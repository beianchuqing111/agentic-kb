"""图导入流水线:文件 → 分块 → 上下文增强 → 三元组抽取 → 写 Neo4j。

    load ──→ 分块 ──→ 定位语 ──→ 三元组抽取 ──→ 实体向量 + 块节点 + 关系
     └────────── 和 IngestPipeline 共用 ──────────┘   └── 图这边独有 ──┘

前处理那半段直接调 `prepare_document`,不重写 —— 两条后端必须切出
**一模一样**的块,否则「切换后端」就不是等价替换了。

为什么不用 PropertyGraphIndex.insert_nodes 写图
---------------------------------------------
检索侧确实用 LlamaIndex(`PropertyGraphStore.vector_query` + `get_rel_map`,
注意**不是** `VectorContextRetriever` —— 那个类只在下面第 2 条里作为
打分口径的参照被提到,全项目从未实例化),
但**写入侧自己写**,有四个具体理由,都是实测出来的:

1. **块会被编码两遍**。`embed_kg_nodes=True` 那一段会重新编码所有块文本,
   而我们上一步刚为 Qdrant 编码过(而且编的是「定位语 + 原文」)。
   自己写就能复用同一批向量 —— 省一次全量前向,更重要的是保证
   两条后端里同一块的向量**逐位一致**。让 LlamaIndex 再编一遍的话,
   它编的是纯 `text`,没有定位语,切换后端等于换了套向量。

2. **实体向量被元数据污染**。`SimpleLLMPathExtractor` 把块的**整个 metadata**
   拷进每个实体/关系的 properties(`EntityNode(name=subj, properties=metadata)`),
   而 `_insert_nodes` 编码实体用的是 `str(kg_node)`:

       EntityNode.__str__ → f"{name} ({properties})"
       → "1号杆塔 ({'doc_id': 'a3f…', 'chunk_index': 7, 'source': 'D:\\…'})"

   也就是说文档 id、块序号、文件路径全都被编进实体向量里。这里只编码
   **实体名**,并且把 properties 精简掉(见 `_strip_properties`)。

3. **每次 insert 都刷一遍 schema**。`supports_structured_queries` 为真,
   `_insert_nodes` 末尾无条件 `get_schema(refresh=True)`;按文档导入
   就是「文档数」次全图扫描,大库上会明显变慢。自己写就只在需要时刷。

4. **同一实体被编码 N 次**。一篇文档提了 50 次「1号杆塔」,上游就编 50 次。
   这里先按 id 去重再编码。注意去重只做在**编码**上:「块提到实体」的
   提及边必须全量写(边是证据,去重即删证据,见 `_collect`)。

另外上游的 `delete_llama_nodes(ref_doc_ids=...)` 按 `ref_doc_id` 找节点,
和我们存的 `doc_id` 对不上(见 graph_store.delete_doc_graph 的说明)。

一个踩过的坑
-----------
`KG_NODES_KEY` / `KG_RELATIONS_KEY` 的字面值是 `"nodes"` / `"relations"`,
抽取器会把它们从 `node.metadata` 里 **pop 掉**。所以块的 metadata 里
**不能**出现叫 `nodes` 或 `relations` 的键 —— 会被静默吃掉。当前用的键
(doc_id/chunk_index/source/title/section)都不冲突,加新键时要注意。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

from config import IngestConfig, get_settings
from embed.bge_m3 import get_embedder
from ingest.graph_extract import build_extractor
from ingest.loader import Document, discover, load_many, stable_doc_id
from ingest.pipeline import EMBED_GROUP, IngestStats, prepare_document
from llm.client import LLMClient, get_llm
from store.graph_store import Neo4jGraphStore, get_graph_store
from store.qdrant_store import Chunk as StoredChunk
from store.qdrant_store import QdrantStore, get_store, point_id

logger = logging.getLogger(__name__)


def chunk_node_id(doc_id: str, chunk_index: int) -> str:
    """块在**两个库里的同一个 id**。

    直接复用 Qdrant 的确定性 point_id(uuid5),于是:

      1. 图里的 ChunkNode.id 和 Qdrant 的点 id 一致 —— 从图反查到块之后,
         拿这个 id 就能把正文捞回来,不需要在别处冗余存一份
      2. 重导同一篇文档时 id 不变,`MERGE` 天然覆盖,不会长出第二套块
      3. `triplet_source_id`(实体/关系上指向来源块的字段)因此也是个稳定值,
         重导后仍能正确回溯

    注意 LlamaIndex 默认用 `node.hash` 当块 id,那是**内容哈希** ——
    正文改一个字 id 就全变,增量重导会留下一堆对不上的旧节点。
    """
    return point_id(doc_id, chunk_index)


def _strip_properties(props: dict, source_id: str) -> dict:
    """把实体/关系的 properties 精简成只剩来源块 id。

    抽取器塞进来的那堆块元数据(doc_id / chunk_index / source / …)对图没用,
    却会:
      - 被 `SET e += row.properties` 写到每个实体上,同一个实体被不同块写到,
        属性就在文档之间来回跳,查出来是随机的
      - 让 `EntityNode.__str__` 变成「名字 (一大坨字典)」,污染实体向量

    `triplet_source_id` 必须留:检索时 `_get_nodes_with_score` 靠它把
    命中的三元组和来源块关联起来(include_text),删文档时也靠它把
    这篇文档产出的关系清掉。

    顺带一提,残留的 `_node_type` 之类键也在这里一并去掉。
    """
    return {"triplet_source_id": source_id} if source_id else {}


class GraphIngestPipeline:
    """图导入编排。接口和 IngestPipeline 对齐,落点多了 Neo4j。"""

    def __init__(
        self,
        store: QdrantStore | None = None,
        graph: Neo4jGraphStore | None = None,
        cfg: IngestConfig | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        s = get_settings()
        self.cfg = cfg or s.ingest
        self.settings = s
        self.store = store or get_store()
        self.graph = graph or get_graph_store()
        self.llm = llm or get_llm()
        self._embedder = None
        self._extractor = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def extractor(self):
        # 抽取器持有 LLM 适配器,构造一次复用 —— 每次导入新建会重连客户端
        if self._extractor is None:
            self._extractor = build_extractor()
        return self._extractor

    # ----------------------------------------------------------------- #
    # 对外入口(和 IngestPipeline 同形)
    # ----------------------------------------------------------------- #

    def ingest_path(
        self,
        root: str | Path,
        recursive: bool = True,
        force: bool = False,
        skip_errors: bool = True,
    ) -> IngestStats:
        t0 = time.time()
        stats = IngestStats()

        files = discover(root, recursive=recursive)
        stats.files_seen = len(files)
        if not files:
            logger.warning("没找到支持的文档:%s", root)
            stats.elapsed = time.time() - t0
            return stats

        logger.info("发现 %d 个待导入文件", len(files))
        docs, errors = load_many(files, on_error="skip" if skip_errors else "raise")
        stats.errors.extend(errors)
        stats.files_loaded = len(docs)

        self._ingest_docs(docs, stats, force=force)
        stats.elapsed = time.time() - t0
        logger.info("图导入完成:%s", stats.summary())
        return stats

    def ingest_text(
        self,
        text: str,
        source: str,
        title: str = "",
        force: bool = False,
        extra: dict | None = None,
    ) -> IngestStats:
        t0 = time.time()
        stats = IngestStats(files_seen=1, files_loaded=1)
        if not text or not text.strip():
            stats.elapsed = time.time() - t0
            return stats

        doc = Document(
            doc_id=stable_doc_id(source),
            text=text,
            source=source,
            title=title or source,
            metadata={"ext": "text", **(extra or {})},
        )
        self._ingest_docs([doc], stats, force=force)
        stats.elapsed = time.time() - t0
        return stats

    # ----------------------------------------------------------------- #
    # 核心
    # ----------------------------------------------------------------- #

    def _ingest_docs(
        self, docs: Sequence[Document], stats: IngestStats, force: bool = False
    ) -> None:
        for doc in docs:
            try:
                self._ingest_one(doc, stats, force=force)
            except Exception as exc:  # noqa: BLE001
                logger.exception("图导入失败: %s", doc.source)
                stats.errors.append((doc.source, str(exc)))
                stats.chunks_failed += 1

    def _ingest_one(self, doc: Document, stats: IngestStats, force: bool = False) -> None:
        prep = prepare_document(doc, self.cfg, self.llm, self.store)
        if prep is None:
            return
        chunks, hashes, contexts = prep.chunks, prep.hashes, prep.contexts

        # 增量判断要比 hybrid 多问一句:图那边完整吗?
        # 只信 Qdrant 的话,「块写进去了但建图那步挂了」这种半截状态
        # 会被判成「未变化」永久跳过 —— 之后图检索一直少一块,而且没有报错。
        if prep.unchanged and not force:
            on_graph = self.graph.doc_chunk_count(doc.doc_id)
            if on_graph == len(chunks):
                stats.docs_unchanged += 1
                logger.debug("未变化,跳过:%s (%d 块)", doc.source, len(chunks))
                return
            logger.warning(
                "%s 的块在 Qdrant 里没变,但图上有 %d/%d 块 —— 补建图",
                doc.source, on_graph, len(chunks),
            )

        # 变了(或图不全)→ 两个库都按文档整体重写,保持两边一致
        if not force:
            if not prep.unchanged:
                self.store.delete_by_doc(doc.doc_id)
            self.graph.delete_doc_graph(doc.doc_id)

        logger.info("图导入 %s:%d 块%s", doc.source, len(chunks),
                    "(force 全量重写)" if force else "")

        stats.contextual_missing += sum(1 for c in contexts if not c)

        written = 0
        for g0 in range(0, len(chunks), EMBED_GROUP):
            g1 = min(g0 + EMBED_GROUP, len(chunks))
            written += self._write_group(
                doc, chunks[g0:g1], contexts[g0:g1], hashes[g0:g1], stats
            )

        stats.docs_indexed += 1
        stats.chunks_written += written

    def _write_group(
        self,
        doc: Document,
        group: list,
        group_ctx: list[str],
        group_hashes: list[str],
        stats: IngestStats,
    ) -> int:
        """处理一批块:编码 → 抽取 → 写 Qdrant → 写图。返回写入的块数。"""
        # 编码的是「定位语 + 原文」,和图/Qdrant 的既有口径一致
        embed_texts = [
            f"{ctx}\n{ch.text}" if ctx else ch.text
            for ch, ctx in zip(group, group_ctx)
        ]
        res = self.embedder.encode(
            embed_texts, batch_size=self._batch_size(len(group))
        )

        # --- 1. 落 Qdrant(和 hybrid 后端同一份数据,见模块说明)---
        stored = [
            StoredChunk(
                doc_id=doc.doc_id,
                chunk_index=ch.index,
                text=ch.text,
                dense=res.dense[i],
                sparse=res.sparse[i],
                context=group_ctx[i],
                source=doc.source,
                title=doc.title,
                content_hash=group_hashes[i],
                extra={
                    "section": ch.section,
                    "doc_hash": doc.content_hash,
                    **(doc.metadata or {}),
                },
            )
            for i, ch in enumerate(group)
        ]
        written = self.store.upsert(stored)

        # --- 2. 抽出三元组 ---
        # 抽取时喂「定位语 + 原文」:定位语说的就是「这段在讲什么」,
        # 对判断哪些实体值得抽是有用信号。
        extract_nodes = [
            self._chunk_node(doc, ch, embed_texts[i]) for i, ch in enumerate(group)
        ]
        extracted = self.extractor(extract_nodes, show_progress=False)

        entities, relations, mentions = self._collect(extracted)
        if not entities and not relations:
            # 抽取全失败(比如 LLM 没配)不该连块都写不进去 ——
            # 块已经在 Qdrant 里了,这条只是提醒图这部分是空的
            logger.warning("%s 这批 %d 块没抽出任何三元组", doc.source, len(group))
            return written

        # --- 3. 写图 ---
        # 块的正文用原文(引用要原文),但向量用上面那批 embed_texts 的 ——
        # 和 Qdrant payload 的口径一致:text 存原文,向量编「定位语 + 原文」
        store_nodes = [
            self._chunk_node(
                doc, ch, ch.text, embedding=[float(x) for x in res.dense[i]]
            )
            for i, ch in enumerate(group)
        ]

        # 实体向量:只编码名字。理由见模块开头第 2 点。
        to_embed = [e for e in entities if e.name]
        vecs = self._encode_entities([e.name for e in to_embed])
        for e, v in zip(to_embed, vecs):
            e.embedding = v

        pg = self.graph.property_graph_store()
        pg.upsert_llama_nodes(store_nodes)
        pg.upsert_nodes(entities)
        pg.upsert_relations(relations)
        # 提及边单独补一遍:upsert_nodes 里那段只连得出「每个实体一块」的边,
        # 因为 triplet_source_id 是单值属性(见 graph_store.link_mentions)
        self.graph.link_mentions(mentions)

        stats.entities_written += len(entities)
        stats.relations_written += len(relations)
        return written

    @staticmethod
    def _chunk_node(
        doc: Document, ch, text: str, embedding: list[float] | None = None
    ):
        """建一个块节点。抽取和落库**共用这一个构造函数**,不各写一遍。

        两处各写一遍的话,迟早出现「三元组挂在 A 版本节点上、检索时按 B 版本
        的 id 反查」—— 而两边都是合法的 TextNode,谁都不会报错。

        必须设 `NodeRelationship.SOURCE`,这不是装饰。`upsert_llama_nodes` 会调
        `node_to_metadata_dict`,里面有一行

            metadata["doc_id"] = node.ref_doc_id or "None"

        把我们 `_chunk_meta` 里填的 doc_id **原样覆盖掉**。不设这个关系时
        ref_doc_id 是 None,于是 Neo4j 上每个块节点的 doc_id 都变成字符串
        "None" —— doc_chunk_count / delete_doc_graph 按 doc_id 什么都查不到:
        重导不会清旧块,图里慢慢堆起互相矛盾的旧数据,**而且全程没有报错**。
        (实测踩过:删文档返回 0,块和实体一个没少。)
        """
        from llama_index.core.schema import (
            NodeRelationship,
            RelatedNodeInfo,
            TextNode,
        )

        return TextNode(
            id_=chunk_node_id(doc.doc_id, ch.index),
            text=text,
            metadata=GraphIngestPipeline._chunk_meta(doc, ch),
            embedding=embedding,
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc.doc_id)
            },
        )

    @staticmethod
    def _chunk_meta(doc: Document, ch) -> dict[str, Any]:
        """块的 metadata —— 会原样变成 Neo4j 上的属性,`doc_id` 是增量重导的关键。

        键名注意别撞 `nodes` / `relations`,抽取器会把这两个 pop 掉。
        """
        return {
            "doc_id": doc.doc_id,
            "chunk_index": int(ch.index),
            "source": doc.source,
            "title": doc.title,
            "section": ch.section,
        }

    def _collect(self, extracted) -> tuple[list, list, list[tuple[str, str]]]:
        """收实体、关系,以及「块 → 实体」的提及对。

        三者的去重口径**不一样**,混在一起就会丢东西:

          entities —— 按 id 去重。同一个实体在 20 个块里出现 20 次,`MERGE`
                      会让它们合成一个节点,但**编码**是按条数来的,不去重
                      就白编 19 次。编码是这里唯一贵的事情。
          relations —— 按 (头, 类型, 尾) 去重,否则同一条边被写 N 遍。
          mentions —— 按 **(块, 实体) 对**去重,但**绝不能按实体去重**。边本身
                      就是「这块提到过这个实体」这条事实,按实体去重即删证据:
                      实体上的 triplet_source_id 是单值属性,去重后只剩一块
                      连得上边,图反查块时会少召回一大批
                      (见 graph_store.link_mentions)。
        """
        from llama_index.core.graph_stores.types import (
            KG_NODES_KEY,
            KG_RELATIONS_KEY,
        )
        from llama_index.core.graph_stores.types import EntityNode, Relation

        seen: set[str] = set()
        entities: list = []
        rel_seen: set[tuple[str, str, str]] = set()
        relations: list = []
        mention_seen: set[tuple[str, str]] = set()
        mentions: list[tuple[str, str]] = []

        for node in extracted:
            source_id = node.id_ or ""
            for kg_node in node.metadata.pop(KG_NODES_KEY, []) or []:
                if not isinstance(kg_node, EntityNode):
                    continue
                kg_node.properties = _strip_properties(kg_node.properties, source_id)
                # 同一个块里同一个实体可能被不同的三元组各带出一次,去成一对
                if source_id and (source_id, kg_node.id) not in mention_seen:
                    mention_seen.add((source_id, kg_node.id))
                    mentions.append((source_id, kg_node.id))
                if kg_node.id in seen:
                    continue
                seen.add(kg_node.id)
                entities.append(kg_node)
            for kg_rel in node.metadata.pop(KG_RELATIONS_KEY, []) or []:
                if not isinstance(kg_rel, Relation):
                    continue
                kg_rel.properties = _strip_properties(kg_rel.properties, source_id)
                key = (kg_rel.source_id, kg_rel.label, kg_rel.target_id)
                if key in rel_seen:
                    continue
                rel_seen.add(key)
                relations.append(kg_rel)

        logger.debug(
            "本批抽取: %d 实体(去重后) / %d 提及对 / %d 关系",
            len(entities), len(mentions), len(relations),
        )
        return entities, relations, mentions

    def _encode_entities(self, names: list[str]) -> list[list[float]]:
        if not names:
            return []
        res = self.embedder.encode(names, batch_size=self._batch_size(len(names)))
        return [[float(x) for x in v] for v in res.dense]

    def _batch_size(self, n: int) -> int:
        base = self.settings.embed.batch_size
        return max(1, min(base, n))


_pipeline: GraphIngestPipeline | None = None


def get_graph_pipeline() -> GraphIngestPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GraphIngestPipeline()
    return _pipeline


__all__ = [
    "GraphIngestPipeline",
    "chunk_node_id",
    "get_graph_pipeline",
]
