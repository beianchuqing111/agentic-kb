"""GraphRAG 后端:实体向量定位 → 多跳展开 → 块取证 → 统一重排。

检索漏斗
-------
    query
      ├─ bge-m3 稠密 ─→ Neo4j `entity` 向量索引 → 种子实体 top-K
      │                        ↓
      │                 max_hops 跳展开(get_rel_map) → 三元组事实
      │                        ↓
      │                 反查「提到这些实体的块」→ 取回正文(图命中的块)
      │
      └─ bge-m3 稠密+稀疏 ─→ Qdrant RRF → 候选块(字面/语义命中的块)
                                ↓
             两者合并去重 → bge-reranker-v2 重排 → 阈值过滤 → top-N

为什么最后还要并上 Qdrant 那一路
-------------------------------
只靠图会有一个硬伤:**没有实体命中的问题就返回空**。像「继电保护有哪些
要求」这种,查询词里没有一个能对上图中的实体名,种子实体全是很弱的
匹配,出来的块自然不对。而这类问题恰恰是向量检索最擅长的。

微软那套 GraphRAG 的 local search 也是这么做的:实体 + 关系 + 文本单元
一起进上下文。图给的是**多跳可达性**(查询词和答案之间隔了几个实体),
向量给的是**字面与语义的相似**。两者互补,不是二选一。

所以这个后端 = 混合检索 + 图带来的额外召回,不会比纯混合检索更差。

关于块的两份数据
--------------
导入时块同时写了 Qdrant 和 Neo4j(见 ingest/graph_pipeline.py),
**用的是同一批向量**(「定位语 + 原文」)。这里图反查块之后是从 Qdrant
取正文的 —— 因为 Qdrant 里存着完整 payload(引用要的 source/title),
Neo4j 那边只存了块的属性。块 id 两边一致(uuid5),所以拿得回来。
"""

from __future__ import annotations

import logging
from typing import Any

from config import get_settings
from ingest.graph_pipeline import GraphIngestPipeline
from ingest.pipeline import IngestStats
from retrieve.backends.base import BackendHealth, BaseBackend
from retrieve.hybrid import (
    HybridRetriever,
    RetrievalDebug,
    RetrievedChunk,
    apply_rerank,
)
from retrieve.reranker import get_reranker
from store.graph_store import INTERNAL_RELS, MENTIONS_REL, get_graph_store
from store.qdrant_store import get_store

logger = logging.getLogger(__name__)

BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"

# 图反查出来的块最多取多少。种子实体通常 10 个左右,每个实体牵扯的块
# 可能几十个,不设上限会灌一大堆弱相关块进重排,把重排算力全花掉。
GRAPH_CHUNK_LIMIT = 60


class GraphRAGBackend(BaseBackend):
    name = "graphrag"

    def __init__(
        self,
        graph=None,
        store=None,
        pipeline: GraphIngestPipeline | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        s = get_settings()
        self.settings = s
        self.graph = graph or get_graph_store()
        self.store = store or get_store()
        self.pipeline = pipeline or GraphIngestPipeline(
            store=self.store, graph=self.graph
        )
        # 块的候选召回复用混合检索那一套(稠密+稀疏+RRF),
        # 但不让它重排 —— 重排要等图那边的候选并进来之后统一做一次
        self.retriever = retriever or HybridRetriever(store=self.store)
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(self.settings.retrieval)
        return self._reranker

    # ----------------------------------------------------------------- #
    # 写
    # ----------------------------------------------------------------- #

    def ingest_path(self, root, recursive: bool = True, force: bool = False) -> IngestStats:
        self.store.ensure_collection(dense_dim=self.settings.embed.dense_dim)
        return self.pipeline.ingest_path(root, recursive=recursive, force=force)

    def ingest_text(
        self,
        text: str,
        source: str,
        title: str = "",
        force: bool = False,
        extra: dict | None = None,
    ) -> IngestStats:
        if not text or not text.strip():
            return IngestStats()
        self.store.ensure_collection(dense_dim=self.settings.embed.dense_dim)
        return self.pipeline.ingest_text(
            text, source, title=title, force=force, extra=extra
        )

    # ----------------------------------------------------------------- #
    # 读
    # ----------------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        debug: RetrievalDebug | None = None,
        **kwargs: Any,
    ) -> list[RetrievedChunk]:
        query = (query or "").strip()
        if not query:
            return []

        # 同 HybridBackend.retrieve:空库要在这一层拦住。
        # 图这边的链路更长 —— 图反查和向量召回各自都要碰 Qdrant,不拦的话
        # 404 会从两个不同的地方冒出来,报错还不一样。
        if not self.store.exists():
            return []

        cfg = self.settings.retrieval
        top_k = top_k or cfg.rerank_top_n

        # 复用 RetrievalDebug 的两个计数字段装图这边的量:实体数 / 事实数。
        # 字段名是混合检索那边起的,但对排查「图到底有没有起作用」够用 ——
        # 实体数为 0 就说明退化成纯向量检索了。
        facts, entity_ids, _ = self._graph_context(query)
        if debug is not None:
            debug.dense_hits = len(entity_ids)
            debug.sparse_hits = len(facts)

        # --- 候选块:图反查的 + 向量召回的 ---
        graph_chunks = self._chunks_mentioning(entity_ids)
        fused = self.retriever.retrieve(
            query, top_k=cfg.fusion_top_k, use_rerank=False
        )

        merged = self._merge(graph_chunks, fused)
        if not merged:
            return []

        # 把事实挂到相关块上 —— 上层(ReAct/生成)要能看见「为什么这些块被拉进来」
        self._attach_facts(merged, facts)

        # --- 统一重排一次 ---
        if not cfg.rerank_enabled:
            return merged[:top_k]
        return apply_rerank(
            query, merged, top_k,
            cfg=cfg, reranker=self.reranker, debug=debug,
        )

    # ----------------------------------------------------------------- #
    # 图这一步
    # ----------------------------------------------------------------- #

    def _graph_context(self, query: str) -> tuple[list[dict], list[str], dict[str, float]]:
        """查种子实体并多跳展开。返回 (事实列表, 实体 id 列表, 实体分数字典)。"""
        from llama_index.core.vector_stores.types import VectorStoreQuery

        cfg = self.settings.graphrag
        qvec = [float(x) for x in self.retriever.embedder.encode(query).dense[0]]

        pg = self.graph.property_graph_store()
        try:
            nodes, scores = pg.vector_query(
                VectorStoreQuery(query_embedding=qvec, similarity_top_k=cfg.vector_top_k)
            )
        except Exception as exc:  # noqa: BLE001
            # 图这一路挂了不该让整个检索失败 —— 下面还有向量那一路兜底。
            # 但这一定要记日志:静默降级会让"图检索没效果"变成一个查不出原因的谜。
            logger.warning("实体向量检索失败,本次只走向量召回: %s", exc)
            return [], [], {}

        if not nodes:
            return [], [], {}

        entity_ids = [n.id for n in nodes]
        seed_scores = {n.id: float(s) for n, s in zip(nodes, scores)}

        # 多跳展开交给 LlamaIndex 的 get_rel_map —— 路径去重、深度控制、
        # 排除 SOURCE 这类内部关系都在里面,不值得自己重写
        try:
            triplets = pg.get_rel_map(
                nodes, depth=cfg.max_hops, limit=cfg.vector_top_k * 5,
                ignore_rels=list(INTERNAL_RELS),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("多跳展开失败,只用种子实体: %s", exc)
            triplets = []

        # 去重是必须的。`get_rel_map` 的 Cypher 是
        #     UNWIND range(0, size(id_list) - 1) AS idx ... ORDER BY idx LIMIT n
        # 也就是**按种子实体逐个展开**,同一条边只要从多个种子都走得到,就会被
        # 重复返回一次。实测 6 个种子实体把 4 条边展开成了 12 条 ——
        # 数字虚高,后面 _attach_facts 还得白跑一遍。
        facts: list[dict] = []
        seen_facts: dict[tuple[str, str, str], dict] = {}
        for t in triplets or []:
            head, rel, tail = t[0], t[1], t[2]
            # 这条事实的置信度取两端实体里更强的那个 —— 与上游
            # VectorContextRetriever 的口径一致
            score = max(
                seed_scores.get(head.id, 0.0), seed_scores.get(tail.id, 0.0)
            )
            key = (head.id, rel.label, tail.id)
            prev = seen_facts.get(key)
            if prev is not None:
                # 重复出现时取更高的分(不同种子给出的置信度不一样)
                prev["score"] = max(prev["score"], score)
                continue

            fact = {
                "head": head.id,
                "relation": rel.label,
                "tail": tail.id,
                "score": score,
            }
            seen_facts[key] = fact
            facts.append(fact)
            entity_ids.extend([head.id, tail.id])

        # 展开后可能冒出新的实体,去重保序
        entity_ids = list(dict.fromkeys(entity_ids))
        return facts, entity_ids, seed_scores

    # ----------------------------------------------------------------- #
    # 块取证
    # ----------------------------------------------------------------- #

    def _chunks_mentioning(self, entity_ids: list[str]) -> list[tuple[RetrievedChunk, float, list[str]]]:
        """反查「哪些块提到了这些实体」,再把正文从 Qdrant 取回来。

        这一步是图检索真正的价值所在:实体多跳走到的那些块,**正文里往往
        没有查询词**,向量检索够不着。比如问「1号杆塔有什么缺陷」,
        「绝缘子破损」那块可能通篇不提杆塔 —— 但杆塔的边连到了绝缘子。

        返回的分数是「这块提到了几个种子实体」,只用来排个初序;
        最终顺序由重排决定。
        """
        if not entity_ids:
            return []

        rows = self.graph.run(
            f"""
            MATCH (c:`{BASE_NODE_LABEL}`)-[:{MENTIONS_REL}]->(e:`{BASE_ENTITY_LABEL}`)
            WHERE e.id IN $ids
            RETURN c.doc_id AS doc_id, c.chunk_index AS chunk_index,
                   count(DISTINCT e.id) AS hits,
                   collect(DISTINCT e.id) AS ents
            ORDER BY hits DESC
            LIMIT $lim
            """,
            ids=entity_ids,
            lim=GRAPH_CHUNK_LIMIT,
        )
        if not rows:
            return []

        keys = [
            (r["doc_id"], int(r["chunk_index"]))
            for r in rows
            if r.get("doc_id") is not None and r.get("chunk_index") is not None
        ]
        records = self.store.fetch_chunks(keys)

        by_key = {
            ((r.payload or {}).get("doc_id", ""), int((r.payload or {}).get("chunk_index", 0))): r
            for r in records
        }

        out: list[tuple[RetrievedChunk, float, list[str]]] = []
        missing = 0
        for r in rows:
            doc_id, idx = r.get("doc_id"), r.get("chunk_index")
            if doc_id is None or idx is None:
                continue
            rec = by_key.get((doc_id, int(idx)))
            if rec is None:
                # 图里有块、Qdrant 里没有 —— 两边不一致,说明导入中断过。
                # 记下来,别静默跳过。
                missing += 1
                continue
            hits = float(r.get("hits") or 0)
            out.append((RetrievedChunk.from_payload(rec.payload or {}), hits, list(r.get("ents") or [])))

        if missing:
            logger.warning(
                "有 %d 个块在图里但 Qdrant 里没有 —— 可能上次导入中断,"
                "建议对相关文档 force 重导", missing,
            )
        return out

    # ----------------------------------------------------------------- #
    # 合并 / 挂事实
    # ----------------------------------------------------------------- #

    @staticmethod
    def _merge(
        graph_chunks: list[tuple[RetrievedChunk, float, list[str]]],
        fused: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """按 (doc_id, chunk_index) 去重合并两路候选。

        同一个块可能既被图反查到、又被向量召回。合并时必须保留**向量那一路
        的分数**(那是真实相似度),图那一路的 hits 只作为 meta 里的辅助信息 ——
        否则重排前的初序会被一个量纲完全不同的数字带偏。
        """
        merged: dict[tuple[str, int], RetrievedChunk] = {}
        ents_of: dict[tuple[str, int], list[str]] = {}

        for r in fused:
            merged[(r.doc_id, r.chunk_index)] = r

        for rc, hits, ents in graph_chunks:
            key = (rc.doc_id, rc.chunk_index)
            ents_of[key] = ents
            if key in merged:
                merged[key].meta.setdefault("graph_hits", hits)
                continue
            rc.meta["graph_hits"] = hits
            rc.meta["from_graph"] = True
            merged[key] = rc

        for key, r in merged.items():
            if key in ents_of:
                r.meta["entities"] = ents_of[key]

        return list(merged.values())

    @staticmethod
    def _attach_facts(
        chunks: list[RetrievedChunk], facts: list[dict]
    ) -> None:
        """把和块相关的三元组挂到块上(meta['facts'])。

        一个块提到的实体若出现在某条事实的两端,这条事实就和它相关。
        上限 8 条:事实是给模型看的补充线索,塞太多会挤掉正文。
        """
        if not facts:
            return
        by_entity: dict[str, list[dict]] = {}
        for f in facts:
            by_entity.setdefault(f["head"], []).append(f)
            by_entity.setdefault(f["tail"], []).append(f)

        for r in chunks:
            ents = r.meta.get("entities") or []
            seen: set[tuple[str, str, str]] = set()
            picked: list[dict] = []
            for e in ents:
                for f in by_entity.get(e, []):
                    k = (f["head"], f["relation"], f["tail"])
                    if k in seen:
                        continue
                    seen.add(k)
                    picked.append(f)
                    if len(picked) >= 8:
                        break
                if len(picked) >= 8:
                    break
            if picked:
                r.meta["facts"] = picked

    # ----------------------------------------------------------------- #
    # 运维
    # ----------------------------------------------------------------- #

    def health(self) -> BackendHealth:
        q = self.store.health()
        g = self.graph.health()
        detail = {
            "qdrant": q.get("collections") if q.get("ok") else None,
            "neo4j": (
                f"{g.get('edition')} {g.get('version')}" if g.get("ok") else None
            ),
            "collection": self.settings.qdrant.collection,
            "database": self.settings.neo4j.database,
            "llm_configured": self.settings.llm.configured,
            "extraction": self.settings.llm.configured,
            "rerank": self.settings.retrieval.rerank_enabled,
        }
        ok = bool(q.get("ok")) and bool(g.get("ok"))
        err = ""
        if not q.get("ok"):
            err = f"Qdrant: {q.get('error')}"
        elif not g.get("ok"):
            err = f"Neo4j: {g.get('error')}"
        return BackendHealth(ok=ok, backend=self.name, detail=detail, error=err)

    def stats(self) -> dict[str, Any]:
        docs = self.store.list_docs() if self.store.exists() else []
        g = self.graph.stats()
        return {
            "backend": self.name,
            "collection": self.settings.qdrant.collection,
            "database": self.settings.neo4j.database,
            "docs": len(docs),
            "chunks": self.store.count() if self.store.exists() else 0,
            "entities": g.get("entities", 0),
            "relations": g.get("relations", 0),
            "entities_with_embedding": g.get("entities_with_embedding", 0),
            "documents": docs,
        }

    def close(self) -> None:
        self.graph.close()


__all__ = ["GraphRAGBackend"]
