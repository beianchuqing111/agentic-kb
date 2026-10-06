"""Neo4j 图存储:自研的图写入/检索层,底层用的是 LlamaIndex 的
`Neo4jPropertyGraphStore`(经 `store/pg_store.py` 子类化成免 APOC 版),
外加两件 LlamaIndex 不管的事。

**注意这里不是 `PropertyGraphIndex`。** 全项目从未构造过 `PropertyGraphIndex`
(那个编排层),理由见 `ingest/graph_pipeline.py` 开头:它的 `insert_nodes`
会把块编码两遍、把块 metadata 编进实体向量、每次 insert 刷全图 schema。
我们只取 LlamaIndex 的两样东西 —— `SimpleLLMPathExtractor`(抽取)和
`PropertyGraphStore`(存储抽象)—— 写入和多跳检索自己写。

一、向量索引维度必须我们抢先建
-----------------------------
LlamaIndex 的 Neo4jPropertyGraphStore 建的是:

    CREATE VECTOR INDEX entity IF NOT EXISTS FOR (m:__Entity__) ON m.embedding

**没写 vector.dimensions**,于是维度就取决于 Neo4j 版本:
  - 5.26.9(本项目实测):索引配置里**根本没有** dimensions 键 —— 动态维度,
    按写入的第一个向量定。
  - 5.11~5.17 时代:默认 1536(照 OpenAI 的维度),写 1024 维的 bge-m3 向量会报错。

所以显式写死 1024 有两个好处,都和版本无关:
  1. 消除版本间行为差异 —— 换台机器重装 Neo4j 不会突然变了维度规则
  2. 维度不匹配会在**写入前**暴露成明确错误,而不是让动态索引
     默默接受了某个来路不明的维度

好在 LlamaIndex 用了 IF NOT EXISTS。只要我们在创建 store **之前**
先把索引建好,它那句就自动变成空操作。这个顺序不能反 ——
反了就得先 DROP 再重建。

二、中文全文索引 LlamaIndex 完全不建
----------------------------------
它只建 UNIQUE 约束和一个向量索引,没有任何 fulltext 索引。
而中文实体名匹配(「北京」查「北京市」这类)要靠全文检索兜底,
Neo4j 底层是 Lucene,中文得显式配 `cjk` 分析器。
Lucene 的 CJKAnalyzer 按双字切分,是中文最稳的选择。

三、实体对齐靠向量相似度
----------------------
「北京」/「北京市」/「首都」指同一个东西,图里却是三个节点,
多跳查询就会断链。这是中文 GraphRAG 最容易被漏掉的一层。
这里用 Neo4j 原生的向量索引找近邻(避免 O(N²) 两两比对),
再用并查集把候选归组,最后合并节点。
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from config import GraphRAGConfig, Neo4jConfig, get_settings

logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "entity"           # 必须和 LlamaIndex 里的常量一致
FULLTEXT_INDEX_NAME = "entity_name_cjk"

# LlamaIndex 用的标签,前缀双下划线
BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
# 块 -> 实体 的边,由 pg_store 写入。<-[:MENTIONS]- 表示「e 被 c 提到」
MENTIONS_REL = "MENTIONS"

# 内部边,不是语义关系:块→实体的提及边,以及 LlamaIndex 可能写的来源边。
# 统计「图谱有多少关系」时必须排掉,否则提及边会把数字抬好几倍
# (实测 3 条真关系 + 7 条提及边被读成 10)。
# 注意上游 get_rel_map 自己就在 Cypher 里硬排除了 MENTIONS,
# 所以事实列表那边不用我们操心 —— 会污染的是统计数字。
INTERNAL_RELS = (MENTIONS_REL, "SOURCE")

DENSE_DIM = 1024

# 关系类型白名单:合并节点时要拼进 Cypher(纯 Cypher 没法把关系类型参数化),
# 所以只能做字符串插值。为了不引入注入面,类型必须匹配这个正则。
_REL_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class Neo4jGraphStore:
    """Neo4j 连接 + 索引管理 + 实体对齐。

    这一层管连接和 schema;**检索不在这里**,而在
    `retrieve/backends/graphrag_backend.py` —— 它拿 `property_graph_store()`
    出来的 store 调 `vector_query`(种子实体)和 `get_rel_map`(多跳展开)。
    两个方法都来自 LlamaIndex 的 `PropertyGraphStore`,与
    `PropertyGraphIndex` 无关(那个类全项目没构造过)。
    """

    def __init__(
        self,
        cfg: Neo4jConfig | None = None,
        graph_cfg: GraphRAGConfig | None = None,
    ) -> None:
        s = get_settings()
        self.cfg = cfg or s.neo4j
        self.graph_cfg = graph_cfg or s.graphrag
        self._driver: Any = None
        self._pg_store: Any = None

    # ----------------------------------------------------------------- #
    # 连接
    # ----------------------------------------------------------------- #

    @property
    def driver(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.cfg.uri,
                auth=(self.cfg.username, self.cfg.password),
                max_connection_lifetime=self.cfg.max_connection_lifetime,
            )
            self._driver.verify_connectivity()
            logger.info("连接 Neo4j: %s (db=%s)", self.cfg.uri, self.cfg.database)
        return self._driver

    def session(self):
        return self.driver.session(database=self.cfg.database)

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.session() as s:
            return [dict(r) for r in s.run(cypher, **params)]

    def close(self) -> None:
        if self._pg_store is not None:
            try:
                self._pg_store.close()
            except Exception:  # noqa: BLE001
                pass
            self._pg_store = None
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def health(self) -> dict[str, Any]:
        try:
            rows = self.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            )
            info = rows[0] if rows else {}
            return {
                "ok": True,
                "server": info.get("name"),
                "version": info.get("version"),
                "edition": info.get("edition"),
                "uri": self.cfg.uri,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "uri": self.cfg.uri, "error": str(exc)}

    # ----------------------------------------------------------------- #
    # 索引
    # ----------------------------------------------------------------- #

    def ensure_indexes(self, dense_dim: int = DENSE_DIM) -> dict[str, Any]:
        """建好检索要用的索引。**必须在创建 PropertyGraphStore 之前调用**。

        幂等:全用 IF NOT EXISTS,重复调用无害。
        """
        created: dict[str, Any] = {}

        # 1. 向量索引 —— 显式给维度,否则 Neo4j 按 1536 建,和 bge-m3 的 1024 对不上
        self.run(
            f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (m:`{BASE_ENTITY_LABEL}`) ON m.embedding
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            dim=dense_dim,
        )
        actual = self._vector_index_dim()
        created["vector_index"] = {
            "name": VECTOR_INDEX_NAME,
            "dimensions": actual,
            "expected": dense_dim,
            "ok": actual in (None, dense_dim),
        }
        if actual is not None and actual != dense_dim:
            raise ValueError(
                f"向量索引 {VECTOR_INDEX_NAME} 的维度是 {actual},但需要 {dense_dim}。"
                f"多半是先建了 store、索引已按别的维度定型。"
                f"请先 DROP INDEX {VECTOR_INDEX_NAME} 再重跑 ensure_indexes()。"
                f"(图数据不会因此丢失,重跑一次索引即可)"
            )

        # 2. 中文全文索引 —— LlamaIndex 完全不建,得自己加
        self.run(
            f"""
            CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
            FOR (n:`{BASE_ENTITY_LABEL}`) ON EACH [n.id, n.name]
            OPTIONS {{indexConfig: {{`fulltext.analyzer`: 'cjk'}}}}
            """
        )
        created["fulltext_index"] = {
            "name": FULLTEXT_INDEX_NAME,
            "analyzer": self._fulltext_analyzer(FULLTEXT_INDEX_NAME),
        }

        # 3. 唯一约束(和 LlamaIndex 建的一致,提前建掉避免并发建)
        for label in (BASE_NODE_LABEL, BASE_ENTITY_LABEL):
            self.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) "
                "REQUIRE n.id IS UNIQUE"
            )
        created["constraints"] = [BASE_NODE_LABEL, BASE_ENTITY_LABEL]

        logger.info("Neo4j 索引就绪: %s", created)
        return created

    def list_indexes(self) -> list[dict[str, Any]]:
        rows = self.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
            "RETURN name, type, labelsOrTypes, properties, options"
        )
        return rows

    def _vector_index_dim(self) -> int | None:
        rows = self.run(
            "SHOW VECTOR INDEXES YIELD name, options WHERE name = $n RETURN options",
            n=VECTOR_INDEX_NAME,
        )
        if not rows:
            return None
        opts = rows[0].get("options") or {}
        cfg = opts.get("indexConfig") or {}
        dim = cfg.get("vector.dimensions")
        return int(dim) if dim is not None else None

    def _fulltext_analyzer(self, name: str) -> str | None:
        rows = self.run(
            "SHOW FULLTEXT INDEXES YIELD name, options WHERE name = $n RETURN options",
            n=name,
        )
        if not rows:
            return None
        opts = rows[0].get("options") or {}
        cfg = opts.get("indexConfig") or {}
        return cfg.get("fulltext.analyzer")

    # ----------------------------------------------------------------- #
    # LlamaIndex 接入
    # ----------------------------------------------------------------- #

    def property_graph_store(self) -> Any:
        """拿到底层的 LlamaIndex `PropertyGraphStore`(免 APOC 子类)。

        调用方是 `graphrag_backend`(`vector_query` / `get_rel_map`)和
        自研写入流水线(`upsert_nodes` / `upsert_relations`),不是
        `PropertyGraphIndex` —— 那个类没被构造过。

        第一次调用前会先把索引建好 —— 顺序很关键,见模块开头。
        """
        if self._pg_store is None:
            # 用自己这个子类,不用父类 —— 父类在 refresh_schema / upsert_*
            # 四处依赖 APOC,而本机 Neo4j Community 没装 APOC,构造就会抛
            # ProcedureNotFound。详见 store/pg_store.py 的说明。
            from store.pg_store import APOCFreeNeo4jPropertyGraphStore

            self.ensure_indexes()

            self._pg_store = APOCFreeNeo4jPropertyGraphStore(
                username=self.cfg.username,
                password=self.cfg.password,
                url=self.cfg.uri,
                database=self.cfg.database,
                refresh_schema=True,
                # 查询结果里塞了太多 schema 噪声时打开,代价是多一次 LLM 调用;
                # 这里关掉,需要时在检索器上单独控制
                sanitize_query_output=False,
                # 我们自己建索引,别让它插手(它那句 IF NOT EXISTS 本来就是空操作,
                # 关掉只是省一次 round trip)
                create_indexes=True,
            )
            logger.info("PropertyGraphStore 就绪 (db=%s)", self.cfg.database)
        return self._pg_store

    # ----------------------------------------------------------------- #
    # 统计 / 清理
    # ----------------------------------------------------------------- #

    def stats(self) -> dict[str, Any]:
        try:
            rows = self.run(
                f"""
                MATCH (e:`{BASE_ENTITY_LABEL}`)
                WITH count(e) AS entities
                MATCH ()-[r]->()
                WHERE NOT type(r) IN $internal
                RETURN entities, count(r) AS relations
                """,
                internal=list(INTERNAL_RELS),
            )
            with_emb = self.run(
                f"MATCH (e:`{BASE_ENTITY_LABEL}`) WHERE e.embedding IS NOT NULL "
                "RETURN count(e) AS c"
            )
            # 实体节点也带 `__Node__` 标签(`upsert_nodes` 里 MERGE 的就是它,
            # 之后再补 `__Entity__`),所以直接数 `__Node__` 会把实体一起数进去 ——
            # 3 块 + 4 实体被读成「7 块」,排查时白绕一圈。
            chunks = self.run(
                f"MATCH (n:`{BASE_NODE_LABEL}`) WHERE NOT n:`{BASE_ENTITY_LABEL}` "
                "RETURN count(n) AS c"
            )
            row = rows[0] if rows else {}
            return {
                "entities": row.get("entities", 0),
                "relations": row.get("relations", 0),
                "entities_with_embedding": (with_emb[0]["c"] if with_emb else 0),
                "chunks": (chunks[0]["c"] if chunks else 0),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def clear(self) -> None:
        """清空图。**危险操作**,只应在显式要求时调用。

        分批删除 —— 一次性 DETACH DELETE 在大图上会吃满事务内存。
        """
        while True:
            rows = self.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c")
            if not rows or not rows[0]["c"]:
                break
        logger.warning("Neo4j 图已清空")

    # ----------------------------------------------------------------- #
    # 按文档增删(增量重导)
    # ----------------------------------------------------------------- #
    #
    # 用自己写的 Cypher,而不是 LlamaIndex 的 delete_llama_nodes(ref_doc_ids=...):
    #
    #   1. 它**只删节点**。这些块产出的关系(实体与实体之间的边)原封不动,
    #      而实体节点是跨文档共享的,DETACH DELETE 块带不走它们 —— 于是重导
    #      之后旧关系还在,和新抽出来的并成两份,谁也没报错。
    #   2. 它按 properties 里的 `ref_doc_id` 找节点。那个值是
    #      `node_to_metadata_dict` 从 `node.ref_doc_id` 灌进去的,我们设了
    #      SOURCE 关系让它等于 doc_id(见 graph_pipeline._chunk_node),所以
    #      键能对上;但块节点一旦换了构造方式,这里就会静默地「删了个空」。
    #      不依赖那层间接 —— 直接按我们自己的 doc_id 查。

    def link_mentions(self, pairs: Sequence[tuple[str, str]]) -> int:
        """补全「块 → 实体」的 MENTIONS 边。**必须在 upsert_nodes 之后调用**。

        为什么不能只靠 upsert_nodes 里那一段
        ----------------------------------
        `upsert_nodes` 连 MENTIONS 用的是实体上的 `triplet_source_id`,而那是个
        **单值属性**:`SET e += row.properties` 之后它只留得下最后写进去的那一块。
        于是同一批里一个实体出现在 20 个块,就只有 1 个块和它连上了边 ——
        图反查块(`_chunks_mentioning`)会少召回 19 个块,**而且不报错**。
        (实测:一批 2 块、5 个提及对,只连出 3 条边。)

        编码才是贵的那个(每个实体名一次前向),边是便宜的。所以去重只做在
        编码上,边必须按「(块, 实体) 对」全量写。

        pairs 为 (块 id, 实体 id);两端都必须已存在,查不到的行直接跳过
        (实体可能因标签非法等原因没能落地)。返回**处理到的对数**,不是
        新增边数 —— MERGE 到已存在的边上也会计入。
        """
        if not pairs:
            return 0
        uniq = list(dict.fromkeys(pairs))
        total = 0
        for i in range(0, len(uniq), 1000):
            rows = self.run(
                f"""
                UNWIND $rows AS row
                MATCH (c:`{BASE_NODE_LABEL}` {{id: row[0]}})
                MATCH (e:`{BASE_ENTITY_LABEL}` {{id: row[1]}})
                MERGE (c)-[:{MENTIONS_REL}]->(e)
                RETURN count(*) AS c
                """,
                rows=[list(p) for p in uniq[i : i + 1000]],
            )
            total += int(rows[0]["c"]) if rows else 0
        return total

    def doc_chunk_count(self, doc_id: str) -> int:
        """这篇文档在图上有多少块。用来发现「Qdrant 有、图里没有」的半截状态。"""
        rows = self.run(
            f"MATCH (c:`{BASE_NODE_LABEL}`) WHERE c.doc_id = $d RETURN count(c) AS c",
            d=doc_id,
        )
        return int(rows[0]["c"]) if rows else 0

    def delete_doc_graph(self, doc_id: str) -> dict[str, int]:
        """把一篇文档在图上的痕迹清掉:块、这些块产出的关系、以及因此孤立的实体。

        顺序不能变,理由如下 ——
          1. 先记下这些块提到过哪些实体(删完块就查不到了)
          2. 再删「由这些块抽出来的关系」。实体节点是**跨文档共享**的,
             DETACH DELETE 块只能带走 MENTIONS 边,带不走实体与实体之间的边;
             那些边的 triplet_source_id 指向已删的块,留着就是永不更新的陈旧事实
          3. 然后删块本身
          4. 最后清理孤立实体
        """
        rows = self.run(
            f"""
            MATCH (c:`{BASE_NODE_LABEL}`)-[:{MENTIONS_REL}]->(e:`{BASE_ENTITY_LABEL}`)
            WHERE c.doc_id = $d
            RETURN collect(DISTINCT e.id) AS ids, collect(DISTINCT c.id) AS chunk_ids
            """,
            d=doc_id,
        )
        entity_ids = (rows[0]["ids"] if rows else None) or []
        chunk_ids = (rows[0]["chunk_ids"] if rows else None) or []

        # 块 id 从 Qdrant 侧更全(图里可能因为上次导入中断而缺块),
        # 但这里只需要图上的部分,漏掉的本来就没写进来
        rels = self.run(
            f"MATCH ()-[r]->() WHERE r.triplet_source_id IN $ids "
            "DELETE r RETURN count(r) AS c",
            ids=chunk_ids,
        )
        chunks = self.run(
            f"MATCH (c:`{BASE_NODE_LABEL}`) WHERE c.doc_id = $d "
            "DETACH DELETE c RETURN count(c) AS c",
            d=doc_id,
        )
        orphans = self._prune_orphan_entities(entity_ids)

        return {
            "relations_deleted": int(rels[0]["c"]) if rels else 0,
            "chunks_deleted": int(chunks[0]["c"]) if chunks else 0,
            "orphan_entities_deleted": orphans,
        }

    def _prune_orphan_entities(self, entity_ids: Sequence[str]) -> int:
        """删掉「没有任何块再提到、且邻居也都孤立的」实体。

        只处理这次删块波及到的实体(不是全图扫描)—— 大图上全图找孤儿很贵。

        判据是**保守**的:只要某个邻居实体还有块在提,这个实体就留着。
        因为邻居还活着,说明它俩之间那条边可能仍由活着的块支撑,
        而实体节点是跨文档合并的,单看「它自己有没有 MENTIONS」会误删。
        宁可留下少量陈旧节点(影响小、可事后人工清理),也不要删掉
        仍然有效的边 —— 后者会让别的文档的检索凭空断链,且不可恢复。
        """
        if not entity_ids:
            return 0
        rows = self.run(
            f"""
            MATCH (e:`{BASE_ENTITY_LABEL}`) WHERE e.id IN $ids
              AND NOT (e)<-[:{MENTIONS_REL}]-()
              AND NOT EXISTS {{
                MATCH (e)--(o:`{BASE_ENTITY_LABEL}`)
                WHERE (o)<-[:{MENTIONS_REL}]-()
              }}
            DETACH DELETE e
            RETURN count(e) AS c
            """,
            ids=list(entity_ids),
        )
        n = int(rows[0]["c"]) if rows else 0
        if n:
            logger.info("清理孤立实体 %d 个(它们不再被任何块提到)", n)
        return n

    # ----------------------------------------------------------------- #
    # 实体对齐(中文 GraphRAG 最容易被漏的一层)
    # ----------------------------------------------------------------- #

    @staticmethod
    def normalize_name(name: str) -> str:
        """实体名归一化。

        只做安全的后处理 —— 全角转半角、去空白、去包裹的引号书名号。
        刻意**不做**「去掉『市/省/公司』后缀」这类激进规则:
        那会把「北京市」和「北京市政」合成一个,得不偿失。
        真正的同义合并交给向量相似度。
        """
        if not name:
            return ""
        # 全角 -> 半角(仅对 ASCII 可见区,中文标点保持原样)
        out = []
        for ch in name:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:
                out.append(chr(code - 0xFEE0))
            elif code == 0x3000:
                out.append(" ")
            else:
                out.append(ch)
        s = "".join(out).strip()
        s = re.sub(r"\s+", "", s)                      # 中文里的空格基本是噪声
        s = s.strip("「」『』【】《》\"'`()（）[]")        # 包裹符号
        return s

    def find_duplicate_groups(
        self, threshold: float | None = None, top_k: int = 10
    ) -> list[list[str]]:
        """找出应合并的实体组。

        用 Neo4j 原生向量索引取近邻,避免 O(N²) 两两比对。
        近邻关系是「A 像 B、B 像 C」这种链式结构,所以用并查集归组,
        而不是简单地把每对都合并 —— 否则 A-B、B-C 会把顺序搞出差异。
        """
        threshold = self.graph_cfg.entity_merge_threshold if threshold is None else threshold

        pairs = self.run(
            f"""
            MATCH (a:`{BASE_ENTITY_LABEL}`)
            WHERE a.embedding IS NOT NULL
            CALL db.index.vector.queryNodes($idx, $k, a.embedding)
              YIELD node AS b, score
            WHERE b.id <> a.id AND score >= $th
            RETURN a.id AS a, b.id AS b, score
            """,
            idx=VECTOR_INDEX_NAME,
            k=top_k,
            th=threshold,
        )
        if not pairs:
            return []

        # 并查集
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for p in pairs:
            union(p["a"], p["b"])

        groups: dict[str, list[str]] = defaultdict(list)
        for name in list(parent):
            groups[find(name)].append(name)
        return [sorted(g) for g in groups.values() if len(g) > 1]

    def merge_entity_group(self, names: Sequence[str], canonical: str | None = None) -> dict[str, Any]:
        """把一个组里的实体合并到代表节点上。

        canonical 不给就自动选:连接度最高的那个(它承载的关系最多,
        作为合并中心语义最完整);并列时取名字最短的。

        纯 Cypher 没法把关系类型参数化,所以这里先查出相邻的关系类型,
        逐个校验后拼进语句。关系类型做了正则白名单,不引入注入面。
        """
        if len(names) < 2:
            return {"merged": 0}

        if canonical is None:
            rows = self.run(
                f"""
                MATCH (e:`{BASE_ENTITY_LABEL}`) WHERE e.id IN $names
                OPTIONAL MATCH (e)-[r]-()
                RETURN e.id AS id, count(r) AS degree, size(e.id) AS len
                ORDER BY degree DESC, len ASC
                LIMIT 1
                """,
                names=list(names),
            )
            if not rows:
                return {"merged": 0}
            canonical = rows[0]["id"]

        dups = [n for n in names if n != canonical]
        moved_rels = 0

        for dup in dups:
            # 查出这个重复节点上的关系类型(参数化做不到,只能逐个来)
            rel_rows = self.run(
                f"MATCH (d:`{BASE_ENTITY_LABEL}` {{id: $dup}})-[r]-() "
                "RETURN DISTINCT type(r) AS t",
                dup=dup,
            )
            types = [r["t"] for r in rel_rows if r["t"]]
            for t in types:
                if not _REL_TYPE_RE.match(t):
                    logger.warning("跳过可疑的关系类型: %r", t)
                    continue
                # 出边:dup -[t]-> other,改接到 canonical
                self.run(
                    f"""
                    MATCH (d:`{BASE_ENTITY_LABEL}` {{id: $dup}})-[r:`{t}`]->(o)
                    WHERE NOT (canonical:`{BASE_ENTITY_LABEL}` {{id: $canon}})-[:`{t}`]->(o)
                    MATCH (canonical:`{BASE_ENTITY_LABEL}` {{id: $canon}})
                    CREATE (canonical)-[n:`{t}`]->(o)
                    SET n += properties(r)
                    RETURN count(n) AS c
                    """,
                    dup=dup,
                    canon=canonical,
                )
                # 入边:other -[t]-> dup
                self.run(
                    f"""
                    MATCH (o)-[r:`{t}`]->(d:`{BASE_ENTITY_LABEL}` {{id: $dup}})
                    WHERE NOT (o)-[:`{t}`]->(canonical:`{BASE_ENTITY_LABEL}` {{id: $canon}})
                    MATCH (canonical:`{BASE_ENTITY_LABEL}` {{id: $canon}})
                    CREATE (o)-[n:`{t}`]->(canonical)
                    SET n += properties(r)
                    RETURN count(n) AS c
                    """,
                    dup=dup,
                    canon=canonical,
                )
                moved_rels += 1

            # 记下别名,方便以后回溯 —— 合并是破坏性的,留个痕迹
            self.run(
                f"""
                MATCH (d:`{BASE_ENTITY_LABEL}` {{id: $dup}})
                MATCH (c:`{BASE_ENTITY_LABEL}` {{id: $canon}})
                SET c.aliases = coalesce(c.aliases, []) + [d.id]
                DETACH DELETE d
                """,
                dup=dup,
                canon=canonical,
            )

        logger.info("合并实体组 %s -> %r (处理 %d 个重复,涉及 %d 类关系)",
                    list(names), canonical, len(dups), moved_rels)
        return {"canonical": canonical, "merged": len(dups), "rel_types": moved_rels}

    def align_entities(self, threshold: float | None = None) -> dict[str, Any]:
        """跑一轮实体对齐。导入完成后调用。"""
        if not self.graph_cfg.entity_merge_enabled:
            return {"skipped": True, "reason": "GRAPH_MERGE_ENTITIES 关掉了"}

        groups = self.find_duplicate_groups(threshold)
        if not groups:
            return {"groups": 0, "merged": 0}

        total = 0
        for g in groups:
            r = self.merge_entity_group(g)
            total += r.get("merged", 0)
        return {"groups": len(groups), "merged": total}


_store: Neo4jGraphStore | None = None


def get_graph_store() -> Neo4jGraphStore:
    global _store
    if _store is None:
        _store = Neo4jGraphStore()
    return _store
