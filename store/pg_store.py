"""不依赖 APOC 的 Neo4j PropertyGraphStore。

为什么需要这个子类
-----------------
LlamaIndex 的 `Neo4jPropertyGraphStore` 有四处硬依赖 APOC:

    refresh_schema()      apoc.meta.data / apoc.meta.subGraph / apoc.schema.nodes
    _enhanced_schema_cypher()  apoc.meta.*
    upsert_nodes()        apoc.map.clean / apoc.create.addLabels
    upsert_relations()    apoc.merge.relationship

而 APOC 是 Neo4j 的**插件**,Community 版默认不带 —— 要单独下 jar、
改 neo4j.conf 加 `dbms.security.procedures.unrestricted=apoc.*`、重启服务。

实测本机 Neo4j 5.26.9 community:带 `refresh_schema=True` 构造直接抛

    Neo.ClientError.Procedure.ProcedureNotFound:
    There is no procedure with the name `apoc.meta.data` registered ...

**但读路径完全不用 APOC。** `vector_query` / `get_rel_map` / `structured_query`
全是普通 Cypher,所以向量检索本身不受影响 —— 只有「建图」和「刷 schema」会挂。
这一点很坑:导入会先烧掉几百次 LLM 实体抽取调用,才在写图那一步失败,
而错误信息是一句「没有 apoc.meta.data」,和实体抽取看不出关系。

所以这里只重写那四个方法,用等价的普通 Cypher 实现:

    refresh_schema      apoc.meta.*          -> db.schema.nodeTypeProperties() 等内置过程
    upsert_nodes        apoc.map.clean       -> SET e += row.properties
                        apoc.create.addLabels -> 按标签分组后插值(Cypher 标签不能参数化)
    upsert_relations    apoc.merge.relationship -> MERGE (a)-[r:类型]->(b)

标签插值会引入注入面,所以插值前先过白名单正则 —— 和 graph_store.py 里
`merge_entity_group` 处理关系类型用的是同一套办法,不引入新范式。

如果以后装上了 APOC,这个子类仍然能用(行为等价),所以不值得加开关来回切。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    Relation,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

logger = logging.getLogger(__name__)

BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
VECTOR_INDEX_NAME = "entity"          # 必须和 graph_store.py 里的一致
CHUNK_SIZE = 1000
MENTIONS_REL = "MENTIONS"
# 图里表示「块来自文档」的关系,不是语义关系,列 schema 时要排掉
KG_SOURCE_REL = "SOURCE"

# Cypher 没法把标签参数化,只能插值。插值前必须过这个正则。
#
# 注意**不能只放 ASCII**:Neo4j 的标签是支持 Unicode 的,中文类型名完全合法。
# 一开始这里写的是 ^[A-Za-z_][A-Za-z0-9_]*$,结果中文语料抽出来的
# 「设备」「缺陷」「包含」全被判非法、统统回退成 Entity ——
# 图会退化成一张没有类型的边列表,GraphRAG 的类型结构全丢,
# 而现场只留下一行 warning,几乎不可能发现。
#
# 这里挡的是真正会破坏 Cypher 的字符:反引号、引号、反斜杠、空白、
# 大括号、方括号、圆括号、分号、逗号、$ 和 <>。
#
# 反过来,像 `/` `+` `-` `.` `·` 这些**看着可疑但其实安全**的字符是放行的 ——
# 标签在 Cypher 里是用反引号包起来的(`` e:`输电/杆塔` ``),这些字符
# 逃不出反引号。放行是为了尽量保住 LLM 抽出来的原始类型名,
# 而不是把它降级成 Entity 把类型结构丢掉。
_LABEL_RE = re.compile(r"^[\w㐀-䶿一-鿿·.\-+/]{1,64}$", re.UNICODE)

# 标签非法时的兜底。宁可所有实体挤在一个通用标签下,也不要因为
# 模型吐了个 `输电线路/杆塔` 这种带斜杠的标签就让整批导入崩掉。
_FALLBACK_ENTITY_LABEL = "Entity"
_FALLBACK_REL_LABEL = "RELATED_TO"


def _dump(obj: Any) -> dict:
    """pydantic v2 下 `.dict()` 已废弃,优先用 `model_dump()`。

    输出内容两者一致,只是前者会刷一屏 DeprecationWarning,
    而导入时每个实体都调一次,日志会被淹掉。
    """
    fn = getattr(obj, "model_dump", None)
    if callable(fn):
        return fn()
    return obj.dict()


def _safe_label(raw: Any, fallback: str) -> str:
    """校验标签能不能安全插进 Cypher。不合法就回退,不抛异常。

    实体类型是 LLM 抽取出来的,属于**不可信输入** —— 直接拼进 Cypher
    就是注入面。而中文字符、斜杠、空格在标签里都不合法。
    """
    s = str(raw or "").strip()
    if _LABEL_RE.match(s):
        return s
    if s:
        logger.warning("标签 %r 不适合做 Cypher 标签,回退成 %r", s[:40], fallback)
    return fallback


class APOCFreeNeo4jPropertyGraphStore(Neo4jPropertyGraphStore):
    """把 APOC 依赖换掉的 Neo4j 图存储。接口和父类完全一致。"""

    # ----------------------------------------------------------------- #
    # schema
    # ----------------------------------------------------------------- #

    def refresh_schema(self) -> None:
        """用 Neo4j 内置的 db.schema.* 过程拼 schema,不碰 APOC。

        `get_schema_str` 只读 node_props / rel_props / relationships 三个键,
        所以这里只需要把它们填对。
        """
        node_props = self.structured_query(
            """
            CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes, mandatory
            RETURN nodeLabels AS labels,
                   collect({property: propertyName, type: propertyTypes[0], mandatory: mandatory}) AS properties
            """
        ) or []

        rel_props = self.structured_query(
            """
            CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes, mandatory
            RETURN relType AS type,
                   collect({property: propertyName, type: propertyTypes[0], mandatory: mandatory}) AS properties
            """
        ) or []

        # 抽样出「哪种节点连了哪种节点、用的什么关系」,给 text-to-cypher 用。
        # LIMIT 是因为大图上全量 DISTINCT 很贵,而 schema 只需要样本。
        relationships = self.structured_query(
            f"""
            MATCH (a)-[r]->(b)
            WHERE NOT type(r) IN ['{MENTIONS_REL}', '{KG_SOURCE_REL}', '_Bloom_HAS_SCENE_']
            RETURN DISTINCT head(labels(a)) AS start, type(r) AS type, head(labels(b)) AS end
            LIMIT 1000
            """
        ) or []

        try:
            constraint = self.structured_query("SHOW CONSTRAINTS")
            index = self.structured_query("SHOW INDEXES")
        except Exception as exc:  # noqa: BLE001 - 只读用户可能没权限看 schema
            logger.debug("取约束/索引失败(多半是权限): %s", exc)
            constraint, index = [], []

        self.structured_schema = {
            # 键必须是**字符串**:db.schema.nodeTypeProperties() 给的 labels 是列表,
            # 直接拿列表当字典键会 TypeError
            "node_props": {
                self._primary_label(el["labels"]): el["properties"] for el in node_props
            },
            "rel_props": {el["type"]: el["properties"] for el in rel_props},
            "relationships": relationships,
            "metadata": {"constraint": constraint, "index": index},
        }
        self.schema = "\n".join(
            f"节点 {k}: {[p['property'] for p in v]}"
            for k, v in self.structured_schema["node_props"].items()
        )

    @staticmethod
    def _primary_label(labels: List[str]) -> str:
        """从标签列表里挑一个当键 —— 跳过 __Entity__/__Node__ 这类内部标签。"""
        for lb in labels or []:
            if lb not in (BASE_ENTITY_LABEL, BASE_NODE_LABEL):
                return lb
        return "__Entity__"

    # ----------------------------------------------------------------- #
    # 写:节点
    # ----------------------------------------------------------------- #

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**_dump(item), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**_dump(item), "id": item.id})

        if chunk_dicts:
            for i in range(0, len(chunk_dicts), CHUNK_SIZE):
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (c:`{BASE_NODE_LABEL}` {{id: row.id}})
                    SET c.text = row.text, c:Chunk
                    WITH c, row
                    SET c += row.properties
                    WITH c, row
                    WHERE row.embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
                    RETURN count(*)
                    """,
                    param_map={"data": chunk_dicts[i : i + CHUNK_SIZE]},
                )

        if not entity_dicts:
            return

        # 按标签分组:标签没法参数化,分组后每组只需要一条语句,
        # 而不是每行一条(那样 N 个实体就是 N 次 round trip)
        grouped: Dict[str, List[dict]] = {}
        for row in entity_dicts:
            lb = _safe_label(row.get("label"), _FALLBACK_ENTITY_LABEL)
            grouped.setdefault(lb, []).append(row)

        for label, rows in grouped.items():
            for i in range(0, len(rows), CHUNK_SIZE):
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (e:`{BASE_NODE_LABEL}` {{id: row.id}})
                    SET e += row.properties
                    SET e.name = row.name, e:`{BASE_ENTITY_LABEL}`, e:`{label}`
                    WITH e, row
                    CALL (e, row) {{
                        WITH e, row
                        WHERE row.embedding IS NOT NULL
                        CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
                        RETURN count(*) AS count
                    }}
                    WITH e, row
                    WHERE row.properties IS NOT NULL
                      AND row.properties.triplet_source_id IS NOT NULL
                    MERGE (c:`{BASE_NODE_LABEL}` {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:{MENTIONS_REL}]-(c)
                    """,
                    param_map={"data": rows[i : i + CHUNK_SIZE]},
                )

    # ----------------------------------------------------------------- #
    # 写:关系
    # ----------------------------------------------------------------- #

    def upsert_relations(self, relations: List[Relation]) -> None:
        if not relations:
            return

        grouped: Dict[str, List[dict]] = {}
        for r in relations:
            row = _dump(r)
            lb = _safe_label(row.get("label"), _FALLBACK_REL_LABEL)
            grouped.setdefault(lb, []).append(row)

        for label, rows in grouped.items():
            for i in range(0, len(rows), CHUNK_SIZE):
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (source:`{BASE_NODE_LABEL}` {{id: row.source_id}})
                    ON CREATE SET source:Chunk
                    MERGE (target:`{BASE_NODE_LABEL}` {{id: row.target_id}})
                    ON CREATE SET target:Chunk
                    WITH source, target, row
                    MERGE (source)-[rel:`{label}`]->(target)
                    SET rel += coalesce(row.properties, {{}})
                    RETURN count(*)
                    """,
                    param_map={"data": rows[i : i + CHUNK_SIZE]},
                )


__all__ = [
    "APOCFreeNeo4jPropertyGraphStore",
    "BASE_ENTITY_LABEL",
    "BASE_NODE_LABEL",
    "VECTOR_INDEX_NAME",
]
