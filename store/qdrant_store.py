"""Qdrant 存储层:单 collection,稠密 + 稀疏双字段。

为什么是一个 collection 而不是两个
---------------------------------
稠密和稀疏描述的是**同一个** chunk。分两个 collection 就要保证两边
删除/更新同步,还要在应用层做 join —— 白白引入一处不一致的来源。
Qdrant 从 1.7 起原生支持 sparse vector,两个字段放一起,
一次 upsert 落盘,一次 query 出结果。

关于 RRF 的 k —— 这里有个到处都在传的错
------------------------------------
网上(以及大部分博客和教程)都说 RRF 的 k 是 60。那是原论文的推荐值。
但 **Qdrant 服务端的默认值是 2**,不是 60,而且它用零基 rank
(第一名 rank=0):

    score(d) = Σ 1 / (k + (r_d + 1)/w_r - 1)

k 从 v1.16.0 起可以传参。不传就等于默认拿了 k=2 —— 行为差别很大:
k=2 时 rank0 得 1/2、rank1 得 1/3,头名权重是次名的 1.5 倍,明显的
赢者通吃;k=60 时是 1/60 对 1/61,几乎只是"数票"。
所以这里显式传 config 里的 rrf_k,不依赖服务端默认值。

关于稀疏向量的 IDF
-----------------
建索引时**不开** Modifier.IDF。bge-m3 的学到的权重自带重要度,
再乘一遍 IDF 就是重复计权。(只有走自建 BM25 那一路、存的是原始
词频时才需要 IDF,那种情况下 IDF 应该放在**查询向量**里。)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from config import QdrantConfig, RetrievalConfig, get_settings
from embed.sparse_convert import to_sparse_vector
from store.versioning import (
    STATUS_CURRENT,
    STATUS_FIELD,
    STATUS_SUPERSEDED,
    VERSION_FIELDS,
    status_of,
)

logger = logging.getLogger(__name__)

# 固定命名空间,用来把 (doc_id, chunk_index) 映射成稳定的 point id。
# ⚠️ 这个 UUID 一旦上线就不能改 —— 改了等于所有已有数据的 id 全部失效。
_POINT_NAMESPACE = uuid.UUID("a3f1c2d4-5b6e-4f7a-8c9d-0e1f2a3b4c5d")

# 一批 upsert 多少条。太大单次请求体过大,太小来回开销高。
UPSERT_BATCH = 256

# payload 索引。放在模块级是为了让「建 collection」和「给已有 collection
# 补索引」走的是**同一份清单** —— 写成两份,加字段时必然只改一处。
PAYLOAD_INDEXES: tuple[tuple[str, "qm.PayloadSchemaType"], ...] = (
    ("doc_id", qm.PayloadSchemaType.KEYWORD),
    ("content_hash", qm.PayloadSchemaType.KEYWORD),
    ("source", qm.PayloadSchemaType.KEYWORD),
    ("chunk_index", qm.PayloadSchemaType.INTEGER),
    # status:每一次召回都会带上它(见 store/versioning.py),是这条查询
    # 路径上唯一的过滤字段。不建索引的话它退化成逐点全扫 ——
    # 加了过滤反而比不加慢,而且不报错。
    (STATUS_FIELD, qm.PayloadSchemaType.KEYWORD),
)


def point_id(doc_id: str, chunk_index: int) -> str:
    """确定性 id:同一文档重导时覆盖而不是新增。

    用 uuid5 而不是自增数字,是因为自增 id 在多进程导入时会撞;
    用 uuid5 而不是随机 uuid4,是为了幂等 —— 重跑导入不会产生重复。
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}::{chunk_index}"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Chunk:
    """落库的最小单元。向量已经算好了,存储层不做嵌入。"""

    doc_id: str
    chunk_index: int
    text: str                                  # 原文,用于展示和引用
    dense: Sequence[float]
    sparse: Mapping[str, float]                # {"token_id": weight}
    context: str = ""                          # Anthropic 上下文检索写的那句定位语
    source: str = ""                           # 文件路径 / URL
    title: str = ""
    content_hash: str = ""                     # 原文指纹,用于增量去重
    # 版本化字段。见 `store/versioning.py` —— 状态取值和缺省语义在那边定义,
    # 这里只是把它落到 payload 上,不要在这一层另写一套判断。
    # `status` 默认就给 `current`:新入库的东西一律有效,而且**显式写出来**,
    # 不靠"缺字段当有效"那条兼容路径 —— 那条是给加字段之前的老数据用的。
    status: str = STATUS_CURRENT
    doc_version: str = ""
    effective_from: str = ""
    effective_to: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return point_id(self.doc_id, self.chunk_index)

    @property
    def embed_text(self) -> str:
        """真正被编码的文本 = 定位语 + 原文。

        存的时候分开存(引用要原文),编码时拼起来 —— 这样上下文增强
        可以随时开关而不影响已存数据的内容。
        """
        if self.context:
            return f"{self.context}\n{self.text}"
        return self.text

    def to_point(self, dense_field: str, sparse_field: str) -> qm.PointStruct:
        sv = to_sparse_vector(self.sparse)
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "context": self.context,
            "source": self.source,
            "title": self.title,
            "content_hash": self.content_hash,
            "char_count": len(self.text),
            "ingested_at": _now_iso(),
            # status **一定写**,哪怕是默认值:缺字段在 Qdrant 里是"匹配不上
            # 等值条件"的,召回侧要靠一个额外的 OR 分支才兜得住。新数据不留
            # 这个坑,那个 OR 就只需要服务于老数据。
            "status": self.status,
        }
        # 其余版本字段是**可选**的,认不出就不写 —— 写空字符串会让"有版本号
        # 但为空"和"没有版本号"变成两种状态,而它们没有任何行为差异。
        for fname, val in (
            ("doc_version", self.doc_version),
            ("effective_from", self.effective_from),
            ("effective_to", self.effective_to),
        ):
            if val:
                payload[fname] = val
        # extra 只放标量 —— 嵌套结构没法建 payload 索引,过滤会退化成全扫
        for k, v in self.extra.items():
            if k == STATUS_FIELD and v != self.status:
                # `extra` 是从文档 metadata 直接铺开的,内容不受这一层控制。
                # 放它覆盖 status,等于让一篇文档的 metadata 把整篇从召回里
                # 抹掉 —— 而且是那种不报错、只是"查不到"的抹法。取值恰好
                # 相同就放行,那种情况下本来也没有差别。
                raise ValueError(
                    f"extra 里的 {STATUS_FIELD}={v!r} 与 Chunk.status={self.status!r} "
                    "冲突。状态只能由 mark_superseded 改,不能从 metadata 塞进来。"
                )
            if isinstance(v, (str, int, float, bool)) or v is None:
                payload[k] = v
        return qm.PointStruct(
            id=self.id,
            vector={dense_field: [float(x) for x in self.dense], sparse_field: sv},
            payload=payload,
        )


class QdrantStore:
    """collection 的建/写/查。线程安全交给底层 HTTP 客户端。"""

    def __init__(
        self,
        cfg: QdrantConfig | None = None,
        retrieval: RetrievalConfig | None = None,
    ) -> None:
        s = get_settings()
        self.cfg = cfg or s.qdrant
        self.retrieval = retrieval or s.retrieval
        self._client: QdrantClient | None = None

    # ----------------------------------------------------------------- #
    # 连接
    # ----------------------------------------------------------------- #

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"url": self.cfg.url, "timeout": self.cfg.timeout}
            if self.cfg.api_key:
                kwargs["api_key"] = self.cfg.api_key
            self._client = QdrantClient(**kwargs)
            logger.info("连接 Qdrant: %s", self.cfg.url)
        return self._client

    def health(self) -> dict[str, Any]:
        try:
            info = self.client.get_collections()
            return {"ok": True, "collections": [c.name for c in info.collections]}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ----------------------------------------------------------------- #
    # collection 生命周期
    # ----------------------------------------------------------------- #

    def exists(self) -> bool:
        return self.client.collection_exists(self.cfg.collection)

    def ensure_collection(self, dense_dim: int = 1024, recreate: bool = False) -> None:
        """建立 collection。已存在时**不**动它 —— 除非显式 recreate=True。"""
        if self.exists():
            if not recreate:
                self._check_dim(dense_dim)
                # ⚠️ 这个分支里**也必须**补索引。
                # 加字段时最容易漏的就是这里:老库早就存在,`ensure_collection`
                # 一看存在就返回,新加的索引永远建不上 —— 而过滤照样能跑,
                # 只是退化成逐点全扫。没有报错、没有日志,只是所有查询
                # 一起变慢,没人会往"少建了个索引"上想。
                self.ensure_payload_indexes()
                return
            logger.warning("recreate=True,正在删除已有 collection: %s", self.cfg.collection)
            self.client.delete_collection(self.cfg.collection)

        self.client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config={
                self.cfg.dense_field: qm.VectorParams(
                    size=dense_dim,
                    distance=qm.Distance.COSINE,
                    # 建 HNSW 索引:召回快。数据量小时关掉能省内存,但生产默认开
                    on_disk=False,
                )
            },
            sparse_vectors_config={
                self.cfg.sparse_field: qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False),
                    # ⚠️ 保持 NONE。开了 IDF 就是和 bge-m3 的权重重复计权,
                    # 详见模块开头的说明。
                    modifier=qm.Modifier.NONE,
                )
            },
        )

        self.ensure_payload_indexes()

        logger.info(
            "已建 collection %s (dense=%s/%d, sparse=%s)",
            self.cfg.collection, self.cfg.dense_field, dense_dim, self.cfg.sparse_field,
        )

    def ensure_payload_indexes(self) -> None:
        """把 payload 索引补齐。**幂等**,已存在就什么都不做。

        抽成公开方法是为了让"已存在的 collection"也能补上新加的索引 ——
        见上面 `ensure_collection` 里那段。Qdrant 的 `create_payload_index`
        对已存在的索引是幂等的(重复建不报错),所以直接建、不用先查。
        """
        # 没有索引,按 doc_id 删除/按 hash 去重会退化成全表扫
        for fname, schema in PAYLOAD_INDEXES:
            self.client.create_payload_index(
                collection_name=self.cfg.collection,
                field_name=fname,
                field_schema=schema,
            )

    def _check_dim(self, dense_dim: int) -> None:
        """已经存在的 collection 维度对不上时**报错而不是覆盖** ——
        静默重建等于悄悄删光用户的数据。"""
        info = self.client.get_collection(self.cfg.collection)
        params = info.config.params.vectors
        if isinstance(params, dict) and self.cfg.dense_field in params:
            actual = params[self.cfg.dense_field].size
            if actual != dense_dim:
                raise ValueError(
                    f"collection {self.cfg.collection!r} 的稠密维度是 {actual},"
                    f"但当前配置要写 {dense_dim}。"
                    f"确认要重建请显式传 recreate=True(会删掉已有数据)。"
                )

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.cfg.collection, exact=True).count

    def info(self) -> dict[str, Any]:
        if not self.exists():
            return {"exists": False, "collection": self.cfg.collection}
        i = self.client.get_collection(self.cfg.collection)
        return {
            "exists": True,
            "collection": self.cfg.collection,
            "points": i.points_count,
            "status": str(i.status),
            "dense_dim": i.config.params.vectors[self.cfg.dense_field].size,
        }

    # ----------------------------------------------------------------- #
    # 写入
    # ----------------------------------------------------------------- #

    def upsert(self, chunks: Iterable[Chunk], wait: bool = True) -> int:
        """批量写入。id 是确定性的,重复写同一 chunk 就是覆盖。"""
        buf: list[qm.PointStruct] = []
        total = 0
        for ch in chunks:
            buf.append(ch.to_point(self.cfg.dense_field, self.cfg.sparse_field))
            if len(buf) >= UPSERT_BATCH:
                total += self._flush(buf, wait=False)
                buf = []
        if buf:
            total += self._flush(buf, wait=wait)
        return total

    def _flush(self, points: list[qm.PointStruct], wait: bool) -> int:
        self.client.upsert(
            collection_name=self.cfg.collection, points=points, wait=wait
        )
        return len(points)

    def existing_hashes(self, hashes: Sequence[str]) -> set[str]:
        """这批内容指纹里,库里已经有哪些 —— 增量导入靠它跳过没变过的块。"""
        if not hashes or not self.exists():
            return set()
        found: set[str] = set()
        # MatchAny 一次问一批,比逐个 count 快得多
        for i in range(0, len(hashes), 512):
            batch = list(hashes[i : i + 512])
            records, _ = self.client.scroll(
                collection_name=self.cfg.collection,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="content_hash", match=qm.MatchAny(any=batch))]
                ),
                limit=len(batch),
                with_payload=["content_hash"],
                with_vectors=False,
            )
            for r in records:
                h = (r.payload or {}).get("content_hash")
                if h:
                    found.add(h)
        return found

    def fetch_chunks(
        self, keys: Sequence[tuple[str, int]]
    ) -> list[qm.Record]:
        """按 (doc_id, chunk_index) 精确取块。

        图检索要用它:实体向量搜到「1号杆塔」,多跳走到「绝缘子」,
        而那些提到绝缘子的块**不包含查询词**,稠密/稀疏召回都够不着。
        只能靠图反查出块 id 再把块本身捞回来。

        这里用的是确定性 point_id(uuid5),所以「查得到 id」等价于
        「拿得回内容」—— 不需要在别处冗余存一份正文。
        """
        if not keys or not self.exists():
            return []
        ids = [point_id(d, i) for d, i in keys]
        out: list[qm.Record] = []
        for i in range(0, len(ids), 512):
            out.extend(
                self.client.retrieve(
                    collection_name=self.cfg.collection,
                    ids=ids[i : i + 512],
                    with_payload=True,
                    with_vectors=False,
                )
            )
        return out

    def delete_by_doc(self, doc_id: str) -> None:
        """按文档删除。区别于「清空整个 collection」——
        清空是个危险操作,不该出现在常规流程里。"""
        self.client.delete(
            collection_name=self.cfg.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )

    def list_docs(self) -> list[dict[str, Any]]:
        """列出库里有哪些文档 —— 运维查询,不参与检索。

        带上版本字段:界面上要能标出「这篇已废止」。**不能靠再查一次**
        —— 列表和标记是两次独立的读,中间有人标记了失效,列表就会显示
        「有效」而它其实已经查不到了 (见 `retrieve/hybrid.py` 里同一条
        理由)。所以版本信息跟着列表本身一起取出来。

        `status` 是**文档级**的,但底层是逐块存的,所以这里按块聚合:
        只要有一块是 superseded 就算这篇被标记过 —— 偏向"标出来"而不是
        "藏起来",因为漏标会让用户对着查不到的结果找不到原因。真的有块
        不一致时额外给 `mixed=True`,让界面能提示"这篇状态不统一"。
        """
        if not self.exists():
            return []
        seen: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.cfg.collection,
                limit=512,
                offset=offset,
                with_payload=[
                    "doc_id", "source", "title",
                    STATUS_FIELD, *VERSION_FIELDS,
                ],
                with_vectors=False,
            )
            for r in records:
                p = r.payload or {}
                d = p.get("doc_id")
                if not d:
                    continue
                st = status_of(p)
                if d not in seen:
                    seen[d] = {
                        "doc_id": d,
                        "source": p.get("source", ""),
                        "title": p.get("title", ""),
                        "chunks": 0,
                        "status": st,
                        "mixed": False,
                        # 版本字段取**第一块非空**的值。同一篇文档的各块应当
                        # 一致(入库时按文件名统一写),真不一致时 `mixed`
                        # 会把状态标出来,版本号取哪个都不至于误导。
                        "doc_version": "",
                        "effective_from": "",
                        "effective_to": "",
                    }
                rec = seen[d]
                rec["chunks"] += 1
                if st != rec["status"]:
                    rec["mixed"] = True
                    if st == STATUS_SUPERSEDED:
                        # 一篇里出现了失效块 → 整篇按"已标记"显示
                        rec["status"] = STATUS_SUPERSEDED
                for k in VERSION_FIELDS:
                    if not rec[k] and p.get(k):
                        rec[k] = str(p.get(k))
            if offset is None:
                break
        return sorted(seen.values(), key=lambda x: x["doc_id"])

    # ----------------------------------------------------------------- #
    # payload 就地改(受控写工具 / 版本回溯)
    # ----------------------------------------------------------------- #

    def doc_records(self, doc_id: str) -> list[qm.Record]:
        """按 doc_id 取出全部块记录(带 payload,**不带向量**)。

        改字段之前得先知道原值 —— 回滚要用,见 `agent/tools.py` 的
        `mark_superseded`。不取向量是因为这条路径只动元数据,把 1024 维
        稠密向量拉回来纯属浪费带宽。
        """
        if not doc_id or not self.exists():
            return []
        out: list[qm.Record] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.cfg.collection,
                scroll_filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                ),
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(records)
            if offset is None:
                break
        return out

    def set_payload(
        self,
        doc_id: str,
        payload: Mapping[str, Any],
        *,
        chunk_index: int | None = None,
    ) -> int:
        """就地改 payload,**不碰向量**。返回改了几块。

        为什么不是 delete + 重新 upsert:那要重算嵌入(贵),而且中间态
        一旦失败这篇文档就整篇没了 —— 一次「改元数据」不该有丢数据的
        可能。set_payload 是原地改,失败就还是老值。

        `chunk_index` 给了就只改那一块,否则改整篇。
        """
        recs = self.doc_records(doc_id)
        ids = [
            r.id
            for r in recs
            if chunk_index is None or (r.payload or {}).get("chunk_index") == chunk_index
        ]
        if not ids:
            return 0
        self.client.set_payload(
            collection_name=self.cfg.collection,
            payload=dict(payload),
            # 用显式 id 列表而不是 FilterSelector:我们刚刚才 scroll 出这批
            # id,期间若有别的写入落了同一篇文档,按 filter 改会连新块一起改掉,
            # 而按 id 改只动我们看过的那批 —— 审计里记的"改了几块"才是真的。
            points=qm.PointIdsList(points=ids),
            wait=True,
        )
        return len(ids)

    # ----------------------------------------------------------------- #
    # 检索
    # ----------------------------------------------------------------- #

    def query_hybrid(
        self,
        dense: Sequence[float] | None,
        sparse: Mapping[str, float] | None,
        limit: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        rrf_k: int | None = None,
        query_filter: qm.Filter | None = None,
        weights: tuple[float, float] | None = None,
    ) -> list[qm.ScoredPoint]:
        """双路召回 + 服务端 RRF 融合。

        只有一路有向量时自动退化成单路查询,不做无意义的融合。
        weights=(稠密权重, 稀疏权重) 是可选的加权 RRF;
        加权版比裸 RRF 更容易调,但需要真的调过再用,别默认开。
        """
        r = self.retrieval
        limit = limit or r.fusion_top_k
        dense_top_k = dense_top_k or r.dense_top_k
        sparse_top_k = sparse_top_k or r.sparse_top_k
        k = r.rrf_k if rrf_k is None else rrf_k

        prefetches: list[qm.Prefetch] = []
        if dense is not None:
            prefetches.append(
                qm.Prefetch(
                    query=[float(x) for x in dense],
                    using=self.cfg.dense_field,
                    limit=dense_top_k,
                    filter=query_filter,
                )
            )
        sparse_sv = None
        if sparse:
            sparse_sv = to_sparse_vector(sparse)
            if sparse_sv.indices:
                prefetches.append(
                    qm.Prefetch(
                        query=sparse_sv,
                        using=self.cfg.sparse_field,
                        limit=sparse_top_k,
                        filter=query_filter,
                    )
                )

        if not prefetches:
            return []

        # 单路:直接查,别套融合 —— 融合一层只有一路的排名纯属浪费
        if len(prefetches) == 1:
            p = prefetches[0]
            resp = self.client.query_points(
                collection_name=self.cfg.collection,
                query=p.query,
                using=p.using,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            return list(resp.points)

        rrf = qm.Rrf(k=k, weights=list(weights) if weights else None)
        resp = self.client.query_points(
            collection_name=self.cfg.collection,
            prefetch=prefetches,
            query=qm.RrfQuery(rrf=rrf),
            limit=limit,
            with_payload=True,
        )
        return list(resp.points)

    def query_paths(
        self,
        dense: Sequence[float] | None,
        sparse: Mapping[str, float] | None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        query_filter: qm.Filter | None = None,
    ) -> tuple[list[qm.ScoredPoint], list[qm.ScoredPoint]]:
        """分别返回两路的原始排名,不做融合。

        两个用处:
          - 调试「到底哪一路召回了正确结果」—— 只看融合后的名单是看不出来的
          - 应用层自己融合(比如要在 RRF 前加业务规则)
        """
        r = self.retrieval
        dense_top_k = dense_top_k or r.dense_top_k
        sparse_top_k = sparse_top_k or r.sparse_top_k

        d_hits: list[qm.ScoredPoint] = []
        if dense is not None:
            resp = self.client.query_points(
                collection_name=self.cfg.collection,
                query=[float(x) for x in dense],
                using=self.cfg.dense_field,
                limit=dense_top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            d_hits = list(resp.points)

        s_hits: list[qm.ScoredPoint] = []
        if sparse:
            sv = to_sparse_vector(sparse)
            if sv.indices:
                resp = self.client.query_points(
                    collection_name=self.cfg.collection,
                    query=sv,
                    using=self.cfg.sparse_field,
                    limit=sparse_top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )
                s_hits = list(resp.points)

        return d_hits, s_hits


_client_singleton: QdrantStore | None = None


def get_store() -> QdrantStore:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = QdrantStore()
    return _client_singleton


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Any]],
    k: int = 60,
    key=lambda x: x,
) -> list[tuple[Any, float]]:
    """应用层 RRF。跟 Qdrant 的公式对齐(零基 rank):

        score(d) = Σ 1 / (k + r_d)

    这里用零基是因为要和 Qdrant 服务端的结果可比 —— 混用两套 rank
    基准是排查排序问题时的经典陷阱。
    """
    scores: dict[Any, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            kk = key(item)
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
