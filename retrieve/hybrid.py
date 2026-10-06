"""混合检索:bge-m3 双路召回 → 服务端 RRF → 客户端交叉编码器重排。

漏斗形状
-------
    query
      ├─ bge-m3 一次前向 ─┬─ 稠密  → Qdrant dense   top-50
      │                   └─ 稀疏  → Qdrant sparse  top-50
      │                              ↓
      │                      服务端 RRF(k=60) 融合   top-20
      │                              ↓
      └──────────────────── bge-reranker-v2-m3 重排 → 阈值过滤 → top-5

为什么要三级而不是一步到位
  - 稠密和稀疏各召 50 条,是因为它们**各自都会漏**:稠密对同义不同字强、
    稀疏(在中文上退化成字符级)对精确字面强。取并集比任何单路都全。
  - RRF 只用了排名不用分数,所以不用管两路分数量纲不可比这件事。
  - 重排最贵(每对候选都要过一次 Transformer),所以放在最后、候选最少的时候。
    反过来拿重排去做召回是不可行的。

一个容易忽略的点
  重排分数过 sigmoid 后分布极其尖锐(实测相关 0.9+,不相关 0.04 以下),
  所以固定取 top-N 会白送一堆近零分的垃圾进上下文。这里加了分数阈值,
  并且保证至少留 rerank_min_keep 条 —— 两条规则缺一不可:
  只有阈值会在全不相关时返回空,只有 top-N 会在只有一条相关时塞四条垃圾。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from qdrant_client import models as qm

from config import RetrievalConfig, get_settings
from embed.bge_m3 import get_embedder
from retrieve.reranker import get_reranker
from store.qdrant_store import QdrantStore, get_store, point_id
from store.versioning import status_of, visible_filter

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """一条检索结果。带够引用所需的信息,不只是文本。"""

    text: str
    doc_id: str
    chunk_index: int
    source: str = ""
    title: str = ""
    context: str = ""

    score: float = 0.0            # 最终排序分(重排分,或未重排时的 RRF 分)
    rrf_score: float = 0.0
    rerank_score: float | None = None
    dense_rank: int | None = None   # 在稠密那一路的排名(0 基),None = 没被召回
    sparse_rank: int | None = None
    # 版本信息。**必须跟着结果一路走到界面上**:用户看到一条答案,
    # 要能当场知道「这条是现行版还是已被废止的那一版」。检索时若把
    # 它丢在 payload 里不带出来,前端就只能再按 doc_id 回头查一次库,
    # 而那次查询拿到的状态**可能已经和这次召回时不同**(中间有人标记了
    # 失效)—— 界面上就会出现"召回到了却说它是失效的"这种自相矛盾的显示。
    # 空串 = 没写过这个字段,不是"未知版本"。
    status: str = ""
    doc_version: str = ""
    effective_from: str = ""
    effective_to: str = ""
    # 后端特有的溯源信息。混合检索往里放 ranking 细节,
    # 图检索往里放命中的实体和多跳路径 —— 上层统一按 RetrievedChunk 处理。
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """给 LLM 看的来源标签。要求能唯一定位到文档里的位置。"""
        name = self.title or self.source or self.doc_id
        return f"{name} (块 {self.chunk_index})"

    @classmethod
    def from_payload(cls, payload: dict, score: float = 0.0) -> "RetrievedChunk":
        """从 Qdrant 的 payload 建一条结果。

        图检索那条路也要用:它按 id 从 Qdrant 取回块(不是打分检索),
        所以拿到的是 payload 而不是 ScoredPoint,但字段口径必须一致 ——
        否则同一篇文档经两条后端出来,引用标签会不一样。
        """
        p = payload or {}
        return cls(
            text=p.get("text", ""),
            doc_id=p.get("doc_id", ""),
            chunk_index=int(p.get("chunk_index", 0) or 0),
            source=p.get("source", ""),
            title=p.get("title", ""),
            context=p.get("context", ""),
            score=float(score),
            rrf_score=float(score),
            # 走 `status_of` 而不是直接取 `p.get("status")`:那一个函数是
            # 「缺字段/取值不认识一律当 current」这条约定的**唯一**实现,
            # 这里再写一遍 `or STATUS_CURRENT` 就是第二份口径 —— 以后改
            # 约定必然只改一处,另一处变成鬼故事。
            status=status_of(p),
            doc_version=str(p.get("doc_version") or ""),
            effective_from=str(p.get("effective_from") or ""),
            effective_to=str(p.get("effective_to") or ""),
        )

    @classmethod
    def from_point(cls, point: qm.ScoredPoint) -> "RetrievedChunk":
        return cls.from_payload(point.payload or {}, score=float(point.score))


@dataclass
class RetrievalDebug:
    """检索过程的可观测信息。

    生产环境排查「为什么这条没被召回」时,只看最终 5 条是查不出问题的 ——
    得知道它到底有没有进候选、在每一路排第几。
    """

    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    reranked_hits: int = 0
    dropped_by_threshold: int = 0
    dense_order: list[str] = field(default_factory=list)
    sparse_order: list[str] = field(default_factory=list)
    fused_order: list[str] = field(default_factory=list)


def apply_rerank(
    query: str,
    results: list[RetrievedChunk],
    top_k: int,
    cfg: RetrievalConfig,
    reranker: Any,
    debug: RetrievalDebug | None = None,
) -> list[RetrievedChunk]:
    """交叉编码器重排 + 阈值过滤。写成自由函数是给图后端复用的。

    两条后端如果各自实现一遍「阈值 + 最少保留条数」,迟早会在某次调参里
    只改一边,然后同一个问题经不同后端返回的条数不一样 —— 这种不一致
    在线上极难定位(因为两个后端单独看都"对")。
    """
    if not results:
        return []

    # 重排看的是和原文一起编码的那段文本(定位语 + 原文),
    # 但展示给用户/LLM 的仍是原文。两者不能混。
    passages = [
        f"{r.context}\n{r.text}" if r.context else r.text for r in results
    ]
    hits = reranker.rerank(query, passages, top_n=len(results))

    reranked: list[RetrievedChunk] = []
    for h in hits:
        r = results[h.index]
        r.rerank_score = h.score
        r.score = h.score
        reranked.append(r)

    # 阈值过滤:先按分数砍掉近零分的,再取 top_k。
    # 但至少保留 rerank_min_keep 条 —— 全都不相关时也要给调用方
    # 一个"确实没有好结果"的信号,而不是空列表(空列表会被误读成检索故障)。
    kept = [
        r for r in reranked
        if (r.rerank_score or 0.0) >= cfg.rerank_min_score
    ]
    dropped = len(reranked) - len(kept)
    if len(kept) < cfg.rerank_min_keep:
        kept = reranked[: cfg.rerank_min_keep]

    if debug is not None:
        debug.reranked_hits = len(reranked)
        debug.dropped_by_threshold = dropped

    return kept[:top_k]


class HybridRetriever:
    """混合检索主入口。"""

    def __init__(
        self,
        store: QdrantStore | None = None,
        cfg: RetrievalConfig | None = None,
    ) -> None:
        self.cfg = cfg or get_settings().retrieval
        self.store = store or get_store()
        self._embedder = None
        self._reranker = None

    # 模型都是重对象,延迟到真正要用时再加载
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(self.cfg)
        return self._reranker

    # ----------------------------------------------------------------- #
    # 主入口
    # ----------------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        query_filter: qm.Filter | None = None,
        use_rerank: bool | None = None,
        debug: RetrievalDebug | None = None,
        include_superseded: bool | None = None,
        explain: bool = False,
    ) -> list[RetrievedChunk]:
        """检索。返回按相关性降序的结果。

        `include_superseded=None`(默认)用检索配置里的口径;显式传 True/False
        则**只影响这一次调用**。见下面「版本过滤」那段。

        `explain=True` 会把每条结果**是被哪一路召回的、那一路里排第几**
        填进 `dense_rank`/`sparse_rank`。代价是额外两次 Qdrant 查询(见
        下面那段),所以默认不做。**和 `debug` 不是一回事**:`debug` 是
        「把这次检索的中间量收集到一个对象里给我看」,`explain` 是
        「把解释写回结果本身,好让它一路序列化到前端」。两个都传就都做。
        """
        query = (query or "").strip()
        if not query:
            return []

        cfg = self.cfg
        top_k = top_k or cfg.rerank_top_n
        if use_rerank is None:
            use_rerank = cfg.rerank_enabled

        # --- 0. 版本过滤 ---
        # 在这里合一次,而不是在三个 `query_filter=` 调用点各写一遍:
        # 下面还有 `query_paths` 那条**只为可观测性**的支路(debug 用),
        # 漏掉它的后果是"调试视图里能看见已被标记失效的块,正式结果里没有"
        # —— 排查版本相关的问题时,这个假象比没有调试信息更坏。
        #
        # 口径:配置里的 `include_superseded` 是**默认值**,调用方可以显式
        # 覆盖单次调用。「查废止条款原来怎么写的」是一次明确的、用户自己
        # 按下去的动作,不给这个开关的话,前端就只能让用户改环境变量重启 ——
        # 那个做不到的开关等于没有这个功能。
        #
        # 之所以默认写在配置里而不是让每次调用自己决定:不显式指定的场合
        # (CLI、评测、智能体的工具调用)必须**行为一致**。同一批题跑评测时
        # 有的召回废止条款、有的不召回,那批数字就没法比了。
        include_hist = (
            cfg.include_superseded if include_superseded is None else include_superseded
        )
        if not include_hist:
            query_filter = visible_filter(query_filter)

        # --- 1. 一次前向出双向量 ---
        q = self.embedder.encode(query)
        q_dense, q_sparse = q.dense[0], q.sparse[0]

        # --- 2. 双路召回 + 服务端 RRF ---
        fused = self.store.query_hybrid(
            q_dense,
            q_sparse,
            limit=cfg.fusion_top_k,
            query_filter=query_filter,
        )
        if not fused:
            return []

        results = [RetrievedChunk.from_point(p) for p in fused]

        if debug is not None:
            debug.fused_hits = len(fused)
            debug.fused_order = [r.doc_id for r in results]

        if debug is not None or explain:
            # 单独再问一次两路的原始排名 —— 融合后名次里看不出「是谁召回了它」。
            #
            # 为什么默认不做:`query_hybrid` 是一次带两个 prefetch 的查询,
            # 稠密和稀疏各已经搜过一遍了,但**融合后的结果不带每路的原始名次**,
            # 想要名次就没有别的办法,只能把两路再各查一遍。这等于把 ANN 的
            # 搜索量翻倍,而换来的只是可解释性 —— 所以它必须由调用方明确要
            # (`explain=True`),或者是在开调试视图(`debug`)。默认路径一次
            # 都不多查。
            try:
                dh, sh = self.store.query_paths(q_dense, q_sparse, query_filter=query_filter)
                dense_order = [(p.payload or {}).get("doc_id", "") for p in dh]
                sparse_order = [(p.payload or {}).get("doc_id", "") for p in sh]
                if debug is not None:
                    debug.dense_hits = len(dh)
                    debug.sparse_hits = len(sh)
                    debug.dense_order = dense_order
                    debug.sparse_order = sparse_order
                # 名次要绑到**块**上,不是绑到文档上:一篇规程有六七个块,
                # 按 doc_id 建表会被同一篇文档后面的块反复覆盖,于是这篇文档
                # 的每个块都拿到同一个名次 —— 而「这块是稠密路第 2 还是第 40」
                # 正是要看的东西。point id 是 (doc_id, chunk_index) 的确定性
                # 函数(store.point_id),所以不必往 payload 里加字段就能对上。
                d_rank = {p.id: i for i, p in enumerate(dh)}
                s_rank = {p.id: i for i, p in enumerate(sh)}
                for r in results:
                    pid = point_id(r.doc_id, r.chunk_index)
                    r.dense_rank = d_rank.get(pid)
                    r.sparse_rank = s_rank.get(pid)
            except Exception as exc:  # noqa: BLE001 - 调试信息不该影响主流程
                logger.debug("抓取原始排名失败: %s", exc)

        # --- 3. 重排 ---
        if use_rerank:
            results = self._apply_rerank(query, results, top_k, debug)
        else:
            results = results[:top_k]

        return results

    def _apply_rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int,
        debug: RetrievalDebug | None,
    ) -> list[RetrievedChunk]:
        return apply_rerank(
            query, results, top_k, cfg=self.cfg, reranker=self.reranker, debug=debug
        )


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def format_context(chunks: Sequence[RetrievedChunk], max_chars: int = 6000) -> str:
    """把检索结果拼成给 LLM 的上下文块。

    每条都带来源标签 —— 没有标签的上下文等于没法引用,
    而没法引用的 RAG 在生产里基本等于不可信。
    这里也做了总长截断:上下文塞太满会挤掉对话历史,反而降低回答质量。
    """
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks, 1):
        block = f"[{i}] 来源:{c.citation}\n{c.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
