"""交叉编码器重排(bge-reranker-v2-m3)。

为什么必须客户端做
-----------------
Qdrant 服务端能跑 RRF 融合(FusionQuery),但跑不了交叉编码器:
它需要把 (query, passage) **成对**送进一个 Transformer 前向,
不是向量点积能表达的。所以重排只能在应用侧做 ——
这也是整条链里唯一一处"拿到候选后还要再算一遍"的地方,延迟主要在这。

两级检索的分工
-------------
  召回(双路 + RRF):要的是**快**和**全**,宁可多召回些不相关的
  重排(交叉编码器)  :要的是**准**,在几十条候选里挑出真正相关的几条

所以召回阶段配置的是 dense_top_k=50 / sparse_top_k=50,融合后 20 条进重排,
最后取 5 条。这个漏斗形状是刻意的:重排很贵,不能拿它去做召回。

关于分数
-------
normalize=True 会过一层 sigmoid,把 logits 压到 (0,1)。
好处是可以跨 query 比较、设阈值;坏处是**不同 query 之间仍然不可比**,
只能在同一 query 内排序。别拿它当作"相关性百分比"展示给用户。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from config import EmbedConfig, RetrievalConfig, assert_hf_offline_env, get_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankHit:
    """一条重排结果。index 指回原始候选列表的下标。"""

    index: int
    score: float
    text: str


class Reranker:
    """bge-reranker-v2-m3 的薄封装。模型约 2.3G,延迟加载 + 单例。"""

    def __init__(
        self,
        cfg: RetrievalConfig | None = None,
        embed_cfg: EmbedConfig | None = None,
    ) -> None:
        s = get_settings()
        self.cfg = cfg or s.retrieval
        self.embed_cfg = embed_cfg or s.embed
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> Any:
        assert_hf_offline_env()

        from FlagEmbedding import FlagReranker

        device = self.embed_cfg.device
        use_fp16 = self.embed_cfg.use_fp16 and device.startswith("cuda")
        logger.info("加载重排模型: %s device=%s fp16=%s",
                    self.cfg.rerank_model, device, use_fp16)

        try:
            model = FlagReranker(
                self.cfg.rerank_model, use_fp16=use_fp16, device=device
            )
        except TypeError:
            # 老版本没有 device 参数
            logger.warning("该版本 FlagReranker 不接受 device 参数,交给 torch 自动选择")
            model = FlagReranker(self.cfg.rerank_model, use_fp16=use_fp16)

        # 默认 512,常把长块截断。调大一点能让交叉编码器看到更多上下文,
        # 代价是显存和延迟。1024 是精度/成本的折中。
        try:
            model.model.max_seq_length = self.cfg.rerank_max_length
        except Exception:  # noqa: BLE001
            logger.debug("rerank_max_length 设置失败,沿用库默认值")

        logger.info("重排模型就绪")
        return model

    def rerank(
        self,
        query: str,
        texts: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankHit]:
        """按与 query 的相关性重排 texts,返回降序结果。

        top_n 为 None 时返回全部(调用方一般只要前几条)。
        """
        if not texts:
            return []
        if not query.strip():
            # 没有 query 就无从重排,原序返回 —— 别让模型白白跑一遍
            return [RerankHit(i, 0.0, t) for i, t in enumerate(texts)]

        pairs = [[query, t] for t in texts]
        scores = self.model.compute_score(
            pairs,
            normalize=True,
            max_length=self.cfg.rerank_max_length,
        )

        # 单条时 compute_score 返回标量而不是列表,统一成列表
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        if len(scores) != len(texts):
            raise RuntimeError(
                f"重排返回 {len(scores)} 个分数,但有 {len(texts)} 条文本 —— 对不上"
            )

        hits = [RerankHit(i, scores[i], texts[i]) for i in range(len(texts))]
        # 稳定排序:分数相同时保持召回阶段的相对顺序(那个顺序本身有信息量)
        hits.sort(key=lambda h: -h.score)

        if top_n is not None:
            hits = hits[:top_n]
        return hits


_reranker: Reranker | None = None
_reranker_lock = threading.Lock()


def get_reranker(cfg: RetrievalConfig | None = None) -> Reranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker(cfg)
    return _reranker
