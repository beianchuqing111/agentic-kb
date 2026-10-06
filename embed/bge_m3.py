"""bge-m3 嵌入器:一次前向,同时出稠密向量和稀疏权重。

为什么是 bge-m3
---------------
它一个模型顶三个:
  - dense  : 1024 维稠密向量,语义匹配
  - sparse : lexical_weights,{"token_id": 权重},SPLADE 那一类的**学习式**稀疏
  - colbert: 多向量细粒度匹配(本项目不开,存储要涨十倍以上)

关键点是这三者是**同一次前向**出来的,不是跑三遍模型。
稀疏那一路尤其容易误会 —— 它不是 BM25,是模型学出来的词权重,
所以不要把 BM25 的 IDF 再叠上去(见 sparse_convert.py 的说明)。

关于归一化
---------
Qdrant 用 COSINE 距离时内部会归一化,理论上可以省掉这一步。
但归一化之后「点积 == 余弦」,任何需要手算相似度的地方
(比如后面实体对齐拿稠密向量算相似度)都不用再惦记这茬。
代价是一次 O(n) 的逐行缩放,忽略不计。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from config import EmbedConfig, assert_hf_offline_env, get_settings

logger = logging.getLogger(__name__)

# bge-m3 的稠密维度是固定的。写成常量而不是从模型里读,
# 是为了在**加载模型之前**就能建 Qdrant collection。
DENSE_DIM = 1024


@dataclass
class EmbedResult:
    """一次编码的产出。稠密是矩阵,稀疏是等长的字典列表。"""

    dense: np.ndarray                  # (n, 1024) float32,已 L2 归一化
    sparse: list[dict[str, float]]     # n 个 {"token_id": weight}

    def __len__(self) -> int:
        return int(self.dense.shape[0])

    def __post_init__(self) -> None:
        if self.dense.shape[0] != len(self.sparse):
            raise ValueError(
                f"稠密 {self.dense.shape[0]} 条、稀疏 {len(self.sparse)} 条,对不上"
            )

    def items(self):
        """按条迭代 (dense_vec, sparse_weights),导入时用起来顺手。"""
        for i in range(len(self)):
            yield self.dense[i], self.sparse[i]


class BGEM3Embedder:
    """bge-m3 的薄封装。

    模型 1.2G,加载一次几秒到几十秒,所以延迟加载 + 单例。
    业务代码只应通过 get_embedder() 拿实例,不要自己 new。
    """

    def __init__(self, cfg: EmbedConfig | None = None) -> None:
        self.cfg = cfg or get_settings().embed
        self._model: Any = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- #
    # 模型加载
    # ----------------------------------------------------------------- #

    @property
    def model(self) -> Any:
        """延迟加载。双检锁 —— 多线程下别把 1.2G 的模型加载两遍。"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> Any:
        # 模型已缓存在本地时也必须完全离线加载 —— 否则一次网络抖动就会
        # 让整批导入挂掉,而故障现象是个「和模型无关」的 HTTP 错误。
        assert_hf_offline_env()

        from FlagEmbedding import BGEM3FlagModel

        # CPU 上没有 fp16 这条快路,硬开会直接报错或者悄悄变慢
        use_fp16 = self.cfg.use_fp16 and self.cfg.device.startswith("cuda")
        logger.info(
            "加载 bge-m3: model=%s device=%s fp16=%s",
            self.cfg.model_name, self.cfg.device, use_fp16,
        )

        try:
            model = BGEM3FlagModel(
                self.cfg.model_name, use_fp16=use_fp16, device=self.cfg.device
            )
        except TypeError:
            # 老版本 FlagEmbedding 的构造签名里没有 device,只能靠 torch 自己选
            logger.warning("该版本 BGEM3FlagModel 不接受 device 参数,交给 torch 自动选择")
            model = BGEM3FlagModel(self.cfg.model_name, use_fp16=use_fp16)

        # 默认 8192,长文档能整段塞进去。导入时按块编码,通常远小于这个值
        try:
            model.max_seq_length = self.cfg.max_length
        except Exception:  # noqa: BLE001 - 属性名随版本变动,设不上就算了
            logger.debug("max_seq_length 设置失败,沿用库默认值")

        logger.info("bge-m3 就绪")
        return model

    # ----------------------------------------------------------------- #
    # 编码
    # ----------------------------------------------------------------- #

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> EmbedResult:
        """文本 → (稠密向量, 稀疏权重)。

        查询和文档走的是同一条路 —— bge-m3 不需要 bge-large 那种
        "为查询加指令前缀" 的写法,这也是它省事的地方之一。
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        if not items:
            return EmbedResult(
                dense=np.zeros((0, DENSE_DIM), dtype=np.float32), sparse=[]
            )

        out = self.model.encode(
            items,
            batch_size=batch_size or self.cfg.batch_size,
            max_length=max_length or self.cfg.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # 多向量不开:存储代价十倍以上
        )

        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        if dense.ndim == 1:  # 单条时某些版本会降成一维
            dense = dense.reshape(1, -1)

        self._l2_normalize_inplace(dense)

        sparse: list[dict[str, float]] = []
        for weights in out["lexical_weights"]:
            # 键统一成 str:FlagEmbedding 不同版本给过 int 和 str 两种,
            # 而 to_sparse_vector 是按 str 键解析的
            sparse.append({str(k): float(v) for k, v in weights.items()})

        return EmbedResult(dense=dense, sparse=sparse)

    def encode_one(self, text: str) -> tuple[np.ndarray, dict[str, float]]:
        """单条便捷入口,返回 (向量, 权重)。查询时用。"""
        res = self.encode(text)
        return res.dense[0], res.sparse[0]

    @staticmethod
    def _l2_normalize_inplace(mat: np.ndarray) -> None:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        # 全零行(空串编码)要防一下,否则除法出 nan,存进 Qdrant 会很难查
        np.divide(mat, np.maximum(norms, 1e-12), out=mat)

    @property
    def dense_dim(self) -> int:
        return DENSE_DIM


# --------------------------------------------------------------------- #
# 单例
# --------------------------------------------------------------------- #

_embedder: BGEM3Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder(cfg: EmbedConfig | None = None) -> BGEM3Embedder:
    """进程内共享一个嵌入器。"""

    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = BGEM3Embedder(cfg)
    return _embedder
