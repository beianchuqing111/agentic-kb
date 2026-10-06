"""把项目的 bge-m3 包成 LlamaIndex 的 BaseEmbedding。

为什么必须用**同一个**嵌入器,而不能让 LlamaIndex 自己加载
------------------------------------------------------
两条后端要能切换,而 GraphRAG 这路除了往 Neo4j 的 `entity` 向量索引里
写实体向量,还要往 Qdrant 写节点向量 —— 用的是**同一批 bge-m3 向量**。

如果让 LlamaIndex 用 `HuggingFaceEmbedding("BAAI/bge-m3")` 自己加载一份:
  1. 显存/内存里会有**两份** bge-m3(1.2G × 2),RTX 5080 上白占
  2. 更麻烦的是两个实例的向量空间虽然在理论上相同,但只要任何一边的
     normalize / max_length / dtype 设置不一致,检索质量就会悄悄变差,
     而且完全不报错
  3. Neo4j 那个 `entity` 向量索引的维度是我们在 ensure_indexes 里
     写死 1024 的,喂进去一个不同来源的向量就撞维度错误

所以这里走适配器:所有嵌入都穿过同一个 `get_embedder()` 单例。

实现上有个细节:不把 embedder 存在 pydantic 字段里。
BaseEmbedding 是 pydantic 模型,塞一个非 pydantic 的普通对象进去要么
被拒绝、要么需要 PrivateAttr 加一堆配置;直接在方法里取单例最省事,
也天然保证「和检索用的是同一个实例」。

同理**不重新声明 embed_batch_size** —— BaseEmbedding 已经定义了这个字段
(带 gt=0 约束),覆盖一遍会把那条校验悄悄丢掉。
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding

from config import get_settings
from embed.bge_m3 import DENSE_DIM, get_embedder


class BGEM3Embedding(BaseEmbedding):
    """bge-m3 的 LlamaIndex 适配器。只出稠密向量 ——
    LlamaIndex 的图/向量存储只认单向量,稀疏那一路是 Qdrant 专用的。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", get_settings().embed.model_name)
        kwargs.setdefault("embed_batch_size", get_settings().embed.batch_size)
        super().__init__(**kwargs)

    # ----------------------------------------------------------------- #
    # 同步
    # ----------------------------------------------------------------- #

    def _get_query_embedding(self, query: str) -> List[float]:
        # bge-m3 查询和文档走**同一条编码路径**,不加任何指令前缀
        # (这点和 bge-large 那类不一样,加了前缀反而变差)
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 覆写批量方法:bge-m3 一次前向比逐条快得多
        if not texts:
            return []
        return self._embed(list(texts))

    # ----------------------------------------------------------------- #
    # 异步 —— 模型是同步的 GPU 推理,丢线程池里跑,别阻塞事件循环
    # ----------------------------------------------------------------- #

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await asyncio.to_thread(self._embed, [query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await asyncio.to_thread(self._embed, [text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed, list(texts))

    # ----------------------------------------------------------------- #

    @staticmethod
    def _embed(texts: List[str]) -> List[List[float]]:
        res = get_embedder().encode(texts)
        return [v.tolist() for v in res.dense]

    @property
    def dense_dim(self) -> int:
        return DENSE_DIM
