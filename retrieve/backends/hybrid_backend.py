"""混合向量后端:bge-m3 双路召回 → Qdrant 服务端 RRF → 客户端重排。

这一层很薄 —— 它只是把已经写好的 IngestPipeline 和 HybridRetriever
拼成一个符合 BaseBackend 契约的对象。真正干活的代码在 ingest/ 和
retrieve/hybrid.py 里,这里不重复实现任何检索逻辑。

薄是好事:后端抽象一旦开始「顺便做点自己的处理」,两条路的行为就会
悄悄发散,切换后端不再是等价替换。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import get_settings
from ingest.pipeline import IngestPipeline, IngestStats
from retrieve.backends.base import BackendHealth, BaseBackend
from retrieve.hybrid import HybridRetriever, RetrievalDebug, RetrievedChunk
from store.qdrant_store import QdrantStore, get_store

logger = logging.getLogger(__name__)


class HybridBackend(BaseBackend):
    name = "hybrid"

    def __init__(
        self,
        store: QdrantStore | None = None,
        pipeline: IngestPipeline | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        s = get_settings()
        self.settings = s
        self.store = store or get_store()
        self.pipeline = pipeline or IngestPipeline(store=self.store)
        self.retriever = retriever or HybridRetriever(store=self.store)

    # ----------------------------------------------------------------- #
    # 写
    # ----------------------------------------------------------------- #

    def ingest_path(
        self,
        root: str | Path,
        recursive: bool = True,
        force: bool = False,
    ) -> IngestStats:
        # 确保 collection 存在 —— 首次导入不该因为「忘了建库」而失败
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
        # 库还不存在时必须在这里返回空,不能让异常冒到 Qdrant。
        # 不设这道闸的后果:新用户装完直接 `kb.py search "…"`(还没 ingest),
        # Qdrant 抛 404 "Collection doesn't exist",CLI 打一整屏 traceback ——
        # 而 cmd_search 里明明备好了「库里可能还没有文档」那句友好提示,
        # 只是永远走不到。检索是只读操作,"空库"是正常状态不是错误。
        if not self.store.exists():
            return []
        return self.retriever.retrieve(query, top_k=top_k, debug=debug, **kwargs)

    # ----------------------------------------------------------------- #
    # 运维
    # ----------------------------------------------------------------- #

    def health(self) -> BackendHealth:
        h = self.store.health()
        detail = {
            "qdrant": h.get("collections"),
            "collection": self.settings.qdrant.collection,
            "points": self.store.count() if h["ok"] else 0,
            "llm_configured": self.settings.llm.configured,
            "contextual": (
                self.settings.ingest.contextual_enabled
                and self.settings.llm.configured
            ),
            "rerank": self.settings.retrieval.rerank_enabled,
        }
        return BackendHealth(ok=bool(h["ok"]), backend=self.name,
                             detail=detail, error=h.get("error", ""))

    def stats(self) -> dict[str, Any]:
        if not self.store.exists():
            # 空库这条路也要带 backend —— 少一个键,cmd_stats 就会打出「后端 ?」。
            # 而「还没导入任何东西」恰恰是最常见的一次调用。
            return {
                "backend": self.name,
                "collection": self.settings.qdrant.collection,
                "docs": 0,
                "chunks": 0,
            }
        docs = self.store.list_docs()
        return {
            "backend": self.name,
            "collection": self.settings.qdrant.collection,
            "docs": len(docs),
            "chunks": self.store.count(),
            "documents": docs,
        }
