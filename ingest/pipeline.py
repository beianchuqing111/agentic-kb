"""导入流水线:文件 → 分块 → 上下文增强 → 嵌入 → 落 Qdrant。

    load_file ──→ chunk_text ──→ enrich_chunks ──→ bge-m3 ──→ upsert
                 (中文分句)      (Anthropic 定位语)   (一次前向出双向量)

增量导入的规则(这里是整个流水线最需要想清楚的地方)
------------------------------------------------
每块算一个 **原文** 的 sha1 当 content_hash,然后:

  所有块的 hash 都在库里  → 整篇跳过,连 LLM 和嵌入都不跑
  有一部分 hash 不在库里  → 这篇文档变了 → **先 delete_by_doc 再整体重写**

为什么是「先删再写」而不是「按 hash 逐块补」:
  分块是位置敏感的。在文档开头插一段,后面所有块的 index 全变。
  逐块补的话,新块写进去、旧块(index 变了但 hash 还在)留在库里 ——
  同一篇文档会有两套并存的块,检索命中哪一套看运气。
  整体重写虽然多花点算力,但结果是确定的。

为什么 content_hash 只算**原文**、不算拼了定位语的 embed_text:
  定位语是 LLM 生成的,每次重跑措辞都会有点不同。算进去的话,
  每次导入都会「发现文档变了」,然后全量重嵌 —— 又贵又不可重放。
  只算原文让导入真正幂等。代价是想换一批更好的定位语必须显式 force=True,
  这个代价是值得的,也是可预期的。
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from config import IngestConfig, get_settings
from embed.bge_m3 import get_embedder
from ingest.chunker import chunk_text
from ingest.contextual import enrich_chunks
from ingest.loader import Document, discover, load_file, load_many, stable_doc_id
from llm.client import LLMClient, get_llm
from store.qdrant_store import Chunk as StoredChunk
from store.qdrant_store import QdrantStore, get_store

logger = logging.getLogger(__name__)

# 一次处理多少块。太大单批内存和显存都吃紧,太小来回开销高。
EMBED_GROUP = 128


def chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class IngestStats:
    """导入结果。生产环境必须看得见这些数字 ——
    "导完了"和"导进去了 60%"是完全不同的两件事。"""

    files_seen: int = 0
    files_loaded: int = 0
    docs_indexed: int = 0
    docs_unchanged: int = 0
    chunks_written: int = 0
    chunks_failed: int = 0
    contextual_missing: int = 0
    elapsed: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    # 只有图导入会填这两个。放在共用结构里而不是另开一个返回值类型,
    # 是为了让后端接口只认一种导入结果 —— hybrid 那里恒为 0。
    entities_written: int = 0
    relations_written: int = 0

    def summary(self) -> str:
        parts = [
            f"扫描 {self.files_seen} 个文件",
            f"读入 {self.files_loaded}",
            f"新建/更新 {self.docs_indexed} 篇",
            f"未变化跳过 {self.docs_unchanged} 篇",
            f"写入 {self.chunks_written} 块",
        ]
        if self.entities_written or self.relations_written:
            parts.append(
                f"图谱 {self.entities_written} 实体 / {self.relations_written} 关系"
            )
        if self.chunks_failed:
            parts.append(f"失败 {self.chunks_failed}")
        if self.contextual_missing:
            parts.append(f"{self.contextual_missing} 块无定位语")
        if self.errors:
            parts.append(f"错误 {len(self.errors)} 条")
        parts.append(f"耗时 {self.elapsed:.1f}s")
        return ",".join(parts)


@dataclass
class PreparedDoc:
    """一篇文档的「前处理」结果,两条导入路径共用。

    分成两半是因为 hybrid 和 graphrag 的差异**只在后半段**:
    前处理(分块 → 增量判断 → 上下文增强)完全一样,落点不一样
    (一个只写 Qdrant,一个还要往 Neo4j 建图)。

    把这半段抽出来是为了让两条路**不可能**在分块/定位语上发散 ——
    否则同一个问题经两条后端检索,拿到的块边界和文本都不一样,
    「切换后端」就不再是等价替换了。
    """

    doc: Document
    chunks: list
    contexts: list[str]
    hashes: list[str]
    unchanged: bool = False


def prepare_document(
    doc: Document,
    cfg: IngestConfig,
    llm: LLMClient,
    store: QdrantStore,
) -> PreparedDoc | None:
    """分块 + 增量判断 + 上下文增强。分块后为空时返回 None。

    增量判断只看 Qdrant —— 图那边是否完整由调用方自己再确认
    (图导入会额外查一次「这篇在图上有几块」,见 GraphIngestPipeline)。
    """
    chunks = chunk_text(doc.text, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        logger.warning("分块后为空,跳过:%s", doc.source)
        return None

    hashes = [chunk_hash(c.text) for c in chunks]
    unchanged = len(store.existing_hashes(hashes)) == len(hashes)

    contexts: list[str] = []
    if not unchanged:
        contexts = enrich_chunks(doc.text, chunks, cfg, llm)

    return PreparedDoc(
        doc=doc, chunks=chunks, contexts=list(contexts),
        hashes=hashes, unchanged=unchanged,
    )


class IngestPipeline:
    """导入编排。"""

    def __init__(
        self,
        store: QdrantStore | None = None,
        cfg: IngestConfig | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.cfg = cfg or get_settings().ingest
        self.store = store or get_store()
        self.llm = llm or get_llm()
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # ----------------------------------------------------------------- #
    # 对外入口
    # ----------------------------------------------------------------- #

    def ingest_path(
        self,
        root: str | Path,
        recursive: bool = True,
        force: bool = False,
        skip_errors: bool = True,
    ) -> IngestStats:
        """导入一个文件或整个目录。"""
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
        logger.info("导入完成:%s", stats.summary())
        return stats

    def ingest_text(
        self,
        text: str,
        source: str,
        title: str = "",
        force: bool = False,
        extra: dict | None = None,
    ) -> IngestStats:
        """导入一段裸文本(比如网页搜索的结果)。"""
        t0 = time.time()
        stats = IngestStats(files_seen=1, files_loaded=1)

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
                # 单篇失败不该中断整批 —— 导 500 篇挂在第 200 篇上最气人
                logger.exception("导入失败: %s", doc.source)
                stats.errors.append((doc.source, str(exc)))
                stats.chunks_failed += 1

    def _ingest_one(self, doc: Document, stats: IngestStats, force: bool = False) -> None:
        prep = prepare_document(doc, self.cfg, self.llm, self.store)
        if prep is None:
            return
        chunks, hashes, contexts = prep.chunks, prep.hashes, prep.contexts

        # --- 增量判断 ---
        if prep.unchanged and not force:
            stats.docs_unchanged += 1
            logger.debug("未变化,跳过:%s (%d 块)", doc.source, len(chunks))
            return

        # 文档变了 → 整体重写。见模块开头关于「先删再写」的说明。
        if not force:
            self.store.delete_by_doc(doc.doc_id)

        logger.info("导入 %s:%d 块%s", doc.source, len(chunks),
                    "(force 全量重写)" if force else "")

        stats.contextual_missing += sum(1 for c in contexts if not c)

        # --- 嵌入 + 写入 ---
        written = 0
        for g0 in range(0, len(chunks), EMBED_GROUP):
            g1 = min(g0 + EMBED_GROUP, len(chunks))
            group = chunks[g0:g1]
            group_ctx = contexts[g0:g1]

            # 编码的是 embed_text(定位语 + 原文),存的 text 是原文
            embed_texts = [
                f"{ctx}\n{ch.text}" if ctx else ch.text
                for ch, ctx in zip(group, group_ctx)
            ]
            res = self.embedder.encode(embed_texts, batch_size=self._batch_size(len(group)))

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
                    content_hash=hashes[g0 + i],
                    extra={
                        "section": ch.section,
                        "doc_hash": doc.content_hash,
                        **(doc.metadata or {}),
                    },
                )
                for i, ch in enumerate(group)
            ]
            written += self.store.upsert(stored)

            if g1 < len(chunks):
                logger.debug("  %s: %d/%d 块", doc.source, g1, len(chunks))

        stats.docs_indexed += 1
        stats.chunks_written += written

    def _batch_size(self, n: int) -> int:
        """编码批大小。长文本要调小,不然一次前向就 OOM。"""
        base = get_settings().embed.batch_size
        return max(1, min(base, n))
