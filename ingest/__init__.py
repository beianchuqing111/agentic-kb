"""导入层:加载 → 中文分块 → 上下文增强 → 嵌入 → 落库。"""

from ingest.chunker import Chunk as TextChunk
from ingest.chunker import chunk_text, split_sentences
from ingest.contextual import enrich_chunks
from ingest.graph_pipeline import GraphIngestPipeline, get_graph_pipeline
from ingest.loader import (
    SUPPORTED_EXTS,
    Document,
    discover,
    load_file,
    load_many,
    normalize_text,
    stable_doc_id,
)
from ingest.pipeline import IngestPipeline, IngestStats, chunk_hash

__all__ = [
    "TextChunk",
    "chunk_text",
    "split_sentences",
    "enrich_chunks",
    "Document",
    "discover",
    "load_file",
    "load_many",
    "normalize_text",
    "stable_doc_id",
    "SUPPORTED_EXTS",
    "IngestPipeline",
    "IngestStats",
    "chunk_hash",
    "GraphIngestPipeline",
    "get_graph_pipeline",
]
