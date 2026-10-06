"""嵌入层:把文本变成可存进 Qdrant 的稠密向量和稀疏向量。"""

from embed.bge_m3 import BGEM3Embedder, get_embedder
from embed.sparse_convert import to_sparse_vector

__all__ = ["BGEM3Embedder", "get_embedder", "to_sparse_vector"]
