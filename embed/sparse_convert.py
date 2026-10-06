"""稀疏权重的格式转换。

bge-m3 的 lexical_weights 长这样:
    {"2134": 0.21, "98711": 0.08, ...}
键是 XLM-R 词表的 token id **字符串**,值是学到的权重。

Qdrant 要的是:
    SparseVector(indices=[2134, 98711, ...], values=[0.21, 0.08, ...])
索引是 **int**。

转换本身只有几行,但有个坑值得写下来:
Qdrant 的稀疏向量打分是**点积**,不带 IDF。bge-m3 学到的权重已经是
SPLADE 那一类「已含重要度」的表征,所以不要再加 Modifier.IDF ——
那等于把 IDF 乘两遍。只有当你存的是原始词频(自建 BM25 那一路)时,
才需要显式把 IDF 放进查询向量里。
"""

from __future__ import annotations

from typing import Dict, Mapping

from qdrant_client.models import SparseVector


def to_sparse_vector(weights: Mapping[str, float] | Dict[str, float]) -> SparseVector:
    """{"token_id_str": weight} -> SparseVector(indices, values)。"""
    if not weights:
        return SparseVector(indices=[], values=[])

    indices: list[int] = []
    values: list[float] = []
    for token, weight in weights.items():
        try:
            idx = int(token)
        except (TypeError, ValueError):
            # 不是数字键就跳过,而不是整个崩掉 —— 单条脏数据不该毁掉一次导入
            continue
        w = float(weight)
        if w <= 0.0:
            continue
        indices.append(idx)
        values.append(w)

    return SparseVector(indices=indices, values=values)
