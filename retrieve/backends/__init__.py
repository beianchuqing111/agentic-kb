"""后端工厂。`KB_BACKEND=hybrid|graphrag` 一个配置切换。

为什么要有这层工厂
-----------------
上层(agent/、CLI、web)只拿 `get_backend()`,不去管现在是哪条路。
切换后端是**改配置**,不是改代码 —— 这是用户明确要的能力。

一个容易搞错的点:两个后端实现的是同一套接口,但**不是同一种东西**
------------------------------------------------------------
  hybrid   : 块级检索。稠密+稀疏 → RRF → 重排。快,覆盖字面与语义。
  graphrag : 块级检索 + 图。多一层「实体 → 多跳 → 反查块」,
             能把正文里没有查询词的块捞回来,还能给出关系型的事实。

graphrag 不是 hybrid 的替代品,而是它的**超集**(见 graphrag_backend 的
模块说明)。所以「不知该选哪个」时的默认值是 hybrid:它快、依赖少
(不需要 Neo4j 和 LLM),而 graphrag 要额外承担建图的 LLM 成本。
"""

from __future__ import annotations

import logging

from config import BackendType, get_settings
from retrieve.backends.base import BackendHealth, BaseBackend
from retrieve.backends.graphrag_backend import GraphRAGBackend
from retrieve.backends.hybrid_backend import HybridBackend

logger = logging.getLogger(__name__)

_backend: BaseBackend | None = None


def build_backend(name: BackendType | str | None = None) -> BaseBackend:
    """按名字新建一个后端(不缓存)。测试和「临时切一下看看」用这个。"""
    kind = BackendType(name or get_settings().backend)

    if kind == BackendType.GRAPHRAG:
        return GraphRAGBackend()
    return HybridBackend()


def get_backend() -> BaseBackend:
    """取全局后端单例。

    缓存的是**实例**而不是「按名字查表」,因为模型(bge-m3、重排器)
    都挂在实例上,重建一次就是重新加载几个 G 的权重。
    所以运行中改 KB_BACKEND 不会热切换 —— 要么重启,要么显式
    `set_backend()`,不做隐式魔法。
    """
    global _backend
    if _backend is None:
        _backend = build_backend()
        logger.info("知识库后端: %s", _backend.name)
    return _backend


def set_backend(backend: BaseBackend | None) -> None:
    """显式换掉全局后端(主要给测试和脚本用)。"""
    global _backend
    _backend = backend


__all__ = [
    "BaseBackend",
    "BackendHealth",
    "GraphRAGBackend",
    "HybridBackend",
    "build_backend",
    "get_backend",
    "set_backend",
]
