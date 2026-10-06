"""知识库后端接口。两条路(混合向量 / GraphRAG)都实现这一套。

为什么要这层抽象
---------------
用户要的是「两种后端可以切换」。如果让调用方自己判断
「现在是 hybrid 还是 graphrag,该调哪个方法」,那个判断会渗透到每一处
调用点;以后加第三种后端就要改所有地方。

这里定死的契约只有四个动作:
    ingest_path / ingest_text  —— 写
    retrieve                   —— 读
    health / stats             —— 运维

**统一返回 RetrievedChunk**。图检索的结果比向量检索多带了实体和路径,
那些塞进 meta,不另开类型 —— 上层做引用、做重排、做喂给 LLM 的上下文,
不需要知道后端是哪种。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingest.pipeline import IngestStats
from retrieve.hybrid import RetrievedChunk


@dataclass
class BackendHealth:
    ok: bool
    backend: str
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def describe(self) -> str:
        if self.ok:
            return f"[{self.backend}] ok {self.detail}"
        return f"[{self.backend}] 不可用: {self.error}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "backend": self.backend,
            "detail": self.detail,
            "error": self.error,
        }


class BaseBackend(ABC):
    """所有后端实现的接口。"""

    #: 配置里 KB_BACKEND 用的值
    name: str = "base"

    # ----------------------------------------------------------------- #
    # 写
    # ----------------------------------------------------------------- #

    @abstractmethod
    def ingest_path(
        self,
        root: str | Path,
        recursive: bool = True,
        force: bool = False,
    ) -> IngestStats:
        """导入文件或目录。"""

    @abstractmethod
    def ingest_text(
        self,
        text: str,
        source: str,
        title: str = "",
        force: bool = False,
        extra: dict | None = None,
    ) -> IngestStats:
        """导入一段裸文本(网页搜索结果、临时粘贴的内容)。"""

    # ----------------------------------------------------------------- #
    # 读
    # ----------------------------------------------------------------- #

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None, **kwargs: Any) -> list[RetrievedChunk]:
        """检索。返回按相关性降序的结果。"""

    # ----------------------------------------------------------------- #
    # 运维
    # ----------------------------------------------------------------- #

    @abstractmethod
    def health(self) -> BackendHealth:
        """连通性 + 依赖状态。给 /health 用。"""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """库内容概况:多少文档、多少块/实体。"""

    def close(self) -> None:
        """释放连接。默认什么都不做。"""
