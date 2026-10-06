"""Tavily 网页搜索。薄封装:把 SDK 的异常归一成 `WebSearchError`。

几个刻意的取舍
-------------
**不开 include_answer**。Tavily 能直接返回一段合成好的答案,但那是一段
无法核验的文本:模型把它抄进最终回答之后,引用标签指向的文章里可能
根本没有这句话。ReAct 的价值就在于「先取证据再下结论」,这里必须拿到
是**检索结果**,合成交给模型自己做。

**不开 include_raw_content**。原始正文动辄几万字,一条就能把 Observation
预算吃光(见 config.AgentConfig.tool_result_max_chars)。Tavily 的
`content` 摘要字段粒度刚好。

**失败不抛给上层**。搜索失败是很平常的事(限流、断网、key 过期),此时
模型完全可以改用知识库回答。所以这里抛 `WebSearchError`,由工具层接住
转成 Observation —— 对话不该因为一次网络抖动就整个崩掉。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config import WebSearchConfig, get_settings

logger = logging.getLogger(__name__)


class WebSearchError(RuntimeError):
    """网页搜索不可用。工具层应把它转成 Observation。"""


@dataclass
class WebResult:
    """一条网页结果。字段是 Tavily 返回里我们真正要用的那几个。"""

    title: str
    url: str
    content: str = ""
    score: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WebResult | None":
        """从 Tavily 的一条结果建对象。没有 url 的直接丢弃 —— 引用不了。"""
        url = (d.get("url") or "").strip()
        if not url:
            return None
        return cls(
            title=(d.get("title") or url).strip(),
            url=url,
            content=(d.get("content") or "").strip(),
            score=float(d.get("score") or 0.0),
        )


class WebSearcher:
    """Tavily 客户端。每次查询现成一个 SDK 客户端成本很低,但仍缓存复用。"""

    def __init__(self, cfg: WebSearchConfig | None = None) -> None:
        self.cfg = cfg or get_settings().web
        self._client: Any = None

    @property
    def configured(self) -> bool:
        return self.cfg.configured

    def _get_client(self):
        if not self.configured:
            raise WebSearchError(
                "TAVILY_API_KEY 没配,联网搜索不可用。"
                "请改用 search_knowledge_base 回答,或告知用户无法联网。"
            )
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.cfg.api_key)
        return self._client

    def search(self, query: str, max_results: int | None = None) -> list[WebResult]:
        """搜一次。返回空列表 = 没搜到(不是错);失败才抛 WebSearchError。"""
        query = (query or "").strip()
        if not query:
            return []

        n = max_results or self.cfg.max_results
        try:
            resp = self._get_client().search(
                query,
                max_results=n,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
                timeout=self.cfg.timeout,
            )
        except WebSearchError:
            raise
        except Exception as exc:  # noqa: BLE001
            # 原始异常里可能带 URL 和 request id,对排查有用,原样带上
            raise WebSearchError(f"Tavily 请求失败: {exc}") from exc

        raw = (resp or {}).get("results") or []
        out = [r for r in (WebResult.from_dict(d) for d in raw) if r is not None]
        if not out and raw:
            logger.warning("Tavily 返回 %d 条但都没有 url,已全部丢弃", len(raw))
        logger.debug("Tavily %r -> %d 条", query, len(out))
        return out


_searcher: WebSearcher | None = None


def get_searcher(cfg: WebSearchConfig | None = None) -> WebSearcher:
    global _searcher
    if _searcher is None or cfg is not None:
        _searcher = WebSearcher(cfg)
    return _searcher


__all__ = ["WebResult", "WebSearchError", "WebSearcher", "get_searcher"]
