"""智能体层:ReAct 循环 + 工具集(知识库检索 / 联网搜索 / 文档清单)。"""

from agent.react import (
    AgentCall,
    AgentResult,
    AgentStep,
    ReActAgent,
    build_agent,
    parse_action,
    parse_final_answer,
    parse_thought,
    truncate_observation,
)
from agent.tools import (
    DOCS_TOOL,
    KB_TOOL,
    WEB_TOOL,
    Tool,
    ToolRegistry,
    build_default_tools,
    build_registry,
    format_hits,
)
from agent.websearch import WebResult, WebSearchError, WebSearcher, get_searcher

__all__ = [
    "AgentCall",
    "AgentResult",
    "AgentStep",
    "ReActAgent",
    "build_agent",
    "parse_action",
    "parse_final_answer",
    "parse_thought",
    "truncate_observation",
    "Tool",
    "ToolRegistry",
    "build_default_tools",
    "build_registry",
    "format_hits",
    "KB_TOOL",
    "WEB_TOOL",
    "DOCS_TOOL",
    "WebResult",
    "WebSearchError",
    "WebSearcher",
    "get_searcher",
]
