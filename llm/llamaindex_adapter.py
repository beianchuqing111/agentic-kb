"""把项目的 LLMClient 包成 LlamaIndex 的 CustomLLM。

为什么需要它
-----------
GraphRAG 那条路里,实体抽取用的 `SimpleLLMPathExtractor` 要求传一个
LlamaIndex 的 LLM 对象(注意不是 `PropertyGraphIndex` —— 那个类全项目
没构造过,见 `ingest/graph_pipeline.py` 开头)。
而项目里所有 LLM 调用都要走 `LLMClient` —— 那里统一做了重试、
退避、用量统计。如果这里改用 `OpenAILike`,用量统计就漏了一半,
账单上的数字会对不上。

关于 CustomLLM 是 pydantic 模型这件事
------------------------------------
不能在类里声明 `client: LLMClient` 这样的字段 —— 它既不是 pydantic
模型也没有任意类型配置,会被拒绝。用模块级单例(`get_llm()`)最省事,
反正本来就该全局共享一个客户端。

只实现三个抽象方法(metadata / complete / stream_complete)。
`chat` / `achat` 在基类里有默认实现,会转发到 complete,
所以不覆写也不会漏功能。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.custom import CustomLLM

from config import get_settings
from llm.client import get_llm


def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    """LlamaIndex 的 ChatMessage → OpenAI 的 messages。

    注意 role:LlamaIndex 有自己的 MessageRole 枚举,直接 str() 出来是
    "MessageRole.USER" 这种,不是 "user"。必须走 .value。
    """
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        # LlamaIndex 用 TOOL / FUNCTION 表态,OpenAI 兼容端点认 "tool"
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        out.append({"role": role, "content": m.content or ""})
    return out


class ProjectLLM(CustomLLM):
    """复用项目 LLMClient 的 LlamaIndex LLM 适配器。"""

    #: 覆盖上下文窗口 —— 基类默认 3900 太小,会让 LlamaIndex 内部
    #: 的提示词组装过早截断长 chunk(比如一个 4000 字的表格块)。
    context_window: int = 32768
    num_output: int = 2048

    #: 这个适配器自己的 completion 预算,`None` = 用 LLMClient 的全局值。
    #: 存在的理由:同一个模型在不同任务上需要的预算差很多。实体抽取要的
    #: 是一小段 JSON,可推理模型会先在思维链上花掉几千 token —— 全局 8192
    #: 在长 chunk 上会被推理吃干净,正文一个字不剩。给抽取单独放宽,
    #: 交互式问答那边不受影响(那边走 agent/react.py,压根不经过这里)。
    #: 字段名**故意**不叫 max_tokens:别覆盖 pydantic 基类可能的同名声明。
    default_max_tokens: Optional[int] = None

    @property
    def metadata(self) -> LLMMetadata:
        cfg = get_settings().llm
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            # 是对话模型 —— 设 False 的话 LlamaIndex 会拿纯文本补全的方式
            # 拼提示词,对 DeepSeek/Qwen 这类只会对话的端点直接报错
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=cfg.model or "unknown",
            system_role=MessageRole.SYSTEM,
        )

    # ----------------------------------------------------------------- #

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        r = get_llm().chat(
            [{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens") or self.default_max_tokens,
        )
        return CompletionResponse(
            text=r.text,
            raw={"usage": r.total_tokens},
        )

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """流式。底层没接流式接口,这里退化成「一次性产出」。

        LlamaIndex 抽实体走的是 complete,不走这条;但抽象方法必须实现。
        真做流式要改 LLMClient 的调用方式,收益只在交互式问答,
        而交互式那条走的是 agent/react.py,不经过 LlamaIndex。
        """
        resp = self.complete(prompt, formatted=formatted, **kwargs)

        def gen() -> CompletionResponseGen:
            yield resp

        return gen()

    # ----------------------------------------------------------------- #
    # 异步:基类默认用 asyncio.to_thread 包同步版,这里显式写出来
    # 只是为了和同步版保持一样的行为,不引入额外的线程切换开销
    # ----------------------------------------------------------------- #

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        return await asyncio.to_thread(self.complete, prompt, formatted, **kwargs)

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return await asyncio.to_thread(lambda: self.chat(messages, **kwargs))

    # ----------------------------------------------------------------- #
    # 对话路径覆写:基类的 chat 会把 ChatMessage 拼成一个大字符串再走
    # complete,那样会丢掉 system / assistant 的角色区分,对抽取任务的
    # 指令遵循有明显影响。这里直接把角色映射传下去。
    # ----------------------------------------------------------------- #

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        r = get_llm().chat(
            _to_openai_messages(messages),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens") or self.default_max_tokens,
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=r.text),
            raw={"usage": r.total_tokens},
        )


_llm_adapter: Optional[ProjectLLM] = None


def get_llamaindex_llm(default_max_tokens: Optional[int] = None) -> ProjectLLM:
    """默认返回全局单例。

    传 `default_max_tokens` 时返回**一个新实例**(带自己的预算),不污染单例 ——
    抽取路径要放宽预算,问答路径不该跟着变。适配器本身很轻,底层
    `LLMClient` 仍然共享同一个,用量统计不会分叉。
    """
    global _llm_adapter
    if default_max_tokens is not None:
        return ProjectLLM(default_max_tokens=default_max_tokens)
    if _llm_adapter is None:
        _llm_adapter = ProjectLLM()
    return _llm_adapter


__all__ = ["ProjectLLM", "get_llamaindex_llm"]
