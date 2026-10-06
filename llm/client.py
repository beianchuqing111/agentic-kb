"""LLM 客户端:OpenAI 兼容接口的一层薄封装。

为什么自己封而不用 LlamaIndex 的 LLM
----------------------------------
因为有三处**不是标准 RAG 流程**的地方要用它,而且是并发的:
  - 上下文感知检索:每块写一句定位语(几百次并发调用)
  - GraphRAG 实体抽取:每个 chunk 抽一次(几百次并发调用)
  - ReAct:对话式单次调用

LlamaIndex 那套是给 pipeline 内部用的,拿它做批量并发不顺手。
底层都用同一个 openai SDK,只是一层更贴合本项目的壳。

三家的兼容性
-----------
DeepSeek / Qwen(DashScope 兼容模式)/ Kimi 都提供 OpenAI 兼容端点,
base_url 一换即可,代码不用动。注意 Qwen 的兼容模式 base_url 是
`https://dashscope.aliyuncs.com/compatible-mode/v1`,不是官网那个。

关于重试
-------
429 是这里最常见的错误,而且**并发调高只会让它更糟** ——
服务端按账号限速,压并发只是把排队从服务端挪到客户端。
所以这里做的是「指数退避 + 尊重 Retry-After」,不是「多开几个线程」。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from config import LLMConfig, get_settings

logger = logging.getLogger(__name__)


class LLMNotConfigured(RuntimeError):
    """没配 key 就调用。宁可显式报错,也不要静默返回空字符串 ——
    那会让上层把「模型没跑」误当成「模型认为没有实体」。"""


class LLMError(RuntimeError):
    """重试耗尽后的失败。"""


# 模型爱把 JSON 包在 ```json 里,或者前面加一句「好的,以下是结果:」
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_JSON_START = re.compile(r"[\[{]")


def _find_json_span(s: str, start: int) -> int | None:
    """从 start 处的 [ 或 { 开始,找到**配对**的收尾符下标。

    不能用 rfind:模型爱在 JSON 后面补一句「以上」,
    而中文正文里出现 } 或 ] 是很正常的(比如「参见 {附录A}」),
    rfind 会一把抓到最后那个,拼出一段语法上不成立的字符串。

    这里做带深度计数的扫描,并且**跳过字符串字面量内部**的括号 ——
    否则 `{"text": "}"}` 这种会在引号里那个 } 上提前收尾。
    """
    if start >= len(s) or s[start] not in "[{":
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return i
        # 中途深度变负说明括号本来就配不平,直接放弃
        elif depth < 0:
            return None

    return None


def extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON。三级退让,能救回大部分脏输出。

    改结构化的输出是模型的通病,直接 json.loads 失败率不低。
    但**不要退让到底**:三级都失败就抛,让调用方知道这次真的没解析出来,
    而不是返回个空 dict 让上层以为「模型说没有」。
    """
    if not text:
        raise ValueError("模型输出为空")

    s = text.strip()

    # 1. 直接就是 JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2. 被 ``` 包起来了
    m = _FENCE_RE.search(s)
    if m:
        inner = m.group(1).strip()
        try:
            # 围栏内容本身就可能是「JSON + 一句废话」,所以还要再过一遍第 3 步
            return json.loads(inner)
        except json.JSONDecodeError:
            s = inner

    # 3. 前后有废话 —— 定位第一个 [ 或 {,再按括号配对找它的收尾
    m = _JSON_START.search(s)
    if m:
        start = m.start()
        end = _find_json_span(s, start)
        if end is not None:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"无法从输出里解析出 JSON(前 200 字):{text[:200]!r}")


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 推理模型(deepseek-v4-flash 等)把思维链花在 reasoning_content 上,
    # 那部分 token 也计费,但不进 content。单独记一笔,账单才对得上。
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """线程安全的 OpenAI 兼容客户端。"""

    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or get_settings().llm
        self._client: Any = None
        self._lock = threading.Lock()
        # 累计用量。批量导入时这个数字直接决定账单,得看得见。
        self._usage = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0}
        self._usage_lock = threading.Lock()

    # ----------------------------------------------------------------- #
    # 基础
    # ----------------------------------------------------------------- #

    @property
    def configured(self) -> bool:
        return self.cfg.configured

    def require_configured(self) -> None:
        if not self.configured:
            raise LLMNotConfigured(
                "LLM_API_KEY 没配。在 .env 里填上 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。\n"
                "  DeepSeek: LLM_BASE_URL=https://api.deepseek.com/v1  LLM_MODEL=deepseek-chat\n"
                "  Qwen    : LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                "            LLM_MODEL=qwen-plus\n"
                "这个 key 是上下文增强、GraphRAG 实体抽取、ReAct 三处共用的。"
            )

    @property
    def client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI

                    self.require_configured()
                    self._client = OpenAI(
                        api_key=self.cfg.api_key,
                        base_url=self.cfg.base_url,
                        timeout=self.cfg.timeout,
                        max_retries=0,   # 重试自己控制,SDK 的重试看不到 Retry-After
                    )
        return self._client

    @property
    def usage(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def _record(self, r: ChatResult) -> None:
        with self._usage_lock:
            self._usage["prompt"] += r.prompt_tokens
            self._usage["completion"] += r.completion_tokens
            self._usage["reasoning"] += r.reasoning_tokens
            self._usage["calls"] += 1

    # ----------------------------------------------------------------- #
    # 调用
    # ----------------------------------------------------------------- #

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
    ) -> ChatResult:
        """一次对话补全。带退避重试。"""
        self.require_configured()

        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": list(messages),
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if stop:
            kwargs["stop"] = list(stop)
        if json_mode:
            # 不是所有兼容端点都支持 response_format,失败时退化成
            # 「靠提示词要 JSON + extract_json 兜底」
            kwargs["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                # json_mode 不被支持 —— 去掉重试一次,别白等退避
                if json_mode and ("response_format" in msg or "json_object" in msg):
                    logger.warning("该端点不支持 response_format,改用提示词约束 JSON")
                    kwargs.pop("response_format", None)
                    json_mode = False
                    continue

                last_exc = exc
                if attempt == self.cfg.max_retries - 1:
                    raise LLMError(
                        f"LLM 调用失败(试了 {self.cfg.max_retries} 次): {exc}"
                    ) from exc

                delay = self._backoff_delay(exc, attempt)
                logger.warning("LLM 调用出错(第 %d 次),%.1fs 后重试: %s",
                               attempt + 1, delay, msg[:160])
                time.sleep(delay)
        else:
            raise LLMError(f"LLM 调用失败: {last_exc}")

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        u = getattr(resp, "usage", None)
        finish = getattr(choice, "finish_reason", None)
        details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0

        result = ChatResult(
            text=text,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            reasoning_tokens=reasoning,
        )
        self._record(result)

        # 推理模型有个很难查的失败模式:max_tokens 全被 reasoning_content 吃掉,
        # content 一个 token 都没轮到,于是 content 是空串、但调用"成功"。
        # 实测 deepseek-v4-flash 在 max_tokens=16 时就是这样:completion_tokens=16、
        # reasoning_tokens=16、text=''。返回空串是最坏的结果 ——
        # ReAct 会以为模型什么都没说,白跑一轮格式纠错,最后把空串当答案交出去。
        # 这里必须显式报错,不然人会对着一个"没有报错但没答案"的界面发呆。
        if not text:
            consumed = result.completion_tokens
            if reasoning or finish == "length":
                raise LLMError(
                    f"模型只产出了推理内容,没有正文(completion={consumed} 个 token,"
                    f"其中推理 {reasoning} 个,finish_reason={finish!r})。\n"
                    f"这是推理模型把 max_tokens 花在思维链上的典型表现。"
                    f"把 .env 里的 LLM_MAX_TOKENS 调大(当前 {self.cfg.max_tokens}),"
                    f"或换一个非推理模型。"
                )
            if finish == "content_filter":
                raise LLMError(f"输出被内容策略拦截(finish_reason={finish!r})。")
            raise LLMError(
                f"模型返回了空内容(finish_reason={finish!r},"
                f"completion={consumed} 个 token)。"
            )

        return result

    @staticmethod
    def _backoff_delay(exc: Exception, attempt: int) -> float:
        """指数退避。服务端给了 Retry-After 就听它的。"""
        resp = getattr(exc, "response", None)
        if resp is not None:
            ra = None
            try:
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            except Exception:  # noqa: BLE001
                ra = None
            if ra:
                try:
                    return min(float(ra), 60.0)
                except (TypeError, ValueError):
                    pass
        return min(2.0 ** attempt, 30.0)

    def chat_json(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """要一段 JSON 回来。解析失败会重试一次(模型偶尔会跑偏)。"""
        r = self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        try:
            return extract_json(r.text)
        except ValueError:
            logger.warning("JSON 解析失败,追加一句要求后重试")
            retry_msgs = list(messages) + [
                {"role": "user", "content": "只输出合法 JSON,不要任何解释、不要 markdown 代码块。"}
            ]
            r = self.chat(retry_msgs, temperature=0, max_tokens=max_tokens, json_mode=True)
            return extract_json(r.text)

    # ----------------------------------------------------------------- #
    # 批量并发
    # ----------------------------------------------------------------- #

    def map_batch(
        self,
        fn: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        workers: int = 4,
        desc: str = "批量 LLM 调用",
        on_error: Callable[[Any, Exception], None] | None = None,
    ) -> list[Any]:
        """并发跑 fn(item)。**保持输入顺序**,失败的位置是 None。

        为什么不用 as_completed 的返回顺序:上游要把结果和原文档对齐,
        顺序错位是那种不报错、只是数据全串了的 bug。

        workers 默认 4 —— 不是越大越好,见模块开头关于 429 的说明。
        """
        items = list(items)
        if not items:
            return []

        results: list[Any] = [None] * len(items)
        # 同一个 LLM 客户端被多线程共用,底层 httpx 是线程安全的
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    # 单条失败不该毁掉整批导入
                    logger.warning("%s: 第 %d 条失败: %s", desc, i, exc)
                    results[i] = None
                    if on_error is not None:
                        try:
                            on_error(items[i], exc)
                        except Exception:  # noqa: BLE001
                            pass
                done += 1
                if done % 20 == 0 or done == len(items):
                    logger.info("%s: %d/%d", desc, done, len(items))
        return results


_llm: LLMClient | None = None
_llm_lock = threading.Lock()


def get_llm(cfg: LLMConfig | None = None) -> LLMClient:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = LLMClient(cfg)
    return _llm
