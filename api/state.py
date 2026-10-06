"""API 层的进程状态:一把锁、一个后端选择器、一个写工具确认通道。

这个模块存在的唯一理由是**后端是全局单例**
----------------------------------------
`retrieve/backends.get_backend()` 缓存的 `_backend` 是个**进程级可变全局**,
而 `webui.py` 的 `backend_for()` 会调 `set_backend()` 把它换掉。于是:

    请求 A:set_backend(graphrag) ──┐
    请求 B:      set_backend(hybrid) ──► A 后面的检索跑在 hybrid 上

这不是"慢一点"或者"偶发不一致",是**结果串台**:A 拿着图检索的期待,
收到的是纯向量的结果,而两者在界面上都长得像正常输出。更坏的是
`ReActAgent` 的工具表是在**构造时**从 `get_backend()` 取后端的 ——
所以锁必须覆盖「切后端 → 构造 agent → 跑完」整段,不能只锁检索那一下。

所以:**一把大锁,整段包住**。代价是没有并发;收益是不会有任何一个请求
跑在别人的后端上。`webui.py` 用 `default_concurrency_limit=1` 表达的是
同一件事(它的注释写着「后端是全局单例,一边导入一边问答会互相踩」),
这里用显式的锁,顺带把"谁在等"说得更清楚。

为什么不用 `build_backend()` 每个请求新建一个
------------------------------------------
那才是"真正的并发",但每个后端实例身上挂着 bge-m3(1.2G)和
bge-reranker-v2-m3 —— 每请求重建 = 每请求重载几 G 权重,几个请求之后
显存就爆了。这不是省事,是唯一可行的做法。

⚠️ 锁是 `threading.Lock` 而不是 `asyncio.Lock`:FastAPI 的同步路由跑在
线程池里,SSE 那一路还会把整段工作丢进独立线程跑(见 `app.py`)。跨线程
的互斥只能用线程锁。
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agent.permissions import AuditLog, WriteGuard
from config import BackendType, get_settings
from retrieve.backends import build_backend, get_backend, set_backend
from retrieve.backends.base import BaseBackend

logger = logging.getLogger(__name__)

__all__ = [
    "BACKEND_LOCK",
    "resolve_backend",
    "current_backend",
    "hold_backend",
    "api_guard",
    "WRITE_ACTOR",
]

#: 全局互斥。**所有**会碰后端或要跑 LLM 的操作都从这里过。
BACKEND_LOCK = threading.Lock()

#: 审计里标记来源。API 触发的写操作和终端里敲的写操作必须分得开 ——
#: 出了事第一个要回答的问题是"这是谁按的",而审计只记时间戳的话答不上来。
WRITE_ACTOR = "api"


def current_backend() -> BaseBackend:
    """取"此刻在用的那个后端",**不拿锁**。

    ⚠️ 只给「不切后端也不写库」的只读路由用(目前只有 `/api/health` 和
    `/api/backends`)。别拿它去跑检索 —— 那样就没有任何东西能保证
    取到的后端和后面的调用是同一个。

    存在的理由就是那个例外:健康检查必须在导入进行中(锁被占着几分钟)
    也能立刻回答,否则前端的"服务不可用"横幅会在最不该出现的时候出现。

    首次调用时 `get_backend()` 内部会懒建一个外壳(不加载任何模型权重,
    见模块头)。理论上两个线程同时第一次进来会各建一个 —— 这是**良性**的:
    两个外壳都合法,`_backend` 会被赋成其中一个,另一个被回收。
    相比"为它去抢那把大锁"(等于让健康检查排在导入后面),这个代价是划算的。
    """
    return get_backend()


def resolve_backend(name: str | None) -> str:
    """把请求里的后端名归一化。给不出合法名字就用配置里的默认值。

    名字不认识时**不报错、退回默认**是刻意的:后端名是请求体里一个
    可选字段,前端传了个旧版本残留的值('vector' 之类)时,让整个检索
    失败不如按默认跑 —— 但这个退回会记进日志,免得"怎么老是 hybrid"
    变成一桩悬案。
    """
    default = get_settings().backend.value
    if not name:
        return default
    try:
        return BackendType(name).value
    except ValueError:
        logger.warning("不认识的后端名 %r,退回默认 %r", name, default)
        return default


def select_backend(name: str | None) -> BaseBackend:
    """按名字切全局后端并返回它。

    **必须在 `hold_backend()` 里面调。** 单独调它会留下一个被换掉的
    全局状态,下一个请求拿到的后端就不是它以为的那个了。
    """
    want = resolve_backend(name)
    current = get_backend()
    if current.name == want:
        return current
    # 已经建过的后端不重建 —— HybridBackend/GraphRAGBackend 都是轻壳子,
    # 真正的模型权重挂在更下层的单例上(get_embedder / get_reranker),
    # 所以这里换掉外壳不会重载模型。重建才是贵的那个。
    fresh = _CACHE.get(want)
    if fresh is None:
        fresh = build_backend(want)
        _CACHE[want] = fresh
    set_backend(fresh)
    logger.info("API 切后端:%s → %s", current.name, want)
    return fresh


#: 建过的后端外壳。和 `retrieve/backends` 的 `_backend` 不同:那个是
#: "当前用哪个",这个是"建过哪些"。分开是为了让来回切换不用重建外壳,
#: 也为了让下面的断言能检查"我没把两个名字指向同一个实例"。
_CACHE: dict[str, BaseBackend] = {}


@contextmanager
def hold_backend(name: str | None = None) -> Iterator[BaseBackend]:
    """拿锁 → (要切就切后端) → 交给调用方 → 放锁。

    这是**唯一**允许切后端的口子。整段在锁里,所以不存在"两个请求同时
    认为自己持有后端 X"。

    `name` 为空 = **用服务端当前那个**,不是"退回配置默认值"。这两者
    在第一个请求上恰好相等,之后就分岔了:前端的选择器是个跨页签的
    单选按钮(和 `webui.py` 那一排 radio 一样),用户在检索页选了
    graphrag、切到文档页,文档页应该还是 graphrag。每次省略都弹回配置
    默认的话,界面上会出现"我明明选了 graphrag,列表却是 hybrid 的"
    —— 而两个后端的文档列表本来就一模一样,这种不一致根本看不出来,
    只会让人以后不敢信这个选择器。
    """
    with BACKEND_LOCK:
        yield select_backend(name) if name else get_backend()


def api_guard(
    *,
    allow_write: bool = False,
    confirm_write_tools: list[str] | None = None,
    exports_dir: Path | None = None,
    audit: AuditLog | None = None,
) -> WriteGuard:
    """给 API 请求拼一个权限层。

    这里要正面回答一个问题:**没有终端,谁来点那个头?**

    直接把 `confirmer` 省掉是最省事的,但那等于写工具永远不可用
    (`permissions.authorize` 第 2 闸写着「没有 confirmer 也算拒绝」,
    这条不能破)。所以 API 必须自己给一个确认通道,而这个通道的口径是:

      - 闸 1 `allow_write` —— 服务端/请求级总开关,和 CLI 的 `--allow-write` 同义。
      - 闸 2 `confirm_write_tools` —— **逐个工具点名**。

    为什么要第二个字段,而不是"`allow_write=true` 就等于全同意":
    那样两闸就塌成一闸,`allow_write` 一个人说了算。分开之后,一个
    `allow_write=true` 但没点名任何工具的请求**照样被拒**(闸 2 拒),
    前端必须明确写出「我允许调用 export_report」这件事。这是无头环境下
    能保住"显式确认"语义的最强形式 —— 代价是它批准的是**工具**而不是
    **这一次调用的参数**(参数是模型后面才生成的,API 形态下没法先看后批)。
    这个代价在面试里要如实说,别吹成"等价于人工确认"。

    `exports_dir` 必须和交给 `build_default_tools()` 的是**同一个值**:
    闸 3 校验的是一个根、工具往里写的是另一个根的话,校验就形同虚设
    (见 `agent/tools.py:build_registry` 的同一条注释)。这里不做默认值,
    由调用方显式给 —— 默认值会让"忘了传"变成一个静默的弱校验。
    """
    approved = {t.strip().lower() for t in (confirm_write_tools or []) if t.strip()}

    def confirm(name: str, arg: str) -> bool:
        return name.strip().lower() in approved

    return WriteGuard(
        allow_write=allow_write,
        confirmer=confirm,
        exports_dir=exports_dir,
        audit=audit,
        actor=WRITE_ACTOR,
    )
