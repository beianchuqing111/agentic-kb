"""受控写工具的权限层:白名单 → 确认 → 参数校验 → 审计。

为什么单独一个模块
-----------------
`agent/tools.py` 的模块头写着「工具本身都无副作用(只读),即便模型被骗
也没有可破坏的东西」。那句话在只有只读工具时是成立的,而且是那一层真正的
兜底 —— 提示词注入没有银弹,但「注入成功也无事可做」是硬保证。

**挂上写工具,这条硬保证就没了。** 注入的目标从「把回答带偏」升级成
「让智能体替你写文件、改索引」。所以写工具不能只是「再注册一个 Tool」,
它必须同时带上三道闸:

  1. **白名单** —— 只有显式声明了写意图的工具能产生副作用。闸门按 `kind`
     判,不按工具名判:名字是字符串,可以重名、可以有大小写变体,而
     `kind` 是注册时定死的。
  2. **显式确认** —— 没人点头就拒绝(fail closed)。特别注意
     「配不出 confirmer」**不等于**「默认同意」:无头环境(批处理、定时
     任务、API 调用)本来就没人能确认,那里就该拒绝,而不是放行。
  3. **参数校验** —— 工具参数直接来自模型输出,和用户输入一样不可信。
     导出路径必须是裸文件名且 resolve 之后仍落在 `exports/` 内,另加后缀
     白名单(能写 `.bat` 的导出工具等于送了一条执行路径)。

审计另算
--------
上面三条管「事前能不能拦」,审计管「事后能不能查」。拦得住但查不到,
出了事说不清;查得到但拦不住,事情已经发生了。两件都要。

**写入时机是「先意图,后执行,再结果」。** 只在副作用之后写日志的话,
"进程在写文件那一瞬间崩了"这条路径在日志里是空白的 —— 而它恰恰最需要
留痕。所以先落一条 `phase=intent`(含参数),执行完再落一条 `phase=result`
(含成败摘要),两条靠同一个 `call_id` 串起来。

**被拒的调用也留痕**(`phase=denied`)。这一条容易被漏掉:只记成功的话,
日志上一切正常,而系统可能正被反复试探 —— 越权尝试正是注入的指纹。

日志写失败要**抛**,不吞:审计落不下去的时候照常执行,等于把这道闸关了。
`ToolRegistry.run` 会把这种异常也转成拒绝 —— 失败方向是「不许写」,不是
「随便写」。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from config import EXPORT_DIR, LOG_DIR

logger = logging.getLogger(__name__)

#: 工具的两种副作用性质。写在 permissions 而不是 tools 里,是因为
#: 权限层要按它判闸,而 tools 反过来要 import 本模块 —— 常量放这里
#: 才不会绕成一个环。
READ = "read"
WRITE = "write"

#: 导出只允许这几种后缀。**这不是洁癖**:导出目录里的文件是能被双击
#: 打开的,`.bat` / `.ps1` / `.exe` 落进去就等于绕开权限层拿到一条执行
#: 路径。用白名单而不是黑名单 —— 黑名单永远漏。
ALLOWED_EXPORT_SUFFIXES = frozenset({".md", ".txt", ".json", ".csv"})

#: 文件名长度上限。Windows 上过长的名字直接 OSError,而那时半成品
#: 可能已经落盘了,不如提前拒。
EXPORT_NAME_MAX = 80

#: 审计记录里参数/结果摘要的上限 —— 正文动辄几万字,不能整段进日志。
AUDIT_TEXT_MAX = 500

#: (工具名, 参数) -> 是否同意执行。
Confirmer = Callable[[str, str], bool]


class ToolDenied(PermissionError):
    """工具在**执行阶段**拒绝了这次调用(参数校验不过、数据状态不安全)。

    为什么要有这个异常,而不是让工具返回一句「错误:...」:工具返回字符串时
    `ToolRegistry` 只看到"调用完成了",审计里会记成 `phase=result, ok=True`
    —— 于是一次被拒的路径穿越尝试,和一次正常的导出**长得一模一样**。
    审计日志的价值就在于区分这两件事。

    继承 `PermissionError` 是为了兼容那些按它来接的调用方,但真正重要的是
    它带着一个**给模型看的理由**:权限层据此记 `phase=denied`。

    ⚠️ 边界:只有**安全性质的拒绝**走这里(沙箱越界、可执行后缀、数据处于
    不能安全处理的状态)。「没找到这个 doc_id」「这次查询没结果」这类是
    **正常的工具结果**,照常记 result —— 把「不许」和「没有」混成一种,
    审计就读不出攻击信号了。
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _clip(text: str, limit: int = AUDIT_TEXT_MAX) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"…(+{len(text) - limit} 字)"


# --------------------------------------------------------------------- #
# 审计
# --------------------------------------------------------------------- #


class AuditLog:
    """append-only 的 JSONL 审计日志。

    append-only 在这里是**约定**而不是**强制** —— 普通文件系统给不了
    WORM 语义。能做的只有:只以 `"a"` 模式打开、不做原地截断、不提供
    任何删除接口。真正的防篡改要靠日志外送(WORM 存储 / 采集到 SIEM),
    那是部署层面的事,不在这里假装已经做到了。
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path else (LOG_DIR / "audit.jsonl")

    def record(
        self,
        *,
        call_id: str,
        phase: str,
        tool: str,
        arg: str = "",
        ok: bool | None = None,
        detail: str = "",
        actor: str = "",
    ) -> dict:
        rec: dict = {
            "ts": _now_iso(),
            "call_id": call_id,
            "phase": phase,
            "tool": tool,
            "arg": _clip(arg),
            "actor": actor or os.getenv("USERNAME") or os.getenv("USER") or "",
        }
        if ok is not None:
            rec["ok"] = bool(ok)
        if detail:
            rec["detail"] = _clip(detail)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            # fsync:崩在「写了但没落盘」上,这条审计就等于没写,
            # 而它记的正是"接下来要动数据"这件事。
            os.fsync(f.fileno())
        return rec

    def tail(self, n: int = 20) -> list[dict]:
        """读最近 n 条(自检用)。"""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()][-n:]


# --------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------- #


def safe_export_path(name: str, *, exports_dir: Path | None = None) -> Path:
    """把模型给的文件名解析成 `exports/` 里的一个绝对路径,越界就拒。

    三类要挡的东西:

      - **目录穿越**:`../../config.py`
      - **绝对路径**:Windows 上 `Path("a") / "D:\\\\x"` 会**整个丢掉左边**
        (pathlib 的规则:右操作数是绝对路径时直接替换左操作数),不挡
        就是任意位置写入
      - **可执行后缀**:见 ``ALLOWED_EXPORT_SUFFIXES`` 的注释

    **带目录成分的一律拒,而不是悄悄取 basename。** 静默净化会把一次
    注入尝试伪装成一次正常调用 —— 参数看着合法、日志里也看不出异常,
    而这正是审计最该看见的东西。宁可报错。

    最终判据是 `resolve()` 之后的 `is_relative_to`,不是字符串前缀:
    字符串比较挡不住 `..`、挡不住符号链接、也挡不住 Windows 的短名。
    """

    raw = (name or "").strip().strip("`'\"")
    if not raw:
        raise PermissionError("导出文件名是空的")
    if raw in {".", ".."}:
        raise PermissionError(f"导出文件名不合法:{raw!r}")
    if "/" in raw or "\\" in raw or Path(raw).is_absolute() or Path(raw).drive:
        raise PermissionError(
            f"导出文件名不能带目录成分(挡目录穿越与绝对路径):{raw!r}"
        )
    if len(raw) > EXPORT_NAME_MAX:
        raise PermissionError(f"导出文件名太长({len(raw)} > {EXPORT_NAME_MAX}):{raw[:40]!r}…")

    suffix = Path(raw).suffix.lower()
    if suffix not in ALLOWED_EXPORT_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_EXPORT_SUFFIXES))
        raise PermissionError(f"不允许导出 {suffix or '(无后缀)'} 文件;只允许 {allowed}")

    root = (exports_dir or EXPORT_DIR).resolve()
    target = (root / raw).resolve()
    if not target.is_relative_to(root):
        # basename 已经取过,还能跑到这里说明中间有符号链接之类的花活
        raise PermissionError(f"导出路径越出了 {root}:{raw!r}")
    return target


def unique_path(target: Path) -> Path:
    """同名文件已存在就加序号,**不覆盖**。

    导出是追加语义:覆盖会让上一次的产物无声消失,而调用方通常
    想不到自己刚毁掉了什么。
    """
    if not target.exists():
        return target
    for i in range(1, 1000):
        cand = target.with_name(f"{target.stem}-{i}{target.suffix}")
        if not cand.exists():
            return cand
    raise PermissionError(f"{target.name} 的同名文件已超过 999 个,不再自动改名")


# --------------------------------------------------------------------- #
# 授权
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ticket:
    """一次**已授权**的写操作。`settle` 靠它把结果记回同一个 call_id。"""

    call_id: str
    tool: str
    arg: str


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    ticket: Ticket | None = None


class WriteGuard:
    """写工具的三道闸 + 审计。`ToolRegistry.run` 是唯一的把关点。

    只读工具**不进这个类**:它们既不需要授权也不需要审计(`run` 里直接
    短路返回)。把只读调用也灌进审计日志,只会让真正要看的写记录淹在
    噪声里。
    """

    def __init__(
        self,
        *,
        allow_write: bool = False,
        confirmer: Confirmer | None = None,
        exports_dir: Path | None = None,
        audit: AuditLog | None = None,
        actor: str = "",
    ) -> None:
        self.allow_write = allow_write
        self.confirmer = confirmer
        self.exports_dir = Path(exports_dir) if exports_dir else EXPORT_DIR
        self.audit = audit or AuditLog()
        self.actor = actor

    def authorize(
        self, *, name: str, kind: str, requires_confirm: bool, arg: str
    ) -> Decision:
        """闸 1 + 闸 2,通过则落一条 intent 并发出 ticket。"""
        if kind != WRITE:
            return Decision(True)

        def deny(reason: str) -> Decision:
            """拒绝,并且**留痕**。

            成功的写操作记的是「发生了什么」,被拒的写操作记的才是
            「有人试过什么」—— 提示词注入、模型幻觉出来的越界参数、
            一次写错路径的手滑,全都只在拒绝记录里看得见。只记成功的话,
            审计日志会显示一切正常,而系统可能正在被反复试探。

            这里**不吞异常**:审计写不下去时异常会冒到 `ToolRegistry.run`,
            那一层同样按拒绝处理。方向始终是"不许写"。
            """
            self.audit.record(
                call_id=uuid.uuid4().hex[:12],
                phase="denied",
                tool=name,
                arg=arg or "",
                ok=False,
                detail=reason,
                actor=self.actor,
            )
            return Decision(False, reason)

        # 闸 1:总开关。默认关,要用的人显式打开。
        if not self.allow_write:
            return deny(f"写工具 {name} 未启用(allow_write=False,默认关闭)")

        # 闸 2:确认。**没有 confirmer 也算拒绝**。
        if requires_confirm:
            if self.confirmer is None:
                return deny(
                    f"写工具 {name} 需要人工确认,但当前环境没有确认通道(按拒绝处理)"
                )
            try:
                agreed = bool(self.confirmer(name, arg or ""))
            except Exception as exc:  # noqa: BLE001
                # 确认环节自己出错时**按拒绝处理**。这里绝不能"出错了就放行"。
                return deny(f"确认环节出错,按拒绝处理:{type(exc).__name__}: {exc}")
            if not agreed:
                return deny(f"用户拒绝执行写工具 {name}")

        # 闸 3(路径穿越、后缀白名单)是参数级的,由各写工具自己调
        # `safe_export_path` 之类的校验 —— 权限层不知道每个工具的参数长什么样。
        ticket = Ticket(call_id=uuid.uuid4().hex[:12], tool=name, arg=arg or "")
        # 先写意图再放行:哪怕接下来崩了,日志里也能看到"它本来要干什么"。
        self.audit.record(
            call_id=ticket.call_id, phase="intent", tool=name, arg=ticket.arg, actor=self.actor
        )
        return Decision(True, ticket=ticket)

    def settle(self, ticket: Ticket | None, *, ok: bool, detail: str = "") -> None:
        """执行完了,把成败记回同一行。只读工具(无 ticket)直接跳过。"""
        if ticket is None:
            return
        self.audit.record(
            call_id=ticket.call_id,
            phase="result",
            tool=ticket.tool,
            arg=ticket.arg,
            ok=ok,
            detail=detail,
            actor=self.actor,
        )

    def deny(self, ticket: Ticket | None, *, reason: str) -> None:
        """**执行阶段**的拒绝(工具抛了 `ToolDenied`)。

        闸 1 / 闸 2 的拒绝走 `authorize` 里的 `deny()`(那时还没有 ticket,
        调用连 intent 都没落);这里是过了前两闸、拿到 ticket 之后才被工具
        自己拒掉的,所以它会把那条 intent 收尾成 `denied` 而不是 `result`。
        两种都记 `phase=denied`,查询时一条 `where phase==denied` 就能捞全。
        """
        if ticket is None:
            return
        self.audit.record(
            call_id=ticket.call_id,
            phase="denied",
            tool=ticket.tool,
            arg=ticket.arg,
            ok=False,
            detail=reason,
            actor=self.actor,
        )


def make_cli_confirmer() -> Confirmer:
    """终端里问一句 y/N。

    **默认 N**:直接回车、EOF、Ctrl-C 全算拒绝。读不到输入时返回 False
    而不是 True —— 这是这道闸唯一说得通的默认值。
    """

    def confirm(name: str, arg: str) -> bool:
        try:
            ans = input(
                f"\n[写操作] {name}\n  参数: {_clip(arg, 200)}\n  确认执行? [y/N] "
            )
        except (EOFError, KeyboardInterrupt):
            print("\n(没读到确认,按拒绝处理)")
            return False
        return ans.strip().lower() in {"y", "yes", "是", "确认"}

    return confirm


__all__ = [
    "READ",
    "WRITE",
    "ALLOWED_EXPORT_SUFFIXES",
    "EXPORT_NAME_MAX",
    "AuditLog",
    "Confirmer",
    "Decision",
    "Ticket",
    "ToolDenied",
    "WriteGuard",
    "make_cli_confirmer",
    "safe_export_path",
    "unique_path",
]
