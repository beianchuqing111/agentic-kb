"""版本化索引:一篇文档「是第几版」「还有效吗」。

为什么单独立一层
----------------
这套词汇(`status` 的取值、字段名、召回侧的过滤条件)同时被三方用到:

  - `store/`   —— 写 payload、建索引、回填老数据
  - `retrieve/`—— 召回时下过滤条件
  - `agent/`   —— `mark_superseded` 工具改这个字段

放在三方里的任何一方,另外两方就得反向依赖它。而 `agent/tools.py` 已经
`import retrieve.backends` 了,再让 `retrieve/` 去 import `agent/` 就是循环
导入。所以立一个**谁都可以依赖**的中立模块,依赖方向是单向的:

    agent/ ──▶ store/versioning ◀── retrieve/
                  ▲
                  └── store/qdrant_store

「缺字段 = 有效」是这层的核心约定
---------------------------------
加字段之前入库的 point 没有 `status`。要是把「没有这个字段」当成"未知"
或者干脆当成"已失效",那么**第一次上线就会让整个库从召回里消失** ——
而且不报错,只是什么都检索不到。

所以两个地方必须用同一套语义,漏一个就会出现「读出来是 current,却召回
不到」的鬼故事:

  1. `status_of()`      —— 代码里读它
  2. `current_filter()` —— Qdrant 里筛它(用 `IsEmptyCondition` 兜住缺字段)

**只用其中一处是不够的**:读的地方宽松、筛的地方严格,数据会在两处对不上。
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from qdrant_client import models as qm

__all__ = [
    "STATUS_FIELD",
    "STATUS_CURRENT",
    "STATUS_SUPERSEDED",
    "VERSION_FIELDS",
    "status_of",
    "is_superseded",
    "current_filter",
    "visible_filter",
    "parse_version_hint",
]


#: payload 里装状态的字段名。改它等于让所有老数据失效,别改。
STATUS_FIELD = "status"

STATUS_CURRENT = "current"
STATUS_SUPERSEDED = "superseded"

#: 除 `status` 外的版本字段。都是**可选**的:没有就是"没写",不当默认值用。
#: `doc_version` 是给人看的版本号(如 `v2`),`effective_from`/`effective_to`
#: 是生效区间,ISO 日期字符串 —— 存字符串而不是时间戳,是因为它主要给人和
#: LLM 看,而排序需求用 ISO 字符串的字典序就能满足。
VERSION_FIELDS = ("doc_version", "effective_from", "effective_to")


# --------------------------------------------------------------------------- #
# 读
# --------------------------------------------------------------------------- #

def status_of(payload: Mapping[str, Any] | None) -> str:
    """这块的版本状态。**字段缺失或取值不认识,一律当 `current`。**

    这里故意不返回 "unknown":调用方拿到 unknown 之后必须自己决定怎么处理,
    十有八九会被顺手写成一个默认分支,那个分支什么行为就看谁写的了。缺省
    直接收敛到 `current`,和 `current_filter()` 的 OR 分支严格对应。

    取值不认识(比如手改坏的 `"current "` 带空格)也当 `current`:这是
    **宽进**的一侧 —— 一个拼错的 status 让文档从召回里消失,比让它多召回
    一次危险得多,因为前者不报错。
    """
    v = (payload or {}).get(STATUS_FIELD)
    return v if v in (STATUS_CURRENT, STATUS_SUPERSEDED) else STATUS_CURRENT


def is_superseded(payload: Mapping[str, Any] | None) -> bool:
    """只判断"要不要按失效挡掉"。取反比直接比字符串少一处漏改。"""
    return status_of(payload) == STATUS_SUPERSEDED


# --------------------------------------------------------------------------- #
# 筛
# --------------------------------------------------------------------------- #

def current_filter() -> qm.Filter:
    """Qdrant 侧的「这块还有效」。

    写成 **OR**:`status == current` **或** 根本没有 `status` 字段。

    为什么必须有第二个分支:Qdrant 的 `MatchValue` 不匹配"字段不存在"的
    point。只写等值条件,回填之前入库的老数据会被**静默**过滤掉 ——
    不报错、不告警,查询就是"没有结果"。这跟 `status_of()` 的缺省语义
    是同一条:代码里当它有效,过滤器就得放它过去。
    """
    return qm.Filter(
        should=[
            qm.FieldCondition(key=STATUS_FIELD, match=qm.MatchValue(value=STATUS_CURRENT)),
            qm.IsEmptyCondition(is_empty=qm.PayloadField(key=STATUS_FIELD)),
        ]
    )


def visible_filter(extra: qm.Filter | None = None) -> qm.Filter:
    """召回侧的统一入口:只留有效版本,并且**不吞掉调用方自己的条件**。

    合并方式是「把对方的 filter 当成一个条件塞进 `must`」,而不是拆开它的
    must/should/must_not 再拼回去。拆开拼要正确复现 `should` 的语义(是
    "任一"还是"全部",嵌套几层)非常容易出错,而**拼错一次就是静默放宽或
    收紧召回** —— 查询照样返回结果,只是结果不对。嵌套 filter 是 Qdrant
    原生支持的,交给它做与运算,语义没有解释空间。

    调用方传 `None`(绝大多数情况)就直接返回有效期条件。
    """
    not_superseded = current_filter()
    if extra is None:
        return not_superseded
    return qm.Filter(must=[extra, not_superseded])


# --------------------------------------------------------------------------- #
# 从文件名认版本
# --------------------------------------------------------------------------- #

# `v2` / `V1.3`。两侧都**不能紧挨着 ASCII 字母数字**。
#
# 为什么用"ASCII 字母数字"而不是"必须是分隔符":中文文件名里 `细则v2.md`
# 比 `细则_v2.md` 更常见,要求前面是分隔符会把最常见的那种漏掉 —— 而漏掉
# 是**静默**的,文件照样入库,只是永远没有版本号。(第一版就是这么写的,
# 实测 `缺陷定级细则v2_2025-06-01.md` 只认出了日期。)
#
# 代价是 `1234v1.2` 这种"文号直接连着版本号"的写法认不出来。宁可漏,
# 不可错:认错的版本号会被当成真的去比较和展示,比没有更坏。
_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])[vV](\d+(?:\.\d+)*)(?![A-Za-z0-9])")
# 前后不能贴着别的数字,否则 `20250601` 里的片段会被切出来当日期。
_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def parse_version_hint(name: str) -> dict[str, str]:
    """从文件名里认出「第几版」「从哪天生效」。认不出就返回空 dict。

    这是**约定**,不是抽取:只认 `v2`、`v1.2` 和 ISO 日期这三种写法,认不出
    就不写字段。宁可没有版本号,也不要一个错的 —— 错的版本号会被当成真的
    去比较,比没有更坏。

    两个日期(如 `2025-06-01_2026-01-01`)按先后当作生效区间的两端。

    返回值只含 `doc_version` / `effective_from` / `effective_to`,**永远不含
    `status`** —— 新入库的东西一律是有效的,状态由 `mark_superseded` 改。
    让它出现在这里,等于给"从文件名把文档标成失效"开了个口子。
    """
    out: dict[str, str] = {}
    if not name:
        return out

    m = _VERSION_RE.search(name)
    if m:
        out["doc_version"] = f"v{m.group(1)}"

    dates = _DATE_RE.findall(name)
    if len(dates) >= 2:
        out["effective_from"], out["effective_to"] = sorted(dates)[0], sorted(dates)[1]
    elif len(dates) == 1:
        out["effective_from"] = dates[0]

    return out
