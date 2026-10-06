"""领域对象 → JSON 可序列化的 dict。

为什么要单独一层,而不是让路由里随手 `dataclasses.asdict`:

1. **`citation` 是个 property,不是字段。** `asdict` 看不见它 —— 而它恰恰是
   这个产品最要紧的那个字符串(答案里的「《X》 (块 3)」要能和结果列表对上)。
   路由里各写各的,迟早有一条路忘了带。

2. **`RetrievedChunk.meta` 是两条后端混用的口袋。** 混合检索往里塞 ranking
   细节,图检索往里塞 `from_graph` / `entities` / `facts`。前端要的是
   「这条是不是图捞回来的」「命中哪些实体」这些**语义**,不是 meta 的原始形状。
   把 meta 原样丢给前端,等于把两套内部约定焊死进前端代码,以后改一个键名
   就要动前端。

3. **`None` 和 `0.0` 必须区分开。** `rerank_score=None` 是"没开重排",
   `0.0` 是"重排给了零分"。前端把 `None` 当 `0` 显示,就会出现"没开重排"
   和"重排认为完全无关"看起来一模一样。所以**缺值一律保留 `None`**,不给
   默认值 —— 数字字段只有真算过才有意义。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from agent.react import AgentResult, AgentStep
from ingest.pipeline import IngestStats
from retrieve.hybrid import RetrievedChunk

__all__ = [
    "hit_to_dict",
    "hits_to_list",
    "steps_to_list",
    "result_to_dict",
    "ingest_stats_to_dict",
]


def _facts(meta: dict) -> list[dict]:
    """把 meta 里的三元组整成前端好渲染的形状。

    原形是 `{"head":…, "relation":…, "tail":…}` 再加几个内部字段。这里
    **只挑这三个**,不是省事 —— 图那边的事实记录还带着 score、来源块 id
    之类的内部信息,原样透出去会在前端形成一个没人维护的窄接口。
    """
    out: list[dict] = []
    for f in meta.get("facts") or []:
        if not isinstance(f, dict):
            continue
        out.append(
            {
                "head": str(f.get("head") or ""),
                "relation": str(f.get("relation") or ""),
                "tail": str(f.get("tail") or ""),
            }
        )
    return out


def hit_to_dict(r: RetrievedChunk) -> dict[str, Any]:
    """一条检索结果。字段形状见 `api/CONTRACT.md` 的 `Hit`。"""
    meta = r.meta or {}
    return {
        # citation 是 property,`asdict` 拿不到 —— 显式算出来
        "citation": r.citation,
        "text": r.text,
        "context": r.context,
        "doc_id": r.doc_id,
        "chunk_index": r.chunk_index,
        "source": r.source,
        "title": r.title,
        # 分数字段保留 None(见模块头第 3 条)
        "score": r.score,
        "rrf_score": r.rrf_score,
        "rerank_score": r.rerank_score,
        "dense_rank": r.dense_rank,
        "sparse_rank": r.sparse_rank,
        # 并排版的分数,给前端排序/画柱状图用:上面几个是"原始字段",
        # 带 None;这一份是"永远有数"的视图,免得前端每处都写 `?? 0`。
        # 两者同值,不是两套口径。
        "scores": {
            "final": r.score,
            "rrf": r.rrf_score,
            "rerank": r.rerank_score,
            "dense_rank": r.dense_rank,
            "sparse_rank": r.sparse_rank,
        },
        # 图检索专属。混合检索永远是 false / 空数组 —— 前端可以无条件读。
        "from_graph": bool(meta.get("from_graph")),
        "entities": [str(e) for e in (meta.get("entities") or [])],
        "facts": _facts(meta),
        "graph_hits": int(meta.get("graph_hits") or 0),
        # 版本。空串 = 当时没写这个字段,前端按"现行版"显示即可。
        "status": r.status,
        "doc_version": r.doc_version,
        "effective_from": r.effective_from,
        "effective_to": r.effective_to,
    }


def hits_to_list(hits: Iterable[RetrievedChunk]) -> list[dict[str, Any]]:
    return [hit_to_dict(h) for h in hits]


def steps_to_list(steps: Sequence[AgentStep]) -> list[dict[str, Any]]:
    return [
        {
            "index": s.index,
            "thought": s.thought,
            "action": s.action,
            "action_input": s.action_input,
            "observation": s.observation,
            "truncated": s.truncated,
            "repeated": s.repeated,
            "note": s.note,
        }
        for s in steps
    ]


def result_to_dict(result: AgentResult, *, elapsed_ms: int = 0) -> dict[str, Any]:
    """ReAct 的最终结果。

    流式那一路的 `done` 事件带的就是这个函数的产物 —— 两条路的响应体
    **同构**是刻意的,前端可以用同一段代码收尾(见 `react.py` 的 `done` 注释)。
    """
    return {
        "question": result.question,
        "answer": result.answer,
        "stop_reason": result.stop_reason,
        "steps": steps_to_list(result.steps),
        "usage": dict(result.usage or {}),
        "warnings": list(result.warnings or []),
        "elapsed_ms": elapsed_ms,
    }


def ingest_stats_to_dict(stats: IngestStats, *, elapsed_ms: int = 0) -> dict[str, Any]:
    """导入结果。

    `errors` 在 `IngestStats` 里是 `list[tuple[str, str]]`;JSON 里给成
    对象数组,让前端能分列显示"哪个文件"和"为什么" —— 拼成一个字符串,
    前端就只能整段显示,长错误信息会糊成一片。
    """
    return {
        "files_seen": stats.files_seen,
        "files_loaded": stats.files_loaded,
        "docs_indexed": stats.docs_indexed,
        "docs_unchanged": stats.docs_unchanged,
        "chunks_written": stats.chunks_written,
        "chunks_failed": stats.chunks_failed,
        "contextual_missing": stats.contextual_missing,
        "entities_written": stats.entities_written,
        "relations_written": stats.relations_written,
        "elapsed_ms": elapsed_ms or int((stats.elapsed or 0.0) * 1000),
        "errors": [{"file": str(f), "error": str(e)} for f, e in (stats.errors or [])],
    }
