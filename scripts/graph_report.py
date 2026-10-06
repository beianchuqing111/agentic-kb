"""把 hybrid vs graphrag 的对照出成一份报告(eval/reports/graphrag_ab.md)。

为什么要脚本而不是手写
----------------------
待补实证清单 #6 的问法是「多跳到底是靠向量还是靠图」——
面试里会被追问的是**这两个数字怎么来的**。手写一份 md 数字好看但复核不了;
这里每一格都从基线 JSON 现算,重跑一次就能自证。

判读口径:先看「有没有变化」,再看「往哪变」
------------------------------------------
这份报告最重要的一条纪律是**不把「没变化」当结论**。图上加了一整条多跳通路,
而分数逐位相同,只有两种可能:

  1. 图那一路压根没接上(配置事故) —— 报告里有 `graph_hits` 一栏专门分开这两者
  2. 语料太小,61 块里没有需要跳的空间

两者在聚合分数上**长得一模一样**。所以本报告除了分数,还必须给出
「graphrag 那一档实际从图上多召回了多少块」—— 没有这个数字,
「GraphRAG 没用」和「GraphRAG 没跑」就分不开。

用法:
    python scripts/graph_report.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.report import comparability_problems, print_diff  # noqa: E402

BASELINES = ROOT / "eval" / "baselines"
OUT = ROOT / "eval" / "reports" / "graphrag_ab.md"

BASE = "b1b2_graph.json"
HYBRID = "no-threshold+wide"
GRAPHRAG = "graphrag:no-threshold+wide"
KS = [1, 3, 5, 10]

#: 语料的题型分布里,只有这几类**理论上**能体现多跳通路的价值。
#: `lexical`/`semantic` 这些单跳题两档本来就该一样 —— 它们的作用是
#: **对照组**:如果连它们都变了,那变的不是多跳能力,而是别的东西串味了。
HOP_TYPES = ("multi_hop",)


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _capture(fn, *a, **kw) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **kw)
    return buf.getvalue()


def _one_run(base: dict, runkey: str, label: str) -> dict:
    """从一份基线里摘出单个 run、换一个公共 key,好让 print_diff 能拿它两两比。"""
    d = {k: v for k, v in base.items() if k != "runs"}
    d["runs"] = {label: base["runs"][runkey]}
    return d


def _row(run: dict, k: int) -> str:
    a = run["aggregates"][str(k)]
    return (f"| {k} | {a['hit']:.4f} | {a['rr']:.4f} | {a['ndcg']:.4f} | "
            f"{a['recall']:.4f} | {a['n']} |")


def _by_type_rows(run: dict, k: int) -> list[tuple[str, dict]]:
    bt = (run.get("by_type") or {}).get(str(k)) or {}
    return sorted(bt.items(), key=lambda kv: kv[0])


def _graph_usage(run: dict) -> dict:
    """graphrag 这一档实际从图上拿到了什么。

    `per_item` 里每题记了返回的块数;`graph_hits` 是可选的(只有图那一路会填)。
    这里不假设字段一定在 —— 缺了就报「没记」,不拿 0 冒充 0 个图命中。
    """
    per_item = run.get("per_item") or []
    with_hits = [d for d in per_item if "graph_hits" in d]
    return {
        "n_items": len(per_item),
        "n_recorded": len(with_hits),
        "total": sum(int(d.get("graph_hits") or 0) for d in with_hits),
        "any": sum(1 for d in with_hits if (d.get("graph_hits") or 0) > 0),
    }


def main() -> int:
    base = _load(BASELINES / BASE)
    runs = base.get("runs") or {}
    for key in (HYBRID, GRAPHRAG):
        if key not in runs:
            print(f"!! 基线里没有 run `{key}`,只有:{sorted(runs)}", file=sys.stderr)
            return 1

    hy, gr = runs[HYBRID], runs[GRAPHRAG]

    L: list[str] = []
    ap = L.append

    ap("# GraphRAG vs 纯向量:多跳到底有没有用")
    ap("")
    ap(f"语料 `{base['corpus']['slug']}` {base['corpus']['n_docs']} 篇 / "
       f"{base['corpus']['n_chunks']} 块;问答集 {base['qa']['n_kept']} 题"
       f"(可答 {base['qa']['n_answerable']} / 反例 {base['qa']['n_negative']})。")
    ap("")
    ap("两档由**同一个 spec、同一个 top_k、同一份语料与问答集**跑出,"
       "唯一差别是 `--retriever`(`hybrid` vs `graphrag`)。"
       "graphrag 那一档多一条「图上多跳」的召回通路,融合进同一份 RRF 结果。")
    ap("")
    ap("---")
    ap("")

    # ---------------- 1. 头条 ----------------
    ap("## 1. 总指标")
    ap("")
    ap(f"| k | hit | mrr | ndcg | recall | n |  ← **hybrid**（`{HYBRID}`）")
    ap("|---|---|---|---|---|---|---|")
    for k in KS:
        ap(_row(hy, k))
    ap("")
    ap(f"| k | hit | mrr | ndcg | recall | n |  ← **graphrag**（`{GRAPHRAG}`）")
    ap("|---|---|---|---|---|---|---|")
    for k in KS:
        ap(_row(gr, k))
    ap("")

    a_h = hy["aggregates"]["1"]
    a_g = gr["aggregates"]["1"]
    ap("**@1 对照:**")
    ap("")
    ap("| 指标 | hybrid | graphrag | 变化 |")
    ap("|---|---|---|---|")
    for label, key in [("hit@1", "hit"), ("mrr@1", "rr"),
                       ("ndcg@1", "ndcg"), ("recall@1", "recall")]:
        b, a = a_h[key], a_g[key]
        mark = "" if abs(a - b) < 1e-9 else (" ↑" if a > b else " ↓")
        ap(f"| {label} | {b:.4f} | {a:.4f} | {a - b:+.4f}{mark} |")
    ap("")

    # ---------------- 2. 分题型 ----------------
    ap("## 2. 分题型(多跳看这里)")
    ap("")
    ap("这一节是本次对照的**主要读法**。`multi_hop` 是多跳通路唯一能体现价值的题型;")
    ap("其余题型在这里是**对照组** —— 单跳题两档本就该一致,若它们也动了,")
    ap("说明变的不是多跳能力,而是别的东西串了味。")
    ap("")
    for k in (1, 5):
        hbt = dict(_by_type_rows(hy, k))
        gbt = dict(_by_type_rows(gr, k))
        ap(f"### k={k}")
        ap("")
        ap("| 题型 | n | hybrid hit | graphrag hit | 变化 | hybrid ndcg | graphrag ndcg |")
        ap("|---|---|---|---|---|---|---|")
        for t in sorted(set(hbt) | set(gbt)):
            h, g = hbt.get(t, {}), gbt.get(t, {})
            nh, ng = int(h.get("n", 0)), int(g.get("n", 0))
            hh, gh = float(h.get("hit", 0.0)), float(g.get("hit", 0.0))
            hn, gn = float(h.get("ndcg", 0.0)), float(g.get("ndcg", 0.0))
            mark = "" if abs(gh - hh) < 1e-9 else (" ↑" if gh > hh else " ↓")
            star = " **←**" if t in HOP_TYPES else ""
            ap(f"| `{t}`{star} | {nh}/{ng} | {hh:.4f} | {gh:.4f} | {gh - hh:+.4f}{mark} | "
               f"{hn:.4f} | {gn:.4f} |")
        ap("")

    # ---------------- 3. 图到底被用上了没有 ----------------
    ap("## 3. 图那一档实际召回了多少(先看这个,再看分数)")
    ap("")
    ap("**这一节没有,下面的分数就不能读。** 图上加了一条通路而分数纹丝不动,")
    ap("只有两种解释:通路没接上(配置事故),或语料太小区分不出来。")
    ap("两者在聚合分上完全一样,只能靠这里分开。")
    ap("")
    usage = _graph_usage(gr)
    ap("| 项 | 值 |")
    ap("|---|---|")
    ap(f"| 记了图命中的题数 | {usage['n_recorded']} / {usage['n_items']} |")
    ap(f"| 至少有一条图命中的题 | {usage['any']} |")
    ap(f"| 图命中块数合计 | {usage['total']} |")
    ap("")
    if usage["n_recorded"] == 0:
        ap("> ⚠ **基线里没有逐题的图命中记录** —— 无法区分「图没接上」和「图接上了但没用」。")
        ap("> 需要在 `run_spec` 里把 `graph_hits` 记进 `per_item` 才能判读。")
    elif usage["any"] == 0:
        ap("> ⚠ **没有任何一题从图上多召回块** —— 这一档等价于跑了一遍纯向量。")
        ap("> 下面的分数**不是** GraphRAG 的效果,是 hybrid 的效果。")
    else:
        ap(f"> 有 {usage['any']} 题用到了图,合计多召回 {usage['total']} 块。")
        ap("> 分数差异可以归因到这条通路。")
    ap("")

    # ---------------- 4. 可比性 ----------------
    ap("## 4. 可比性检查")
    ap("")
    problems = comparability_problems(hy, gr)
    ap("`comparability_problems(hybrid_run, graphrag_run)`:")
    ap("")
    if not problems:
        ap("- 空清单。")
    else:
        ap(f"- 报了 **{len(problems)}** 项:")
        for p in problems:
            ap(f"  - `{p}`")
        ap("")
        ap("  判读口径同 CT 报告:**看清单里有几项**。这里的 run 名字带 `graphrag:` 前缀,")
        ap("  所以「配置变体不同」这一项必然出现 —— 它**就是本次的干预本身**。")
        ap("  只要它没同时报出问答集 sha1 / 分块大小 / 嵌入模型 / 重排模型,就是干净对照。")
    ap("")

    # ---------------- 5. 逐题 ----------------
    ap("## 5. 逐题原始输出(`report.py`)")
    ap("")
    ap("下面直接贴 `print_diff` 的原话,不转述 —— 聚合看不见的那一层在这里。")
    ap("")
    ap("```")
    ap(_capture(
        print_diff,
        _one_run(base, HYBRID, "h"),
        _one_run(base, GRAPHRAG, "g"),
        ks=[1, 5],
    ).strip())
    ap("```")
    ap("")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
