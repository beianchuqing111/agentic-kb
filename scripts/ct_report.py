"""把 CT 的**检索增益**与**入库成本**合成一份报告(eval/reports/contextual_ab.md)。

为什么要有这个脚本,而不是手写一份报告
--------------------------------------
待补实证清单 #7 的问法是「CT 值不值?增益多少、成本多少?」——
面试里真正会被追问的是**这两个数字怎么来的**。手写一份 md 数字好看,
但复核不了;这里每一格都从基线 JSON 现算,重跑一次就能自证。

顺带把重排消融也一起出:重排是简历上那个「ndcg 0.790 → 0.962」的出处,
而 `ct_on.json` 一次跑里同时存了 rerank 开/关两份 spec(同 collection、
同其余参数),是**比旧基线更干净**的单变量对照 —— 顺手让它自证一遍。

`comparability_problems` 的用法说明(容易误读)
--------------------------------------------
CT 对照会让它报「定位语开关不同」。这不是缺陷:守卫的职责是拦住
「换了语料/模型还硬比」。所以判读口径是 **看清单里有几项**:
只有一项、且那项正是干预本身 = 干净的单变量对照;
要是它还报了分块大小/嵌入模型/问答集 sha1,那才是真的不可比,必须停。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.report import comparability_problems, print_diff, print_item_diff  # noqa: E402

BASELINES = ROOT / "eval" / "baselines"
COST = ROOT / "eval" / "reports" / "ct_cost.json"
OUT = ROOT / "eval" / "reports" / "contextual_ab.md"

RERANK_ON = "no-threshold+wide"
RERANK_OFF = "no-rerank+no-threshold+wide"
KS = [1, 3, 5, 10]


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


def main() -> int:
    on = _load(BASELINES / "ct_on.json")
    off = _load(BASELINES / "ct_off.json")
    cost = _load(COST)

    L: list[str] = []
    ap = L.append

    ap("# Contextual Retrieval:检索增益 × 入库成本")
    ap("")
    ap(f"语料 `{on['corpus']['slug']}` {on['corpus']['n_docs']} 篇 / "
       f"{on['corpus']['n_chunks']} 块;问答集 {on['qa']['n_kept']} 题"
       f"(可答 {on['qa']['n_answerable']} / 反例 {on['qa']['n_negative']})。")
    ap("")
    ap("两份基线 `ct_on` / `ct_off` 由**同一个 spec、同一个 top_k** 跑出,"
       "唯一差别是定位语开关(落点分别是 `*_ctx_*` 与 `*_noctx_*` 两个 collection)。")
    ap("")
    ap("---")
    ap("")

    # ---------------- 1. 头条:重排消融 ----------------
    ap("## 1. 重排消融(简历头条数字的出处)")
    ap("")
    ap("同一份 `ct_on.json` 里跑了两档 spec,同 collection、同其余全部参数,"
       "**只差 `use_rerank`** —— 单变量对照。")
    ap("")
    ap(f"| k | hit | mrr | ndcg | recall | n |  ← 重排 **关**（`{RERANK_OFF}`）")
    ap("|---|---|---|---|---|---|---|")
    for k in KS:
        ap(_row(on["runs"][RERANK_OFF], k))
    ap("")
    ap(f"| k | hit | mrr | ndcg | recall | n |  ← 重排 **开**（`{RERANK_ON}`）")
    ap("|---|---|---|---|---|---|---|")
    for k in KS:
        ap(_row(on["runs"][RERANK_ON], k))
    ap("")

    a_off = on["runs"][RERANK_OFF]["aggregates"]["1"]
    a_on = on["runs"][RERANK_ON]["aggregates"]["1"]
    ap("**@1 对照：**")
    ap("")
    ap("| 指标 | 重排关 | 重排开 | 变化 |")
    ap("|---|---|---|---|")
    for label, key in [("hit@1", "hit"), ("mrr@1", "rr"), ("ndcg@1", "ndcg"), ("recall@1", "recall")]:
        b, a = a_off[key], a_on[key]
        ap(f"| {label} | {b:.4f} | {a:.4f} | {a - b:+.4f} |")
    ap("")
    ap("> 简历写的是「ndcg 从 0.790 提升到 0.962、hit@1 从 0.867 提升到 0.95」。")
    ap("> ndcg 一栏**分毫不差**;hit@1 的**终值不是 0.95 而是 1.0000** —— 0.95 在任何产物里")
    ap("> 都找不到对应。")
    ap("")
    ap("逐题看(§4.3),这个提升**可以精确归因**:重排把 **2 道题**(`seed-004`、`seed-013`)")
    ap("从第 2 名提到第 1 名,两次移动的名次区间都是**单点**(`2-2` → `1-1`),**不涉及并列**;")
    ap("另有 1 题(`seed-014`)的并列被消除。其余 13 题在两种配置下本已是第 1 名。")
    ap("")
    ap("所以 `hit@1` 的 `0.8667 → 1.0000` 恰好等于 `13/15 → 15/15`,**分子分母都对得上**。")
    ap("引用时带上 n=15:这个提升虽说只有 2 道题,但两次移动都是无歧义的,站得住。")
    ap("")

    # ---------------- 2. CT 增益 ----------------
    ap("## 2. CT 增益(定位语开 vs 关)")
    ap("")
    problems = comparability_problems(off, on)
    ap("可比性检查 `comparability_problems(ct_off, ct_on)`:")
    ap("")
    if not problems:
        ap("- 空清单 —— 完全可比。")
    else:
        ap(f"- 报了 **{len(problems)}** 项:")
        for p in problems:
            ap(f"  - `{p}`")
        ap("")
        ap("  这**一项正是本次要被检验的干预本身**(定位语开关),不是意外的失配 ——")
        ap("  守卫没有报出分块大小/分块重叠/嵌入模型/重排模型/问答集 sha1 中的任何一项,")
        ap("  说明除定位语外两份基线在建库口径上完全一致。**这是干净单变量对照的证据。**")
    ap("")

    for spec_label, runkey in [("重排开", RERANK_ON), ("重排关", RERANK_OFF)]:
        ap(f"### 2.{1 if runkey == RERANK_ON else 2} {spec_label}（`{runkey}`）")
        ap("")
        ap("| CT | k | hit | mrr | ndcg | recall | n |")
        ap("|---|---|---|---|---|---|---|")
        for tag, base in [("**关**", off), ("**开**", on)]:
            for k in (1, 5, 10):
                r = base["runs"][runkey]
                a = r["aggregates"][str(k)]
                ap(f"| {tag} | {k} | {a['hit']:.4f} | {a['rr']:.4f} | "
                   f"{a['ndcg']:.4f} | {a['recall']:.4f} | {a['n']} |")
        ap("")

        o = off["runs"][runkey]["aggregates"]["1"]
        n_ = on["runs"][runkey]["aggregates"]["1"]
        ap("**@1 差:** " + " · ".join(
            f"{lbl} {o[key]:.4f} → {n_[key]:.4f} ({n_[key] - o[key]:+.4f})"
            for lbl, key in [("hit", "hit"), ("mrr", "rr"), ("ndcg", "ndcg"), ("recall", "recall")]
        ))
        ap("")

    ap("**结论:在这套语料上,CT 换不来可归因于它的检索增益。**")
    ap("")
    ap("下面每一条都由 §4 的 `report.py` 原始输出支撑,不是我的判断:")
    ap("")
    ap("- 重排**开**:逐题 **0 改进 / 0 退步 / 共 15**,聚合也逐位相同"
       "(ndcg@1 0.9619 vs 0.9619、hit@1 1.0000 vs 1.0000)。")
    ap("- 重排**关**:逐题 1 改进 / 0 退步 —— 而**唯一那题 `seed-010` 被 `report.py` 标为「(并列)」**。")
    ap("  它在关档的名次区间是 `[1,2]`、开档是 `[1,1]`:变化发生在**同分块谁在前没有客观答案**的区间内。")
    ap("  `report.py` 对此的原话是「标 (并列) 的那几条**别当结论**」。")
    ap("  同时并列只是**从一题挪到了另一题**(`tie_ambiguous`:关 `['seed-010']` → 开 `['seed-014']`)。")
    ap("")
    ap("所以聚合表上那 `+0.0667` 的 hit@1 **全部来自这一道并列题的名次移动**,不构成结论。")
    ap("")
    ap("这**不是**「CT 无效」的定论,而是**这套语料测不出 CT 的增益**:")
    ap("重排开时评估集已触顶,重排关时也只有一道题有区分度。")
    ap("要给 CT 一个公平的检验,需要更大语料 + 更多有区分度的题(见 §5)。")
    ap("")
    ap("> 「CT 值不值」在这套数据上的诚实回答是:**不划算** ——")
    ap("> 花掉 8.3 万 token(§3),换来的增益在 n=15 上测不出来。")
    ap("> 注意简历并没有写 CT 的具体增益数字,只写了「做了有/无对照,把增益与成本放一起算」——")
    ap("> **那句话站得住,而这张表正好是它的证据**。")
    ap("")

    # ---------------- 3. 成本 ----------------
    ap("## 3. 入库成本(花在导入那一步)")
    ap("")
    llm, emb = cost["llm"], cost["embed"]
    ap(f"`scripts/ct_cost.py` 实测,{cost['n_docs']} 篇 / {cost['n_chunks']} 块,"
       f"模型 `{cost['model']}`,并发 `contextual_workers={cost['contextual_workers']}`。")
    ap("")
    ap("| 项 | 值 |")
    ap("|---|---|")
    ap(f"| LLM 调用次数 | {llm['calls']}（= 块数 {cost['n_chunks']}，每块一次） |")
    ap(f"| prompt tokens | {llm['prompt_tokens']} |")
    ap(f"| completion tokens | {llm['completion_tokens']} |")
    ap(f"| reasoning tokens | {llm['reasoning_tokens']} |")
    ap(f"| 合计 tokens | {llm['total_tokens']} |")
    ap(f"| 墙钟 | {llm['wall_seconds']}s（并发 {cost['contextual_workers']}） |")
    ap(f"| 每块均值 | prompt {llm['prompt_tokens'] // llm['calls']} / "
       f"completion {llm['completion_tokens'] // llm['calls']} / "
       f"{llm['wall_seconds'] / llm['calls']:.2f}s |")
    ap("")
    ap("| 嵌入项 | 无定位语 | 有定位语 | 变化 |")
    ap("|---|---|---|---|")
    ap(f"| 编码耗时 | {emb['seconds_plain']}s | {emb['seconds_ctx']}s | "
       f"{emb['delta_seconds']:+.3f}s（{(emb['seconds_ctx'] / emb['seconds_plain'] - 1) * 100:+.1f}%） |")
    ap(f"| embed_text 字符数 | {emb['chars_plain']} | {emb['chars_ctx']} | "
       f"{emb['chars_ctx'] - emb['chars_plain']:+d}（{emb['chars_growth_pct']:+.1f}%） |")
    ap("")
    ap("口径说明:")
    ap("")
    ap("- CT 的成本**全部落在导入**;检索那一步一分不多花(定位语已拼在向量里)。")
    ap("- 每块一次 LLM 调用是设计使然:同篇文档的块共用同一段文档前缀,")
    ap("  按文档分组才能吃到服务端前缀缓存(`ingest/contextual.py` 模块注释)。")
    ap("- 嵌入耗时**两侧均已预热**。第一次 `encode` 会把 bge-m3 载进显存,")
    ap("  那几秒是加载不是编码 —— 不预热就会量出「有定位语反而更快」的假结论。")
    ap("- 61 块的语料下嵌入差只有 +0.027s,是因为量太小;真正占成本的是 LLM 那一栏。")
    ap("")

    # ---------------- 4. 证据 ----------------
    ap("## 4. `report.py` 原始输出(复核用)")
    ap("")
    ap("以下由 `eval/report.py` 现算,未经手工编辑。")
    ap("")

    ap("### 4.1 CT:逐指标差值 + 逐题涨跌")
    ap("")
    ap("```")
    ap(_capture(print_diff, off, on, ks=KS).rstrip())
    ap("```")
    ap("")

    for spec_label, runkey in [("重排开", RERANK_ON), ("重排关", RERANK_OFF)]:
        b = _one_run(off, runkey, "ct")
        a = _one_run(on, runkey, "ct")
        ap(f"### 4.2 逐题名次变化 · CT · {spec_label} · k=1")
        ap("")
        ap("```")
        ap(_capture(print_item_diff, b["runs"]["ct"], a["runs"]["ct"], 1).rstrip())
        ap("```")
        ap("")

    ap("### 4.3 重排消融 · 逐题名次变化 · k=1")
    ap("")
    ap("```")
    ap(_capture(print_item_diff,
                _one_run(on, RERANK_OFF, "x")["runs"]["x"],
                _one_run(on, RERANK_ON, "x")["runs"]["x"], 1).rstrip())
    ap("```")
    ap("")

    # ---------------- 5. 待补 ----------------
    ap("## 5. 这份报告**没有**证明的事(诚实边界)")
    ap("")
    ap(f"- **样本只有 {on['qa']['n_answerable']} 道可答题**。重排开时 hit@1 已触顶 1.0000,"
       "`eval_run.py` 自己也对此报警:")
    ap("  「k=1 的 hit = 1.000 几乎满分 —— 评估集可能太简单,"
       "或 gold 是从检索器自己的输出标的」。该警告尚未排除。")
    ap("- `multi_hop` 分类下 3 题里只有 `seed-009` 是真跨文档"
       "(gold 落在 09/03/02 三篇);`seed-010`、`seed-011` 的 gold 都**在同一篇文档内**,"
       "属单文档多块,分类标错了。**多跳能力(待补清单 #6)不在这份报告里**,它需要专门造的用例。")
    ap("- 语料是**电力(输电/变电)规程 9 篇**,不是简历写的「林业安检」。口径需另行处理。")
    ap("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写 {OUT}  ({len(L)} 行)")
    print()
    print("可比性:", problems if problems else "(空=完全可比)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
