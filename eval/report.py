"""表格打印与基线读写。

这一层同时拥有**结果记录的形状**(`ItemDetail` / `RunResult` / 基线 JSON):
runner 负责生产它们,本模块负责把它们变成文字或落盘。把形状放在这里而不是
runner 里,是为了让 `runner → report` 单向依赖,不出现循环。

## 基线里为什么必须存逐题明细

聚合的差值会掩盖「修好 3 题、弄坏 2 题」—— 两边的 `hit@5` 一样,系统已经不是
同一个系统了。逐题列表才是人能顺着往下查的入口,所以基线里存它,而不只是一个数。

## 基线比对为什么必须拒绝跨来源

`qa.sha1` / `chunk_size` / 定位语开关 / `embed_model` / `rerank_model` 任一不同,
两份数字就没有可比性。而这是**必然会发生的**失败:人改了一条问题、重跑、
然后得出「检索器变了」的结论。`qa.sha1` 存在的全部理由就是拦住这一步。

**注意 `retrieval` 配置不在这个清单里** —— 它恰恰是**该**变的:改 `rrf_k`、
改阈值本来就是基线比对的内容。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from config import BASE_DIR
from eval.metrics import (
    METRIC_FIELDS,
    METRIC_LABELS,
    NEGATIVE_FIELDS,
    Aggregate,
    QueryScore,
)

__all__ = [
    "BASELINE_DIR",
    "SCHEMA_VERSION",
    "ItemDetail",
    "NegativeDetail",
    "RunResult",
    "baseline_dir",
    "resolve_baseline_path",
    "build_baseline",
    "save_baseline",
    "load_baseline",
    "comparability_problems",
    "print_context",
    "print_run",
    "print_by_type",
    "print_negatives",
    "print_diff",
    "print_misses",
    "dump_json",
]

BASELINE_DIR = BASE_DIR / "eval" / "baselines"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------- #
# 结果记录形状
# --------------------------------------------------------------------- #


@dataclass
class ItemDetail:
    """一道可答题的明细。存进基线,供人定位具体退步。

    **刻意不含 k** —— 名次与返回条数都不随 k 变,k 只改窗口大小。
    把 k 变成字段会逼出「同一题存 N 份」,而基线文件是拿来看的,不是拿来膨胀的。
    """

    item_id: str
    type: str
    question: str
    rank: int | None  # 第一个 gold 的名次,None = 没命中
    n_returned: int  # 阈值之后真实返回了几条(未被 k 截断)
    n_gold: int
    # rank 的**并列区间**:同分的块可以任意排,所以真名次落在 [lo, hi] 内。
    # lo != hi ⟹ 这一题的名次是并列对里谁在前决定的,单题的名次变化不算结论。
    rank_lo: int | None = None
    rank_hi: int | None = None
    # 阈值砍掉了几条。>0 且本题 miss 是「排到了但被砍」,和「根本没召回」
    # 是两种完全不同的病,调的药也不同。
    dropped: int = 0
    # [(gold 标签 "文件#块", 该 gold 在返回表里的名次(1 起,0=没召回))]
    golds: list[tuple[str, int]] = field(default_factory=list)
    top1: str | None = None  # 首位的标签
    top1_score: float = 0.0
    # 标签 → "dense" / "sparse" / "both" / "-"(两路都没召回它所属的文档)。
    # **文档级**:见 runner 模块头,`dense_order` 是 doc_id 列表。
    doc_source: dict[str, str] = field(default_factory=dict)
    # 返回的块里,有几条是**图那一路**多召回的(hybrid 档恒为 0)。
    #
    # 这个字段存在的唯一目的,是把「图没接上」和「图接上了但没用」分开:
    # 两者在聚合分上完全一样(都是没变化),而前者是配置事故,必须能被指认。
    # 没有它,一份「graphrag 分数 = hybrid 分数」的报告什么也证明不了。
    graph_hits: int = 0

    @property
    def missed(self) -> bool:
        return self.rank is None


@dataclass
class NegativeDetail:
    """一道反例在某个 k 上的明细。"""

    item_id: str
    question: str
    top1: str | None
    top1_score: float
    leak: float
    n_returned: int
    foreign_gold: float
    threshold: float


@dataclass
class RunResult:
    """一个配置变体的完整结果。"""

    spec: str
    use_rerank: bool
    retrieval: dict[str, Any]
    effective_k: int
    ks: list[int]
    #: 哪个检索后端跑出来的(`hybrid` / `graphrag`)。进基线是为了让 JSON 自解释 ——
    #: 否则读到 `graphrag:no-threshold+wide` 的人只能靠 run 名字去猜。
    retriever: str = "hybrid"
    # str(k) → 聚合值。键转字符串是为了直接进 JSON。
    aggregates: dict[str, dict[str, float]] = field(default_factory=dict)
    by_type: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    negatives: dict[str, dict[str, float]] = field(default_factory=dict)
    # 逐题明细**不按 k 分组**:明细里没有一个字段依赖 k(见 ItemDetail)
    per_item: list[ItemDetail] = field(default_factory=list)
    negative_items: list[NegativeDetail] = field(default_factory=list)
    # 诊断项,不是成绩:见 print_run 里的说明。
    nonempty_rate: dict[str, float] = field(default_factory=dict)
    n_answerable: int = 0
    n_negative: int = 0
    # 名次落在并列区间里的题(rank_lo != rank_hi)。这些题的数字由并列定序决定,
    # 逐题对比时要打折看 —— 见 `_stable_hits` / `_tie_band`。
    tie_ambiguous: list[str] = field(default_factory=list)

    def agg(self, k: int) -> Aggregate:
        d = self.aggregates.get(str(k), {})
        return Aggregate(n=int(d.get("n", 0)), values={x: v for x, v in d.items() if x != "n"})

    def nagg(self, k: int) -> Aggregate:
        d = self.negatives.get(str(k), {})
        return Aggregate(n=int(d.get("n", 0)), values={x: v for x, v in d.items() if x != "n"})


# --------------------------------------------------------------------- #
# 基线读写
# --------------------------------------------------------------------- #


def baseline_dir() -> Path:
    return BASELINE_DIR


def resolve_baseline_path(name_or_path: str) -> Path:
    """`--save-baseline before` / `--baseline before` → 具体文件路径。

    带目录分隔符或 `.json` 后缀的当路径;否则当名字,落到 `eval/baselines/<名字>.json`。
    """
    p = Path(name_or_path)
    if p.suffix == ".json" or p.parent != Path("."):
        return p if p.is_absolute() else (BASE_DIR / p)
    return BASELINE_DIR / f"{name_or_path}.json"


def build_baseline(
    *,
    corpus: Mapping[str, Any],
    qa: Mapping[str, Any],
    ingest: Mapping[str, Any],
    collection: str,
    ks: Sequence[int],
    runs: Mapping[str, RunResult],
) -> dict[str, Any]:
    """组装基线 JSON。

    `created_at` 用本地时间:这份文件的读者是人,不是机器,时区标记比 UTC 好读。
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": dict(corpus),
        "qa": dict(qa),
        "ingest": dict(ingest),
        "collection": collection,
        "ks": list(ks),
        "runs": {
            name: {
                "spec": r.spec,
                "retriever": r.retriever,
                "use_rerank": r.use_rerank,
                "retrieval": r.retrieval,
                "effective_k": r.effective_k,
                "n_answerable": r.n_answerable,
                "n_negative": r.n_negative,
                "nonempty_rate": r.nonempty_rate,
                # 并列区间里的题。存进基线是为了让**将来**的逐题 diff 能看见
                # 「这条变化发生在一次抛硬币上」,而不是记成真实退步。
                "tie_ambiguous": r.tie_ambiguous,
                "aggregates": r.aggregates,
                "by_type": r.by_type,
                "negatives": r.negatives,
                "per_item": [asdict(d) for d in r.per_item],
                "negative_items": [asdict(d) for d in r.negative_items],
            }
            for name, r in runs.items()
        },
    }


def save_baseline(payload: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return p


def load_baseline(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"基线文件不存在:{p}")
    return json.loads(p.read_text(encoding="utf-8"))


#: 比对可比性时必须逐字相同的字段。路径 → 人话名字。
#: **故意不含 `retrieval`** —— 改检索参数正是比对要做的事。
_COMPARABLE_FIELDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("qa", "sha1"), "问答集 sha1"),
    (("ingest", "chunk_size"), "分块大小"),
    (("ingest", "chunk_overlap"), "分块重叠"),
    (("ingest", "contextual_enabled"), "定位语开关"),
    (("ingest", "embed_model"), "嵌入模型"),
    (("ingest", "rerank_model"), "重排模型"),
)


def _dig(d: Mapping[str, Any] | None, path: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def comparability_problems(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    """两份基线能不能比。返回**具体哪个字段不同**的清单;空清单 = 能比。

    只说「不可比」而不说哪里不同,等于把排查丢回给用户 ——
    而这个检查的全部价值就是省掉那次排查。
    """
    problems: list[str] = []

    b_ver, a_ver = before.get("schema_version"), after.get("schema_version")
    if b_ver != a_ver:
        problems.append(f"基线格式版本不同(前 {b_ver} / 后 {a_ver})")

    for path, label in _COMPARABLE_FIELDS:
        bv, av = _dig(before, path), _dig(after, path)
        if bv != av:
            problems.append(f"{label}不同:前 = {bv!r},后 = {av!r}")

    bks, aks = before.get("ks") or [], after.get("ks") or []
    if list(bks) != list(aks):
        problems.append(f"k 取值不同:前 = {bks},后 = {aks}")

    bspec, aspec = sorted((before.get("runs") or {})), sorted((after.get("runs") or {}))
    if bspec != aspec:
        problems.append(f"配置变体不同:前 = {bspec},后 = {aspec}")

    return problems


def dump_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# --------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------- #

_RULE = "─" * 74


def _num(x: float, width: int = 7) -> str:
    return f"{x:>{width}.3f}"


def print_context(
    *,
    corpus_root: Path,
    n_docs: int,
    n_chunks: int,
    qa_path: Path,
    qa_sha1: str,
    n_kept: int,
    n_answerable: int,
    n_negative: int,
    collection: str,
    contextual: bool,
    ks: Sequence[int],
    top_k: int,
) -> None:
    """开跑前把「这次测的是什么」打全。

    这些行不是装饰:一份分数离开语料、问答集版本、定位语开关就没有意义,
    而事后要重建这个上下文,成本远高于现在打这几行。
    """
    print(_RULE)
    print(f"语料    {corpus_root}")
    print(f"        {n_docs} 篇 / {n_chunks} 块   定位语 = {'开' if contextual else '关'}")
    print(f"问答集  {qa_path}")
    print(
        f"        sha1 {qa_sha1}   keep {n_kept} 条"
        f"(可答 {n_answerable} / 反例 {n_negative})"
    )
    print(f"库      {collection}")
    print(f"k       {','.join(str(k) for k in ks)}   top_k={top_k}")
    print(_RULE)


def print_run(run: RunResult, *, ks: Sequence[int], show_binary: bool = True) -> None:
    """打一个配置变体的主表。"""
    r = run.retrieval
    print()
    print(f"◆ {run.spec}")
    # 关重排时那两个阈值**根本不参与**(`retrieve` 走的是 `results[:top_k]`),
    # 照原样打印会让人以为「阈值 0.05 是生效的」,进而拿它解释漏召回。
    if run.use_rerank:
        thr = (
            f"   阈值 rerank_min_score={r.get('rerank_min_score')}"
            f" / min_keep={r.get('rerank_min_keep')}"
        )
    else:
        thr = "   阈值 **未生效**(重排关时 rerank_min_score / min_keep 均不参与)"
    print(f"   重排 = {'开' if run.use_rerank else '关'}{thr}")
    print(
        f"   fusion_top_k={r.get('fusion_top_k')}  dense_top_k={r.get('dense_top_k')}"
        f"  sparse_top_k={r.get('sparse_top_k')}  rrf_k={r.get('rrf_k')}"
        f"   → effective_k = {run.effective_k}"
    )
    if run.n_answerable == 0:
        print("   (没有可答题,跳过指标)")
        return

    head = "   k    n   " + "".join(f"{METRIC_LABELS[f]:>9}" for f in METRIC_FIELDS)
    if show_binary:
        head += f"{'非空率':>9}"
    print(head)
    for k in ks:
        a = run.agg(k)
        if a.is_empty:
            continue
        row = f"  {k:>3}  {a.n:>3}   "
        row += "".join(f"{_num(a.get(f)):>9}" for f in METRIC_FIELDS)
        if show_binary:
            row += f"{_num(run.nonempty_rate.get(str(k), 0.0)):>9}"
        print(row)

    print()
    if run.use_rerank:
        print(
            "   非空率是**诊断项,不是成绩**:rerank_min_keep="
            f"{r.get('rerank_min_keep')} 把它钉在 1.000 附近是设计使然,"
            "别读成「检索质量好」。"
        )
    else:
        print(
            "   非空率是**诊断项,不是成绩**:本配置无阈值,1.000 只说明"
            "「top_k 窗口被填满了」,与质量无关。"
        )
    print("   宏平均(每题一票),不是 micro —— 多 gold 题不放大权重。")
    n_amb = len(run.tie_ambiguous)
    if n_amb:
        shown = ",".join(run.tie_ambiguous[:8]) + ("…" if n_amb > 8 else "")
        print(
            f"   ⚠ {n_amb}/{run.n_answerable} 题的**首个 gold 落在并列区间内**({shown})"
            " —— 同分的块谁在前没有客观答案,名次已按 doc_id 定序以求可复现,"
            "但**这些题的名次变化不构成结论**。"
        )


def print_by_type(run: RunResult, ks: Sequence[int], *, min_n: int = 1) -> None:
    """按题型分组。这是最高价值的切面:哪一类引擎动到了。

    逐个 k 打完整表会得到 N 张几乎重复的表(4 个 k 就是 4 张),
    所以压成两段:**跨 k 的 hit 网格**(看一类题要多深才捞得到)
    + **单一 k 的完整指标明细**。分类别完整数据都在基线 JSON 里,不缺。
    """
    grids = {k: run.by_type.get(str(k)) for k in sorted(ks)}
    grids = {k: g for k, g in grids.items() if g}
    if not grids:
        return
    types = sorted({t for g in grids.values() for t in g})
    if not types:
        return

    def _n_of(t: str) -> int:
        return max(int(grids[k][t].get("n", 0)) for k in grids if t in grids[k])

    print()
    print("   按题型 (hit)")
    print(
        "     " + f"{'type':<12}{'n':>4}"
        + "".join(f"{'@' + str(k):>8}" for k in grids)
    )
    for t in types:
        cells = "".join(
            f"{_num(grids[k][t].get('hit')):>8}" if t in grids[k] else f"{'-':>8}"
            for k in grids
        )
        print(f"     {t:<12}{_n_of(t):>4}{cells}")
    small = [t for t in types if _n_of(t) < 3]
    if small:
        print(f"     ⚠ {','.join(small)} 的 n < 3 —— 均值是噪声,别拿它下结论")

    # 完整明细只打在「用户今天实际拿到的量」那个 k 上(等于出厂的 rerank_top_n)
    detail_k = 5 if 5 in grids else max(grids)
    print()
    print(f"   按题型明细 (k={detail_k})")
    print(
        "     " + f"{'type':<12}{'n':>4}"
        + "".join(f"{METRIC_LABELS[f]:>9}" for f in METRIC_FIELDS)
    )
    for t in types:
        a = grids[detail_k][t]
        n = int(a.get("n", 0))
        if n < min_n:
            continue
        print(
            f"     {t:<12}{n:>4}"
            + "".join(f"{_num(a.get(f)):>9}" for f in METRIC_FIELDS)
        )


def print_negatives(run: RunResult, k: int) -> None:
    """反例族单独打,**不进任何平均**。

    没有阈值时 `leak` / `above_threshold_rate` / `mean_returned`
    是**结构性常数**(只要返回非空就恒为 1.000 / top_k),报出来会被读成
    「系统乱了」,而其实只说明「没设阈值」。所以关重排时这三项打 n/a,
    只留有意义的分数底与外来 gold 比例。
    """
    a = run.nagg(k)
    if a.is_empty:
        return
    print()
    if run.use_rerank:
        print(f"   反例族 (k={k})  n={a.n}   阈值 = {a.get('_threshold', 0.0):.3f}")
        print(
            f"     leak@k            {_num(a.get('leak'))}"
            "   ← 越低越好;阈值参数**唯一**的判别性指标"
        )
        print(f"     above_threshold   {_num(a.get('above_threshold_rate'))}")
        print(
            f"     mean_returned     {_num(a.get('mean_returned'))}"
            "   ← min_keep 钉住的底,不是好成绩"
        )
        score_note = "重排分"
    else:
        print(f"   反例族 (k={k})  n={a.n}   **本配置无阈值**(重排关)")
        print(
            "     leak / above_threshold / mean_returned 在本配置下是结构性常数,"
            "不报 —— 要看它们请开重排"
        )
        score_note = "RRF 分,只在同一配置内可比"
    print(
        f"     top1_score        mean {_num(a.get('top1_score_mean'))}"
        f"   p90 {_num(a.get('top1_score_p90'))}   ← 噪声底({score_note})"
    )
    print(
        f"     top1_is_foreign_gold {_num(a.get('top1_is_foreign_gold'))}"
        "   ← top-1 是**别的题**的 gold,领域覆盖过宽的具体证据"
    )


def print_misses(run: RunResult, k: int, *, limit: int = 15) -> None:
    """打没命中的题,附「排到了但被阈值砍掉」的区分。

    `k` 只影响「哪些题算 miss」这一个判断(`rank > k` 也算 miss),
    明细本身与 k 无关 —— 所以基线里只需要存一份。
    """
    items = [d for d in run.per_item if d.rank is None or d.rank > k]
    if not items:
        print(f"\n   k={k}:全部命中。")
        return
    print(f"\n   k={k} 未命中 {len(items)}/{run.n_answerable} 题:")
    for d in items[:limit]:
        print(f"     {d.item_id:<10} {d.type:<11} 返回 {d.n_returned} 条")
        print(f"        {d.question}")
        if d.golds:
            where = "、".join(f"{lbl} 第{r}名" if r else f"{lbl} 未召回" for lbl, r in d.golds)
            print(f"        gold: {where}")
        if d.dropped:
            print(
                f"        ⚠ 阈值另砍掉 {d.dropped} 条 —— 与「根本没召回」是两种病"
            )
        if d.top1:
            print(f"        首位: {d.top1}  ({d.top1_score:.3f})")
    if len(items) > limit:
        print(f"     …另有 {len(items) - limit} 题")


def print_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    ks: Sequence[int],
    fields: Sequence[str] = METRIC_FIELDS,
    negative_fields: Sequence[str] = NEGATIVE_FIELDS,
) -> None:
    """逐变体、逐 k 打差值,并把逐题涨跌单独列出来。"""
    b_runs = before.get("runs") or {}
    a_runs = after.get("runs") or {}

    for spec in sorted(set(b_runs) & set(a_runs)):
        b_run, a_run = b_runs[spec], a_runs[spec]
        print()
        print(_RULE)
        print(f"对比 {before.get('created_at', '?')} → {after.get('created_at', '?')}   ◆ {spec}")
        print(_RULE)

        for k in ks:
            bk, ak = str(k), str(k)
            b_agg = (b_run.get("aggregates") or {}).get(bk)
            a_agg = (a_run.get("aggregates") or {}).get(ak)
            if not b_agg or not a_agg:
                continue
            print()
            print(f"   k={k}   可答题 n={int(a_agg.get('n', 0))}")
            print("     指标        前        后        增量")
            for f in fields:
                bv, av = float(b_agg.get(f, 0.0)), float(a_agg.get(f, 0.0))
                d = av - bv
                mark = " " if abs(d) < 1e-9 else ("↑" if d > 0 else "↓")
                print(f"     {METRIC_LABELS[f]:<10} {bv:>7.3f}   {av:>7.3f}   {d:>+8.3f} {mark}")

            b_neg = (b_run.get("negatives") or {}).get(bk)
            a_neg = (a_run.get("negatives") or {}).get(ak)
            if b_neg and a_neg:
                print()
                print("     反例族")
                for f in negative_fields + ("top1_score_mean", "top1_score_p90"):
                    bv, av = float(b_neg.get(f, 0.0)), float(a_neg.get(f, 0.0))
                    d = av - bv
                    mark = " " if abs(d) < 1e-9 else ("↑" if d > 0 else "↓")
                    print(f"     {f:<16} {bv:>7.3f}   {av:>7.3f}   {d:>+8.3f} {mark}")

            print_item_diff(b_run, a_run, k)


def print_item_diff(b_run: Mapping[str, Any], a_run: Mapping[str, Any], k: int) -> None:
    """「修好 3 题、弄坏 2 题」—— 聚合看不见的那一层。

    用的是**该 k 下的名次**而不是 k 无关的命中与否:第 8 名挪到第 2 名是真实进步,
    而 `hit@10` 两边都是 1,聚合完全看不见它。
    """
    b_items = {d["item_id"]: d for d in (b_run.get("per_item") or [])}
    a_items = {d["item_id"]: d for d in (a_run.get("per_item") or [])}
    both = sorted(set(b_items) & set(a_items))
    if not both:
        return

    def rr(d: Mapping[str, Any]) -> float:
        r = d.get("rank")
        return 1.0 / r if isinstance(r, int) and 0 < r <= k else 0.0

    # 两份基线里落在并列区间的题。这些题的名次由同分的块谁在前决定,
    # 涨跌可能只是抛硬币,所以在这里标出来而不是算进「真实涨跌」。
    tied = set(b_run.get("tie_ambiguous") or []) | set(a_run.get("tie_ambiguous") or [])

    improved, regressed = [], []
    for iid in both:
        bv, av = rr(b_items[iid]), rr(a_items[iid])
        if av > bv:
            improved.append(iid)
        elif av < bv:
            regressed.append(iid)

    def _mark(ids: list[str]) -> str:
        return ", ".join(f"{i}(并列)" if i in tied else i for i in ids)

    print()
    print(f"     逐题 (k={k},按名次):改进 {len(improved)} / 退步 {len(regressed)} / 共 {len(both)}")
    if improved:
        print(f"       改进: {_mark(improved)}")
    if regressed:
        print(f"       退步: {_mark(regressed)}")
    if not improved and not regressed:
        print("       (逐题无变化 —— 若聚合也没动,先怀疑配置没生效)")
    if tied:
        moved = (set(improved) | set(regressed)) & tied
        note = f",其中 {', '.join(sorted(moved))} 的涨跌发生在并列区间内" if moved else ""
        print(
            f"       ⚠ 本次有 {len(tied)} 题名次落在并列区间(同分块谁在前无客观答案){note}"
            " —— 标 (并列) 的那几条别当结论"
        )
