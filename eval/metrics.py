"""检索指标。**纯函数,无模型、无网络、无 IO、无项目内依赖。**

指标一律自己写,不 import LlamaIndex 的
--------------------------------------
计划阶段本打算直接借 `llama_index.core.evaluation` 的 `HitRate` / `MRR` /
`Precision` / `NDCG`,读源码之后否掉了 `Precision`。它的实现是:

    retrieved_set = set(retrieved_ids); expected_set = set(expected_ids)
    precision = len(retrieved_set & expected_set) / len(retrieved_set)

**除的是「返回条数」而不是 k,而且它根本没有 k 参数。** 在
`rerank_min_keep=1` 这种保证返回非空的配置下,若阈值只放行 1 条且恰好是 gold,
`precision = 1/1 = 1.0` —— 系统正在失败,它报满分。这不是小数点上的分歧,
是方向性的:返回得越少、分越高,而这个指标本该惩罚「只返回一条」。

它还会在 `expected_ids` 为空时 `ValueError` 崩掉,不是得 0 分,所以反例根本喂不进去。

→ `precision_at_k` 除以 **k**。少返回就等于少了 k 个位置里的若干个,
这正是消费者感受到的损失(`rerank_min_keep=1` 下每个查询都只拿到 1 条,
`precision@10` 的上限就是 0.1,这个惩罚是**故意要的**)。

→ `NDCG` 也自己写:`hasattr(llama_index.core.evaluation, "NDCG")` 实测为 **False**
(它在 `...evaluation.retrieval.metrics` 子模块里),而且那个实现按 docstring 只支持
**二值**相关性 —— 分级 nDCG 它给不了。

`check_eval.py` 里仍会拿 `HitRate` / `MRR` / 二值 `NDCG` 做**交叉验证**(同输入比分数),
以换取「不是我们自己编错了」的信心。**`Precision` 明确排除在交叉验证之外**,
理由就是上面那段。这符合本项目一贯做法:分块器、JSON 抽取都是手写并写明理由的。

名次口径
--------
`ranked` 传**完整**返回列表(不预截断),`k` 由本模块截断。这样 `n_returned`
能反映真实返回条数(阈值砍掉了多少条),而非被 k 抹平。

键一律用 `(doc_id, chunk_index)`。**不要用 `dense_rank` / `sparse_rank`** ——
`hybrid.py:236-240` 里它们是**文档级**的,同一文档的所有块拿到相同 rank,
拿来做块级排序判断会得出错误结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# 块的唯一标识:(doc_id, chunk_index)。和 store 里 point_id 的一一对应关系一致。
ChunkKey = tuple[str, int]

# 可答题的指标名。顺序即报告里的打印顺序。
METRIC_FIELDS = ("hit", "rr", "ndcg", "ndcg_binary", "precision", "recall")

# 报告里用的短名。`rr` 对外的名字是 mrr —— 逐个查的倒数名次取平均就是 MRR。
METRIC_LABELS = {
    "hit": "hit",
    "rr": "mrr",
    "ndcg": "ndcg",
    "ndcg_binary": "ndcg_bin",
    "precision": "precision",
    "recall": "recall",
}

# 反例族的指标名。`leak` 是**阈值参数唯一的判别性指标**,别的指标对阈值几乎不敏感。
NEGATIVE_FIELDS = ("leak", "above_threshold_rate", "top1_is_foreign_gold", "mean_returned")


# --------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------- #


def mean(values: Sequence[float]) -> float:
    """算术平均。空序列返回 0.0。

    空返回 0.0 是有风险的(和「全错」同形),所以**调用方必须同时报 n**:
    `Aggregate.n == 0` 和 `命中率真的是 0` 是两件事。这就是 `Aggregate` 里
    `n` 不是装饰的原因。
    """
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数,和 `numpy.percentile` 默认口径一致。

    为什么不用 `statistics.quantiles`:它要求至少两个样本,且切分方式
    (exclusive/inclusive)对同一个输入给不同的 p90,`numpy` 口径是大家
    拿笔算时的预期。为什么不用 numpy:本模块要在没有 torch / 没有装科学计算栈的
    机器上跑得起来,而这里只需要 8 行。
    """
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def unique_window(ranked: Sequence[ChunkKey], k: int) -> list[ChunkKey]:
    """截断到前 k 个,并**按首次出现去重**。

    去重是必要的:融合层理论上可能把同一个块吐两次。不去重的话
    `hit@3` 会因为一条重复项被挤掉一个真名次而虚低,
    `first_hit_rank` 也会指向一个「其实没有第 3 个位置」的位置。

    先截断再取重,所以窗口长度可能 < k —— 这正是对的:重复项提供不了新信息,
    它不该顶替一个本该被看到的位置。
    """
    if k <= 0:
        # 静默返回空窗口会让 k=0 看起来像「全错」而不是调用方写错了。
        raise ValueError(f"k 必须 >= 1,实得 {k}")
    out: list[ChunkKey] = []
    seen: set[ChunkKey] = set()
    for key in ranked[:k]:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def first_hit_rank(window: Sequence[ChunkKey], gold: Iterable[ChunkKey]) -> int | None:
    """第一个命中的名次(1 起)。没命中返回 None(而不是 0)。

    `None` 和 `0` 的区别很实在:`0` 参与 `1/0` 会炸,参与排序会排到最前面。
    让它无法被误用,比写注释提醒靠得住。
    """
    gs = set(gold)
    for i, key in enumerate(window, 1):
        if key in gs:
            return i
    return None


# --------------------------------------------------------------------- #
# 单项指标(均为「给定窗口」的纯函数)
# --------------------------------------------------------------------- #


def hit_at_k(ranked: Sequence[ChunkKey], gold: Iterable[ChunkKey], k: int) -> float:
    """窗口里有没有 gold。有→1.0,无→0.0。"""
    return 1.0 if first_hit_rank(unique_window(ranked, k), gold) is not None else 0.0


def reciprocal_rank_at_k(ranked: Sequence[ChunkKey], gold: Iterable[ChunkKey], k: int) -> float:
    """第一个 gold 的倒数名次,没有则 0.0。逐题平均即 MRR。"""
    r = first_hit_rank(unique_window(ranked, k), gold)
    return 1.0 / r if r else 0.0


def precision_at_k(ranked: Sequence[ChunkKey], gold: Iterable[ChunkKey], k: int) -> float:
    """**除以 k**,不是除以返回条数 —— 这是与 LlamaIndex 分道扬镳的地方。

    见模块头:除以返回条数会在「只返回 1 条且恰好正确」时报 1.0,
    而那正是我们要判为最差的形态。
    """
    window = unique_window(ranked, k)
    gs = set(gold)
    return sum(1 for key in window if key in gs) / k


def recall_at_k(ranked: Sequence[ChunkKey], gold: Iterable[ChunkKey], k: int) -> float:
    """命中的 gold 占全部 gold 的比例。多 gold 题给部分分。

    `gold` 为空(反例)时返回 0.0 —— 反例永远不该走到这里,
    所以这更像是防御而不是语义。反例走 `score_negative`。
    """
    gs = set(gold)
    if not gs:
        return 0.0
    window = unique_window(ranked, k)
    return len({key for key in window if key in gs}) / len(gs)


def ndcg_at_k(
    ranked: Sequence[ChunkKey],
    gold_grades: Mapping[ChunkKey, int],
    k: int,
    *,
    binary: bool = False,
) -> float:
    """nDCG@k。`binary=True` 时把所有相关度压成 1,与 LlamaIndex 的口径对齐以便交叉验证。

    分级用标准的 `gain = 2^g - 1`、`discount = log2(i + 2)`(`i` 从 0 起,
    即第 1 名除以 `log2(2)=1` 不衰减)。

    IDCG 用**金标自己的理想排序**并按 k 截断 —— 不截断的话,一道有 8 个 gold
    而只能看到 10 个位置的题,IDCG 里塞进了看不到的位置,分母虚高、分数虚低。
    """
    if not gold_grades:
        return 0.0

    window = unique_window(ranked, k)

    def gain(g: int) -> int:
        if binary:
            return 1 if g >= 1 else 0
        return (1 << g) - 1  # 2^g - 1

    dcg = 0.0
    for i, key in enumerate(window):
        g = gold_grades.get(key, 0)
        if g:
            dcg += gain(g) / math.log2(i + 2)

    ideal_grades = sorted((1 if binary else g for g in gold_grades.values()), reverse=True)[:k]
    idcg = sum(gain(g) / math.log2(i + 2) for i, g in enumerate(ideal_grades))

    # idcg == 0 只在「所有 grade 都是 0」时发生,而 schema 校验器不允许
    # 列在 expected 里的 gold 是 grade 0。走到这里说明调用方绕过了校验,
    # 返回 0.0 而不是抛异常,是为了让评估本身别因为一条坏数据整场跑不完。
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------- #
# 逐题评分
# --------------------------------------------------------------------- #


@dataclass
class QueryScore:
    """一道**可答题**在某个 k 上的得分。"""

    item_id: str = ""
    type: str = ""
    k: int = 0

    hit: float = 0.0
    rr: float = 0.0
    ndcg: float = 0.0
    ndcg_binary: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    first_rank: int | None = None   # 1 起;None = 没命中
    n_found: int = 0                # 窗口内命中的 gold 数
    n_gold: int = 0                 # 本题 gold 总数
    n_returned: int = 0             # **完整**返回条数(未被 k 截断),反映阈值砍了多少

    def value(self, field_name: str) -> float:
        return float(getattr(self, field_name))


def score_query(
    ranked: Sequence[ChunkKey],
    gold_grades: Mapping[ChunkKey, int],
    k: int,
    *,
    item_id: str = "",
    type: str = "",
) -> QueryScore:
    """算一道可答题的全部指标。

    `gold_grades`: `{key: grade}`。只关心「是否相关」的指标会把 grade 二值化。
    `ranked`: **完整**返回列表,不预截断。
    """
    if not gold_grades:
        raise ValueError(
            f"{item_id or '?'}: score_query 只接可答题;"
            "expected 为空的题目是反例,走 score_negative"
        )
    window = unique_window(ranked, k)
    gold_keys = set(gold_grades)
    found = [key for key in window if key in gold_keys]
    rank = first_hit_rank(window, gold_keys)

    return QueryScore(
        item_id=item_id,
        type=type,
        k=k,
        hit=1.0 if rank is not None else 0.0,
        rr=1.0 / rank if rank else 0.0,
        ndcg=ndcg_at_k(ranked, gold_grades, k),
        ndcg_binary=ndcg_at_k(ranked, gold_grades, k, binary=True),
        precision=len(found) / k,
        recall=len(found) / len(gold_keys),
        first_rank=rank,
        n_found=len(found),
        n_gold=len(gold_keys),
        n_returned=len(ranked),
    )


# --------------------------------------------------------------------- #
# 反例族
# --------------------------------------------------------------------- #


@dataclass
class NegativeScore:
    """一道**反例**的得分。指标名和可答题刻意不重叠,免得被误并进平均。"""

    item_id: str = ""
    type: str = "negative"

    top1_key: ChunkKey | None = None
    top1_score: float = 0.0         # 无返回时 0.0
    leak: float = 0.0               # top-1 分 ≥ 阈值(系统**自称**相关了)
    above_threshold_rate: float = 0.0   # 返回里超阈值的比例
    top1_is_foreign_gold: float = 0.0   # top-1 是**别的题**的 gold

    n_returned: int = 0
    threshold: float = 0.0          # 记下当时阈值 —— 分数离开阈值就没有意义

    # 指标名与字段名在这里有一处刻意的错位:`mean_returned` 是 `n_returned`
    # 的**宏平均**,报告和基线里该叫 mean_returned(它确实是个均值,
    # 而 n_returned 单看会被读成「总数」)。对外用指标名、对内用字段名,
    # 在这里显式对上,不要去改 `n_returned` 的名字 ——
    # 逐题明细里它必须还是「这一条返回了几条」。
    _ALIAS = {"mean_returned": "n_returned"}

    def value(self, field_name: str) -> float:
        return float(getattr(self, self._ALIAS.get(field_name, field_name)))


def score_negative(
    ranked_scores: Sequence[tuple[ChunkKey, float]],
    threshold: float,
    *,
    foreign_golds: Iterable[ChunkKey] = (),
    item_id: str = "",
    type: str = "negative",
) -> NegativeScore:
    """算一道反例的分数分布特征。

    `ranked_scores` 是**带分数**的完整返回:重排开着时给重排分,
    关掉时给 RRF 分。用哪个由 runner 决定并记进基线 —— 本模块只管算,
    不猜。**阈值必须和这批分数同源**,否则 `leak` 是无意义的数字。

    反例的价值全在「分数分布」上:`expected == []` 会让它们给任何
    「命中率」类平均贡献恒定的 0,既稀释了均值,又对参数变化完全不可见。
    所以它们单独成族,评的是「会不会硬凑」。
    """
    seen: set[ChunkKey] = set()
    uniq: list[tuple[ChunkKey, float]] = []
    for key, sc in ranked_scores:
        if key in seen:
            continue
        seen.add(key)
        uniq.append((key, sc))

    above = sum(1 for _, sc in uniq if sc >= threshold)
    top1_key, top1_score = uniq[0] if uniq else (None, 0.0)
    foreign = set(foreign_golds)

    return NegativeScore(
        item_id=item_id,
        type=type,
        top1_key=top1_key,
        top1_score=top1_score,
        leak=1.0 if (top1_key is not None and top1_score >= threshold) else 0.0,
        above_threshold_rate=above / len(uniq) if uniq else 0.0,
        top1_is_foreign_gold=1.0 if (top1_key is not None and top1_key in foreign) else 0.0,
        n_returned=len(uniq),
        threshold=threshold,
    )


# --------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------- #


@dataclass
class Aggregate:
    """一组得分的宏平均。

    **宏平均(逐题平均),不是 micro。** 多 gold 题上两者不同:micro 会把
    「有 8 个 gold 的题」的权重放大 8 倍。我们关心的是「一个典型查询体验如何」,
    所以每题一票。报告里必须写明这一点,否则读数字的人会拿它跟别人
    micro 口径的公告值比。
    """

    n: int = 0
    values: dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.n == 0

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)

    def as_dict(self) -> dict[str, float]:
        return {"n": float(self.n), **self.values}


def aggregate(scores: Sequence[QueryScore]) -> Aggregate:
    """可答题的宏平均。"""
    return Aggregate(
        n=len(scores),
        values={f: mean([s.value(f) for s in scores]) for f in METRIC_FIELDS},
    )


def aggregate_by_type(scores: Sequence[QueryScore]) -> dict[str, Aggregate]:
    """按题型分组。

    这是**最高价值的切面** —— 整个问题就是「哪一类检索引擎动到了」。
    但报告必须同时打分组内的 n:3 题一类的均值是噪声,
    把它和 40 题一类的均值并排打而不标 n,是最容易让人过度解读的排版错误。
    """
    groups: dict[str, list[QueryScore]] = {}
    for s in scores:
        groups.setdefault(s.type or "unknown", []).append(s)
    return {t: aggregate(g) for t, g in groups.items()}


def aggregate_negatives(scores: Sequence[NegativeScore]) -> Aggregate:
    """反例族的宏平均 + top1 分位数。

    分位数是必要的:`top1_score_mean` 会被几条极高分拉走,
    而真正决定「用户会不会被误导」的是**高分那一端**,所以 p90 才是要看的那一列。
    """
    vals = {f: mean([s.value(f) for s in scores]) for f in NEGATIVE_FIELDS}
    top1 = [s.top1_score for s in scores]
    vals["top1_score_mean"] = mean(top1)
    vals["top1_score_p90"] = percentile(top1, 90)
    return Aggregate(n=len(scores), values=vals)


# --------------------------------------------------------------------- #
# 对比
# --------------------------------------------------------------------- #


def diff_aggregates(before: Aggregate, after: Aggregate, *, fields: Sequence[str] = ()) -> dict[str, tuple[float, float, float]]:
    """`{指标: (前, 后, 增量)}`。字段取两者并集,以便跨族对比也能用。"""
    names = list(fields) or sorted(set(before.values) | set(after.values))
    return {
        f: (before.get(f), after.get(f), after.get(f) - before.get(f))
        for f in names
    }


def diff_items(before: Sequence[QueryScore], after: Sequence[QueryScore], *, field_name: str = "hit") -> dict[str, list[str]]:
    """逐题对比,分成 `improved / regressed / unchanged`。

    聚合的差值会掩盖「修好 3 题、弄坏 2 题」;逐题列表才是人能顺着定位的入口。
    只比较**两边都出现**的题:新增/删除的题不是涨跌,混进来会污染判读。
    """
    b = {s.item_id: s for s in before}
    a = {s.item_id: s for s in after}
    out: dict[str, list[str]] = {"improved": [], "regressed": [], "unchanged": []}
    for item_id in sorted(set(b) & set(a)):
        bv, av = b[item_id].value(field_name), a[item_id].value(field_name)
        if av > bv:
            out["improved"].append(item_id)
        elif av < bv:
            out["regressed"].append(item_id)
        else:
            out["unchanged"].append(item_id)
    return out


def same_aggregate(a: Aggregate, b: Aggregate) -> bool:
    """两份聚合是否**逐位相同**。用于「配置没生效」的自检断言。

    这里用精确相等而不是容差,是因为要判的是**同一段代码对同一份输入**是否
    给出了逐位相同的结果 —— 浮点运算在确定性输入上是确定性的。
    容差会把这个性质测没了:真跑过两次的微小差异与「复用上次结果」的区别,
    恰恰在低位比特里。

    注意它**不是**通用的「no-op 检测器」:换一个对该语料根本不起作用的参数,
    结果本来就该逐位相同。断言里要挑一个保证会动的参数。
    """
    return a.n == b.n and a.values == b.values


__all__ = [
    "ChunkKey",
    "METRIC_FIELDS",
    "METRIC_LABELS",
    "NEGATIVE_FIELDS",
    "mean",
    "percentile",
    "unique_window",
    "first_hit_rank",
    "hit_at_k",
    "reciprocal_rank_at_k",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "QueryScore",
    "score_query",
    "NegativeScore",
    "score_negative",
    "Aggregate",
    "aggregate",
    "aggregate_by_type",
    "aggregate_negatives",
    "diff_aggregates",
    "diff_items",
    "same_aggregate",
]
