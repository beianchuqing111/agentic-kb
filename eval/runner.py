"""真导入、真检索、算分、比基线。

`scripts/eval_run.py` 是薄壳,真正的顺序在这里。

## 顺序为什么是这个顺序

1. **`get_settings()`** —— 一切参数的出处
2. **定语料与问答集** —— 找不到就说清怎么造,不要跑到一半才炸
3. **加载 + 校验问答集** —— `keep` 为 0 直接中止(空集会打印出一张全 0 的表,看着像灾难)
4. **锚点解析与前置校验,在任何查询之前** —— 锚点失效时**拒绝出分**。理由见
   `corpus.py` 模块头:锚点位移会被误读成检索退步,让人去调一个没坏的参数
5. **导入** —— collection 名里带定位语开关与语料指纹
6. **导入后复验** —— 磁盘上有不等于库里有
7. **逐题检索、算分、聚合**
8. **可选存/比基线**

## 三个必须显式传参的地方

- **`top_k` 必须显式传。** `retrieve()` 里 `top_k = top_k or cfg.rerank_top_n`,
  默认 5 —— 不传的话 `hit@10` 是幻觉数。本模块每次都显式传。
- **k 有天花板**:`effective_k = min(top_k, fusion_top_k, dense_top_k + sparse_top_k)`,
  因为融合结果先被 `limit=fusion_top_k` 收口(`hybrid.py` 的 `query_hybrid` 调用),
  重排只能在这批候选里排。超过天花板的 k **从报告里删掉**而不是打成 0 ——
  打成 0 会把「这个 k 测不了」伪装成「这个 k 全错」。
- **不用 `get_retriever()`。** 它是单例,配置改不了。本模块直接
  `HybridRetriever(store=..., cfg=...)`,这样 `--set` 与 preset 才是真的。

## `dense_order` / `sparse_order` 是文档级

`hybrid.py:236-240` 把 `debug.dense_order` 建成 `{doc_id: rank}`,同一文档的
**所有块拿到相同 rank**。所以本模块的 `doc_source` 归因也是**文档级**的
(「这个块所属的文档被哪一路召回」),它在单块文档上才等于块级归因。
**不要**拿它做块级排序判断 —— 那是这个评估设施最容易犯的错。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

from config import BASE_DIR, RetrievalConfig, Settings, get_settings
from eval import corpus as C
from eval import report as R
from eval.metrics import (
    ChunkKey,
    METRIC_FIELDS,
    aggregate,
    aggregate_by_type,
    aggregate_negatives,
    score_negative,
    score_query,
    unique_window,
)
from eval.schema import EvalItem, EvalSet, EvalSetError, load_qa, make_meta
from ingest import IngestPipeline
from ingest.graph_pipeline import GraphIngestPipeline
from retrieve import HybridRetriever, RetrievalDebug
from store.graph_store import Neo4jGraphStore

__all__ = ["PRESETS", "PRESET_HELP", "RunSpec", "build_specs", "main"]

# --------------------------------------------------------------------- #
# 配置变体
# --------------------------------------------------------------------- #

#: preset 名 → 覆盖在 `.env` 之上的 RetrievalConfig 字段。
#: 每个都对应一个**具体想回答的问题**,不是「多给几个档位」。
PRESETS: dict[str, dict[str, Any]] = {
    # 照 .env,现实基线
    "shipped": {},
    # 从不加载 2.3G 重排器 —— 没下载模型的机器也能跑的第一个档
    "no-rerank": {"rerank_enabled": False},
    # 头条测量配置:关掉阈值,测的是**排序器**而不是 rerank_min_score
    "no-threshold": {"rerank_min_score": 0.0, "rerank_min_keep": 0},
    # 让 k=10 真有空间(fusion_top_k 默认 20,top_k=10 时够,但 k=20 就不够)
    "wide": {"fusion_top_k": 50},
    # Qdrant 自身 RRF 的默认值是 2,而这里是 60(见 store/qdrant_store.py:10-21):
    # 60 接近「名次几乎不影响权重」,2 是赢者通吃。这是保证会动的演示用例。
    "dense-rrf-k2": {"rrf_k": 2},
}

PRESET_HELP: dict[str, str] = {
    "shipped": "照 .env,现实基线",
    "no-rerank": "关重排 —— 不需要 2.3G 重排器",
    "no-threshold": "关阈值(min_score=0/min_keep=0)—— 头条测量配置",
    "wide": "fusion_top_k=50 —— 让 k=10 真有空间",
    "dense-rrf-k2": "RRF k=60→2 —— 保证会动的演示",
}

#: 可选的检索后端。加一个后端不等于加一个旋钮 —— 它得先在 `run_spec`
#: 里真的被构造出来,否则就是「看着在、其实没接线」。
RETRIEVERS: frozenset[str] = frozenset({"hybrid", "graphrag"})

#: 本进程内改不了、只能靠重启生效的键(`get_reranker` 缓存单例,
#: `retrieve/reranker.py:144-150` 只在第一次调用时读 cfg)。
#: **必须显式告警**:评估工具最经典的坑就是「改了参数、数字没动、以为结论是没影响」。
INERT_IN_PROCESS: dict[str, str] = {
    "rerank_model": "重排器是模块单例,同进程内换模型不会生效(要换得新起一个进程)",
    "rerank_max_length": "重排器是模块单例,同进程内改 max_length 不会生效",
}


@dataclass(frozen=True)
class RunSpec:
    """一个要跑的配置变体。"""

    name: str
    #: 喂给检索器的完整配置
    retrieval: RetrievalConfig
    #: `None` = 跟随 `retrieval.rerank_enabled`;否则显式覆盖 `retrieve(use_rerank=)`
    use_rerank: bool | None = None
    #: `hybrid` = 纯块级混合检索;`graphrag` = 混合检索 + 图上多跳来的额外召回。
    #: 两者的结果都进同一套打分,所以**只有这个字段不同**才是干净对照。
    retriever_kind: str = "hybrid"

    def label(self) -> str:
        r = self.retrieval
        bits = [
            f"retriever={self.retriever_kind}",
            f"rerank={self.use_rerank if self.use_rerank is not None else r.rerank_enabled}",
            f"min_score={r.rerank_min_score}",
            f"min_keep={r.rerank_min_keep}",
            f"fusion_top_k={r.fusion_top_k}",
            f"rrf_k={r.rrf_k}",
        ]
        return f"{self.name} ({', '.join(bits)})"


def _coerce(field_type: str, raw: str, key: str) -> Any:
    """`--set` 的字符串 → 声明类型。转不了就抛,不静默吞。"""
    t = field_type.strip().strip("'\"")
    if t in ("bool", "bool | None"):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on", "y"):
            return True
        if low in ("0", "false", "no", "off", "n"):
            return False
        raise ValueError(f"--set {key}: 要布尔值,实得 {raw!r}(可写 true/false/1/0)")
    if t == "int":
        return int(raw)
    if t == "float":
        return float(raw)
    if t == "str":
        return raw
    # 兜底:按数字解析,失败就当字符串
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """`["rrf_k=2", "rerank_min_score=0.0"]` → 校验过的覆盖字典。

    对着 `dataclasses.fields(RetrievalConfig)` 校验,**未知键是硬错误**。
    拼错一个键、参数静默无效、结论却是「这个参数没影响」—— 这是评估工具里
    代价最高的一类 bug,因为它的产出是一条**错误的结论**,而不是一个报错。
    """
    known = {f.name: f.type for f in dataclasses.fields(RetrievalConfig)}
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set 需要 key=value 形式,实得 {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in known:
            near = sorted(k for k in known if key.lower() in k.lower() or k.lower() in key.lower())
            hint = f";最接近的:{near}" if near else ""
            raise ValueError(
                f"--set {key}: RetrievalConfig 没有这个字段{hint}\n"
                f"  可用的键:{', '.join(sorted(known))}"
            )
        # 转换失败要给出**键名和期望类型**,而不是把 `invalid literal for int()`
        # 原样抛给用户 —— 那个报错里没有键名,而一次 --set 可以写十几个键。
        try:
            out[key] = _coerce(known[key], raw, key)
        except (TypeError, ValueError) as exc:
            t = known[key].strip().strip("'\"")
            raise ValueError(f"--set {key}={raw!r}:转不成 {t} —— {exc}") from None
    return out


def build_specs(
    base: RetrievalConfig,
    presets: Sequence[str],
    overrides: Mapping[str, Any] | None = None,
    retrievers: Sequence[str] | None = None,
) -> list[RunSpec]:
    """preset 列表 → RunSpec 列表。

    每个 preset 都从**同一个 base** 派生(不互相叠加、不链式),
    `--set` 则叠加在**所有** spec 之上 —— 因为它的用途是「在这批档位上统一挪一个旋钮」。

    多个 `--preset` 得到**多个 spec**(一趟跑完、模型只加载一次),这适合做档位横评。
    但「阈值中性 **且** 放宽融合池」这类**组合**要的是**一个** spec,
    所以名字里可以用 `+`(或 `,`)合并:`--preset no-threshold+wide`。
    合并顺序即书写顺序,后者覆盖前者。

    `retrievers` 是**另一根轴**:`hybrid` 与 `graphrag` 各自跑一遍全部 preset。
    两个轴是**笛卡尔积**而不是叠加 —— 因为「graphrag + no-rerank」这类组合
    才是想知道「图的增益是不是被重排吃掉」时该看的东西。

    命名规则:`hybrid` 保持**裸名**(`no-threshold+wide`),这样既有基线
    (`before` / `rrf60` / `ct_on` …)的 run key 一个都不用改;其它后端加前缀
    (`graphrag:no-threshold+wide`)。同一个进程里跑两条后端时,名字必须能分开,
    否则后者会把前者的结果覆盖掉 —— 而覆盖是静默的。

    `no-rerank` 被强制排到最前:它不需要加载 2.3G 重排器,先跑它能让
    「只是想看看混合检索什么水平」的人立刻拿到数字,而不用先等模型下载。
    """
    names = list(presets) or ["shipped"]
    kinds = list(retrievers) or ["hybrid"]
    unknown_kind = [k for k in kinds if k not in RETRIEVERS]
    if unknown_kind:
        raise ValueError(
            f"未知检索后端:{', '.join(unknown_kind)}\n"
            f"  可用:{', '.join(sorted(RETRIEVERS))}"
        )

    specs: list[RunSpec] = []
    for kind in kinds:
        for name in names:
            parts = [p for p in name.replace(",", "+").split("+") if p]
            unknown = [p for p in parts if p not in PRESETS]
            if unknown:
                raise ValueError(
                    f"未知 preset:{', '.join(unknown)}\n"
                    f"  可用:{', '.join(f'{k}({v})' for k, v in PRESET_HELP.items())}\n"
                    f"  组合用 + 连接,例如 --preset no-threshold+wide"
                )
            kw: dict[str, Any] = {}
            for part in parts:  # 后者覆盖前者
                kw.update(PRESETS[part])
            kw.update(overrides or {})
            kw.setdefault("rerank_enabled", base.rerank_enabled)
            bare = "+".join(parts)
            specs.append(
                RunSpec(
                    name=bare if kind == "hybrid" else f"{kind}:{bare}",
                    retrieval=dataclasses.replace(base, **kw),
                    retriever_kind=kind,
                )
            )

    specs.sort(key=_spec_order)
    return specs


def _spec_order(s: RunSpec) -> tuple[int, int, str]:
    """排序只为**读起来顺**:先 no-rerank(不必加载重排器),再按后端分组。"""
    bare = s.name.split(":", 1)[-1]
    return (
        0 if bare == "no-rerank" else 1,
        0 if s.retriever_kind == "hybrid" else 1,
        bare,
    )


def effective_k(spec: RunSpec, top_k: int) -> int:
    """这个配置下**真的**能看到几个位置。

    融合结果先被 `limit=fusion_top_k` 收口,再重排、再 `[:top_k]`。
    所以天花板是 `min(top_k, fusion_top_k)`,而融合池本身又不可能超过
    两路召回条数之和(`dense_top_k + sparse_top_k`)。
    """
    r = spec.retrieval
    return min(top_k, r.fusion_top_k, r.dense_top_k + r.sparse_top_k)


# --------------------------------------------------------------------- #
# 问答集选取
# --------------------------------------------------------------------- #


def select_items(es: EvalSet, *, include_drafts: bool) -> EvalSet:
    """按 status 挑出参与打分的题。

    返回**新的 EvalSet**,所以 `sha1()` 是对实际参与打分的那个集合算的。
    这是对的:`--include-drafts` 之后的问题集和之前不是同一个集合,
    拿它们的分数做 diff 没有意义,而 `comparability_problems` 正好会拦住。
    """
    wanted = {"keep"} | ({"draft"} if include_drafts else set())
    items = [i for i in es.items if i.status in wanted]
    return EvalSet(items=items, meta=dict(es.meta), path=es.path)


# --------------------------------------------------------------------- #
# 导入
# --------------------------------------------------------------------- #


def _open_eval_store(settings: Settings, collection: str):
    """**只**构造评估用的 store,永远不碰生产 collection。

    让「不碰正式库」成为代码性质而不是口头约定:名字前缀不对就拒绝开跑。
    """
    from store import QdrantStore

    if collection == settings.qdrant.collection:
        raise RuntimeError(
            f"评估 collection 与生产 collection 同名({collection})—— 拒绝运行。\n"
            f"  这是代码级护栏:评估会写入/删除数据,绝不能落在正式库上"
        )
    if not collection.startswith("agentic_kb_eval_"):
        raise RuntimeError(
            f"评估 collection 必须以 agentic_kb_eval_ 开头,实得 {collection!r}"
        )
    return QdrantStore(cfg=dataclasses.replace(settings.qdrant, collection=collection))


def _open_eval_graph(settings: Settings) -> Neo4jGraphStore:
    """给评估开图。**和 Qdrant 那边不一样:这里没有第二张图可开。**

    Neo4j Community 只有**一个** database(`config.Neo4jConfig.database`),
    没法像 collection 那样给评估单开一个库。所以隔离只能靠 `source` 过滤
    (见 `GraphRAGBackend.sources`),而不是靠「另开一张图」。

    代价要说清楚:评估**会写进生产图所在的那个库**,和正式语料共存。
    因此:
      - 作用域过滤是**必须**的,不是可选的优化 —— 没有它,评估的图查询
        会看见生产实体,而报告上看不出任何异常。
      - 反过来也别指望 `clear()` 清理评估痕迹 —— 那会把生产图一起清掉。
    """
    store = Neo4jGraphStore(cfg=settings.neo4j)
    health = store.health()
    if not health.get("ok"):
        raise RuntimeError(
            f"连不上 Neo4j({settings.neo4j.uri}):{health.get('error')}\n"
            f"  graphrag 这一档需要它。先跑 scripts\\start_neo4j.bat。"
        )
    # 顺手把版本打出来。上面那段「只有一个库」的推理成立与否取决于
    # edition —— 换成 Enterprise 就该重新考虑要不要单开一个 database,
    # 而这个前提不该只活在注释里。
    print(
        f"  Neo4j {health.get('server')} {health.get('version')}"
        f" ({health.get('edition')}) — "
        + (
            "按 source 过滤隔离"
            if health.get("edition") == "community"
            else "注意:这个版本支持多库,可以不再靠属性过滤"
        )
    )
    return store


def ingest_corpus(
    *,
    root: Path,
    settings: Settings,
    store,
    contextual: bool,
    force: bool,
) -> Any:
    """把语料导进评估 collection。

    `cfg.contextual_enabled` 走 `dataclasses.replace` 而不是改全局 ——
    全局改法在 A/B 两档连着跑时会串味(第二档继承第一档的设置)。
    """
    ingest_cfg = dataclasses.replace(settings.ingest, contextual_enabled=contextual)
    pipeline = IngestPipeline(store=store, cfg=ingest_cfg)
    return pipeline.ingest_path(root, recursive=True, force=force, skip_errors=True)


# --------------------------------------------------------------------- #
# 检索与评分
# --------------------------------------------------------------------- #


def _doc_source_map(debug: RetrievalDebug) -> dict[str, str]:
    """doc_id → 被哪一路召回。**文档级**,理由见模块头。"""
    d = set(debug.dense_order or ())
    s = set(debug.sparse_order or ())
    out: dict[str, str] = {}
    for doc_id in d | s:
        in_d, in_s = doc_id in d, doc_id in s
        out[doc_id] = "both" if (in_d and in_s) else ("dense" if in_d else "sparse")
    return out


def _label(key: ChunkKey, labels: Mapping[ChunkKey, str]) -> str:
    return labels.get(key) or f"{key[0][:8]}#{key[1]}"


def build_retriever(
    spec: RunSpec,
    *,
    store,
    graph=None,
    graph_sources: Sequence[str] | None = None,
):
    """按 `spec.retriever_kind` 造检索器。

    两条路**必须**都拿到评估侧的 store(和评估侧的图作用域)。默认参数是
    生产库,而评估的块在评估 collection 里 —— 一旦退回默认,图反查会在生产库里
    查不到块,表现为「图的召回全是 missing」,数字莫名其妙地低,而报错一个没有。
    """
    if spec.retriever_kind == "hybrid":
        return HybridRetriever(store=store, cfg=spec.retrieval)

    if spec.retriever_kind != "graphrag":
        raise ValueError(f"未知 retriever_kind: {spec.retriever_kind!r}")

    if graph is None:
        raise RuntimeError(
            "spec 要求 graphrag,但没传图实例 —— 拒绝退回生产图。\n"
            "  构造 GraphRAGBackend 时省略 graph 会默认拿 get_graph_store(),\n"
            "  那是生产图;而评估的块只在评估 collection 里,eval 侧反查会全部落空。\n"
            "  这类失败不报错、只是分数变低,最难查 —— 所以在这里直接拦住。"
        )
    from retrieve.backends import GraphRAGBackend

    return GraphRAGBackend(
        store=store,
        graph=graph,
        cfg=spec.retrieval,
        sources=graph_sources,
    )


def run_spec(
    spec: RunSpec,
    *,
    store,
    items: Sequence[EvalItem],
    anchors: C.AnchorReport,
    ks: Sequence[int],
    top_k: int,
    foreign_golds: set[ChunkKey],
    graph=None,
    graph_sources: Sequence[str] | None = None,
) -> R.RunResult:
    """跑一个配置变体,返回可直接落盘的结果。"""
    eff = effective_k(spec, top_k)
    use_rerank = spec.use_rerank if spec.use_rerank is not None else spec.retrieval.rerank_enabled

    # store 也要拿到**本 spec** 的 retrieval,不能让它留着建库时的那份。
    # `query_hybrid` 从 `self.retrieval` 读 `dense_top_k` / `sparse_top_k` /
    # `rrf_k`(只有 `fusion_top_k` 是 retrieve 显式传进去的),而 `_open_eval_store`
    # 建库时用的是**生产**配置 —— 于是 `--set rrf_k=2` 这类键会**静默无效**,
    # 更糟的是 `effective_k` 会照 spec 算出一个**根本不存在**的天花板,
    # `hit@10` 就成了幻觉数。这与计划里点名的 `get_reranker` 单例是同一类坑:
    # 旋钮看着在、其实没接线。
    # 评估是单进程顺序跑的,store 也是评估专用的,所以这里直接绑过去。
    store.retrieval = spec.retrieval

    retriever = build_retriever(
        spec, store=store, graph=graph, graph_sources=graph_sources
    )

    # (doc_id, chunk_index) → "文件#块",只为人看得懂
    labels: dict[ChunkKey, str] = {}
    for g in anchors.all_golds:
        labels[g.key] = g.label

    res = R.RunResult(
        spec=spec.name,
        use_rerank=bool(use_rerank),
        retrieval=dataclasses.asdict(spec.retrieval),
        effective_k=eff,
        ks=list(ks),
        retriever=spec.retriever_kind,
    )

    per_k_scores: dict[int, list] = {k: [] for k in ks}
    per_k_neg: dict[int, list] = {k: [] for k in ks}
    nonempty: list[float] = []
    tie_ambiguous: list[str] = []

    for item in items:
        golds = anchors.resolved.get(item.id, [])
        debug = RetrievalDebug()  # 每次新的:便宜,且避免上一题的漏斗计数串味
        try:
            hits = retriever.retrieve(
                item.question, top_k=top_k, use_rerank=use_rerank, debug=debug
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"检索 {item.id} 时出错:{exc}\n"
                f"  这一条的问题文本:{item.question!r}"
            ) from exc

        # 钉死并列分数的先后,否则同一份数据两次运行会给出不同的名次 ——
        # 实测过,不是理论担忧。见 `_stable_hits`。
        hits = _stable_hits(hits)

        ranked: list[ChunkKey] = [(h.doc_id, h.chunk_index) for h in hits]
        ordered = unique_window(ranked, len(ranked)) if ranked else []
        nonempty.append(1.0 if ranked else 0.0)
        dropped = int(debug.dropped_by_threshold or 0)
        srcmap = _doc_source_map(debug)

        if item.is_negative:
            # 阈值必须与这批分数**同源**:重排开了给重排分、关掉给 RRF 分。
            # 而阈值只在重排分支里存在,所以关掉重排时用 0.0 表示「没有阈值」——
            # 那时 `leak` 退化成「返回非空」,这个退化是诚实的,记进基线里的
            # `_threshold` 会让人看见。
            threshold = float(spec.retrieval.rerank_min_score) if use_rerank else 0.0
            # score_negative 不接受 k —— 反例评的是分数分布(会不会硬凑),
            # 不是「前几条里有没有」。所以一个配置下只算一次,所有 k 共用。
            neg = score_negative(
                [((h.doc_id, h.chunk_index), float(h.score or 0.0)) for h in hits],
                threshold,
                foreign_golds=foreign_golds,
                item_id=item.id,
            )
            for k in ks:
                per_k_neg[k].append(neg)
            res.negative_items.append(
                R.NegativeDetail(
                    item_id=item.id,
                    question=item.question,
                    top1=_label(neg.top1_key, labels) if neg.top1_key else None,
                    top1_score=neg.top1_score,
                    leak=neg.leak,
                    n_returned=neg.n_returned,
                    foreign_gold=neg.top1_is_foreign_gold,
                    threshold=neg.threshold,
                )
            )
            continue

        grades = {g.key: g.grade for g in golds}
        for k in ks:
            per_k_scores[k].append(
                score_query(
                    ranked, grades, k, item_id=item.id, type=item.type
                )
            )

        # 明细与 k 无关(名次、返回条数都不随 k 变),所以只存一份
        rank_lo, rank_hi = _tie_band(hits, grades)
        if rank_lo != rank_hi:
            tie_ambiguous.append(item.id)
        res.per_item.append(
            R.ItemDetail(
                item_id=item.id,
                type=item.type,
                question=item.question,
                rank=_rank_of_first(ordered, grades),
                rank_lo=rank_lo,
                rank_hi=rank_hi,
                n_returned=len(ranked),
                n_gold=len(golds),
                dropped=dropped,
                golds=[
                    (_label(g.key, labels), _position(ordered, g.key)) for g in golds
                ],
                top1=_label(ordered[0], labels) if ordered else None,
                top1_score=float(hits[0].score or 0.0) if hits else 0.0,
                doc_source={
                    _label(g.key, labels): srcmap.get(g.doc_id, "-") for g in golds
                },
                graph_hits=sum(1 for h in hits if h.meta.get("from_graph")),
            )
        )

    # ---- 聚合 ----
    res.n_answerable = len(per_k_scores[ks[0]]) if ks else 0
    res.n_negative = len(per_k_neg[ks[0]]) if ks else 0
    res.tie_ambiguous = tie_ambiguous
    ne = sum(nonempty) / len(nonempty) if nonempty else 0.0

    for k in ks:
        qs = per_k_scores[k]
        agg = aggregate(qs)
        res.aggregates[str(k)] = {"n": agg.n, **{f: agg.get(f) for f in METRIC_FIELDS}}
        res.nonempty_rate[str(k)] = ne
        by_type = aggregate_by_type(qs)
        res.by_type[str(k)] = {
            t: {"n": a.n, **{f: a.get(f) for f in METRIC_FIELDS}}
            for t, a in by_type.items()
        }
        neg = aggregate_negatives(per_k_neg[k])
        vals = dict(neg.values)
        # 阈值记进基线 —— 分数离开阈值就没有意义,而重排开关决定有没有阈值
        vals["_threshold"] = (
            float(spec.retrieval.rerank_min_score) if use_rerank else 0.0
        )
        res.negatives[str(k)] = {"n": neg.n, **vals}

    return res


def _stable_hits(hits: Sequence[Any]) -> list[Any]:
    """把**并列分数**的先后钉死,让同一份数据两次运行给出同一个名次。

    为什么必须要这一步:RRF 的分数会**结构性**地产生精确并列 ——
    一块在稠密第 0 / 稀疏第 1、另一块在稠密第 1 / 稀疏第 0,
    两者都是 1/(k+0) + 1/(k+1),按分排序根本分不开。语料越小、
    两路候选越接近,这种交换并列就越密。

    而 Qdrant 服务端对并列的顺序**不保证稳定**。实测(seed 语料 seed-010):
    `04#1` 与 `04#5` 同分 0.033060110,同一进程内连续 6 次查询顺序就翻转,
    于是「并列对里谁在前」成了抛硬币,整个评估集的 hit@1 在 0.800 / 0.867
    之间来回跳(0.067 的摆幅,比大多数参数改动的效果都大)。

    对做回归追踪的工具来说,不可复现的基线等于没有基线。所以这里自己定序:
    分数降序、同分按 (doc_id, chunk_index)。**只改并列的先后,
    不改分数、不改返回集合**;并列区间的宽度由 `_tie_band` 另外如实报出来。
    """
    return sorted(hits, key=lambda h: (-float(h.score or 0.0), h.doc_id, h.chunk_index))


def _tie_band(
    hits: Sequence[Any], gold_keys: Collection[ChunkKey]
) -> tuple[int | None, int | None]:
    """首个 gold 在「并列怎么排都合法」前提下的名次区间 (最好, 最坏)。

    最好 = 1 + 分数**严格更高**的块数;最坏 = 分数**不低于**它的块数
    (同分的块可以任意排在它前面)。两者不等 ⟹ 这一题的名次是一次抛硬币:
    定序让**本次**可复现,但并列本身是真实存在的,**单题的名次变化
    不构成结论**。区间为 None ⟹ 一个 gold 都没召回(那是真的没召回,
    与并列无关)。
    """
    scores = [float(h.score or 0.0) for h in hits]
    gs = set(gold_keys)
    best_gold = [
        s for h, s in zip(hits, scores) if (h.doc_id, h.chunk_index) in gs
    ]
    if not best_gold:
        return None, None
    top = max(best_gold)
    higher = sum(1 for s in scores if s > top)
    tied = sum(1 for s in scores if s == top)
    return higher + 1, higher + tied


def _warn_on_noop(results: Mapping[str, R.RunResult]) -> None:
    """两个 retrieval 不同的 spec 给出逐位相同的分数 ⟹ 有一个旋钮没接线。

    只比聚合与逐题名次,不比时间戳之类的元数据。同名 spec 自然跳过。
    """
    runs = list(results.values())
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            # 参数相同**且**后端相同才跳过。后端不同也算「本该有差别」——
            # 换了一整条检索通路而分数一位不差,和换了个旋钮而分数一位不差
            # 是同一个信号:那个差异没接线。
            if a.retrieval == b.retrieval and a.retriever == b.retriever:
                continue
            if a.aggregates == b.aggregates and [
                (d.item_id, d.rank) for d in a.per_item
            ] == [(d.item_id, d.rank) for d in b.per_item]:
                if a.retriever != b.retriever:
                    hits = sum(d.graph_hits for d in b.per_item) + sum(
                        d.graph_hits for d in a.per_item
                    )
                    print(
                        f"  ⚠ {a.spec}({a.retriever}) 与 {b.spec}({b.retriever}) "
                        "分数**逐位相同**。\n"
                        f"    逐题的图命中合计 = {hits}。"
                        + (
                            "  → 图那一路**一条都没多召回**,这一档等价于纯向量;"
                            "先别把它当「GraphRAG 没用」的结论,查通路接上了没。"
                            if hits == 0
                            else "  → 图确实召回了块但没改变名次,这才是一个真结论。"
                        )
                    )
                    continue
                diff = {
                    k: (a.retrieval.get(k), b.retrieval.get(k))
                    for k in set(a.retrieval) | set(b.retrieval)
                    if a.retrieval.get(k) != b.retrieval.get(k)
                }
                print(
                    f"  ⚠ {a.spec} 与 {b.spec} 的参数不同({diff}),"
                    "但分数**逐位相同** —— 大概率有一个旋钮没接线(静默无效)。"
                    "先别读这组对比;查一下该键是不是只在 store 侧或模块单例里生效。"
                )


def _rank_of_first(ordered: Sequence[ChunkKey], gold: Mapping[ChunkKey, int]) -> int | None:
    gs = set(gold)
    for i, key in enumerate(ordered, 1):
        if key in gs:
            return i
    return None


def _position(ordered: Sequence[ChunkKey], key: ChunkKey) -> int:
    """某个 gold 在返回表里的名次(1 起)。0 = 没召回。"""
    for i, k in enumerate(ordered, 1):
        if k == key:
            return i
    return 0


# --------------------------------------------------------------------- #
# 反例体检(--verify-negatives)
# --------------------------------------------------------------------- #


def verify_negatives(
    retriever: HybridRetriever,
    items: Sequence[EvalItem],
    *,
    top_k: int = 10,
    margin: float = 1.0,
) -> tuple[list[tuple[str, float, float]], float]:
    """**纯稀疏、关重排**扫一遍,看反例的召回强度是不是和真题一个量级。

    为什么要这一条:一个和真实块共享罕见词的「反例」其实是**标错的答案题**,
    它会永远伪装成检索失败 —— 你会去调参数救一个根本不存在的目标。
    按文档提示出反例时这种错很容易犯(「拿这里出现的实体配这里没有的取值」,
    模型经常只做前半句)。

    判据用**相对**量而不是绝对阈值:Qdrant 稀疏分是学出来的词权内积,
    没有可跨语料搬用的绝对刻度。所以拿「可答题的 top-1 稀疏分」的中位数当基准,
    反例的 top-1 分达到它 × margin 就报出来。判据不需要调参,也不需要标定。

    返回 `(告警行, 基准中位数)`;告警行为 `(item_id, 该题 top-1 稀疏分, 基准)`。
    """
    store = retriever.store

    def top1_sparse(q: str) -> float:
        q = (q or "").strip()
        if not q:
            return 0.0
        vec = retriever.embedder.encode(q).sparse[0]
        pts = store.query_hybrid(None, vec, limit=top_k, sparse_top_k=top_k)
        return float(pts[0].score) if pts else 0.0

    base_scores = sorted(top1_sparse(i.question) for i in items if not i.is_negative)
    if not base_scores:
        return [], 0.0
    mid = base_scores[len(base_scores) // 2]

    flagged: list[tuple[str, float, float]] = []
    for it in items:
        if not it.is_negative:
            continue
        s = top1_sparse(it.question)
        if s >= mid * margin:
            flagged.append((it.id, s, mid))
    return flagged, mid


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_run.py",
        description=(
            "检索质量评估:导入语料、跑指标、存档基线。"
            "**不调用任何 LLM 配额**(只有开启定位语的导入阶段会调用)。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  eval_run.py --preset no-rerank\n"
            "  eval_run.py --preset no-threshold+wide --save-baseline before\n"
            "  eval_run.py --preset no-threshold+wide --set rrf_k=2 --baseline before\n"
            "  eval_run.py --preset no-threshold+wide --no-contextual --save-baseline noctx\n"
            "  eval_run.py --preset no-threshold+wide --retriever hybrid,graphrag\n"
            "  eval_run.py --locate \"值班人员每四小时抄录\"\n"
            "  eval_run.py --purge\n"
            "注意 --preset 的组合用 + 连接(no-threshold+wide);写成两个 --preset\n"
            "  是**两档横评**,不是一次组合。\n"
        ),
    )
    p.add_argument("--corpus", default=None, help="语料目录(默认 eval/corpus/seed)")
    p.add_argument("--qa", default=None, help="问答集 JSONL(默认 eval/qa/<语料 slug>.jsonl)")
    p.add_argument("--ks", default="1,3,5,10", help="要报的 k,逗号分隔(默认 1,3,5,10)")
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="传给 retrieve() 的 top_k。**必须显式传** —— retrieve 默认吃 rerank_top_n(5),"
        "不传的话 hit@10 是幻觉数",
    )
    p.add_argument(
        "--preset",
        action="append",
        default=[],
        metavar="NAME",
        help="可重复;组合用 + 连接成一个 spec(如 no-threshold+wide)。"
        + " / ".join(f"{k}={v}" for k, v in PRESET_HELP.items()),
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="可重复。对着 RetrievalConfig 校验,未知键是硬错误",
    )
    p.add_argument(
        "--retriever",
        default="hybrid",
        metavar="KIND[,KIND]",
        help="检索后端,逗号分隔(hybrid / graphrag)。每个后端都跑一遍全部 preset,"
        "是**笛卡尔积**不是叠加。graphrag 需要 Neo4j 起来、且图里已导入本语料"
        "(--graph-ingest);它的 run key 带 `graphrag:` 前缀。",
    )
    p.add_argument(
        "--graph-ingest",
        action="store_true",
        help="把评估语料抽进图(要 LLM,有成本)。**只对该语料** —— 靠 source "
        "作用域隔离,不动生产图。不加这个开关时,图里有什么就用什么。",
    )
    p.add_argument(
        "--no-contextual",
        action="store_true",
        help="导入时不生成定位语。落到**另一个 collection**(开关进名字)",
    )
    p.add_argument("--baseline", default=None, metavar="NAME", help="与这个基线比")
    p.add_argument("--save-baseline", default=None, metavar="NAME", help="把结果存成这个基线")
    p.add_argument("--reingest", action="store_true", help="强制重新导入(force=True)")
    p.add_argument("--purge", action="store_true", help="先删掉评估 collection 再跑")
    p.add_argument("--include-drafts", action="store_true", help="把 draft 状态的题也算进去")
    p.add_argument("--show-misses", action="store_true", help="列出未命中的题")
    p.add_argument("--show-items", action="store_true", help="列出所有题的逐题明细")
    p.add_argument("--verify-negatives", action="store_true", help="反例体检:纯稀疏扫一遍")
    p.add_argument(
        "--locate", default=None, metavar="TEXT",
        help="只查这段文字落在语料的第几块,然后退出(不连任何服务)",
    )
    p.add_argument("--json", default=None, metavar="OUT", help="把本次结果写成 JSON")
    p.add_argument("--dry-run", action="store_true", help="只打印计划,不连 Qdrant、不加载模型")
    return p


def parse_ks(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 1:
            raise ValueError(f"--ks 里出现了 {v}:k 必须 >= 1")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("--ks 是空的")
    return sorted(out)


def _print_status_counts(es: EvalSet) -> None:
    counts = es.status_counts()
    print(
        "问答集状态:"
        + "  ".join(f"{k}={counts.get(k, 0)}" for k in ("keep", "draft", "drop", "skip"))
    )
    print("题型分布:  " + "  ".join(f"{k}={v}" for k, v in sorted(es.type_counts().items())))
    for w in es.warnings():
        print(f"  ⚠ {w}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --- 0. 目录 ---
    # Settings.ensure_dirs() 定义了但全项目没人调用,所以 data/ 一直不存在。
    # 评估设施当它的第一个调用者:基线要落盘,日志要目录。
    try:
        s = get_settings()
        s.ensure_dirs()
    except Exception as exc:  # noqa: BLE001
        print(f"读不到配置:{exc}", file=sys.stderr)
        return 1

    # --- 1. 语料 ---
    try:
        root = C.resolve_corpus(args.corpus)
    except C.CorpusError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    docs, _load_errors = C.load_doc_chunks(
        root, chunk_size=s.ingest.chunk_size, chunk_overlap=s.ingest.chunk_overlap
    )
    n_chunks = sum(len(v) for v in docs.values())

    # --- --locate:定位器,不连任何服务 ---
    if args.locate:
        hits = C.locate_text(docs, args.locate)
        print(f"在 {root} 里找 {args.locate!r}:命中 {len(hits)} 处")
        for h in hits:
            print(f"  {h}")
        return 0

    # --- 2. 问答集 ---
    qa_path = (
        Path(args.qa)
        if args.qa
        else (BASE_DIR / "eval" / "qa" / f"{C.corpus_slug(root)}.jsonl")
    )
    if not qa_path.is_file():
        print(
            f"问答集不存在:{qa_path}\n"
            f"  先出题(**会花钱**):\n"
            f"    scripts\\eval_gen.py --corpus {root} --out {qa_path} --dry-run\n"
            f"    scripts\\eval_gen.py --corpus {root} --out {qa_path} --yes\n"
            f"  然后人工筛:把 draft 里好的行复制进定稿文件并改 status/type/grade。",
            file=sys.stderr,
        )
        return 1

    try:
        es_raw = load_qa(qa_path)
    except EvalSetError as exc:
        print(f"问答集读不了:{exc}", file=sys.stderr)
        return 1

    es = select_items(es_raw, include_drafts=args.include_drafts)
    print()
    _print_status_counts(es_raw)
    if args.include_drafts:
        print("  (--include-drafts:参与打分的是 keep + draft)")

    problems = es.validate()
    if problems:
        print("\n问答集校验失败:", file=sys.stderr)
        for m in problems:
            print(f"  - {m}", file=sys.stderr)
        return 1

    if not es.kept:
        print(
            "\n参与打分的题是 0 条 —— 中止。\n"
            "  空集会打印出一张全 0 的表,看着像灾难,其实什么都没测。\n"
            "  把要用的题改成 status=keep,或加 --include-drafts。",
            file=sys.stderr,
        )
        return 1

    # --- 3. meta 对不上就拒绝 ---
    bad_meta = False
    if es.meta:
        for key, actual in (
            ("chunk_size", s.ingest.chunk_size),
            ("chunk_overlap", s.ingest.chunk_overlap),
        ):
            expected = es.meta.get(key)
            if expected is not None and int(expected) != int(actual):
                print(
                    f"问答集的 meta.{key} = {expected},但当前配置是 {actual} —— 拒绝运行。\n"
                    f"  锚点是在 {expected} 下标的;换参数后 chunk_index 会整体位移,"
                    f"分数会掉但那不是检索退步。",
                    file=sys.stderr,
                )
                bad_meta = True
    if bad_meta:
        return 1

    # --- 4. 配置变体 ---
    try:
        overrides = parse_overrides(args.set)
        retrievers = [r.strip() for r in str(args.retriever).split(",") if r.strip()]
        specs = build_specs(s.retrieval, args.preset, overrides, retrievers=retrievers)
        ks = parse_ks(args.ks)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    for key in INERT_IN_PROCESS:
        if key in overrides:
            print(f"  ⚠ --set {key}:{INERT_IN_PROCESS[key]}", file=sys.stderr)

    contextual = not args.no_contextual
    collection = C.collection_name(root, contextual=contextual)

    if args.dry_run:
        print()
        print(R._RULE)
        print("DRY-RUN —— 不连 Qdrant、不加载模型、不写文件")
        print(R._RULE)
        print(f"语料    {root}  {len(docs)} 篇 / {n_chunks} 块")
        print(
            f"问答集  {qa_path}  keep {len(es.kept)} 条"
            f"(可答 {len(es.answerable)} / 反例 {len(es.negatives)})"
        )
        print(f"库      {collection}   定位语 = {'开' if contextual else '关'}")
        print(f"k       {','.join(map(str, ks))}   top_k={args.top_k}")
        print(
            "后端    "
            + ",".join(retrievers)
            + ("   (graphrag:先 --graph-ingest 把图建起来)" if "graphrag" in retrievers else "")
        )
        print()
        print("将跑的配置:")
        for sp in specs:
            # effective_k 放前面:名字列的中文/全角字符宽度不一,右对齐对不齐,
            # 而数字列对齐才有用
            print(f"  effective_k={effective_k(sp, args.top_k):<4} {sp.label()}")
        for key in sorted(INERT_IN_PROCESS):
            if key in overrides:
                print(f"  ⚠ --set {key}:{INERT_IN_PROCESS[key]}")
        print()
        if contextual:
            print(
                f"导入时**会调用 LLM** 生成定位语,按文档分批,约 {len(docs)} 次调用。\n"
                f"  (已经导入过且内容未变会跳过 —— 增量判定按块的内容指纹)"
            )
        else:
            print("定位语关闭,导入阶段**不调用 LLM**。")
        return 0

    # --- 5. 开评估库 ---
    try:
        store = _open_eval_store(s, collection)
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    if args.purge:
        if store.exists():
            store.client.delete_collection(collection)
            print(f"已删除 collection: {collection}")
        else:
            print(f"collection 不存在,无需删除:{collection}")

    store.ensure_collection(dense_dim=s.embed.dense_dim)

    # --- 6. 导入 ---
    before = store.count()
    need = before == 0 or args.reingest or args.purge
    if need:
        print()
        print(f"导入 {root} → {collection} …")
        t0 = time.time()
        try:
            stats = ingest_corpus(
                root=root, settings=s, store=store, contextual=contextual, force=args.reingest
            )
        except Exception as exc:  # noqa: BLE001
            print(f"导入失败:{exc}", file=sys.stderr)
            return 1
        print(f"  {stats.summary()}")
        for path, msg in list(stats.errors)[:10]:
            print(f"  ⚠ {path}: {msg}")
        n = store.count()
        print(f"  库内现有 {n} 块(导入前 {before}),耗时 {time.time() - t0:.1f}s")
        if stats.docs_unchanged and not args.reingest:
            print(
                f"  ⚠ 有 {stats.docs_unchanged} 篇被判为「未变化」而跳过重写。"
                f"若你刚改过定位语开关,确认 collection 名带对了 ctx/noctx ——"
                f"否则两次 A/B 读的是同一个库。"
            )
        if n == 0:
            print("导入后库仍是空的 —— 中止。", file=sys.stderr)
            return 1
    else:
        print(f"\n复用已有 collection {collection}({before} 块)。加 --reingest 可强制重导。")

    # --- 7. 锚点:前置校验 + 导入后复验,**任何查询之前** ---
    report = C.resolve_anchors(
        es,
        root,
        chunk_size=s.ingest.chunk_size,
        chunk_overlap=s.ingest.chunk_overlap,
        docs=docs,
    )
    report.issues.extend(C.verify_ingested(store, report))
    if report.issues:
        print()
        print(C.format_issues(report.issues), file=sys.stderr)
        return 1
    print(f"  {report.summary()}   ✓")

    # --- 7.5 图:作用域 = 向量侧能返回的那些 source ---
    #
    # 放在锚点校验**之后**是有意的:graphrag 那条路要花 LLM 的钱,评估集自己
    # 有问题就不该往下烧。锚点挂了会 return 1,走不到这里。
    graph = None
    graph_sources: list[str] | None = None
    if "graphrag" in retrievers:
        # 作用域不是手写的常量,而是**从评估库自己读出来**的:只有这样,
        # 「图能跳到的范围」才和「向量能召回的范围」是同一个集合。
        # 手写一份清单的话,两边会各自漂移,而漂移的症状是分数变低 ——
        # 没人能看出来那是配置错了还是检索真的差。
        graph_sources = sorted(
            {str(d.get("source") or "") for d in store.list_docs()} - {""}
        )
        if not graph_sources:
            print("评估库里没有文档,拿不到图的作用域 —— 中止。", file=sys.stderr)
            return 1
        graph = _open_eval_graph(s)

        if args.graph_ingest:
            print()
            print(f"抽取实体入图({len(docs)} 篇 / {n_chunks} 块,调用 LLM)…")
            t0 = time.time()
            try:
                gstats = GraphIngestPipeline(
                    store=store, graph=graph, cfg=s.ingest
                ).ingest_path(root, recursive=True, force=args.reingest)
            except Exception as exc:  # noqa: BLE001
                print(f"图导入失败:{exc}", file=sys.stderr)
                return 1
            print(f"  {gstats.summary()}")
            for path, msg in list(gstats.errors)[:10]:
                print(f"  ⚠ {path}: {msg}")
            print(f"  耗时 {time.time() - t0:.1f}s")

        # 入图之后**当场核一遍**:作用域内的块在图上有没有对应的 __Node__。
        # 不核的话,「图是空的」和「图建好了但这题确实跳不到」在报告里长得
        # 一模一样 —— 都是 multi_hop 全错。前者是配置事故,必须当场喊出来。
        in_scope = graph.count_chunks(sources=graph_sources)
        print()
        print(
            f"图    作用域 {len(graph_sources)} 个 source,"
            f"作用域内的块 {in_scope} / 向量侧 {store.count()}"
        )
        if in_scope == 0:
            print(
                "  ⚠ 图里**作用域内一个块都没有** —— graphrag 这一档只会退化成"
                "纯向量,跑出来的分不叫「GraphRAG 的效果」。\n"
                "    先加 --graph-ingest 把图建起来(或确认 Neo4j 里 source 路径"
                "与评估库一致)。",
                file=sys.stderr,
            )

    R.print_context(
        corpus_root=root,
        n_docs=len(docs),
        n_chunks=n_chunks,
        qa_path=qa_path,
        qa_sha1=es.sha1(),
        n_kept=len(es.kept),
        n_answerable=len(es.answerable),
        n_negative=len(es.negatives),
        collection=collection,
        contextual=contextual,
        ks=ks,
        top_k=args.top_k,
    )

    # --- 8. 跑 ---
    foreign: set[ChunkKey] = set()
    for item in es.answerable:
        foreign.update(report.keys(item.id))

    results: dict[str, R.RunResult] = {}
    for sp in specs:
        eff = effective_k(sp, args.top_k)
        if args.top_k > sp.retrieval.fusion_top_k:
            print(
                f"  ⚠ {sp.name}:top_k={args.top_k} > fusion_top_k="
                f"{sp.retrieval.fusion_top_k} —— 融合结果先被收口,"
                f"真正能看到的位置只有 {eff} 个。超过的 k 会从报告里删掉。",
                file=sys.stderr,
            )
        dropped_ks = [k for k in ks if k > eff]
        if dropped_ks:
            print(
                f"  ⚠ {sp.name}:k={dropped_ks} 超过 effective_k={eff},"
                f"**从报告里删掉**(打成 0 会把「测不了」伪装成「全错」)。",
                file=sys.stderr,
            )
        eff_ks = [k for k in ks if k <= eff] or [eff]
        t0 = time.time()
        results[sp.name] = run_spec(
            sp,
            store=store,
            items=list(es.kept),
            anchors=report,
            ks=eff_ks,
            top_k=args.top_k,
            foreign_golds=foreign,
            graph=graph,
            graph_sources=graph_sources,
        )
        r = results[sp.name]
        R.print_run(r, ks=eff_ks)
        R.print_by_type(r, eff_ks)
        R.print_negatives(r, min(eff_ks))
        if args.show_items:
            for d in r.per_item:
                mark = "✓" if d.rank else "✗"
                print(
                    f"     {mark} {d.item_id:<10} rank={d.rank}"
                    f" 返回{d.n_returned} 砍{d.dropped} {d.type}"
                )
        if args.show_misses:
            R.print_misses(r, min(eff_ks))
        print(f"   ({time.time() - t0:.1f}s)")

    # 「不能是 no-op」自检:retrieval 不同的两个 spec 却给出**逐位相同**的聚合,
    # 说明有一个旋钮根本没接线(最经典的就是 store 侧的 rrf_k / dense_top_k,
    # 以及被模块单例忽略的 rerank_model)。这比分数难看更该报警 ——
    # 分数难看至少是真的,静默无效会让人对着假数据调参数。
    _warn_on_noop(results)

    # 防循环自检:均值应落在宽区间内,否则先怀疑评估集而不是检索器。
    #
    # 必须**逐 k 检查**,不能只看 min(ks):头部 k 天然是最难的那个,
    # 只盯它会让这条断言几乎永不触发 —— 而 `hit@5 = 1.000` 恰恰是计划里
    # 点名的「评估集太简单」信号。打到 stdout 而不是 stderr,因为它要贴在
    # 数字旁边被看到;丢进 stderr 就会被 `> run.log` 冲散。
    for name, r in results.items():
        head = min(r.ks)
        h_head = r.agg(head).get("hit") or 0.0
        if h_head >= 0.99:
            print(
                f"  ⚠ {name}:k={head} 的 hit = {h_head:.3f} 几乎满分 —— 评估集可能太简单,"
                "或 gold 是从检索器自己的输出标的(自证循环)。抽查几条再下结论。"
            )
        elif h_head <= 0.05:
            print(
                f"  ⚠ {name}:k={head} 的 hit = {h_head:.3f} 几乎全错 —— 先怀疑锚点与导入,"
                "而不是先调参数。"
            )
        # 深 k 饱和是正常的(窗口越大越容易捞到),所以只作提示不作警告,
        # 但它必须被说出来:这些 k 上的「命中率」已经失去分辨率。
        saturated = [k for k in r.ks if (r.agg(k).get("hit") or 0.0) >= 0.99]
        if saturated and head not in saturated:
            print(
                f"  · {name}:k={','.join(str(k) for k in saturated)} 的 hit 已到 1.000 —— "
                f"这些 k 上命中率**没有分辨率**,要看 mrr/ndcg 或把 ks 往小取。"
            )

    # --- 9. 反例体检 ---
    if args.verify_negatives:
        sp = specs[0]
        retriever = HybridRetriever(store=store, cfg=sp.retrieval)
        print()
        print("反例体检(纯稀疏、关重排):")
        flagged, mid = verify_negatives(retriever, list(es.kept), top_k=args.top_k)
        print(f"   可答题 top-1 稀疏分中位数 = {mid:.4f}(基准)")
        if flagged:
            for iid, sc, m in flagged:
                print(
                    f"   ⚠ {iid} 的 top-1 稀疏分 {sc:.4f} ≥ 基准 {m:.4f} —— "
                    f"纯词法都能这么强,它可能**其实是可答题**,标注要人工复核"
                )
        else:
            print("   全部反例的 top-1 稀疏分都低于基准 —— 没发现标错的反例。")

    # --- 10. 基线 ---
    payload = R.build_baseline(
        corpus={
            "root": str(root),
            "slug": C.corpus_slug(root),
            "n_docs": len(docs),
            "n_chunks": n_chunks,
        },
        qa={
            "path": str(qa_path),
            "sha1": es.sha1(),
            "n_kept": len(es.kept),
            "n_answerable": len(es.answerable),
            "n_negative": len(es.negatives),
            "include_drafts": bool(args.include_drafts),
        },
        ingest={
            "chunk_size": s.ingest.chunk_size,
            "chunk_overlap": s.ingest.chunk_overlap,
            "contextual_enabled": contextual,
            "embed_model": s.embed.model_name,
            "rerank_model": s.retrieval.rerank_model,
        },
        collection=collection,
        ks=ks,
        runs=results,
    )

    if args.baseline:
        bpath = R.resolve_baseline_path(args.baseline)
        try:
            before_payload = R.load_baseline(bpath)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"读不了基线:{exc}", file=sys.stderr)
            return 1
        problems = R.comparability_problems(before_payload, payload)
        if problems:
            print(f"\n两份基线不可比 —— 拒绝出 diff({bpath}):", file=sys.stderr)
            for m in problems:
                print(f"  - {m}", file=sys.stderr)
            print(
                "\n  这不是小事:改了一条问题、重跑、然后以为「检索器变了」,"
                "是这套设施最该拦住的一种错误结论。",
                file=sys.stderr,
            )
            return 1
        R.print_diff(before_payload, payload, ks=ks)

    if args.save_baseline:
        p = R.save_baseline(payload, R.resolve_baseline_path(args.save_baseline))
        print(f"\n基线已存:{p}")

    if args.json:
        p = R.dump_json(payload, args.json)
        print(f"结果已写:{p}")

    return 0
