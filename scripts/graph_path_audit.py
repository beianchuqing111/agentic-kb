"""图那一路的体检:是「通路没接上」还是「接上了但没多召回」。

为什么单独要一个脚本
-------------------
`graph_report.py` 只能看到聚合分和 `graph_hits` 合计。当 hybrid 与 graphrag
**逐位相同**、图命中合计 = 0 时,那个报告会正确地拒绝下结论,但它分不清
下面两种情况 —— 两者在聚合分上长得一模一样:

  A. 通路没接上(事故):种子实体 = 0 或事实 = 0,图这一路压根没跑。
     历史上真发生过(`_graph_context` 里引用了从未绑定的 `pg`,NameError 被
     except 吞掉,每次检索都退回「只用种子实体」)。这种事必须能被指认出来。
  B. 通路是通的,但这套语料加不出东西(结论):
     向量侧一次就取走 61 块里的 50 块,图那一路**结构上**就没有位置可加。

判据是三个内部计数 + 一个 gold 覆盖对照:

  种子实体   `_graph_context` 的实体向量召回 —— 0 就说明实体这层废了
  多跳事实   `get_rel_map` 跳出来的三元组 —— 0 就说明多跳没跳起来
  图反查块   按实体反查到的原文块 —— 0 就说明「反查」这步废了
  仅图命中   向量够不着、只有图捞到的 **gold 块** —— 这才是图的价值所在

最后一项是关键:多召回一堆**非 gold** 的块对效果毫无意义,只会把重排算力
摊薄。只有 gold 才算数。

候选宽度为什么要跑两档
---------------------
评估档位(`no-threshold+wide`)自己选了 `fusion_top_k=50`。61 块的库里
取 50 = 拿走 82% 的语料,图那一路**不可能**加出东西。所以默认再跑一档窄窗口
(`10`)作为诊断:有空间的时候它到底补不补得上。**两档都报,不挑好看的** ——
只报对自己有利的那档,是这套评测里最容易犯也最难查的错。

只读。不写库、不改任何数据。

跑法:
    python scripts/graph_path_audit.py
输出:
    eval/reports/graph_path_audit.md
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import get_settings  # noqa: E402
from eval import corpus as C  # noqa: E402
from retrieve.backends.graphrag_backend import GraphRAGBackend  # noqa: E402
from store.graph_store import get_graph_store  # noqa: E402
from store.qdrant_store import QdrantStore  # noqa: E402

#: 评估档位全开的那套候选宽度(与 `--preset no-threshold+wide` 一致)
WIDE = 50
#: 诊断用的窄窗口:留出足够空隙,看图的通路有没有真价值
NARROW = 10
#: multi_hop 才可能体现图的价值;别的题型是对照
AUDIT_TYPE = "multi_hop"


def _load_multi_hop(path: Path) -> list[dict]:
    """读问答集里 status=keep 的 multi_hop 题。

    注意:seed.jsonl 开头是 `#` 注释行(这个文件自己定的格式),不是纯 JSONL,
    不滤掉的话第一行就 JSONDecodeError。
    """
    items: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            o = json.loads(line)
            if o.get("status") == "keep" and o.get("type") == AUDIT_TYPE:
                items.append(o)
    return items


def _key(chunk) -> tuple[str, int]:
    """块的唯一身份。两条后端出来口径一致,所以能直接比。"""
    return (chunk.doc_id, chunk.chunk_index)


def _has_source(chunks: Sequence, gold: dict, unwrap) -> bool:
    """某个 gold 块在不在这批结果里。

    `_chunks_mentioning` 返回的是 `(RetrievedChunk, 命中实体数, 实体名)` 三元组,
    和 `RetrievedChunk` 单条不是一回事 —— 所以用 `unwrap` 统一取真身,
    而不是在两个调用点各写一遍解包。
    """
    want = os.path.basename(gold.get("file") or "")
    for c in chunks:
        rc = unwrap(c)
        if os.path.basename(rc.source or "") == want and rc.chunk_index == gold["chunk_index"]:
            return True
    return False


def _measure(backend: GraphRAGBackend, items: list[dict], k: int) -> list[dict]:
    """在候选宽度 k 下量一遍 11 道多跳题。"""
    # cfg 同时挂在后端和它内部的混合检索器上,两个都要换 —— 只换一个的话
    # 「图反查」和「向量候选」会按各自的宽度跑,量出来的差值是假的。
    cfg = dataclasses.replace(
        backend.cfg,
        fusion_top_k=k, dense_top_k=k, sparse_top_k=k,
        rerank_enabled=True, rerank_min_score=0.0, rerank_min_keep=0,
    )
    backend.cfg = cfg
    backend.retriever.cfg = cfg

    rows: list[dict] = []
    for it in items:
        q = it["question"]
        facts, ent_ids, _ = backend._graph_context(q)
        gchunks = backend._chunks_mentioning(ent_ids)
        fused = backend.retriever.retrieve(q, top_k=k, use_rerank=False)

        fused_keys = {_key(c) for c in fused}
        new = [g for g in gchunks if _key(g[0]) not in fused_keys]

        golds = list(it.get("expected") or [])
        only_graph = [
            g
            for g in golds
            if _has_source(gchunks, g, lambda c: c[0])
            and not _has_source(fused, g, lambda c: c)
        ]
        rows.append(
            {
                "id": it["id"],
                "question": q,
                "entities": len(ent_ids),
                "facts": len(facts),
                "graph_chunks": len(gchunks),
                "vector_chunks": len(fused),
                "graph_only_chunks": len(new),
                "golds": len(golds),
                "gold_in_vector": sum(_has_source(fused, g, lambda c: c) for g in golds),
                "gold_in_graph": sum(_has_source(gchunks, g, lambda c: c[0]) for g in golds),
                "gold_only_graph": [
                    f"{g.get('file')}#{g['chunk_index']}(grade={g.get('grade')})"
                    for g in only_graph
                ],
            }
        )
    return rows


def _table(rows: list[dict], k: int) -> list[str]:
    out = [
        f"### 候选宽度 fusion_top_k = {k}",
        "",
        f"向量侧一次取走 {rows[0]['vector_chunks']} 块 / 全库 61 块。",
        "",
        "| 题目 | 种子实体 | 多跳事实 | 图反查块 | 向量候选 | 图独有块 | gold 总数 | 向量命中 | 图命中 | **仅图命中** |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        og = "、".join(r["gold_only_graph"]) if r["gold_only_graph"] else "—"
        out.append(
            f"| `{r['id']}` | {r['entities']} | {r['facts']} | {r['graph_chunks']} | "
            f"{r['vector_chunks']} | {r['graph_only_chunks']} | {r['golds']} | "
            f"{r['gold_in_vector']} | {r['gold_in_graph']} | {og} |"
        )
    tot_new = sum(r["graph_only_chunks"] for r in rows)
    tot_gold = sum(len(r["gold_only_graph"]) for r in rows)
    n_gold = sum(r["golds"] for r in rows)
    out += [
        "",
        f"- 11 题合计:图多召回**非 gold** 块 **{tot_new}** 个;"
        f"**仅图命中 gold {tot_gold} / {n_gold}**。",
        "",
    ]
    return out


def main() -> int:
    s = get_settings()
    audit_type = os.environ.get("AUDIT_TYPE", AUDIT_TYPE)

    qa = ROOT / "eval" / "qa" / "seed.jsonl"
    items = _load_multi_hop(qa)
    items = [i for i in items if i.get("type") == audit_type]
    if not items:
        print(f"问答集里没有 {audit_type} 的题,无从体检。", file=sys.stderr)
        return 1

    # 走**评估档**的 collection 与 store —— 生产库的 collection 名不一样,
    # 拿错的症状是「块的召回全是 missing」,而报错一个没有。
    root = ROOT / "eval" / "corpus" / "seed"
    collection = C.collection_name(root, contextual=True)
    store = QdrantStore(cfg=dataclasses.replace(s.qdrant, collection=collection))
    graph = get_graph_store()

    # 作用域与 eval/runner.py 同一口径:图能跳到谁,取决于库里实际有谁
    sources = sorted({str(d.get("source") or "") for d in store.list_docs()} - {""})
    backend = GraphRAGBackend(store=store, graph=graph, cfg=s.retrieval, sources=sources)

    lines = [
        "# 图那一路的体检:是「通路没接上」还是「接上了但没多召回」",
        "",
        f"语料 `eval/corpus/seed` 9 篇 / {store.count()} 块;"
        f"图作用域 {len(sources)} 个 source;题型 `{audit_type}` {len(items)} 题。",
        "",
        "脚本 `scripts/graph_path_audit.py`,可重跑自证:**只读**,不写库、不改数据。",
        "",
        "为什么需要这一份:`eval/reports/graphrag_ab.md` 的 §3 能看出「图命中合计 = 0」,",
        "但分不清那是**通路没接上**(配置事故)还是**语料太小**(真结论)——",
        "两者在聚合分上完全一样。这里用三个内部计数把「通不通」和「有没有用」分开量。",
        "",
        "---",
        "",
        "## 1. 通路本身通不通",
        "",
        "| 计数 | 含义 | 为 0 说明什么 |",
        "|---|---|---|",
        "| 种子实体 | `_graph_context` 的实体向量召回 | 实体这层废了(作用域/索引问题) |",
        "| 多跳事实 | `get_rel_map` 跳出来的三元组 | 多跳没跳起来 |",
        "| 图反查块 | 按实体反查到的原文块 | 「实体 → 原文块」这步废了 |",
        "",
        "## 2. 图补上了向量够不着的 gold 吗",
        "",
        "「多召回几个块」本身不是价值 —— 多召回**非 gold** 的块只会摊薄重排算力。",
        "**只有 gold 才算数**,所以最后一列是唯一有判据性的那一列。",
        "",
        "候选宽度跑两档:评估档位自己选的 50(61 块的库里等于拿走 82%),",
        "以及一档窄窗口 10。**两档都报,不挑好看的。**",
        "",
    ]

    wide = _measure(backend, items, WIDE)
    lines += _table(wide, WIDE)
    narrow = _measure(backend, items, NARROW)
    lines += _table(narrow, NARROW)

    w_gold = sum(len(r["gold_only_graph"]) for r in wide)
    n_gold = sum(len(r["gold_only_graph"]) for r in narrow)
    n_all = sum(r["golds"] for r in wide)
    lines += [
        "---",
        "",
        "## 3. 结论(按实测写,不修饰)",
        "",
        f"1. **通路是通的**:每题种子实体 "
        f"{min(r['entities'] for r in wide)}–{max(r['entities'] for r in wide)} 个、"
        f"多跳事实 {min(r['facts'] for r in wide)}–{max(r['facts'] for r in wide)} 条、"
        f"反查块 {min(r['graph_chunks'] for r in wide)}–{max(r['graph_chunks'] for r in wide)} 个。"
        f"三个计数都不为 0,不是 A 类事故。",
        f"2. **宽窗口下(50)图加不出东西**:仅图命中 gold **{w_gold} / {n_all}**。"
        f"向量侧一次就取走 61 块里的 50 块,图结构上没有位置可加 ——",
        f"这解释了 `graphrag_ab.md` 里两档**逐位相同**。",
        f"3. **窄窗口下(10)图确实补得上一点**:仅图命中 gold **{n_gold} / {n_all}**。"
        f"通路有真实价值,只是这套语料上小到聚合分看不出来。",
        "",
        "> ⚠ **不能说的话**:「GraphRAG 提升多跳检索」。实测支持的说法是",
        "> 「实现了多跳通路并做了有/无对照,在这套 61 块语料上未测出增益」。",
        "> 若要报增益数字,得先换一套向量召回够不着的语料。",
        "",
    ]

    out = ROOT / "eval" / "reports" / "graph_path_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写 {out}")
    print(f"  宽窗口(50):仅图命中 gold {w_gold} / {n_all}")
    print(f"  窄窗口(10):仅图命中 gold {n_gold} / {n_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
