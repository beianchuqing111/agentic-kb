"""GraphRAG 自检:图写入 → 实体向量 → 多跳 → 反查块 → 与向量候选合并重排。

**不需要 LLM**:三元组抽取用桩函数喂进去,块向量和重排都是本地模型。
所以这条链路现在就能验,不必等 API key。桩函数刻意照抄上游
`SimpleLLMPathExtractor` 的行为(把块的整个 metadata 拷进实体属性),
这样「属性被剥干净」这件事每次都会被重新验证一遍。

Qdrant 用独立 collection(agentic_kb_selftest_graph),不碰正式库。
Neo4j Community 只有一个库,没法隔离 —— 所以这里只用 `selftest_` 前缀的
实体名和 doc_id,退出时按 doc_id 精确删掉,绝不 clear 整张图。

跑法:
    set PYTHONIOENCODING=utf-8
    python scripts\\check_graphrag.py

重点验证:
  1. 块上的 doc_id 真的落对了(不是字符串 "None")—— 增量重导的前提
  2. 每个「块提到实体」都有 MENTIONS 边,不是每个实体只连一块
  3. 实体属性干净:只剩 triplet_source_id,没有 doc_id/路径那堆污染
  4. 图能把**正文里没有查询词**的块捞出来(这是图检索唯一不可替代的价值)
  5. 两路候选合并后只重排一次,且图那一路的分数不会被当成相似度
  6. 事实能挂到块上(meta['facts']),上层能看见「为什么这些块被拉进来」
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from embed import get_embedder  # noqa: E402
from ingest import GraphIngestPipeline, stable_doc_id  # noqa: E402
from retrieve.backends import GraphRAGBackend  # noqa: E402
from retrieve.hybrid import RetrievalDebug  # noqa: E402
from store import QdrantStore, get_graph_store  # noqa: E402

TEST_COLLECTION = "agentic_kb_selftest_graph"
DOC_PREFIX = "selftest_graph"

# 语料设计的关键在 b 块:**正文里一个查询词都没有**(不提绝缘子,也没有
# 「缺陷/问题/故障」),它只能靠「横担」这个实体从图上转两跳被捞出来。
# 这正是图检索存在的理由 —— 向量检索对它是够不着的。
DOCS = [
    (
        "a",
        "测试杆塔位于测试A线上,是本次巡检的重点。测试绝缘子安装在横担上,"
        "属于关键受力部件,需要重点检查。",
        [
            ("测试杆塔", "位于", "测试A线"),
            ("测试绝缘子", "位于", "横担"),
        ],
    ),
    (
        "b",
        "瓷套出现裂纹时必须立即更换,否则在风振下可能断裂。判据来自运维规程,"
        "现场用卡尺测量裂纹深度。",
        [("瓷套", "存在缺陷", "裂纹"), ("横担", "承载", "瓷套")],
    ),
]
# 图上的实体连接:测试绝缘子 -位于-> 横担 <-承载- 瓷套 -存在缺陷-> 裂纹
# 查询从「测试绝缘子」出发,2 跳就能走到 瓷套/裂纹,从而反查到 b 块。

QUERY = "测试绝缘子有什么问题"


class C:
    """假的 chunk 对象,只需要 index/text/section 三个字段。"""

    def __init__(self, index, text):
        self.index, self.text, self.section = index, text, ""


class StubExtractor:
    """照上游 SimpleLLMPathExtractor 的行为:把块的整个 metadata 拷进
    实体/关系的 properties。污染就是这么来的,正好验证 _collect 剥得掉。"""

    def __init__(self, triples):
        self.triples = triples  # (doc_id, chunk_index) -> [(头, 关系, 尾)]

    def __call__(self, nodes, show_progress=False):
        from llama_index.core.graph_stores.types import (
            KG_NODES_KEY,
            KG_RELATIONS_KEY,
            EntityNode,
            Relation,
        )

        for n in nodes:
            md = n.metadata.copy()
            key = (md.get("doc_id"), md.get("chunk_index"))
            kn, kr = [], []
            for subj, rel, obj in self.triples.get(key, []):
                sn = EntityNode(name=subj, properties=md)
                on = EntityNode(name=obj, properties=md)
                kn += [sn, on]
                kr.append(
                    Relation(
                        label=rel, source_id=sn.id, target_id=on.id, properties=md
                    )
                )
            n.metadata[KG_NODES_KEY] = kn
            n.metadata[KG_RELATIONS_KEY] = kr
        return nodes


def main() -> int:
    s = get_settings()
    emb = get_embedder()
    failures: list[str] = []

    qcfg = dataclasses.replace(s.qdrant, collection=TEST_COLLECTION)
    store = QdrantStore(cfg=qcfg, retrieval=s.retrieval)
    g = get_graph_store()

    print("=" * 72)
    print(f"Qdrant: {qcfg.url} / {TEST_COLLECTION}")
    print(f"Neo4j : {s.neo4j.uri} / {s.neo4j.database}   (只写 {DOC_PREFIX}_* ,退出时精确删除)")
    print(f"重排: {s.retrieval.rerank_enabled}  阈值: {s.retrieval.rerank_min_score}  "
          f"图: top_k={s.graphrag.vector_top_k} hops={s.graphrag.max_hops}")
    h, gh = store.health(), g.health()
    if not h["ok"]:
        print(f"❌ 连不上 Qdrant: {h['error']}\n   先跑 scripts\\start_qdrant.bat")
        return 1
    if not gh["ok"]:
        print(f"❌ 连不上 Neo4j: {gh['error']}\n   先跑 scripts\\start_neo4j.bat")
        return 1
    print("=" * 72)

    doc_ids = {name: stable_doc_id(f"{DOC_PREFIX}://{name}") for name, _, _ in DOCS}
    triples = {
        (doc_ids[name], 0): [(a, r, b) for a, r, b in tl]
        for name, _, tl in DOCS
    }

    pipeline = GraphIngestPipeline(store=store, graph=g, cfg=s.ingest)
    pipeline._extractor = StubExtractor(triples)
    backend = GraphRAGBackend(graph=g, store=store, pipeline=pipeline)

    try:
        # ---------------- 1. 写入 ----------------
        print("\n[1] 写入两篇文档(块 → Qdrant + Neo4j,实体 → 图)")
        store.ensure_collection(dense_dim=emb.dense_dim, recreate=True)
        g.ensure_indexes(dense_dim=emb.dense_dim)
        for name, text, _ in DOCS:
            st = pipeline.ingest_text(text, source=f"{DOC_PREFIX}://{name}", title=f"自检文档{name}")
            print(f"    {name}: {st.summary()}")
            if st.docs_indexed != 1 or st.chunks_written != 1:
                failures.append(f"{name}: 期望 1 篇 1 块,实得 {st.docs_indexed}/{st.chunks_written}")

        gs = g.stats()
        print(f"    图: {gs}")
        print(f"    Qdrant: count={store.count()}")

        # ---------------- 2. doc_id 与 MENTIONS ----------------
        print("\n[2] 块的 doc_id / MENTIONS 边")
        for name, did in doc_ids.items():
            n = g.doc_chunk_count(did)
            print(f"    {name}: 图上有 {n} 块")
            if n != 1:
                failures.append(f"{name}: doc_chunk_count={n},应为 1(doc_id 没落对?)")

        rows = g.run(
            "MATCH (c:__Node__)-[:MENTIONS]->(e:__Entity__) "
            "WHERE e.id STARTS WITH '测试' OR e.id IN ['横担','瓷套','裂纹'] "
            "RETURN c.doc_id AS d, count(DISTINCT e.id) AS n ORDER BY d"
        )
        for r in rows:
            print(f"    {r['d']}: 提到 {r['n']} 个实体")
        total_m = sum(r["n"] for r in rows)
        # a 块:测试杆塔/测试A线/测试绝缘子/横担 = 4;b 块:瓷套/裂纹/横担 = 3
        if total_m != 7:
            failures.append(f"MENTIONS 对数={total_m},应为 7(每块提到的每个实体都要连边)")

        # ---------------- 3. 实体属性干净 ----------------
        print("\n[3] 实体属性(应只剩 triplet_source_id,没有 doc_id/路径污染)")
        ents = g.run(
            "MATCH (e:__Entity__) WHERE e.id STARTS WITH '测试' "
            "RETURN e.id AS id, keys(e) AS k ORDER BY id"
        )
        for r in ents:
            keys = sorted(x for x in r["k"] if not x.startswith("_"))
            print(f"    {r['id']} -> {keys}")
            extra = set(keys) - {"id", "name", "embedding", "triplet_source_id"}
            if extra:
                failures.append(f"实体 {r['id']} 属性有污染: {sorted(extra)}")

        # ---------------- 4. 图这一步单独看 ----------------
        print(f"\n[4] 图这一步直接调(查询:{QUERY!r})")
        facts, ent_ids, seed = backend._graph_context(QUERY)
        print(f"    种子实体(向量命中): {sorted(seed, key=lambda x: -seed[x])}")
        print(f"    展开后实体 {len(ent_ids)} 个,事实 {len(facts)} 条:")
        for f in facts:
            print(f"      {f['head']} --{f['relation']}--> {f['tail']}  (score={f['score']:.4f})")
        if len(facts) < 2:
            failures.append(f"多跳只拿到 {len(facts)} 条事实,期望 ≥2(测试绝缘子→横担→瓷套/裂纹)")
        if not any(f["head"] == "测试绝缘子" for f in facts):
            failures.append("多跳没从「测试绝缘子」展开出任何事实")

        graph_chunks = backend._chunks_mentioning(ent_ids)
        got = [rc.doc_id for rc, _, _ in graph_chunks]
        print(f"    反查到的块: {[d[:12] + '…' for d in got]}")
        b_hit = doc_ids["b"] in got
        print(f"    {'✅' if b_hit else '❌'} b 块(正文里没有查询词)被图捞到了")
        if not b_hit:
            failures.append("图反查没能捞到 b 块 —— 图检索等于没用")

        # --------- 5. 向量那一路自己能不能捞到 b 块 ---------
        print("\n[5] 对照:纯向量召回(不重排)能不能捞到 b 块")
        vec = backend.retriever.retrieve(QUERY, top_k=s.retrieval.fusion_top_k, use_rerank=False)
        vec_ids = [r.doc_id for r in vec]
        print(f"    向量候选 {len(vec_ids)} 条: {[d[:12] + '…' for d in vec_ids]}")
        if doc_ids["b"] not in vec_ids:
            print("    ✅ b 块不在向量候选里 —— 这一条完全是图的贡献")
        else:
            print("    (b 块向量也召回了,说明这组语料不够刁钻;机制本身没问题)")

        # ---------------- 6. 端到端 ----------------
        print(f"\n[6] 端到端 backend.retrieve()")
        dbg = RetrievalDebug()
        hits = backend.retrieve(QUERY, debug=dbg)
        print(f"    漏斗: 实体 {dbg.dense_hits} 个 / 事实 {dbg.sparse_hits} 条 "
              f"→ 重排 {dbg.reranked_hits} → 阈值砍 {dbg.dropped_by_threshold} → 最终 {len(hits)}")
        if not hits:
            failures.append("retrieve() 返回空")
        for r in hits:
            tag = "图" if r.meta.get("from_graph") else "向量"
            print(f"      [{tag}] rerank={r.rerank_score:.4f}  {r.citation}  {r.text[:22]}…")
            if r.meta.get("entities"):
                print(f"            命中实体: {r.meta['entities']}")
            for f in r.meta.get("facts") or []:
                print(f"            事实: {f['head']} --{f['relation']}--> {f['tail']}")

        if not any(r.meta.get("facts") for r in hits):
            failures.append("没有任何块挂上 facts —— _attach_facts 没生效")
        if not store.exists() or store.count() != 2:
            failures.append(f"Qdrant 里应有 2 块,实得 {store.count()}")

        # 空查询
        if backend.retrieve("") != []:
            failures.append("空查询应返回 []")

        print(f"\n{'=' * 72}")
        if failures:
            print("失败项:")
            for f in failures:
                print(f"  ❌ {f}")
            return 1
        print("全部通过 ✅")
        return 0

    finally:
        # 只按 doc_id 精确删,不动别人的数据
        try:
            for name, did in doc_ids.items():
                r = g.delete_doc_graph(did)
                print(f"\n(清理 {name}: {r})")
            left = {n: g.doc_chunk_count(d) for n, d in doc_ids.items()}
            print(f"(清理后残留块数: {left})")
            if any(left.values()):
                print("⚠️  有残留 —— delete_doc_graph 没删干净")
        except Exception as exc:  # noqa: BLE001
            print(f"\n(图清理失败: {exc})")
        try:
            store.client.delete_collection(TEST_COLLECTION)
            print(f"(已删测试 collection {TEST_COLLECTION})")
        except Exception as exc:  # noqa: BLE001
            print(f"(collection 清理失败: {exc})")
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
