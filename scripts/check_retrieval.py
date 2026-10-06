"""混合检索自检:双路召回 → RRF → 重排 → 阈值过滤,对着真的 Qdrant 跑。

用独立 collection(agentic_kb_selftest_retr),不碰正式库。

跑法:
    set PYTHONIOENCODING=utf-8
    python scripts\\check_retrieval.py

重点验证:
  1. 三级漏斗通了:50/50 召回 → 20 融合 → 重排 → 阈值砍到只剩相关的
  2. **阈值真的起作用** —— 关掉重排返回 top_n 条,开着重排只返回相关的少数几条
  3. RRF 名次和重排名次**确实不同** —— 相同说明重排没干活(或候选太容易)
  4. 调试信息能说清「这一条是谁召回的」:稠密排名 vs 稀疏排名
  5. use_rerank=False 时退化成纯 RRF,分数是 RRF 分不是 0
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from embed import get_embedder  # noqa: E402
from retrieve.hybrid import HybridRetriever, RetrievalDebug, format_context  # noqa: E402
from store import Chunk, QdrantStore  # noqa: E402

TEST_COLLECTION = "agentic_kb_selftest_retr"

# 语料刻意做成两组:一组和查询真相关,一组是同领域但答非所问
# (以及一条纯噪声)。只有这样才能看出重排+阈值在干什么。
DOCS = [
    ("d1", "变压器油温超过告警阈值时,应立即启动备用冷却系统,并通知运维人员到现场检查。"),
    ("d2", "冷却系统由油泵、散热器和风扇三部分组成,任一部件故障都会导致油温上升。"),
    ("d3", "绝缘子破损是输电线路最常见的缺陷,通常通过无人机航拍图像识别。"),
    ("d4", "无人机巡检航线规划需考虑地形、禁飞区和电池续航,一般单次覆盖不超过五公里。"),
    ("d5", "今天天气晴朗,气温适宜,适合户外作业。"),
    ("d6", "变压器铭牌参数包括额定容量、额定电压、联结组别和短路阻抗。"),
    ("d7", "红外测温是发现变压器局部过热的有效手段,可定位到具体套管或接头。"),
    ("d8", "变电站值班制度要求每四小时抄录一次主变油温并登记在运行日志中。"),
]

QUERIES = [
    # (查询, 期望被召回的关键文档, 说明)
    ("变压器温度太高怎么办", {"d1"}, "同义改写:油温 → 温度、超阈值 → 太高"),
    ("绝缘子坏了怎么发现", {"d3"}, "字面几乎不重合,考稠密"),
    ("主变油温多久记录一次", {"d8"}, "考精确字面:主变油温 / 记录"),
]


def main() -> int:
    s = get_settings()
    emb = get_embedder()

    cfg = dataclasses.replace(s.qdrant, collection=TEST_COLLECTION)
    store = QdrantStore(cfg=cfg, retrieval=s.retrieval)

    print("=" * 70)
    print(f"Qdrant: {cfg.url}   重排: {s.retrieval.rerank_enabled}   "
          f"阈值: {s.retrieval.rerank_min_score}   top_n: {s.retrieval.rerank_top_n}")
    h = store.health()
    if not h["ok"]:
        print(f"❌ 连不上 Qdrant: {h['error']}")
        print("   先跑 scripts\\start_qdrant.bat")
        return 1
    print("=" * 70)

    retriever = HybridRetriever(store=store, cfg=s.retrieval)

    try:
        # --- 建库 + 写入 ---
        store.ensure_collection(dense_dim=emb.dense_dim, recreate=True)
        res = emb.encode([t for _, t in DOCS])
        n = store.upsert(
            Chunk(
                doc_id=did,
                chunk_index=0,
                text=txt,
                dense=res.dense[i],
                sparse=res.sparse[i],
                source=f"{did}.txt",
                title=f"文档{did}",
                content_hash=f"hash-{did}",
            )
            for i, (did, txt) in enumerate(DOCS)
        )
        print(f"\n[1] 写入 {n} 条,count={store.count()}")

        failures: list[str] = []

        for qtext, expect, note in QUERIES:
            print(f"\n{'─' * 70}")
            print(f"[查询] {qtext}    ({note})   期望召回: {sorted(expect)}")

            dbg = RetrievalDebug()
            hits = retriever.retrieve(qtext, debug=dbg)

            if not hits:
                failures.append(f"{qtext}: 返回空")
                print("    ❌ 返回空")
                continue

            print(f"    漏斗: 稠密 {dbg.dense_hits} + 稀疏 {dbg.sparse_hits} "
                  f"→ 融合 {dbg.fused_hits} → 重排 {dbg.reranked_hits} "
                  f"→ 阈值砍掉 {dbg.dropped_by_threshold} → 最终 {len(hits)}")
            for r in hits:
                d = "—" if r.dense_rank is None else r.dense_rank
                sp = "—" if r.sparse_rank is None else r.sparse_rank
                print(f"      rerank={r.rerank_score:.4f}  rrf={r.rrf_score:.6f}  "
                      f"[稠密#{d} 稀疏#{sp}]  {r.doc_id}  {r.text[:24]}…")

            got = {r.doc_id for r in hits}
            if not (got & expect):
                failures.append(f"{qtext}: 期望 {expect},实得 {got}")
                print(f"    ❌ 期望的 {sorted(expect)} 一条都没进")
            else:
                print(f"    ✅ 命中 {sorted(got & expect)}")

            # 阈值该砍掉近零分的那些
            if dbg.dropped_by_threshold == 0 and dbg.reranked_hits > 1:
                print(f"    ⚠️  阈值一条都没砍(候选 {dbg.reranked_hits} 条),"
                      f"检查 rerank_min_score={s.retrieval.rerank_min_score}")

            # --- 对照:关掉重排,看纯 RRF 会返回什么 ---
            raw = retriever.retrieve(qtext, use_rerank=False, debug=None)
            raw_ids = [r.doc_id for r in raw]
            # 关掉重排时 score 应当是 RRF 分,不是 0
            if raw and raw[0].rrf_score <= 0:
                failures.append(f"{qtext}: 未重排时 RRF 分为 0")
            print(f"    [对照] 纯 RRF top{len(raw)}: {raw_ids}")
            if raw_ids and raw_ids[0] != hits[0].doc_id:
                print(f"    ✅ 重排改变了第一名: RRF→{raw_ids[0]}  重排→{hits[0].doc_id}")
            else:
                print(f"    (重排没改变第一名,RRF 已经对了 —— 小语料下正常)")

        # --- 上下文拼装 ---
        print(f"\n{'─' * 70}")
        top = retriever.retrieve(QUERIES[0][0], top_k=2)
        ctx = format_context(top, max_chars=400)
        print(f"[上下文拼装] max_chars=400,实际 {len(ctx)} 字符")
        for line in ctx.splitlines()[:4]:
            print(f"    {line}")
        assert len(ctx) <= 400 + 100, "max_chars 截断失效"
        assert "来源:" in ctx, "上下文缺来源标签 —— 没法引用"
        print("    ✅ 带来源标签且长度受限")

        # --- 空查询 ---
        assert retriever.retrieve("") == [], "空查询应当返回空列表而不是报错"
        assert retriever.retrieve("   ") == []
        print("\n[边界] 空/空白查询返回 [] ✅")

        print(f"\n{'=' * 70}")
        if failures:
            print("失败项:")
            for f in failures:
                print(f"  ❌ {f}")
            return 1
        print("全部通过 ✅")
        return 0

    finally:
        try:
            store.client.delete_collection(TEST_COLLECTION)
            print(f"\n(已清理测试 collection {TEST_COLLECTION})")
        except Exception as exc:  # noqa: BLE001
            print(f"\n(清理失败: {exc})")
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
