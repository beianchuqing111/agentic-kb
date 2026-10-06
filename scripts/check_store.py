"""存储层自检:对着真的 Qdrant 跑一遍建库→写入→混合检索→清理。

用独立 collection(agentic_kb_selftest),不碰正式库。

跑法:
    set PYTHONIOENCODING=utf-8
    python scripts\\check_store.py

重点验证:
  1. 单 collection 里稠密 + 稀疏双字段能不能建起来
  2. 服务端 RrfQuery(rrf=Rrf(k=...)) 到底收不收 k 参数
  3. **k=60 和 Qdrant 默认 k=2 排出来的名次是否真的不同** ——
     如果相同,说明 k 根本没生效,config 里那个 rrf_k 就是摆设
  4. 单路(仅稠密 / 仅稀疏)能不能自动退化,不套无意义的融合
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from embed import get_embedder  # noqa: E402
from store import Chunk, QdrantStore  # noqa: E402
from store.qdrant_store import point_id  # noqa: E402

TEST_COLLECTION = "agentic_kb_selftest"

DOCS = [
    ("d1", "变压器油温超过告警阈值时,应立即启动备用冷却系统,并通知运维人员到现场检查。"),
    ("d2", "冷却系统由油泵、散热器和风扇三部分组成,任一部件故障都会导致油温上升。"),
    ("d3", "绝缘子破损是输电线路最常见的缺陷,通常通过无人机航拍图像识别。"),
    ("d4", "无人机巡检航线规划需考虑地形、禁飞区和电池续航,一般单次覆盖不超过五公里。"),
    ("d5", "今天天气晴朗,气温适宜,适合户外作业。"),
]

QUERY = "变压器温度太高怎么办"


def main() -> int:
    s = get_settings()
    emb = get_embedder()

    # 用独立 collection,别把自检数据写进正式库
    cfg = dataclasses.replace(s.qdrant, collection=TEST_COLLECTION)
    store = QdrantStore(cfg=cfg, retrieval=s.retrieval)

    print("=" * 66)
    print(f"Qdrant: {cfg.url}")
    h = store.health()
    if not h["ok"]:
        print(f"❌ 连不上 Qdrant: {h['error']}")
        print("   先跑 scripts\\start_qdrant.bat")
        return 1
    print(f"健康检查: ok,现有 collection={h['collections']}")
    print("=" * 66)

    try:
        # --- 1. 建库 ---
        store.ensure_collection(dense_dim=emb.dense_dim, recreate=True)
        print(f"\n[1] 建库成功: {store.info()}")

        # 幂等性:同一 (doc_id, idx) 必须得到同一个 id
        assert point_id("d1", 0) == point_id("d1", 0), "point_id 不稳定"
        assert point_id("d1", 0) != point_id("d2", 0), "point_id 撞了"
        print(f"    point_id 确定性: d1#0={point_id('d1', 0)[:8]}… ✅")

        # --- 2. 写入 ---
        res = emb.encode([t for _, t in DOCS])
        chunks = [
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
        ]
        n = store.upsert(chunks)
        print(f"\n[2] 写入 {n} 条,库内 count={store.count()}")
        assert store.count() == len(DOCS)

        # 幂等:再写一遍,数量不该变
        store.upsert(chunks)
        assert store.count() == len(DOCS), "重复写入产生了重复点,幂等性被破坏"
        print("    重复 upsert 后 count 不变 ✅ (覆盖而非新增)")

        # --- 3. 增量去重 ---
        found = store.existing_hashes([f"hash-{d}" for d, _ in DOCS[:3]] + ["hash-nonexistent"])
        print(f"\n[3] existing_hashes: 问 4 个,命中 {len(found)} 个 (应为 3)")
        assert len(found) == 3, f"去重查询不准: {found}"

        # --- 4. 混合检索 ---
        q = emb.encode(QUERY)
        qd, qs = q.dense[0], q.sparse[0]
        print(f"\n[4] 查询: {QUERY!r}")

        hits = store.query_hybrid(qd, qs, limit=5)
        print(f"    RRF(k={s.retrieval.rrf_k}) 融合结果:")
        for pt in hits:
            print(f"      {pt.score:.6f}  {pt.payload['doc_id']}  "
                  f"{pt.payload['text'][:26]}…")

        # --- 5. k 到底生没生效 ---
        hits_k2 = store.query_hybrid(qd, qs, limit=5, rrf_k=2)
        order60 = [p.payload["doc_id"] for p in hits]
        order2 = [p.payload["doc_id"] for p in hits_k2]
        print(f"\n[5] k={s.retrieval.rrf_k} 顺序: {order60}")
        print(f"    k=2 (Qdrant 默认) 顺序: {order2}")
        if order60 == order2:
            print("    ⚠️  两者顺序相同 —— k 可能没传到服务端。"
                  "小样本下有可能真的相同,但值得警惕。")
        else:
            print("    ✅ 顺序不同,证明 k 确实作用到了服务端融合。")

        # --- 6. 单路退化 ---
        d_only = store.query_hybrid(qd, None, limit=3)
        s_only = store.query_hybrid(None, qs, limit=3)
        print(f"\n[6] 仅稠密 top3: {[p.payload['doc_id'] for p in d_only]}")
        print(f"    仅稀疏 top3: {[p.payload['doc_id'] for p in s_only]}")
        assert d_only and s_only, "单路查询返回空"
        assert len(d_only) == 3 and len(s_only) == 3

        # 两路的原始排名 —— 调试「是谁召回了正确答案」时唯一有用的东西
        dh, sh = store.query_paths(qd, qs)
        print(f"    原始稠密排名: {[p.payload['doc_id'] for p in dh]}")
        print(f"    原始稀疏排名: {[p.payload['doc_id'] for p in sh]}")
        print(f"    ↑ 融合后的第一名 {order60[0]},看它是不是两路都进了前列")

        # --- 7. 按文档删除 ---
        store.delete_by_doc("d5")
        after = store.count()
        print(f"\n[7] 删除 d5 后 count={after} (应为 {len(DOCS) - 1})")
        assert after == len(DOCS) - 1, "按文档删除没生效"

        docs = store.list_docs()
        print(f"    剩余文档: {[d['doc_id'] for d in docs]}")

        print("\n全部通过 ✅")
        return 0

    finally:
        # 自检不留垃圾
        try:
            store.client.delete_collection(TEST_COLLECTION)
            print(f"\n(已清理测试 collection {TEST_COLLECTION})")
        except Exception as exc:  # noqa: BLE001
            print(f"\n(清理失败: {exc})")
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
