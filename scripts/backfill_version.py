"""给加 `status` 字段**之前**入库的 point 补上 `status = current`。

为什么要回填,而不是靠"缺字段当有效"
------------------------------------
召回侧的过滤条件(见 `store/versioning.py`)已经写成
`status == 'current' OR status 不存在`,所以**不回填也不会查不到** ——
但那是个兜底分支,不是常态:

  1. 它让每次查询都要多带一个 OR 条件,而索引对 `IsEmptyCondition`
     的帮助有限,这部分是全扫。
  2. 更要命的是它把两种数据永远混在一起:真正"新入库、显式写了 current"
     的和"老数据、碰巧没这个字段"的,查询上完全等价。以后要是有人收紧
     过滤条件(比如把 OR 去掉),老数据会**静默消失** —— 那种事故查起来
     要命,因为它表现得像"检索坏了",而不是"少了个字段"。

回填是**就地改 payload**,不重算嵌入 —— 不需要 bge-m3,不需要 LLM,
不改任何一块正文,只在 payload 上加一个键。跑一次几秒钟。

用法
----
    python scripts/backfill_version.py            # 看有多少要补(不改)
    python scripts/backfill_version.py --apply    # 真改

**默认是 dry-run**。这个脚本动的是整个 collection 的数据,而"跑一下看看"
是人的本能 —— 默认改成写入的话,迟早有人在没备份的库上跑它。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import models as qm  # noqa: E402

from config import get_settings  # noqa: E402
from store.qdrant_store import QdrantStore, get_store  # noqa: E402
from store.versioning import STATUS_CURRENT, STATUS_FIELD  # noqa: E402


def count_missing(store) -> int:
    return store.client.count(
        collection_name=store.cfg.collection,
        count_filter=qm.Filter(
            must=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key=STATUS_FIELD))]
        ),
        exact=True,
    ).count


def count_total(store) -> int:
    return store.client.count(collection_name=store.cfg.collection, exact=True).count


def main() -> int:
    ap = argparse.ArgumentParser(description="回填 status 字段")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="真的写入。不给这个参数只统计,不改任何数据",
    )
    ap.add_argument(
        "--collection",
        default="",
        help="覆盖 collection 名(默认用 .env 里的 QDRANT_COLLECTION)",
    )
    args = ap.parse_args()

    cfg = get_settings()
    if args.collection:
        # `--collection` 存在的理由:评测用的那几个 collection
        # (`agentic_kb_eval_*`)平时不在 `.env` 里,要回填得能指名字。
        import dataclasses

        cfg = dataclasses.replace(
            cfg, qdrant=dataclasses.replace(cfg.qdrant, collection=args.collection)
        )
        store = QdrantStore(cfg.qdrant, cfg.retrieval)
    else:
        store = get_store()

    name = store.cfg.collection
    print(f"collection: {name}")

    if not store.exists():
        print("不存在。没有要回填的东西。")
        return 0

    total = count_total(store)
    missing = count_missing(store)
    have = total - missing
    print(f"总块数 {total}:已有 {STATUS_FIELD} 的 {have} 块,缺的 {missing} 块")

    if not args.apply:
        print("\n(dry-run,未改动任何数据。加 --apply 真写)")
        return 0

    # 索引先补上。回填给每个点写上 status,而索引是**建索引那一刻**对已有
    # 数据生效的 —— 先写数据再建索引,那批数据不一定会被索引进覆盖,
    # 结果就是"字段有了、过滤还是全扫"。顺序反过来没有这个问题。
    store.ensure_payload_indexes()
    print(f"payload 索引已确保(含 {STATUS_FIELD})")

    if missing == 0:
        print("没有要补的,收工。")
        return 0

    # 一次 filter 改完,不走 scroll + 逐条 —— 回填写的值是**同一个常量**,
    # 不依赖任何逐点的信息,分批反而多出"改到一半断了"的中间态。
    # ⚠️ 这跟 `QdrantStore.set_payload` 里刻意用 PointIdsList 的理由不冲突:
    # 那里改的是"我刚看过的那几块",中途有新块落进来就不能连它一起改;
    # 这里要改的**就是**"所有还没有这个字段的块",判据本身就是 filter。
    store.client.set_payload(
        collection_name=name,
        payload={STATUS_FIELD: STATUS_CURRENT},
        points=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key=STATUS_FIELD))]
            )
        ),
        wait=True,
    )

    left = count_missing(store)
    print(f"回填完成:改了 {missing - left} 块,还剩 {left} 块没有 {STATUS_FIELD}")
    if left:
        print("⚠️ 还有没补上的 —— 别当成功,查一下写入是不是被限流或中断了。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
