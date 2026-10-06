"""量 Contextual Retrieval 的**入库成本**,和检索增益配对使用。

为什么要单独量
------------
「CT 值不值」这个问题只回答一半是没用的:`eval_run.py` 出的是**增益**,
而 CT 的钱花在**导入**那一步(每块一次 LLM 调用),检索那一步一分不多花。
所以增益表和成本表必须一起看,否则「CT 提升了 x」这个结论是悬空的。

这个脚本只做计量,**不写任何存储**:
- 不连 Qdrant、不 upsert、不动现有 collection
  (直接 reingest 会重写 collection,定位语措辞每次略有不同,
   会把刚跑出来的检索基线弄脏 —— 计量不该有这种副作用)
- 分块走 `chunk_text`,和 `ingest/pipeline.py:prepare_document` 完全同参,
  所以块数、块边界和真实导入一致

量三样:
1. **LLM**:调用次数 / prompt / completion / reasoning token / 墙钟
2. **嵌入**:拼了定位语 vs 不拼,编码耗时差(本地 GPU,不花钱但占时间)
3. **体积**:embed_text 字符数涨幅 —— 定位语把每块喂给编码器的文本撑大多少

成本口径说明:定位语按文档分组、块级并发(`contextual_workers`),
墙钟是并发下的真实耗时;token 数与并发无关,是账单口径。

两个量成本的坑(都踩过,写在代码里免得下次再踩)
------------------------------------------------
一、**测嵌入前必须预热**。第一次 `encode` 会把 bge-m3 载进显存,那几秒是
   加载不是编码。先量的那一侧会平白多背一次加载 —— 不预热就会得出
   「有定位语反而快 5.4 秒」这种一眼假的结论(实测踩到过)。
二、**别为了重量成本重跑 LLM**。定位语文本随跑随丢的话,想复核嵌入耗时
   就得再调 61 次 LLM。所以块文本和定位语都落盘,配合 `--embed-only`
   可以只重量嵌入。

用法:
    python scripts/ct_cost.py               # 全量:调 LLM + 量嵌入
    python scripts/ct_cost.py --embed-only  # 只重量嵌入,复用已落盘文本,不再调 LLM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from embed.bge_m3 import get_embedder  # noqa: E402
from ingest.chunker import chunk_text  # noqa: E402
from ingest.contextual import enrich_chunks  # noqa: E402
from ingest.loader import discover, load_many  # noqa: E402
from llm.client import get_llm  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "eval" / "reports" / "ct_cost.json"


def _measure_embed(plain_all: list[str], ctx_all: list[str]) -> dict:
    """嵌入耗时差。**两侧都在预热之后量**,顺序不构成偏差。"""
    embedder = get_embedder()
    n = max(1, len(plain_all))
    bs = min(get_settings().embed.batch_size, n)

    # 预热:见模块开头「坑一」。这一条不预热,结论就是假的。
    embedder.encode(["预热"], batch_size=1)

    texts_ctx = [f"{c}\n{t}" if c else t for c, t in zip(ctx_all, plain_all)]

    ts = time.perf_counter()
    embedder.encode(plain_all, batch_size=bs)
    t_plain = time.perf_counter() - ts

    ts = time.perf_counter()
    embedder.encode(texts_ctx, batch_size=bs)
    t_ctx = time.perf_counter() - ts

    chars_plain = sum(len(t) for t in plain_all)
    chars_ctx = sum(len(t) for t in texts_ctx)

    print()
    print("── 嵌入(本地 GPU,不花钱但占时间;两侧均已预热) ──")
    print(f"  无定位语  {t_plain:.3f}s   {chars_plain} 字")
    print(f"  有定位语  {t_ctx:.3f}s   {chars_ctx} 字  "
          f"(+{chars_ctx - chars_plain} 字, +{(chars_ctx / chars_plain - 1) * 100:.1f}%)")
    print(f"  差值      {t_ctx - t_plain:+.3f}s  ({(t_ctx / t_plain - 1) * 100:+.1f}%)")

    return {
        "seconds_plain": round(t_plain, 3),
        "seconds_ctx": round(t_ctx, 3),
        "delta_seconds": round(t_ctx - t_plain, 3),
        "chars_plain": chars_plain,
        "chars_ctx": chars_ctx,
        "chars_growth_pct": round((chars_ctx / chars_plain - 1) * 100, 2),
    }


def _load_existing() -> dict | None:
    if not OUT.exists():
        return None
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="量 Contextual Retrieval 的入库成本")
    ap.add_argument("--embed-only", action="store_true",
                    help="只重量嵌入耗时(复用已落盘的块文本,不再调 LLM)")
    args = ap.parse_args(argv)

    # ---------------- 只重量嵌入 ----------------
    if args.embed_only:
        prev = _load_existing()
        if not prev:
            print(f"!! 没有可复用的计量结果:{OUT}\n   先跑一次全量:python scripts/ct_cost.py")
            return 1
        plain_all = prev.get("plain_texts")
        ctx_all = prev.get("ctx_texts")
        if not plain_all or not ctx_all:
            print("!! 已存的计量结果里没有块文本(旧版本产物)。\n"
                  "   跑一次全量把文本落盘,之后就能只重量嵌入了。")
            return 1
        prev["embed"] = _measure_embed(plain_all, ctx_all)
        OUT.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"已更新 {OUT}")
        return 0

    # ---------------- 全量 ----------------
    cfg = get_settings().ingest
    if not cfg.contextual_enabled:
        print("!! .env 里定位语是关的,量不了。先打开 contextual_enabled 再跑。")
        return 1

    llm = get_llm()
    if not llm.configured:
        print("!! LLM 未配置(.env 缺 LLM_API_KEY),量不了。")
        return 1

    root = Path(__file__).resolve().parents[1] / "eval" / "corpus" / "seed"
    docs, errors = load_many(discover(root, recursive=True), on_error="raise")
    if errors:
        print("!! 读取语料出错:", errors)
        return 1

    print(f"语料 {root}")
    print(f"文档 {len(docs)} 篇")
    print(f"模型 {llm.cfg.model}   并发 contextual_workers={cfg.contextual_workers}")
    print(f"定位语上限 {cfg.contextual_max_chars} 字")
    print()

    usage0 = llm.usage
    t0 = time.perf_counter()

    per_doc: list[dict] = []
    ctx_all: list[str] = []
    plain_all: list[str] = []

    for d in docs:
        chunks = chunk_text(d.text, cfg.chunk_size, cfg.chunk_overlap)
        plain_all.extend(c.text for c in chunks)

        ts = time.perf_counter()
        ctxs = enrich_chunks(d.text, chunks, cfg, llm)
        dt = time.perf_counter() - ts

        ctx_all.extend(ctxs)
        per_doc.append({
            "source": d.source,
            "n_chunks": len(chunks),
            "n_ctx": sum(1 for c in ctxs if c),
            "ctx_chars": sum(len(c) for c in ctxs),
            "seconds": round(dt, 3),
        })
        print(f"  {d.source:<44} {len(chunks):>3} 块  "
              f"{sum(len(c) for c in ctxs):>5} 字定位语  {dt:>6.2f}s")

    wall = time.perf_counter() - t0
    usage1 = llm.usage

    d_prompt = usage1["prompt"] - usage0["prompt"]
    d_comp = usage1["completion"] - usage0["completion"]
    d_reason = usage1["reasoning"] - usage0["reasoning"]
    d_calls = usage1["calls"] - usage0["calls"]
    n_chunks = len(plain_all)

    print()
    print("── LLM(花真钱的那一半) ──")
    print(f"  调用次数        {d_calls}   (块数 {n_chunks})")
    print(f"  prompt tokens   {d_prompt}")
    print(f"  completion      {d_comp}")
    print(f"  reasoning       {d_reason}")
    print(f"  合计 tokens     {d_prompt + d_comp}")
    print(f"  墙钟            {wall:.2f}s  (并发 {cfg.contextual_workers})")
    if d_calls:
        print(f"  每块均值        prompt {d_prompt / d_calls:.0f} / "
              f"completion {d_comp / d_calls:.0f} / {wall / d_calls:.2f}s")

    embed = _measure_embed(plain_all, ctx_all)

    payload = {
        "corpus": str(root),
        "n_docs": len(docs),
        "n_chunks": n_chunks,
        "model": llm.cfg.model,
        "contextual_workers": cfg.contextual_workers,
        "contextual_max_chars": cfg.contextual_max_chars,
        "llm": {
            "calls": d_calls,
            "prompt_tokens": d_prompt,
            "completion_tokens": d_comp,
            "reasoning_tokens": d_reason,
            "total_tokens": d_prompt + d_comp,
            "wall_seconds": round(wall, 3),
        },
        "embed": embed,
        "per_doc": per_doc,
        # 落盘块文本与定位语,供 --embed-only 复核嵌入耗时(见模块开头「坑二」)
        "plain_texts": plain_all,
        "ctx_texts": ctx_all,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"已写 {OUT}")
    print("  (块文本与定位语已一并落盘,之后可用 --embed-only 只重量嵌入)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
