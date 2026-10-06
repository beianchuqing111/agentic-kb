"""嵌入层自检:不连任何外部服务,只验 bge-m3 这一层。

跑法(py310):
    set PYTHONIOENCODING=utf-8
    python scripts\\check_embed.py

检查项:
  1. embed 包能导入(等价于 embed/__init__.py 的导出都真实存在)
  2. 稠密向量形状 / 是否归一化 / 有没有 nan
  3. 稀疏权重的键是不是数字字符串 —— 不是的话 to_sparse_vector 会静默丢光
  4. 中文稀疏命中的到底是哪些 token(肉眼确认分词合理,不是按字切碎)
  5. 归一化后点积 == 余弦相似度
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能从任意 cwd 跑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from embed import get_embedder, to_sparse_vector  # noqa: E402

TEXTS = [
    "变压器的油温超过了告警阈值,需要立即检查冷却系统。",
    "输电线路巡检中,绝缘子破损是最常见的缺陷类型之一。",
    "今天天气不错,适合出门散步。",  # 故意放一条无关的,看稀疏权重能不能区分开
]


def main() -> int:
    print("=" * 62)
    emb = get_embedder()
    print(f"模型: {emb.cfg.model_name}  device={emb.cfg.device}  fp16={emb.cfg.use_fp16}")
    print("=" * 62)

    res = emb.encode(TEXTS)
    print(f"\n[1] 编码条数: {len(res)}  稠密形状: {res.dense.shape}")
    assert res.dense.shape == (len(TEXTS), 1024), "稠密维度不是 1024"

    norms = np.linalg.norm(res.dense, axis=1)
    print(f"[2] 每行 L2 范数: {np.round(norms, 6).tolist()}")
    assert np.allclose(norms, 1.0, atol=1e-4), "稠密向量没有归一化"
    assert not np.isnan(res.dense).any(), "稠密向量里出现 nan"
    assert (res.dense != 0).any(axis=1).all(), "有整行全零 —— 空文本或编码失败"

    d = res.dense
    dot = float(d[0] @ d[1])
    cos = float(d[0] @ d[1] / (np.linalg.norm(d[0]) * np.linalg.norm(d[1])))
    print(f"[3] 归一化后 点积={dot:.6f} 余弦={cos:.6f}  (应相等)")
    assert abs(dot - cos) < 1e-5, "点积与余弦不等,归一化有问题"

    # --- 稀疏 ---
    w0 = res.sparse[0]
    print(f"\n[4] 第 0 条稀疏权重个数: {len(w0)}")
    assert w0, "稀疏权重是空的"

    bad_keys = [k for k in w0 if not str(k).isdigit()]
    print(f"    非数字键: {bad_keys if bad_keys else '无 ✅'}")
    assert not bad_keys, "存在非数字键,to_sparse_vector 会把它们全部丢弃"

    sv = to_sparse_vector(w0)
    print(f"    -> SparseVector: {len(sv.indices)} 个索引, "
          f"类型={type(sv.indices[0]).__name__}, 前 5 个={sv.indices[:5]}")
    assert isinstance(sv.indices[0], int), "索引不是 int,Qdrant 会拒收"
    assert len(sv.indices) == len(sv.values)
    assert len(sv.indices) == len(w0), f"转换丢了权重: {len(sv.indices)} != {len(w0)}"

    # 把 token id 解回文字 —— 肉眼确认中文不是被按单字切碎
    try:
        tok = emb.model.tokenizer
        top = sorted(w0.items(), key=lambda kv: -kv[1])[:12]
        words = tok.convert_ids_to_tokens([int(k) for k, _ in top])
        print("\n[5] 权重最高的 12 个 token:")
        for (tid, wt), word in zip(top, words):
            print(f"      {tid:>7}  {wt:.4f}  {word!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[5] token 反解失败(不影响功能): {exc}")

    # 稀疏重叠看的是**字面**共享,不是主题。
    # XLM-R 把中文按字/子词切("变压器" → 变+压+器),所以学习式稀疏
    # 在中文上退化成一个字符级的词法信号:同主题但不同字,重叠可以是 0。
    # 这不是缺陷,而是双路召回存在的理由 —— 纯稀疏救不了同义不同字,
    # 稠密那一路才是主力,RRF 负责把两边捏到一起。
    sets = [set(w) for w in res.sparse]
    print(f"\n[6] 稀疏字面重叠: 文0∩文1={len(sets[0] & sets[1])}  "
          f"文0∩文2={len(sets[0] & sets[2])}   (低于预期属正常,见脚本注释)")

    print("\n全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
