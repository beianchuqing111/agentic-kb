"""评估集出题 —— 薄壳,真正的逻辑在 `eval/generate.py`。

**这个脚本会花 LLM 配额**(`eval_run.py` 不会)。调用数是「每篇参与出题的文档 1 次
(出题)+ 1 次(出反例)」,所以先跑 `--dry-run` 看计划是零成本的:
它读语料、算抽样、打印调用数,一次 API 都不发。

跑法:
    scripts\\eval_gen.py --dry-run                      # 零调用,先看计划
    scripts\\eval_gen.py --limit 3 --per-doc 2          # 小规模试
    scripts\\eval_gen.py --corpus D:\\ai-agent-book --limit 30 --dry-run
    scripts\\eval_gen.py --corpus D:\\ai-agent-book --yes   # 真跑(超 20 次调用要 --yes)

产出的东西**全是 `status: "draft"`**,落在 `eval/qa/<slug>.draft.jsonl`:

    它是一份**待筛的候选**,不是评估集。生成器只保证「它给的那个片段答得上」,
    并不保证那是唯一答案 —— 两块都答得上的题,人得把 `expected` 加宽。
    直接把 draft 当评估集用,分数会偏高,而且偏高得看不出原因。

筛题的动作就是这份文件存在的理由:把要保留的行复制进定稿文件
(如 `eval/qa/seed.jsonl`),`status` 改成 `keep`,补齐或收窄 `expected`。
从 draft 到定稿没有自动通道 —— 那一步就是全部价值所在。

注意:`seed.draft.jsonl` 里如果一条 `keep` 都没有,它是**加载不了**的
(`load_qa` 会报「评估集是空的」)。这是故意的 —— 一份没有一条题经过人工确认的
文件,不该被当成评估集跑出分数来。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 stdout 能打中文:沿用 kb.py:41-45 的做法,省得每个调用方都记
# 解释器全路径 + PYTHONIOENCODING。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 老解释器/被重定向时没有 reconfigure
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.generate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
