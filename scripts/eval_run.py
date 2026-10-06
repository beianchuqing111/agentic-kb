"""检索质量评估 —— 薄壳,真正的顺序在 `eval/runner.py`。

**不调用任何 LLM 配额**(唯一例外:开启定位语的导入阶段会调用 LLM 生成定位语,
约等于文档数那么多次。想完全不花钱就先跑一次带定位语的导入,之后都复用)。

用**独立的评估 collection**(`agentic_kb_eval_<slug>_<ctx|noctx>_<指纹>`),
不碰正式库 `agentic_kb` —— 这条是代码级护栏,不是口头约定。

跑法:
    scripts\\eval_run.py --preset no-rerank
    scripts\\eval_run.py --preset no-threshold --wide --save-baseline before
    scripts\\eval_run.py --preset no-threshold --wide --set rrf_k=2 --baseline before
    scripts\\eval_run.py --locate "值班人员每四小时抄录"
    scripts\\eval_run.py --purge

和 `check_*.py` 的分工:`check_*.py` 回答「流程对不对」,这里回答「召回好不好」。
所以这里**没有** `try/finally: delete_collection` —— 每次调参都重嵌 60 块、
重跑定位语会让「改一版跑一次」这个循环彻底不可用,而那个循环就是全部意义。
清理靠显式 `--purge`。

退出码:0 = 跑通(哪怕分数难看),1 = 基础设施问题(连不上、锚点失效、
文件缺失、基线不可比)。**分数下降不算失败** —— 这里不做回归门禁。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 stdout 能打中文:沿用 kb.py:41-45 的做法,省得每个调用方都记
# 解释器全路径 + PYTHONIOENCODING。errors="replace" 是为了在 GBK 控制台上
# 至少还能看到其余输出,而不是整个脚本崩掉。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 老解释器/被重定向时没有 reconfigure
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
