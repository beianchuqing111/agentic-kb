"""检索质量评估:给定「查询 → 期望命中块」,算出指标并支持前后对比。

和 `scripts/check_*.py` 那套自检的分工
-------------------------------------
自检验证的是**流程对不对**(能不能读 GBK、增量会不会重复写、阈值有没有生效),
跑一次全绿就够,所以它们用一次性 collection、跑完全清理、**不花一分钱**。

这一套验证的是**召回质量好不好**,回答的是「我这次改动让它变好还是变坏了」。
两者不重叠:自检全绿的系统完全可能召回得很难看。

因此子模块的依赖方向是刻意安排的 —— 越靠前越纯,越靠后越脏:

    metrics.py   纯函数,无依赖、无模型、无网络、无 IO
    schema.py    纯数据 + JSONL 读写,只依赖标准库
    corpus.py    语料定位与锚点解析,依赖 store(读库校验锚点)
    runner.py    真导入、真检索,依赖 ingest / retrieve
    report.py    纯打印与基线读写
    generate.py  唯一会花钱的模块(要 LLM 出题)

`metrics.py` 和 `schema.py` 不 import 项目里任何东西,所以它们能在没有 Qdrant、
没有 torch、没有 key 的机器上单测 —— `scripts/check_eval.py` 的第一批断言就是
靠这个性质才能离线跑。

注意本包名 `eval` 会**遮蔽 Python 内置的 `eval()`**吗?不会 —— 内置函数在
`builtins` 里,不是模块。但 `import eval` 在某些场景下确实会让人误读,
所以包内一律写 `from eval.metrics import ...` 全路径,不写 `from . import ...`,
让调用点一眼看出说的是本包。
"""
