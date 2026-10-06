"""HTTP API 层(FastAPI)。

分工,别串:

    serialize.py   领域对象 → JSON。**唯一**的转换层。
    state.py       进程级共享状态:那把必须存在的锁、后端选择器、写权限通道。
    app.py         路由。只做「请求 → 一次下层调用 → JSON」的翻译。
    CONTRACT.md    对外契约。前端按它写,改任何一处都要三边一起改。

这一层**不做业务判断** —— 检索口径、权限规则、版本过滤、分块策略全在下层。
每在这里多写一个 if,就多一个和 CLI / Gradio 不一致的地方,而那种不一致
极难发现:`/api/search` 和 Gradio 的「检索」页签给出的结果不一样时,没人
会在第一时间怀疑是 API 层多加了个判断。

本 `__init__` 刻意**不导入 app** —— 导入它会把 fastapi/uvicorn 一起拖进来,
而 `api.serialize`、`api.state` 是可以被脚本和自检单独用的。
"""
