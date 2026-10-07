# 架构与环境

为什么这么设计、代码放在哪、跑在什么上面。

---

## 1. 两个后端,一套块

`hybrid` 和 `graphrag` 用**同一套分块、同一批向量**(见 `ingest/pipeline.py`
的 `prepare_document`,两条路径共用)。切换后端不会换掉块,只是多了图那一路召回。
`graphrag` 后端最后仍会并上向量那一路 —— 因为图有个硬伤:查询词对不上任何
实体名时会返回空,而那类问题恰恰是向量检索最擅长的。

## 2. 为什么写入侧不用 `PropertyGraphIndex.insert_nodes`

检索侧确实用 LlamaIndex(`PropertyGraphStore.vector_query` / `get_rel_map`),
但图**写入**是自己写的,四个实测理由记在 `ingest/graph_pipeline.py` 开头:
块被编码两遍、实体向量被元数据污染、每次 insert 刷全图 schema、同一实体编码 N 次。

## 3. 单例与并发

`retrieve.backends.get_backend` 是个**模块级单例**,ReAct 的工具表也是构造时
从它取的。两套界面的并发都被限成 1,理由就是它:

- **Gradio** 默认会并发跑多个请求,那就成了一边导入一边问答、两边共用同一个
  后端和同一个 Qdrant store。它因此设成 `concurrency=1`。
- **React** 那边是**前端自己排队**(`frontend/src/api.js`),服务端再用
  `BACKEND_LOCK` 兜一道。哪些请求不走队列、为什么,那份文件的开头写了。

本机单人用,串行反而是对的 —— 界面不会因为你手快连点两下就出错。

**但这条保证只在进程内。** 后端单例是**每进程一份**,跨进程没有锁:
`webui.py` 和 `scripts/api_server.py` 同时对着一个库做写操作时,
`content_hash` 的判断会互相打架,表现是"我明明重导了,界面却说没变化"。
只读没问题。

## 4. ReAct 为什么是文本协议而不是 function calling

`llm/client.py` 是纯文本对话客户端,表达不了 `tool_calls` / `role="tool"`。
硬走原生工具调用就得把客户端改成多态的,还要处理「有的兼容端点支持、
有的不支持」。文本协议的代价是格式可能跑偏,所以 `agent/react.py` 里有三道纠错:
`stop` 序列截断、格式纠错重试(上限 `AGENT_FORMAT_RETRIES`)、
最后把散文当答案返回而不是报错。

## 5. 工具返回的一切都不可信

知识库正文来自导入的文档,网页来自公网,两者都可能写着「忽略以上指令」。
所以工具的 Observation 都夹在 `<<<UNTRUSTED_DATA …>>>` 之间,系统提示词
明确声明那是资料不是指令。这层不是银弹,真正的兜底是只读工具**不可破坏任何东西**。

写工具打破这条保证,所以它们带三道闸且默认全关 —— 见
[usage.md §3.4](usage.md#34-写工具默认全关)。

## 6. 目录

```
config.py            所有配置(读 .env,单例 get_settings)
kb.py / kb.bat       命令行入口
webui.py / webui.bat Gradio 前端(四个页签)—— **保留**,调试用
api/                 HTTP 层(FastAPI;React 前端和别的程序走这里)
  app.py             路由
  schemas.py         请求体模型(pydantic)
  serialize.py       响应体拼装(把 RetrievedChunk 摊平成 JSON)
  state.py           进程级单例 + BACKEND_LOCK(一次只准一个请求碰后端)
  CONTRACT.md        接口契约。**改行为先改它**,附录有逐控件的能力对照表
frontend/            React 界面(Vite;与 Gradio 并存,README 在那层)
uploads/             前端上传的文件落这里(自动建;块里的 source 就是它)
embed/               bge-m3 稠密+稀疏
store/               qdrant_store(块) / graph_store(Neo4j,含 delete_doc_graph)
ingest/              loader → chunker → contextual(定位语) → pipeline / graph_pipeline
retrieve/            hybrid(稠密+稀疏+RRF+重排) / reranker / backends(hybrid / graphrag)
llm/                 OpenAI 兼容客户端(chat / chat_json / map_batch)
agent/               react(文本 ReAct) / tools(三个工具) / websearch(Tavily)
eval/                检索质量评估(见 evaluation.md)
  schema.py           问答集读写 + validate()
  metrics.py          纯指标函数(hit/mrr/ndcg/precision/recall)
  corpus.py           语料定位、doc_id 解析、锚点校验
  runner.py           隔离 collection 导入、跑、聚合、比对
  report.py           表格打印、逐题 diff、基线读写
  generate.py         LLM 出题引擎(唯一花钱的部分)
  README.md           评估设施自己的说明(放**这层**,不放 corpus/seed/ ——
                      discover() 只跳过 `~$` 和 `.` 前缀,**不跳过 README.md**,
                      放进去会被当成语料入库)
  corpus/seed/        种子语料 9 篇,**只放可入库文档**
  qa/seed.jsonl       人工筛后定稿
  qa/*.draft.jsonl    出题产物,待筛,可重生
  baselines/          基线 JSON
scripts/             自检脚本 + 服务启动
  api_server.py       起 HTTP API(127.0.0.1:8000;`--reload` 别和前端调试一起用)
  check_eval.py       评估设施自检(不花钱、不调 LLM)
  check_api.py        接口自检(默认档要跑一次真上传,见脚本开头的说明)
  eval_run.py         跑评估(不花钱)
  eval_gen.py         出题(花钱 —— 先 --dry-run)
qdrant_storage/      Qdrant 数据(别手删)
neo4j/               Neo4j 5.26.9 + 自带 JRE + downloads(安装包缓存)
```

## 7. 环境

- Python:**anaconda py310**(`C:\Users\Administrator\anaconda3\envs\py310`)
- **torch 不要重装** —— py310 里已有 2.9.1+cu130,CUDA 可用(RTX 5080);
  换成 CPU 版会让 bge-m3 慢十倍以上
- bge-m3 和 bge-reranker-v2-m3 已缓存,所以默认离线可用。
  **换机器**首次要下模型:设 `KB_HF_ONLINE=1` 跑一次导入,下完改回
- Qdrant:6343/6344,数据在 `qdrant_storage/`(独立于 imgsearch 在 6333 上的数据)
- Neo4j:Community 5.26.9,自带 JRE,`neo4j/server/data`;
  重建步骤(两个安装包已缓存在 `neo4j/downloads/`)写在
  `scripts\start_neo4j.bat` 顶部注释里
