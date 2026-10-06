# agentic-kb HTTP API 契约

前端按这份契约写。**这份文件是唯一口径** —— 后端实现和前端消费都以它为准,
改任何一处都要同时改这里、后端、前端。

Base:`http://127.0.0.1:8000`,全部挂在 `/api` 下。

> 本文件描述的是**已实现的**行为。凡是这里写了、代码里没有的,按契约算缺陷;
> 凡是代码里有、这里没写的,前端不许依赖。契约里留一个编出来的字段,
> 前端就会照着一个永远为 null 的值写逻辑。
>
> 按这条规矩,写这份文件时砍掉过一个字段:单路由双模式的 ingest
> (实际是 `/api/ingest` 与 `/api/ingest/upload` 两条路由)。
> 而 `llm_calls` 一度也在砍掉的名单上 —— 它看着像个"入库统计",但
> `IngestStats` 里根本没有这个字段。后来发现 `webui.py` 的入库页一直在报
> 「本次入库花了多少次模型调用」,用的办法是**量进程级调用计数器的前后差值**
> (`get_llm().usage["calls"]`,见 `webui.py:_ingest_cost_note`)。
> 也就是说这个能力是真实存在的,只是不属于某次 `IngestStats`。
> 所以它留了下来,并在下面写明了它的口径 —— **全局计数器的差值,
> 不是这次入库独占的消耗**。抹掉它才是丢能力。
>
> 同样按这条规矩补进来的是 `/api/search` 的 `explain`:契约一直写着
> `scores.dense_rank`/`sparse_rank` 的 null 是什么意思,但那段算名次的代码
> 被关在 `if debug is not None:` 里,API 从不传 `debug` —— 于是这两个字段
> **恒为 null**,而契约说它们有意义。现在有了显式开关,默认不付代价。

---

## 0. 四条贯穿全局的约定

### 0.1 后端是全局单例,所以请求是**串行**的

`retrieve/backends.get_backend()` 缓存实例(bge-m3 权重挂在上面,重建一次
就是重载几个 G),而且 `set_backend()` 改的是**进程级全局变量**。所以:

- 每个请求开头按 `backend` 字段切后端,**这之后**才构造 agent
  (工具表是构造时从 `get_backend()` 取的 —— 所以"切后端"和"构造 agent"
  之间不能被别的请求插进来)。
- 整个「切后端 → 检索/问答 → 收集结果」必须在**一把锁**里。不锁的话请求 A
  把后端切成 graphrag、请求 B 切成 hybrid,两个请求会互相把对方的后端换掉 ——
  这不是"慢一点",是**结果串台**:A 拿着图检索的期待收到纯向量的结果,
  而两者在界面上都长得像正常输出。
- 前端不要指望并发。一次只发一个检索/问答请求(导入也是)。这和 Gradio 那边
  `default_concurrency_limit=1` 是同一条约束,不是 API 的缺陷。
- 请求排队时**没有**进度提示,前端只能显示"等待中"。这是已知的取舍:
  串行是为了正确性,不是为了性能。

### 0.2 `backend` 字段:省略 = 用**服务端当前那个**

所有带 `backend` 的请求(可选字段),语义统一为:

| 传值 | 行为 |
|---|---|
| 合法后端名(`"hybrid"` / `"graphrag"`) | 切到它,并在后续请求里**保持** |
| 省略 / `null` | **用服务端当前的后端**,不是"退回配置默认值" |
| 不认识的名字 | 记一条 warning,退回配置默认值 |

第二行是刻意的,别按常识猜:这两者在第一个请求上恰好相等,之后就分岔了。
前端的选择器是个跨页签的单选按钮(和 `webui.py` 那一排 radio 一样),
用户在检索页选了 graphrag、切到文档页,文档页应该**还是** graphrag。
每次省略都弹回配置默认的话,界面上会出现"我明明选了 graphrag,列表却是
hybrid 的" —— 而两个后端的文档列表本来就一模一样,这种不一致根本看不出来,
只会让人以后不敢信这个选择器。

响应体里一定回显 `backend`(实际生效的那个),前端以它为准渲染选择器。

### 0.3 `Hit`(检索结果)的字段

`RetrievedChunk` 的直译。**引用回溯原文是这个产品的核心交互,字段不能砍。**

```jsonc
{
  "citation": "绝缘子检测作业指导书 (块 3)",  // 给人和 LLM 看的来源标签
  "text": "……块正文……",
  "context": "……定位语(导入时模型写的"这段在讲什么")……",  // 可能为空串
  "doc_id": "3b31671de10f71ad",
  "chunk_index": 3,
  "source": "D:\\docs\\绝缘子检测作业指导书.md",
  "title": "绝缘子检测作业指导书",
  "score": 0.4947,          // 最终排序分
  "rrf_score": 0.0328,
  "rerank_score": 0.4947,   // null = 没开重排;0.0 = 重排认为完全无关。**别把 null 当 0**
  "dense_rank": 0,          // 见下面「两路名次」:默认 null,要 explain=true 才有数
  "sparse_rank": 2,         // 同上
  "scores": {               // 永远有数的并行视图,前端排序/画柱状图用它,省得每处写 ?? 0
    "final": 0.4947,
    "rrf": 0.0328,
    "rerank": null,         // 同 rerank_score,这里也保留 null
    "dense_rank": 0,
    "sparse_rank": 2
  },
  "from_graph": false,      // true = 这条是图多跳捞回来的
  "entities": ["绝缘子", "裂纹"],   // 命中实体,可能为空数组
  "facts": [                // 关系型事实,可能为空数组
    {"head": "绝缘子", "relation": "导致", "tail": "放电"}
  ],
  "graph_hits": 0,          // 本题从图那一路拿到的块数。**hybrid 永远 0,不是 bug**
  "status": "current",      // "current" | "superseded" | ""(老数据没写这个字段)
  "doc_version": "v2",      // 可能为空串(文件名里没写)
  "effective_from": "2025-06-01",
  "effective_to": ""
}
```

**两路名次(`dense_rank` / `sparse_rank`)单独说,因为它们的取值取决于请求参数:**

- **默认(`explain` 缺省或 false)恒为 `null`。** 这不是"两路都没召回它" ——
  是**没算**。算名次要额外付两次 Qdrant 查询:`query_hybrid` 已经把稠密/稀疏
  各搜过一遍,但**融合后的结果不带每路的原始名次**,想要名次只能把两路再各查
  一遍(等于 ANN 搜索量翻倍)。所以它由调用方显式要。
- **`explain: true` 时才有值**:`0` 基名次,`null` 才是"这一路确实没召回它"。
- **名次是块级的,不是文档级的。** 一篇规程有六七个块,按 `doc_id` 取名次会让
  同一篇文档的每个块共用一个数 —— 用它做"是哪一路召回的这一块"的判断会失真。
- **走 `graphrag` 后端时,只从图那一路来的块两个名次都是 `null`,而且这是对的:**
  两路向量召回**确实**没召回它,它是靠实体多跳捞上来的。前端显示成「来自图」
  即可,不要补 0 假装它排第一。
- 前端**不要**为了拿名次去默认打开 `explain`。只在用户真的要看"双路贡献"时开。

**问答的响应里没有 `Hit` —— 这一条决定了引用回溯只能做在检索页。**

`POST /api/ask` 返回的是 `answer` + `steps[]`,而 `steps[]` 的字段只有
`index / thought / action / action_input / observation / truncated / repeated / note`
(见 `serialize.steps_to_list`)——**没有任何块对象**。工具返回的检索结果被
工具层渲染成了 `observation` 文本(`format_hits`),结构到那儿就没有了。
所以:

- 问答页能显示引用**文字**(`observation` 里有 `标题 (块 N)` 和 `出处: <路径>`),
  但拿不到 `doc_id`/`chunk_index` 这些能点开回溯的字段。
- **Gradio 的问答页也一样**(`webui.py:do_ask` 只渲染 `render_steps()` 的文本),
  所以这不是 React 版的倒退,是两条前端共同的口径。
- 想点开引用回溯原文,走**检索页**(`/api/search` 的 `hit` 字段齐全);
  要在问答页也能点,得先改 `AgentStep` 让它带着 hits —— 那是后端改动,
  不在契约允许前端自行发挥的范围内。

两条容易踩的:

- **`scores` 不是"另一套口径"**,它和上面几个字段同值,只是把 null 也照原样
  带过来。它存在的原因是让前端不用在每个渲染点都判断 null。
- **`graph_hits` 恒为 0 不代表图没接上。** 评估语料的 `fusion_top_k=50` 会在
  61 块的库里一次取走 50 块,图那一路**结构上没有位置可加**,于是
  graphrag 和 hybrid 的结果逐位相同。这是实测过的、有报告的结论
  (`eval/reports/graph_path_audit.md`:通路是通的 —— 每题种子实体 26–53 个、
  多跳事实 16–44 条;窄窗口 `fusion_top_k=10` 时图能补上 1/23 的 gold)。
  前端**不要**因为"选了 graphrag 却没变化"就报错或提示异常。

### 0.4 错误

非 2xx 一律返回:

```jsonc
{ "detail": "人话错误说明" }
```

前端按 `detail` 原样显示即可。码的含义:

| 码 | 何时 |
|---|---|
| `400` | 参数不对(缺字段、类型错、`path` 不存在、后端名... ) —— 附**人话** `detail` |
| `503` | 依赖没起(Qdrant / Neo4j 连不上) |
| `500` | 其他 |

pydantic 的校验失败**不**直接透传(那玩意是一串 JSON path,给用户看等于没写),
统一转成一行说明缺什么/错在哪。

---

## `GET /api/health`

```jsonc
{
  "ok": true,
  "backend": "hybrid",
  "detail": { "collection": "agentic_kb", "points": 61, ... },  // 依赖探测结果
  "error": "",                       // ok=false 时是人话原因
  "config": {
    "llm_configured": true,
    "llm_model": "deepseek-chat",
    "tavily_configured": true,
    "rerank_enabled": true,
    "contextual_enabled": true,
    "include_superseded": false,     // 服务端默认口径(前端勾选框的初始值)
    "allow_write": false             // 服务端是否允许写工具
  }
}
```

**这个端点不拿锁、也不切后端** —— 它必须在后端正忙(导入中)时也能立刻回答,
不然前端的"服务不可用"横幅会在最不该出现的时候出现。

## `GET /api/backends`

```jsonc
{ "backends": ["hybrid", "graphrag"], "default": "hybrid", "current": "hybrid" }
```

`current` 是服务端此刻实际在用的那个(见 §0.2)。

## `POST /api/search`

请求:

```jsonc
{
  "query": "绝缘子破损判据",     // 必填,非空
  "top_k": 5,                    // 可选,默认取配置
  "backend": "hybrid",           // 可选,见 §0.2
  "include_superseded": false,   // 可选,不传 = 服务端默认口径
  "explain": false               // 可选,true = 填 Hit 的两路名次,见 §0.3
}
```

响应:

```jsonc
{
  "backend": "hybrid",
  "elapsed_ms": 412,
  "include_superseded": false,
  "hits": [ /* Hit[] */ ],
  "llm_text": "……format_hits() 出来的、真正喂给模型的那段文本……"
}
```

`llm_text` 永远返回。前端「看看模型眼中的样子」用它 —— 这是这套系统里
"检索结果和模型输入是否一致"的唯一可核对处。

`explain` **只影响 `hits` 里两个名次字段的取值,不改变排序**(实测同一批
结果逐位相同)。它默认关,因为开着要多付两次 Qdrant 查询。

`/api/ask` **没有** `explain`,原因详见 §0.3 末尾:问答的响应里**根本没有
结构化的检索结果**,名次没有落脚的地方。

## `POST /api/ask`

请求:

```jsonc
{
  "question": "绝缘子出现裂纹怎么办",   // 必填
  "backend": "hybrid",                  // 可选,见 §0.2
  "include_superseded": false,
  "allow_write": false,                 // 闸 1:请求级写意图,默认 false
  "confirm_write_tools": [],            // 闸 2:逐个点名的写工具,见下
  "stream": true                        // true=SSE,false=等完整结果
}
```

### 写权限的两道闸(无头环境下的取舍)

没有终端,谁来点那个"确认"?`permissions.authorize` 的第 2 闸写着
**「没有 confirmer 也算拒绝」**,这条不能破。所以 API 自己给一个确认通道:

- **闸 1 `allow_write`** —— **两把钥匙,而且必须同时插进去**:
  请求里的 `allow_write` **和服务端的 `AGENT_ALLOW_WRITE` 两个都为 true 才放行**
  (实现是 `requested and s_master`,见 `api/app.py:_build_agent`)。
  只带请求字段、服务端没开时**按拒绝处理**,并且会在服务端日志里留一条
  warning —— 静默忽略一个安全字段是最坏的做法。
  `GET /api/health` 的 `config.allow_write` 就是这个服务端天花板,前端据此
  决定要不要把写工具的 UI 显示出来。
- **闸 2 `confirm_write_tools`** —— **逐个工具点名**,如
  `["export_report"]`。不在名单里的写工具一律拒。

分开的理由:如果 `allow_write=true` 就等于全同意,两闸就塌成一闸。
分开之后,一个 `allow_write=true` 但没点名工具的请求**照样被拒**。

⚠️ **如实说清的边界**:它批准的是**工具**,不是**这一次调用的参数** ——
参数是模型后面才生成的,API 形态下没法"先看后批"。所以它**不等价于**
人工确认,Gradio 那边逐次弹窗才是。这一点别在简历/面试里吹过头。

**第三道闸在工具执行时**:参数校验(导出文件名不能带目录成分 —— 挡目录穿越
与绝对路径)。它校验的根目录和工具实际写入的根目录必须是**同一个**
(`exports_dir` 要同时交给 guard 和 `build_default_tools`),否则这道闸是空的。
`scripts/check_api.py` 的第 9 节把三道闸分开测 —— 合起来测的话,闸 1 关着时
后面两道根本不会执行,断言就成了摆设。

### 非流式(`stream: false`)

```jsonc
{
  "question": "绝缘子出现裂纹怎么办",
  "answer": "……",
  "stop_reason": "final_answer",   // final_answer | max_iterations | unparsed_output
  "steps": [
    {"index":1,"thought":"…","action":"search_knowledge_base","action_input":"绝缘子裂纹",
     "observation":"…","truncated":false,"repeated":false,"note":""}
  ],
  "usage": {"prompt": 123, "completion": 45, "calls": 3},
  "warnings": ["…"],
  "elapsed_ms": 18320
}
```

### 流式(`stream: true`)—— SSE

`Content-Type: text/event-stream`,每条 `data: <JSON>\n\n`。事件类型:

| `type` | 何时 | 关键字段 |
|---|---|---|
| `start` | 循环开始 | `question`, `max_iter` |
| `action` | **工具执行前**(可能几秒到几十秒) | `index`, `tool`, `input`, `thought` |
| `step` | 工具执行完 | `index`, `tool`, `input`, `thought`, `observation`, `truncated`, `repeated`, `note` |
| `warning` | 格式纠错 / 轮次用尽等 | `message`, 可能带 `index` |
| `done` | 结束 | `result` = 和**非流式响应体完全一样的对象** |
| `error` | 出错 | `detail` |

前端要点:

- **`action` 事件是防"卡死感"的关键** —— 检索和联网各自可能几秒,必须在
  收到 `action` 时就显示「正在调用 X…」,而不是等 `step`。这个事件是**为了
  前端而存在的**,不是内部日志的副产品。
- 收完 `done` 就别再读了(后端随后关闭流)。把 `done.result` 当作最终状态,
  和走非流式拿到的对象是同构的 —— 前端可以用同一段代码收尾。
- `error` 之后流结束。**拿到 `error` 也要收尾**(渲染错误态),别让界面停在
  最后一个 `action` 上。
- 锁在**工作线程**里持有,所以 SSE 期间别的请求在等;这是 §0.1 的必然结果。

## `POST /api/ingest`

导入**服务端上已有的路径**。JSON 请求体:

```jsonc
{ "path": "D:\\docs", "recursive": true, "force": false, "backend": "hybrid" }
```

## `POST /api/ingest/upload`

导入**上传的文件**。`multipart/form-data`:

| 字段 | 类型 | 说明 |
|---|---|---|
| `files` | File[] | 必填,可多个 |
| `recursive` | bool | 默认 true |
| `force` | bool | 默认 false |
| `backend` | str | 可选,见 §0.2 |

上传的文件按**原始文件名**落到 `uploads/`(路径由 `config.UPLOAD_DIR` 给),
再从那里导入 —— 块里记的 `source` 因此才是看得懂的路径。**不要**直接用
上传的临时路径入库:临时路径会进 `source`,溯源就指向一个过一会儿就不存在
的地方,而且看起来完全正常。Gradio 那边同样落到 `uploads/` 就是这个理由。

**为什么是两个路由而不是一个双模式路由**:一个路由同时吃
`application/json` 和 `multipart/form-data` 要靠嗅探 Content-Type,
校验错误(比如 JSON 缺 `path`)会以两种完全不同的形状冒出来,前端得写两套
错误处理。拆开之后两条路的请求体、错误、`saved` 字段各自干净。

响应(两个路由一致):

```jsonc
{
  "backend": "hybrid",
  "elapsed_ms": 10400,
  "files_seen": 1,
  "files_loaded": 1,
  "docs_indexed": 1,
  "docs_unchanged": 0,
  "chunks_written": 61,
  "chunks_failed": 0,
  "contextual_missing": 0,      // 开启定位语但没能生成的块数(>0 说明 LLM 那段有问题)
  "entities_written": 0,        // 只有 graphrag 那条路会写
  "relations_written": 0,
  "llm_calls": 61,              // 见下。**可能为 null**
  "saved": ["D:\\agentic-kb\\uploads\\x.md"],   // **仅 upload 路由有**,落盘后的路径
  "errors": [ {"file": "x.pdf", "error": "ValueError: …"} ]
}
```

`errors` 是**对象数组**不是字符串数组 —— 前端要能分列显示"哪个文件"和
"为什么"。拼成一个字符串就只能整段显示,长错误信息会糊成一片。

**`llm_calls` 的口径要说清楚,因为它不是"这次入库的消耗"。** 它取的是
**进程级调用计数器的前后差值**(`get_llm().usage["calls"]`,和
`webui.py:_ingest_cost_note` 报"本次入库花了多少次模型调用"是同一个办法)。
所以:

- 它**只在本进程**意义明确。因为后端是串行的(§0.1),同一时刻不会有第二个
  请求在偷偷调模型,差值基本就等于这次入库 —— 这是这个数能用的前提,不是巧合。
- **`null` 不是 0,是"读不到"**:计数器拿不到时返回 `null`,前端显示成「—」,
  **不要**显示 0(0 的意思是"这次入库一次模型都没调",那是另一回事)。
- 它包含**建图**那部分的调用(走 graphrag 时),不只是写定位语。

单文件失败**不影响其他文件**,失败项进 `errors`,成功项照常计入上面的计数。
`errors` 非空但 HTTP 200 是正常状态(部分成功),前端要显示这批失败清单。

## `GET /api/docs?q=<可选过滤>`

```jsonc
{
  "total_docs": 9, "total_chunks": 61, "filtered": 9,
  "docs": [
    {"doc_id":"…","title":"…","source":"…","chunks":7,
     "status":"current","mixed":false,
     "doc_version":"v2","effective_from":"","effective_to":""}
  ]
}
```

- `status: "superseded"` 的文档前端要**明确标出来**(灰掉/加"已废止"徽章)——
  不然用户会奇怪为什么它检索不到。这个标记是**检索行为的一部分**,不是装饰。
- `mixed: true` = 这篇文档内部块的 `status` 不一致(有的 current 有的
  superseded)。聚合口径是"只要有一块失效,整篇按已标记显示",因为漏标会
  让用户对着查不到的结果找不到原因。`mixed` 是给界面提示"这篇状态不统一"
  用的。
- 状态是**逐块**存的,一篇文档的 `status` 是从块聚合出来的,不是一个独立字段。

## `GET /api/stats`

`backend.stats()` 的直译。key 随后端不同:

```jsonc
// hybrid
{ "backend":"hybrid", "collection":"agentic_kb", "docs":9, "chunks":61,
  "documents":[ /* 同 /api/docs 的 docs */ ] }

// graphrag —— 多三个图里的量
{ "backend":"graphrag", "collection":"…", "database":"neo4j",
  "docs":9, "chunks":61,
  "entities":412, "relations":388, "entities_with_embedding":412,
  "documents":[ … ] }
```

空库时 hybrid 只返回 `backend/collection/docs/chunks`(没有 `documents`)——
**"还没导入任何东西"是最常见的一次调用**,前端要能处理 `documents` 缺席。

## `GET /api/audit?n=20`

Batch 2 的写操作审计,给前端「审计」页签用。读 `logs/audit.jsonl` 的**末尾** n 条。

```jsonc
{ "records": [ {"ts":"…","call_id":"…","phase":"intent|result|denied","tool":"…","arg":"…","ok":true,"detail":"…","actor":"…"} ] }
```

- 一次写操作产生**至少两条**:`phase=intent`(副作用之前落盘,fsync)和
  `phase=result`。前端按 `call_id` 把一对配起来显示,不要按顺序配对 ——
  并发的写会交错。
- `phase=denied` = 被闸拦下的**越权尝试**(比如没点名工具却调了写工具)。
  它和"写失败"是两回事,前端要用不同的样式 —— **把被拒绝的攻击和写入失败
  混在一起显示,等于把安全事件藏进了错误日志里**。
- 文件不存在 → `{"records": []}`,不报错。

## `GET /api/eval?full=0`

Batch 1 的评测基线,读 `eval/baselines/*.json`。**数字原样透传,不做二次计算** ——
在 API 里重算一遍"hit@1"就会多出一个可能和基线文件不一致的口径,
而前端展示的必须是跑出来的那个数。

```jsonc
{
  "baselines": [
    {
      "name": "b1b2_graph",                       // 文件名(无扩展名)
      "created_at": "2026-10-06T18:11:23",
      "schema_version": 1,
      "collection": "agentic_kb_eval_seed_ctx_b364d64b",
      "corpus": {"root":"…","slug":"seed","n_docs":9,"n_chunks":61},
      "qa": {"path":"…","sha1":"39c3c3eea092f43f","n_kept":30,"n_answerable":25,"n_negative":5},
      "ks": [1,3,5,10],
      "runs": {
        "no-threshold+wide": {
          "spec":"no-threshold+wide", "retriever":"hybrid", "use_rerank":true,
          "effective_k":10,
          "aggregates": { "1": {"n":25,"hit":1.0,"rr":1.0,"ndcg":0.9543,
                                "ndcg_binary":1.0,"precision":1.0,"recall":0.6867}, … },
          "by_type": { "1": {"multi_hop": {"n":11,"hit":1.0, …}}, … },
          "negatives": { … }
        },
        "graphrag:no-threshold+wide": { … }
      }
    }
  ]
}
```

- **`metrics` 不存在**,别找它。真数字在 `runs[<key>].aggregates[<k>]`,
  逐题型在 `by_type[<k>][<type>]`。`aggregates` 的键是**字符串** k。
- 默认(`full=0`)**剥掉** `per_item` / `negative_items`(逐题明细,占了文件
  九成体积)。要逐题明细传 `full=1`。
- `by_type` 里 `n < 3` 的题型,均值是噪声 —— 前端显示时要标注,别让
  「n=2 的 1.000」和「n=25 的 0.95」看起来同样可信。
- 文件不存在 → `{"baselines": []}`,**不要报错** —— 没跑过评测是正常状态。
- 目录里可能有同一实验的多个基线(`b1b2_graph.json` 与
  `b1b2_graph_prefix.json` 内容相同、只是名字不同)。按 `created_at` 倒序,
  前端默认展示最新的一份即可。

---

## 附:验收对照

Batch 4 的验收是"**逐条对比 `webui.py` 四个 tab 的能力,确认无遗漏**"。
下表按 `webui.py:547` 起的 `gr.Blocks` 布局逐控件列,左列是 Gradio 控件,
右列是它对应的路由/字段。**Gradio 是基准,不是上限** —— API 多出来的能力
另列一表,前端可以做,但不做也不算缺。

### 全局(所有 tab 共用)

| Gradio 控件 | 对应的 API |
|---|---|
| 后端单选 `backend`(`webui.py:555`) | 每个请求体的 `backend` 字段;选项与当前值取自 `GET /api/backends` |
| 状态徽标 `status_badge`(LLM/Tavily 是否配好,`:565`) | `GET /api/health` → `config.llm_configured` / `llm_model` / `tavily_configured` |

### 📥 导入文档(`:572`)

| Gradio 控件 | 对应的 API |
|---|---|
| `files` 多文件上传(`:580`) | `POST /api/ingest/upload`(multipart `files`) |
| `path_text` 服务器路径(`:586`) | `POST /api/ingest`(`path`) |
| `force` 强制重导(`:591`) | 两条路由的 `force` |
| `recursive` 递归子目录(`:595`) | 两条路由的 `recursive` |
| `ingest_btn`(`:596`) | 同上 |
| `ingest_out`(`:598`) | 响应体 `files_seen` … `errors`(见上面那张响应表) |
| 入库花了多少次模型调用(webui 的 `_ingest_cost_note`) | 响应 `llm_calls`(**同口径**:全局计数器差值) |

### 🔍 检索(`:611`)

| Gradio 控件 | 对应的 API |
|---|---|
| `query`(`:617`) | `query` |
| `top_k` 滑杆(`:621`) | `top_k` |
| `raw` 「附上给模型看的文本」(`:625`) | `llm_text`(**永远返回**,不必先勾) |
| `search_btn`(`:626`) | `POST /api/search` |
| `search_out`(`:627`) | `hits[]` 全字段 + `elapsed_ms` |

### 💬 问答(`:636`)

| Gradio 控件 | 对应的 API |
|---|---|
| `question`(`:642`) | `question` |
| `verbose` 「完整轨迹」(`:645`) | 前端用 `steps[]` 自己拼(见下面那条说明) |
| `ask_btn`(`:649`) | `POST /api/ask` |
| `answer_out`(`:650`) | `answer` + `stop_reason` |
| `detail_out`(`:651`,即 `render_steps` + `render_usage` + warnings) | `steps[]` + `usage` + `warnings` |

> **`verbose` 那条不是缺失,是给了更多。** Gradio 的「完整轨迹」调
> `result.transcript()`,而 `transcript()` 只用 `steps` + `warnings`,并且
> 把每条 observation 再 `_clip_for_log` 到 **200 字**(`react.py:215`)。
> API 的 `steps[].observation` 是 `AgentStep` 里的原值,没有这道日志截断 ——
> 所以前端拿 `steps[]` 能拼出**比 Gradio 更长**的轨迹。要短就自己截。

### 📊 库状态(`:662`)

| Gradio 控件 | 对应的 API |
|---|---|
| `refresh_btn` 「刷新体检/计数」(`:665`) | `GET /api/health`(`detail` = 连通性)+ `GET /api/stats`(计数) |
| `docs_filter` 过滤框(`:666`) | `GET /api/docs?q=` |
| `docs_btn` 「列出文档」(`:669`) | `GET /api/docs` |
| `docs_out`(`:670`) | `documents[]` |
| `status_out`(`:672`) | `health.detail` + `stats` |

### API 有、Gradio 没有的(前端可做,不做不算缺)

| 能力 | 路由/字段 | 为什么值得做 |
|---|---|---|
| 评测数字(命中的那批真实基线) | `GET /api/eval` | Gradio 完全没有这一栏;这是简历里那些数字的出处 |
| 审计日志(谁在什么时候调了写工具) | `GET /api/audit` | 配合写权限,是"可追溯"那一半 |
| 放行已失效版本 | 请求体 `include_superseded` | Gradio 只能改环境变量重启 |
| 双路召回贡献 | `POST /api/search` 的 `explain` | 见 §0.3 |
| 写工具授权 | `allow_write` + `confirm_write_tools` | Gradio 靠命令行 `--allow-write` + 逐次弹窗 |
| 流式中间步骤 | `stream: true`(SSE) | Gradio 是整段阻塞返回,长问答期间界面一动不动 |
| 上传扩展名校验 | `/api/ingest/upload` 的 400 | Gradio 把不支持的扩展名直接扔给 loader |

**仍然保持同一条并发约束**:一次只发一个会碰后端或跑模型的请求。
Gradio 用 `default_concurrency_limit=1` 表达,这里用服务端的锁表达(§0.1)。
