# 使用手册

安装、起服务、命令、配置、排错。想先看懂设计再看怎么用,去
[architecture.md](architecture.md);想知道那些数字怎么来的,去
[evaluation.md](evaluation.md)。

---

## 1. 起服务(每次开机都要,两个窗口别关)

```bat
D:\agentic-kb\scripts\start_qdrant.bat
D:\agentic-kb\scripts\start_neo4j.bat        REM 只有 graphrag 后端需要
```

双击也行。**关掉窗口 = 停服务**;数据落在 `qdrant_storage\` 和 `neo4j\server\data\`,
重启不丢。Qdrant 占 6343/6344(和 imgsearch 的 6333 是两套,互不干扰),
Neo4j 占 7474/7687。

先确认起来了:

```bat
kb.bat health
```

看到 `✅ 全部就绪` 或 `⚠️ 后端可用,但 LLM 未配置` 都算正常。

## 2. 填 LLM key(只有 `ask` 和"导入时补定位语"需要)

打开 `D:\agentic-kb\.env`,填 `LLM_API_KEY`。DeepSeek 或 Qwen 都行:

| 用哪家 | `LLM_BASE_URL` | `LLM_MODEL` |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Kimi | `https://api.moonshot.cn/v1` | `kimi-k3` |

`.env` 已在 `.gitignore` 里,**别把 key 贴进对话/截图/issue**。

**不填也能用** —— `health` / `ingest` / `search` / `docs` / `stats` 全都照常,
只有 `ask` 会明确报错告诉你去填。详见 [§6](#6-没配-llm-key-会怎样)。

## 3. 命令

`kb.bat` 放在项目根,**在任何目录下都能直接调用**(内部用 `%~dp0` 定位)。
它只做两件事:切 UTF-8 代码页、用 py310 解释器。

| 命令 | 干什么 | 要 key? |
|---|---|---|
| `kb.bat health` | 体检:配置 + Qdrant/Neo4j 通不通 | 否 |
| `kb.bat stats` | 各项计数(多少文档/块/实体) | 否 |
| `kb.bat ingest <路径>` | 导入文件或目录 | 否¹ |
| `kb.bat search <查询词>` | 只检索,看召回效果 | 否 |
| `kb.bat ask <问题>` | ReAct 智能问答 | **是** |
| `kb.bat ask <问题> --allow-write` | 同上,并允许写工具(每次单独确认) | **是** |
| `kb.bat docs` | 列出库里的文档 | 否 |

¹ 没 key 也能导入,但每个块前面会缺「定位语」,召回质量下降。填上 key 后重导即可补齐。

全局开关 `--backend {hybrid,graphrag}` 和 `-v` **放在子命令前后都行**:

```bat
kb.bat --backend graphrag health
kb.bat stats --backend graphrag
kb.bat ask "问题" -v
```

### 3.1 ingest —— 导入

```bat
kb.bat ingest D:\docs                  REM 递归整个目录
kb.bat ingest D:\docs\规程.pdf          REM 单个文件也可以
kb.bat ingest D:\docs -f                REM 强制全量重导(忽略内容哈希)
kb.bat ingest D:\docs --no-recursive    REM 不递归子目录
```

支持的格式:`.txt .md .markdown .text .log .pdf .docx`。

**重复导入是安全的。** 每个文档按 `content_hash` 判断变没变,没变就跳过
(`未变化跳过 N 篇`),变了才重写。所以:

- 往目录里丢新文件 → 直接再跑一次 `ingest`,老文件不会白重算。
- 之前没配 key 导入的 → 配好 key 后再跑一次 `ingest`,**不用先删**,
  缺定位语的块会被补齐。

导入会打印:扫描/读入/新建更新/未变化跳过/写入块数,以及耗时。
有文件出错会列出来(网络抖动导致某个 PDF 没进来属于这类)。

### 3.2 search —— 只检索

```bat
kb.bat search "绝缘子破损判据"            REM 默认最多 5 条
kb.bat search "绝缘子破损判据" -k 10      REM 要更多
kb.bat search "绝缘子破损判据" --raw      REM 额外打印"模型看到的那份文本"
```

`5` 是上限不是保证:低于 `RERANK_MIN_SCORE` 的会被挡掉,所以常常只返回
一两条(实测一句很具体的查询:命中的那条 0.9897,次相关 0.1463,其余全被挡)。

输出里每条带:出处、分数、正文片段;graphrag 模式下还会带 `[图]` 标记、
命中实体、以及 `头 --关系--> 尾` 的关系三元组。`--raw` 那份就是 ReAct 里
模型实际读到的格式,调提示词时对着它看最直观。

**这是最该先用的命令。** `ask` 回答不对时,先 `search` 看检索本身对不对,
能直接分清是"检索没召回到"还是"模型没答好"。

### 3.3 ask —— 问答

```bat
kb.bat ask "绝缘子出现裂纹怎么办"
kb.bat ask "绝缘子出现裂纹怎么办" -v      REM 打完整 Thought/Action 轨迹
```

模型会自己决定:查知识库(可查多次)、联网搜、还是列文档清单,然后作答。
输出包含答案、检索过程、用到的工具、停止原因、token 用量。

**退出码**:`0` = 正常给出最终答案;`2` = 没给出(轮数用尽/格式一直跑偏),
答案仍会打印,但脚本里调用可以靠这个码发现问题。

### 3.4 写工具(默认全关)

前面四个工具都是**只读**的:即便提示词注入成功,模型也没有可破坏的东西 ——
这是防注入真正的兜底。写工具打破这条保证,所以它们带三道闸,而且**默认全关**。

| 工具 | 干什么 | 闸 |
|---|---|---|
| `export_report` | 把整理好的内容导出成 `exports/` 下的文件 | 白名单 + 确认 + 参数校验 |
| `mark_superseded` | 把一篇文档标记失效 / 撤销标记(只改 payload,不删向量) | 同上 |

打开方式(**两个都得给**):

```bat
kb.bat ask "把结论导出成报告.md" --allow-write
```

只在 `.env` 里写 `AGENT_ALLOW_WRITE=1` **不够**:闸 1 过了,闸 2 还要一个
确认通道,而那条路不挂 confirmer,于是写操作照样一次都成功不了 ——
只会把拒绝理由从「未启用」换成「没有确认通道」。

```bat
set AGENT_ALLOW_WRITE=1
kb.bat ask "把结论导出成报告.md"      REM 仍然是拒绝:没有确认通道
```

三道闸分别在拦什么:

1. **白名单** —— 按 `Tool.kind` 判,不按工具名判(名字可以重名、可以变大小写,
   `kind` 是注册时定死的)。
2. **显式确认** —— 每次写操作在终端单独问一次,默认 N;直接回车 / EOF /
   Ctrl-C 全算拒绝。**「配不出确认通道」不等于「默认同意」**:无头环境
   (批处理、定时任务、API)本来就没人能点头,那里就该拒绝。
3. **参数校验** —— 导出路径必须是**裸文件名**,`resolve()` 之后仍落在
   `exports/` 内,后缀限 `.md`/`.txt`/`.json`/`.csv`。带目录成分的一律拒,
   **不悄悄取 basename**:静默净化会把一次注入尝试伪装成一次正常调用。
   能写 `.bat` 的导出工具等于送了一条执行路径,所以后缀走白名单。

**审计**:每次写调用往 `AGENT_AUDIT_LOG`(默认 `logs/audit.jsonl`)追加三条 ——
`intent`(执行**之前**写,含参数)→ `result`(成败摘要)或 `denied`(被拒),
靠同一个 `call_id` 串起来。三条都要:

- 只记结果不留意图:"进程在写文件那一瞬间崩了"这条路径在日志里就是空白的,
  而它恰恰最需要留痕。
- 只记成功不记拒绝:日志上一切正常,而系统可能正被反复试探 ——
  **越权尝试正是注入的指纹**。

**可逆**:`mark_superseded` 不删数据,只改 payload 里的状态字段,并把原值
存在 `previous_status` 里,再调一次带 `restore` 就回到原状;重复标记是幂等的
(不会把回滚点覆盖掉)。导出走 `.part` 临时文件 + 原子 rename,中途失败不留
半截文件;同名不覆盖,自动加序号。

标记的效果(不再被召回)见下一节。

### 3.5 版本化索引:换了版怎么处理

规程换版是这类知识库的日常。做法是「**先把新版导进来,再把旧版标记失效**」,
不是删旧版 —— 废止的条款以后还要能查。

```bat
python kb.py ingest .\新版规程\            REM 1. 新版入库(旧版仍在)
kb.bat ask "把旧版规程标记失效" --allow-write   REM 2. 旧版标记失效
```

**默认就查不到旧版了**,不需要额外参数。要翻废止条款时:

```bat
set INCLUDE_SUPERSEDED=1
kb.bat search "架空线路巡视周期"
```

前端上这是检索框旁边的「含废止版本」勾选框,勾上只影响**这一次查询** ——
环境变量只是默认值。不给按次覆盖的口,前端就只能让人改配置文件再重启。

字段落在 Qdrant payload 上:

| 字段 | 含义 | 从哪来 |
|---|---|---|
| `status` | `current` / `superseded` | 入库时写 `current`,只有 `mark_superseded` 能改 |
| `doc_version` | 版本号,如 `v2` | 文件名约定 |
| `effective_from` / `effective_to` | 生效/失效日期 | 文件名约定 |

版本字段是从**文件名**认出来的,认得出的写法是 `规程v2_2025-06-01.md` 这种:
`vN`(或 `vN.N`)当版本号,ISO 日期当生效日期,两个日期就是区间。认不出**就不写** ——
错的版本号会被当成真的去比较,比没有更坏。

关于「加字段之前入库的老数据」——这是这块最容易出人命的地方:

- **缺 `status` 的块一律当有效**。代码里(`status_of`)和 Qdrant 过滤条件
  (`current_filter`)用的是同一套语义,后者写成
  `status == 'current' OR status 不存在`。
  只对一处做兼容,结果就是**整个库查不到东西,而且不报错**。
- 想彻底不带那条 OR 分支,跑一次回填(**只改 payload,不重算向量**):

```bat
python scripts\backfill_version.py            REM 先看有多少要补
python scripts\backfill_version.py --apply    REM 真写
```

> 新增 payload 字段时记得看 `store/qdrant_store.py` 的 `PAYLOAD_INDEXES`:
> **已存在的 collection 不会重建**,`ensure_collection` 走的是"存在就返回"
> 那条路。索引没补上的表现是所有查询一起变慢,不报错(所以那里现在会顺手
> 把索引补齐)。

### 3.6 docs / stats / health

```bat
kb.bat docs                    REM 列出所有文档 + 每篇多少块
kb.bat docs --filter 规程       REM 按文件名/标题子串过滤
kb.bat stats
kb.bat health
```

各命令退出码:`health` 后端不可用返 1;`search`/`docs` 没结果返 1;
`ingest` 有文件出错返 1;`ask` 见上。

---

## 4. 网页界面

有两套,**能力对齐**(`api/CONTRACT.md` 附录有一张逐控件比过的对照表):

| | 什么时候用 |
|---|---|
| **Gradio**(`webui.py`) | 调试首选。少一层构建、少一个进程,排查"是后端不对还是前端不对"更快 |
| **React**(`frontend/`) | 日常用。能力一样,但检索各路得分、问答的中间步骤、引用回溯都做得细 |

两边**不要同时对同一个库做写操作** —— 后端单例是每进程一份,跨进程没有锁。

### 4.1 Gradio

不想记命令就用这个,四个页签对应上面 §3 的命令。

```bat
cd /d D:\agentic-kb
webui.bat
```

浏览器会自动打开 <http://127.0.0.1:7860>。参数:

| 参数 | 用途 |
|---|---|
| `--port 7861` | 换端口(7860 被占时) |
| `--no-browser` | 不自动开浏览器 |
| `--share` | 生成公网临时链接。**等于把知识库内容暴露到公网**,用前先想清楚 |
| `-v` | 打印调试日志(会带出本项目自己的 INFO,第三方噪声仍然是压掉的) |

#### 四个页签

| 页签 | 干什么 | 要 LLM key 吗 |
|---|---|---|
| **导入文档** | 上传文件 或 填服务器路径 → 分块/嵌入/落库 | 要(补定位语);没有也能导,只是会标「缺定位语」 |
| **检索** | 只查库看召回,不调 LLM | **不要** |
| **问答** | ReAct 智能体:自己决定查库/联网/多轮检索 | 要 |
| **库状态** | 文档清单、块数、体检 | 不要 |

顶部有个**后端**单选(混合检索 / GraphRAG),四个页签共用 —— 它同时会去改
ReAct 的工具表来源,所以页面上的选择和 `kb.py --backend` 是一回事。

#### 上传的文件会先拷进 `uploads/`

不是多此一举。块里记的 `source` 就是**导入那一刻的文件路径**:直接拿 Gradio
给的临时路径去导,每篇文档的出处都会变成
`C:\Users\...\Temp\gradio\a1b2c3\xxx.pdf` —— 不能看、不能按名字过滤、
重导时路径还对不上,`docs --filter` 也就废了。所以先按原始文件名落到
`uploads/`,再从那里导入。

同名文件会覆盖,但**内容变了会提示**(哈希不同),不会静默吃掉你改过的版本。

#### 导入会花钱,界面上会告诉你花了几次

上下文增强(定位语)是**每块一次** LLM 调用。导入结束后界面上会写
「本次共发出 N 次 LLM 调用」,拿的是 LLM 客户端的全局计数器做的差 ——
不是估计值。嵌入和重排(BGE-M3 / bge-reranker-v2-m3)全程在本地,不计费。

同一个文档**内容没变就不会重算**,`content_hash` 相同的直接跳过,所以
反复点「导入」不会重复计费。想强制重算就勾「强制重写」。

#### 为什么并发被限成 1

见 [architecture.md 的「单例与并发」](architecture.md#3-单例与并发)。

### 4.2 React(FastAPI + Vite)

两个进程,后端先起:

```bash
python scripts/api_server.py     # 127.0.0.1:8000
cd frontend && npm install && npm run dev    # http://localhost:5173
```

浏览器开 **`localhost:5173`**,别换成 `127.0.0.1` —— Vite 绑的是 `::1`,换地址
会直接被拒,和防火墙无关。调前端时**不要**给 `api_server.py` 加 `--reload`
(uvicorn 会重建后端单例,每次都重新加载 bge-m3)。

并发限成 1 这件事在 React 这边是**前端自己排队**(`frontend/src/api.js`),
理由和上面 §4.1 一样;哪些请求不走队列、为什么,那份文件的开头写了。
端口、构建、`null` vs `0.0`、引用回溯这些细节见 `frontend/README.md`。

---

## 5. 从零到能问答

```bat
REM ① 起服务(两个窗口)
scripts\start_qdrant.bat
scripts\start_neo4j.bat

REM ② 体检
kb.bat health

REM ③ 导入
kb.bat ingest D:\你的文档目录 -f

REM ④ 确认进去了
kb.bat docs
kb.bat stats

REM ⑤ 确认检索能召回
kb.bat search "一个你确定文档里有的短语"

REM ⑥ 填 .env 里的 LLM_API_KEY,然后
kb.bat ask "你的问题" -v
```

**切 GraphRAG:** 改 `.env` 的 `KB_BACKEND=graphrag`,或临时 `--backend graphrag`。
切了之后**必须重新导入**(图那一路要跑实体抽取,是导入时做的),这次会花 LLM 钱:

```bat
kb.bat ingest D:\你的文档目录 --backend graphrag
```

两个后端共用同一套分块和同一批向量,切换不会换掉块,只是多了图那一路召回。
图模式下最后仍会并上向量那一路 —— 因为查询词对不上任何实体名时图会返回空,
而那类问题恰恰是向量检索最擅长的。

## 6. 没配 LLM key 会怎样

系统**不会坏,只会降级**。三处功能共用这一个 key:

| 功能 | 没 key 时 |
|---|---|
| 导入时的上下文定位语 | 跳过,块前面少一句话,召回质量下降;填上后重导可补齐 |
| GraphRAG 实体抽取 | 跳过,图是空的 |
| `ask` | 明确报错并提示你先用 `search` 验证检索 |

`health` / `ingest` / `search` / `docs` / `stats` 全部照常,一个 API 调用都不发。

## 7. 出问题怎么查

**排查顺序固定是 `health` → `search` → `ask`。** 这个顺序能直接把问题
切成三块:服务/配置 、检索 、模型。

| 现象 | 多半是 |
|---|---|
| `health` 报 Qdrant 不可用 | 服务窗口关了 → 跑 `scripts\start_qdrant.bat` |
| graphrag 下报 Neo4j 连不上 | 同上,跑 `scripts\start_neo4j.bat` |
| `search` 说"没有检索到任何内容" | 库里是空的,先 `kb.bat docs` 确认;注意**只要库里有块就一定会返回至少一条**(见下面「阈值」那条) |
| `search` 结果不相关 | 换陈述式短语,别用整句问句;`--raw` 看模型视角 |
| `ask` 报 LLM_API_KEY 没配 | 填 `.env`,见 §2 |
| `ask` 答得不对但 `search` 是对的 | 是模型/提示词的问题,不是检索 |
| graphrag 下 `ask` 像没用上图 | `kb.bat stats --backend graphrag` 看 `entities` 是不是 0 |
| 输出里一堆 `pre tokenize: 100%\|…` | 正常。那是 bge-m3 编码的进度条,不是错误,忽略即可 |
| 中文输出乱码 | 没用 `kb.bat`,而是直接跑了 `python kb.py` 且没设 `PYTHONIOENCODING` |
| `ModuleNotFoundError: dotenv` | 用了裸 `python`。必须 py310,见 [architecture.md §3](architecture.md#3-环境) |
| 首次导入卡在下载模型 | 首次要下 bge-m3,见 [architecture.md §3](architecture.md#3-环境) |

## 8. 配置(.env)

改完 `.env` **下一次跑 `kb.bat` 就生效**,不用重启任何东西
(配置是进程启动时读的;Qdrant / Neo4j 那两个服务根本不读 `.env`)。

| 键 | 默认 | 说明 |
|---|---|---|
| `LLM_API_KEY` | *(空)* | **待填**。三处共用 |
| `LLM_BASE_URL` / `LLM_MODEL` | DeepSeek | 见 §2 表格 |
| `KB_BACKEND` | `hybrid` | `hybrid` \| `graphrag` |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://127.0.0.1:6343` / `agentic_kb` | |
| `QDRANT_EXE` / `QDRANT_STORAGE` | 见 `.env` | 只给 `start_qdrant.bat` 用 |
| `NEO4J_URI` / `_USERNAME` / `_PASSWORD` / `_DATABASE` | 本地 bolt:// | 已装好,密码 `agentic-kb-local` |
| `TAVILY_API_KEY` | 已配 | 联网搜索 |
| `RERANK_MIN_SCORE` / `RERANK_MIN_KEEP` | `0.05` / `1` | 见下面「阈值」那条 |
| `AGENT_MAX_ITER` / `AGENT_TOOL_MAX_CHARS` | `8` / `8000` | ReAct 轮数、单条观察上限 |
| `KB_DEVICE` | 自动 | `cuda` / `cpu` |
| `KB_HF_ONLINE` | 关 | 首次下模型时设 `1`,下完关掉 |

`.env.example` 里每一项都有注释,含上面这些的备选写法。

### 阈值:`rerank_min_keep` 默认 1,所以"检索不到"几乎不会发生

bge-reranker 过 sigmoid 后分布**极尖**:相关文档 0.91~0.98,不相关的全在 0.04
以下。所以固定取 top-5 会把 4 条近乎零分的垃圾塞给模型,才有了
`RERANK_MIN_SCORE=0.05` 这道阈值。

但候选全低于阈值时仍会保留 top1 —— 因为空列表会被上层误读成"检索故障"。
**副作用:只要库里有块,任何查询都会返回至少一条。** `search` 返回空
只意味着**库是空的**。想让不相关的结果真的返回空,把 `RERANK_MIN_KEEP` 设 0。

## 9. 自检脚本(都不花钱)

这些脚本**不用 LLM key**,也**不发网络请求**,跑完自己清理测试数据。

```bat
python scripts\check_embed.py        REM 嵌入层:稠密/稀疏维度、批处理
python scripts\check_store.py        REM Qdrant 读写
python scripts\check_ingest.py       REM 加载→分块→去重→落库
python scripts\check_retrieval.py    REM 混合检索 + 重排
python scripts\check_graphrag.py     REM 图写入→多跳→反查块→合并重排(需要 Neo4j)
python scripts\check_agent.py        REM 解析器 / 工具层 / ReAct 循环
```

注意直接 `python` 跑不起来(裸 python 没 dotenv)。用 py310 全路径:

```bat
C:\Users\Administrator\anaconda3\envs\py310\python.exe scripts\check_agent.py
```

`check_graphrag.py` 和 `check_agent.py` 用**独立的 Qdrant collection**,
Neo4j 那侧只用 `selftest_*` 前缀并按 doc_id 精确删除 —— **不碰正式数据**。

跑完全套(含 `check_eval.py`):

```bat
C:\Users\Administrator\anaconda3\envs\py310\python.exe scripts\check_eval.py
```

> `scripts\eval_gen.py`(出题)**不是**自检脚本,它调 LLM 花钱。
> 见 [evaluation.md](evaluation.md)。
