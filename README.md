# agentic-kb

`D:\agentic-kb` —— 本地知识库。混合检索(稠密 + 稀疏 + RRF + 重排)打底,
可选切成 GraphRAG,上面架一个 ReAct 智能体:能查库、能联网(Tavily)、能多轮追问。

```
                     ┌──────────────── ReAct 智能体 ────────────────┐
   用户提问 ─────────▶│  Thought → Action → Observation → … → 答案   │
                     └───────┬──────────────┬──────────────┬────────┘
                             │              │              │
                  search_knowledge_base    │        list_documents
                             │        search_web           │
                             ▼              ▼              ▼
                    ┌─────────────┐   ┌──────────┐   ┌──────────┐
                    │ 后端(可切)  │   │ Tavily   │   │ Qdrant   │
                    └──────┬──────┘   └──────────┘   │ 文档清单  │
                           │                         └──────────┘
              ┌────────────┴────────────┐
              ▼                         ▼
        hybrid 后端                 graphrag 后端
   稠密+稀疏(bge-m3)→RRF        ┌─ 实体向量 → 多跳展开 → 反查块
        →bge-reranker 重排        └─ 向量那一路(同上)
              │                         │
              ▼                         ▼
           Qdrant                 Qdrant + Neo4j

   写工具(默认全关) ──▶ 三道闸 ──▶ export_report / mark_superseded
                        白名单 → 显式确认 → 参数校验 ──▶ logs/audit.jsonl
```

---

## 一分钟版

```bat
cd /d D:\agentic-kb
kb.bat health                              REM 体检,先看这个
kb.bat ingest D:\docs -f                   REM 导入(文件或目录都行)
kb.bat search "绝缘子破损判据"              REM 只检索,不花钱
kb.bat ask "绝缘子出现裂纹怎么办" -v        REM 问答,要 LLM key

webui.bat                                  REM 或者:图形界面,浏览器里点
```

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

### ingest —— 导入

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

### search —— 只检索

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

### ask —— 问答

```bat
kb.bat ask "绝缘子出现裂纹怎么办"
kb.bat ask "绝缘子出现裂纹怎么办" -v      REM 打完整 Thought/Action 轨迹
```

模型会自己决定:查知识库(可查多次)、联网搜、还是列文档清单,然后作答。
输出包含答案、检索过程、用到的工具、停止原因、token 用量。

**退出码**:`0` = 正常给出最终答案;`2` = 没给出(轮数用尽/格式一直跑偏),
答案仍会打印,但脚本里调用可以靠这个码发现问题。

### 写工具(默认全关)

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

> ⚠️ 目前 `mark_superseded` 只**写**状态字段,**召回侧还没有按它过滤**
> (那是版本化索引那一批的事),所以标记过的文档仍会被检索到。工具描述里
> 也是这么写的 —— 别在描述里承诺做不到的事。

### docs / stats / health

```bat
kb.bat docs                    REM 列出所有文档 + 每篇多少块
kb.bat docs --filter 规程       REM 按文件名/标题子串过滤
kb.bat stats
kb.bat health
```

各命令退出码:`health` 后端不可用返 1;`search`/`docs` 没结果返 1;
`ingest` 有文件出错返 1;`ask` 见上。

---

## 4. 网页界面(Gradio)

不想记命令就用这个,四个页签对着下面 §3 的三档能力。

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

### 四个页签

| 页签 | 干什么 | 要 LLM key 吗 |
|---|---|---|
| **导入文档** | 上传文件 或 填服务器路径 → 分块/嵌入/落库 | 要(补定位语);没有也能导,只是会标「缺定位语」 |
| **检索** | 只查库看召回,不调 LLM | **不要** |
| **问答** | ReAct 智能体:自己决定查库/联网/多轮检索 | 要 |
| **库状态** | 文档清单、块数、体检 | 不要 |

顶部有个**后端**单选(混合检索 / GraphRAG),四个页签共用 —— 它同时会去改
ReAct 的工具表来源,所以页面上的选择和 `kb.py --backend` 是一回事。

### 上传的文件会先拷进 `uploads/`

不是多此一举。块里记的 `source` 就是**导入那一刻的文件路径**:直接拿 Gradio
给的临时路径去导,每篇文档的出处都会变成
`C:\Users\...\Temp\gradio\a1b2c3\xxx.pdf` —— 不能看、不能按名字过滤、
重导时路径还对不上,`docs --filter` 也就废了。所以先按原始文件名落到
`uploads/`,再从那里导入。

同名文件会覆盖,但**内容变了会提示**(哈希不同),不会静默吃掉你改过的版本。

### 导入会花钱,界面上会告诉你花了几次

上下文增强(定位语)是**每块一次** LLM 调用。导入结束后界面上会写
「本次共发出 N 次 LLM 调用」,拿的是 LLM 客户端的全局计数器做的差 ——
不是估计值。嵌入和重排(BGE-M3 / bge-reranker-v2-m3)全程在本地,不计费。

同一个文档**内容没变就不会重算**,`content_hash` 相同的直接跳过,所以
反复点「导入」不会重复计费。想强制重算就勾「强制重写」。

### 为什么并发被限成 1

`retrieve.backends.get_backend` 是个**模块级单例**,ReAct 的工具表也是构造时
从它取的。Gradio 默认会并发跑多个请求,那就成了一边导入一边问答、两边共用同一个
后端和同一个 Qdrant store。本机单人用,串行反而是对的 —— 界面不会因为你手快
连点两下就出错。

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

两个后共用同一套分块和同一批向量,切换不会换掉块,只是多了图那一路召回。
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
| `ModuleNotFoundError: dotenv` | 用了裸 `python`。必须 py310,见 §13 |
| 首次导入卡在下载模型 | 首次要下 bge-m3,见 §13 |

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

跑完全套(含上面那节说的"不花钱"部分):

```bat
C:\Users\Administrator\anaconda3\envs\py310\python.exe scripts\check_eval.py
```

---

## 9.5 检索质量评估(`eval/`)

**先说钱。这一节的两个脚本,一个花钱一个不花:**

| 脚本 | 花钱? | 干什么 |
|---|---|---|
| `scripts\eval_run.py` | **不花** | 跑已有问答集,算指标,存/比基线 |
| `scripts\eval_gen.py` | **花**(调 LLM 出题) | 从语料生成待筛的问答初稿 |

上面 §9 那节的小标题是"都不花钱" —— 那是自检脚本的性质,`eval_gen.py` **不是**
自检脚本,别照着那节的习惯随手跑。它每次调用都要过 `--dry-run` 看计划。

### 它解决什么问题

前 6 个 `check_*.py` 验证的是**流程对不对**,验证不了**召回质量好不好**。
改分块大小、换 reranker、调 `rerank_min_score`、改 RRF 的 `k`、加/去定位语 ——
在评估集出现之前,判断好坏的唯一依据是打开界面看几条。有了它,
这些决策从"双方讲道理"变成"改前后各跑一次,看 `hit@5` 和 `by_type` 里的数字"。

### 三条用法

```bat
REM 一、多个 spec 并排跑 —— 最快的用法,一次进程加载模型
REM   注意 preset 是**用 + 拼成一个 spec**,不是两个独立的 flag
python scripts\eval_run.py --preset no-rerank+no-threshold+wide --preset no-threshold+wide

REM 二、存基线再改一个旋钮,看涨跌
python scripts\eval_run.py --preset no-threshold+wide --save-baseline before
python scripts\eval_run.py --preset no-rerank+no-threshold+wide --baseline before

REM 三、回答"定位语到底值多少召回" —— 独立 collection,两种都要重导一次
python scripts\eval_run.py --no-contextual --save-baseline noctx

REM 四、换语料:先出题(花钱),人工筛,再跑(不花钱)
python scripts\eval_gen.py --corpus D:\some\docs --limit 30 --dry-run
python scripts\eval_gen.py --corpus D:\some\docs --limit 30 --yes
REM   ↑ 产物是 eval\qa\<slug>.draft.jsonl,全是 status="draft",**不是评估集**
REM     人工把要留的行复制进 eval\qa\<slug>.jsonl 并改成 status="keep"
python scripts\eval_run.py --corpus D:\some\docs
```

### 挑旋钮挑**真会动**的那个(实测)

种子上"重排开/关"的实测差(其余全同):

| @1 | 重排关 | 重排开 |
|---|---|---|
| hit / mrr | 0.867 | **1.000** |
| ndcg | 0.790 | 0.962 |
| recall | 0.678 | 0.811 |
| **contextual 的 hit@1** | **0.500** | **1.000** |

那一行 `contextual` 就是"定位语 + 重排"合起来值多少钱的具体证据 ——
代词密集段(「该系统」「上述参数」)本身就是靠重排救回来的。

**反面教材:`rrf_k` 在这个语料上根本不动。** 计划里原以为它是"保证会动的
干净演示"(依据是 `store/qdrant_store.py` 那段 k=2 赢者通吃的说明),实测
`--set rrf_k=2` 与 k=60 **逐题完全一致**。原因不是旋钮没接线 ——
`check_store.py` 把两条路的分数打出来后一眼可辨:

```
k=60:  d1 = 1/60+1/60 = .0333    d2 = 1/61+1/61 = .0328
k=2:   d1 = 1/2 +1/2  = 1.0      d2 = 1/3 +1/3  = .667     ← 顺序完全一样
```

RRF 的 k **只改分数的刻度,不改名次**,除非稠密与稀疏两路**互相不同意**。
这个语料的词法和语义排序高度一致,所以 k 怎么调名次都不动。要看它的效果,
得找一个两路分道扬镳的查询 —— 或直接看 `top1_score_mean`(它确实变了:
无阈值下 0.033 是 RRF 分的量级)。

> 这条正是这套设施的价值所在:它把一个"想当然会动"的旋钮变成了
> "拿数字证明它不动"。报告里那句 `(逐题无变化 —— 若聚合也没动,
> 先怀疑配置没生效)` 就是逼你去分辨"真没动"还是"没接线"。

`--dry-run` 一次 API 都不发,只读语料、算抽样、打印调用数。超过 20 次调用
要求显式 `--yes`。

### 四条规矩(改这块之前必须知道)

1. **`k` 有天花板。** `effective_k = min(top_k, fusion_top_k, dense_top_k,
   sparse_top_k)`,不是 `top_k` 想多大就多大。`--top-k 10` 配默认
   `fusion_top_k=5` 时 `hit@10` 是**幻觉数**。runner 会告警并指名两个数字,
   `effective_k` 也记进基线;超过它的 k **从报告里删掉**,不打成 0。

2. **头条数字在"阈值中性"配置下算。** 开着 `rerank_min_score` 时
   `recall@10` 测的**根本不是排序器**,是阈值。两个数字印在一起,人就会得出
   "新重排器更差了"而其实只动了阈值。所以头条用 `no-threshold`
   (`rerank_min_score=0.0, rerank_min_keep=0`),`shipped` 并排报。

3. **反例独立成族,绝不进入任何平均。** 反例的 `expected` 是空的,
   混进平均会给每个数贡献恒定的 0 —— 既稀释,又**对参数变化完全不可见**。
   它们单独按分数分布评(`leak@k` / `top1_score_mean`)。另外
   LlamaIndex 那几个 metric 遇到空 `expected` 会直接 `ValueError`,
   想混也混不进去。

4. **比对模式拒绝跨不可比来源做 diff。** `qa.sha1`、`chunk_size`、
   `contextual_enabled`、`embed_model`、`rerank_model` 任一不同就拒绝并
   指名哪个字段不同。这不是洁癖 —— "改了一行问题、重跑、然后得出
   '检索器变了'的结论"是**必然会发生**的失败,`qa.sha1` 存在的全部理由
   就是拦它。同理 `by_type` 那种 3 题一类的均值是噪声,报告里每类都打 `n`。

### 指标口径

可答题:`hit@k` / `mrr@k` / `ndcg@k`(分级 + 二值并排)/ `precision@k` /
`recall@k`,**按题宏平均**。`--ks 1,3,5,10`,其中 **5 = 出厂
`rerank_top_n`,也就是你今天实际拿到的量**。

指标全部**自己写**(`eval/metrics.py`,纯函数,无模型无网络),不 import
LlamaIndex 的。原因见 §11 的 `Precision` 那条 —— 它的分母是"返回条数"而不是
`k`,方向是反的。LlamaIndex 的实现只在 `check_eval.py` 里当交叉验证的参照。

`no-threshold` 那档还有个诊断项 `nonempty_rate`,**它是 `rerank_min_keep=1`
的产物,不是质量**(见 §8 那条阈值说明)—— 打出来是为了让那个 1.0 有出处。

---

## 10. 架构要点

### 两个后端,一套块

`hybrid` 和 `graphrag` 用**同一套分块、同一批向量**(见 `ingest/pipeline.py`
的 `prepare_document`,两条路径共用)。切换后端不会换掉块,只是多了图那一路召回。
`graphrag` 后端最后仍会并上向量那一路 —— 因为图有个硬伤:查询词对不上任何
实体名时会返回空,而那类问题恰恰是向量检索最擅长的。

### 为什么写入侧不用 `PropertyGraphIndex.insert_nodes`

检索侧确实用 LlamaIndex(`PropertyGraphStore.vector_query` / `get_rel_map`),
但图**写入**是自己写的,四个实测理由记在 `ingest/graph_pipeline.py` 开头:
块被编码两遍、实体向量被元数据污染、每次 insert 刷全图 schema、同一实体编码 N 次。

### ReAct 为什么是文本协议而不是 function calling

`llm/client.py` 是纯文本对话客户端,表达不了 `tool_calls` / `role="tool"`。
硬走原生工具调用就得把客户端改成多态的,还要处理「有的兼容端点支持、
有的不支持」。文本协议的代价是格式可能跑偏,所以 `agent/react.py` 里有三道纠错:
`stop` 序列截断、格式纠错重试(上限 `AGENT_FORMAT_RETRIES`)、
最后把散文当答案返回而不是报错。

### 工具返回的一切都不可信

知识库正文来自导入的文档,网页来自公网,两者都可能写着「忽略以上指令」。
所以工具的 Observation 都夹在 `<<<UNTRUSTED_DATA …>>>` 之间,系统提示词
明确声明那是资料不是指令。这层不是银弹,真正的兜底是三个工具都**只读**。

## 11. 踩过的坑(改动前先读)

- **块的 `doc_id` 曾经全是字符串 `"None"`。** LlamaIndex 的
  `node_to_metadata_dict` 会执行 `metadata["doc_id"] = node.ref_doc_id or "None"`,
  把我们填的 doc_id **原样覆盖**。不设 `NodeRelationship.SOURCE` 就会中招,
  后果是删文档返回 0、重导不清旧块,图里慢慢堆起互相矛盾的旧数据**而全程不报错**。
  现在靠 `GraphIngestPipeline._chunk_node` 统一设 SOURCE 关系,别再各写一遍构造。

- **「块提到实体」的 MENTIONS 边必须全量写。** 实体上的 `triplet_source_id`
  是**单值属性**,靠它连边只能连出一条。边就是证据,去重即删证据 ——
  编码可以去重(那才贵),边不行。见 `store/graph_store.link_mentions`。

- **空库检索必须自己拦,不能让它问到 Qdrant。** 装完还没 `ingest` 就
  `kb.py search "…"`,Qdrant 直接抛 404 `Collection doesn't exist`,
  CLI 打一整屏 traceback —— 而 `cmd_search` 里明明备好了「库里可能还没有文档」
  那句友好提示,只是永远走不到。两个后端现在都在 `retrieve` 开头
  `if not self.store.exists(): return []`。**检索是只读操作,空库是正常状态。**

- **`rerank_min_keep` 默认 1,所以「检索不到」几乎不会发生。** 候选全低于阈值时
  仍保留 top1(空列表会被上层误读成检索故障)。也就是说只要库里有块,
  任何查询都会返回至少一条。`search` 返回空只意味着**库是空的**。
  这个行为在 `scripts/check_agent.py` 里被显式验证(用空库跑那一路)。

- **argparse 的全局开关不会自动认子命令后面的位置。** `kb.py ask "x" -v`
  会报 `unrecognized arguments: -v`,而人写命令的习惯就是把开关放最后。
  修法是把 `--backend`/`-v` 也挂到每个子 parser 上(见 `kb.py`
  的 `_add_global_flags`),但子 parser 那份的 `default` **必须是
  `argparse.SUPPRESS`** —— 否则 `kb.py --backend graphrag stats` 里,
  子 parser 会用自己那份默认值 `None` 把主 parser 解析好的 `graphrag`
  悄悄覆盖掉,`--backend` 就此静默失效。

- **`.bat` 文件必须纯 ASCII。** cmd.exe 按 OEM 代码页(这里是 936)解析批处理,
  UTF-8 的中文会被拆碎、甚至破坏行边界,于是 cmd 去**执行**注释的碎片,
  刷出一屏「'xxx' 不是内部或外部命令」。

  这条**实测复现过**,不是理论风险:`scripts\start_qdrant.bat` 原来带中文注释,
  从默认控制台跑会刷出 `'/6344)+'`、`'drant.exe'`、`'ot'` 这种错误
  —— 词被从中间切断,说明行边界真的破了。当时以为"没出问题",只是因为
  在你已经 `chcp 65001` 过的窗口里跑,换个窗口/双击就炸。

  **在开头加 `chcp 65001` 并不能修**(我试过):cmd 是**打开文件时**用当时的
  代码页解码整个文件的,不是逐行按当前代码页解码,所以第二行再切代码页已经
  晚了——那几行中文 echo 照样被拆。唯一的修法是让文件本身是 ASCII。
  现在 `kb.bat` 和 `scripts\start_*.bat` 全部纯 ASCII,改之前先想清楚这点。

- **必须用 py310 的完整路径。** 裸 `python` 会 `ModuleNotFoundError: dotenv`。
  `kb.bat` 里写死了路径,可用 `KB_PYTHON` 覆盖。

- **`HF_HUB_OFFLINE=1` 由 `config.py` 自动设**(`setdefault`)。它必须在
  huggingface 相关库被导入**之前**生效,所以 import 顺序是:
  先 `config`,再 `embed`/`retrieve`。为什么非关不可:模型已完整缓存在本地时,
  `BGEM3FlagModel` 加载**仍然**会向 huggingface.co 发一次 HEAD 查最新版本;
  这次请求的成败和模型能不能加载**毫无关系**,但它失败时异常会一路冒出来
  把整篇文档的导入搞挂。首次下载模型时用 `KB_HF_ONLINE=1` 打开,下完关掉。

- **别在导入流程里调 `g.clear()`。** Neo4j Community 只有一个库,
  clear 会连正式数据一起清掉。删文档走 `delete_doc_graph(doc_id)`。

- **块的 metadata 里不能出现 `nodes` / `relations` 这两个键名。**
  `KG_NODES_KEY` / `KG_RELATIONS_KEY` 的字面值就是它们,抽取器会把它们 pop 掉。

- **第三方噪声要按 logger 名压,而且要放在模块顶 —— 理由是"会被当模块导入",
  不是"导入期就会打"。** 两条烦人的话是
  `torchao: Skipping import of cpp extensions ...` 和
  `torch.distributed.elastic: Redirects are currently not supported in Windows`。

  它们确实都写 **stderr**、都走 `logger.warning`、logger 名就是
  `torchao` / `torch.distributed.elastic`,所以 `setLevel(ERROR)` 压得住。
  但**触发时机不是 `import webui`**:实测 `import webui` 跑完,
  `sys.modules` 里既没有 `torchao` 也没有 `torch.distributed.elastic`,
  真正的触发点是**第一次构造嵌入模型**时由 `FlagEmbedding` 把 torchao 拖进来。
  (我一开始把原因写成"`from retrieve.backends import` 在导入期就打了",
  是错的 —— 记在这里免得下次又想当然。)

  放在模块顶的真正理由是:`webui.py` 会被当**模块**导入(自检脚本,
  以及直接调 handler 做验证的那条路),那条路不经 `main()`,
  写在那里就等于没写。

  另一半是个真会咬人的细节:`logging.basicConfig` 在根 logger 已有 handler 时
  是 **no-op**。模块顶 basicConfig 过一次之后,`_setup_logging` 里再调它
  **改不了级别**,必须显式 `logging.getLogger().setLevel(...)` ——
  否则 `-v` 静默失效,人会以为"verbose 没用"。

- **界面上的数字宁可不报,也不能报得比实际乐观。** 导入结语原来写死一句
  「没有 LLM 调用」,但上下文增强是**每块一次**调用 —— 配了 key 的用户导
  4 个块就花了 4 次,界面却说没花。这类"界面比实际乐观"的错最伤信任,
  比报不出来更糟。现在拿 LLM 客户端的全局计数器做差,如实报次数
  (`webui._ingest_cost_note`)。同理 `_llm_calls()` 在读不到时返回 `None`
  而不是 `0`。

- **自检脚本的断言不能跟着本机状态翻脸,失败文案更不能说反话。**
  `check_ingest.py` 原来写死 `assert payload["context"] == ""`、
  理由是「LLM 未配置,优雅降级」—— 这在 `.env` 补上 `LLM_API_KEY` 之后**必然失败**,
  且失败文案是「context 为空」(说的正好是反的),看到的人第一反应是"产品坏了",
  实际只是机器状态变了。**一个随环境翻脸的断言比没有断言更糟:它教人忽略失败。**
  现在按真正的不变式判:`context 为空 ⟺ 没配 LLM`,两边都成立。

- **上下文增强的定位语要喂给模型,不能只喂给嵌入。** 它有两半价值:一半是
  拼进被嵌入的文本以提升召回(写入侧),另一半是让模型知道「这块在原文的
  什么位置、讲的是哪件事」。`RetrievedChunk.context` 一直有值,但
  `agent/tools.py` 的 `format_hits` 早期压根没引用它 —— 模型看不到。
  块往往很短(一条判据、一行参数),少了这半,模型判断不出该不该信,
  容易把孤立数字当结论。现在 Observation 里会多一行 `定位: …`,
  前端「检索」页签也会用引用块显示。

以下五条是建 `eval/` 时踩出来的,但**每一条都同时是检索侧的性质**,
不只在评估里成立:

- **`dense_rank` / `sparse_rank` 是文档级,不是块级。** `retrieve/hybrid.py`
  里 `d_rank = {pid: i for i, pid in enumerate(debug.dense_order)}`,而
  `dense_order` 装的是 **doc_id**。同一篇文档的多个块拿到**完全相同**的
  rank。拿它做块级命中或名次判断会得出错误结论 —— 它只能回答
  "这条是稠密还是稀疏捞上来的"。块级判定只能用 `doc_id + chunk_index`。

- **锚点失效 ≠ 检索变差。** 评估集的 gold 是 `(文件, chunk_index)`。
  在目标块**之前**增删字符会让 chunk_index 整体位移,锚点指向别的块,
  分数掉下来 —— 但检索器没问题,是评估集过期了。混为一谈会让人去调一个
  没坏的参数。所以 gold 里还存了目标块正文首 30 字,开跑前比对,
  对不上就报"锚点失效"并**拒绝出分**,而不是照常打印一张误导性的低分表。

- **LlamaIndex 的 `Precision` 分母是"返回条数"而不是 `k`,会系统性虚高。**
  `1/len(retrieved_set)`。实测:retrieved=`["d1"]`、expected=`["d1"]`、k=3 时
  它给 `1.0`,而我们给 `1/3`。也就是说**返回得越少它越高兴,方向是反的** ——
  在 `rerank_min_keep=1` 下它几乎永远报满分。它也不接受 `k` 参数,
  且空 `expected` 会直接 `ValueError`(反例喂不进去)。所以评估指标全部自己写。

- **`get_reranker()` 是模块级单例,第二次起忽略 `cfg`。**
  `if _reranker is None:` 才读配置,所以同进程内 `--set rerank_model=…`
  **静默不生效** —— 这是评估工具最经典的坑。`rerank_min_score` /
  `rerank_min_keep` 是安全的(`apply_rerank` 从检索器的 cfg 读)。

- **RRF 的 `k` 是刻度旋钮,不是排序旋钮。** `store/qdrant_store.py` 开头那段
  说 k=2 赢者通吃是对的,但**只对分数刻度成立** —— 实测 k=2 与 k=60 在本语料上
  名次逐题一致。k 要改到名次,前提是稠密与稀疏两路**互相不同意**;
  两路排序一致时(本语料正是)任何 k 都得出同一个顺序。所以别拿它当
  "保证会动"的对照,那会让你以为配置没生效。`check_store.py` 的
  `[5]` 段就是把两种 k 的顺序和分数并排打出来的地方(它自己也带一句
  "小样本下有可能真的相同,但值得警惕"的提示)。

- **`no-rerank` 与 `no-threshold` 是两回事,别混。** `no-threshold` 是
  `rerank_min_score=0.0 / min_keep=0`,**重排仍然开着**。要真关掉重排、
  省下 2.3G 模型加载,用 `no-rerank`。两者可以叠加:
  `--preset no-rerank+no-threshold+wide`。

- **定位语开关必须进 collection 名。** 否则第二次导入时 `unchanged` 判定
  命中、一个块都不写,你以为在做 A/B,其实两次读的是同一个库 —— 分数一样,
  还会被读成"定位语没用"。名字里带 `ctx` / `noctx` 就杜绝了这种假阴性。

## 12. 目录

```
config.py            所有配置(读 .env,单例 get_settings)
kb.py / kb.bat       命令行入口
webui.py / webui.bat Gradio 前端(四个页签)
uploads/             前端上传的文件落这里(自动建;块里的 source 就是它)
embed/               bge-m3 稠密+稀疏
store/               qdrant_store(块) / graph_store(Neo4j,含 delete_doc_graph)
ingest/              loader → chunker → contextual(定位语) → pipeline / graph_pipeline
retrieve/            hybrid(稠密+稀疏+RRF+重排) / reranker / backends(hybrid / graphrag)
llm/                 OpenAI 兼容客户端(chat / chat_json / map_batch)
agent/               react(文本 ReAct) / tools(三个工具) / websearch(Tavily)
eval/                检索质量评估(见 §9.5)
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
  check_eval.py       评估设施自检(不花钱、不调 LLM)
  eval_run.py         跑评估(不花钱)
  eval_gen.py         出题(花钱 —— 先 --dry-run)
qdrant_storage/      Qdrant 数据(别手删)
neo4j/               Neo4j 5.26.9 + 自带 JRE + downloads(安装包缓存)
```

## 13. 环境

- Python:**anaconda py310**(`C:\Users\Administrator\anaconda3\envs\py310`)
- **torch 不要重装** —— py310 里已有 2.9.1+cu130,CUDA 可用(RTX 5080);
  换成 CPU 版会让 bge-m3 慢十倍以上
- bge-m3 和 bge-reranker-v2-m3 已缓存,所以默认离线可用。
  **换机器**首次要下模型:设 `KB_HF_ONLINE=1` 跑一次导入,下完改回
- Qdrant:6343/6344,数据在 `qdrant_storage/`(独立于 imgsearch 在 6333 上的数据)
- Neo4j:Community 5.26.9,自带 JRE,`neo4j/server/data`;
  重建步骤(两个安装包已缓存在 `neo4j/downloads/`)写在
  `scripts\start_neo4j.bat` 顶部注释里
