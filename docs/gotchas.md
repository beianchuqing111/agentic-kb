# 踩过的坑(改动前先读)

这份清单里每一条都**实际发生过**,而且几乎每一条的失败方式都是
**不报错**——所以不能靠"跑一遍看看"发现。改相应模块之前先扫一眼。

---

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

以下几条是建 `eval/` 时踩出来的,但**每一条都同时是检索侧的性质**,
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
