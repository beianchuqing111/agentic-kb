"""Agent 自检:解析器 → 工具层 → 受控写工具 → ReAct 循环。

**不需要 LLM API key**:ReAct 循环喂的是脚本化的假 LLM(见 ScriptedLLM),
它按事先写好的台词逐轮返回。这样格式解析、工具分发、截断、轮次上限、
重复调用短路这些**逻辑**全都能验,而不用花一分钱。

真模型才能验的只有一件事:它是否愿意按格式输出。那属于提示词质量,
不属于代码正确性 —— 混在一起测只会让「红了」的时候分不清是代码坏了
还是模型今天心情不好。

Qdrant 用独立 collection(agentic_kb_selftest_agent),退出时删掉。
**不碰 Neo4j**:这个自检走 HybridBackend,图那一路单独由
check_graphrag.py 验。

联网搜索不真发请求(省 quota,也不该让自检依赖外网)—— 注入假搜索器,
只验「结果怎么变成 Observation」和「失败/未配置怎么降级」。

跑法:
    set PYTHONIOENCODING=utf-8
    python scripts\\check_agent.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AgentConfig, WebSearchConfig, get_settings  # noqa: E402
from embed import get_embedder  # noqa: E402
from ingest import IngestPipeline, stable_doc_id  # noqa: E402
from llm.client import ChatResult  # noqa: E402
from retrieve.backends import HybridBackend, set_backend  # noqa: E402
from retrieve.hybrid import HybridRetriever, RetrievedChunk  # noqa: E402
from store.versioning import STATUS_FIELD  # noqa: E402
from qdrant_client import models as qm  # noqa: E402

from store import QdrantStore  # noqa: E402
from agent.react import (  # noqa: E402
    STOP_SEQUENCES,
    ReActAgent,
    parse_action,
    parse_final_answer,
    parse_thought,
    truncate_observation,
)
from agent.tools import (  # noqa: E402
    DOCS_TOOL,
    EXPORT_TOOL,
    KB_TOOL,
    MARK_TOOL,
    STATUS_CURRENT,
    STATUS_SUPERSEDED,
    WEB_TOOL,
    Tool,
    ToolRegistry,
    build_default_tools,
    build_registry,
    format_hits,
)
from agent.websearch import WebResult, WebSearchError, WebSearcher  # noqa: E402

TEST_COLLECTION = "agentic_kb_selftest_agent"
DOC_NAME = "自检规程"
DOC_TEXT = (
    "瓷套出现裂纹时必须立即更换,否则在风振下可能断裂。"
    "判据来自运维规程,现场用卡尺测量裂纹深度并记录。"
    "绝缘子破损等级按裂纹长度划分为三级,超过三级立即停电处理。"
)
HIT_QUERY = "绝缘子破损判据"
# 用来跑「库里没有」那一路的 collection:建好但**一个点都不写**
EMPTY_COLLECTION = "agentic_kb_selftest_agent_empty"

# 写工具专用的长文档。**单独一篇,不去撑大 DOC_TEXT** —— 上面那篇是检索
# 用例的夹具,长度和措辞牵着命中/阈值那几条断言,动它等于顺手改了一组
# 已经调好的期望值。而「混合状态」用例非得要一篇多块文档不可
# (chunk_size=512,这篇约 1200 字 → 3 块)。
WRITE_DOC_NAME = "自检缺陷定级细则"
WRITE_DOC_TEXT = (
    "第一条 为规范输变电设备缺陷管理,按缺陷严重程度分为紧急、重大、一般三级。"
    "紧急缺陷是指威胁设备安全运行、随时可能造成事故的缺陷,发现后应立即处理,"
    "处理时限不超过二十四小时;重大缺陷是指短期内可能发展为事故的缺陷,"
    "处理时限不超过七天;一般缺陷是指对安全运行影响较小、可列入计划检修的缺陷,"
    "处理时限不超过一个检修周期。"
    "第二条 缺陷定级由运维班组初判,由运维分部专业工程师复核定级结论,"
    "定级结论应当记录在设备缺陷管理台账中,并同步录入生产管理系统。"
    "定级存在争议时,由运维分部组织专业会商确定,会商记录随缺陷档案一并保存。"
    "第三条 消缺完成后应当由发起人以外的人员进行验收,验收内容包括缺陷现象是否消除、"
    "设备运行参数是否恢复正常、是否存在遗留问题。验收不合格的应当重新纳入缺陷管理流程,"
    "并按原等级重新计时,不得以验收不合格为由延长处理时限。"
    "第四条 因故不能在规定时限内完成消缺的,应当在时限届满前办理延期手续,"
    "说明延期理由、临时管控措施和计划完成时间,经运维分部批准后生效。"
    "延期次数一般不超过一次,延期后的累计时限不得超过原时限的两倍。"
    "第五条 缺陷管理台账应当按月统计、按季分析,统计内容包括缺陷发现数量、"
    "各等级占比、按期消缺率、超期未消缺数量以及重复性缺陷情况。"
    "重复性缺陷应当开展专项分析,查明根因,不得简单以消缺了事。"
    "第六条 本细则由运维分部负责解释,自发布之日起施行。"
)

# 同一部规程的两版,用来验版本化召回。**两版对同一个问题都能答**是刻意的:
# 旧版要是本来就不相关,"标记失效后不再被召回"什么都证明不了 —— 它本来
# 也不会出现。两版内容只在数字上有区别,和真实换版的情形一致。
# source 里带 `v1_2024-01-01` 这种写法,是为了顺带验"文件名约定 → 版本字段"
# 这条链路真的通了。
VER_DOC_V1 = "selftest_agent://架空线路巡视规程v1_2024-01-01"
VER_DOC_V2 = "selftest_agent://架空线路巡视规程v2_2025-06-01"
VER_QUERY = "架空线路的巡视周期是多久"
VER_TEXT_V1 = (
    "架空线路巡视周期规定:城区线路每十五天至少巡视一次,郊区及农村线路每三十天至少一次。"
    "红外测温每季度开展一次,重点测量导线接头、耐张线夹和并沟线夹的温度。"
    "巡视发现的缺陷按缺陷定级流程处理,通道内的树障应当在发现后三十天内清理完毕。"
)
VER_TEXT_V2 = (
    "架空线路巡视周期规定:城区线路每七天至少巡视一次,郊区及农村线路每十五天至少一次。"
    "红外测温每月开展一次,重点测量导线接头、耐张线夹和并沟线夹的温度。"
    "巡视发现的缺陷按缺陷定级流程处理,通道内的树障应当在发现后十五天内清理完毕。"
)


# --------------------------------------------------------------------- #
# 替身
# --------------------------------------------------------------------- #


class ScriptedLLM:
    """按台词逐轮返回的假 LLM。

    接口只需要 `require_configured` 和 `chat` —— 这正是 ReActAgent 用到的
    全部。同时记录每次收到的 messages,用来断言「Observation 确实回灌了」
    「stop 序列确实传了」这类否则很难看出来的事情。
    """

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.calls: list[list[dict[str, str]]] = []
        self.stops: list[object] = []

    @property
    def configured(self) -> bool:
        return True

    def require_configured(self) -> None:
        return None

    def chat(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        json_mode=False,
        stop=None,
    ) -> ChatResult:
        self.calls.append([dict(m) for m in messages])
        self.stops.append(stop)
        if self.script:
            text = self.script.pop(0)
        else:
            text = "Final Answer: (脚本已用尽)"
        return ChatResult(text=text, prompt_tokens=10, completion_tokens=5)


class StubSearcher:
    """假搜索器。可切到「失败」和「未配置」两种降级路径。"""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.queries: list[str] = []

    def search(self, query: str, max_results=None) -> list[WebResult]:
        self.queries.append(query)
        if self.mode == "error":
            raise WebSearchError("Tavily 请求失败: 429 rate limited")
        return [
            WebResult(
                title=f"关于 {query} 的维基条目",
                url=f"https://example.com/{abs(hash(query)) % 10000}",
                content=f"{query} 的定义与常见做法……(此处是网页摘要)",
                score=0.91,
            )
        ]


def act(name: str, arg: str, thought: str = "先查一下") -> str:
    return f"Thought: {thought}\nAction: {name}\nAction Input: {arg}"


# --------------------------------------------------------------------- #
# 1. 纯函数
# --------------------------------------------------------------------- #


def check_parsers(failures: list[str]) -> None:
    print("\n[1] 解析与截断(纯函数,不依赖任何服务)")

    cases: list[tuple[str, object]] = [
        # (输入, 期望的 Final Answer 或 None)
        ("Thought: 够了\nFinal Answer: 答案是 42", "答案是 42"),
        ("**Final Answer**: 加粗也能认", "加粗也能认"),
        ("最终答案:中文标签", "中文标签"),
        # 「答案:」不算 —— Thought 里这么写很常见,误判会让循环早退
        ("Thought: 答案:42\nAction: search_web\nAction Input: x", None),
        ("Thought: 还在想", None),
    ]
    for text, want in cases:
        got = parse_final_answer(text)
        ok = got == want
        print(f"    {'✅' if ok else '❌'} Final Answer {text[:28]!r} -> {got!r}")
        if not ok:
            failures.append(f"parse_final_answer({text[:24]!r}) = {got!r},期望 {want!r}")

    acases: list[tuple[str, tuple[str, str] | None]] = [
        (
            "Thought: t\nAction: search_knowledge_base\nAction Input: 绝缘子",
            ("search_knowledge_base", "绝缘子"),
        ),
        # 反引号包裹 + 输入外层是**弯引号**(开闭字符不同,单独判「首尾同字符」会漏)
        ("Action: `search_web`\nAction Input: “光伏板 清洗”", ("search_web", "光伏板 清洗")),
        # 折行的长查询要**拼起来**而不是截掉第二行 —— 截掉等于丢掉半个检索意图
        (
            "Action: search_knowledge_base\nAction Input: 绝缘子破损\n的判据是什么",
            ("search_knowledge_base", "绝缘子破损 的判据是什么"),
        ),
        # 但模型把下一段思考/下一步续在输入里时必须切掉
        (
            "Action: search_web\nAction Input: 真查询\nThought: 我还想再查一次",
            ("search_web", "真查询"),
        ),
        (
            "Action: search_web\nAction Input: 真查询\n\n补充说明不该进来",
            ("search_web", "真查询"),
        ),
        ("Thought: 只想了一半", None),  # 没有 Action
        ("Action: search_web", None),  # 有 Action 但没有输入 —— 不猜
        # 加粗标签也要认
        ("**Action**: search_web\n**Action Input**: 加粗", ("search_web", "加粗")),
    ]
    for text, want in acases:
        got = parse_action(text)
        pair = (got.name, got.arg) if got else None
        ok = pair == want
        print(f"    {'✅' if ok else '❌'} Action {text[:34]!r} -> {pair}")
        if not ok:
            failures.append(f"parse_action({text[:30]!r}) = {pair},期望 {want}")

    th = parse_thought("Thought: 先查知识库\nAction: search_web\nAction Input: x")
    if "先查知识库" not in th:
        failures.append(f"parse_thought 没抽出 Thought: {th!r}")
    print(f"    {'✅' if '先查知识库' in th else '❌'} Thought -> {th!r}")

    # --- 截断 ---
    short = "短内容"
    out, cut = truncate_observation(short, 100, 0.6)
    if out != short or cut:
        failures.append("短文本不该被截断")
    print(f"    {'✅' if not cut else '❌'} 短文本原样返回")

    long_text = "".join(f"第{i}段。\n" for i in range(4000))  # ≈2.7 万字
    for limit in (1000, 8000):
        out, cut = truncate_observation(long_text, limit, 0.6)
        ok = cut and len(out) <= limit and out.startswith(long_text[:20]) and out.rstrip().endswith("段。")
        head_len = int((limit - len("\n…(中间省略 9999999 字,末尾保留)…\n")) * 0.6)
        print(
            f"    {'✅' if ok else '❌'} 截断 limit={limit}: "
            f"原 {len(long_text)} → {len(out)} 字,头部 {head_len} / 尾部保留"
        )
        if not ok:
            failures.append(
                f"truncate_observation(limit={limit}) 结果 {len(out)} 字,"
                f"cut={cut},超出上限或头尾没保住"
            )
    # 尾部必须真的在 —— 报错信息都在末尾,只留头等于丢掉最该看的
    tail_probe = "A" * 5000 + "!!!关键错误在末尾!!!"
    out, _ = truncate_observation(tail_probe, 1000, 0.6)
    if "关键错误在末尾" not in out:
        failures.append("截断把尾部丢了 —— 报错信息全在末尾,这条必须保住")
    print(f"    {'✅' if '关键错误在末尾' in out else '❌'} 尾部内容保留")


# --------------------------------------------------------------------- #
# 2. 工具层
# --------------------------------------------------------------------- #


def check_tools(failures: list[str]) -> tuple[QdrantStore, QdrantStore] | None:
    s = get_settings()
    emb = get_embedder()

    qcfg = dataclasses.replace(s.qdrant, collection=TEST_COLLECTION)
    store = QdrantStore(cfg=qcfg, retrieval=s.retrieval)
    h = store.health()
    if not h["ok"]:
        print(f"❌ 连不上 Qdrant: {h['error']}\n   先跑 scripts\\start_qdrant.bat")
        return None

    # 知识库工具走 retrieve.backends 的模块单例,这里把它换成测试后端,
    # 免得自检往正式 collection 里写数据
    backend = HybridBackend(store=store)
    set_backend(backend)

    print("\n[2] 工具层")
    store.ensure_collection(dense_dim=emb.dense_dim, recreate=True)
    pipe = IngestPipeline(store=store)
    st = pipe.ingest_text(DOC_TEXT, source="selftest_agent://规程", title=DOC_NAME)
    print(f"    语料写入: {st.summary()}")
    if st.chunks_written < 1:
        failures.append("测试语料没写进去,后面的知识库工具没法验")
    # 写工具那一段要一篇多块文档(「混合状态」用例必须有两块才构造得出来)
    pipe.ingest_text(WRITE_DOC_TEXT, source="selftest_agent://细则", title=WRITE_DOC_NAME)

    searcher = StubSearcher()
    reg = ToolRegistry(
        build_default_tools(
            searcher=searcher,  # type: ignore[arg-type]
            store_override=store,
        )
    )

    # --- 命中 ---
    obs = reg.run(KB_TOOL, HIT_QUERY)
    ok = "UNTRUSTED_DATA" in obs and DOC_NAME in obs and "[1]" in obs
    print(f"    {'✅' if ok else '❌'} 知识库命中({HIT_QUERY}):{len(obs)} 字")
    print(f"        {obs.splitlines()[0][:70]}…")
    if not ok:
        failures.append(f"知识库工具没返回预期内容(前 200 字): {obs[:200]!r}")
    if "出处" not in obs:
        failures.append("知识库结果没带出处")
    if "相关度" not in obs:
        failures.append("知识库结果没带相关度")

    # --- 未命中:必须给出下一步建议,而不是空字符串 ---
    #
    # 为什么要专门建一个空 collection,而不是「拿一句不相关的话去查」:
    # `apply_rerank` 里有 `rerank_min_keep`(默认 1)—— 当所有候选都低于阈值时
    # **仍然保留 top1**,因为空列表会被上层误读成检索故障。所以只要知识库里
    # 有哪怕一个块,任何查询都会「命中」一条。真正会返回空的只有
    # 「候选集本身为空」,也就是空库。这也说明这个分支在真实使用中
    # 基本只在库没导数据时才会走到。
    empty_store = QdrantStore(
        cfg=dataclasses.replace(s.qdrant, collection=EMPTY_COLLECTION),
        retrieval=s.retrieval,
    )
    empty_store.ensure_collection(dense_dim=emb.dense_dim, recreate=True)
    set_backend(HybridBackend(store=empty_store))
    obs = reg.run(KB_TOOL, "量子计算机的制冷方案")
    set_backend(backend)
    ok = "没有检索到" in obs and "search_web" in obs
    print(f"    {'✅' if ok else '❌'} 知识库未命中(空库)-> 明确提示可以用联网搜索")
    if not ok:
        failures.append(f"未命中的 Observation 不合格: {obs[:200]!r}")
    if "UNTRUSTED_DATA" in obs:
        # 这句话是替模型做的决策提示,不是资料,不该套不可信包装
        failures.append("未命中的提示被误当成资料包了 UNTRUSTED_DATA")

    # --- 空输入 ---
    obs = reg.run(KB_TOOL, "   ")
    if "查询为空" not in obs:
        failures.append(f"空输入没被拦住: {obs[:120]!r}")
    print(f"    {'✅' if '查询为空' in obs else '❌'} 空输入被拦住")

    # --- 工具名大小写/反引号 ---
    obs = reg.run("`Search_Knowledge_Base`", HIT_QUERY)
    if "没有名为" in obs or "UNTRUSTED_DATA" not in obs:
        failures.append(f"工具名归一化失败: {obs[:120]!r}")
    print(f"    {'✅' if 'UNTRUSTED_DATA' in obs else '❌'} 工具名大小写/反引号归一化")

    # --- 未知工具:要把可用工具回给模型 ---
    obs = reg.run("search_google", "x")
    ok = "没有名为" in obs and KB_TOOL in obs and WEB_TOOL in obs
    print(f"    {'✅' if ok else '❌'} 未知工具 -> 列出可用工具")
    if not ok:
        failures.append(f"未知工具的提示不合格: {obs[:160]!r}")

    # --- 工具抛异常:必须被兜住,不能上抛 ---
    boom = ToolRegistry([Tool("boom", "会炸的工具", "任意", _raise)])
    obs = boom.run("boom", "x")
    ok = "执行失败" in obs and "ValueError" in obs
    print(f"    {'✅' if ok else '❌'} 工具内部异常被兜住 -> {obs[:60]!r}")
    if not ok:
        failures.append(f"工具异常没被兜住: {obs[:160]!r}")

    # --- 网页搜索:成功 / 失败 / 未配置 三条路 ---
    obs = reg.run(WEB_TOOL, "光伏板清洗规范")
    ok = "UNTRUSTED_DATA" in obs and "https://example.com/" in obs
    print(f"    {'✅' if ok else '❌'} 联网搜索格式化正常")
    if not ok:
        failures.append(f"联网搜索结果格式不对: {obs[:200]!r}")
    if searcher.queries != ["光伏板清洗规范"]:
        failures.append(f"搜索器收到的查询不对: {searcher.queries}")

    err_reg = ToolRegistry(build_default_tools(searcher=StubSearcher("error")))  # type: ignore[arg-type]
    obs = err_reg.run(WEB_TOOL, "任意")
    ok = "联网搜索不可用" in obs and "429" in obs
    print(f"    {'✅' if ok else '❌'} 联网失败降级为 Observation(不上抛)")
    if not ok:
        failures.append(f"联网失败的 Observation 不对: {obs[:160]!r}")

    # 未配置 = 空 key 的配置。不能真去改 .env,直接构造一个空 key 的搜索器。
    unconf = WebSearcher(WebSearchConfig(api_key=""))
    unconf_reg = ToolRegistry(build_default_tools(searcher=unconf))
    obs = unconf_reg.run(WEB_TOOL, "任意")
    ok = "TAVILY_API_KEY 没配" in obs
    print(f"    {'✅' if ok else '❌'} 联网未配置 -> 明确报出缺哪个 key")
    if not ok:
        failures.append(f"未配置的 Observation 不对: {obs[:160]!r}")

    # --- 文档清单 ---
    # 篇数从库里现查,不写死 —— 上面又多导了一篇给写工具用,写死数字的
    # 断言会在下次加夹具时以一个和被测行为无关的理由变红。
    obs = reg.run(DOCS_TOOL, "-")
    n_docs = len(store.list_docs())
    ok = DOC_NAME in obs and f"共 {n_docs} 篇" in obs and "按关键词筛选后" not in obs
    print(f"    {'✅' if ok else '❌'} 文档清单 -> {obs.splitlines()[-2] if ok else obs[:60]!r}")
    if not ok:
        failures.append(f"文档清单不对: {obs[:200]!r}")
    obs = reg.run(DOCS_TOOL, "不存在的关键词")
    if "没有文件名或标题包含" not in obs:
        failures.append(f"文档清单过滤没生效: {obs[:160]!r}")
    print(f"    {'✅' if '没有文件名或标题包含' in obs else '❌'} 文档清单关键词过滤")

    # --- format_hits 直接调:图那边的 facts 要能被渲染出来 ---
    fake = RetrievedChunk(
        text="正文", doc_id="d", chunk_index=2, source="s.pdf", title="规程",
        rerank_score=0.8,
        meta={"facts": [{"head": "瓷套", "relation": "存在缺陷", "tail": "裂纹"}],
              "entities": ["瓷套"], "from_graph": True},
    )
    txt = format_hits([fake])
    ok = "[图]" in txt and "瓷套 --存在缺陷--> 裂纹" in txt and "涉及实体" in txt
    print(f"    {'✅' if ok else '❌'} format_hits 渲染图信息(标注 [图] + 关系)")
    if not ok:
        failures.append(f"format_hits 没渲染图信息: {txt[:200]!r}")

    # 写工具自检直接挂在这里跑:它要一份**真实存在的 doc_id**,而语料刚
    # 由上面写进这个测试 collection;另起一套只会重跑一遍嵌入。
    # 它在最后把标记撤销还原,所以不影响后面 ReAct 那一段。
    check_write_tools(failures, store)

    # 版本化召回接在写工具后面:它要用 `mark_superseded` 把一篇真文档标成
    # 失效,再回头验证检索结果 —— 用到的是同一条 `status` 链路的两端。
    check_versioning(failures, store)

    # 两个 collection 都要交回给 main 去删(空的那个也要)
    return store, empty_store


def _raise(arg: str) -> str:
    raise ValueError("故意失败")


# --------------------------------------------------------------------- #
# 3. 受控写工具
# --------------------------------------------------------------------- #


def check_write_tools(failures: list[str], store: QdrantStore) -> None:
    """写工具的三道闸 + 审计 + 可逆。

    **不碰正式 exports/ 和正式 logs/audit.jsonl** —— 全程落在
    `tempfile.TemporaryDirectory` 里。自检往生产日志里灌数据,往后就再也
    分不清哪条是自检、哪条是真事。

    `allow_write` 一律**显式传**,不读 `AgentConfig` —— 否则 `.env` 里
    把 `AGENT_ALLOW_WRITE` 打开的人,自检会静默变成另一套行为。
    """
    print("\n[3] 受控写工具(权限 + 审计)")

    docs = store.list_docs()
    if not docs:
        failures.append("写工具自检拿不到测试文档 —— 前面的语料没写进去")
        return
    # 挑**块数最多**的那篇。下面「混合状态」用例要求同一 doc 下至少两块
    # —— 要是选到单块文档,那条断言会被自己的前置条件悄悄跳过,而跳过的
    # 安全测试比没有这条测试更坏:报告上看着是绿的。
    best = max(docs, key=lambda d: d.get("chunks", 0))
    if best.get("chunks", 0) < 2:
        failures.append(f"自检语料里没有多块文档(最多 {best.get('chunks')} 块),混合状态用例无法构造")
        return
    doc_id = best["doc_id"]

    tmp = tempfile.TemporaryDirectory(prefix="kb_selftest_")
    root = Path(tmp.name)
    exports = root / "exports"
    audit_path = root / "audit.jsonl"

    def make_registry(**kw) -> ToolRegistry:
        cfg = dataclasses.replace(get_settings().agent, audit_log=str(audit_path))
        # store_override 必须传:不传的话 mark_superseded 会去改**正式 collection**。
        # 这条自检最危险的地方不是断言失败,是它自己写错了地方。
        return build_registry(cfg, exports_dir=exports, store_override=store, **kw)

    def eff_status() -> set[str]:
        """有效状态。**字段缺失等价于 current**(Batch 3 就是这么定的),
        所以比较状态时统一走这个函数 —— 直接比 `payload["status"]` 会把
        「没写字段」和「写了个 current」当成两回事,而那只是同一件事的两种
        表示。"""
        return {
            ((r.payload or {}).get("status") or STATUS_CURRENT)
            for r in store.doc_records(doc_id)
        }

    def audit() -> list[dict]:
        if not audit_path.exists():
            return []
        return [
            json.loads(x)
            for x in audit_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]

    agree = lambda n, a: True  # noqa: E731
    exported = lambda: sorted(p.name for p in exports.iterdir()) if exports.exists() else []  # noqa: E731

    # --- 闸 1:默认关 ---
    reg = make_registry(allow_write=False)
    obs = reg.run(EXPORT_TOOL, "不该出现.md | 正文")
    ok = "未启用" in obs and not exported()
    print(f"    {'✅' if ok else '❌'} 闸1 默认关:写工具被拒且没落盘")
    if not ok:
        failures.append(f"默认关没生效: {obs[:160]!r} 落盘={exported()}")
    # 工具**确实挂在表里** —— 不然上面的"被拒"就只是"根本没这个工具",
    # 那样测的是查表逻辑,不是权限层。这两件事必须分得开。
    if EXPORT_TOOL not in reg.names or MARK_TOOL not in reg.names:
        failures.append("写工具没挂进工具表,上面的拒绝不是权限层给的")
    print(f"    {'✅' if EXPORT_TOOL in reg.names else '❌'} 写工具已挂进工具表(拒绝来自权限层)")

    # --- 闸 2:没人能确认 ≠ 默认同意 ---
    reg = make_registry(allow_write=True)
    obs = reg.run(EXPORT_TOOL, "a.md | 正文")
    ok = "没有确认通道" in obs and not exported()
    print(f"    {'✅' if ok else '❌'} 闸2 无确认通道 -> 拒绝(不是默认同意)")
    if not ok:
        failures.append(f"无 confirmer 时没拒绝: {obs[:160]!r}")

    reg = make_registry(allow_write=True, confirmer=lambda n, a: False)
    obs = reg.run(EXPORT_TOOL, "b.md | 正文")
    ok = "拒绝" in obs and not exported()
    print(f"    {'✅' if ok else '❌'} 闸2 用户拒绝 -> 不执行")
    if not ok:
        failures.append(f"用户拒绝后仍执行: {obs[:160]!r}")

    def boom(n: str, a: str) -> bool:
        raise RuntimeError("确认通道炸了")

    reg = make_registry(allow_write=True, confirmer=boom)
    obs = reg.run(EXPORT_TOOL, "c.md | 正文")
    ok = "确认环节出错" in obs and not exported()
    print(f"    {'✅' if ok else '❌'} 闸2 确认环节自身出错 -> 按拒绝(失败方向是安全侧)")
    if not ok:
        failures.append(f"确认环节出错时没拒绝: {obs[:160]!r}")

    # --- 闸 3:参数校验。全部点头同意,只让文件名不合法 ---
    reg = make_registry(allow_write=True, confirmer=agree)
    bad_names = [
        ("../evil.md", "目录穿越"),
        ("..\\evil.md", "反斜杠穿越"),
        ("sub/evil.md", "带目录成分"),
        ("D:\\tmp\\evil.md", "绝对路径"),
        ("C:evil.md", "驱动器相对路径"),
        ("evil.bat", "可执行后缀"),
        ("evil.ps1", "可执行后缀"),
        ("evil.md.exe", "双后缀"),
        ("x" * 90 + ".md", "超长文件名"),
    ]
    for name, label in bad_names:
        obs = reg.run(EXPORT_TOOL, f"{name} | 正文")
        ok = "导出被拒绝" in obs
        print(f"    {'✅' if ok else '❌'} 闸3 拒绝{label}:{name[:32]!r}")
        if not ok:
            failures.append(f"非法文件名 {name!r}({label})没被拒: {obs[:160]!r}")
    # 上面九次尝试**一个文件都不该落盘**
    leftover = exported()
    ok = not leftover
    print(f"    {'✅' if ok else '❌'} 闸3 九次越权尝试零落盘")
    if not ok:
        failures.append(f"越权尝试留下了文件: {leftover}")

    # --- 正常导出:真落盘 + provenance + 同名不覆盖 ---
    obs = reg.run(EXPORT_TOOL, "报告.md | 第一版结论")
    first = exports / "报告.md"
    body = first.read_text(encoding="utf-8") if first.exists() else ""
    ok = (
        first.exists()
        and "已导出到" in obs
        and "第一版结论" in body
        and "受控写工具导出" in body  # provenance 头是自动加的,不是模型写的
    )
    print(f"    {'✅' if ok else '❌'} 正常导出:落盘 {first.name}({len(body)} 字,含来源头)")
    if not ok:
        failures.append(f"导出结果不对: {obs[:160]!r} body={body[:120]!r}")

    reg.run(EXPORT_TOOL, "报告.md | 第二版结论")
    second = exports / "报告-1.md"
    ok = second.exists() and "第一版结论" in first.read_text(encoding="utf-8")
    print(f"    {'✅' if ok else '❌'} 同名不覆盖:第二份落到 {second.name},第一份原样")
    if not ok:
        failures.append(f"同名导出覆盖了原文件: {exported()}")

    # 不给文件名 -> 自动命名,而不是报错
    reg.run(EXPORT_TOOL, "只有正文没有文件名")
    ok = any(p.name.startswith("report-") for p in exports.iterdir())
    print(f"    {'✅' if ok else '❌'} 省略文件名 -> 自动按时间命名")
    if not ok:
        failures.append(f"自动命名没生效: {exported()}")

    # --- 审计:intent 配对、先意图后执行、被拒也留痕、只读不入账 ---
    # 一条写调用的收尾只有两种:`result`(真执行了,无论成败)或 `denied`
    # (被拒了)。闸 1/闸 2 的拒绝没有 intent(那时连参数都没进工具),
    # 所以是 `intents ⊆ terminal`,不是相等。
    recs = audit()
    intents = {r["call_id"] for r in recs if r["phase"] == "intent"}
    results = {r["call_id"] for r in recs if r["phase"] == "result"}
    denied = [r for r in recs if r["phase"] == "denied"]
    terminal = results | {r["call_id"] for r in denied}
    ok = bool(intents) and intents <= terminal and results <= intents
    print(f"    {'✅' if ok else '❌'} 审计:每条 intent 都有收尾、每个 result 都有 intent({len(intents)} 条)")
    if not ok:
        failures.append(
            f"审计配对不全: intent={len(intents)} result={len(results)} denied={len(denied)}"
        )

    # 顺序上 intent 必须先于同一 call_id 的收尾记录 —— 这条是"崩了也有据可查"的全部依据
    def at(cid: str, *phases: str) -> int:
        return next(i for i, r in enumerate(recs) if r["call_id"] == cid and r["phase"] in phases)

    order_ok = all(at(cid, "intent") < at(cid, "result", "denied") for cid in intents & terminal)
    print(f"    {'✅' if order_ok else '❌'} 审计:每条 intent 都先于对应的收尾记录落盘")
    if not order_ok:
        failures.append("存在收尾记录早于 intent —— 先写意图这条没做到")

    # 被拒的尝试分两类,都要留痕:
    #   - 闸 3 的 9 次越权 -> 过了前两闸、拿到 ticket,由**工具**抛 ToolDenied
    #   - 闸 1 / 闸 2 的 4 次   -> 在 authorize 里就被拒,连 ticket 都没有
    # 第二类以前完全没记录(日志上一切正常,而系统正被反复试探)。
    traversal = [r for r in denied if "导出被拒绝" in (r.get("detail") or "")]
    gate = [r for r in denied if "导出被拒绝" not in (r.get("detail") or "")]
    ok = len(traversal) >= len(bad_names) and len(gate) >= 1
    print(
        f"    {'✅' if ok else '❌'} 审计:被拒的尝试也留痕"
        f"(越权 {len(traversal)} 条 / 闸门 {len(gate)} 条)"
    )
    if not ok:
        failures.append(f"越权尝试没有留痕: 越权={len(traversal)} 闸门={len(gate)}")

    before = len(audit())
    reg.run(DOCS_TOOL, "-")
    ok = len(audit()) == before
    print(f"    {'✅' if ok else '❌'} 审计:只读工具不入账(不被写记录淹没)")
    if not ok:
        failures.append("只读工具被写进了审计日志")

    # --- mark_superseded:标记 -> 幂等 -> 撤销,且正文/向量没被动过 ---
    recs0 = store.doc_records(doc_id)
    before_status = eff_status()
    before_text = (recs0[0].payload or {}).get("text", "")

    obs = reg.run(MARK_TOOL, f"{doc_id} | 2025-06 已废止")
    cur = store.doc_records(doc_id)
    status = {(r.payload or {}).get("status") for r in cur}
    prev = {(r.payload or {}).get("previous_status") for r in cur}
    ok = status == {STATUS_SUPERSEDED} and prev == {STATUS_CURRENT} and "标记为已失效" in obs
    print(f"    {'✅' if ok else '❌'} 标记失效:status={status} previous_status={prev}")
    if not ok:
        failures.append(f"标记失效不对: {obs[:120]!r} status={status} prev={prev}")

    # payload 是**合并**写入:正文必须原样还在(否则这个工具就成了篡改工具)
    after_text = (store.doc_records(doc_id)[0].payload or {}).get("text", "")
    ok = after_text == before_text and bool(after_text)
    print(f"    {'✅' if ok else '❌'} 标记只改状态字段,正文原样({len(after_text)} 字)")
    if not ok:
        failures.append("标记失效把正文也改了 —— set_payload 应该是合并语义")

    # 幂等:重复标记**不能覆盖 previous_status**
    obs = reg.run(MARK_TOOL, f"{doc_id} | 再标一次")
    prev2 = {(r.payload or {}).get("previous_status") for r in store.doc_records(doc_id)}
    ok = "无需重复标记" in obs and prev2 == {STATUS_CURRENT}
    print(f"    {'✅' if ok else '❌'} 重复标记幂等:previous_status 仍是 {prev2}")
    if not ok:
        failures.append(f"重复标记覆盖了回滚点: prev={prev2} —— 撤销就回不去了")

    # 撤销 -> 回到原状态,且辅助字段清空
    obs = reg.run(MARK_TOOL, f"{doc_id} | restore")
    cur = store.doc_records(doc_id)
    status = eff_status()
    leftovers = {k for r in cur for k in ("previous_status", "superseded_at", "superseded_reason")
                 if (r.payload or {}).get(k) is not None}
    ok = status == before_status and not leftovers and "撤销标记" in obs
    print(f"    {'✅' if ok else '❌'} 撤销标记:回到 {status},辅助字段已清空")
    if not ok:
        failures.append(f"撤销没回到原状态: status={status} 残留={leftovers}")

    # 标记一次能撤销一次,撤销后还能再标 —— 可逆不是一次性的
    reg.run(MARK_TOOL, doc_id)
    obs = reg.run(MARK_TOOL, f"{doc_id} | restore")
    ok = eff_status() == before_status
    print(f"    {'✅' if ok else '❌'} 可逆是反复可用的(标记→撤销→再标记→撤销)")
    if not ok:
        failures.append("第二次往返失败了")

    # --- 混合状态:半篇被标过 -> 拒绝,而不是一刀切 ---
    # 这是**安全性质的拒绝**,必须走 ToolDenied(审计记 denied)。构造方式
    # 是绕过工具直接改一块的 payload —— 模拟"上一次操作只做了一半"。
    # 文档已在上面选过(`multi`),这里不再判长度 —— 跳过会伪装成通过。
    store.set_payload(doc_id, {"status": STATUS_SUPERSEDED}, chunk_index=0)
    before_mixed = eff_status()
    obs = reg.run(MARK_TOOL, f"{doc_id} | 又标一次")
    ok = "混合状态" in obs and eff_status() == before_mixed
    print(f"    {'✅' if ok else '❌'} 混合状态 -> 拒绝且不落任何改动({sorted(before_mixed)})")
    if not ok:
        failures.append(f"混合状态没被拦住或已经改了数据: {obs[:120]!r}")

    # 收尾:把那一块还原,后面的用例还在同一个 doc 上跑
    store.set_payload(doc_id, {"status": STATUS_CURRENT}, chunk_index=0)
    ok = eff_status() == {STATUS_CURRENT}
    print(f"    {'✅' if ok else '❌'} 混合状态还原后状态一致({sorted(eff_status())})")
    if not ok:
        failures.append(f"还原失败: {sorted(eff_status())}")

    # --- 参数缺失/不存在:给模型的提示要能指导下一步,且不产生副作用 ---
    obs = reg.run(MARK_TOOL, "   ")
    ok = "请给出" in obs
    print(f"    {'✅' if ok else '❌'} 空参数被拦住并说明怎么填")
    if not ok:
        failures.append(f"空参数的提示不合格: {obs[:120]!r}")

    obs = reg.run(MARK_TOOL, "这个docid不存在")
    ok = "没有 doc_id" in obs and "list_documents" in obs
    print(f"    {'✅' if ok else '❌'} 不存在的 doc_id -> 指向 list_documents")
    if not ok:
        failures.append(f"未知 doc_id 的提示不合格: {obs[:120]!r}")

    tmp.cleanup()


# --------------------------------------------------------------------- #
# 3.5 版本化索引
# --------------------------------------------------------------------- #


def check_versioning(failures: list[str], store: QdrantStore) -> None:
    """标记失效 → 召回侧真的不再返回它 → 开关打开又能召回。

    这条是 Batch 3 的**验收条件**,不是"字段写进去了"的检查。字段写进去
    没人读是这轮改动之前的状态,而那种状态下所有断言都能绿 ——
    所以这里一律从**检索结果**看,不看 payload。
    """
    print("\n[3.5] 版本化索引(标记失效 → 召回侧过滤)")

    s = get_settings()
    pipe = IngestPipeline(store=store)
    pipe.ingest_text(VER_TEXT_V1, source=VER_DOC_V1, title="架空线路巡视规程")
    pipe.ingest_text(VER_TEXT_V2, source=VER_DOC_V2, title="架空线路巡视规程")

    v1 = stable_doc_id(VER_DOC_V1)
    v2 = stable_doc_id(VER_DOC_V2)

    base = dataclasses.replace(s.retrieval, include_superseded=False, rerank_enabled=False)
    with_hist = dataclasses.replace(base, include_superseded=True)
    # 直接建两个检索器,而不是改全局配置:全局那份是 `frozen` 的单例,
    # 动它会把同一进程里后面所有用例一起带偏 —— 自检自己制造的这种
    # 串扰最难查。
    r_cur = HybridRetriever(store=store, cfg=base)
    r_hist = HybridRetriever(store=store, cfg=with_hist)

    def hits(r: HybridRetriever) -> set[str]:
        return {c.doc_id for c in r.retrieve(VER_QUERY, top_k=10)}

    # --- 前置:两版都召得回 ---
    # 没有这一步,后面"旧版不在了"完全可能是"它本来就没被召回"。
    got = hits(r_cur)
    ok = v1 in got and v2 in got
    print(f"    {'✅' if ok else '❌'} 基线:两版都能召回(共 {len(got)} 块)")
    if not ok:
        failures.append(f"基线不成立,后面的过滤结论没意义: v1={v1 in got} v2={v2 in got}")
        return

    # --- 文件名约定 → 版本字段 ---
    recs = store.doc_records(v1)
    p = (recs[0].payload or {}) if recs else {}
    ok = p.get("doc_version") == "v1" and p.get("effective_from") == "2024-01-01"
    print(f"    {'✅' if ok else '❌'} 文件名认出版本: v1={p.get('doc_version')} 生效={p.get('effective_from')}")
    if not ok:
        failures.append(f"文件名里的版本约定没进 payload: {p.get('doc_version')!r} {p.get('effective_from')!r}")

    # --- 标记失效(走真工具,不走 set_payload)---
    tmp = tempfile.TemporaryDirectory(prefix="kb_selftest_ver_")
    audit_path = Path(tmp.name) / "audit.jsonl"
    cfg = dataclasses.replace(s.agent, audit_log=str(audit_path))
    reg = build_registry(
        cfg, allow_write=True, confirmer=lambda *a: True,
        exports_dir=Path(tmp.name) / "exports", store_override=store,
    )
    obs = reg.run(MARK_TOOL, f"{v1} | 2025-06 换版")
    if "已把" not in obs:
        failures.append(f"标记失效没成功,后面几条都验不了: {obs[:120]!r}")
        print(f"    ❌ 标记失效失败: {obs[:100]!r}")
        tmp.cleanup()
        return

    # --- 核心:旧版不再召回,新版还在 ---
    got = hits(r_cur)
    ok = v1 not in got and v2 in got
    print(f"    {'✅' if ok else '❌'} 标记后:旧版不再召回、新版仍在(旧版在={v1 in got})")
    if not ok:
        failures.append(f"过滤没生效: 召回结果 {sorted(got)}")

    # --- 开关打开:历史版本可召回 ---
    got = hits(r_hist)
    ok = v1 in got
    print(f"    {'✅' if ok else '❌'} 打开 include_superseded:历史版本可召回")
    if not ok:
        failures.append("开关打开了旧版还是召回不到 —— 那不是过滤,是数据没了")

    # --- 老数据兼容:没有 status 字段的块**必须**照样召回 ---
    # 这是"缺字段当有效"那条约定的唯一验证点。它要是错了,表现是升级之后
    # 整个库查不到东西 —— 而且不报错。
    store.client.delete_payload(
        collection_name=store.cfg.collection,
        keys=[STATUS_FIELD],
        points=qm.PointIdsList(points=[r.id for r in store.doc_records(v2)]),
        wait=True,
    )
    left = (store.doc_records(v2)[0].payload or {})
    got = hits(r_cur)
    ok = STATUS_FIELD not in left and v2 in got
    print(f"    {'✅' if ok else '❌'} 老数据(无 status 字段)仍能召回")
    if not ok:
        failures.append(f"缺字段的老数据被挡掉了: 字段在={STATUS_FIELD in left} 召回={v2 in got}")

    # --- 撤销后回到可召回 ---
    reg.run(MARK_TOOL, f"{v1} | restore")
    got = hits(r_cur)
    ok = v1 in got
    print(f"    {'✅' if ok else '❌'} 撤销标记后旧版回到召回结果")
    if not ok:
        failures.append("撤销标记后旧版仍然召回不到")

    tmp.cleanup()


# --------------------------------------------------------------------- #
# 4. ReAct 循环
# --------------------------------------------------------------------- #


def check_react(failures: list[str], cfg: AgentConfig) -> None:
    print("\n[4] ReAct 循环(脚本化假 LLM)")

    searcher = StubSearcher()
    reg = ToolRegistry(build_default_tools(searcher=searcher))  # type: ignore[arg-type]

    # --- A. 一轮检索后作答 ---
    llm = ScriptedLLM([act(KB_TOOL, HIT_QUERY), "Thought: 够了\nFinal Answer: 按规程,裂纹需立即更换 [自检规程 (块 0)]"])
    r = ReActAgent(llm=llm, registry=reg, cfg=cfg).run("绝缘子出现裂纹怎么办")
    ok = (
        r.stop_reason == "final_answer"
        and r.iterations == 1
        and "立即更换" in r.answer
        and r.steps[0].action == KB_TOOL
    )
    print(f"    {'✅' if ok else '❌'} A 一轮检索后作答:stop={r.stop_reason} 轮数={r.iterations}")
    if not ok:
        failures.append(f"A 场景不对: {r.stop_reason=} {r.iterations=} {r.answer[:80]!r}")
    # Observation 必须作为 user 消息回灌(不是 role=tool,兼容端点会 400)
    second = llm.calls[1]
    ok = (
        second[0]["role"] == "system"
        and second[-1]["role"] == "user"
        and second[-1]["content"].startswith("Observation: ")
        and any(m["role"] == "assistant" for m in second)
    )
    print(f"    {'✅' if ok else '❌'} A Observation 以 user 消息回灌(并保留 assistant 原文)")
    if not ok:
        failures.append(f"A 消息编排不对: {[m['role'] for m in second]}")
    # stop 序列必须传,否则模型会自己编 Observation
    if llm.stops[0] != STOP_SEQUENCES:
        failures.append(f"A 没传 stop 序列: {llm.stops[0]!r}")
    print(f"    {'✅' if llm.stops[0] == STOP_SEQUENCES else '❌'} A 每轮都带 stop 序列(防编造 Observation)")

    # --- B. 重复调用被短路 ---
    llm = ScriptedLLM([
        act(KB_TOOL, HIT_QUERY),
        act(KB_TOOL, HIT_QUERY),          # 完全一样 -> 不该再执行
        "Final Answer: 好",
    ])
    r = ReActAgent(llm=llm, registry=reg, cfg=cfg).run("重复调用测试")
    ok = r.steps[1].repeated and "结果不会变化" in r.steps[1].observation and r.iterations == 2
    print(f"    {'✅' if ok else '❌'} B 重复调用被短路(未执行第二次)")
    if not ok:
        failures.append(f"B 重复调用没短路: {r.steps[1].observation[:80]!r}")

    # --- C. 格式跑偏 -> 纠正 -> 成功 ---
    llm = ScriptedLLM([
        "我想想啊,这个问题有点复杂。",           # 没有 Action 也没有 Final Answer
        act(KB_TOOL, HIT_QUERY),
        "Final Answer: 纠正后答出来了",
    ])
    r = ReActAgent(llm=llm, registry=reg, cfg=cfg).run("格式纠正测试")
    ok = r.stop_reason == "final_answer" and r.iterations == 1 and r.warnings
    print(f"    {'✅' if ok else '❌'} C 格式跑偏 -> 纠正 -> 继续(警告 {len(r.warnings)} 条)")
    if not ok:
        failures.append(f"C 纠正路径不对: {r.stop_reason=} {r.iterations=} {r.warnings}")
    # 纠正提示要以 user 消息送进去
    if "不符合要求" not in llm.calls[1][-1]["content"]:
        failures.append("C 第二轮没有送纠正提示")

    # --- D. 一直不按格式 -> 兜底把散文当答案,而不是空手而归 ---
    cfg1 = dataclasses.replace(cfg, max_format_retries=2, max_iterations=6)
    llm = ScriptedLLM(["散1", "散2", "散3", "散4"])
    r = ReActAgent(llm=llm, registry=reg, cfg=cfg1).run("兜底测试")
    ok = r.stop_reason == "unparsed_output" and r.answer == "散3" and r.iterations == 0
    print(f"    {'✅' if ok else '❌'} D 连续不合格式 -> 原样作为答案返回({r.answer!r})")
    if not ok:
        failures.append(f"D 兜底不对: {r.stop_reason=} {r.answer!r}")
    # 纠正次数是有限的,不能把轮次全耗在纠正上:
    # 第 3 次不合格式时 fmt_retries(=3) 已超过 max_format_retries(=2),直接兜底 ——
    # 所以只调了 3 次 LLM,而不是把 max_iterations 用完
    if len(llm.calls) != 3:
        failures.append(f"D 纠正次数不对: 调了 {len(llm.calls)} 次 LLM,应为 3")
    print(f"    {'✅' if len(llm.calls) == 3 else '❌'} D 纠正次数受 max_format_retries 限制({len(llm.calls)} 次调用)")

    # --- E. 一直在调工具 -> 触发轮次上限 -> 强制用现有资料作答 ---
    cfg2 = dataclasses.replace(cfg, max_iterations=2)
    llm = ScriptedLLM([
        act(KB_TOOL, HIT_QUERY, "第一轮"),
        act(WEB_TOOL, "另一个查询", "第二轮"),
        "Final Answer: 强制作答的内容",
    ])
    r = ReActAgent(llm=llm, registry=reg, cfg=cfg2).run("轮次上限测试")
    ok = (
        r.stop_reason == "max_iterations"
        and r.iterations == 2
        and r.answer == "强制作答的内容"
        and any("最大轮次" in w for w in r.warnings)
    )
    print(f"    {'✅' if ok else '❌'} E 轮次上限 -> 强制作答:{r.answer!r}")
    if not ok:
        failures.append(f"E 上限路径不对: {r.stop_reason=} {r.iterations=} {r.answer[:60]!r}")
    # 强制作答那一次不该再带 stop 序列(要整段答案)
    if llm.stops[-1] is not None:
        failures.append("E 强制作答还传了 stop 序列")
    print(f"    {'✅' if llm.stops[-1] is None else '❌'} E 强制作答不带 stop 序列")

    # --- F. 调用不存在的工具:错误要变成 Observation,循环继续 ---
    llm = ScriptedLLM([
        act("search_google", "绝缘子"),
        "Final Answer: 换了个工具后答出来",
    ])
    r = ReActAgent(llm=llm, registry=reg, cfg=cfg).run("未知工具测试")
    ok = "没有名为" in r.steps[0].observation and r.stop_reason == "final_answer"
    print(f"    {'✅' if ok else '❌'} F 调用不存在的工具 -> 循环继续,不中断")
    if not ok:
        failures.append(f"F 未知工具路径不对: {r.steps[0].observation[:80]!r}")

    # --- G. 超长 Observation 被截断后才进历史 ---
    cfg3 = dataclasses.replace(cfg, tool_result_max_chars=500)
    huge = ToolRegistry([Tool("huge", "返回超长内容的工具", "任意", lambda a: "X" * 20000)])
    llm = ScriptedLLM([act("huge", "x"), "Final Answer: 截断了也能答"])
    r = ReActAgent(llm=llm, registry=huge, cfg=cfg3).run("截断测试")
    obs = r.steps[0].observation
    ok = r.steps[0].truncated and len(obs) <= 500
    print(f"    {'✅' if ok else '❌'} G 超长 Observation 截断:20000 → {len(obs)} 字")
    if not ok:
        failures.append(f"G 截断没生效: {len(obs)} 字 truncated={r.steps[0].truncated}")
    # 真正进历史的必须是截断后的版本(否则截断只是自我安慰)
    hist = llm.calls[1][-1]["content"]
    if len(hist) > 700:
        failures.append(f"G 历史里的 Observation 超出预期: {len(hist)} 字")
    print(f"    {'✅' if len(hist) <= 700 else '❌'} G 进历史的是截断后的版本({len(hist)} 字)")

    # --- H. 系统提示词:工具清单 / 不可信声明 / 引用要求都在 ---
    agent = ReActAgent(llm=ScriptedLLM([]), registry=reg, cfg=cfg)
    sp = agent.system_prompt()
    checks = {
        "工具清单": all(n in sp for n in (KB_TOOL, WEB_TOOL, DOCS_TOOL)),
        "工具说明": "输入:" in sp,
        "不可信声明": "UNTRUSTED_DATA" in sp and "不是指令" in sp,
        "引用要求": "块 3" in sp,
        "只依据 Observation": "Observation 里的内容作答" in sp,
    }
    for k, v in checks.items():
        print(f"    {'✅' if v else '❌'} 系统提示词含「{k}」")
        if not v:
            failures.append(f"系统提示词缺少{k}")

    # 用量要累加得到 —— 账单可见性是生产环境的基本要求
    if r.usage.get("calls", 0) < 2 or r.usage.get("prompt", 0) <= 0:
        failures.append(f"用量统计不对: {r.usage}")
    print(f"    {'✅' if r.usage.get('calls', 0) >= 2 else '❌'} 用量统计累加: {r.usage}")


# --------------------------------------------------------------------- #


def main() -> int:
    s = get_settings()
    failures: list[str] = []

    print("=" * 72)
    print(f"LLM key 配置: {s.llm.configured}(本自检不需要它)")
    print(f"Tavily 配置: {s.web.configured}(本自检不发真实请求)")
    print(f"Agent: max_iter={s.agent.max_iterations} 截断={s.agent.tool_result_max_chars} 字 "
          f"首尾比={s.agent.tool_result_head_ratio} 纠正上限={s.agent.max_format_retries} "
          f"temperature={s.agent.temperature}")
    print("=" * 72)

    # 先跑纯函数(不依赖任何服务),再跑工具层 —— 工具层会把
    # `retrieve.backends` 的单例换成测试 collection,于是紧随其后的
    # ReAct 循环里,知识库工具查的是测试数据而不是正式库。
    # (Qdrant 连不上时工具层会返回 None,ReAct 那一段照样能跑完 ——
    #  它的断言只依赖脚本化 LLM 和假搜索器,不依赖检索结果的好坏。)
    check_parsers(failures)

    stores: tuple[QdrantStore, ...] = ()
    try:
        out = check_tools(failures)
        if out is not None:
            stores = out
        check_react(failures, s.agent)
    finally:
        for i, name in enumerate((TEST_COLLECTION, EMPTY_COLLECTION)):
            try:
                stores[i].client.delete_collection(name)
                print(f"(已删测试 collection {name})")
            except Exception as exc:  # noqa: BLE001
                print(f"(collection {name} 清理失败: {exc})")
        for st in stores:
            try:
                st.close()
            except Exception:  # noqa: BLE001
                pass
        # 别把测试后端留在单例里
        set_backend(None)

    print(f"\n{'=' * 72}")
    if failures:
        print("失败项:")
        for f in failures:
            print(f"  ❌ {f}")
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
