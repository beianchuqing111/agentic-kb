"""Agent 自检:解析器 → 工具层 → ReAct 循环。

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AgentConfig, WebSearchConfig, get_settings  # noqa: E402
from embed import get_embedder  # noqa: E402
from ingest import IngestPipeline, stable_doc_id  # noqa: E402
from llm.client import ChatResult  # noqa: E402
from retrieve.backends import HybridBackend, set_backend  # noqa: E402
from retrieve.hybrid import RetrievedChunk  # noqa: E402
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
    KB_TOOL,
    WEB_TOOL,
    Tool,
    ToolRegistry,
    build_default_tools,
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
    obs = reg.run(DOCS_TOOL, "-")
    ok = DOC_NAME in obs and "共 1 篇" in obs
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

    # 两个 collection 都要交回给 main 去删(空的那个也要)
    return store, empty_store


def _raise(arg: str) -> str:
    raise ValueError("故意失败")


# --------------------------------------------------------------------- #
# 3. ReAct 循环
# --------------------------------------------------------------------- #


def check_react(failures: list[str], cfg: AgentConfig) -> None:
    print("\n[3] ReAct 循环(脚本化假 LLM)")

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
