"""agentic-kb 命令行入口。

    python kb.py health                     体检:各子系统通不通、配没配
    python kb.py ingest D:\\docs             导入文档(支持 pdf/md/txt/docx…)
    python kb.py search "绝缘子破损判据"      只检索,不用 LLM(不需要 key)
    python kb.py ask "绝缘子出现裂纹怎么办"    ReAct 智能问答(需要 LLM key)
    python kb.py docs                       列出库里的文档
    python kb.py stats                      各项计数

Windows 上请用 kb.bat(它会设好 UTF-8 代码页和解释器路径),
否则中文输出会乱码、或者 import 不到 dotenv。

命令分两档,是为了让**没配 LLM key 也能用**:

  search / ingest / docs / stats / health  纯本地,零 API 调用
  ask                                      需要 LLM key(ReAct 要模型来决策)

这也是排查问题的顺序:ask 不好使的时候,先 `health` 再 `search`,
就能分清是模型的问题还是检索的问题。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 必须在 transformers 被导入**之前**设。它控制的是 transformers 自己的
# 日志级别:那个「You're using a XLMRobertaTokenizerFast tokenizer…」
# 每次都刷一行,和用户要做的事毫无关系。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 没配 PYTHONIOENCODING 时,Windows 控制台默认 GBK —— 中文答案一打印就
# UnicodeEncodeError。这里兜一层:实在编不出来就用 ? 顶上,绝不因为
# 「输出不了」而丢掉整个回答。真正的解法是 kb.bat 里的 chcp 65001。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

from config import BackendType, get_settings  # noqa: E402
from llm.client import LLMNotConfigured  # noqa: E402
from retrieve.backends import build_backend, get_backend, set_backend  # noqa: E402
from agent.tools import format_hits  # noqa: E402

log = logging.getLogger("kb")


# --------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------- #


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # 这些库的话都和用户要做的事无关,不压掉会淹没真正的输出:
    #   httpx/openai/qdrant… —— INFO 级别每条 HTTP 请求一行
    #   torchao              —— 导入时打一行「cpp extensions 版本不匹配」
    #   torch.distributed.elastic —— 导入时必打一行「Redirects are currently
    #      not supported in Windows」,来自它的 get_libc() 里一句
    #      logger.warning,在任何 Windows 机器上都出现。它走的是标准 logging,
    #      所以按 logger 名压掉就行 —— 用 TORCH_CPP_LOG_LEVEL 是没用的
    #      (那是 C++ 侧的日志级别,实测拦不住这条)。
    for noisy in (
        "httpx", "httpcore", "neo4j", "qdrant_client", "openai", "urllib3",
        "torchao", "transformers", "sentence_transformers", "huggingface_hub",
        "torch.distributed.elastic",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _pick_backend(name: str | None):
    """按 --backend 覆盖全局后端单例。"""
    if not name:
        return get_backend()
    b = build_backend(BackendType(name))
    set_backend(b)
    return b


def _fmt_score(r) -> str:
    s = r.rerank_score if r.rerank_score is not None else r.score
    return f"{s:.4f}"


# --------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------- #


def cmd_health(args) -> int:
    s = get_settings()
    # 先在前面解析后端 —— 下面报的「后端」「要不要连 Neo4j」都得按**生效的**
    # 那个说。用 s.backend 的话,--backend 覆盖时这两处会各说各话
    # (实测:头部写着 hybrid,服务段却已经在报 neo4j 了)。
    backend = _pick_backend(args.backend)
    is_graph = backend.name == BackendType.GRAPHRAG.value

    print("=" * 68)
    print("配置")
    print(f"  后端      : {backend.name}"
          f"{'   (--backend 覆盖)' if args.backend else ''}   collection: {s.qdrant.collection}")
    print(f"  Qdrant    : {s.qdrant.url}")
    if is_graph:
        print(f"  Neo4j     : {s.neo4j.uri}   database: {s.neo4j.database}")
    print(f"  LLM       : {'已配置 ' + s.llm.model if s.llm.configured else '❌ 未配置(LLM_API_KEY 为空)'}")
    print(f"              {s.llm.base_url}")
    print(f"  Tavily    : {'已配置' if s.web.configured else '❌ 未配置(TAVILY_API_KEY 为空)'}")
    print(f"  重排      : {'开' if s.retrieval.rerank_enabled else '关'}"
          f"   阈值 {s.retrieval.rerank_min_score}(至少保留 {s.retrieval.rerank_min_keep} 条)")

    print("\n服务")
    h = backend.health()
    # 键长度不一(collection / llm_configured …),对齐宽度按实际的算
    w = max((len(k) for k in h.detail), default=0)
    for k, v in h.detail.items():
        print(f"  {k:<{w}} : {v if v is not None else '—'}")
    if not h.ok:
        print(f"  ❌ {h.error}")

    # 图后端才需要 Neo4j;hybrid 后端根本不去连它,所以只在图模式下多报一节
    if is_graph:
        from store.graph_store import get_graph_store

        graph = get_graph_store()
        g = graph.health()
        print(f"  Neo4j 连通 : {'✅' if g.get('ok') else '❌ ' + str(g.get('error'))}")
        if g.get("ok"):
            print(f"  图规模     : {graph.stats()}")

    print("\n" + "=" * 68)
    if not h.ok:
        print("❌ 结论:后端不可用。先跑 scripts\\start_qdrant.bat"
              + ("、scripts\\start_neo4j.bat" if is_graph else ""))
        return 1
    if not s.llm.configured:
        print("⚠️  后端可用,但 LLM 未配置:search/ingest 能用,ask 不能用。")
        print("   在 .env 里填 LLM_API_KEY(DeepSeek 或 Qwen 都行,见 llm/client.py 的提示)。")
        return 0
    print("✅ 全部就绪。")
    return 0


def cmd_ingest(args) -> int:
    s = get_settings()
    root = Path(args.path)
    if not root.exists():
        print(f"❌ 路径不存在: {root}")
        return 1

    if not s.llm.configured:
        # 不是错误,但必须说清楚代价:没有定位语,上下文感知检索那块能力是缺的
        print("⚠️  LLM 未配置 —— 本次导入跳过上下文定位语生成。")
        print("   文档仍会被导入和检索,但「块前面补一句这份文档在讲什么」那步没有,")
        print("   召回质量会明显下降(尤其是块本身很短、指代很多的时候)。")
        print("   填上 LLM_API_KEY 后重新导入即可补齐 —— 重导会按 content_hash 覆盖,不用先删。\n")

    backend = _pick_backend(args.backend)
    print(f"导入 {root}  →  后端 {backend.name}"
          f"{'  (force 全量重写)' if args.force else ''}")
    t0 = time.time()
    st = backend.ingest_path(root, recursive=not args.no_recursive, force=args.force)
    dt = time.time() - t0

    print(f"\n{st.summary()}")
    if st.contextual_missing:
        print(f"  ⚠️  {st.contextual_missing} 个块没有定位语(LLM 没配或调用失败)")
    if st.errors:
        print(f"  ❌ {len(st.errors)} 个文件出错:")
        for src, err in st.errors[:10]:
            print(f"     {src}: {err}")
    print(f"  耗时 {dt:.1f}s")
    return 0 if not st.errors else 1


def cmd_search(args) -> int:
    backend = _pick_backend(args.backend)
    t0 = time.time()
    hits = backend.retrieve(args.query, top_k=args.top_k)
    dt = time.time() - t0

    if not hits:
        print("没有检索到任何内容。")
        print("  · 库里可能还没有文档 → python kb.py docs")
        print("  · 或者换一组关键词(用陈述式短语比整句问句好)")
        return 1

    print(f"{len(hits)} 条结果(耗时 {dt:.1f}s):\n")
    for i, r in enumerate(hits, 1):
        tag = "[图] " if r.meta.get("from_graph") else ""
        print(f"[{i}] {tag}{r.citation}   分数 {_fmt_score(r)}")
        if r.source and r.source != r.citation:
            print(f"    出处: {r.source}")
        body = " ".join(r.text.split())
        print("    " + (body[:300] + "…" if len(body) > 300 else body))
        if r.meta.get("entities"):
            print(f"    命中实体: {', '.join(str(e) for e in r.meta['entities'][:10])}")
        for f in (r.meta.get("facts") or [])[:5]:
            print(f"    关系: {f['head']} --{f['relation']}--> {f['tail']}")
        print()
    # format_hits 是给模型看的口径,顺手打出来便于对照「模型看到的是什么」
    if args.raw:
        print("-" * 68)
        print(format_hits(hits))
    return 0


def cmd_ask(args) -> int:
    from agent.react import ReActAgent

    s = get_settings()
    try:
        agent = ReActAgent()
        r = agent.run(args.question)
    except LLMNotConfigured as exc:
        print(f"❌ {exc}\n")
        print("ask 需要模型来决策(该查什么、够不够、怎么回答)。")
        print("先用这个看看检索本身是否正常:")
        print(f'    python kb.py search "{args.question}"')
        return 1

    print("=" * 68)
    print(r.answer)
    print("=" * 68)

    if r.steps:
        print(f"检索过程({r.iterations} 轮):")
        for st in r.steps:
            mark = " [重复,已短路]" if st.repeated else (" [已截断]" if st.truncated else "")
            print(f"  {st.index}. {st.action}({st.action_input}){mark}")
            if st.thought:
                print(f"     Thought: {st.thought[:120]}")
            if st.note:
                print(f"     {st.note}")
    sources = r.sources_called
    if sources:
        print(f"用到的工具: {', '.join(sources)}")
    print(f"停止原因: {r.stop_reason}   LLM 调用 {r.usage.get('calls', 0)} 次,"
          f"tokens {r.usage.get('prompt', 0)}+{r.usage.get('completion', 0)}")

    for w in r.warnings:
        print(f"⚠️  {w}")
    if args.verbose:
        print("\n" + "-" * 68 + "\n完整轨迹:\n")
        print(r.transcript())
    # 没给出 Final Answer 的路径要显式失败 —— 脚本化调用时能靠退出码发现问题
    return 0 if r.stop_reason == "final_answer" else 2


def cmd_docs(args) -> int:
    from store.qdrant_store import get_store

    store = get_store()
    if not store.exists():
        print("知识库还是空的。先导入文档:python kb.py ingest <目录>")
        return 1
    docs = store.list_docs()
    if not docs:
        print("知识库还是空的。")
        return 1

    if args.filter:
        kw = args.filter.lower()
        docs = [d for d in docs
                if kw in (d.get("source") or "").lower() or kw in (d.get("title") or "").lower()]
        if not docs:
            print(f"没有文件名或标题包含 {args.filter!r} 的文档。")
            return 1

    total_chunks = sum(d.get("chunks", 0) for d in docs)
    print(f"{len(docs)} 篇文档,共 {total_chunks} 块:\n")
    for d in docs:
        name = d.get("title") or d.get("source") or d.get("doc_id")
        print(f"  {d.get('chunks', 0):>5} 块  {name}")
        if d.get("source") and d.get("title") and d["source"] != d["title"]:
            print(f"         {d['source']}")
    return 0


def cmd_stats(args) -> int:
    backend = _pick_backend(args.backend)
    st = backend.stats()
    docs = st.pop("documents", [])
    print(f"后端 {st.pop('backend', '?')}")
    for k, v in st.items():
        print(f"  {k:<24}: {v}")
    print(f"  {'documents':<24}: {len(docs)} 篇")
    return 0


# --------------------------------------------------------------------- #


def _add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    """把 --backend / -v 同时挂到主 parser 和每个子 parser 上。

    argparse 默认只认子命令**之前**的全局开关,`kb.py ask "x" -v` 会直接报
    "unrecognized arguments: -v" —— 而人写命令时习惯把开关放最后。两边都挂
    是标准解法,但子 parser 那份的 default 必须是 SUPPRESS:否则
    `kb.py --backend graphrag stats` 里,子 parser 会用自己那份默认值(None)
    把主 parser 已经解析好的 graphrag **覆盖掉**,于是 --backend 静默失效。
    SUPPRESS 表示"没写这个参数就根本不设这个属性",正好不会覆盖。
    """
    d = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--backend", choices=[b.value for b in BackendType], default=d,
        help="临时覆盖后端(默认取 .env 的 KB_BACKEND)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=d, help="打开调试日志",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kb.py",
        description="agentic-kb:混合检索 + GraphRAG + ReAct 的本地知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子:\n"
            "  python kb.py health\n"
            "  python kb.py ingest D:\\docs -f\n"
            '  python kb.py search "绝缘子破损判据" -k 5\n'
            '  python kb.py ask "绝缘子出现裂纹怎么办" -v\n'
        ),
    )
    _add_global_flags(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    def sub_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text)
        _add_global_flags(sp, suppress=True)
        return sp

    sub_parser("health", "体检:配置 + 各服务连通性").set_defaults(fn=cmd_health)
    sub_parser("stats", "各项计数").set_defaults(fn=cmd_stats)

    pi = sub_parser("ingest", "导入文档")
    pi.add_argument("path")
    pi.add_argument("-f", "--force", action="store_true", help="强制全量重写(忽略内容哈希)")
    pi.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    pi.set_defaults(fn=cmd_ingest)

    ps = sub_parser("search", "只检索(不需要 LLM)")
    ps.add_argument("query")
    ps.add_argument("-k", "--top-k", type=int, default=None, help="返回条数")
    ps.add_argument("--raw", action="store_true", help="额外打印给模型看的那份文本")
    ps.set_defaults(fn=cmd_search)

    pa = sub_parser("ask", "ReAct 智能问答(需要 LLM)")
    pa.add_argument("question")
    pa.set_defaults(fn=cmd_ask)

    pd = sub_parser("docs", "列出库里的文档")
    pd.add_argument("--filter", default="", help="按文件名/标题子串过滤")
    pd.set_defaults(fn=cmd_docs)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
