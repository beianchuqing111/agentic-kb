"""agentic-kb 的 Gradio 前端。

    webui.bat                       起界面(推荐,它会设好代码页和 py310 路径)
    python webui.py --port 7860     直接跑

四个页签,对着 CLI 的三档能力:

    导入文档   上传文件 或 填服务器上的路径 → 分块/嵌入/落库(可带定位语)
    检索       只查库,不需要 LLM,用来验证召回
    问答       ReAct 智能体:自己决定查库/联网/多次检索
    库状态     文档清单 + 计数 + 体检

**为什么上传的文件要先拷到 uploads/ 再导入**:块里记的 `source` 就是导入时
的文件路径。直接拿 Gradio 给的临时路径去导入,每篇文档的出处都会变成
`C:\\Users\\...\\Temp\\gradio\\xxxx\\xxx.pdf` —— 不能看、不能按名字过滤、
重导时路径还对不上。所以先按原始文件名落到 uploads/,再从那里导入。

**并发**:Gradio 默认并发跑多个请求,而后端是个全局单例
(`retrieve.backends.get_backend`),ReAct 的工具也从它取 —— 一边导入一边
问答会互相踩。所以 queue 的并发限成 1。本机单人用,串行反而是对的。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# 和 kb.py 一样的顺序约束:这两个环境变量必须在 transformers / huggingface
# 相关库被导入**之前**设上。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 控制台是 GBK 时中文日志一打印就 UnicodeEncodeError。用 replace 兜住,
# 宁可个别字变 ? 也不能因为「输出不了」把整个请求搞挂。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

# --- 静态噪声抑制 ---
#
# 两条和用户要做的事完全无关、却很容易漏到控制台上的话:
#   torchao                    "Skipping import of cpp extensions ..."
#   torch.distributed.elastic  "Redirects are currently not supported in Windows"
#
# 实测(2026-09-16):这两条都写 **stderr**,且都走 logger.warning,
# logger 名就叫 `torchao` / `torch.distributed.elastic` ——
# 所以按名字 setLevel(ERROR) 是压得住的(加名单后 stderr 全空,验证过)。
#
# 但它们**不是在本文件导入期触发的**:`import webui` 跑完,
# sys.modules 里既没有 torchao 也没有 elastic。真正的触发点是
# **第一次构造嵌入模型**的时候,由 FlagEmbedding 把 torchao 拖进来。
#
# 那为什么仍要放在模块顶、而不是 main() 里的 _setup_logging?
# 因为 webui.py 会被当**模块**导入(自检脚本、以及上面刚说的那条路),
# 那条路根本不经过 main()。放在这里,「先建好 logger 并设级别」这件事
# 就不可能被漏掉 —— 和顶部 HF_HUB_OFFLINE 是同一个思路:
# 模块级副作用的位置本身就是语义。
_NOISY_LOGGERS = (
    "httpx", "httpcore", "neo4j", "qdrant_client", "openai", "urllib3",
    "torchao", "transformers", "sentence_transformers", "huggingface_hub",
    "torch.distributed.elastic", "asyncio", "multipart",
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for _n in _NOISY_LOGGERS:
    logging.getLogger(_n).setLevel(logging.ERROR)

from config import BackendType, get_settings  # noqa: E402
from llm.client import LLMError, LLMNotConfigured, get_llm  # noqa: E402
from retrieve.backends import build_backend, set_backend  # noqa: E402
from agent.tools import format_hits  # noqa: E402

# gradio 必须排在 config 之后导入。config 在导入期用 setdefault 设
# HF_HUB_OFFLINE,而 huggingface_hub 一旦先被导入就把离线常量定死了,
# config.assert_hf_offline_env() 会直接抛错(见 config.py 里的说明)。
import gradio as gr  # noqa: E402

log = logging.getLogger("webui")

UPLOAD_DIR = BASE_DIR / "uploads"

# 支持的格式 —— 和 ingest/loader.py 的 SUPPORTED_EXTS 保持一致。
# Gradio 的文件选择器按扩展名过滤,漏了 .markdown/.text/.log 会让用户
# 选不到本可以导入的文件。
UPLOAD_EXTS = [".txt", ".md", ".markdown", ".text", ".log", ".pdf", ".docx"]


# --------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------- #


def _setup_logging(verbose: bool) -> None:
    """把根 logger 调到想要的级别。

    模块顶已经 basicConfig 过一次了(为了赶在重导入之前压噪声),而
    basicConfig 在有 handler 时是 **no-op** —— 这里再调一次不会改级别。
    所以必须显式 setLevel,否则 `--verbose` 静默失效、`-v` 看不到任何
    多出来的东西,人会以为"verbose 没用"。
    """
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.WARNING)
    # 名单和模块顶那份是同一个(见 _NOISY_LOGGERS)。它们被显式设过级别,
    # 所以上面调高根的级别也带不动它们 —— 这正是想要的:
    # --verbose 该给的是本项目自己的调试信息,不是 httpx 的每一轮请求。
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.ERROR)


_log = logging.getLogger(__name__)


def _llm_calls() -> int | None:
    """全局 LLM 调用次数。读不到就返回 None。

    **不要在有异常时伪造成 0** —— 这个数字要直接写进界面告诉用户
    「这次导入花了多少调用」,报 0 比报不出来更糟。
    """
    try:
        return int(get_llm().usage.get("calls", 0))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------- #
# 后端(全局单例 + 缓存)
# --------------------------------------------------------------------- #

_BACKENDS: dict[str, object] = {}


def backend_for(name: str):
    """取后端并把它设为全局单例。

    必须设全局:ReAct 的工具表是构造时从 `get_backend()` 拿的
    (见 agent/tools.py 的 build_default_tools),不设的话页面上选了 graphrag,
    工具却还在查 hybrid。
    """
    key = (name or BackendType.HYBRID.value).strip()
    b = _BACKENDS.get(key)
    if b is None:
        b = build_backend(BackendType(key))
        _BACKENDS[key] = b
    set_backend(b)
    return b


# --------------------------------------------------------------------- #
# 上传落盘
# --------------------------------------------------------------------- #


def save_uploads(files) -> tuple[list[Path], list[str]]:
    """把 Gradio 的临时文件按原始文件名拷进 uploads/。

    返回 (落盘后的路径列表, 提示信息列表)。
    """
    if not files:
        return [], []
    if not isinstance(files, (list, tuple)):
        files = [files]

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    notes: list[str] = []
    for f in files:
        src = Path(str(f))
        # gr.File 给的是 NamedString,带 orig_name;没有就退回临时文件名
        name = getattr(f, "orig_name", None) or src.name
        # 只取文件名部分 —— orig_name 理论上就是文件名,但不信外部输入
        name = Path(str(name)).name
        if not name:
            continue
        dst = UPLOAD_DIR / name
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            notes.append(f"⚠️ `{name}` 在 uploads/ 里已有同名文件,已用新上传的覆盖")
        shutil.copy2(src, dst)
        saved.append(dst)
    return saved, notes


# --------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------- #


def _fmt_score(r) -> str:
    s = r.rerank_score if r.rerank_score is not None else r.score
    return f"{s:.4f}"


def render_hits(hits) -> str:
    if not hits:
        return (
            "### 没有检索到任何内容\n"
            "· 库里可能还没有文档 → 去「库状态」看一眼\n"
            "· 或者换一组关键词(陈述式短语比整句问句好)\n"
        )
    out = [f"**{len(hits)} 条结果**\n"]
    for i, r in enumerate(hits, 1):
        tag = "🕸️ " if r.meta.get("from_graph") else ""
        out.append(f"---\n### {i}. {tag}{r.citation}　`{_fmt_score(r)}`")
        if r.source and r.source != r.citation:
            out.append(f"**出处**:`{r.source}`")
        # 定位语(导入时模型写的那句「这段在原文的哪里、讲的是哪件事」)。
        # 界面上要看得见:排查召回问题时,最常要判断的就是
        # 「块被选对了没有、定位语有没有把它带到正确的语境里」。
        # 它是灰底引用块,和正文视觉上分开。
        if r.context:
            out.append(f"> 📍 **定位语**:{' '.join(r.context.split())}")
        body = " ".join((r.text or "").split())
        if len(body) > 700:
            body = body[:700] + " …"
        out.append("")
        out.append(body)
        if r.meta.get("entities"):
            ents = ", ".join(str(e) for e in r.meta["entities"][:10])
            out.append(f"\n**命中实体**:{ents}")
        facts = (r.meta.get("facts") or [])[:5]
        if facts:
            lines = "\n".join(
                f"- `{f['head']}` --{f['relation']}--> `{f['tail']}`" for f in facts
            )
            out.append(f"\n**关系**:\n{lines}")
    return "\n".join(out)


def render_steps(result) -> str:
    if not result.steps:
        return "_没有调用任何工具,直接作答。_"
    out = [f"**{result.iterations} 轮检索**\n"]
    for st in result.steps:
        marks = []
        if getattr(st, "repeated", False):
            marks.append("重复,已短路")
        if getattr(st, "truncated", False):
            marks.append("已截断")
        suffix = f"　`[{', '.join(marks)}]`" if marks else ""
        out.append(f"**{st.index}. `{st.action}`**({st.action_input}){suffix}")
        if st.thought:
            out.append(f"> Thought:{st.thought[:200]}")
        if getattr(st, "note", ""):
            out.append(f"> {st.note}")
        out.append("")
    return "\n".join(out)


def render_usage(result) -> str:
    u = result.usage or {}
    parts = [
        f"停止原因 `{result.stop_reason}`",
        f"LLM 调用 **{u.get('calls', 0)}** 次",
        f"tokens {u.get('prompt', 0)} + {u.get('completion', 0)}",
    ]
    if u.get("reasoning"):
        # 推理模型的思维链 token 也计费,不显示出来账单会看不懂
        parts.append(f"其中推理 **{u['reasoning']}**")
    return "　|　".join(parts)


# --------------------------------------------------------------------- #
# 事件处理
# --------------------------------------------------------------------- #


def do_ingest(files, path_text, force, recursive, backend_name, progress=gr.Progress()):
    # progress 的默认值必须是个 Progress 实例 —— Gradio 靠它认参数
    # (`special_args` 检查的是 `isinstance(default, Progress)`,写成 None 不会注入)。
    s = get_settings()
    lines: list[str] = []

    targets: list[Path] = []
    saved, notes = save_uploads(files)
    targets.extend(saved)
    if saved:
        lines.append("**已接收上传**:")
        lines.extend(f"- `{p.name}`" for p in saved)
        lines.append(f"\n(存到 `{UPLOAD_DIR}`)\n")

    raw = (path_text or "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.exists():
            lines.append(f"❌ 路径不存在:`{p}`")
            return "\n".join(lines)
        targets.append(p)
        lines.append(f"**已接收路径**:`{p}`\n")

    if not targets:
        return "❌ 没有要导入的东西 —— 上传文件,或在下面填一个路径。"

    if not s.llm.configured:
        lines.append(
            "⚠️ **LLM 未配置** —— 本次导入会跳过「上下文定位语」。\n"
            "文档仍能导入和检索,但召回质量会明显下降"
            "(尤其是块本身很短、指代很多的时候)。\n"
            "填上 `LLM_API_KEY` 后**重新导入即可补齐**,不用先删。\n"
        )

    backend = backend_for(backend_name)
    totals = {
        "files": 0, "loaded": 0, "indexed": 0, "skipped": 0,
        "chunks": 0, "missing": 0, "entities": 0, "relations": 0,
    }
    errors: list[str] = []
    t0 = time.time()
    calls0 = _llm_calls()

    for i, target in enumerate(targets):
        if progress is not None:
            try:
                progress(
                    (i / len(targets)),
                    desc=f"导入 {target.name}({i + 1}/{len(targets)})",
                )
            except Exception:  # noqa: BLE001
                pass
        try:
            st = backend.ingest_path(target, recursive=recursive, force=force)
        except Exception as exc:  # noqa: BLE001
            log.exception("导入失败: %s", target)
            errors.append(f"`{target.name}` → {type(exc).__name__}: {exc}")
            continue
        totals["files"] += st.files_seen
        totals["loaded"] += st.files_loaded
        totals["indexed"] += st.docs_indexed
        totals["skipped"] += st.docs_unchanged
        totals["chunks"] += st.chunks_written
        totals["missing"] += st.contextual_missing
        totals["entities"] += getattr(st, "entities_written", 0)
        totals["relations"] += getattr(st, "relations_written", 0)
        for src, err in st.errors:
            errors.append(f"`{src}` → {err}")

    if progress is not None:
        try:
            progress(1.0, desc="完成")
        except Exception:  # noqa: BLE001
            pass

    dt = time.time() - t0
    lines.append(f"### 后端 `{backend.name}` 导入完成({dt:.1f}s)\n")
    lines.append("| 项 | 数 |\n|---|---:|")
    lines.append(f"| 扫描文件 | {totals['files']} |")
    lines.append(f"| 读入 | {totals['loaded']} |")
    lines.append(f"| 新建/更新 | {totals['indexed']} |")
    lines.append(f"| 未变化跳过 | {totals['skipped']} |")
    lines.append(f"| 写入块 | {totals['chunks']} |")
    if totals["entities"] or totals["relations"]:
        lines.append(f"| 图谱实体 | {totals['entities']} |")
        lines.append(f"| 图谱关系 | {totals['relations']} |")
    if totals["missing"]:
        lines.append(f"| 缺定位语的块 | {totals['missing']} |")

    if totals["indexed"] == 0 and totals["skipped"] and not errors:
        lines.append(
            "\n✅ 内容都没变(`content_hash` 相同),全部跳过 —— 这是正常的,"
            "不会重复计费。想强制重导就勾上「强制重写」。"
        )
    if errors:
        lines.append(f"\n### ❌ {len(errors)} 个文件出错\n")
        lines.extend(f"- {e}" for e in errors[:20])
        if len(errors) > 20:
            lines.append(f"- …还有 {len(errors) - 20} 个")

    lines.append("\n" + _ingest_cost_note(calls0, _llm_calls(), totals))
    return "\n".join(lines)


def _ingest_cost_note(calls0: int | None, calls1: int | None, totals: dict) -> str:
    """导入结语:这次到底花没花 LLM 调用。

    之前这里写死一句「没有 LLM 调用」,是错的 —— 上下文增强(定位语)是
    **每块一次**调用,配了 key 的用户导 4 个块就悄悄花了 4 次,界面却说没花。
    这种"界面比实际乐观"的错最伤信任,所以改成拿全局计数器做差如实报。

    顺带解释**为什么**要有这笔开销,以及不想要时怎么办 ——
    本地嵌入/重排是免费的这点仍然成立,值得讲清楚。
    """
    if calls0 is None or calls1 is None:
        return ("> (读不到 LLM 调用计数,无法确认本次是否产生调用。"
                "嵌入与重排始终是本地模型,不花钱。)")

    used = calls1 - calls0
    if used:
        model = get_settings().llm.model
        return (
            f"> 本次共发出 **{used} 次 LLM 调用**(模型 `{model}`)。\n"
            "> 这笔开销来自**上下文增强**:每块让模型写一句定位语,"
            "再把它和正文一起送去嵌入,短块的召回会明显变好。\n"
            "> 嵌入和重排(BGE-M3 / bge-reranker-v2-m3)全程在本地跑,不计费。\n"
            "> 不想花这笔钱:清空 `.env` 里的 `LLM_API_KEY` 再导 —— "
            "文档照样能导入和检索,只是召回差些(界面上会标「缺定位语」)。"
        )

    if totals["indexed"] == 0 and totals["skipped"]:
        return ("> 本次没有 LLM 调用 —— 文档都未变化,整体跳过,"
                "定位语也没有重算。这正是 `content_hash` 想省下来的钱。")

    return ("> 本次没有 LLM 调用:嵌入与重排都是本地模型,"
            "定位语这一步未启用(没配 `LLM_API_KEY`,或后端不需要)。")


def do_search(query, top_k, backend_name, raw):
    query = (query or "").strip()
    if not query:
        return "❌ 查询是空的。"
    backend = backend_for(backend_name)
    t0 = time.time()
    try:
        hits = backend.retrieve(query, top_k=int(top_k or 5))
    except Exception as exc:  # noqa: BLE001
        log.exception("检索失败")
        return f"❌ 检索失败:{type(exc).__name__}: {exc}"
    dt = time.time() - t0

    md = render_hits(hits)
    md = f"_后端 `{backend.name}`,耗时 {dt:.1f}s_\n\n" + md
    if raw and hits:
        md += "\n\n---\n### 给模型看的那份文本\n```\n" + format_hits(hits) + "\n```"
    return md


def do_ask(question, backend_name, verbose):
    question = (question or "").strip()
    if not question:
        return "❌ 问题为空。", ""

    # 先设全局后端再构造 agent —— 工具表是构造时从 get_backend() 取的
    backend_for(backend_name)
    from agent.react import ReActAgent

    try:
        result = ReActAgent().run(question)
    except LLMNotConfigured as exc:
        return (
            f"❌ **{exc}**\n\n"
            "「问答」需要模型来决策(该查什么、够不够、怎么回答)。\n\n"
            "先用「检索」页签验证召回本身没问题。"
        ), ""
    except LLMError as exc:
        return f"❌ **模型调用失败**\n\n```\n{exc}\n```", ""
    except Exception as exc:  # noqa: BLE001
        log.exception("问答失败")
        return f"❌ {type(exc).__name__}: {exc}", ""

    detail = render_steps(result) + "\n\n---\n\n" + render_usage(result)
    if result.warnings:
        detail += "\n\n" + "\n".join(f"⚠️ {w}" for w in result.warnings)
    if verbose:
        detail += "\n\n---\n\n### 完整轨迹\n```\n" + result.transcript() + "\n```"

    answer = result.answer or "_(模型没有给出内容)_"
    if result.stop_reason != "final_answer":
        answer += (
            f"\n\n> ⚠️ 这轮没有正常给出最终答案(`stop_reason={result.stop_reason}`),"
            "上面这段是兜底返回的模型原文。"
        )
    return answer, detail


def do_docs(filter_text):
    from store.qdrant_store import get_store

    store = get_store()
    if not store.exists():
        return "库还是空的。去「导入文档」页签导点东西。"
    docs = store.list_docs()
    if not docs:
        return "库还是空的。"

    kw = (filter_text or "").strip().lower()
    if kw:
        docs = [
            d for d in docs
            if kw in (d.get("source") or "").lower()
            or kw in (d.get("title") or "").lower()
        ]
        if not docs:
            return f"没有文件名或标题包含 `{filter_text}` 的文档。"

    total = sum(d.get("chunks", 0) for d in docs)
    out = [f"**{len(docs)} 篇文档,共 {total} 块**\n", "| 块数 | 文档 | 路径 |", "|---:|---|---|"]
    for d in sorted(docs, key=lambda x: -(x.get("chunks") or 0)):
        title = d.get("title") or d.get("doc_id") or ""
        src = d.get("source") or ""
        out.append(f"| {d.get('chunks', 0)} | {title} | `{src}` |")
    return "\n".join(out)


def do_status(backend_name):
    backend = backend_for(backend_name)
    lines = [f"### 后端 `{backend.name}`\n"]

    try:
        h = backend.health()
        lines.append(f"**连通性**:{'✅ 正常' if h.ok else '❌ ' + str(h.error)}\n")
        if h.detail:
            lines.append("| 项 | 值 |\n|---|---|")
            for k, v in h.detail.items():
                lines.append(f"| {k} | {v if v is not None else '—'} |")
    except Exception as exc:  # noqa: BLE001
        log.exception("体检失败")
        lines.append(f"**连通性**:❌ {type(exc).__name__}: {exc}\n")

    try:
        st = backend.stats()
        st.pop("documents", None)
        lines.append("\n| 计数 | 值 |\n|---|---|")
        for k, v in st.items():
            lines.append(f"| {k} | {v} |")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"\n❌ 统计失败:{type(exc).__name__}: {exc}")

    s = get_settings()
    lines.append("\n| 配置 | 值 |\n|---|---|")
    lines.append(f"| LLM | {'✅ ' + s.llm.model if s.llm.configured else '❌ 未配置'} |")
    lines.append(f"| Tavily | {'✅' if s.web.configured else '❌ 未配置'} |")
    lines.append(f"| 重排 | {'开' if s.retrieval.rerank_enabled else '关'}"
                 f" (阈值 {s.retrieval.rerank_min_score},至少留 {s.retrieval.rerank_min_keep} 条) |")
    lines.append(f"| 上下文定位语 | {'开' if s.ingest.contextual_enabled else '关'} |")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# 界面
# --------------------------------------------------------------------- #

CSS = """
.hit-panel { max-height: 640px; overflow-y: auto; }
footer { visibility: hidden; }
"""


def build_ui():
    s = get_settings()
    default_backend = s.backend.value

    with gr.Blocks(title="agentic-kb", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown(
            "# agentic-kb\n"
            "本地知识库:混合检索(稠密 + 稀疏 + RRF + 重排)"
            ",可切 GraphRAG,上面架一个 ReAct 智能体。"
        )

        with gr.Row():
            backend = gr.Radio(
                choices=[b.value for b in BackendType],
                value=default_backend,
                label="后端",
                info="hybrid = 稠密+稀疏+RRF+重排;graphrag = 额外走 Neo4j 的实体多跳",
                scale=2,
            )
            # Markdown 组件不收 scale(gradio 5.16 里只有布局组件和输入类组件收),
            # 想要占宽就包一层 Column。
            with gr.Column(scale=3):
                status_badge = gr.Markdown(
                    f"**LLM** {'✅ `' + s.llm.model + '`' if s.llm.configured else '❌ 未配置(问答不可用)'}"
                    f"　**Tavily** {'✅' if s.web.configured else '❌'}"
                )

        with gr.Tabs():
            # ---------------- 导入 ----------------
            with gr.Tab("📥 导入文档"):
                gr.Markdown(
                    "上传的文件会以**原始文件名**存到 `agentic-kb/uploads/`,"
                    "再从那里导入 —— 这样块里记的「出处」才是个人看得懂、"
                    "重导时也对得上的路径。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        files = gr.File(
                            label="上传文件(可多选)",
                            file_count="multiple",
                            file_types=UPLOAD_EXTS,
                            height=220,
                        )
                        path_text = gr.Textbox(
                            label="或者填服务器上的路径(文件或目录)",
                            placeholder=r"例如 D:\docs 或 D:\docs\规程.pdf",
                        )
                        with gr.Row():
                            force = gr.Checkbox(
                                label="强制重写", value=False,
                                info="忽略内容哈希,全部重新分块嵌入(费时,但能补齐定位语)",
                            )
                            recursive = gr.Checkbox(label="递归子目录", value=True)
                        ingest_btn = gr.Button("开始导入", variant="primary")
                    with gr.Column(scale=1):
                        ingest_out = gr.Markdown(
                            "_还没开始。_\n\n"
                            "**重复导入是安全的** —— 按内容哈希判断,没变的会跳过。\n\n"
                            "**graphrag 后端**的导入会跑实体抽取,会调用 LLM,"
                            "比 hybrid 慢很多也费钱。"
                        )
                ingest_btn.click(
                    do_ingest,
                    inputs=[files, path_text, force, recursive, backend],
                    outputs=ingest_out,
                )

            # ---------------- 检索 ----------------
            with gr.Tab("🔍 检索"):
                gr.Markdown(
                    "只查库,**不需要 LLM key,不发任何外部请求**。"
                    "问答答得不对时先来这里 —— 能直接分清是「没召回到」还是「模型没答好」。"
                )
                with gr.Row():
                    query = gr.Textbox(
                        label="查询", scale=4,
                        placeholder="用陈述式短语,例如:绝缘子破损判据",
                    )
                    top_k = gr.Slider(
                        1, 20, value=s.retrieval.rerank_top_n, step=1,
                        label="最多返回条数", scale=2,
                    )
                    raw = gr.Checkbox(label="附上给模型看的文本", value=False, scale=1)
                search_btn = gr.Button("检索", variant="primary")
                search_out = gr.Markdown(elem_classes="hit-panel")
                search_btn.click(
                    do_search, inputs=[query, top_k, backend, raw], outputs=search_out,
                )
                query.submit(
                    do_search, inputs=[query, top_k, backend, raw], outputs=search_out,
                )

            # ---------------- 问答 ----------------
            with gr.Tab("💬 问答"):
                gr.Markdown(
                    "ReAct 智能体自己决定:查知识库(可查多次)、联网搜、还是列文档清单。"
                    "**需要 LLM key。**"
                )
                with gr.Row():
                    question = gr.Textbox(
                        label="问题", scale=4, placeholder="例如:绝缘子出现裂纹怎么办",
                    )
                    verbose = gr.Checkbox(
                        label="显示完整轨迹", value=False,
                        info="把模型每一步的 Thought/Action 原文都打出来",
                    )
                ask_btn = gr.Button("提问", variant="primary")
                answer_out = gr.Markdown()
                detail_out = gr.Markdown()
                ask_btn.click(
                    do_ask, inputs=[question, backend, verbose],
                    outputs=[answer_out, detail_out],
                )
                question.submit(
                    do_ask, inputs=[question, backend, verbose],
                    outputs=[answer_out, detail_out],
                )

            # ---------------- 库状态 ----------------
            with gr.Tab("📊 库状态"):
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("刷新体检 / 计数", variant="primary")
                        docs_filter = gr.Textbox(
                            label="文档清单按名称过滤", placeholder="留空 = 全部",
                        )
                        docs_btn = gr.Button("列出文档")
                        docs_out = gr.Markdown()
                    with gr.Column():
                        status_out = gr.Markdown()
                refresh_btn.click(do_status, inputs=[backend], outputs=status_out)
                docs_btn.click(do_docs, inputs=[docs_filter], outputs=docs_out)
                demo.load(do_status, inputs=[backend], outputs=status_out)

    return demo


# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="webui.py", description="agentic-kb 前端")
    p.add_argument("--host", default=os.getenv("KB_UI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("KB_UI_PORT", "7860")))
    p.add_argument("--share", action="store_true", help="生成公网临时链接(谨慎用)")
    p.add_argument("--no-browser", action="store_true", help="不要自动开浏览器")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    s = get_settings()
    print("=" * 68)
    print("agentic-kb 前端")
    print(f"  后端默认 : {s.backend.value}")
    print(f"  LLM      : {'已配置 ' + s.llm.model if s.llm.configured else '❌ 未配置(问答不可用)'}")
    print(f"  Tavily   : {'已配置' if s.web.configured else '❌ 未配置'}")
    print(f"  上传目录 : {UPLOAD_DIR}")
    print(f"  地址     : http://{args.host}:{args.port}")
    print("=" * 68)
    print("Qdrant / Neo4j 没起的话,先跑 scripts\\start_qdrant.bat"
          "(graphrag 还要 scripts\\start_neo4j.bat)")

    demo = build_ui()
    # 并发限 1:后端是全局单例,一边导入一边问答会互相踩。本机单人用,串行是对的。
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
