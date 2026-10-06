"""HTTP 路由。

这个文件**只做翻译**:请求体 → 一次下层调用 → JSON。检索口径、权限规则、
版本过滤、分块策略全在下层,这里一个都不重实现。

两处例外,都是为了"多一层判断"这件事本身有正当理由:

1. `/api/health` **不拿锁** —— 它必须在导入进行中也能立刻回答(见契约 §health)。
2. SSE 那一路把整段工作丢进**独立线程** —— 理由见下面 `_sse_stream`。

> 契约在 `api/CONTRACT.md`。这份文件里的每个响应形状都以它为准,
> 改一处就要三边一起改(契约 / 这里 / 前端)。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

# 和 webui.py 同一条:必须在导入任何会拉 transformers 的模块之前设好,
# 否则加载模型时会往 stdout 刷一堆进度条,把日志冲掉。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

from agent.permissions import AuditLog  # noqa: E402
from agent.tools import build_registry, format_hits  # noqa: E402
from api import serialize as S  # noqa: E402
from api.schemas import AskRequest, IngestPathRequest, SearchRequest  # noqa: E402
from api.state import api_guard, current_backend, hold_backend  # noqa: E402
from config import (  # noqa: E402
    BASE_DIR,
    EXPORT_DIR,
    UPLOAD_DIR,
    UPLOAD_EXTS,
    BackendType,
    get_settings,
)

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]

#: `eval/baselines/` —— 评测基线。API 只读它,从不写。
BASELINES_DIR = BASE_DIR / "eval" / "baselines"

#: SSE 静默多久算"卡住了"。
#:
#: 这**不是**延迟上限,是**卡死**探针:每轮工具调用前都会发 `action`,
#: 所以两次事件之间的间隔上限 = 一次 LLM 调用。模型思考久一点完全正常,
#: 但一个线程真死了的话,没有这个超时前端会永远停在最后一个 `action` 上 ——
#: 那种"界面看着像在运行、其实早就没了"的状态最难排查。
_SSE_IDLE_TIMEOUT = 600.0


# --------------------------------------------------------------------------- #
# 错误:统一成契约 §0.4 的形状
# --------------------------------------------------------------------------- #

#: 真正属于"依赖没起"的异常。
#:
#: ⚠️ 这里**故意不含** `neo4j.exceptions.Neo4jError`:它的覆盖面里有 Cypher
#: 语法/语义错误 —— 那是**我们的 bug**,不是 Neo4j 挂了。把它算成 503 会让
#: 人跑去重启数据库,而真正该做的是看 traceback。宁可不分类。
def _is_dependency_down(exc: BaseException) -> bool:
    names = [
        "qdrant_client.http.exceptions.ResponseHandlingException",
        "UnexpectedResponse",
        "httpx.ConnectError",
        "httpx.ConnectTimeout",
        "httpx.ReadTimeout",
        "httpx.TimeoutException",
        "openai.APIConnectionError",
        "openai.APITimeoutError",
        "ServiceUnavailable",
        "SessionExpired",
    ]
    mod = type(exc).__module__ or ""
    name = type(exc).__name__
    if name in names:
        return True
    for pat in ("qdrant_client", "httpx", "urllib3", "neo4j", "openai"):
        if mod.startswith(pat):
            # 上面已排除 Neo4jError;剩下的 neo4j 异常(连接层)算依赖没起
            return not (name == "Neo4jError")
    return False


def _http_error(exc: BaseException) -> HTTPException:
    """把下层异常翻成 HTTP 状态 + 人话 detail。

    500 的 detail **保留异常类型名**:只写一句"服务器内部错误"的话,
    用户连搜索都不知道搜什么,而 `ValidationError: …` 至少能指向代码。
    """
    if isinstance(exc, HTTPException):
        return exc
    detail = f"{type(exc).__name__}: {exc}".strip()
    if _is_dependency_down(exc):
        return HTTPException(
            status_code=503,
            detail=f"依赖服务连不上({detail})。检查 Qdrant / Neo4j 是否已启动。",
        )
    return HTTPException(status_code=500, detail=detail)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _llm_calls() -> int | None:
    """全局 LLM 调用次数。读不到返回 None。

    `webui.py` 有一份同名的。这里没有 import 它 —— 那个模块 `import gradio`,
    为了一个 5 行函数把整个 Gradio 前端拖进 API 进程不值得(而且启动会慢一大截)。

    **读不到就返回 None,不伪造成 0**:这个数字要直接写进响应告诉调用方
    "这次导入花了多少模型调用",报 0 比报不出来更糟(webui 那条注释同理)。
    """
    try:
        from llm.client import get_llm

        return int(get_llm().usage.get("calls", 0))
    except Exception:  # noqa: BLE001
        return None


#: 比对同名上传时一次读多少字节。1 MiB 是随手取的:够大(几十 MB 的
#: PDF 也就几十次循环),又不会为了比一个哈希把整份文件读进内存。
_DIGEST_BLOCK = 1 << 20


def _digest(path: Path) -> str:
    """文件的 sha256,边读边算。

    只用来判断"同名文件的**内容**变没变",不是入库那条链上的
    `content_hash`(`ingest/loader.py` 算的是**正文文本**的 sha1)。
    两个哈希的输入都不一样,别拿来互相印证。
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(_DIGEST_BLOCK), b""):
            h.update(blk)
    return h.hexdigest()


def _build_agent(payload: AskRequest):
    """按请求拼一个 agent(**必须在锁里调**)。

    必须在锁里的原因:工具表是构造时从 `get_backend()` 取后端的 ——
    锁外面构造的话,构造和 `hold_backend` 之间会被别的请求换掉后端,
    于是"选了 graphrag、工具却查 hybrid"。见 `api/state.py` 的模块头。
    """
    from agent.react import ReActAgent

    s = get_settings()
    # 写权限要**两把钥匙**:环境变量是运维给这台机器的上限,请求体是调用方
    # 这一次的意图。各自单独都不该能写 ——
    #
    #   * 只看请求体:任何能发 HTTP 的人都能让 agent 落盘,`AGENT_ALLOW_WRITE`
    #     就成了一个**看着在、其实没接线**的旋钮(这个项目刚为 `use_rerank`
    #     踩过一次)。
    #   * 只看环境变量:等于"开了就谁都能写",没有第二次确认的机会。
    s_master = bool(s.agent.allow_write)
    requested = bool(payload.allow_write)
    if requested and not s_master:
        logger.warning(
            "请求带了 allow_write=true,但服务端 AGENT_ALLOW_WRITE=false —— 按拒绝处理。"
        )
    allow = requested and s_master

    guard = api_guard(
        allow_write=allow,
        # 总开关关着时把**点名名单**清空(注意:清的是 `confirm_write_tools`
        # 这个"批准了哪些"的列表,**不是**工具表本身 —— 写工具仍在 registry 里,
        # 模型仍会去调,然后在 `ToolRegistry.run` 的授权环节被拒并留下一条
        # `denied` 审计)。清名单是为了免得日志里出现"用户点名了 export_report"
        # 却仍被拒,看起来像名单没生效。
        confirm_write_tools=payload.confirm_write_tools if allow else [],
        exports_dir=EXPORT_DIR,
        audit=AuditLog(s.agent.audit_log),
    )
    registry = build_registry(
        s.agent,
        guard=guard,
        include_superseded=payload.include_superseded,
        exports_dir=EXPORT_DIR,
    )
    return ReActAgent(registry=registry)


def _ask_blocking(payload: AskRequest) -> dict[str, Any]:
    """跑一轮问答,返回**和流式 `done.result` 同构**的那个对象。

    非流式路由和 SSE 都走这里 —— 两条路各写一份组装代码的话,
    它们迟早会在某个字段上分岔,而前端按"同构"写的收尾逻辑就会在
    其中一条路上静默出错。
    """
    t0 = time.time()
    with hold_backend(payload.backend):
        result = _build_agent(payload).run(payload.question)
    return S.result_to_dict(result, elapsed_ms=int((time.time() - t0) * 1000))


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _sse_response(payload: AskRequest) -> StreamingResponse:
    """把一轮问答**丢进独立线程**,事件经队列流出去。

    为什么必须开线程:`ReActAgent.run()` 是同步阻塞的,而且一跑就是几十秒
    (中间全是网络 IO)。直接在 async 生成器里 `await` 它会**冻住整个事件
    循环** —— 连 `/api/health` 都会没响应,而那正是契约里专门承诺
    "忙的时候也能答"的端点。`asyncio.to_thread` 也解决不了:那只是把阻塞
    调用挪走,拿不到中途的事件。

    所以:worker 线程持有那把大锁(和别的请求互斥,契约 §0.1),
    边跑边把事件塞进队列;事件循环这边只等着取。

    队列**不设上限**:设了的话,消费者慢了就会反过来阻塞 worker ——
    而 worker 正卡在 LLM 调用里,等于让一个渲染慢的前端拖慢一次推理。
    代价是断线后事件会堆在内存里直到 run 结束,数量是轮次级别(个位数),
    可忽略。
    """
    q: queue.Queue = queue.Queue()
    sentinel = object()
    t0 = time.time()

    def sink(event: dict[str, Any]) -> None:
        if event.get("type") == "done":
            # `done` 里装的是 AgentResult **对象**,不能直接 JSON 化。
            # 在这里换成和非流式响应体**同构**的 dict —— 前端于是可以用
            # 同一段代码收尾,不用为两条路各写一份。
            event = {
                "type": "done",
                "result": S.result_to_dict(
                    event["result"], elapsed_ms=int((time.time() - t0) * 1000)
                ),
            }
        q.put(event)

    def worker() -> None:
        try:
            with hold_backend(payload.backend):
                _build_agent(payload).run(payload.question, on_event=sink)
        except Exception as exc:  # noqa: BLE001
            logger.exception("问答失败")
            # 出错也要给前端一个**可收尾**的事件。只在服务端记日志的话,
            # 界面会永远停在最后一个 action 上。
            q.put({"type": "error", "detail": str(_http_error(exc).detail)})
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, name="api-ask", daemon=True).start()

    async def gen() -> Iterator[str]:
        while True:
            try:
                item = await asyncio.to_thread(q.get, True, _SSE_IDLE_TIMEOUT)
            except queue.Empty:
                yield _sse(
                    {
                        "type": "error",
                        "detail": (
                            f"超过 {int(_SSE_IDLE_TIMEOUT)}s 没有新事件,已断开。"
                            "服务端可能仍在跑(见日志)。"
                        ),
                    }
                )
                return
            if item is sentinel:
                return
            yield _sse(item)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # nginx 默认会把响应缓冲起来,那 SSE 就变成"憋到最后一次性吐"
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #

def create_app() -> FastAPI:
    app = FastAPI(
        title="agentic-kb API",
        description="法规知识库智能体 —— 检索 / 问答 / 导入 / 运维",
        version="1.0.0",
    )

    # ----------------------------------------------------------------- #
    # 错误处理
    # ----------------------------------------------------------------- #

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(_request, exc: RequestValidationError):
        """pydantic 的报错**不直接透传**。

        它给的是一串 `loc` / `type` / `ctx` 的 JSON path(形如
        `body -> top_k -> less_than_equal`),直接丢给用户看等于没写错误信息。
        这里压成一行:哪个字段、为什么。
        """
        parts = []
        for e in exc.errors()[:5]:
            loc = ".".join(str(x) for x in e.get("loc", ()) if x != "body")
            parts.append(f"{loc or '(请求体)'}: {e.get('msg', '不合法')}")
        more = len(exc.errors()) - 5
        detail = "请求参数有问题 —— " + ";".join(parts)
        if more > 0:
            detail += f"(另有 {more} 处)"
        return JSONResponse(status_code=400, content={"detail": detail})

    @app.exception_handler(Exception)
    async def _on_unhandled(_request, exc: Exception):
        err = _http_error(exc)
        if err.status_code >= 500:
            logger.exception("未处理的异常")
        return JSONResponse(status_code=err.status_code, content={"detail": err.detail})

    # ----------------------------------------------------------------- #
    # 运维
    # ----------------------------------------------------------------- #

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """**不拿锁、不切后端。** 见契约 §health。

        它必须在导入进行中(锁被占着几分钟)也能立刻回答 —— 不然前端
        "服务不可用"的横幅会在最不该出现的时候出现。
        """
        s = get_settings()
        backend = current_backend()
        try:
            h = backend.health().to_dict()
        except Exception as exc:  # noqa: BLE001
            # 探测本身炸了 = 这个后端不可用,不是服务器 500。
            # 健康检查的意义就是在这种情况下**仍然能回答**。
            logger.warning("健康探测失败: %s", exc)
            h = {
                "ok": False,
                "backend": backend.name,
                "detail": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "ok": bool(h.get("ok")),
            "backend": backend.name,
            "detail": h.get("detail") or {},
            "error": h.get("error") or "",
            "config": {
                "llm_configured": bool(s.llm.configured),
                "llm_model": s.llm.model,
                "tavily_configured": bool(s.web.configured),
                "rerank_enabled": bool(s.retrieval.rerank_enabled),
                # 两个阈值。`webui.py` 的「库状态」一直显示它们,契约这边
                # 之前只给了开关 —— 而"重排开着但一条都没留下"和"重排关着"
                # 在界面上长得一样,分不出来。数就是它们:
                # 低于 `rerank_min_score` 的丢掉,但至少留 `rerank_min_keep` 条。
                "rerank_min_score": s.retrieval.rerank_min_score,
                "rerank_min_keep": s.retrieval.rerank_min_keep,
                "contextual_enabled": bool(s.ingest.contextual_enabled),
                "include_superseded": bool(s.retrieval.include_superseded),
                "allow_write": bool(s.agent.allow_write),
            },
        }

    @app.get("/api/backends")
    def backends() -> dict[str, Any]:
        return {
            "backends": [b.value for b in BackendType],
            "default": get_settings().backend.value,
            "current": current_backend().name,
        }

    @app.get("/api/tools")
    def tools() -> dict[str, Any]:
        """工具清单。**前端闸 2 的复选框就是用这个渲染的。**

        为什么非要给前端一条接口,而不是让它写死 `export_report` /
        `mark_superseded`:写死的话,以后重命名一个写工具,前端那个复选框
        就会**静默地什么都不批准** —— 用户勾了它、以为批准了,而请求里
        带的是一个不存在的名字。那正是这个项目刚为 `use_rerank` 踩过的
        「看着在、其实没接线」,不该在前端再长一个。

        `allow_write` 回显的是**服务端**的总开关(`AGENT_ALLOW_WRITE`)。
        它是闸 1 的一半,另一半才是请求里的 `allow_write`(见上面
        `_build_agent`)。开关关着的时候,点名哪些工具**没有任何用** ——
        调用方要求前端据此把闸 2 的控件禁掉并说明原因,而不是让用户
        勾了半天才发现全被拒。

        **不拿锁**:这里只是把 `Tool` 对象构造出来读它们的元数据,
        不碰后端单例(每个工具都是**调用时**才去取 store / retriever)。
        所以导入进行中它照样能立刻回答。
        """
        s = get_settings()
        # 显式 `allow_write=False`:这条路由**只读元数据**,构造出来的
        # 这个 registry 永远不会被 `run()`,给它写权限是白给。
        reg = build_registry(s.agent, allow_write=False, exports_dir=EXPORT_DIR)
        return {
            "allow_write": bool(s.agent.allow_write),
            "tools": [
                {
                    "name": t.name,
                    "kind": t.kind,
                    "requires_confirm": bool(t.requires_confirm),
                    "description": t.description,
                    "parameter": t.parameter,
                }
                for t in reg.tools.values()
            ],
        }

    # ----------------------------------------------------------------- #
    # 检索
    # ----------------------------------------------------------------- #

    @app.post("/api/search")
    def search(payload: SearchRequest) -> dict[str, Any]:
        t0 = time.time()
        try:
            with hold_backend(payload.backend) as backend:
                hits = backend.retrieve(
                    payload.query,
                    top_k=payload.top_k,
                    include_superseded=payload.include_superseded,
                    explain=payload.explain,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("检索失败")
            raise _http_error(exc) from exc

        s = get_settings()
        return {
            "backend": backend.name,
            "elapsed_ms": int((time.time() - t0) * 1000),
            # 回显**实际生效**的口径:请求没传时前端要知道服务端用的是什么
            # (见契约 §0.2 的同一条理由)。
            "include_superseded": (
                s.retrieval.include_superseded
                if payload.include_superseded is None
                else payload.include_superseded
            ),
            "hits": S.hits_to_list(hits),
            # 永远返回。这是"检索结果"和"模型实际看到的东西"之间
            # **唯一**能核对的地方 —— 少了它,同一个问题检索页和问答页
            # 给出不同结论时无从查起。
            "llm_text": format_hits(hits) if hits else "",
        }

    # ----------------------------------------------------------------- #
    # 问答
    # ----------------------------------------------------------------- #

    @app.post("/api/ask")
    def ask(payload: AskRequest):
        if not payload.stream:
            try:
                return _ask_blocking(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("问答失败")
                raise _http_error(exc) from exc
        return _sse_response(payload)

    # ----------------------------------------------------------------- #
    # 导入
    # ----------------------------------------------------------------- #

    def _do_ingest(
        targets: list[Path], *, recursive: bool, force: bool, backend_name: str | None
    ) -> dict[str, Any]:
        """两个 ingest 路由共用的下半段。

        只把"怎么拿到 targets"留给各自的入口 —— 落盘/校验那部分两条路
        确实不同,但导入本身和计数完全相同。分开写必然出现"JSON 那条路
        少算了一个字段"这种偏差。
        """
        t0 = time.time()
        calls0 = _llm_calls()
        totals = dict.fromkeys(
            (
                "files_seen",
                "files_loaded",
                "docs_indexed",
                "docs_unchanged",
                "chunks_written",
                "chunks_failed",
                "contextual_missing",
                "entities_written",
                "relations_written",
            ),
            0,
        )
        errors: list[dict[str, str]] = []

        with hold_backend(backend_name) as backend:
            for target in targets:
                try:
                    st = backend.ingest_path(target, recursive=recursive, force=force)
                except Exception as exc:  # noqa: BLE001
                    # 单个文件失败**不拖垮整批**:Gradio 那边就是这行为,
                    # 而"一个坏 pdf 让整次导入白跑"是不可接受的。
                    logger.exception("导入失败: %s", target)
                    errors.append({"file": target.name, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                totals["files_seen"] += st.files_seen
                totals["files_loaded"] += st.files_loaded
                totals["docs_indexed"] += st.docs_indexed
                totals["docs_unchanged"] += st.docs_unchanged
                totals["chunks_written"] += st.chunks_written
                totals["chunks_failed"] += st.chunks_failed
                totals["contextual_missing"] += st.contextual_missing
                totals["entities_written"] += getattr(st, "entities_written", 0)
                totals["relations_written"] += getattr(st, "relations_written", 0)
                for src, err in st.errors:
                    errors.append({"file": Path(str(src)).name, "error": str(err)})
            name = backend.name

        calls1 = _llm_calls()
        return {
            "backend": name,
            "elapsed_ms": int((time.time() - t0) * 1000),
            **totals,
            # 全局计数器做差。**不是** IngestStats 上的字段 —— 这个数来自
            # LLM 客户端自己的累计调用数,所以只能这么量。
            # 有值就一定有开销:定位语(contextual)是每块一次调用,
            # 界面写死"本次没有 LLM 调用"是错的(webui 那边踩过)。
            "llm_calls": (
                None if calls0 is None or calls1 is None else calls1 - calls0
            ),
            "errors": errors,
        }

    @app.post("/api/ingest")
    def ingest_path(payload: IngestPathRequest) -> dict[str, Any]:
        # 路径不存在 → 400 而不是 500:这是**请求**错了,不是服务器坏了。
        p = Path(payload.path)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"路径不存在:{p}")
        return _do_ingest(
            [p], recursive=payload.recursive, force=payload.force, backend_name=payload.backend
        )

    @app.post("/api/ingest/upload")
    def ingest_upload(
        files: list[UploadFile] = File(...),
        recursive: bool = Form(True),
        force: bool = Form(False),
        backend: str | None = Form(None),
    ) -> dict[str, Any]:
        """落到 `uploads/` 再导入,不直接用临时路径。

        临时路径会进块的 `source` 字段,溯源就指向一个过一会儿就不存在的
        地方 —— 而界面上看起来完全正常。Gradio 那边也是这个理由。
        """
        if not files:
            raise HTTPException(status_code=400, detail="没有收到文件。")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        notes: list[str] = []
        for f in files:
            # 只取文件名部分。`filename` 是**外部输入**,带 `..\..\` 就能写到
            # uploads/ 外面去 —— 不信它。
            name = Path(str(f.filename or "")).name
            if not name:
                raise HTTPException(status_code=400, detail="上传的文件没有文件名。")
            ext = Path(name).suffix.lower()
            if ext not in UPLOAD_EXTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型 {ext!r}。允许:{', '.join(UPLOAD_EXTS)}",
                )
            dst = UPLOAD_DIR / name
            # 同名覆盖要**说出来**。`webui.py` 的 `save_uploads` 也做这件事,
            # 这里之前漏了 —— 而漏掉的后果不是报错,是静默:上传一份同名但
            # 内容不同的文件,旧的那份就没了,界面上却一路正常。`uploads/`
            # 里的路径正是块 `source` 的来源,"我导过这篇"的溯源码被换掉
            # 而没人提,是最难查的一类不一致。
            #
            # 只比内容,不比时间戳:同一份文件重新上传一遍不该报警。
            before = _digest(dst) if dst.exists() else None
            h = hashlib.sha256()
            with dst.open("wb") as out:
                while True:
                    blk = f.file.read(_DIGEST_BLOCK)
                    if not blk:
                        break
                    h.update(blk)
                    out.write(blk)
            if before is not None and before != h.hexdigest():
                notes.append(f"{name} 已在 uploads/ 里且内容不同,已覆盖")
            saved.append(dst)

        out = _do_ingest(
            saved, recursive=recursive, force=force, backend_name=backend
        )
        out["saved"] = [str(p) for p in saved]
        out["notes"] = notes
        return out

    # ----------------------------------------------------------------- #
    # 文档 / 统计 / 审计 / 评测
    # ----------------------------------------------------------------- #

    @app.get("/api/docs")
    def docs(q: str = "") -> dict[str, Any]:
        from store.qdrant_store import get_store

        # 走锁:它要和"导入中"(改了库)互斥,口径和 Gradio 的 concurrency=1 一致。
        with hold_backend(None):
            store = get_store()
            if not store.exists():
                return {"total_docs": 0, "total_chunks": 0, "filtered": 0, "docs": []}
            all_docs = store.list_docs()

        total_chunks = sum(d.get("chunks") or 0 for d in all_docs)
        kw = (q or "").strip().lower()
        shown = all_docs
        if kw:
            shown = [
                d
                for d in all_docs
                if kw in (d.get("source") or "").lower()
                or kw in (d.get("title") or "").lower()
            ]
        # 前端要能显示"筛选后 / 总数",所以两个数都给 —— 只给 filtered 的话
        # 用户分不清"库里就这些"和"筛掉了大半"。
        return {
            "total_docs": len(all_docs),
            "total_chunks": total_chunks,
            "filtered": len(shown),
            "docs": sorted(shown, key=lambda d: -(d.get("chunks") or 0)),
        }

    @app.get("/api/docs/{doc_id}")
    def doc_chunks(doc_id: str) -> dict[str, Any]:
        """一篇文档的**全部块**,按 `chunk_index` 排好序 —— 给"点引用回溯原文"用。

        为什么非要有这条路由:检索结果里的 `hit.text` 只是**那一块**的正文,
        而人要看的是"这句话在原文里是什么位置、前后在讲什么"。只给一块的话,
        引用能看见但回溯不了 —— 用户没法确认它是不是被断章取义了,而这正是
        "引用可核验"唯一有意义的地方。

        **Gradio 没有这个能力**(它的库状态页只列文档,不展开正文),所以这条
        API 比 Gradio 多。计划里 Batch 4 的验收写着"前端能点开引用回溯到原文",
        这是它需要的那块后端。

        ⚠️ 返回的是**全文块**,没有做条数上限:一篇文档在这里就是几个到几十个
        块(自检语料最多 7 块/篇),分页反而会让"回溯"变成翻页。真接了大文档
        要改成分页,那时按 `chunk_index` 游标切。
        """
        from store.qdrant_store import get_store

        # 走锁,理由和 `/api/docs` 一样:它读的是导入会改的那张表,
        # 要和"导入中"互斥(口径同 Gradio 的 concurrency=1)。
        with hold_backend(None):
            store = get_store()
            if not store.exists():
                raise HTTPException(status_code=404, detail=f"文档不存在:{doc_id}")
            records = store.doc_records(doc_id)

        if not records:
            raise HTTPException(status_code=404, detail=f"文档不存在:{doc_id}")

        rows = []
        for rec in records:
            p = dict(rec.payload or {})
            rows.append(
                {
                    "chunk_index": int(p.get("chunk_index") or 0),
                    "text": p.get("text") or "",
                    # 定位语(导入时模型写的"这段在讲什么"),可能为空串 ——
                    # 回溯时要的就是它:块本身看不出自己在全文的位置
                    "context": p.get("context") or "",
                    "status": p.get("status") or "",
                    "doc_version": p.get("doc_version") or "",
                }
            )
        rows.sort(key=lambda r: r["chunk_index"])
        head = dict(records[0].payload or {})
        return {
            "doc_id": doc_id,
            "source": head.get("source") or "",
            "title": head.get("title") or "",
            "total": len(rows),
            "chunks": rows,
        }

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        try:
            with hold_backend(None) as backend:
                data = backend.stats()
                name = backend.name
        except Exception as exc:  # noqa: BLE001
            logger.exception("取统计失败")
            raise _http_error(exc) from exc

        # 空库时下层不给 `documents`(见契约 §stats)—— **原样透传**,
        # 不在这里补一个空列表:补了的话前端就分不清"还没导东西"和
        # "导了但一篇都没读进来",而这两个的处置完全不同。
        out = {"backend": name}
        if isinstance(data, dict):
            out.update(data)
            out["backend"] = name
        else:
            out["detail"] = data
        return out

    @app.get("/api/audit")
    def audit(n: int = 20) -> dict[str, Any]:
        """读审计日志末尾 n 条。

        按 call_id 配对是**前端**的事 —— 并发写会交错,按顺序配对会张冠李戴
        (见契约 §audit)。
        """
        n = max(1, min(int(n), 1000))
        path = Path(get_settings().agent.audit_log)
        if not path.exists():
            # 还没发生过任何写操作。**这不是错误** —— 刚从没写过库的状态
            # 打开页面时必现,报错会让用户以为审计坏了。
            return {"records": []}

        records: list[dict[str, Any]] = []
        bad = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in deque(f, maxlen=n):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 只跳过坏行,不整段失败:审计日志可能正被另一个进程
                        # 追加(读到写了一半的行是正常的)。
                        bad += 1
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"读审计日志失败:{exc}") from exc

        if bad:
            logger.warning("审计日志有 %d 行无法解析(已跳过)", bad)
        return {"records": records}

    @app.get("/api/eval")
    def eval_baselines(full: int = 0) -> dict[str, Any]:
        """评测基线**原样透传**,不做二次计算。

        在这个层重算一遍 hit@1,就多出一个可能和基线文件不一致的口径 ——
        而前端要展示的必须是**跑出来的那个数**(契约 §eval)。
        """
        if not BASELINES_DIR.is_dir():
            return {"baselines": []}

        out: list[dict[str, Any]] = []
        for p in sorted(BASELINES_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("跳过读不了的基线 %s: %s", p.name, exc)
                continue
            if not isinstance(data, dict):
                continue
            if not full:
                runs = data.get("runs")
                if isinstance(runs, dict):
                    for run in runs.values():
                        if isinstance(run, dict):
                            # 逐题明细占文件九成体积,默认剥掉。
                            run.pop("per_item", None)
                            run.pop("negative_items", None)
            data["name"] = p.stem
            out.append(data)

        # 最新的排前面。`created_at` 缺失的排最后 —— 排序键缺失时不能让它
        # 抢到首位,否则一份残缺文件会把最新那份挤下去。
        out.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
        return {"baselines": out}

    return app


app = create_app()
