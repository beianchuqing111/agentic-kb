"""API 层自检:逐条走一遍契约里的路由,对着真的 Qdrant / Neo4j 跑。

用 `TestClient` 在**进程内**跑,不开端口 —— 端口那套留给
`scripts/api_server.py` 手工验收(见 README)。

跑法:
    set PYTHONIOENCODING=utf-8
    python scripts\\check_api.py            # 免费档:不打模型,不花 LLM 调用
    python scripts\\check_api.py --ask      # 额外真跑一轮问答(会花 LLM 调用)

`--ask` 为什么是分开的:检索要加载 bge-m3(十几秒),问答还要连模型。
默认档只验证"契约的形状"和"错误码对不对",这些用不着模型 ——
每次改一行路由都等半分钟加载模型,自检就会没人跑。

重点验证:
  1. 每个路由的响应形状和 `api/CONTRACT.md` 对得上(尤其 `metrics` 不许存在、
     `per_item` 默认被剥掉)
  2. 参数错误一律 400 且 `detail` 是**人话**(不是 pydantic 的 JSON path)
  3. `backend` 字段的三种语义(切/省略=当前/不认识)
  4. 写工具的**两把钥匙**:请求要 `allow_write`,服务端环境变量也得开
  5. SSE 的事件序列:`action` 必须在 `done` 前面(前端靠它防卡死感)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from config import UPLOAD_DIR, get_settings  # noqa: E402

FAILED: list[str] = []


def check(cond: bool, what: str, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {what}")
    else:
        FAILED.append(what)
        print(f"  ❌ {what}" + (f"\n       {detail}" if detail else ""))


#: 拒绝在界面上长什么样。`WriteGuard` 的每条 deny 都经 `ToolRegistry.run`
#: 变成以「错误:」开头的一句 —— 但**只判断这一个前缀不够**:
#: 一个正常执行、只是没法完成的只读工具也会返回「错误:…」(比如列表为空)。
#: 所以这里要同时认出几种具体的拒绝说法。
_DENY_MARKERS = ("拒绝", "未启用", "越界", "只能", "不允许", "不在允许")


def is_denied(out: str) -> bool:
    return out.startswith("错误:") and any(m in out for m in _DENY_MARKERS)


def section(title: str) -> None:
    print(f"\n[{title}]")


#: 自检往库/上传目录里放的东西,跑完必须删干净。
#: `check_store.py` 用的是"另建一个 collection",这里做不到 ——
#: ingest 路由只会往配置里的那个 collection 写。所以改成事后清理,
#: 并且**只删我们自己造的那几个名字**,不做任何"清空库"之类的动作。
SELF_CHECK_NAMES = {
    "自检逃逸.md",
    # 第一版自检用的名字。留着是因为那次跑**真的**往库里塞了一篇,
    # 不列在这里的话它永远躺在库里没人认领。
    "逃逸.md",
}


def cleanup() -> None:
    """删掉自检自己造进库和 uploads/ 的文档。"""
    from store.qdrant_store import get_store

    try:
        store = get_store()
        if store.exists():
            for d in store.list_docs():
                src = Path(str(d.get("source") or "")).name
                if src in SELF_CHECK_NAMES:
                    store.delete_by_doc(d["doc_id"])
                    print(f"     (已从库里删掉自检文档 {src} / {d['doc_id']})")
    except Exception as exc:  # noqa: BLE001
        print(f"     ⚠️ 清理库失败(记得手工删):{exc}")
    for name in SELF_CHECK_NAMES:
        p = UPLOAD_DIR / name
        try:
            if p.exists():
                p.unlink()
                print(f"     (已删掉 uploads/{name})")
        except OSError as exc:
            print(f"     ⚠️ 删 {p} 失败:{exc}")


def main(argv: list[str]) -> int:
    with_ask = "--ask" in argv
    s = get_settings()
    client = TestClient(app)

    # ----------------------------------------------------------------- #
    section("1. GET /api/health —— 不拿锁,形状固定")
    r = client.get("/api/health")
    check(r.status_code == 200, "200", r.text[:200])
    h = r.json()
    for key in ("ok", "backend", "detail", "error", "config"):
        check(key in h, f"顶层有 {key!r}")
    cfg = h.get("config", {})
    for key in (
        "llm_configured",
        "llm_model",
        "tavily_configured",
        "rerank_enabled",
        "contextual_enabled",
        "include_superseded",
        "allow_write",
    ):
        check(key in cfg, f"config 里有 {key!r}")
    print(f"     backend={h['backend']!r} ok={h['ok']} err={h['error'][:60]!r}")
    check(
        h["backend"] in {"hybrid", "graphrag"},
        "backend 是后端名而不是路径",
        h["backend"],
    )

    # ----------------------------------------------------------------- #
    section("2. GET /api/backends")
    r = client.get("/api/backends")
    check(r.status_code == 200, "200", r.text[:200])
    b = r.json()
    check(set(b) == {"backends", "default", "current"}, "键恰好是三个", str(set(b)))
    check("hybrid" in b["backends"] and "graphrag" in b["backends"], "两个后端都在")
    check(b["current"] == h["backend"], "current 和 health 报的一致")

    # ----------------------------------------------------------------- #
    section("3. GET /api/eval —— 评测数字原样透传")
    r = client.get("/api/eval")
    check(r.status_code == 200, "200", r.text[:200])
    ev = r.json()
    base = ev.get("baselines") or []
    check(len(base) > 0, "读到了基线文件(跑过 Batch 1 就该有)", str(len(base)))
    if base:
        first = base[0]
        check("name" in first, "带 name(文件名)")
        check("created_at" in first, "带 created_at")
        check("metrics" not in first, "**没有** metrics(契约明说它不存在)")
        runs = first.get("runs") or {}
        check(isinstance(runs, dict), "runs 是 dict 不是 list", type(runs).__name__)
        if runs:
            k0 = next(iter(runs))
            run0 = runs[k0]
            check("aggregates" in run0, f"runs[{k0!r}] 有 aggregates")
            agg = run0.get("aggregates") or {}
            check(
                all(isinstance(k, str) for k in agg),
                "aggregates 的 k 是**字符串**(不是 int)",
                str([type(k).__name__ for k in list(agg)[:3]]),
            )
            check("per_item" not in run0, "默认剥掉 per_item")
            check("negative_items" not in run0, "默认剥掉 negative_items")
        dates = [str(x.get("created_at") or "") for x in base]
        check(dates == sorted(dates, reverse=True), "按 created_at 倒序")

        r2 = client.get("/api/eval", params={"full": 1})
        check(r2.status_code == 200, "full=1 也是 200")
        full_runs = (r2.json()["baselines"][0].get("runs") or {})
        if full_runs:
            k0 = next(iter(full_runs))
            check("per_item" in full_runs[k0], "full=1 时 per_item 回来了")

    # ----------------------------------------------------------------- #
    section("4. GET /api/audit")
    r = client.get("/api/audit")
    check(r.status_code == 200, "200(文件不存在也不许报错)", r.text[:200])
    check("records" in r.json(), "有 records 键")
    n_before = len(r.json()["records"])
    print(f"     当前 {n_before} 条(没写过就是 0,正常)")
    check(client.get("/api/audit", params={"n": 0}).status_code == 200, "n=0 不炸")
    check(client.get("/api/audit", params={"n": -5}).status_code == 200, "n 为负不炸")
    check(
        client.get("/api/audit", params={"n": "abc"}).status_code == 400,
        "n 不是数字 → 400",
    )

    # ----------------------------------------------------------------- #
    section("5. GET /api/docs")
    r = client.get("/api/docs")
    check(r.status_code == 200, "200", r.text[:200])
    d = r.json()
    check(
        set(d) == {"total_docs", "total_chunks", "filtered", "docs"},
        "键恰好是四个",
        str(set(d)),
    )
    check(d["total_docs"] == d["filtered"], "不带 q 时 filtered == total_docs")
    check(len(d["docs"]) == d["filtered"], "docs 长度和 filtered 一致")
    if d["docs"]:
        one = d["docs"][0]
        for key in (
            "doc_id",
            "title",
            "source",
            "chunks",
            "status",
            "mixed",
            "doc_version",
            "effective_from",
            "effective_to",
        ):
            check(key in one, f"doc 里有 {key!r}")
        ch = [x["chunks"] for x in d["docs"]]
        check(ch == sorted(ch, reverse=True), "按块数倒序")
        title = (d["docs"][0].get("title") or "")[:2]
        if title:
            r2 = client.get("/api/docs", params={"q": title})
            check(
                r2.json()["filtered"] <= d["filtered"], f"q={title!r} 是过滤不是扩增"
            )
    r3 = client.get("/api/docs", params={"q": "绝不可能出现的字符串zzz"})
    check(r3.status_code == 200 and r3.json()["filtered"] == 0, "匹配不到时 filtered=0")
    check(
        r3.json()["total_docs"] == d["total_docs"],
        "**过滤后 total_docs 不变**(前端要能显示 筛/总)",
    )

    # ----------------------------------------------------------------- #
    section("6. GET /api/stats")
    r = client.get("/api/stats")
    check(r.status_code == 200, "200", r.text[:300])
    st = r.json()
    check("backend" in st, "有 backend")
    check(st.get("backend") == h["backend"], "和 health 报的后端一致")
    print(f"     键:{sorted(st)}")

    # ----------------------------------------------------------------- #
    section("7. 参数错误 → 400 且 detail 是人话")
    cases = [
        ("POST", "/api/search", {"json": {}}, "缺 query"),
        ("POST", "/api/search", {"json": {"query": "   "}}, "query 全是空格"),
        ("POST", "/api/search", {"json": {"query": "x", "top_k_": 5}}, "字段名拼错"),
        ("POST", "/api/search", {"json": {"query": "x", "top_k": 0}}, "top_k 越界"),
        ("POST", "/api/search", {"json": {"query": "x", "top_k": 999}}, "top_k 越界(大)"),
        ("POST", "/api/ask", {"json": {"question": ""}}, "缺 question"),
        ("POST", "/api/ingest", {"json": {}}, "缺 path"),
        ("POST", "/api/ingest", {"json": {"path": "Z:/不存在的路径/xx"}}, "路径不存在"),
    ]
    for method, url, kw, what in cases:
        r = client.request(method, url, **kw)
        detail = ""
        try:
            detail = r.json().get("detail", "")
        except Exception:  # noqa: BLE001
            detail = r.text[:120]
        check(r.status_code == 400, f"{what} → 400", f"实际 {r.status_code}: {detail}")
        check(
            isinstance(detail, str) and detail and "->" not in detail,
            f"{what} 的 detail 是人话(不含 JSON path)",
            repr(detail)[:160],
        )
        print(f"     {what}: {detail[:110]}")

    # ----------------------------------------------------------------- #
    section("8. backend 字段的三种语义(契约 §0.2)")
    r = client.get("/api/backends")
    default_name = r.json()["default"]
    # 不认识的名字:退回默认,但**不报错**
    r = client.post("/api/search", json={"query": "x", "backend": "vector-nonsense"})
    check(
        r.status_code in (200, 503, 500),
        "不认识的后端名不返回 400(退回默认)",
        f"{r.status_code}: {r.text[:160]}",
    )
    print(f"     不认识的名字 → {r.status_code}(期望 200,模型没加载时 503/500 也算过)")

    # ----------------------------------------------------------------- #
    section("8b. POST /api/search 的 explain 开关(名次不是摆设)")
    # 这一节存在的理由:契约里 `scores.dense_rank`/`sparse_rank` 写明了
    # 「null = 这一路没召回它」,而这两个字段**曾经永远是 null** ——
    # 算名次的那段代码被关在 `if debug is not None:` 里,而 API 从不传 debug。
    # 一条恒为 null 的字段比没有这个字段更坏:它看起来在工作。
    q = {"query": "绝缘子检测的周期", "top_k": 3}
    r = client.post("/api/search", json=q)
    if r.status_code != 200 or not r.json().get("hits"):
        # 不假装通过:没有真命中就无从判断名次有没有被填上。
        print(
            f"     跳过:检索没给出结果({r.status_code})—— 空库或模型没起来时"
            "名次断言无从下手,不在这里假装通过。"
        )
    else:
        off = r.json()["hits"]
        on_r = client.post("/api/search", json={**q, "explain": True})
        check(on_r.status_code == 200, "explain=true 也返回 200", on_r.text[:200])
        on = on_r.json()["hits"]
        check(
            all(
                h["scores"]["dense_rank"] is None and h["scores"]["sparse_rank"] is None
                for h in off
            ),
            "默认档两路名次全是 null(默认不付那两次额外查询的代价)",
        )
        check(
            any(
                h["scores"]["dense_rank"] is not None or h["scores"]["sparse_rank"] is not None
                for h in on
            ),
            "explain=true 时真的有名次(不是恒为 null 的死字段)",
        )
        check(
            [h["doc_id"] for h in off] == [h["doc_id"] for h in on],
            "explain 只往结果上加信息,不改变排序",
        )
        # graphrag 的 `**kwargs` 是**用来报错**的:explain 没登记进签名的话
        # 这里会 500(TypeError),正是当初 use_rerank 被静默吞掉的那类事故。
        # 依赖没起来时 503 是允许的 —— 但 500 不是。
        rg = client.post("/api/search", json={**q, "backend": "graphrag", "explain": True})
        check(
            rg.status_code in (200, 503),
            "graphrag 后端也认 explain(不因未知参数 500)",
            f"{rg.status_code}: {rg.text[:200]}",
        )

    # ----------------------------------------------------------------- #
    section("9. 写工具的两把钥匙 + 三道闸(不需要模型)")
    import tempfile

    from agent.permissions import AuditLog
    from agent.tools import build_registry
    from api.app import _build_agent
    from api.schemas import AskRequest
    from api.state import api_guard, hold_backend

    master = bool(s.agent.allow_write)
    print(f"     服务端 AGENT_ALLOW_WRITE={master}(钥匙 2)")

    with hold_backend(None):
        # 闸 1 的**上半把**(服务端总开关)。只构造不调用 —— 调用会往真实
        # 审计日志里写记录,自检不该留这种垃圾。
        a = _build_agent(AskRequest(question="x", allow_write=True, confirm_write_tools=["export_report"]))
        check(
            a.registry.guard.allow_write is False,
            "服务端 AGENT_ALLOW_WRITE=false 时,请求带 allow_write 也**不给**写权限",
            f"guard.allow_write={a.registry.guard.allow_write}",
        )
        # 只读工具不受写权限影响 —— 别把权限层做成"关了就什么都不能干"
        out = a.registry.run("list_documents", "")
        check(not is_denied(out), "只读工具(list_documents)不受写权限影响", out[:120])

    # 三道闸各自单独验:在临时目录里造 guard + 审计,用完即弃。
    # 为什么不用 `_build_agent`:它的 guard 闸 1 被环境变量压着,
    # 闸 1 一拦,后面两道根本没机会跑 —— 那样测出来的"闸 3 有效"是假的。
    with tempfile.TemporaryDirectory(prefix="kbselfcheck_") as tmp:
        tmpd = Path(tmp)
        exports = tmpd / "exports"
        exports.mkdir()
        audit_path = tmpd / "audit.jsonl"

        def guard_for(allow: bool, tools: list[str]):
            return build_registry(
                s.agent,
                guard=api_guard(
                    allow_write=allow,
                    confirm_write_tools=tools,
                    exports_dir=exports,
                    audit=AuditLog(audit_path),
                ),
                exports_dir=exports,
            )

        out = guard_for(False, ["export_report"]).run("export_report", "自检.md | hello")
        check(is_denied(out), "闸 1:allow_write=False → 拒", out[:160])

        out = guard_for(True, []).run("export_report", "自检.md | hello")
        check(
            is_denied(out),
            "闸 2:allow_write=True 但**没点名工具** → 仍拒(两闸不塌成一闸)",
            out[:160],
        )

        out = guard_for(True, ["export_report"]).run(
            "export_report", "..\\..\\逃逸.md | hello"
        )
        check(is_denied(out), "闸 3:点名了工具,但路径带 .. → 拒", out[:200])
        check(
            not (tmpd.parent / "逃逸.md").exists() and not (tmpd / "逃逸.md").exists(),
            "闸 3 拒绝后**没有**在 exports/ 之外留下文件",
        )

        out = guard_for(True, ["export_report"]).run("export_report", "自检.md | hello")
        check(not is_denied(out), "三道闸都过了 → 放行", out[:160])
        check((exports / "自检.md").exists(), "文件真的落在 exports/ 里")

        # 审计:拒绝和成功必须分得开 —— 混在一起等于把安全事件藏进错误日志
        if audit_path.exists():
            recs = [json.loads(x) for x in audit_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            phases = [r.get("phase") for r in recs]
            check("denied" in phases, "被拒的尝试进了审计(phase=denied)", str(phases))
            check("intent" in phases, "成功的写先记 intent", str(phases))
            check("result" in phases, "...再记 result", str(phases))
            check(
                all("call_id" in r for r in recs),
                "每条都带 call_id(前端按它配对,不按顺序)",
            )
            print(f"     审计:{phases}")
        else:
            check(False, "审计文件被创建了")

    # ----------------------------------------------------------------- #
    section("10. 上传校验(multipart,不需要模型)")
    r = client.post(
        "/api/ingest/upload",
        files={"files": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    check(r.status_code == 400, "不支持的扩展名 → 400", r.text[:200])
    check(
        ".exe" in json.dumps(r.json(), ensure_ascii=False),
        "detail 里点明了是哪个类型",
        r.text[:200],
    )
    # 带 `..` 的文件名:只取 basename,不许写到 uploads/ 外面。
    # ⚠️ 这条**会真的入库**,所以跑完必须清掉 —— 自检不留垃圾。
    r = client.post(
        "/api/ingest/upload",
        files={"files": ("../自检逃逸.md", b"# selfcheck\n", "text/markdown")},
    )
    check(r.status_code == 200, "带 .. 的文件名不崩", r.text[:200])
    if r.status_code == 200:
        saved = r.json().get("saved") or []
        check(len(saved) == 1, "落盘 1 个文件")
        if saved:
            p = Path(saved[0])
            check(
                p.parent == UPLOAD_DIR and p.name == "自检逃逸.md",
                "落盘路径被压在 uploads/ 下、只取了 basename",
                str(p),
            )
        check(
            any(e.get("file") == "自检逃逸.md" for e in (r.json().get("errors") or [])) is False,
            "这个文件本身没出错",
            str(r.json().get("errors")),
        )
        check("llm_calls" in r.json(), "响应带 llm_calls(入库成本,可为 null)")
        print(f"     llm_calls={r.json().get('llm_calls')}  "
              f"chunks_written={r.json().get('chunks_written')}")

    section("清理(自检不留垃圾)")
    cleanup()

    # ----------------------------------------------------------------- #
    if with_ask:
        section("11. POST /api/ask —— 真跑一轮(花 LLM 调用)")
        q = "绝缘子出现裂纹应该怎么处理"
        r = client.post("/api/ask", json={"question": q, "stream": False}, timeout=300)
        check(r.status_code == 200, "非流式 200", r.text[:300])
        if r.status_code == 200:
            a = r.json()
            for key in ("question", "answer", "stop_reason", "steps", "usage", "warnings", "elapsed_ms"):
                check(key in a, f"有 {key!r}")
            check(
                a.get("stop_reason") in {"final_answer", "max_iterations", "unparsed_output"},
                "stop_reason 是三个合法值之一",
                str(a.get("stop_reason")),
            )
            print(f"     stop_reason={a.get('stop_reason')} steps={len(a.get('steps') or [])} "
                  f"usage={a.get('usage')} {a.get('elapsed_ms')}ms")

        section("12. SSE:事件序列")
        events: list[dict] = []
        with client.stream(
            "POST", "/api/ask", json={"question": q, "stream": True}, timeout=300
        ) as resp:
            check(resp.status_code == 200, "流式 200", str(resp.status_code))
            check(
                resp.headers.get("content-type", "").startswith("text/event-stream"),
                "Content-Type 是 text/event-stream",
                resp.headers.get("content-type", ""),
            )
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                events.append(json.loads(line[len("data: "):]))

        types = [e.get("type") for e in events]
        check(bool(events), "收到了事件", str(types[:8]))
        check(types[0] == "start", "第一条是 start", str(types[:3]))
        check(types[-1] == "done", "最后一条是 done", str(types[-3:]))
        if "action" in types and "done" in types:
            check(types.index("action") < types.index("done"), "action 在 done 之前")
        acts = [i for i, t in enumerate(types) if t == "action"]
        steps = [i for i, t in enumerate(types) if t == "step"]
        if acts and steps:
            check(acts[0] < steps[0], "第一个 action 在第一个 step 之前(**防卡死感**)")
        done = next((e for e in events if e.get("type") == "done"), None)
        if done:
            res = done.get("result") or {}
            check("answer" in res, "done.result 和非流式响应体同构")
            check("steps" in res and "usage" in res, "done.result 带 steps/usage")
            print(f"     事件序列:{types}")

    # ----------------------------------------------------------------- #
    print("\n" + "=" * 62)
    if FAILED:
        print(f"❌ {len(FAILED)} 项未通过:")
        for f in FAILED:
            print(f"   - {f}")
        return 1
    print("全部通过 ✅")
    if not with_ask:
        print("(问答/SSE 两节没跑 —— 加 --ask 会真花 LLM 调用)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
