"""启动 HTTP API。

    python scripts/api_server.py                # 127.0.0.1:8000
    python scripts/api_server.py --port 8100 --reload

`--reload` 只在改后端代码时用;开着它的时候 uvicorn 会**另起一个进程**
跑应用,而进程级的后端单例(以及挂在它上面的模型权重)会在每次改动后
重建 —— 加载 bge-m3 一次就是十几秒,改一行前端无关的代码也照样重来。
调试前端请用 vite 的 proxy,不要用 `--reload`。

**不要**把它和 `webui.py` 同时对着一个库跑写操作。读没问题,但同时导入
会让两边的 `content_hash` 判断互相打架 —— 表现为"我明明重导了,界面
却说没变化"。后端单例是**每进程**一份,跨进程没有锁。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 和 webui.py / scripts/eval_run.py 同一条:Windows 控制台是 GBK,
# 中文日志会直接抛 UnicodeEncodeError 打断启动。errors="replace" 兜底,
# 宁可显示成问号也不要因为一行日志起不来。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # 已被重定向/不支持的流
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from config import UPLOAD_DIR, get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="agentic-kb HTTP API")
    ap.add_argument("--host", default="127.0.0.1", help="默认只监听本机")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="改代码自动重启(别和前端调试一起用)")
    args = ap.parse_args()

    s = get_settings()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  agentic-kb API")
    print("=" * 62)
    print(f"  地址     : http://{args.host}:{args.port}   (文档 /docs)")
    print(f"  后端     : {s.backend.value}    (请求体可用 backend 字段切换)")
    print(f"  集合     : {s.qdrant.collection}")
    print(f"  上传目录 : {UPLOAD_DIR}")
    print(f"  LLM      : {'已配置 ' + s.llm.model if s.llm.configured else '**未配置**(问答和建图不可用)'}")
    print(f"  写工具   : {'开' if s.agent.allow_write else '关(AGENT_ALLOW_WRITE=true 可开)'}")
    print("-" * 62)
    # 默认 `127.0.0.1` 而不是 `0.0.0.0`:这个接口没有认证,写工具一旦开着,
    # 网段里任何人都能让它落盘。要对外提供就自己加反代和鉴权。
    print("  ⚠️  无认证。默认只监听本机,别直接暴露到网络上。")
    print("=" * 62)

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # 访问日志按请求打一行,正好是排查"谁把库导坏了"要的线索,留着。
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
