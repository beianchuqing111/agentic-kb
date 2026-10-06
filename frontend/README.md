# frontend —— React 界面

`webui.py`(Gradio)的替代界面。**Gradio 那份不删**,留在 `webui.py` 当调试面板:
它少一层构建、少一个进程,排查"到底是后端不对还是前端不对"时更好用。

这一层**不含任何业务逻辑**。分块、重排、阈值、权限判定全在后端,前端只负责
把 `/api/*` 的响应摆出来,以及把"这一步发生了什么"说清楚。唯一的例外是引用的
名字解析(见下),那是因为 `/api/ask` 的响应体里不带 Hit 对象,只有名字。

## 跑起来

两个进程,先起后端:

```bash
# 1) 后端。在仓库根目录,用 anaconda py310
python scripts/api_server.py            # 127.0.0.1:8000

# 2) 前端
cd frontend
npm install                             # 第一次
npm run dev                             # http://localhost:5173
```

浏览器开 **`http://localhost:5173`**,不是 `127.0.0.1:5173`。

> **`localhost` 不能换成 `127.0.0.1`。** Vite 5 绑的是 `localhost`,在 Windows 上
> 这个名字先解析到 IPv6 的 `::1` —— 实测当前进程的监听地址就是 `::1:5173`,
> 于是 `http://127.0.0.1:5173` 直接**连接被拒**(curl exit 7),而
> `http://localhost:5173` 正常。`npm run preview` 的 4173 一样。端口本身没问题,
> 是地址族的问题,所以别去查防火墙。

**调前端时不要给 `scripts/api_server.py` 加 `--reload`。** uvicorn 会在子进程里
跑应用,每次改动都重建一次进程级后端单例 —— 也就是重新加载 bge-m3,十几秒。
改前端只要 Vite 热更,后端那个进程一直活着就行(`scripts/api_server.py:6-9`)。
前端通过 Vite 代理访问后端,代理配置在 `vite.config.js`,换后端端口改 `VITE_API_TARGET`。

`strictPort: true`:5173 被占时**直接失败**而不是悄悄换一个端口。换了你粘到浏览器
里的还是旧地址,会对着上一次残留的进程调试。

**别和 `webui.py` 同时对着一个库做写操作。** 后端单例是**每进程**一份,跨进程
没有锁;两边同时导入会让 `content_hash` 的判断互相打架,表现是"我明明重导了,
界面却说没变化"。只读没问题。

## 端口

| 端口 | 谁 | 备注 |
|---|---|---|
| 8000 | `scripts/api_server.py` | FastAPI。默认只监听 `127.0.0.1`,**无认证** |
| 5173 | `npm run dev` | 开发用。绑 `::1`,只能用 `localhost` 访问 |
| 4173 | `npm run preview` | 看构建产物用,同样只能 `localhost` |
| 6343 / 6344 | Qdrant | **不是 6333 —— 那个是 imgsearch 的** |
| 7687 / 7474 | Neo4j | |

## 构建

```bash
npm run build      # → dist/
npm run preview    # 在 4173 上伺服 dist/,并把 /api 代理到 8000
```

`preview` 会沿用 `server.proxy`(Vite 的 `preview.proxy` 默认就是它),所以构建
产物也能直接连后端,不用另外配。

**`api/app.py` 不托管 `dist/`** —— 它没有 `StaticFiles` 挂载,`/api/*` 之外全是
404。要对外发布得另外用一个静态服务器伺服 `dist/`,或者在 `app.py` 里加挂载;
现在 `preview` 是唯一验证构建产物的方式。这是**没做**,不是"配好了没写文档"。

## 队列:为什么前端要自己排队

`src/api.js` 里有一个串行队列。这不是为了防重复点击,是因为 `api/CONTRACT.md`
§0.1 定死了**同一时刻只允许一个碰后端的请求** —— 后端是进程级单例,bge-m3 挂在
模块变量上,Gradio 那边同样是 `concurrency=1`。

不排队的后果不是报错,是**看起来正常**:服务端那把锁会把第二个请求挡住,两个
转圈同时转但后一个一步没动。契约里量过这个场景(导入途中发检索要等 9.5s)。
用户看不到"在排队"就只会重试,队列更长。所以队列深度必须可见 —— `useBusy()`
把它喂给顶栏的「N 在跑 · M 等待中」。

**有些路由不走队列**,判据是 `api/app.py` 里它有没有 `with hold_backend(...)`,
不是随手定的:

| 不走队列 | 为什么 |
|---|---|
| `/api/health`、`/api/backends` | 导入进行中**唯一还能立刻回答**的端点。排进去等于"忙的时候连忙不忙都问不出来" |
| `/api/tools` | 只构造 Tool 对象读元数据,不碰单例 |
| `/api/audit`、`/api/eval` | 只读文件。插入队列的话,导入跑着时就查不了上次的数字和操作记录 |

往 `api.js` 加新路由时,先去看 `app.py` 里那条路由有没有 `hold_backend`,别默认
排队也别默认不排队。

## 目录

```
src/
  App.jsx               外壳:页签、主题、健康轮询、引用抽屉、提示条
  api.js                接口封装 + 串行队列(见上)
  main.jsx              挂载点
  styles.css            全部样式(单文件,主题走 CSS 变量)
  components/
    Nameplate.jsx       顶栏:后端选择、排队指示、健康与配置 tooltip
    SearchTab.jsx       检索:稠密/稀疏/重排/图的各路得分并排
    AskTab.jsx          问答:SSE 流式,思考/工具调用/观察值逐步出
    IngestTab.jsx       导入:路径与上传两条路,结果计数 + 错误明细
    LibraryTab.jsx      库状态:文档列表、块浏览、统计
    EvalTab.jsx         评测:基线列表与对比
    AuditTab.jsx        审计:写工具的操作记录
    CitationDrawer.jsx  引用抽屉:点开引用看原文块
    HitCard.jsx         单条检索结果的卡片
    Bits.jsx            Card / Field / KV / Pill / Banner / RawBox
  lib/
    citations.js        引用标签解析 + 名字→文档解析
    format.js           数字、耗时、`null` 的处理
```

## 几个容易搞错的

- **`null` 和 `0.0` 不是一回事。** 重排分 `null` 是"重排没开",`0.0` 是"重排开了
  并给这一条打了 0 分"。界面上前者显示 `—`(`format.js`),**绝不能显示成 0** ——
  这两种情况该做的处置相反。
- **引用里的「块 N」是 `chunk_index` 原值,不加 1。** 后端 `citation` 就是这个
  拼法(`retrieve/hybrid.py`),显示层照抄,别"顺手修正"成 1-based,否则点过去
  和实际召回的不是同一个块。
- **引用标签是提示词约定,不是协议。** `agent/react.py` 的系统提示词要求
  `[文档名 (块 3)]`,但模型不保证遵守。所以 `citations.js` 的解析是宽容的
  (全角/半角/`【】` 都收),认不出来就原样当文字——硬按一种格式切会把正常
  段落切碎,比少一个可点链接糟得多。
- **引用只能靠名字对回 doc_id。** `/api/ask` 的响应体里**不带 Hit 对象**,
  所以 `App.jsx` 拿文档列表在客户端对。匹配顺序是从精确到宽松,且必须
  先按 `source` 全路径比 —— `title` 可能重复(同一份规程的多个版本),反过来
  先模糊匹配会把 v1 的引用指到 v2 上,而那正好是版本化功能要防的事。
  **对不上就明说,不猜。**
- **`askStream` 用 `fetch` + `ReadableStream`,不是 `EventSource`。** 后者只能
  GET,而这里要 POST 一个 JSON 请求体(问题 + 两个写权限闸门)。用 EventSource
  就得把 `allow_write` 塞进 query string —— 写权限不该长成 URL 上的开关。
- **写权限是两道闸。** 请求体里的 `allow_write` **且** 服务端 master 开关
  都开才可能写;`/api/tools` 报出来的名单决定界面渲染哪些复选框。判据永远在
  后端,前端只是把开关摆出来。

## 和 Gradio 的能力对齐

`api/CONTRACT.md` 的附录有一张**逐控件比过**的对照表(Gradio 是基准)。加新功能
时对着它改,别只改一边 —— 那张表就是为了防"Gradio 有、React 没有"这种静默缺失。
