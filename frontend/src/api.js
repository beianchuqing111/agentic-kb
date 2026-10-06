/**
 * 后端接口封装 + **串行队列**。
 *
 * ## 为什么要有队列
 *
 * 后端是**进程级单例** —— bge-m3 的权重挂在模块变量上,不是每个请求新建
 * 一份(新建一次要十几秒)。所以 `api/CONTRACT.md` §0.1 定死了:同一时刻
 * 只允许一个"碰后端"的请求在跑。Gradio 那边是这个口径(concurrency=1),
 * 这里必须是同一个。
 *
 * 不做会**真的坏掉**,而且坏的方式很隐蔽:两个请求并发时服务端那把锁会把
 * 第二个挡住,所以结果**看起来**是对的 —— 但界面上两个转圈同时在转,
 * 用户以为都在跑,实际上后一个一步都还没动。导入途中发问答就是这个场景,
 * 而契约里 §0.1 专门量过它(搜索要排队 9.5s)。用户看不到"在排队",
 * 只会重试,然后队列更长。
 *
 * 所以排队这件事**必须在前端可见**:`subscribeBusy` 把队列深度暴露给
 * 顶栏,让"等待中"是一个看得见的状态,而不是一个转不完的圈。
 *
 * ## 为什么有的请求不走队列
 *
 * 队列保护的是那把锁,不是"所有请求"。不走锁的路由插进来只会让
 * 健康检查/审计/评测一起变慢,而它们本来就不受影响。每一条都由
 * `api/app.py` 里有没有 `with hold_backend(...)` 决定,不是随手定的。
 */

import { useSyncExternalStore } from 'react'

// --------------------------------------------------------------------------- //
// 队列
// --------------------------------------------------------------------------- //

let _tail = Promise.resolve()
let _state = { running: 0, waiting: 0 }
const _watchers = new Set()

function _set(next) {
  _state = next
  for (const w of _watchers) w(_state)
}

export function subscribeBusy(fn) {
  _watchers.add(fn)
  return () => _watchers.delete(fn)
}

function _snapshot() {
  return _state
}

/** 顶栏用:当前有几个在跑、几个在排队。 */
export function useBusy() {
  return useSyncExternalStore(subscribeBusy, _snapshot, _snapshot)
}

/**
 * 把一个作业排进队列。**前一个失败也照样接着跑** —— 队列后面的人不该
 * 因为前面那次请求 500 就一起死掉。
 */
function enqueue(job) {
  _set({ ..._state, waiting: _state.waiting + 1 })
  const start = () => {
    _set({ running: _state.running + 1, waiting: _state.waiting - 1 })
    return Promise.resolve()
      .then(job)
      .finally(() => {
        _set({ ..._state, running: _state.running - 1 })
      })
  }
  // 两个参数都传 start:成功和失败都继续。单传一个的话,`tail` 一旦
  // rejected,后面所有请求会永远挂着 —— 表现是"页面突然什么都不响应了"。
  const run = _tail.then(start, start)
  _tail = run.catch(() => {})
  return run
}

// --------------------------------------------------------------------------- //
// 底层请求
// --------------------------------------------------------------------------- //

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * 把响应体里的错误信息挖出来。
 *
 * FastAPI 的错误体是 `{"detail": ...}`,而 `detail` **可能是字符串,也可能是
 * 数组**(pydantic 校验失败时是数组)。这不是猜的:`app.py` 的
 * `_on_validation_error` 把它压成了字符串,但 404 之外的路径(pydantic 直接
 * 拦截的)仍会给出数组。两种都得能显示,否则最该看清原因的那条错误反而
 * 显示成 `[object Object]`。
 */
async function _error(res) {
  let detail = ''
  try {
    const body = await res.json()
    detail = body?.detail ?? ''
    if (Array.isArray(detail)) {
      detail = detail
        .map((d) => `${(d.loc || []).join('.')}: ${d.msg || ''}`)
        .join('; ')
    } else if (detail && typeof detail === 'object') {
      detail = JSON.stringify(detail)
    }
  } catch {
    /* 不是 JSON(网关错误页之类),下面用状态码兜底 */
  }
  return new ApiError(detail || `HTTP ${res.status}`, res.status)
}

async function _fetch(path, { method = 'GET', body, form, signal } = {}) {
  const init = { method, signal, headers: {} }
  if (form) {
    // 不设 Content-Type:multipart 的 boundary 得由浏览器自己写,
    // 手写一个会让后端解析不出任何文件。
    init.body = form
  } else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  }
  const res = await fetch(path, init)
  if (!res.ok) throw await _error(res)
  if (res.status === 204) return null
  return res.json()
}

/**
 * 发一个请求。
 *
 * `queued: false` 只给那些**不碰后端单例**的路由用 —— 判据是
 * `app.py` 里它有没有 `with hold_backend(...)`。健康检查尤其重要:
 * 它是导入进行中唯一还能立刻回答的端点(契约 §health),把它塞进队列
 * 会让"服务忙不忙"这个指示自己先卡住。
 */
function req(path, opts = {}) {
  const { queued = true, ...rest } = opts
  const job = () => _fetch(path, rest)
  return queued ? enqueue(job) : job()
}

// --------------------------------------------------------------------------- //
// 路由
// --------------------------------------------------------------------------- //

// 运维 —— 全部不走队列(见上)。
export const health = (signal) => req('/api/health', { queued: false, signal })
export const backends = (signal) => req('/api/backends', { queued: false, signal })

// 检索 —— 走锁,排队。
export const search = (payload, signal) =>
  req('/api/search', { method: 'POST', body: payload, signal })

// 导入 —— 走锁,而且是最占锁的一个。表单字段必须是字符串。
/**
 * 非流式问答。返回体和流式 `done.result` **同构**(契约 §0.2)——
 * 依赖这一点,前端两条路用同一段渲染代码。
 */
export const ask = (payload, signal) =>
  req('/api/ask', { method: 'POST', body: { ...payload, stream: false }, signal })

export const ingestPath = (payload, signal) =>
  req('/api/ingest', { method: 'POST', body: payload, signal })

export const ingestUpload = (files, { recursive, force, backend }, signal) => {
  const form = new FormData()
  for (const f of files) form.append('files', f, f.name)
  form.append('recursive', recursive ? 'true' : 'false')
  form.append('force', force ? 'true' : 'false')
  if (backend) form.append('backend', backend)
  return req('/api/ingest/upload', { method: 'POST', form, signal })
}

// 库状态 —— 两条都走锁,排队。
export const docs = (q = '', signal) =>
  req(`/api/docs${q ? `?q=${encodeURIComponent(q)}` : ''}`, { signal })
export const docChunks = (docId, signal) =>
  req(`/api/docs/${encodeURIComponent(docId)}`, { signal })
export const stats = (signal) => req('/api/stats', { signal })

// 工具清单 —— 只构造 Tool 对象读元数据,不碰后端单例,不走队列。
export const tools = (signal) => req('/api/tools', { queued: false, signal })

// 审计 / 评测 —— 只读文件,不碰单例,不走队列。
// 这两条插进队列的话,导入途中就查不了评测和历史操作了,而它们恰恰是
// "导入跑着的时候我想核对一下上次的数字"最需要的。
export const audit = (n = 50, signal) =>
  req(`/api/audit?n=${n}`, { queued: false, signal })
export const evalBaselines = (signal) => req('/api/eval', { queued: false, signal })

/**
 * 流式问答。**整条流占着队列**,直到 `done`/`error` 或者连接断开。
 *
 * 用 `fetch` + `ReadableStream` 而不是 `EventSource`:后者只能 GET,
 * 而这里要 POST 一个 JSON 请求体(问题 + 两个写权限闸门)。
 * 拿 EventSource 就得把参数塞进 query string,那等于把 `allow_write`
 * 变成一个 URL 上的开关 —— 写权限不该长成那样。
 */
export function askStream(payload, onEvent, { signal } = {}) {
  return enqueue(async () => {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, stream: true }),
      signal,
    })
    if (!res.ok) throw await _error(res)
    if (!res.body) throw new ApiError('响应没有流式主体(浏览器或代理不支持)', 0)

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''

    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      // 统一换行:`sse-starlette` 之类以后换成 \r\n 时这里不会静默失灵。
      buf = buf.replace(/\r\n/g, '\n')
      let cut
      while ((cut = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, cut)
        buf = buf.slice(cut + 2)
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue
          const text = line.slice(5).trim()
          if (!text) continue
          let event
          try {
            event = JSON.parse(text)
          } catch {
            continue // 半个帧,丢掉比让整轮问答崩掉强
          }
          onEvent(event)
        }
      }
    }
  })
}
