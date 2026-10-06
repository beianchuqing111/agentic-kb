/** 显示层的格式化。都写成纯函数,方便在组件里直接调。 */

/** 毫秒 → 人话。1.2s / 340ms / 1m12s。 */
export function ms(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const n = Number(v)
  if (n < 1000) return `${Math.round(n)}ms`
  if (n < 60_000) return `${(n / 1000).toFixed(2)}s`
  const m = Math.floor(n / 60_000)
  const s = Math.round((n % 60_000) / 1000)
  return `${m}m${s}s`
}

/**
 * 分数。**`null` 显示成 `—`,不是 `0`。**
 *
 * 这是 `api/serialize.py` 模块头第 3 条在前端的落点:`rerank_score=null`
 * 是"没开重排",`0.0` 是"重排给了零分"。两者显示成一样的话,用户没法
 * 从界面上看出"这个后端根本没重排"和"重排认为毫不相关"的区别 ——
 * 而这两件事的处置完全相反。
 */
export function score(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const n = Number(v)
  return n.toFixed(digits)
}

export function pct(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${(Number(v) * 100).toFixed(digits)}%`
}

/** 整数千分位。 */
export function int(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return Number(v).toLocaleString('zh-CN')
}

/** 字节 → KB/MB。 */
export function bytes(v) {
  if (!v) return '—'
  const n = Number(v)
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

/**
 * 时间戳 → 本地时间。
 *
 * 服务端给的是 `time.time()` 的秒(浮点),不是毫秒。乘 1000 之前先判一下
 * 量级 —— 判错的话会显示成 1970 年,而且**看起来像个正经日期**,不会引起
 * 怀疑。
 */
export function ts(v) {
  if (v === null || v === undefined || v === '') return '—'
  let n = Number(v)
  if (Number.isNaN(n)) return String(v)
  if (n < 1e11) n *= 1000 // 秒 → 毫秒
  const d = new Date(n)
  if (Number.isNaN(d.getTime())) return String(v)
  const p = (x) => String(x).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(
    d.getHours()
  )}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/**
 * 工具返回的 Observation 是不是一条错误。
 *
 * 判据是 `agent/tools.py` 里**这个项目自己定的**约定:`ToolRegistry.run`
 * 的五处失败返回(工具名对不上、授权被拒、授权环节自己出错、执行抛异常、
 * 工具自身拒了)统统以 `错误:` 开头 —— 半角冒号 `0x3a`,不是全角。这段文本
 * 由工具层拼出来,不是模型写的,所以按前缀判是稳的。
 *
 * 但它**分不出**「被拒」和「执行失败」。那个区分在审计日志里 —— `tools.py`
 * 特意把两者记成不同相(「有人试了不该试的」vs「坏了」),所以这里只说
 * 「这一步返回了错误」,不替审计下结论,也不谎称"被拒"。
 */
export const TOOL_ERROR_PREFIX = '错误:'

export function isToolError(observation) {
  return (
    typeof observation === 'string' &&
    observation.trimStart().startsWith(TOOL_ERROR_PREFIX)
  )
}

/** 状态徽章的分档。空串 = 老数据没写这个字段,按现行版显示。 */
export function statusOf(hit) {
  const s = String(hit?.status || '').toLowerCase()
  if (s === 'superseded') return { key: 'superseded', label: '已失效' }
  if (s === 'current' || s === '') return { key: 'current', label: '现行' }
  return { key: 'other', label: s }
}

/** 截断长文本,给列表用的摘要。 */
export function clip(text, n = 260) {
  const s = String(text || '').replace(/\s+/g, ' ').trim()
  return s.length > n ? `${s.slice(0, n)}…` : s
}
