import { Pill } from './Bits.jsx'
import { useBusy } from '../api.js'

/**
 * 顶栏。
 *
 * 「请求后端」这个下拉是**每次请求**的选择,不是"切换服务端设置"——
 * 这点必须说清楚:服务端只有一个进程级单例,`current_backend()` 会跟着
 * 最后一个带 `backend` 的请求变。把它画成一个"设置项",用户会以为自己
 * 改了全局,而实际上只影响他自己发出去的那一条。
 */
export default function Nameplate({ backend, onBackend, backendList, current, health }) {
  const busy = useBusy()
  const h = health?.data

  const tone = health?.error ? 'bad' : h?.ok ? 'ok' : h ? 'warn' : ''
  const label = health?.error
    ? '服务不可用'
    : h
      ? h.ok
        ? '正常'
        : '后端报错'
      : '探测中…'

  return (
    <header className="nameplate">
      <div className="brand">
        <span className="brand-mark">法规知识库智能体</span>
        <span className="brand-sub">agentic-kb</span>
      </div>

      <div className="nameplate-spacer" />

      <div className="nameplate-group">
        <span className="label" style={{ marginRight: 6 }}>
          请求后端
        </span>
        <select
          className="select"
          style={{ width: 168 }}
          value={backend || ''}
          onChange={(e) => onBackend(e.target.value)}
          title="只影响你接下来发的请求。服务端是单例,它的当前后端会跟着最后一个带 backend 的请求变。"
        >
          <option value="">跟服务端当前({current || '?'})</option>
          {(backendList || []).map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
      </div>

      {/* 排队指示。**必须在顶栏**,因为它是"为什么点了没反应"的唯一答案。 */}
      {busy.running + busy.waiting > 0 ? (
        <Pill tone="accent" pulse title="碰后端的请求是串行的:同一时刻只跑一个">
          {busy.running} 在跑
          {busy.waiting ? ` · ${busy.waiting} 等待中` : ''}
        </Pill>
      ) : null}

      <Pill tone={tone} dot title={h?.error || ''}>
        {label}
      </Pill>

      {h ? (
        <Pill
          title={[
            `LLM: ${h.config?.llm_model || '—'}${h.config?.llm_configured ? '' : '(未配置)'}`,
            // 阈值要跟着开关一起报。「重排开着但一条都没留下」和
            // 「重排关着」在只看开关时长得一模一样,而这两件事的处置相反。
            `重排: ${h.config?.rerank_enabled ? '开' : '关'}` +
              (h.config?.rerank_enabled
                ? ` (阈值 ${h.config?.rerank_min_score},至少留 ${h.config?.rerank_min_keep} 条)`
                : ''),
            `定位语: ${h.config?.contextual_enabled ? '开' : '关'}`,
            `含失效版本: ${h.config?.include_superseded ? '是' : '否'}`,
            `写权限: ${h.config?.allow_write ? '开' : '关'}`,
            `Tavily: ${h.config?.tavily_configured ? '已配' : '未配'}`,
          ].join(' · ')}
        >
          {h.config?.llm_model || '未配模型'}
        </Pill>
      ) : null}
    </header>
  )
}
