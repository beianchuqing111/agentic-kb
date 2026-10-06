/** 共用的小零件。都是无状态的,只做显示。 */

export function Pill({ tone = '', children, dot = false, pulse = false, title }) {
  return (
    <span className={`pill ${tone}${pulse ? ' busy' : ''}`} title={title}>
      {dot ? <span className="dot" /> : null}
      {children}
    </span>
  )
}

export function Card({ title, actions, children, tight = false }) {
  return (
    <section className="card">
      {title ? (
        <header className="card-head">
          <span className="card-title">{title}</span>
          <span className="nameplate-spacer" />
          {actions}
        </header>
      ) : null}
      <div className={`card-body${tight ? ' tight' : ''}`}>{children}</div>
    </section>
  )
}

export function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      {children}
      {hint ? <span className="muted" style={{ fontSize: 12 }}>{hint}</span> : null}
    </label>
  )
}

/** 键值网格。`items` 是 `[key, value]`,值为 null 时显示 `—`。 */
export function KV({ items }) {
  return (
    <div className="kv">
      {items.map(([k, v]) => (
        <div key={k}>
          <span className="k">{k}</span>
          <span className="v">{v === null || v === undefined || v === '' ? '—' : v}</span>
        </div>
      ))}
    </div>
  )
}

export function Banner({ tone = '', children, spinner = false }) {
  return (
    <div className={`banner ${tone}`}>
      {spinner ? <span className="spinner" /> : null}
      <span className="banner-text">{children}</span>
    </div>
  )
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>
}

/** 分数条。`value` 为 null 时**不画 0 长度的条** —— 那看起来像"得了 0 分"。 */
export function Meter({ label, value, digits = 4, max = 1 }) {
  const has = value !== null && value !== undefined && !Number.isNaN(Number(value))
  const w = has ? Math.max(0, Math.min(1, Number(value) / max)) * 100 : 0
  return (
    <span className="meter" title={`${label}: ${has ? value : '没这一项'}`}>
      <span>{label}</span>
      <span className="track">
        {has ? <span className="fill" style={{ width: `${w}%` }} /> : null}
      </span>
      <span className="val">{has ? Number(value).toFixed(digits) : '—'}</span>
    </span>
  )
}

export function RawBox({ label, children }) {
  return (
    <details className="raw-box">
      <summary>{label}</summary>
      <pre className="raw">{typeof children === 'string' ? children : JSON.stringify(children, null, 2)}</pre>
    </details>
  )
}
