import { useState } from 'react'
import * as api from '../api.js'
import { Banner, Card, Field, Pill, RawBox } from './Bits.jsx'
import HitCard from './HitCard.jsx'
import { ms } from '../lib/format.js'

export default function SearchTab({ backend, onTrace }) {
  const [q, setQ] = useState('')
  const [topK, setTopK] = useState('')
  const [superseded, setSuperseded] = useState(false)
  const [explain, setExplain] = useState(false)
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState(null)
  const [err, setErr] = useState('')

  async function run(e) {
    e?.preventDefault()
    const query = q.trim()
    if (!query) {
      setErr('查询不能为空。')
      return
    }
    setBusy(true)
    setErr('')
    try {
      const body = { query, include_superseded: superseded, explain }
      // top_k 不传时**不带这个键**:后端 `extra="forbid"` 会拒掉未知字段,
      // 但传 null 是合法的"用服务端默认"。这里选择"不带",让契约 §0.2
      // 的回显机制告诉我们服务端实际用了什么。
      if (topK.trim()) body.top_k = Number(topK)
      if (backend) body.backend = backend
      setRes(await api.search(body))
    } catch (e2) {
      setErr(e2.message)
      setRes(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel-inner">
      <div>
        <h1 className="page-title">检索</h1>
        <p className="page-note">
          只做检索,不调模型。想看「模型实际拿到的是什么」就展开下面的{' '}
          <em>喂给模型的上下文</em> —— 那是检索页和问答页<strong>唯一</strong>能对上口径的地方。
        </p>
      </div>

      <Card title="查询">
        <form className="stack" onSubmit={run}>
          <Field label="查询文本">
            <textarea
              className="input"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="例如:绝缘子污秽等级的判定依据是什么"
              onKeyDown={(e) => {
                // Ctrl/Cmd+Enter 提交。裸 Enter 留给换行 —— 查询经常是
                // 粘进来的一整句,回车直接提交会把粘贴的多行截断。
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run(e)
              }}
            />
          </Field>

          <div className="row">
            <Field label="返回条数上限">
              <input
                className="input"
                style={{ width: 120 }}
                inputMode="numeric"
                value={topK}
                onChange={(e) => setTopK(e.target.value.replace(/[^\d]/g, ''))}
                placeholder="默认"
              />
            </Field>
            <Field label="请求后端">
              <span className="mono dim" style={{ padding: '7px 0' }}>
                {backend || '(跟服务端当前)'}
              </span>
            </Field>
            <span className="nameplate-spacer" />
            <button className="btn primary" disabled={busy}>
              {busy ? <span className="spinner" /> : null}
              {busy ? '检索中…' : '检索'}
            </button>
          </div>

          <div className="inline">
            <label className="check">
              <input
                type="checkbox"
                checked={explain}
                onChange={(e) => setExplain(e.target.checked)}
              />
              解释召回路径(多两次 Qdrant 查询,较慢)
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={superseded}
                onChange={(e) => setSuperseded(e.target.checked)}
              />
              放行已失效版本
            </label>
          </div>
        </form>
      </Card>

      {err ? <Banner tone="bad">{err}</Banner> : null}

      {res ? (
        <>
          <div className="inline">
            <Pill tone="accent">{res.backend}</Pill>
            <Pill>{res.hits.length} 条</Pill>
            <Pill>{ms(res.elapsed_ms)}</Pill>
            {/* 回显服务端**实际生效**的版本口径:请求没传时前端无从知道,
                而"我以为没放行失效版本"这种误会只能靠它发现。 */}
            <Pill tone={res.include_superseded ? 'warn' : ''}>
              {res.include_superseded ? '含已失效版本' : '只召回现行版'}
            </Pill>
          </div>

          {res.hits.length === 0 ? (
            <Card>
              <div className="empty">
                没有命中。库是空的,或者这个查询在这个后端下确实没有结果。
              </div>
            </Card>
          ) : (
            <ol className="hits">
              {res.hits.map((h, i) => (
                <HitCard key={`${h.doc_id}-${h.chunk_index}-${i}`} hit={h} index={i} onTrace={onTrace} />
              ))}
            </ol>
          )}

          {res.llm_text ? (
            <details className="raw-box">
              <summary>喂给模型的上下文(与问答页同源)</summary>
              <pre className="raw">{res.llm_text}</pre>
            </details>
          ) : null}

          <RawBox label="原始响应 JSON">{res}</RawBox>
        </>
      ) : null}
    </div>
  )
}
