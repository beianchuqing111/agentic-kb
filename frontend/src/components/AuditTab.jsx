import { useEffect, useState } from 'react'
import * as api from '../api.js'
import { Banner, Card, Empty, Field, Pill, RawBox } from './Bits.jsx'
import { ts } from '../lib/format.js'

/**
 * 按 `call_id` 把审计记录配对。
 *
 * 契约 §audit 里点名了这件事**必须**在前端做:并发的写操作会在日志里交错,
 * 按出现顺序去配"上一条是意图、下一条是结果"会张冠李戴 —— 显示出来的
 * 就是"某个危险操作成功了",而它其实被拒了。用 call_id 分组就不会。
 *
 * 四相:denied(授权阶段就拒了,**没有副作用**)、intent(已批准,即将执行)、
 * result(执行完了)。denied 单独成组,因为它压根没有配对的 intent。
 */
function group(records) {
  const by = new Map()
  for (const r of records || []) {
    const id = r.call_id || '(无 call_id)'
    if (!by.has(id)) by.set(id, [])
    by.get(id).push(r)
  }
  const rows = []
  for (const [id, recs] of by) {
    recs.sort((a, b) => String(a.ts).localeCompare(String(b.ts)))
    const denied = recs.find((r) => r.phase === 'denied')
    const intent = recs.find((r) => r.phase === 'intent')
    const result = recs.find((r) => r.phase === 'result')
    rows.push({
      id,
      tool: (denied || intent || result || {}).tool || '',
      arg: (denied || intent || result || {}).arg || '',
      actor: (denied || intent || result || {}).actor || '',
      ts: (denied || intent || result || {}).ts,
      outcome: denied ? 'denied' : result ? (result.ok ? 'ok' : 'failed') : 'pending',
      detail: denied?.detail || result?.detail || '',
      raw: recs,
    })
  }
  // 新的在前。ts 是带时区的 ISO 串,字典序即时间序。
  rows.sort((a, b) => String(b.ts).localeCompare(String(a.ts)))
  return rows
}

const TONE = { denied: 'warn', ok: 'ok', failed: 'bad', pending: '' }
const LABEL = { denied: '被拒', ok: '已执行', failed: '执行失败', pending: '只有意图,没等到结果' }

export default function AuditTab({ versionTick }) {
  const [n, setN] = useState('100')
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [open, setOpen] = useState(null)

  async function load(count = n) {
    setBusy(true)
    setErr('')
    try {
      setData(await api.audit(Number(count) || 100))
    } catch (e) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    load('100')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [versionTick])

  const rows = data ? group(data.records) : []

  return (
    <div className="panel-inner">
      <div>
        <h1 className="page-title">审计</h1>
        <p className="page-note">
          每一次写工具的调用都会留痕,<strong>包括被拒的</strong>。按 <span className="mono">call_id</span> 配对 ——
          并发的写会交错,按顺序配对会张冠李戴。
        </p>
      </div>

      <Card
        title="读取"
        actions={
          <button className="btn small" onClick={() => load()} disabled={busy}>
            刷新
          </button>
        }
      >
        <form
          className="row"
          onSubmit={(e) => {
            e.preventDefault()
            load()
          }}
        >
          <div style={{ width: 160 }}>
            <Field label="末尾多少条记录" hint="上限 1000">
              <input
                className="input"
                inputMode="numeric"
                value={n}
                onChange={(e) => setN(e.target.value.replace(/[^\d]/g, ''))}
              />
            </Field>
          </div>
          <button className="btn" disabled={busy}>
            {busy ? <span className="spinner" /> : null}
            读取
          </button>
        </form>
      </Card>

      {err ? <Banner tone="bad">{err}</Banner> : null}

      {data && data.records.length === 0 ? (
        <Card>
          <Empty>
            审计日志是空的 —— 还没有发生过任何写工具调用。<strong>这不是错误</strong>:从没写过库的时候
            本来就是这样。
          </Empty>
        </Card>
      ) : null}

      {rows.length ? (
        <Card
          title="调用记录"
          actions={
            <span className="inline">
              <Pill>{rows.length} 次调用</Pill>
              <Pill>{data.records.length} 条原始记录</Pill>
            </span>
          }
          tight
        >
          <div className="scroll-x">
            <table className="grid">
              <thead>
                <tr>
                  <th>时间</th>
                  <th>结果</th>
                  <th>工具</th>
                  <th>来源</th>
                  <th>参数</th>
                  <th>说明</th>
                  <th>call_id</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id}>
                    <td className="mono muted">{ts(r.ts)}</td>
                    <td>
                      <Pill tone={TONE[r.outcome]}>{LABEL[r.outcome]}</Pill>
                    </td>
                    <td className="mono">{r.tool}</td>
                    <td className="mono muted">{r.actor}</td>
                    <td
                      className="mono"
                      style={{ whiteSpace: 'normal', maxWidth: '30ch', overflowWrap: 'anywhere' }}
                    >
                      {r.arg || <span className="muted">(空)</span>}
                    </td>
                    <td style={{ whiteSpace: 'normal', maxWidth: '36ch' }}>{r.detail}</td>
                    <td className="mono muted">{r.id}</td>
                    <td>
                      <button
                        className="btn small"
                        onClick={() => setOpen(open === r.id ? null : r.id)}
                      >
                        {open === r.id ? '收起' : '原始'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      {open ? (
        <Card title={`原始记录 ${open}`}>
          <pre className="raw">
            {JSON.stringify(rows.find((r) => r.id === open)?.raw || [], null, 2)}
          </pre>
        </Card>
      ) : null}

      {data ? <RawBox label="原始响应 JSON">{data}</RawBox> : null}
    </div>
  )
}
