import { useEffect, useState } from 'react'
import * as api from '../api.js'
import { Banner, Card, Field, KV, Pill, RawBox } from './Bits.jsx'
import { int } from '../lib/format.js'

/**
 * 库状态:有哪些文档、多少块,以及后端自己的统计。
 *
 * 两个数**分开显示**不是啰嗦:`/api/docs` 的 `total_docs` 是库里的文档数,
 * `/api/stats` 的 `documents` 是**后端**自己数的 —— 两者口径不同
 * (契约 §stats 说明过:空库时后端干脆不给这个字段)。把它们合成一个数,
 * 就等于替后端猜了一个它没说的值。
 */
export default function LibraryTab({ versionTick, onTrace }) {
  const [data, setData] = useState(null)
  const [stats, setStats] = useState(null)
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  async function load(keyword = q) {
    setBusy(true)
    setErr('')
    try {
      // 两条都碰后端(都走 hold_backend),排队由 api.js 统一管 ——
      // 这里连发两个,第二个会自动等第一个。
      const [d, s] = await Promise.all([
        api.docs(keyword),
        api.stats().catch((e) => ({ __error: e.message })),
      ])
      setData(d)
      setStats(s)
    } catch (e) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }

  // 导入完成后由 App 递一个 tick 下来,自动刷新 —— 导完还要手点一下
  // "刷新"才知道成没成,是最容易忘的一步。
  useEffect(() => {
    load('')
    setQ('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [versionTick])

  const docs = data?.docs || []
  const filtered = Boolean(q.trim())

  return (
    <div className="panel-inner">
      <div>
        <h1 className="page-title">库状态</h1>
        <p className="page-note">
          库里的文档和块。点任意一篇可以展开它的全文块 —— 和「回溯原文」看到的是同一份数据。
        </p>
      </div>

      <Card title="查询">
        <form
          className="row"
          onSubmit={(e) => {
            e.preventDefault()
            load()
          }}
        >
          <div className="grow">
            <Field label="按文件名 / 标题过滤">
              <input
                className="input"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="留空 = 全部"
              />
            </Field>
          </div>
          <button className="btn" disabled={busy}>
            {busy ? <span className="spinner" /> : null}
            查询
          </button>
        </form>
      </Card>

      {err ? <Banner tone="bad">{err}</Banner> : null}

      {data ? (
        <div className="inline">
          <Pill tone="accent">{int(data.total_docs)} 篇文档</Pill>
          <Pill>{int(data.total_chunks)} 块</Pill>
          {filtered ? <Pill tone="warn">筛选后 {int(data.filtered)} 篇</Pill> : null}
        </div>
      ) : null}

      {stats?.__error ? (
        <Banner tone="warn">后端统计取不到:{stats.__error}</Banner>
      ) : stats ? (
        <Card title="后端统计(原样透传)">
          <div className="stack">
            {/* 空库时后端不给 documents —— 原样显示它真的没有,而不是补一个 0 */}
            {stats.documents === undefined ? (
              <Banner tone="warn">
                后端这次没有给出 <span className="mono">documents</span> 字段。
                这通常意味着库是空的(契约 §stats),但你也不能从这一条断定
                「一篇都没导进去」—— 上面的文档列表才是这个问题的答案。
              </Banner>
            ) : null}
            <KV
              items={[
                ['后端', stats.backend],
                ...Object.entries(stats)
                  .filter(([k]) => k !== 'backend' && k !== '__error')
                  .map(([k, v]) => [k, typeof v === 'object' ? JSON.stringify(v) : v]),
              ]}
            />
            <RawBox label="原始响应 JSON">{stats}</RawBox>
          </div>
        </Card>
      ) : null}

      <Card title={filtered ? '文档(筛选后)' : '文档'} tight>
        {docs.length === 0 ? (
          <div className="empty">
            {filtered ? '没有匹配的文档。' : '库是空的。去「导入」页加几篇。'}
          </div>
        ) : (
          <div className="scroll-x">
            <table className="grid">
              <thead>
                <tr>
                  <th>标题</th>
                  <th>来源</th>
                  <th className="n">块数</th>
                  <th>状态</th>
                  <th>版本</th>
                  <th>doc_id</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.doc_id}>
                    <td style={{ whiteSpace: 'normal', maxWidth: '28ch' }}>
                      {d.title || <span className="muted">(无标题)</span>}
                    </td>
                    <td
                      className="mono muted"
                      style={{ whiteSpace: 'normal', maxWidth: '36ch', overflowWrap: 'anywhere' }}
                    >
                      {d.source}
                    </td>
                    <td className="n">{d.chunks}</td>
                    <td>
                      <Pill tone={d.status === 'superseded' ? 'warn' : 'ok'}>
                        {d.status === 'superseded' ? '已失效' : d.status || '现行'}
                      </Pill>
                      {d.mixed ? <Pill tone="warn">块状态不一致</Pill> : null}
                    </td>
                    <td className="mono">{d.doc_version || '—'}</td>
                    <td className="mono muted">{d.doc_id}</td>
                    <td>
                      <button
                        className="btn small"
                        onClick={() =>
                          onTrace({ docId: d.doc_id, chunkIndex: -1, label: d.title || d.source })
                        }
                      >
                        看全文
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {data ? <RawBox label="原始响应 JSON">{data}</RawBox> : null}
    </div>
  )
}
