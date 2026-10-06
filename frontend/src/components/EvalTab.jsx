import { Fragment, useEffect, useMemo, useState } from 'react'
import * as api from '../api.js'
import { Banner, Card, Empty, Pill, RawBox } from './Bits.jsx'
import { pct, ts } from '../lib/format.js'

const METRICS = ['hit', 'ndcg', 'rr', 'ndcg_binary', 'precision', 'recall']

function RunTable({ baseline, run }) {
  const ks = Object.keys(run.aggregates || {}).sort((a, b) => Number(a) - Number(b))
  return (
    <div className="scroll-x">
      <table className="grid">
        <thead>
          <tr>
            <th>k</th>
            {/* n 必须和指标并排显示。同一个指标在 n=15 和 n=25 上是两个
                不同的数,藏了 n 就没法比较 —— 而这套基线里两种都有。 */}
            <th className="n">n</th>
            {METRICS.map((m) => (
              <th className="n" key={m}>
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ks.map((k) => {
            const a = run.aggregates[k] || {}
            return (
              <tr key={k} className={Number(k) === Number(run.effective_k) ? 'is-current' : ''}>
                <td className="mono">
                  @{k}
                  {Number(k) === Number(run.effective_k) ? (
                    <span className="muted"> (effective_k)</span>
                  ) : null}
                </td>
                <td className="n">{a.n}</td>
                {METRICS.map((m) => (
                  <td className="n" key={m}>
                    {m === 'hit' ? pct(a[m]) : a[m] === undefined ? '—' : Number(a[m]).toFixed(3)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function ByType({ run }) {
  const ks = Object.keys(run.by_type || {}).sort((a, b) => Number(a) - Number(b))
  if (!ks.length) return null
  const types = [...new Set(ks.flatMap((k) => Object.keys(run.by_type[k] || {})))]
  return (
    <div className="scroll-x">
      <table className="grid">
        <thead>
          <tr>
            <th>题型</th>
            {ks.map((k) => (
              <th className="n" key={k} colSpan={2}>
                命中率@{k} (n)
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {types.map((t) => (
            <tr key={t}>
              <td className="mono">{t}</td>
              {ks.map((k) => {
                const a = run.by_type[k]?.[t]
                return (
                  <Fragment key={k}>
                    <td className="n">{a ? pct(a.hit) : '—'}</td>
                    <td className="n muted">{a ? a.n : '—'}</td>
                  </Fragment>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** 两个 run 逐 k 对照。差值是从数据里算的,不是写死的结论。 */
function Compare({ runA, runB, nameA, nameB }) {
  const ks = Object.keys(runA.aggregates || {}).sort((a, b) => Number(a) - Number(b))
  return (
    <div className="scroll-x">
      <table className="grid">
        <thead>
          <tr>
            <th>k</th>
            <th className="n">hit A</th>
            <th className="n">hit B</th>
            <th className="n">Δhit</th>
            <th className="n">ndcg A</th>
            <th className="n">ndcg B</th>
            <th className="n">Δndcg</th>
          </tr>
        </thead>
        <tbody>
          {ks.map((k) => {
            const a = runA.aggregates[k] || {}
            const b = runB.aggregates[k] || {}
            const dh = a.hit !== undefined && b.hit !== undefined ? b.hit - a.hit : null
            const dn =
              a.ndcg !== undefined && b.ndcg !== undefined ? b.ndcg - a.ndcg : null
            const cell = (d) =>
              d === null ? '—' : `${d > 0 ? '+' : ''}${d.toFixed(3)}`
            return (
              <tr key={k}>
                <td className="mono">@{k}</td>
                <td className="n">{a.hit === undefined ? '—' : pct(a.hit)}</td>
                <td className="n">{b.hit === undefined ? '—' : pct(b.hit)}</td>
                <td className="n" style={{ color: dh > 0 ? 'var(--ok)' : dh < 0 ? 'var(--bad)' : 'inherit' }}>
                  {cell(dh)}
                </td>
                <td className="n">{a.ndcg === undefined ? '—' : a.ndcg.toFixed(3)}</td>
                <td className="n">{b.ndcg === undefined ? '—' : b.ndcg.toFixed(3)}</td>
                <td className="n" style={{ color: dn > 0 ? 'var(--ok)' : dn < 0 ? 'var(--bad)' : 'inherit' }}>
                  {cell(dn)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="muted" style={{ fontSize: 12, padding: '6px 2px 0' }}>
        A = <span className="mono">{nameA}</span>,B = <span className="mono">{nameB}</span>。
        差值为 0 就是<strong>这两档在这套题上测不出差别</strong> —— 和"接近"不是一回事。
      </p>
    </div>
  )
}

function BaselineCard({ b }) {
  const runNames = Object.keys(b.runs || {})
  const [a, setA] = useState(runNames[0] || '')
  const [bb, setB] = useState(runNames[1] || '')
  const [showRaw, setShowRaw] = useState(false)

  // 换了基线就重置选择 —— 否则会保留上一个基线里不存在的 run 名,
  // 渲染出一张空表。
  useEffect(() => {
    setA(runNames[0] || '')
    setB(runNames[1] || '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b.name])

  return (
    <Card
      title={b.name}
      actions={
        <span className="inline">
          {b.created_at ? <Pill>{ts(b.created_at)}</Pill> : <Pill tone="warn">没有 created_at</Pill>}
          {b.corpus ? <Pill>{b.corpus.n_docs} 篇 / {b.corpus.n_chunks} 块</Pill> : null}
          {b.qa ? <Pill>可答题 {b.qa.n_answerable}</Pill> : null}
          {b.qa?.n_negative ? <Pill tone="warn">负例 {b.qa.n_negative}</Pill> : null}
        </span>
      }
    >
      <div className="stack">
        <div className="inline">
          {b.collection ? <Pill title="跑这份基线时用的 Qdrant collection">{b.collection}</Pill> : null}
          {b.ingest ? (
            <Pill>
              {b.ingest.contextual_enabled ? '定位语开' : '定位语关'} / 块 {b.ingest.chunk_size}
            </Pill>
          ) : null}
          {b.ingest?.rerank_model ? <Pill>{b.ingest.rerank_model}</Pill> : null}
          {b.qa?.include_drafts ? <Pill tone="warn">含草稿题</Pill> : null}
        </div>

        {runNames.length === 0 ? (
          <Empty>这份基线里没有 runs。</Empty>
        ) : (
          runNames.map((name) => {
            const run = b.runs[name]
            return (
              <div key={name} className="stack sm">
                <div className="inline">
                  <Pill tone="accent">{name}</Pill>
                  <Pill tone={run.use_rerank ? 'ok' : ''}>
                    重排 {run.use_rerank ? '开' : '关'}
                  </Pill>
                  <Pill>检索器 {run.retriever}</Pill>
                  <Pill>effective_k {run.effective_k}</Pill>
                  {run.n_negative ? <Pill>负例 {run.n_negative}</Pill> : null}
                </div>
                <RunTable baseline={b} run={run} />
                <details className="raw-box">
                  <summary>按题型拆分</summary>
                  <ByType run={run} />
                </details>
              </div>
            )
          })
        )}

        {runNames.length >= 2 && b.runs[a] && b.runs[bb] ? (
          <div className="stack sm">
            <div className="row">
              <label className="field" style={{ flex: '1 1 200px' }}>
                <span className="label">对照 A</span>
                <select className="select" value={a} onChange={(e) => setA(e.target.value)}>
                  {runNames.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field" style={{ flex: '1 1 200px' }}>
                <span className="label">对照 B</span>
                <select className="select" value={bb} onChange={(e) => setB(e.target.value)}>
                  {runNames.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {/* 可比性:语料/题集不同的话这张差值表没有意义。
                基线文件本身带着 corpus 指纹和题集 sha1,这里对一下就报出来,
                而不是让读者自己记得"这两份能比吗"。 */}
            {a === bb ? (
              <Banner tone="warn">A 和 B 是同一档,差值恒为 0。</Banner>
            ) : (
              <Compare runA={b.runs[a]} runB={b.runs[bb]} nameA={a} nameB={bb} />
            )}
          </div>
        ) : null}

        <div>
          <button className="btn small" onClick={() => setShowRaw((v) => !v)}>
            {showRaw ? '收起原始 JSON' : '展开原始 JSON'}
          </button>
        </div>
        {showRaw ? <RawBox label="原始基线 JSON">{b}</RawBox> : null}
      </div>
    </Card>
  )
}

export default function EvalTab() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(true)

  useEffect(() => {
    let alive = true
    api
      .evalBaselines()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(e.message))
      .finally(() => alive && setBusy(false))
    return () => {
      alive = false
    }
  }, [])

  const baselines = data?.baselines || []

  // 有几个不同的语料指纹 —— 拿来提示"这些基线不一定能互相比较"。
  const corpora = useMemo(
    () => [...new Set(baselines.map((b) => b.corpus?.slug || '(未知语料)'))],
    [baselines]
  )

  return (
    <div className="panel-inner">
      <div>
        <h1 className="page-title">评测</h1>
        <p className="page-note">
          下面全部是<strong>跑出来的数</strong>,接口原样透传,这一层不做任何重算 ——
          重算就等于多出一套可能和基线文件不一致的口径。
        </p>
      </div>

      <Card title="怎么读这张表">
        <div className="stack sm">
          <p style={{ margin: 0 }}>
            每个指标都必须连着 <span className="mono">k</span> 和 <span className="mono">n</span> 一起读。
            同一个 <span className="mono">hit</span> 在 n=15 和 n=25 上是两个不同的数;只看百分数
            会得出"提高了"或"退步了"的假结论。表里每一行都把 n 摆在指标旁边,就是为了这个。
          </p>
          <p style={{ margin: 0 }}>
            <span className="mono">effective_k</span> 那一行是这份基线<strong>对外报数时用的档位</strong> ——
            别拿别的 k 去和别的基线比。
          </p>
          <p style={{ margin: 0 }}>
            <strong>对照要一次只变一个变量。</strong> 用下面的 A/B 选择器时,先确认两档除目标变量
            外完全一致(尤其是重排开关):像 <span className="mono">rerank_off</span> 这种名字里写着
            一个变量、实际同时关了另一个的旧基线,拿它做对照会得出错的结论。界面上每个 run 都把
            「重排 开/关」和「检索器」单独列出来了,就是给你核这个的。
          </p>
          <p style={{ margin: 0 }}>
            <strong>关于 GraphRAG:</strong> 这套种子语料只有 9 篇规程 / 61 个块,实体图能「跳」的
            空间本来就很小。把 <span className="mono">hybrid</span> 和 <span className="mono">graphrag</span>
            两档放在 A/B 里比一下 —— 如果差值那一列是 <span className="mono">+0.000</span>,
            那就是<strong>在这套语料上没测出增益</strong>,不是"接近"。简历上对应的措辞也是按这个口径写的。
          </p>
        </div>
      </Card>

      {busy ? <Banner spinner>正在读基线文件…</Banner> : null}
      {err ? <Banner tone="bad">读不到基线:{err}</Banner> : null}

      {!busy && !err && baselines.length === 0 ? (
        <Card>
          <Empty>
            没有基线文件。<span className="mono"> eval/baselines/</span> 里还没有跑出过东西 ——
            先按 README 跑一次评测。
          </Empty>
        </Card>
      ) : null}

      {corpora.length > 1 ? (
        <Banner tone="warn">
          这些基线跨了 {corpora.length} 套语料({corpora.join('、')})。<strong>跨语料的数不可比</strong> ——
          语料指纹一变,召回难度就变了,分数跟着变是必然的。
        </Banner>
      ) : null}

      {baselines.map((b) => (
        <BaselineCard key={b.name} b={b} />
      ))}
    </div>
  )
}
