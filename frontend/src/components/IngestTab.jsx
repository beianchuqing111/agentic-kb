import { useRef, useState } from 'react'
import * as api from '../api.js'
import { Banner, Card, Field, KV, Pill, RawBox } from './Bits.jsx'
import { ms } from '../lib/format.js'

const ACCEPT = '.pdf,.docx,.doc,.txt,.md,.html,.htm,.csv,.json'

export default function IngestTab({ backend, onChanged, health }) {
  const [path, setPath] = useState('')
  const [recursive, setRecursive] = useState(true)
  const [force, setForce] = useState(false)
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState(null)
  const [err, setErr] = useState('')
  const [picked, setPicked] = useState([])
  const fileRef = useRef(null)

  async function runPath(e) {
    e?.preventDefault()
    const p = path.trim()
    if (!p) return setErr('路径不能为空。')
    setBusy(true)
    setErr('')
    setRes(null)
    try {
      const body = { path: p, recursive, force }
      if (backend) body.backend = backend
      const r = await api.ingestPath(body)
      setRes(r)
      onChanged?.()
    } catch (e2) {
      setErr(e2.message)
    } finally {
      setBusy(false)
    }
  }

  async function runUpload(e) {
    e?.preventDefault()
    if (!picked.length) return setErr('先选文件。')
    setBusy(true)
    setErr('')
    setRes(null)
    try {
      const r = await api.ingestUpload(picked, { recursive, force, backend })
      setRes(r)
      setPicked([])
      if (fileRef.current) fileRef.current.value = ''
      onChanged?.()
    } catch (e2) {
      setErr(e2.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel-inner">
      <div>
        <h1 className="page-title">导入</h1>
        <p className="page-note">
          导入会<strong>占用后端</strong>直到跑完(定位语的生成是每块一次模型调用),期间别的请求都在排队 ——
          顶栏的「等待中」就是给这一刻准备的。
        </p>
      </div>

      {/* LLM 没配就先说 —— 别等导完了才在结果里报「缺定位语」。
          同一件事,提前说和事后说不是一回事:提前说,人还可以先去把 key
          配上再导;事后说,这批文件的定位语要么补导要么就这么缺着。
          `webui.py` 也是在**动手前**提示的。 */}
      {health?.data?.config && !health.data.config.llm_configured ? (
        <Banner tone="warn">
          服务端<strong>没配 LLM</strong>(<span className="mono">LLM_API_KEY</span> 为空),
          本次导入会跳过「上下文定位语」。文档照样能进去、也能检索,
          但召回会明显变差 —— 尤其是块本身很短、指代很多的那类。
          填上 key 后<strong>重新导一遍即可补齐</strong>,不用先删。
        </Banner>
      ) : null}

      <Card
        title="选项"
        actions={
          <span className="inline">
            <Pill tone={force ? 'warn' : ''}>{force ? '强制重导' : '跳过未改动'}</Pill>
            <Pill>{recursive ? '递归子目录' : '只导这一层'}</Pill>
          </span>
        }
      >
        <div className="stack sm">
          <label className="check">
            <input type="checkbox" checked={recursive} onChange={(e) => setRecursive(e.target.checked)} />
            递归子目录
          </label>
          <label className="check">
            <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
            忽略内容指纹,全部重导
          </label>
          {force ? (
            <Banner tone="warn">
              重导会<strong>重新生成每一块的定位语</strong>(每块一次模型调用),也可能改掉库里的
              <span className="mono"> doc_id </span>对应关系 —— 老引用会失效。
              只有在定位语确实缺失、或者文件被改过而指纹没认出来时才用它。
            </Banner>
          ) : null}
        </div>
      </Card>

      <div className="row" style={{ alignItems: 'stretch' }}>
        <div className="grow">
          <Card title="从服务器路径导入">
            <form className="stack" onSubmit={runPath}>
              <Field label="服务器上的文件或目录" hint="路径按服务端的文件系统解释,不是你本机的。带引号粘进来也可以,后端会剥掉。">
                <input
                  className="input"
                  value={path}
                  onChange={(e) => setPath(e.target.value)}
                  placeholder="例如 D:\docs\规程"
                />
              </Field>
              <div className="inline">
                <span className="nameplate-spacer" />
                <button className="btn primary" disabled={busy}>
                  {busy ? <span className="spinner" /> : null}
                  {busy ? '导入中…' : '导入'}
                </button>
              </div>
            </form>
          </Card>
        </div>

        <div className="grow">
          <Card title="上传文件">
            <form className="stack" onSubmit={runUpload}>
              <Field label="选择文件" hint="会先落到 uploads/ 再入库 —— 不直接用临时路径,否则块的 source 会指向一个过一会儿就不存在的地方。">
                <input
                  ref={fileRef}
                  className="input"
                  type="file"
                  multiple
                  accept={ACCEPT}
                  onChange={(e) => setPicked(Array.from(e.target.files || []))}
                />
              </Field>
              {picked.length ? (
                <ul className="filelist">
                  {picked.map((f) => (
                    <li key={f.name}>
                      <span>{f.name}</span>
                      <span className="muted">{Math.ceil(f.size / 1024)} KB</span>
                    </li>
                  ))}
                </ul>
              ) : null}
              <div className="inline">
                <span className="nameplate-spacer" />
                <button className="btn primary" disabled={busy || !picked.length}>
                  {busy ? <span className="spinner" /> : null}
                  {busy ? '导入中…' : `上传并导入${picked.length ? ` (${picked.length})` : ''}`}
                </button>
              </div>
            </form>
          </Card>
        </div>
      </div>

      {err ? <Banner tone="bad">{err}</Banner> : null}

      {res ? (
        <Card
          title="本次导入"
          actions={
            <span className="inline">
              <Pill tone="accent">{res.backend}</Pill>
              <Pill>{ms(res.elapsed_ms)}</Pill>
              {/* LLM 调用数:界面**不能**写死"本次没有模型调用" ——
                  定位语是每块一次调用。null 表示拿不到这个计数,不是 0。 */}
              <Pill tone={res.llm_calls ? 'warn' : ''}>
                {res.llm_calls === null || res.llm_calls === undefined
                  ? '模型调用数未知'
                  : `模型调用 ${res.llm_calls} 次`}
              </Pill>
            </span>
          }
        >
          <div className="stack">
            <KV
              items={[
                ['扫描文件', res.files_seen],
                ['成功载入', res.files_loaded],
                ['索引文档', res.docs_indexed],
                ['未改动跳过', res.docs_unchanged],
                ['写入块', res.chunks_written],
                ['失败块', res.chunks_failed],
                ['缺定位语', res.contextual_missing],
                ['写入实体', res.entities_written],
                ['写入关系', res.relations_written],
              ]}
            />

            {res.chunks_failed ? (
              <Banner tone="warn">
                有 {res.chunks_failed} 块没写进去。看下面的错误明细 —— 块级失败不会让
                整批导入回滚,所以剩下的部分是好的。
              </Banner>
            ) : null}

            {res.contextual_missing ? (
              <Banner tone="warn">
                {res.contextual_missing} 块没有定位语。缺定位语会同时削弱召回和模型的判断
                —— 想补齐就对这批文件重新导入并勾上「强制重导」。
              </Banner>
            ) : null}

            {/* 同名覆盖的提示。上传这一步本身没出错,所以它不是"错误明细"
                里的一条 —— 但它改变了 uploads/ 里的内容,而块的 source
                就指着那里。不说出来的话,前后两次导入的溯源码会悄悄对不上。 */}
            {res.notes?.length ? (
              <div className="stack sm">
                {res.notes.map((n, i) => (
                  <Banner tone="warn" key={i}>
                    {n}
                  </Banner>
                ))}
              </div>
            ) : null}

            {res.saved?.length ? (
              <div>
                <div className="label" style={{ marginBottom: 4 }}>
                  已落盘到 uploads/
                </div>
                <ul className="filelist">
                  {res.saved.map((p) => (
                    <li key={p}>
                      <span style={{ overflowWrap: 'anywhere' }}>{p}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {res.errors?.length ? (
              <div>
                <div className="label" style={{ marginBottom: 4 }}>
                  错误明细({res.errors.length})
                </div>
                <div className="scroll-x">
                  <table className="grid">
                    <thead>
                      <tr>
                        <th>文件</th>
                        <th>原因</th>
                      </tr>
                    </thead>
                    <tbody>
                      {res.errors.map((e, i) => (
                        <tr key={i}>
                          <td className="mono">{e.file}</td>
                          <td style={{ whiteSpace: 'normal' }}>{e.error}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}

            <RawBox label="原始响应 JSON">{res}</RawBox>
          </div>
        </Card>
      ) : null}
    </div>
  )
}
