import { useEffect, useRef, useState } from 'react'
import * as api from '../api.js'
import { Banner, Pill } from './Bits.jsx'

/**
 * 回溯原文:把一篇文档的**全部块**摊开,并高亮被引用的那一块。
 *
 * 存在的理由只有一个 —— 让"引用可核验"这句话成立。检索结果里的 `text`
 * 只是那一块的正文,单看它没法判断有没有被断章取义;要判断就必须看到
 * 它在全文里的位置和上下文。`GET /api/docs/{doc_id}` 就是为此加的
 * (见 `api/app.py` 里那条路由的注释)。
 *
 * 高亮块会**自动滚进视野**:一篇文档几十块时,用户点开引用却要自己找
 * 那一块,等于这个功能没做。
 */
export default function CitationDrawer({ target, onClose }) {
  const [state, setState] = useState({ status: 'idle', data: null, error: '' })
  const citedRef = useRef(null)

  useEffect(() => {
    if (!target?.docId) return
    let alive = true
    const ctl = new AbortController()
    setState({ status: 'loading', data: null, error: '' })
    api
      .docChunks(target.docId, ctl.signal)
      .then((data) => {
        if (alive) setState({ status: 'ready', data, error: '' })
      })
      .catch((err) => {
        if (!alive || err.name === 'AbortError') return
        setState({ status: 'error', data: null, error: err.message })
      })
    return () => {
      alive = false
      ctl.abort()
    }
  }, [target?.docId])

  // 数据到了再把高亮块滚进视野。用 rAF 而不是直接调:`scrollIntoView`
  // 在节点刚插入、布局还没算完时会滚到错的位置。
  useEffect(() => {
    if (state.status !== 'ready') return
    const t = requestAnimationFrame(() => {
      citedRef.current?.scrollIntoView({ block: 'center' })
    })
    return () => cancelAnimationFrame(t)
  }, [state.status, target?.chunkIndex])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!target) return null

  const d = state.data
  const title = target.label || d?.title || target.docId

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <aside
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`回溯原文:${title}`}
      >
        <header className="drawer-head">
          <div style={{ minWidth: 0, flex: '1 1 auto' }}>
            <div className="drawer-title">{title}</div>
            <div className="mono muted" style={{ marginTop: 4, overflowWrap: 'anywhere' }}>
              {d?.source || target.source || ''}
            </div>
            <div className="inline" style={{ marginTop: 7 }}>
              <Pill>块 {target.chunkIndex}</Pill>
              {d ? <Pill>全文共 {d.total} 块</Pill> : null}
              <Pill title={target.docId}>{String(target.docId).slice(0, 12)}…</Pill>
            </div>
          </div>
          <button className="btn small" onClick={onClose} aria-label="关闭">
            关闭
          </button>
        </header>

        <div className="drawer-body">
          {state.status === 'loading' ? (
            <div className="drawer-empty">
              <span className="spinner" style={{ display: 'inline-block', marginRight: 8 }} />
              正在取全文块…
              <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
                这条请求要和"导入中"互斥,所以服务端正忙时它会先排队(契约 §0.1)。
              </div>
            </div>
          ) : null}

          {state.status === 'error' ? (
            <div style={{ padding: 16 }}>
              <Banner tone="bad">取不到这篇文档:{state.error}</Banner>
              <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
                可能原因:这篇文档刚被重新导入(块已换了一批),或者检索结果来自
                另一次运行留下的缓存。回检索页重新搜一次即可。
              </div>
            </div>
          ) : null}

          {state.status === 'ready' && d
            ? d.chunks.map((c) => {
                const cited = c.chunk_index === target.chunkIndex
                return (
                  <article
                    className={`chunk${cited ? ' cited' : ''}`}
                    key={c.chunk_index}
                    ref={cited ? citedRef : null}
                  >
                    <div className="chunk-head">
                      <span>块 {c.chunk_index}</span>
                      {cited ? <Pill tone="accent">被引用</Pill> : null}
                      {c.status === 'superseded' ? <Pill tone="warn">块级已失效</Pill> : null}
                      {c.doc_version ? <span>版本 {c.doc_version}</span> : null}
                    </div>
                    {c.context ? <p className="hit-context">{c.context}</p> : null}
                    <p className="chunk-text">{c.text || <span className="muted">(空块)</span>}</p>
                  </article>
                )
              })
            : null}
        </div>
      </aside>
    </>
  )
}
