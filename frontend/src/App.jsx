import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api.js'
import Nameplate from './components/Nameplate.jsx'
import CitationDrawer from './components/CitationDrawer.jsx'
import SearchTab from './components/SearchTab.jsx'
import AskTab from './components/AskTab.jsx'
import IngestTab from './components/IngestTab.jsx'
import LibraryTab from './components/LibraryTab.jsx'
import EvalTab from './components/EvalTab.jsx'
import AuditTab from './components/AuditTab.jsx'
import { Banner, Pill } from './components/Bits.jsx'
import { int } from './lib/format.js'
import { resolveDoc } from './lib/citations.js'

const TABS = [
  { key: 'search', label: '检索', hint: 'search' },
  { key: 'ask', label: '问答', hint: 'react' },
  { key: 'ingest', label: '导入', hint: 'ingest' },
  { key: 'library', label: '库状态', hint: 'docs' },
  { key: 'eval', label: '评测', hint: 'eval' },
  { key: 'audit', label: '审计', hint: 'audit' },
]

const THEMES = ['auto', 'light', 'dark']
const THEME_LABEL = { auto: '跟随系统', light: '亮色', dark: '暗色' }

function useTheme() {
  const [theme, setTheme] = useState(() => {
    // localStorage 在某些上下文里**访问本身就抛**(隐私模式、被禁的站点数据),
    // 所以读也要包 try —— 不包的话页面直接白屏。
    try {
      const v = localStorage.getItem('agentic-kb-theme')
      return THEMES.includes(v) ? v : 'auto'
    } catch {
      return 'auto'
    }
  })

  useEffect(() => {
    // `auto` 时**不写** data-theme:让 `prefers-color-scheme` 那一段生效。
    // 写死成某个值会把系统的选择也一起盖掉。
    if (theme === 'auto') delete document.documentElement.dataset.theme
    else document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('agentic-kb-theme', theme)
    } catch {
      /* 存不下就只影响下次打开,不影响这次 */
    }
  }, [theme])

  return [theme, () => setTheme((t) => THEMES[(THEMES.indexOf(t) + 1) % THEMES.length])]
}

export default function App() {
  const [tab, setTab] = useState('search')
  const [backend, setBackend] = useState('')
  const [backendList, setBackendList] = useState([])
  const [current, setCurrent] = useState('')
  const [health, setHealth] = useState({ data: null, error: '' })
  const [tools, setTools] = useState(null)
  const [docs, setDocs] = useState([])
  const [drawer, setDrawer] = useState(null)
  const [notice, setNotice] = useState(null)
  const [tick, setTick] = useState(0)
  const [theme, cycleTheme] = useTheme()

  // 健康探测。**不走队列**(见 api.js)——它是导入进行中唯一还能立刻
  // 回答的端点,把它排到后面就等于"忙的时候连忙不忙都问不出来"。
  useEffect(() => {
    let alive = true
    let timer = null
    const poll = async () => {
      try {
        const d = await api.health()
        if (alive) setHealth({ data: d, error: '' })
      } catch (e) {
        if (alive) setHealth({ data: null, error: e.message })
      }
      if (alive) timer = setTimeout(poll, 5000)
    }
    poll()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    api.backends().then((d) => {
      setBackendList(d.backends || [])
      setCurrent(d.current || '')
    }).catch(() => {})
    api.tools().then(setTools).catch(() => {})
  }, [])

  // 文档列表:引用解析要用它把「文档名」对到 doc_id。
  // 走队列(它碰后端),失败不吵 —— 解析不到时会给明确提示。
  const refreshDocs = useCallback(() => {
    api
      .docs()
      .then((d) => setDocs(d.docs || []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    refreshDocs()
  }, [refreshDocs, tick])

  useEffect(() => {
    if (!notice) return
    const t = setTimeout(() => setNotice(null), 6000)
    return () => clearTimeout(t)
  }, [notice])

  const openTrace = useCallback((t) => {
    setDrawer({ ...t, _key: Date.now() })
  }, [])

  /**
   * 答案里的引用点开。这里只有**名字**,没有 doc_id ——
   * `/api/ask` 的响应体里不带 Hit 对象(契约 §ask)。所以要拿名字去
   * 文档列表里对。对不上就说清楚,不猜。
   */
  const openCite = useCallback(
    (name, chunk) => {
      const doc = resolveDoc(name, docs)
      if (!doc) {
        setNotice({
          tone: 'warn',
          text: `引用「${name}」在库里对不到文档,没法回溯。可能是模型写的标签和真实标题有出入,也可能是这篇已经被重新导入过。`,
        })
        return
      }
      openTrace({ docId: doc.doc_id, chunkIndex: chunk, label: doc.title || doc.source, source: doc.source })
    },
    [docs, openTrace]
  )

  const onIngestChanged = useCallback(() => {
    setTick((v) => v + 1)
    setNotice({ tone: 'ok', text: '导入完成,库状态和文档列表已刷新。' })
  }, [])

  return (
    <div className="shell">
      <Nameplate
        backend={backend}
        onBackend={setBackend}
        backendList={backendList}
        current={current}
        health={health}
      />

      <div className="body">
        <nav className="rail" aria-label="功能">
          {TABS.map((t) => (
            <button
              key={t.key}
              className="rail-item"
              aria-current={tab === t.key ? 'page' : undefined}
              onClick={() => setTab(t.key)}
            >
              {t.label}
              <span className="rail-hint">{t.hint}</span>
            </button>
          ))}

          <div className="rail-foot">
            {health.data?.config ? (
              <div className="inline" style={{ gap: 4 }}>
                <Pill tone={health.data.config.allow_write ? 'warn' : ''}>
                  写权限 {health.data.config.allow_write ? '开' : '关'}
                </Pill>
              </div>
            ) : null}
            <button className="btn small" onClick={cycleTheme} title="切换主题">
              {THEME_LABEL[theme]}
            </button>
          </div>
        </nav>

        <main className="panel">
          {!health.data && health.error ? (
            <div style={{ marginBottom: 16 }}>
              <Banner tone="bad">
                连不上后端({health.error})。确认
                <span className="mono"> scripts/api_server.py </span>
                在跑 —— 前端只连它的代理,不自己起服务。
              </Banner>
            </div>
          ) : null}

          {health.data && !health.data.ok ? (
            <div style={{ marginBottom: 16 }}>
              <Banner tone="warn">
                后端在,但健康探测没过:{health.data.error || '没给原因'}。
                检索/问答多半会 503,库状态和审计还能看。
              </Banner>
            </div>
          ) : null}

          {tab === 'search' ? (
            <SearchTab backend={backend} onTrace={openTrace} />
          ) : tab === 'ask' ? (
            <AskTab backend={backend} tools={tools} onCite={openCite} onDone={() => setTick((v) => v + 1)} />
          ) : tab === 'ingest' ? (
            <IngestTab backend={backend} onChanged={onIngestChanged} health={health} />
          ) : tab === 'library' ? (
            <LibraryTab versionTick={tick} onTrace={openTrace} />
          ) : tab === 'eval' ? (
            <EvalTab />
          ) : (
            <AuditTab versionTick={tick} />
          )}
        </main>
      </div>

      <CitationDrawer key={drawer?._key} target={drawer} onClose={() => setDrawer(null)} />

      {notice ? (
        <div className="toast-wrap">
          <Banner tone={notice.tone}>{notice.text}</Banner>
        </div>
      ) : null}
    </div>
  )
}
