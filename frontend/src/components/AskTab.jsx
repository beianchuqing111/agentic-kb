import { useEffect, useRef, useState } from 'react'
import * as api from '../api.js'
import { Banner, Card, Field, Pill, RawBox } from './Bits.jsx'
import { isToolError, ms } from '../lib/format.js'
import { parseCitations } from '../lib/citations.js'

/** 把答案里 `[文档名 (块 3)]` 的引用渲染成可点的按钮。 */
function AnswerText({ text, onCite }) {
  return (
    <p className="answer">
      {parseCitations(text).map((seg, i) =>
        seg.type === 'cite' ? (
          <button
            key={i}
            className="cite-link"
            title="回溯到原文"
            onClick={() => onCite(seg.name, seg.chunk)}
          >
            {seg.raw}
          </button>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
    </p>
  )
}

export default function AskTab({ backend, tools, onCite, onDone }) {
  const [q, setQ] = useState('')
  const [stream, setStream] = useState(true)
  const [allowWrite, setAllowWrite] = useState(false)
  const [confirmed, setConfirmed] = useState([])
  const [busy, setBusy] = useState(false)
  const [events, setEvents] = useState([])
  const [result, setResult] = useState(null)
  const [err, setErr] = useState('')

  const abortRef = useRef(null)
  const bodyRef = useRef(null)

  const writeTools = (tools?.tools || []).filter((t) => t.kind === 'write')
  const masterOn = Boolean(tools?.allow_write)

  // 服务端总开关关着 → 闸 2 无论如何都不会生效。
  // 这不是"前端小心一点",是 `_build_agent` 里的实情:总开关关着时它
  // **连工具名单都不往下传**(见 app.py 那段注释)。所以这里必须禁掉并
  // 说明原因,而不是让用户勾了再收到一个"全被拒"。
  useEffect(() => {
    if (!masterOn) {
      setAllowWrite(false)
      setConfirmed([])
    }
  }, [masterOn])

  // 切走标签页时把没跑完的流掐掉 —— 否则它会一直占着后端那把锁,
  // 别的请求全排在后面,而用户已经看不到它在干什么了。
  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  async function run(e) {
    e?.preventDefault()
    const question = q.trim()
    if (!question) return setErr('问题不能为空。')

    const ctl = new AbortController()
    abortRef.current = ctl
    setBusy(true)
    setErr('')
    setEvents([])
    setResult(null)
    requestAnimationFrame(() => bodyRef.current?.scrollIntoView({ block: 'nearest' }))

    const payload = {
      question,
      include_superseded: null,
      allow_write: allowWrite,
      confirm_write_tools: allowWrite ? confirmed : [],
      stream,
    }
    if (backend) payload.backend = backend

    try {
      if (stream) {
        await api.askStream(
          payload,
          (ev) => {
            setEvents((prev) => [...prev, ev])
            if (ev.type === 'done') {
              setResult(ev.result)
              onDone?.(ev.result)
            }
            if (ev.type === 'error') setErr(ev.detail || '服务端报错')
          },
          { signal: ctl.signal }
        )
      } else {
        // 非流式那条路返回的对象和 `done.result` **同构**(契约 §0.2),
        // 所以下面渲染只用一套代码。
        const r = await api.ask(payload, ctl.signal)
        setResult(r)
        onDone?.(r)
      }
    } catch (e2) {
      if (e2.name !== 'AbortError') setErr(e2.message)
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  // 步骤列表:`step` 事件是"做完了",`action` 是"要开始做"。
  // 把 action 也收进列表并标成 running,是为了让最耗时的那一段
  // (检索/联网,几秒到几十秒)**在界面上有东西在动** ——
  // 否则页面会停在上一轮,和卡死没有区别。
  const steps = []
  for (const ev of events) {
    if (ev.type === 'action') steps.push({ kind: 'action', ...ev })
    else if (ev.type === 'step') steps.push({ kind: 'step', ...ev })
  }

  return (
    <div className="panel-inner">
      <div>
        <h1 className="page-title">问答</h1>
        <p className="page-note">
          走 ReAct:模型自己决定查知识库、查网页还是列文档,中间每一步都会显示出来。
          答案里的引用可以点开回溯原文。
        </p>
      </div>

      <Card title="提问">
        <form className="stack" onSubmit={run}>
          <Field label="问题">
            <textarea
              className="input"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="例如:这台主变的消缺时限是多久,依据在哪一条"
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run(e)
              }}
            />
          </Field>

          <div className="inline">
            <label className="check">
              <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
              流式显示中间步骤
            </label>
            <span className="nameplate-spacer" />
            {busy ? (
              <button type="button" className="btn" onClick={() => abortRef.current?.abort()}>
                中断
              </button>
            ) : null}
            <button className="btn primary" disabled={busy}>
              {busy ? <span className="spinner" /> : null}
              {busy ? '运行中…' : '提问'}
            </button>
          </div>
        </form>
      </Card>

      {/* 写权限两把闸。措辞按 app.py 的实现来:请求级意图 AND 服务端总开关。 */}
      <Card
        title="写权限"
        actions={
          <Pill tone={masterOn ? 'warn' : ''}>
            服务端 AGENT_ALLOW_WRITE = {masterOn ? '开' : '关'}
          </Pill>
        }
      >
        <div className="stack sm">
          {!masterOn ? (
            <Banner tone="warn">
              服务端总开关是关的,这一页的写权限<strong>不可能生效</strong>,所以下面两个控件都禁用。
              这不是"前端小心一点",是 <span className="mono">app.py</span> 的实情:它算的是
              <span className="mono"> allow = 请求的 allow_write AND 服务端开关</span>,
              并且把点名名单也一并清空(<span className="mono">confirm_write_tools if allow else []</span>)——
              勾了也送不到执行那一步。
              <br />
              但<strong>写工具本身仍在工具表里</strong>:就算你现在硬发一个带
              <span className="mono"> allow_write=true </span>的请求,模型<strong>照样会去调</strong>
              <span className="mono"> export_report</span>,然后被拒、并在「审计」页留下一条
              <span className="mono"> 被拒 </span>记录(本轮实测就是这个行为)。想真正演练受控写,
              要以 <span className="mono">AGENT_ALLOW_WRITE=true</span> 重启
              <span className="mono"> scripts/api_server.py</span>。
            </Banner>
          ) : (
            <Banner tone="warn">
              打开后模型才<strong>可能</strong>落盘。两把闸都要开:这里的是请求级意图,
              另一个是服务端总开关;再加上下面逐个点名的工具,三个条件同时成立才会真写。
            </Banner>
          )}

          <label className="check">
            <input
              type="checkbox"
              checked={allowWrite}
              disabled={!masterOn}
              onChange={(e) => setAllowWrite(e.target.checked)}
            />
            本次请求带写意图(<span className="mono">allow_write</span>)
          </label>

          <div style={{ paddingLeft: 21 }}>
            <div className="label" style={{ marginBottom: 5 }}>
              点名批准哪些写工具(不勾 = 一个都不批)
            </div>
            {writeTools.length === 0 ? (
              <div className="muted" style={{ fontSize: 12 }}>
                服务端没报告任何写工具。
              </div>
            ) : (
              <div className="stack sm">
                {writeTools.map((t) => (
                  <label className="check" key={t.name} title={t.parameter}>
                    <input
                      type="checkbox"
                      disabled={!masterOn || !allowWrite}
                      checked={confirmed.includes(t.name)}
                      onChange={(e) =>
                        setConfirmed((prev) =>
                          e.target.checked
                            ? [...prev, t.name]
                            : prev.filter((x) => x !== t.name)
                        )
                      }
                    />
                    <span className="mono">{t.name}</span>
                    <span className="muted" style={{ fontSize: 12 }}>
                      {t.description.split(/[。:：]/)[0]}
                    </span>
                  </label>
                ))}
              </div>
            )}
            <p className="muted" style={{ fontSize: 12, margin: '6px 0 0' }}>
              名单来自 <span className="mono">GET /api/tools</span>(写在代码里的真实工具表),
              不是前端写死的字符串 —— 工具改名时这里跟着变,不会出现"勾了但没批准"。
            </p>
          </div>
        </div>
      </Card>

      {err ? <Banner tone="bad">{err}</Banner> : null}

      <div ref={bodyRef} />

      {events.length > 0 ? (
        <Card
          title={busy ? '过程(运行中)' : '过程'}
          actions={
            <span className="inline">
              {busy ? (
                <Pill tone="accent" pulse>
                  正在运行
                </Pill>
              ) : null}
              <Pill>{steps.filter((s) => s.kind === 'step').length} 步</Pill>
            </span>
          }
        >
          <ul className="steps">
            {steps.map((s, i) => {
              const done = s.kind === 'step'
              return (
                <li className="step" key={i} data-state={done ? 'done' : 'running'}>
                  <div className="step-head">
                    <span className="step-idx">#{s.index}</span>
                    <span className="step-tool">{s.tool}</span>
                    <span className="nameplate-spacer" />
                    {!done ? (
                      <>
                        <span className="spinner" />
                        <span className="muted">正在调用…</span>
                      </>
                    ) : s.repeated ? (
                      <Pill tone="warn">重复调用,已短路</Pill>
                    ) : isToolError(s.observation) ? (
                      /* 写工具被拒时 `ToolRegistry.run` 返回的就是这种文本。
                         标成绿色的「完成」会读成"写成功了" —— 而它恰恰是没写。
                         这里只说"返回了错误":被拒 vs 执行失败要看审计页。 */
                      <Pill tone="bad" title="这一步工具返回了错误(可能是被拒,也可能是执行失败)—— 审计页有定论">
                        返回错误
                      </Pill>
                    ) : s.truncated ? (
                      <Pill tone="warn">已截断</Pill>
                    ) : (
                      <Pill tone="ok">完成</Pill>
                    )}
                  </div>
                  <div className="step-body">
                    {s.thought ? <div className="thought">{s.thought}</div> : null}
                    {s.input ? (
                      <div className="mono muted" style={{ overflowWrap: 'anywhere' }}>
                        输入: {s.input}
                      </div>
                    ) : null}
                    {done && s.observation ? <pre className="obs">{s.observation}</pre> : null}
                    {done && s.note ? <div className="muted" style={{ fontSize: 12 }}>{s.note}</div> : null}
                  </div>
                </li>
              )
            })}
          </ul>
        </Card>
      ) : null}

      {(result?.warnings?.length || events.some((e) => e.type === 'warning')) ? (
        <div className="stack sm">
          {events
            .filter((e) => e.type === 'warning')
            .map((e, i) => (
              <Banner tone="warn" key={`e${i}`}>
                {e.message || JSON.stringify(e)}
              </Banner>
            ))}
          {(result?.warnings || []).map((w, i) => (
            <Banner tone="warn" key={`r${i}`}>
              {w}
            </Banner>
          ))}
        </div>
      ) : null}

      {result ? (
        <Card
          title="答案"
          actions={
            <span className="inline">
              <Pill tone={result.stop_reason === 'final_answer' ? 'ok' : 'warn'}>
                停止原因:{result.stop_reason}
              </Pill>
              <Pill>{ms(result.elapsed_ms)}</Pill>
            </span>
          }
        >
          <div className="stack">
            {result.answer ? (
              <AnswerText text={result.answer} onCite={onCite} />
            ) : (
              <Banner tone="warn">
                这一轮没有给出最终答案(停止原因 {result.stop_reason})。
                {result.stop_reason === 'max_iterations'
                  ? '通常是轮次用完了 —— 上面的过程里能看到它卡在哪一步。'
                  : ''}
              </Banner>
            )}

            {result.usage && Object.keys(result.usage).length ? (
              <div className="inline">
                {Object.entries(result.usage).map(([k, v]) => (
                  <Pill key={k}>
                    {k} {typeof v === 'number' ? v.toLocaleString('zh-CN') : String(v)}
                  </Pill>
                ))}
              </div>
            ) : null}

            {/* 审计留痕:写操作跑过之后,这里能直接对上下面的审计页 */}
            <RawBox label="原始结果 JSON">{result}</RawBox>
          </div>
        </Card>
      ) : null}

      {events.length ? <RawBox label={`过程事件(${events.length} 条)`}>{events}</RawBox> : null}
    </div>
  )
}
