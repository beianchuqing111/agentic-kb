import { Meter, Pill } from './Bits.jsx'
import { score, statusOf } from '../lib/format.js'

/**
 * 一条检索结果。
 *
 * 显示口径跟着 `api/serialize.py`:`score`/`rrf_score`/`rerank_score` 是
 * **原始字段,带 null**;`scores{}` 是同值的并排视图。`null` 一律显示成
 * `—` 而不是 `0` —— "没开重排"和"重排给了零分"必须看得出区别。
 *
 * `dense_rank`/`sparse_rank` 是**名次**(第几),不是分数,所以不给它们
 * 画进度条。给名次画条会让人读成"相似度 0.3",那是另一回事。
 */
export default function HitCard({ hit, index, onTrace, rank }) {
  const st = statusOf(hit)
  const hasExplain = hit.dense_rank !== null || hit.sparse_rank !== null

  return (
    <li className="hit">
      <header className="hit-head">
        <span className="rank">{rank ?? index + 1}</span>
        <span className="hit-source" title={hit.citation}>
          {hit.title || hit.source || hit.doc_id}
        </span>
        <span className="mono muted">块 {hit.chunk_index}</span>
        {hit.from_graph ? (
          <Pill tone="accent" title="这条是图多跳那一路捞回来的,不是向量召回">
            图召回
          </Pill>
        ) : null}
        {/* 这个数**不是**分数、也不是"命中了几块"。`graphrag_backend.py` 里它是
            `count(DISTINCT e.id)` —— 本次查询的种子实体里,有几个出现在这一块。
            标签必须写准:写成「图命中 6」会被读成"6 个图命中",那是对不上的。
            为 0 时不显示:「图命中 0」挂在一次刻意关掉图的 hybrid 检索上是噪音。 */}
        {hit.graph_hits ? (
          <Pill title="本次查询的种子实体中有几个出现在这一块(等于下面列的实体数)">
            图中命中 {hit.graph_hits} 个实体
          </Pill>
        ) : null}
        <Pill tone={st.key === 'superseded' ? 'warn' : ''}>{st.label}</Pill>
        {hit.doc_version ? <Pill>v{hit.doc_version}</Pill> : null}
        <span className="nameplate-spacer" />
        {onTrace ? (
          <button
            className="btn small"
            onClick={() =>
              onTrace({
                docId: hit.doc_id,
                chunkIndex: hit.chunk_index,
                label: hit.title || hit.source,
                source: hit.source,
              })
            }
          >
            回溯原文
          </button>
        ) : null}
      </header>

      <div className="hit-body">
        <div className="scores">
          <Meter label="最终" value={hit.scores.final} />
          <Meter label="RRF" value={hit.scores.rrf} digits={5} />
          {/* 没开重排时 rerank 是 null → 显示 —。这正是要区分的那件事。 */}
          <Meter label="重排" value={hit.scores.rerank} />
          {hasExplain ? (
            <span className="meter" title="这条被哪一路召回、在那一路排第几">
              <span>名次</span>
              <span className="val">
                稠密 {hit.dense_rank ?? '—'} / 稀疏 {hit.sparse_rank ?? '—'}
              </span>
            </span>
          ) : null}
          <span className="meter" title="原始 score 字段(未做任何换算)">
            <span>raw</span>
            <span className="val">{score(hit.score)}</span>
          </span>
        </div>

        {hit.context ? (
          <p className="hit-context" title="导入时模型写的定位语:这块在全文的什么位置">
            {hit.context}
          </p>
        ) : null}

        <p className="hit-text">{hit.text}</p>

        {hit.entities?.length ? (
          <div className="entities">
            {hit.entities.map((e) => (
              <span className="entity" key={e}>
                {e}
              </span>
            ))}
          </div>
        ) : null}

        {hit.facts?.length ? (
          <ul className="facts">
            {hit.facts.map((f, i) => (
              <li key={i}>
                {f.head} <span className="rel">--{f.relation}--&gt;</span> {f.tail}
              </li>
            ))}
          </ul>
        ) : null}

        <div className="mono muted" style={{ overflowWrap: 'anywhere', fontSize: 11 }}>
          {hit.source}
        </div>
      </div>
    </li>
  )
}
