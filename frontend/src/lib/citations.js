/**
 * 把答案里的引用标签解析成可点的东西,并解析到具体文档。
 *
 * 引用长什么样**不是**前端定的 —— `agent/react.py:106` 的系统提示词里
 * 写死了:
 *
 *     引用必须标注来源:知识库的块用 `[文档名 (块 3)]` 这种标签
 *
 * 但这是个**提示词约定**,模型不保证 100% 遵守。所以这里的解析是宽容的:
 * 全角/半角括号都收,`【】` 也收,名字里的空白先归一。解析不出来的就
 * 原样当普通文字显示 —— 硬按一种格式切会把正常段落切碎,那比少一个
 * 可点链接糟得多。
 *
 * 网页引用 `[标题](URL)` 是同一个系统提示词里的另一条格式,它**不会**
 * 被这里的正则匹配到(匹配要求括号里是 `块 N`),所以 markdown 链接
 * 不会被误当成知识库引用。
 */

// `[文档名 (块 3)]` / `【文档名 (块 3)】` / 全角括号
const CITE_RE = /[\[【]([^\[\]【】]{1,140}?)\s*[（(]\s*块\s*(\d+)\s*[)）]\s*[\]】]/g

/**
 * 把一段文本切成 `{type:'text'}` 和 `{type:'cite'}` 交替的片段。
 */
export function parseCitations(text) {
  const src = String(text || '')
  const out = []
  let last = 0
  CITE_RE.lastIndex = 0
  let m
  while ((m = CITE_RE.exec(src)) !== null) {
    if (m.index > last) out.push({ type: 'text', text: src.slice(last, m.index) })
    out.push({
      type: 'cite',
      name: m[1].trim(),
      chunk: Number(m[2]),
      raw: m[0],
    })
    last = m.index + m[0].length
  }
  if (last < src.length) out.push({ type: 'text', text: src.slice(last) })
  return out
}

/** 名字归一:全角空格、换行、大小写、扩展名差异都不该影响匹配。 */
function norm(s) {
  return String(s || '')
    .replace(/[\s　]+/g, '')
    .replace(/[《》「」"']/g, '')
    .toLowerCase()
}

function baseName(p) {
  return String(p || '').split(/[\\/]/).pop() || ''
}

/**
 * 按引用里的名字找文档。
 *
 * 匹配顺序是**从精确到宽松**,而且必须先精确:`title` 可能重复(同一份
 * 规程的多个版本),先按 `source` 全路径比才能区分。反过来先模糊匹配
 * 会把「绝缘子检测作业指导书 v1」的引用指到 v2 上,而那正好是版本化
 * 功能要防的事情。
 *
 * 找不到返回 `null` —— **不要**退回"随便挑一个相似的"。指错文档比
 * 指不出来严重得多:用户点开看到一段不像的原文,会开始怀疑整个检索。
 */
export function resolveDoc(name, docs) {
  const n = norm(name)
  if (!n) return null
  for (const d of docs) {
    if (norm(d.title) === n) return d
  }
  for (const d of docs) {
    if (norm(d.source) === n) return d
  }
  for (const d of docs) {
    if (norm(baseName(d.source)) === n) return d
  }
  // 最后才允许"包含"。只在这一层放宽,且要求长度够,避免「规程」两个字
  // 匹配到第一篇文章。
  if (n.length >= 4) {
    for (const d of docs) {
      const t = norm(d.title)
      const s = norm(d.source)
      if ((t && t.includes(n)) || (s && s.includes(n)) || (t && n.includes(t))) return d
    }
  }
  return null
}
