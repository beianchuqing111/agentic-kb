"""语料定位、doc_id 解析、锚点校验。

本地优先:只读文件系统 + 复用 `ingest` 的分块器,不联网。
唯一碰外部服务的是 `verify_ingested`,它查的是**已经导好的** collection。

## 为什么锚点要存「相对文件名 + chunk_index」而不是 point_id

`stable_doc_id(source)`(`ingest/loader.py:71-86`)是
`sha1(规范化绝对路径)[:16]` —— **路径决定,内容无关**:

- 改内容 → id 不变,锚点继续有效
- 改名 / 移动 → id 变,锚点失效

也就是说,把绝对路径写进评估集,项目一搬迁整份问答集就废了。存**语料目录下的
相对名**、运行时再算 id,就只对「改名/移动」敏感 —— 而这件事恰好能在开跑前查出来。
`stable_doc_id` 内部会 `resolve()` 再 lower 再转正斜杠,所以
`D:\\Corpus` 与 `d:\\corpus\\` 算出的 id 相同,即便 Windows 的路径字符串本身大小写不一。

## 锚点失效 ≠ 检索失败

在目标块之前增删字符,会把后面所有块的 `chunk_index` 整体推移一格。此时锚点指向
**别的块**,分数会掉,但**检索器没有任何问题 —— 是评估集过期了**。混为一谈会让人
去调一个没坏的参数。所以每条 gold 除 `(file, chunk_index)` 还存目标块正文里的一段
原样文字(`snippet`),开跑前逐条比对正文,对不上就报「锚点失效」并拒绝出分。

比对用的是 `snippet in chunk.text`(**中段子串**,不是 `startswith`):人工挑的
distinctive 片段往往在块的中后部;而只匹配开头的话,一个纯标题片段会在别的块里
悄悄命中,把位移盖住。位移报错时会把「这段文字现在实际在哪一块」一并算出来 ——
把长排查变成五秒修复。
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from config import BASE_DIR
from ingest import SUPPORTED_EXTS, TextChunk, chunk_text, discover, load_many, stable_doc_id
from store import point_id

__all__ = [
    "ChunkKey",
    "CorpusError",
    "AnchorIssue",
    "AnchorReport",
    "ResolvedGold",
    "TextHit",
    "DEFAULT_CORPUS",
    "default_corpus",
    "resolve_corpus",
    "corpus_slug",
    "corpus_fingerprint",
    "collection_name",
    "load_doc_chunks",
    "locate_text",
    "resolve_anchors",
    "verify_ingested",
    "format_issues",
]

# (语料内相对文件名, chunk_index) —— 与 eval.metrics.ChunkKey 同形。
# 这里不 import metrics,是为了让 corpus 不依赖任何评估语义。
ChunkKey = tuple[str, int]

DEFAULT_CORPUS: Path = BASE_DIR / "eval" / "corpus" / "seed"


class CorpusError(ValueError):
    """语料层面的问题:目录不存在、文件缺失、锚点越界。"""


# ---------------------------------------------------------------- 语料定位


def default_corpus() -> Path:
    """出厂种子语料的绝对路径。"""
    return DEFAULT_CORPUS


def resolve_corpus(spec: str | Path | None) -> Path:
    """`--corpus` 的取值 → 一个确认可用的语料根目录。

    只接受目录,不接受单个文件:锚点是相对**根**的路径,根含糊了锚点就含糊。
    """
    root = Path(spec) if spec else DEFAULT_CORPUS
    if not root.is_absolute():
        root = BASE_DIR / root
    try:
        root = root.resolve()
    except OSError:  # pragma: no cover - 罕见的路径解析失败
        root = root.absolute()

    if not root.is_dir():
        raise CorpusError(f"语料目录不存在或不是目录:{root}")

    files = discover(root)
    if not files:
        exts = ",".join(sorted(SUPPORTED_EXTS))
        raise CorpusError(
            f"语料目录里没有可入库的文档:{root}\n"
            f"  支持的后缀:{exts}\n"
            f"  提醒:discover() 不跳过 README.md,说明性文档要放到语料的上一级"
        )
    return root


_NON_SLUG = re.compile(r"[^a-z0-9]+")


def corpus_slug(root: str | Path) -> str:
    """语料目录 → collection 名里能用的短 slug(取目录名)。"""
    s = _NON_SLUG.sub("_", Path(root).name.lower()).strip("_")
    return s[:24] or "corpus"


def corpus_fingerprint(root: str | Path) -> str:
    """语料根目录的路径指纹(8 位)。

    刻意与 `stable_doc_id` 用同一套规范化(resolve → 正斜杠 → lower):
    同一个目录的不同写法必须落到同一个 collection,不同目录必须落到不同的 ——
    否则会读到上一份语料遗留的点,而症状是「检索结果里混着别的文档」。
    """
    key = str(Path(root).resolve()).replace("\\", "/").lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def collection_name(
    root: str | Path,
    *,
    contextual: bool,
    prefix: str = "agentic_kb_eval",
) -> str:
    """评估专用 collection 名。

    **定位语开关必须进名字。** 增量导入是按块的内容指纹判定的:同一批文档
    换一个开关重新导入,如果 collection 名一样,`existing_hashes` 会把每块都判成
    「没变过」,`docs_unchanged == N`、一块都不重写 —— 你以为在 A/B 两种定位语,
    其实两次读的是同一个库,而且**不会报任何错**。
    """
    return f"{prefix}_{corpus_slug(root)}_{'ctx' if contextual else 'noctx'}_{corpus_fingerprint(root)}"


# ---------------------------------------------------------------- 加载与分块


def _rel_key(root: Path, path: str | Path) -> str:
    """绝对路径 → 语料内的正斜杠相对路径(与 schema 的 `_norm_relpath` 口径一致)。"""
    p = Path(path)
    try:
        return p.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return p.as_posix().replace("\\", "/")


def load_doc_chunks(
    root: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool = True,
    on_error: str = "skip",
) -> tuple[dict[str, list[TextChunk]], list[tuple[str, str]]]:
    """语料目录 → `{相对路径: 分块列表}`,外加加载失败清单。

    分块必须与导入时**逐字一致**:同一个 `chunk_text`、同一套
    `chunk_size` / `chunk_overlap`。这条不是洁癖 —— 生成问答对时报的
    `chunk_index` 必须就是 pipeline 导入时会分配的索引,参数不一致会产出随机偏移的
    锚点,症状和检索 bug 一模一样,而排查方向会完全跑偏。
    """
    root = Path(root).resolve()
    paths = discover(root, recursive=recursive)
    docs, errors = load_many(paths, on_error=on_error)

    out: dict[str, list[TextChunk]] = {}
    for doc in docs:
        chunks = chunk_text(doc.text, chunk_size, chunk_overlap)
        if not chunks:
            errors.append((str(doc.source), "分块后为空"))
            continue
        out[_rel_key(root, doc.source)] = chunks
    return out, errors


def locate_text(
    docs: Mapping[str, Sequence[TextChunk]],
    needle: str,
    *,
    limit: int = 40,
) -> list[TextHit]:
    """在**已分块的本地语料**里找一段文字,返回它落在哪几块。

    这是「这段文字在第几块」的**定位器**,存在的意义是改完语料后重排
    `chunk_index` 锚点。它**刻意不查检索器**:用检索器的输出去定 gold,
    会让评估循环自证 —— 检索器总能捞出它自己认为对的块,分数永远 ~1.0。

    先精确匹配;一个都没命中再退化成大小写不敏感(中文无所谓,英文缩写会用到)。
    """
    hits: list[TextHit] = []
    if not needle:
        return hits

    for fname in sorted(docs):
        for ch in docs[fname]:
            if needle in ch.text:
                hits.append(_hit(fname, ch, needle))
                if len(hits) >= limit:
                    return hits

    if hits:
        return hits

    low = needle.lower()
    for fname in sorted(docs):
        for ch in docs[fname]:
            if low in ch.text.lower():
                hits.append(_hit(fname, ch, needle))
                if len(hits) >= limit:
                    return hits
    return hits


def _hit(fname: str, ch: TextChunk, needle: str) -> "TextHit":
    pos = ch.text.find(needle)
    if pos < 0:
        pos = ch.text.lower().find(needle.lower())
    section = getattr(ch, "section", "") or ""
    return TextHit(
        file=fname,
        chunk_index=ch.index,
        section=section,
        offset=pos,
        preview=ch.text[max(0, pos - 12) : pos + len(needle) + 12].replace("\n", " "),
    )


@dataclass(frozen=True)
class TextHit:
    """一段文字在语料里的落点。`offset` 是块内字符偏移,-1 表示没定位到。"""

    file: str
    chunk_index: int
    section: str
    offset: int
    preview: str

    def __str__(self) -> str:
        sec = f"   [{self.section}]" if self.section else ""
        return f"{self.file}  #{self.chunk_index}{sec}\n      …{self.preview}…"


# ---------------------------------------------------------------- 锚点解析


@dataclass(frozen=True)
class ResolvedGold:
    """一条通过校验的 gold 锚点。`key` 是检索结果能直接对上的块坐标。"""

    key: ChunkKey  # (doc_id, chunk_index)
    file: str  # 语料内相对名
    chunk_index: int
    grade: int
    snippet: str
    doc_id: str

    @property
    def label(self) -> str:
        return f"{self.file}#{self.chunk_index}"


@dataclass(frozen=True)
class AnchorIssue:
    """一条锚点问题。`kind` 决定让人去改什么。"""

    item_id: str
    file: str
    chunk_index: int | None
    kind: str  # missing_file | out_of_range | snippet_mismatch | not_ingested
    detail: str
    suggestion: str = ""

    def __str__(self) -> str:
        head = f"[{self.item_id}] {self.file}"
        if self.chunk_index is not None:
            head += f"#{self.chunk_index}"
        line = f"  {head}\n      {self.kind}: {self.detail}"
        if self.suggestion:
            line += f"\n      → {self.suggestion}"
        return line


@dataclass
class AnchorReport:
    """锚点解析结果。`ok` 为假时**不允许出分**。"""

    resolved: dict[str, list[ResolvedGold]] = field(default_factory=dict)
    issues: list[AnchorIssue] = field(default_factory=list)
    n_items: int = 0
    n_golds: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def keys(self, item_id: str) -> list[ChunkKey]:
        return [g.key for g in self.resolved.get(item_id, ())]

    @property
    def all_keys(self) -> list[ChunkKey]:
        out: list[ChunkKey] = []
        for golds in self.resolved.values():
            out.extend(g.key for g in golds)
        return out

    @property
    def all_golds(self) -> list[ResolvedGold]:
        out: list[ResolvedGold] = []
        for golds in self.resolved.values():
            out.extend(golds)
        return out

    def summary(self) -> str:
        return (
            f"锚点:{self.n_golds} 条 gold / {self.n_items} 题,"
            f"问题 {len(self.issues)} 条"
        )


def resolve_anchors(
    es,
    root: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    docs: Mapping[str, Sequence[TextChunk]] | None = None,
) -> AnchorReport:
    """逐条 gold 对着**本地重新分块**的语料校验,产出可用的块坐标。

    只校验 `keep` 状态的题 —— `draft` 里的锚点本来就是半成品,不该拦住跑分。

    三类失败分开报:

    - `missing_file` —— 文件改名/删除了。给 `difflib` 的近似名字建议
    - `out_of_range` —— 块数变少了,`chunk_index` 越界
    - `snippet_mismatch` —— **最要命的一类**:没报错、块也还在,但内容对不上。
      语料前面增删几个字就会这样。此时会把「这段文字现在实际在哪一块」算出来
    """
    root = Path(root).resolve()
    if docs is None:
        docs, load_errors = load_doc_chunks(
            root, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        load_errors = []

    report = AnchorReport(n_items=0, n_golds=0)
    known = sorted(docs)

    for item in es.kept:
        report.n_items += 1
        if item.is_negative:
            report.resolved[item.id] = []
            continue

        golds: list[ResolvedGold] = []
        for g in item.expected:
            report.n_golds += 1
            where = f"{item.id} 的 gold"

            # 1. 文件在不在
            if g.file not in docs:
                report.issues.append(
                    AnchorIssue(
                        item_id=item.id,
                        file=g.file,
                        chunk_index=g.chunk_index,
                        kind="missing_file",
                        detail=f"语料目录里没有这个文件({where})",
                        suggestion=_suggest_file(g.file, known),
                    )
                )
                continue

            chunks = docs[g.file]
            # 2. chunk_index 越界 —— 文件在,但内容变短了
            if not 0 <= g.chunk_index < len(chunks):
                report.issues.append(
                    AnchorIssue(
                        item_id=item.id,
                        file=g.file,
                        chunk_index=g.chunk_index,
                        kind="out_of_range",
                        detail=(
                            f"{where} 指向第 {g.chunk_index} 块,"
                            f"但该文件现在只有 {len(chunks)} 块(0~{len(chunks) - 1})"
                        ),
                        suggestion=_suggest_index(chunks, g.snippet, g.file),
                    )
                )
                continue

            # 3. 内容对不对 —— 中段子串,不是 startswith
            text = chunks[g.chunk_index].text
            if g.snippet and g.snippet not in text:
                report.issues.append(
                    AnchorIssue(
                        item_id=item.id,
                        file=g.file,
                        chunk_index=g.chunk_index,
                        kind="snippet_mismatch",
                        detail=(
                            f"{where} 说目标块含 {_short(g.snippet)},"
                            f"但第 {g.chunk_index} 块里找不到这段文字"
                        ),
                        suggestion=_suggest_index(chunks, g.snippet, g.file),
                    )
                )
                continue

            doc_id = stable_doc_id(str((root / g.file)))
            golds.append(
                ResolvedGold(
                    key=(doc_id, g.chunk_index),
                    file=g.file,
                    chunk_index=g.chunk_index,
                    grade=g.grade,
                    snippet=g.snippet,
                    doc_id=doc_id,
                )
            )

        report.resolved[item.id] = golds

    for path, msg in load_errors:
        report.issues.append(
            AnchorIssue(
                item_id="-",
                file=_rel_key(root, path),
                chunk_index=None,
                kind="load_failed",
                detail=f"语料文件读不出来:{msg}",
                suggestion="修好这个文件或把它移出语料目录;它现在既不能被评估,也不能被导入",
            )
        )
    return report


def _short(s: str, n: int = 30) -> str:
    return f"「{s[:n]}…」" if len(s) > n else f"「{s}」"


def _suggest_file(missing: str, known: Sequence[str]) -> str:
    near = difflib.get_close_matches(missing, known, n=3, cutoff=0.4)
    if near:
        return "最接近的现有文件名:" + "、".join(near)
    return "语料目录里现有的文件:" + "、".join(known[:8]) + ("…" if len(known) > 8 else "")


def _suggest_index(
    chunks: Sequence[TextChunk],
    snippet: str,
    fname: str,
) -> str:
    """锚点对不上时,直接算出这段文字**现在**在哪一块。"""
    if not snippet:
        return f"该文件现在共 {len(chunks)} 块;建议用 eval_run.py --locate 重新定位"

    where = [c.index for c in chunks if snippet in c.text]
    if where:
        return (
            f"这段文字现在落在 {fname} 的第 {where} 块 —— "
            f"像是锚点位移(在目标块之前增删了字符),把 chunk_index 改成其中之一即可"
        )
    hits = locate_text({fname: chunks}, snippet)
    if hits:
        return f"近似命中:{hits[0]}"
    return (
        "这段文字在当前语料里完全找不到 —— 语料被改过了,"
        "该 gold 需要重新标注,而不是调检索参数"
    )


# ---------------------------------------------------------------- 导入后复验


def verify_ingested(
    store,
    report: AnchorReport,
    *,
    resolve_missing: bool = True,
) -> list[AnchorIssue]:
    """导入之后,用**一次批量 `fetch_chunks`** 复验锚点真的在库里。

    前置校验查的是磁盘上的语料,这里查的是 collection 里实际存着的东西。
    两者会分叉:导入时个别文件读失败被 `skip_errors` 跳过、内容指纹判定
    「没变过」而跳过重写(但库里存的是旧版本)、导入被 Ctrl-C 打断。

    这类分叉如果不管,症状是「所有题都 miss」,而人会去怀疑检索器。
    """
    keys = report.all_keys
    if not keys:
        return []

    records = store.fetch_chunks(keys)
    have: dict[str, str] = {}
    for rec in records:
        payload = dict(rec.payload or {})
        have[str(rec.id)] = str(payload.get("text", ""))

    # 反查:point_id → 我们关心的那条 gold
    wanted: dict[str, ResolvedGold] = {}
    for golds in report.resolved.values():
        for g in golds:
            wanted[point_id(*g.key)] = g

    issues: list[AnchorIssue] = []
    for pid, g in wanted.items():
        if pid not in have:
            issues.append(
                AnchorIssue(
                    item_id="-",
                    file=g.file,
                    chunk_index=g.chunk_index,
                    kind="not_ingested",
                    detail=f"{g.label} 不在 collection 里(语料在磁盘上有,库里没有)",
                    suggestion=(
                        "加 --reingest 重导;若该文件加载报错,先修文件"
                        if resolve_missing
                        else "该文件在导入时被跳过了,先修文件再重导"
                    ),
                )
            )
            continue
        text = have[pid]
        if g.snippet and g.snippet not in text:
            issues.append(
                AnchorIssue(
                    item_id="-",
                    file=g.file,
                    chunk_index=g.chunk_index,
                    kind="snippet_mismatch",
                    detail=f"{g.label} 在库里的正文与锚点对不上(库里存的是旧版本)",
                    suggestion=(
                        "加 --reingest --purge 清掉这个 collection 重导;"
                        "只加 --reingest 可能因为内容指纹命中而跳过重写"
                    ),
                )
            )
    return issues


def format_issues(issues: Iterable[AnchorIssue], *, limit: int = 40) -> str:
    """把问题清单打成一段人能直接动手的文本。"""
    items = list(issues)
    if not items:
        return ""
    lines = [f"锚点校验失败,共 {len(items)} 条 —— 拒绝出分。", ""]
    for iss in items[:limit]:
        lines.append(str(iss))
    if len(items) > limit:
        lines.append(f"  …另有 {len(items) - limit} 条,已省略")
    lines.append("")
    lines.append(
        "  注意:锚点失效**不是检索质量下降**。它说明评估集或语料过期了,"
        "在这个状态下打出的任何分数都不可信,所以这里直接停掉而不是照常打印一张表。"
    )
    return "\n".join(lines)
