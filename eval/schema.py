"""评估集的数据结构与 JSONL 读写。**纯标准库**,不 import 项目里任何东西。

为什么是 JSONL 而不是 YAML
-------------------------
这份文件的存在意义,恰恰落在几个最容易被 YAML 悄悄改型的字段上:

  * YAML 1.1 的 `safe_load` 把 `on/off/yes/no` 读成布尔 —— `note` 里写
    「这条先 on 着」就变成 `True`;
  * `1:30` 会被按 60 进制解析成 90;
  * 前导零的 `03` 可能变成整数 3,而这里的**载荷正是文件名和 chunk_index**。

再加上两条实际得多的理由:
  * `json.loads` 失败能指到**行号**,而 YAML 只给重解析后的字符偏移 ——
    中文 note 里一个全角冒号就能给出让人看不懂的报错;
  * 人工筛的物理动作就是「从 draft 里挑好行,复制进定稿文件再改」,
    一行一条让这个动作就是字面意义的复制粘贴。

注释用「跳过空行与 `#` 开头行」来支持,够用了;`# meta: {...}` 那一行会被
解析成文件级元信息(记录 chunk_size 等),runner 靠它判断评估集是否过期。

锚点为什么是「文件名 + 块序号」而不是 id
--------------------------------------
`stable_doc_id` 是**路径**的 sha1(`ingest/loader.py:12` 说明了为什么必须如此:
否则改一个字就产生新 id、旧数据堆两份)。所以把 doc_id 写死进评估集有两个问题:
一是换台机器换个盘符就全废,二是 16 位 hex 人根本没法读、没法手改。

写成语料目录下的**相对文件名**,runner 运行时算 doc_id,就能扛住项目搬迁、
内容修改、重新导入 —— 只在**文件改名/移动**时失效,而那恰好能在开跑前查出来。

`snippet` 是给这个失效兜底的:在目标块**之前**增删字符会让 chunk_index 整体位移,
锚点就指向别的块了。此时分数会掉,但**检索器没坏,是评估集过期了** ——
两者混为一谈会让人去调一个根本没坏的参数。存一段正文首 30 字,
开跑前比对得上才继续,把这两种失败彻底分开。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# snippet 存多少字。只要能认出「是不是同一块」即可,太长会让手工编辑变累。
SNIPPET_CHARS = 30

# 允许的人工筛状态。runner 只算 keep,其余只打印计数 ——
# 一条写错的 draft 会让评估集静默缩小,比报错更危险,所以要看得见。
STATUSES = ("keep", "draft", "drop", "skip")

# 已知题型。**未知题型只告警不报错** —— 让人能自己加类而不必改代码。
KNOWN_TYPES = (
    "lexical",      # 该由稀疏赢:术语/编号精确匹配
    "semantic",     # 该由稠密赢:换了说法,字面不重合
    "multi_hop",    # 多个 gold 块共同作答
    "contextual",   # 短块脱离语境看不懂,定位语才救得回 —— CONTEXTUAL_ENABLED 的直接考场
    "negative",     # 库内无答案:考「会不会硬凑」
    "ambiguous",    # 有多个合理答案,看排序合不合理
    "single_hop",   # 默认:一块就能答
)

# grade 的含义。分级只服务 nDCG;其余指标一律按「上榜即相关」二值化。
GRADE_ANSWER = 3   # 直接回答
GRADE_SUPPORT = 2  # 强相关,但要结合别的块
GRADE_TOPIC = 1    # 同主题,不是答案
GRADE_MAX = 3


class EvalSetError(ValueError):
    """评估集本身有问题(格式错、字段矛盾、锚点越界)。

    单独一个异常类型,是为了让 CLI 能把它和「基础设施坏了」区分开:
    前者要人去看评估集,后者要人去看 Qdrant。
    """


# --------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------- #


@dataclass
class GoldChunk:
    """一个期望被召回的块。"""

    file: str                        # 语料目录下的相对路径,正斜杠
    chunk_index: int
    grade: int = GRADE_TOPIC         # 缺省按最低档算,免得忘了填就被当成满分
    snippet: str = ""                # 目标块正文首 SNIPPET_CHARS 字,用于检测锚点失效

    def key(self, doc_id: str) -> tuple[str, int]:
        return (doc_id, self.chunk_index)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"file": self.file, "chunk_index": self.chunk_index}
        # grade 缺省值不写进文件 —— 默认值出现在每一行里只会让 diff 变吵
        if self.grade != GRADE_TOPIC:
            d["grade"] = self.grade
        if self.snippet:
            d["snippet"] = self.snippet
        return d

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, where: str) -> GoldChunk:
        if "file" not in raw or "chunk_index" not in raw:
            raise EvalSetError(f"{where}: gold 必须有 file 和 chunk_index,实得 {raw!r}")
        idx = raw["chunk_index"]
        # bool 是 int 的子类,`isinstance(True, int)` 为真 —— 不挡的话
        # `"chunk_index": true` 会被当成 1,锚点静默指错块。
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise EvalSetError(f"{where}: chunk_index 必须是整数,实得 {idx!r}")
        if idx < 0:
            raise EvalSetError(f"{where}: chunk_index 不能为负,实得 {idx}")

        grade = raw.get("grade", GRADE_TOPIC)
        if isinstance(grade, bool) or not isinstance(grade, int):
            raise EvalSetError(f"{where}: grade 必须是整数,实得 {grade!r}")
        if not 1 <= grade <= GRADE_MAX:
            # grade=0 是「列上去了但一点都不相关」—— 自相矛盾。
            # 不能容忍:二值指标把上榜一律当相关,放行 0 会直接虚高。
            raise EvalSetError(
                f"{where}: 列在 expected 里的 gold,grade 必须在 1~{GRADE_MAX},实得 {grade}"
                "(真要表示不相关就别列它)"
            )

        return cls(
            file=_norm_relpath(str(raw["file"]), where=where),
            chunk_index=idx,
            grade=grade,
            snippet=str(raw.get("snippet") or ""),
        )


@dataclass
class EvalItem:
    """一条「问题 → 期望命中」。"""

    id: str
    question: str
    expected: list[GoldChunk] = field(default_factory=list)
    type: str = "single_hop"
    status: str = "keep"
    answer: str = ""
    note: str = ""
    added_by: str = ""
    added_at: str = ""

    @property
    def is_negative(self) -> bool:
        """没有期望块 = 反例。

        由**数据**决定而不是由 `type` 决定,是为了让调用方不必同时信两个字段;
        校验器会强制 `type == "negative"` 与「expected 为空」严格等价,
        所以两者永远一致。
        """
        return not self.expected

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "type": self.type,
            "status": self.status,
            "expected": [g.to_json() for g in self.expected],
            "answer": self.answer,
            "note": self.note,
            "added_by": self.added_by,
            "added_at": self.added_at,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, where: str) -> EvalItem:
        if not isinstance(raw, dict):
            raise EvalSetError(f"{where}: 每一行必须是一个 JSON 对象,实得 {type(raw).__name__}")
        qid = str(raw.get("id") or "").strip()
        question = str(raw.get("question") or "").strip()
        if not qid:
            raise EvalSetError(f"{where}: 缺 id")
        if not question:
            raise EvalSetError(f"{where}: {qid} 缺 question(空查询没法评)")

        status = str(raw.get("status") or "keep").strip()
        if status not in STATUSES:
            raise EvalSetError(
                f"{where}: {qid} 的 status={status!r} 不认识,只能是 {STATUSES}"
            )

        raw_expected = raw.get("expected") or []
        if not isinstance(raw_expected, list):
            raise EvalSetError(f"{where}: {qid} 的 expected 必须是列表")
        expected = [
            GoldChunk.from_json(g, where=f"{where} [{qid}].expected[{i}]")
            for i, g in enumerate(raw_expected)
        ]

        seen: set[tuple[str, int]] = set()
        for g in expected:
            k = (g.file, g.chunk_index)
            if k in seen:
                raise EvalSetError(f"{where}: {qid} 重复列了同一个 gold {k}")
            seen.add(k)

        return cls(
            id=qid,
            question=question,
            expected=expected,
            type=str(raw.get("type") or "single_hop").strip(),
            status=status,
            answer=str(raw.get("answer") or ""),
            note=str(raw.get("note") or ""),
            added_by=str(raw.get("added_by") or ""),
            added_at=str(raw.get("added_at") or ""),
        )


@dataclass
class EvalSet:
    """一份评估集 = 若干题目 + 文件级元信息。"""

    items: list[EvalItem]
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # --- 视图 ---

    @property
    def kept(self) -> list[EvalItem]:
        return [it for it in self.items if it.status == "keep"]

    @property
    def answerable(self) -> list[EvalItem]:
        return [it for it in self.kept if not it.is_negative]

    @property
    def negatives(self) -> list[EvalItem]:
        return [it for it in self.kept if it.is_negative]

    def status_counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for it in self.items:
            out[it.status] = out.get(it.status, 0) + 1
        return out

    def type_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.kept:
            out[it.type] = out.get(it.type, 0) + 1
        return out

    def by_id(self) -> dict[str, EvalItem]:
        return {it.id: it for it in self.items}

    def files(self) -> list[str]:
        """被引用到的所有语料文件(去重、排序)。runner 用它做存在性校验。"""
        return sorted({g.file for it in self.kept for g in it.expected})

    def max_chunk_index(self) -> int:
        """引用到的最大块序号。

        用来做一条体检断言:如果它有 0,说明语料每篇只分出 1 块,
        `chunk_index` 恒等于 0,整套锚点方案等于没测
        (`scripts/check_ingest.py:42-45` 记过这个坑)。
        """
        return max((g.chunk_index for it in self.kept for g in it.expected), default=-1)

    def warnings(self) -> list[str]:
        """不致命、但值得让人看见的问题。"""
        out: list[str] = []

        # draft 里「expected 为空但不是反例」= 没标完。validate() 对它放行
        # (一行没标完不该让整份文件加载不了),但必须在这里说出来 ——
        # 否则一条永远停在中途的题会安静地躺在 draft 里,
        # 每次看 `status_counts()` 都以为「还有 3 条 draft,回头再弄」。
        for it in self.items:
            if it.status == "draft" and it.is_negative and it.type != "negative":
                out.append(
                    f"{it.id}: draft 且 expected 为空、type={it.type!r} —— "
                    "是还没标完的题?补 gold,或者改成 type=negative"
                )

        for it in self.kept:
            if it.type not in KNOWN_TYPES:
                out.append(
                    f"{it.id}: 未知题型 {it.type!r}(不是错误 —— 报告会单独归一类)"
                )
            if len(it.question) < 4:
                out.append(f"{it.id}: 问题过短 {it.question!r},查起来区分度低")
            for g in it.expected:
                if not g.snippet:
                    out.append(
                        f"{it.id}: gold {g.file}#{g.chunk_index} 没有 snippet,"
                        "改文档导致块位移时查不出来"
                    )
                elif len(g.snippet) > SNIPPET_CHARS * 2:
                    out.append(
                        f"{it.id}: gold {g.file}#{g.chunk_index} 的 snippet 过长,"
                        f"建议只留前 {SNIPPET_CHARS} 字"
                    )
        if self.max_chunk_index() < 1:
            out.append(
                "所有 gold 的 chunk_index 都是 0 —— 语料可能短到每篇只出一块,"
                "锚点方案没有被真正检验"
            )
        return out

    # --- 出处指纹 ---

    def sha1(self) -> str:
        """**参与评分的那些题**的内容指纹,用于基线可比性检查。

        为什么只算 `kept`:人把一条题标成 drop 之后,评分集合没变,
        此时若指纹变了,基线对比就会误报「不可比」,护栏变成噪声,人就学会忽略它。

        为什么要排序:重排一下行顺序不该被当成内容变化。

        为什么保证书用 JSON 且带 ensure_ascii=False:直接用 `to_json()` 的
        规范序列化,保证「同一份数据 → 同一个指纹」,而不用另写一套哈希逻辑。
        """
        canonical = [
            json.dumps(it.to_json(), ensure_ascii=False, sort_keys=True)
            for it in sorted(self.kept, key=lambda x: x.id)
        ]
        blob = "\n".join(canonical)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    # --- 校验 ---

    def validate(self) -> list[str]:
        """结构级校验。返回致命问题列表(空 = 通过)。

        致命与告警的分界:凡是**会让指标算错或虚高**的都算致命。
        「题型不认识」不会算错,所以只告警。
        """
        errs: list[str] = []

        ids = [it.id for it in self.items]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            errs.append(f"id 重复: {sorted(dupes)}")

        for it in self.items:
            # 反例与 expected 必须严格等价 —— 否则「可答题但没 gold」这种
            # 数据错误会被当成反例静默吸收进分母,把平均分往下拽。
            #
            # **只对 status=keep 致命**,因为只有 keep 会被算进去。draft 的
            # 「expected 为空但 type 还不是 negative」正是一条题**没标完**的
            # 正常形态(生成器出了题、锚点还没附上 / 人还没判它是不是反例),
            # 而 drop / skip 本就明说了不参与评分。三个都要放行 ——
            # 否则一行没标完,整份文件**加载不了**,`eval_gen` 的产出会
            # 因为一条坏行而全批读不进来。draft 的那种形态在 warnings() 里
            # 照常提醒,只是不拦。
            if it.status == "keep":
                if it.is_negative and it.type != "negative":
                    errs.append(
                        f"{it.id}: expected 为空但 type={it.type!r};"
                        "无反例意图的话请补 gold,有的话请把 type 改成 negative"
                    )
                if it.type == "negative" and not it.is_negative:
                    errs.append(f"{it.id}: type=negative 却列了 {len(it.expected)} 个 gold")

            for g in it.expected:
                if g.file.startswith("/") or (len(g.file) > 1 and g.file[1] == ":"):
                    errs.append(f"{it.id}: gold.file 必须是相对语料的路径,实得 {g.file!r}")

        if not self.kept:
            errs.append("没有任何 status=keep 的题目 —— 评估集是空的")

        return errs


# --------------------------------------------------------------------- #
# JSONL 读写
# --------------------------------------------------------------------- #


def _norm_relpath(raw: str, *, where: str) -> str:
    """把文件名规范化成「正斜杠相对路径」,并挡掉逃出语料目录的写法。"""
    p = raw.replace("\\", "/").strip()
    if not p:
        raise EvalSetError(f"{where}: gold.file 不能为空")
    if p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        raise EvalSetError(f"{where}: gold.file 必须是相对路径,实得 {p!r}")
    # `..` 会让 root/file 指到语料目录外面去 —— 那不是「锚点画错了」,
    # 是能读到不该读的文件。直接拒。
    if ".." in p.split("/"):
        raise EvalSetError(f"{where}: gold.file 不能含 ..,实得 {p!r}")
    return p


def load_qa(path: str | Path) -> EvalSet:
    """读一份 JSONL 评估集。

    容忍空行和 `#` 注释行(人工筛的时候要能写「# 这条待确认」)。
    一行解析失败就抛出,并带上**行号和原文** —— 手工维护的文件里,
    「哪一行坏了」比「坏在哪一列」有用得多。
    """
    p = Path(path)
    if not p.exists():
        raise EvalSetError(
            f"评估集不存在: {p}\n"
            "  还没生成过的话,先跑 scripts\\eval_gen.py 出初稿,人工筛完再跑本脚本。"
        )

    text = p.read_text(encoding="utf-8")
    items: list[EvalItem] = []
    meta: dict[str, Any] = {}

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # `# meta: {...}` 是文件级元信息,其余 `#` 行是纯注释
            head, sep, tail = line.lstrip("#").strip().partition(":")
            if sep and head.strip().lower() == "meta":
                try:
                    meta = json.loads(tail.strip())
                except json.JSONDecodeError as exc:
                    raise EvalSetError(f"{p}:{lineno}: meta 行不是合法 JSON: {exc}") from exc
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalSetError(
                f"{p}:{lineno}: 不是合法 JSON: {exc}\n  该行原文: {line[:200]}"
            ) from exc
        items.append(EvalItem.from_json(raw, where=f"{p}:{lineno}"))

    es = EvalSet(items=items, meta=meta, path=p)
    errs = es.validate()
    if errs:
        raise EvalSetError("评估集校验未通过:\n  - " + "\n  - ".join(errs))
    return es


def dump_qa(es: EvalSet, path: str | Path, *, header: Sequence[str] = ()) -> Path:
    """写一份 JSONL 评估集。

    `ensure_ascii=False` 是必须的 —— 中文以 `\\uXXXX` 形式落盘的话,
    这份文件就没法人工筛了,那等于把设施的核心环节废掉。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if es.meta:
        lines.append("# meta: " + json.dumps(es.meta, ensure_ascii=False, sort_keys=True))
    for h in header:
        lines.append("# " + h)
    for it in es.items:
        lines.append(json.dumps(it.to_json(), ensure_ascii=False))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def make_meta(*, corpus: str, chunk_size: int, chunk_overlap: int, generated_at: str) -> dict[str, Any]:
    """文件头元信息。

    `chunk_size` / `chunk_overlap` 必须记 —— 生成器报的 chunk_index 只有在
    **与导入时同一套分块参数**下才对得上。参数一变,锚点整体位移,
    表现和「检索变差了」一模一样。runner 靠这几个字段拒绝运行明显过期的评估集。
    """
    return {
        "schema_version": 1,
        "corpus": corpus,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "generated_at": generated_at,
    }


__all__ = [
    "SNIPPET_CHARS",
    "STATUSES",
    "KNOWN_TYPES",
    "GRADE_ANSWER",
    "GRADE_SUPPORT",
    "GRADE_TOPIC",
    "GRADE_MAX",
    "EvalSetError",
    "GoldChunk",
    "EvalItem",
    "EvalSet",
    "load_qa",
    "dump_qa",
    "make_meta",
]
