"""中文感知的分块。

为什么不用现成的
---------------
`chapter3/structured-index` 用的是:

    re.split(r'(?<=[.!?])\\s+', text)

这个正则在中文上**等于不切** —— 中文句末是「。!?」,而且后面不跟空格。
结果整段中文变成"一个句子",分块逻辑全部失效,chunk 大小完全失控。
这是那套代码里最致命的几行之一。

中文分句的要点:
  1. 句末符是 。!?…;(全角),英文的 .!?; 也要认
  2. 句末符后面**没有空格**,所以不能靠 \\s+ 判断边界
  3. 句末符后面常跟收尾符号(」』"）)》,这些要一起收进当前句
  4. 连续引号/书名号可能套多层

分块策略是「句子感知的贪心装箱」:先切句,再按顺序往块里塞,
塞不下就换块。**绝不从句子中间切开**,除非单句本身就超过块上限
(那种情况下只能硬切,但会记账)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 句末标点:中文全角 + 英文半角
SENTENCE_END = "。！？；!?;…"
# 句末之后可能紧跟的收尾符号,要和句子一起收进来
CLOSING = "」』”’）)》〉】〕｝\"')]}"

# 章节标题的粗略识别:第X章/节、1.2.3 标题、## 标题
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百零\d]+[章节篇部]"      # 第一章 / 第3节
    r"|[#]{1,6}\s*\S"                                  # markdown 标题
    r"|\d+(?:\.\d+){0,3}[、.\s]\s*\S"                  # 1.2.3 标题
    r")"
)


@dataclass
class Chunk:
    """一块文本 + 它在原文中的位置。

    位置信息不是为了好看 —— 引用要能定位回原文,
    没有 offset 就只能靠字符串查找,重复段落会指错地方。
    """

    text: str
    start: int
    end: int
    index: int
    section: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)


def split_sentences(text: str) -> list[tuple[int, int]]:
    """把文本切成句子,返回每句的 (start, end) 字符偏移。

    只返回偏移不返回字符串,是为了让调用方能按需切片,
    也避免大文档上反复复制字符串。
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    start = i

    while i < n:
        ch = text[i]

        # 换行一律断句 —— 中文里换行往往就是语义边界(列表项、小标题)
        if ch == "\n":
            if i > start and text[start:i].strip():
                spans.append((start, i))
            i += 1
            start = i
            continue

        if ch in SENTENCE_END:
            j = i + 1
            # 吃掉连续的句末符(「真的吗?!!」)和收尾符号(「他说。」」)
            while j < n and (text[j] in SENTENCE_END or text[j] in CLOSING):
                j += 1
            if text[start:j].strip():
                spans.append((start, j))
            i = j
            start = i
            continue

        i += 1

    if start < n and text[start:].strip():
        spans.append((start, n))

    return spans


def _detect_section(line: str) -> str:
    line = line.strip()
    if _HEADING_RE.match(line) and len(line) <= 60:
        return line
    return ""


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """句子感知分块。

    chunk_size / chunk_overlap 的单位是**字符**,不是 token。
    选字符是因为中文的 token/字符比不固定(XLM-R 对中文大约 1 字 ≈ 0.6~1 token),
    按字符数控制更可预测。bge-m3 的 max_length 是 8192 token,
    512 字符远在安全范围内。

    单句超长(比如没有标点的表格行、代码块)时只能硬切,
    但会尽量在标点上找落点,实在没有才真硬切。
    """
    if not text.strip():
        return []

    chunk_size = max(64, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

    spans = split_sentences(text)
    if not spans:
        return []

    # 预先把每句所属的章节标题算出来,后面组装 Chunk 时直接取。
    # heading_starts 记的是「这一句本身就是标题」的偏移 —— 分块时要在这些
    # 位置强制断开。用偏移集合而不是比较字符串,免得正文里恰好出现
    # 和标题一模一样的句子时被误判成标题。
    section_of: dict[int, str] = {}
    heading_starts: set[int] = set()
    current_section = ""
    for s, e in spans:
        head = _detect_section(text[s:e])
        if head:
            current_section = head
            heading_starts.add(s)
        section_of[s] = current_section

    chunks: list[Chunk] = []
    buf: list[tuple[int, int]] = []
    buf_len = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal buf, buf_len, chunk_index
        if not buf:
            return
        s = buf[0][0]
        e = buf[-1][1]
        body = text[s:e].strip()
        if body:
            chunks.append(
                Chunk(
                    text=body,
                    start=s,
                    end=e,
                    index=chunk_index,
                    section=section_of.get(buf[0][0], ""),
                )
            )
            chunk_index += 1
        buf = []
        buf_len = 0

    for span in spans:
        s, e = span
        seg = text[s:e]
        seg_len = e - s
        is_heading = s in heading_starts

        # 单句就超上限:先冲掉手上的,再把这句硬切
        if seg_len > chunk_size:
            flush()
            for hard_s, hard_e in _hard_split(text, s, e, chunk_size):
                body = text[hard_s:hard_e].strip()
                if body:
                    chunks.append(
                        Chunk(
                            text=body,
                            start=hard_s,
                            end=hard_e,
                            index=chunk_index,
                            section=section_of.get(s, ""),
                        )
                    )
                    chunk_index += 1
            continue

        # 标题处强制断开:**块不跨章节**。
        #
        # 不这么做的话,一个块会横跨「一、适用范围」和「二、变压器检查要点」,
        # 而 section 只能记一个值 —— 记的是块内**第一个**句子所属的章节,
        # 于是后面那一节的标签全是错的。存进 payload 再喂给 LLM 就是误导。
        # 而且混了两个主题的块,其向量也是两个主题的平均,谁都不像。
        #
        # 这里刻意**不做重叠**:重叠会把上一节的内容带进新一节,
        # 又把刚修好的边界弄脏了。
        if is_heading and buf:
            flush()

        if buf_len + seg_len > chunk_size and buf:
            flush()
            # 重叠:把上一块的尾部句子挪到新块开头。
            # 目的是让跨块的语义别被拦腰截断 —— 检索时前后文还在。
            if chunk_overlap > 0:
                tail_start = chunks[-1].end
                carry: list[tuple[int, int]] = []
                carry_len = 0
                for ps, pe in reversed(spans):
                    if pe > tail_start:
                        continue
                    if carry_len + (pe - ps) > chunk_overlap:
                        break
                    carry.insert(0, (ps, pe))
                    carry_len += pe - ps
                buf = carry
                buf_len = carry_len

        buf.append(span)
        buf_len += seg_len

    flush()
    return chunks


def _hard_split(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """对超长单句做硬切,但优先在次级标点上落刀。

    逗号、顿号、空格都算次级落点 —— 在逗号后断开比在字中间断开好得多。
    """
    soft = "，,、）) 　"
    out: list[tuple[int, int]] = []
    i = start
    while i < end:
        j = min(i + size, end)
        if j < end:
            # 从 j 往前找最近的次级标点
            k = j
            floor = i + size // 2  # 别为了找标点把块切得太小
            while k > floor and text[k - 1] not in soft:
                k -= 1
            if k > floor:
                j = k
        out.append((i, j))
        i = j
    return out
