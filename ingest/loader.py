"""文档加载:txt / md / pdf / docx → Document。

两个中文环境下必须处理的坑
------------------------
一、编码。Windows 上有人用记事本存的中文 txt 是 **GBK**,不是 UTF-8。
   `open(p, encoding='utf-8')` 会直接 UnicodeDecodeError,而用
   `errors='ignore'` 又会**静默吞掉所有中文** —— 文件能读进来、
   库能建起来、检索永远返回空,查到最后是编码问题。
   所以这里的顺序是:先按字节探 BOM 和 UTF-8 合法性,失败才转 GBK,
   并且**把选中的编码记进 metadata**,出问题时有据可查。

二、doc_id 必须由「来源路径」决定,不能由「内容」决定。
   用内容哈希做 id 的话,改一个字 → id 变了 → 旧 chunk 不会被认为是
   同一篇文档,于是旧数据既不覆盖也不删除,库里堆两份。
   所以 id 用规范化绝对路径的哈希:同一文件重导入 = 同一个 id,
   配合 Chunk.id(也是确定性的)天然幂等。内容变没变交给 content_hash 判断。
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# 支持的后缀。加新格式时 **必须**同步更新 DISPATCH。
TEXT_EXTS = {".txt", ".md", ".markdown", ".text", ".log"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}

SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS | DOCX_EXTS

# 多字节编码按这个顺序试。GB18030 是 GBK/GB2312 的超集,放最后兜底。
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5")

_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t　]+$", re.MULTILINE)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass
class Document:
    """一篇文档的纯文本形态。分块在 pipeline 里做。"""

    doc_id: str
    text: str
    source: str = ""                       # 原始路径(绝对)
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def content_hash(self) -> str:
        """**整篇**文档的指纹。注意和 chunk 级指纹不是一回事:
        这个用来判断「这篇文件整体变没变」,决定要不要重新分块。"""
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- #
# doc_id
# --------------------------------------------------------------------- #

def stable_doc_id(source: str) -> str:
    """来源 → 稳定 id。

    规范化后再哈希,避免同一文件因为写法不同(相对/绝对、正反斜杠、
    大小写)拿到两个 id。Windows 路径大小写不敏感,所以统一 lower。
    """
    p = Path(source)
    try:
        if p.exists():
            key = str(p.resolve())
        else:
            key = str(p.absolute())
    except OSError:
        key = str(source)
    key = key.replace("\\", "/").lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------- #
# 编码
# --------------------------------------------------------------------- #

def read_text(path: Path) -> tuple[str, str]:
    """读出文本,返回 (文本, 实际用的编码)。

    先试多字节编码,成功即止;全失败才用 latin-1 硬解(保证不丢字节),
    并在日志里告警 —— 那种情况基本可以确定文档本身有问题。
    """
    raw = path.read_bytes()
    if not raw:
        return "", "utf-8"

    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue

    logger.warning("%s 用 %s 都解不出来,按 latin-1 硬解(中文大概率是乱码)",
                   path, "/".join(_ENCODINGS))
    return raw.decode("latin-1", errors="replace"), "latin-1"


# --------------------------------------------------------------------- #
# 清洗
# --------------------------------------------------------------------- #

def normalize_text(text: str) -> str:
    """统一换行、压缩空行、去掉行尾空白。

    **不做**全角转半角 —— 中文标点(,。、)在语义上是有用的,
    转成半角反而让分词和展示都变差。只把全角空格归到普通空格
    (它多半是从网页/PDF 复制来的排版残留)。

    NFKC 也不做:那会把「①」变成「1」、「㎡」变成「m2」,
    对有单位/编号的技术文档是破坏性的。
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("　", " ")          # 全角空格
    t = t.replace(" ", " ")          # 不换行空格
    t = t.replace("﻿", "")           # 零宽 BOM
    # 零宽字符:PDF 复制常见,肉眼不可见但会污染分词
    t = t.replace("​", "").replace("‌", "").replace("‍", "")
    t = _TRAILING_WS.sub("", t)
    t = _BLANK_RUN.sub("\n\n", t)
    return t.strip()


def _guess_title(doc: Document) -> str:
    """标题:markdown 一级标题 > 正文第一个短行 > 文件名。"""
    m = _MD_HEADING.search(doc.text[:2000])
    if m:
        return m.group(1).strip()[:120]

    for line in doc.text.split("\n", 30)[:30]:
        s = line.strip()
        if 4 <= len(s) <= 40:
            return s
    return Path(doc.source).stem if doc.source else doc.doc_id


# --------------------------------------------------------------------- #
# 各格式的读取
# --------------------------------------------------------------------- #

def _load_text_file(path: Path) -> Document:
    text, enc = read_text(path)
    text = normalize_text(text)
    doc = Document(
        doc_id=stable_doc_id(str(path)),
        text=text,
        source=str(path),
        metadata={"ext": path.suffix.lower(), "encoding": enc, "size": path.stat().st_size},
    )
    doc.title = _guess_title(doc)
    return doc


def _load_pdf(path: Path) -> Document:
    """按页抽文本。页间用空行隔开 —— 分块时它是天然的断开点。

    抽不到文本的 PDF(扫描件)会得到空字符串,这里**显式报错**,
    而不是返回空文档。空文档进库的后果是检索永远命中不了它,
    却在文档列表里看得见,极难排查。扫描件该走 OCR,那是另一条链路。
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")            # 有些 PDF 只是「限制编辑」,空密码能解
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"PDF 已加密,无法读取: {path}") from exc

    pages: list[str] = []
    empty_pages = 0
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 第 %d 页抽取失败: %s", path.name, i + 1, exc)
            t = ""
        if not t.strip():
            empty_pages += 1
        pages.append(t.strip())

    text = normalize_text("\n\n".join(pages))
    if not text:
        raise ValueError(
            f"PDF 抽不出任何文本(共 {len(reader.pages)} 页,全空): {path}\n"
            f"多半是扫描件。这种要走 OCR,当前 loader 不支持。"
        )
    if empty_pages:
        logger.info("%s 有 %d/%d 页无文本(可能是图片页)",
                    path.name, empty_pages, len(reader.pages))

    doc = Document(
        doc_id=stable_doc_id(str(path)),
        text=text,
        source=str(path),
        metadata={
            "ext": ".pdf",
            "pages": len(reader.pages),
            "empty_pages": empty_pages,
            "size": path.stat().st_size,
        },
    )
    doc.title = _guess_title(doc)
    return doc


def _load_docx(path: Path) -> Document:
    """段落 + 表格。

    表格别丢 —— 中文技术文档里参数表特别多(「额定容量 500kVA」这种),
    而且表格文本按行列顺序拼出来正好是可检索的短语。
    """
    import docx  # python-docx

    d = docx.Document(str(path))
    parts: list[str] = [p.text for p in d.paragraphs]

    for ti, table in enumerate(d.tables):
        parts.append(f"[表 {ti + 1}]")
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            # 去掉被合并单元格造成的重复
            dedup: list[str] = []
            for c in cells:
                if not dedup or dedup[-1] != c:
                    dedup.append(c)
            line = " | ".join(x for x in dedup if x)
            if line:
                parts.append(line)

    text = normalize_text("\n".join(parts))
    if not text:
        raise ValueError(f"docx 里没有文本: {path}")

    doc = Document(
        doc_id=stable_doc_id(str(path)),
        text=text,
        source=str(path),
        metadata={"ext": ".docx", "tables": len(d.tables), "size": path.stat().st_size},
    )
    doc.title = _guess_title(doc)
    return doc


_DISPATCH = {
    **{e: _load_text_file for e in TEXT_EXTS},
    **{e: _load_pdf for e in PDF_EXTS},
    **{e: _load_docx for e in DOCX_EXTS},
}


# --------------------------------------------------------------------- #
# 对外
# --------------------------------------------------------------------- #

def load_file(path: str | Path) -> Document:
    """读单个文件。不支持的格式直接抛错,不静默跳过 ——
    静默跳过会让人以为「导入了 100 篇」,其实只进去 60 篇。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if not p.is_file():
        raise ValueError(f"不是文件: {p}")

    ext = p.suffix.lower()
    loader = _DISPATCH.get(ext)
    if loader is None:
        raise ValueError(
            f"不支持的格式 {ext!r}:{p}\n支持的:{sorted(SUPPORTED_EXTS)}"
        )
    return loader(p)


def discover(root: str | Path, recursive: bool = True) -> list[Path]:
    """扫描目录下所有支持的文件,排序保证可复现。"""
    r = Path(root)
    if r.is_file():
        return [r] if r.suffix.lower() in SUPPORTED_EXTS else []
    if not r.is_dir():
        raise FileNotFoundError(f"路径不存在: {r}")

    it = r.rglob("*") if recursive else r.glob("*")
    files = [
        p for p in it
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and not p.name.startswith("~$")        # Office 锁文件
        and not p.name.startswith(".")         # 隐藏文件
    ]
    return sorted(files, key=lambda p: str(p).lower())


def load_many(
    paths: Iterable[str | Path],
    on_error: str = "raise",
) -> tuple[list[Document], list[tuple[str, str]]]:
    """批量加载,返回 (成功的文档, [(路径, 错误信息)])。

    on_error:
      "raise"  —— 有一个失败就整体失败(用于确知输入干净的场合)
      "skip"   —— 跳过坏文件继续(用于扫描整个目录,个别坏文件不该阻塞)
    """
    docs: list[Document] = []
    errors: list[tuple[str, str]] = []

    for p in paths:
        try:
            docs.append(load_file(p))
        except Exception as exc:  # noqa: BLE001
            if on_error == "raise":
                raise
            logger.warning("加载失败,跳过: %s —— %s", p, exc)
            errors.append((str(p), str(exc)))

    return docs, errors
