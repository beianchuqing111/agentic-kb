"""Anthropic 上下文感知检索(contextual retrieval)的落地。

要解决的问题
-----------
传统分块把一篇文档剁成互不相干的片段。于是「该公司当年营收增长 12%」
这种块根本没法检索 —— 谁是「该公司」?哪一年?块自己不知道,
稠密向量也就编不出有意义的语义,检索必然漏。

做法:每块**入库前**先让 LLM 读整篇文档,写一句定位语说明
「这段讲的是什么、在全文的什么位置」,把定位语拼在原文前面一起编码。
检索时命中的是拼好的文本,展示给用户的仍是原文。

三个容易做错的地方
----------------
一、定位语必须**独立存储**,不能直接改原文。
   直接改原文的话,引用出来的句子会带着 LLM 的话,
   用户看到的内容就不是文档里写的了 —— 这在需要溯源的知识库里是致命的。
   所以 Chunk 里 context 和 text 是两个字段,embed_text 才把它们拼起来。

二、**别让它总结块内容**。定位语是「这段话在讲什么话题、属于文档哪部分」,
   不是块内容的摘要。变成摘要就等于用 LLM 的复述替换了原文,
   检索到了也没法用。提示词里明确禁止复述。

三、文档太长要截断。Anthropic 原文是把整篇文档塞进去,那在长文档上
   会直接爆上下文窗口。这里做头尾截断 —— 文档的**开头**(标题/背景)
   和**结尾**(结论)对定位最有信息量,中间丢了损失最小。

关于成本
-------
每块一次 LLM 调用,一篇 100 块的文档就是 100 次。同一篇文档的所有块
共用同一段文档前缀,所以命中服务端前缀缓存能显著降本
(DeepSeek 默认开,不用配)。这也是为什么按文档分组批处理,
而不是把所有块打平了一起跑 —— 打平了前缀缓存就全失效了。
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

from config import IngestConfig, get_settings
from ingest.chunker import Chunk as TextChunk
from llm.client import LLMClient, get_llm

logger = logging.getLogger(__name__)

# 文档塞给 LLM 的字符上限。超了做头尾截断。
# 8k 上下文的模型也能安全跑;有更大窗口的模型可以调高。
DOC_CHAR_BUDGET = 24000
_HEAD_RATIO = 0.6

SYSTEM_PROMPT = """你是一个检索系统的文档预处理助手。

你会看到一篇完整文档,以及该文档中的一小段。你的任务是给这一小段写一句**定位语**,
让这句话和这段原文拼在一起之后,即使脱离全文也能被准确检索到。

定位语必须做到:
1. 点明这段所说的话题/对象,用文档里出现的**原词**,不要换说法
2. 说明这段在全文中的位置或作用(如「属于设备参数表的第三项」「是故障处理流程的第二步」)
3. 补全指代 —— 把「该公司」「上述设备」「本年」这类指代还原成文档里的具体名称
4. 用文档本身的语言书写(中文文档就写中文)

定位语**严禁**:
- 复述或概括这一小段的内容
- 引入文档中不存在的信息,不要推测、不要补充背景知识
- 超过两三句话

如果这一小段本身已经足够自明(比如它自带完整的标题和主语),
输出一句最简短的定位即可,不要硬凑。

只输出定位语本身,不要任何前缀、解释或引号。"""


def _truncate_doc(text: str, budget: int = DOC_CHAR_BUDGET) -> str:
    """头尾截断。保留了开头和结尾,中间用省略标记。"""
    if len(text) <= budget:
        return text
    head = int(budget * _HEAD_RATIO)
    tail = budget - head
    return (
        text[:head]
        + f"\n\n…(此处省略 {len(text) - budget} 字)…\n\n"
        + text[-tail:]
    )


def _build_messages(doc_text: str, chunk: TextChunk, max_chars: int) -> list[dict[str, str]]:
    user = (
        "这是一篇完整文档:\n"
        "<document>\n"
        f"{doc_text}\n"
        "</document>\n\n"
        "以下是该文档中的一小段:\n"
        "<chunk>\n"
        f"{chunk.text}\n"
        "</chunk>\n\n"
        f"请为这一段写定位语(不超过 {max_chars} 字):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_warned = False
_warn_lock = threading.Lock()


def _warn_once() -> None:
    global _warned
    with _warn_lock:
        if not _warned:
            logger.warning(
                "上下文增强已开启但 LLM 未配置,本次导入跳过定位语生成。"
                "填上 .env 里的 LLM_API_KEY 后重新导入即可补齐 —— "
                "已有数据不需要删,重导会按 content_hash 判断并覆盖。"
            )
            _warned = True


def enrich_chunks(
    doc_text: str,
    chunks: Sequence[TextChunk],
    cfg: IngestConfig | None = None,
    llm: LLMClient | None = None,
) -> list[str]:
    """给一篇文档的所有块生成定位语。返回和 chunks 等长的列表。

    单块失败返回空字符串而不是抛异常 —— 少一句定位语只是那一条检索差一点,
    整篇文档导入失败才是真的损失。失败情况会记日志。
    """
    cfg = cfg or get_settings().ingest
    llm = llm or get_llm()

    if not chunks:
        return []
    if not cfg.contextual_enabled:
        return [""] * len(chunks)
    if not llm.configured:
        _warn_once()
        return [""] * len(chunks)

    doc_view = _truncate_doc(doc_text)
    max_chars = cfg.contextual_max_chars

    def _one(ch: TextChunk) -> str:
        # 超长块单独截一下 —— 定位语只需要看个大意
        view = ch.text if len(ch.text) <= 4000 else ch.text[:4000]
        probe = TextChunk(text=view, start=ch.start, end=ch.end, index=ch.index)
        r = llm.chat(
            _build_messages(doc_view, probe, max_chars),
            temperature=0.0,
            max_tokens=min(300, max_chars * 2),
        )
        s = r.text.strip().strip('"').strip("「」")
        # 模型偶尔会不听话写一长串,硬截一下保护上下文预算
        if len(s) > max_chars:
            s = s[:max_chars].rstrip()
        return s

    results = llm.map_batch(
        _one,
        list(chunks),
        workers=cfg.contextual_workers,
        desc=f"上下文增强({len(chunks)} 块)",
    )

    out = [r or "" for r in results]
    empty = sum(1 for x in out if not x)
    if empty:
        logger.warning("有 %d/%d 块没拿到定位语", empty, len(out))
    return out
