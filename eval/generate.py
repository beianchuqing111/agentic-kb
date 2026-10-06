"""LLM 出题引擎 —— 给评估集出**初稿**,人再筛。

这个模块的产出永远是 `status: "draft"`。它不是评估集,是**待筛的候选**。
`eval/qa/seed.draft.jsonl` 由它生成、可以随手重生、不手工编辑;
人挑过的那些才进 `seed.jsonl`。把这两者混起来,「评估集」就会退化成
「模型说自己会答的题」,而那种题必然全对。

四个刻意的设计决定
----------------
一、**生成器绝不输出 `chunk_index`。**
   锚点是 LLM 唯一会幻觉、而这套设施唯一依赖的字段。块是**框架**选的,
   就由框架附上 `(file, chunk_index)`。模型看到的片段带了框架给的标签
   (`c1`/`c2`…),它只需要回答「这道题问的是哪个片段」——标签认不出来就丢弃该题。
   模型没法用幻觉伪造一个合法标签,于是整类失败被白送般消掉。

二、**按文档分组调用,而不是把所有块打平。**
   同一篇文档的所有片段共用同一段文档前缀,服务端前缀缓存能显著降本。
   打平了前缀缓存就全失效了。这条是照搬 `ingest/contextual.py` 的结构决策。

三、**反例单独一遍、按文档问,而不是针对某块问。**
   「针对某块出反例」是自相矛盾的 —— 反例的定义就是没有目标块。
   所以第二遍的提问单位是文档:「写 1~2 个本文档**答不了**但听起来合理的问题」,
   办法是拿本文档真实出现的实体/参数,配一个本文档没给的取值。

四、**抽样只遍历排序后的列表,绝不遍历集合**,且用 `random.Random(seed)`。
   重跑必须取到同一批样本 —— 否则每次生成都白烧一遍 API 调用,
   而且两次生成的题无法比较。

关于「自证循环」
-------------
生成器只能出「**它给的那个片段**答得上」的题,它**无法知道那个片段是唯一答案**。
两块都答得上的情况要靠人工把 `expected` 加宽 —— 这正是人在筛什么。

**不要**为了省事把检索 top 结果摆给人点选。用检索器自己的输出定 gold
会让评估循环自证,分数永远 ~1.0,整个设施就废了。
`eval/corpus.py:locate_text` 是纯文本查找、刻意不查检索器,理由同此。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from config import BASE_DIR, get_settings
from eval import corpus as C
from eval.schema import EvalItem, EvalSet, GoldChunk, dump_qa, make_meta
from ingest import chunk_text, discover, load_many
from ingest.loader import Document
from llm.client import LLMClient, LLMError, get_llm

logger = logging.getLogger(__name__)

# 短于这个字数的块不值得出题:多半是标题、目录行、表格剩行。
# 有标题(section 非空)的块优先 —— 有主题的块才出得了好题。
MIN_CHUNK_CHARS = 80
# 一次调用最多要几道题。多了模型会开始互相抄袭措辞。
MAX_Q_PER_CALL = 6
# 塞给模型的正文预算,与 contextual.py 同一个口径与理由(头尾截断)。
DOC_CHAR_BUDGET = 24000
_HEAD_RATIO = 0.6
# 超过这个调用数就要求 --yes。380 篇语料全量出题是真金白银。
CONFIRM_ABOVE_CALLS = 20

KNOWN_GEN_TYPES = ("lexical", "semantic", "multi_hop", "contextual", "ambiguous")

QUESTION_SYSTEM = """你是一个检索评测集的出题助手。

你会看到一篇文档,以及该文档中的若干片段(用 c1/c2/… 标记)。你的任务是给每个片段出一道**用户真的会这么问**的问题,使得:光凭这个问题去这个知识库检索,该片段应当是被召回的目标之一。

出题要求:
1. 用文档里出现的**原词**,不要自己造一套同义说法 —— 但也不要原样抄一整个句子当问题。
   抄句子会让词法检索白送命中,测不出语义能力;完全换词又会让题目不公平。
2. 问题必须能被所给片段**独立回答**,不依赖其他片段。
3. 逐题给 type,从下面五种里选:
   - lexical:答案的关键词在片段里字面出现(主要考稀疏/词法检索)
   - semantic:问的是意思,措辞与片段差异较大(主要考稠密检索)
   - contextual:问题里含「该系统」「上述参数」这类指代,必须靠定位语才能对上片段
   - multi_hop:答案要结合文档里不止一处才能得出
   - ambiguous:文档里有多个相似对象,问题需要人判断指的是哪一个
4. 逐题给 answer:该片段给出的答案本身。它**不参与评分**,只是让人筛题时
   不必回头翻原文。片段里没给答案就换一道题 —— 不要编。
5. 逐题给 difficulty(1~3):1=几乎照抄,2=需要理解,3=要绕一下。

严禁:
- 严禁编造文档里不存在的事实、参数、条文号
- 严禁输出 chunk_index、块号、页码之类的定位信息 —— 你只需要说这道题问的是哪个标签
- 问题里不要出现「根据上文」「文中提到」「该片段」这类指代本文档的说法:
  真实用户提问时并不知道你手里有片段表

只输出 JSON。"""

NEGATIVE_SYSTEM = """你要为一份检索评测集出**反例题**(负样本)。

反例题是「在这个知识库里查不到答案」的问题。它的价值全在于**可信**:
一道其实答得上的「反例」会永远伪装成一次检索失败,把人的注意力引到一个没坏的地方。

出一个好反例的唯一办法:拿本文档里**真实出现**的实体、参数、设备名,
去问一个本文档**没有给出**的取值或属性。比如文档讲了某设备在 A 条件下的参数,
就问它在另一个本文档没提过的条件下的参数。

要求:
1. 问题必须**听起来非常合理** —— 像一个真的会用这个知识库的人会问的
2. 问题里的对象**必须来自本文档**(这样它才落在同一条语义轨道上,
   嵌入式检索器才会真的被它骗到;否则它一望即分,等于没测)
3. 但答案必须是本文档没有写的
4. 严禁问与文档领域无关的噪声(「今天天气怎么样」对着电力语料,
   无论检索多烂都会满分)
5. note 里写清「本文档为什么答不了它」—— 这句是给人筛题用的

只输出 JSON。"""


# --------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------- #


@dataclass
class Fragment:
    """一个待出题的片段。`label` 是**框架给**的,不是模型给的。"""

    label: str          # "c1"
    file: str           # 语料内相对路径(正斜杠)
    chunk_index: int
    text: str
    section: str = ""

    @property
    def is_negative(self) -> bool:
        return False


@dataclass
class DocJob:
    """一篇文档 + 它被抽中的片段。"""

    doc: Document
    file: str
    fragments: list[Fragment] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.doc.title or Path(self.file).stem


# --------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------- #


def _fragment_pool(
    doc: Document, file: str, *, chunk_size: int, chunk_overlap: int
) -> list[Fragment]:
    """一篇文档里**值得出题**的块。

    分块用的是导入时那套 `chunk_text` 和同一组 `chunk_size` / `chunk_overlap` ——
    这不是可选项。生成时报的位置必须就是导入时 pipeline 会分配的索引,
    参数不一致会产出随机偏移的锚点,症状和「检索变差了」一模一样,
    而排查方向会完全跑偏。
    """
    good: list[Fragment] = []
    for ch in chunk_text(doc.text, chunk_size, chunk_overlap):
        t = ch.text.strip()
        if len(t) < MIN_CHUNK_CHARS:
            continue
        # 整块就是一行标题的,出不了题
        if t.count("\n") == 0 and len(t) < 60:
            continue
        good.append(
            Fragment(
                label="",  # 抽中之后再编号
                file=file,
                chunk_index=ch.index,
                text=t,
                section=getattr(ch, "section", "") or "",
            )
        )
    # 有 section 的排前面:有主题的块更容易出好题
    good.sort(key=lambda f: (not f.section, f.chunk_index))
    return good


def sample_jobs(
    docs: Sequence[tuple[str, Document]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    per_doc: int,
    max_items: int,
    seed: int,
) -> list[DocJob]:
    """按排序后的文档列表**轮转**抽样,每篇最多 `per_doc` 个片段。

    轮转意味着语料里的每一篇都有机会被抽到,而不是把前面几篇抽干 ——
    评估集要覆盖整份语料,不然它测的是子集,分数没法代表真实检索质量。
    """
    rng = random.Random(seed)
    pools: list[tuple[str, Document, list[Fragment]]] = []
    for file, doc in docs:
        pool = _fragment_pool(doc, file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if pool:
            pools.append((file, doc, pool))

    # 每篇先打乱自己的池子,再逐轮取 —— 等价于「每篇随机取 per_doc 个」,
    # 但保持了对同一 seed 的可复现性。
    for _file, _doc, pool in pools:
        rng.shuffle(pool)

    jobs: list[DocJob] = []
    taken = 0
    for round_i in range(max(0, per_doc)):
        for file, doc, pool in pools:
            if taken >= max_items:
                break
            if round_i >= len(pool):
                continue
            frag = pool[round_i]
            job = next((j for j in jobs if j.file == file), None)
            if job is None:
                job = DocJob(doc=doc, file=file)
                jobs.append(job)
            job.fragments.append(frag)
            taken += 1
        if taken >= max_items:
            break

    # 统一编号:模型只会看到 c1/c2/…,认得出来才准题,认不出来就丢题
    for job in jobs:
        job.fragments.sort(key=lambda f: f.chunk_index)
        for i, frag in enumerate(job.fragments, 1):
            frag.label = f"c{i}"

    # 只留真的抽到片段的文档,且按文件名排序 —— 调用顺序可复现
    jobs = [j for j in jobs if j.fragments]
    jobs.sort(key=lambda j: j.file)
    return jobs


def estimate_calls(jobs: Sequence[DocJob], *, negatives_per_doc: int) -> int:
    """预计调用数。**每篇文档** 1 次出题 + 1 次出反例。"""
    if not jobs:
        return 0
    return len(jobs) + (len(jobs) if negatives_per_doc > 0 else 0)


# --------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------- #


def _truncate_doc(text: str, budget: int = DOC_CHAR_BUDGET) -> str:
    """头尾截断。

    这是 `ingest/contextual.py:_truncate_doc` 的**分叉**(照抄而非 import):
    那边是导入链路的一部分,这边是出题链路,改一边不该动另一边。
    文档的开头(标题/背景)和结尾(结论)对定位最有信息量,中间丢了损失最小。
    """
    if len(text) <= budget:
        return text
    head = int(budget * _HEAD_RATIO)
    tail = budget - head
    return (
        text[:head]
        + f"\n\n…(此处省略 {len(text) - budget} 字)…\n\n"
        + text[-tail:]
    )


def _question_messages(job: DocJob) -> list[dict[str, str]]:
    blocks = "\n\n".join(
        f"<片段 {f.label}>\n{f.text}\n</片段 {f.label}>" for f in job.fragments
    )
    user = (
        f"文档标题:《{job.title}》\n文件名:{job.file}\n\n"
        "文档全文:\n<document>\n"
        f"{_truncate_doc(job.doc.text)}\n"
        "</document>\n\n"
        f"本次要出题的片段:\n{blocks}\n\n"
        "请给上面每一个片段各出 1 道题。只输出 JSON,形如:\n"
        '{"items":[{"fragment":"c1","question":"…","answer":"…",'
        '"type":"lexical","difficulty":2}]}'
    )
    return [
        {"role": "system", "content": QUESTION_SYSTEM},
        {"role": "user", "content": user},
    ]


def _negative_messages(job: DocJob) -> list[dict[str, str]]:
    user = (
        f"文档标题:《{job.title}》\n文件名:{job.file}\n\n"
        "文档全文:\n<document>\n"
        f"{_truncate_doc(job.doc.text)}\n"
        "</document>\n\n"
        "请写 1~2 个本文档**答不了**、但听起来非常合理的问题。只输出 JSON,形如:\n"
        '{"items":[{"question":"…","note":"本文档未提及 …"}]}'
    )
    return [
        {"role": "system", "content": NEGATIVE_SYSTEM},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------- #
# 解析(模型的输出不可信,一律当作待校验的外部输入)
# --------------------------------------------------------------------- #


def _extract_items(raw: Any) -> list[dict[str, Any]]:
    """从模型返回里挖出 items 列表,容错到「能用就用,不能用就丢」。"""
    if isinstance(raw, dict):
        items = raw.get("items") or raw.get("questions") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    return [x for x in items if isinstance(x, dict)]


def _clean_question(s: Any) -> str:
    """问题文本归一化。文档本身不该被写进问题里 —— 用户提问时看不到片段表。"""
    q = str(s or "").strip().strip('"').strip("「」")
    q = " ".join(q.split())
    for bad in ("根据上文", "文中提到", "该片段", "上述片段", "本文档中"):
        q = q.replace(bad, "")
    return q.strip()


def parse_questions(
    raw: Any, job: DocJob
) -> list[tuple[Fragment, str, str, str, int]]:
    """解析出题返回 → `[(片段, 问题, 答案, type, difficulty)]`。

    **认不出的 `fragment` 标签直接丢弃**。这是整套「模型不碰 chunk_index」
    方案的最后一道闸:模型只能回答「问的是 c1 还是 c2」,
    而 c1/c2 是我们自己编的号;它编不出一个我们没给过的合法标签。
    """
    by_label = {f.label: f for f in job.fragments}
    out: list[tuple[Fragment, str, str, str, int]] = []
    for it in _extract_items(raw):
        label = str(it.get("fragment") or it.get("chunk") or "").strip().lower()
        frag = by_label.get(label)
        if frag is None:
            logger.warning("%s: 丢弃一条 —— 认不出的片段标签 %r", job.file, label)
            continue
        q = _clean_question(it.get("question"))
        if len(q) < 6:
            logger.warning("%s: 丢弃一条 —— 问题过短 %r", job.file, q)
            continue
        qtype = str(it.get("type") or "").strip().lower()
        if qtype not in KNOWN_GEN_TYPES:
            # 不丢题,降级成 semantic 并记一笔:题型错比题丢了损失小
            logger.warning("%s: 未知题型 %r,降级为 semantic", job.file, qtype)
            qtype = "semantic"
        try:
            diff = int(it.get("difficulty") or 2)
        except (TypeError, ValueError):
            diff = 2
        diff = min(3, max(1, diff))
        out.append((frag, q, str(it.get("answer") or "").strip(), qtype, diff))
    return out


def parse_negatives(raw: Any) -> list[tuple[str, str]]:
    """解析反例返回 → `[(问题, note)]`。反例没有片段,所以没标签可对。"""
    out: list[tuple[str, str]] = []
    for it in _extract_items(raw):
        q = _clean_question(it.get("question"))
        if len(q) < 6:
            continue
        out.append((q, str(it.get("note") or "").strip()))
    return out


# --------------------------------------------------------------------- #
# 组装成 EvalItem
# --------------------------------------------------------------------- #


def _make_item(
    *,
    item_id: str,
    question: str,
    item_type: str,
    golds: list[GoldChunk],
    answer: str,
    note: str,
    model: str,
    today: str,
) -> EvalItem:
    """**一律 `status="draft"`。**

    生成器产出的东西没有一条可以直接进评分:题目公不公平、
    `expected` 是不是该加宽、反例是不是真答不上,都得人看过。
    默认 draft 让「没筛过」成为默认状态,而不是靠人记得去改。
    """
    return EvalItem(
        id=item_id,
        question=question,
        expected=golds,
        type=item_type,
        status="draft",
        answer=answer,
        note=note,
        added_by=f"eval_gen:{model}",
        added_at=today,
    )


def _snippet_of(frag: Fragment, n: int) -> str:
    from eval.schema import SNIPPET_CHARS

    return frag.text[: min(n, SNIPPET_CHARS)]


# --------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------- #


def plan(
    *,
    corpus: str | Path,
    settings,
    per_doc: int,
    max_items: int,
    seed: int,
    limit: int = 0,
) -> tuple[list[DocJob], list[tuple[str, str]], dict[tuple[str, int], Fragment]]:
    """读语料、算抽样。**零 LLM 调用** —— `--dry-run` 和真跑走的是同一条路。

    `limit` 在**抽样之前**截断文档列表。放在抽样之后是错的:
    轮转抽样是**广度优先**的(先给每篇分 1 个片段,再回头给第 2 个),
    所以 `--max 40 --per-doc 2` 会变成「40 篇各 1 个片段」而不是
    「20 篇各 2 个」。此时再 `--limit 30` 截断,你以为在跑
    「前 30 篇、每篇 2 题」,实际拿到的是「前 30 篇、每篇 1 题」——
    而且另外 10 篇的抽样白算了。先截断就没有这个错位。
    """
    root = C.resolve_corpus(corpus)
    chunk_size = settings.ingest.chunk_size
    chunk_overlap = settings.ingest.chunk_overlap

    paths = discover(root)
    if limit:
        paths = paths[:limit]
    docs, errors = load_many(paths, on_error="skip")
    # 相对路径的口径必须与 corpus.py 一致,否则生成的锚点会指向一个
    # 语料里"不存在"的文件,而 runner 会正确地把它判成锚点失效。
    pairs = [(C._rel_key(root, d.source), d) for d in docs]

    jobs = sample_jobs(
        pairs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        per_doc=per_doc,
        max_items=max_items,
        seed=seed,
    )
    index: dict[tuple[str, int], Fragment] = {
        (f.file, f.chunk_index): f for j in jobs for f in j.fragments
    }
    return jobs, errors, index


def run_generation(
    *,
    jobs: Sequence[DocJob],
    llm: LLMClient,
    settings,
    negatives_per_doc: int,
    workers: int,
) -> tuple[list[EvalItem], list[EvalItem]]:
    """两遍生成。返回 `(可答题, 反例题)`,顺序与 jobs 一致。"""
    model = settings.llm.model

    # 预算必须给足 —— 这是**实测踩过的坑**,不是保守估计。
    #
    # 出反例那侧原来给 1024,结果两篇文档全灭,而且是**两种不同的死法**:
    #   一、思维链把 1024 全吃了 → content="" 但调用"成功"
    #       (client.py 有 LLMError 守卫把它拦下来了,不然就是一条空题静默入库)
    #   二、JSON 写到一半截断 → extract_json 解析失败
    # 两种都是「预算小于推理模型的 reasoning_tokens」,不是模型不会做。
    # reasoning 计入 max_tokens 但不进 content,所以按题量给足是唯一稳妥做法。
    n_frag = max((len(j.fragments) for j in jobs), default=1)
    q_tokens = 2048 + 220 * n_frag
    n_tokens = 2048 + 220 * max(2, negatives_per_doc)

    def _one_doc(job: DocJob) -> list[tuple[Fragment, str, str, str, int]]:
        r = llm.chat_json(
            _question_messages(job), temperature=0.4, max_tokens=q_tokens
        )
        return parse_questions(r, job)

    raw_pos = llm.map_batch(
        _one_doc, list(jobs), workers=workers, desc=f"出题({len(jobs)} 篇)"
    )

    answerable: list[EvalItem] = []
    for i, (job, parsed) in enumerate(zip(jobs, raw_pos), 1):
        if not parsed:
            logger.warning("%s: 一道题都没出出来", job.file)
            continue
        for j, (frag, q, ans, qtype, diff) in enumerate(parsed, 1):
            answerable.append(
                _make_item(
                    item_id=f"gen-a{i:03d}-{j}",
                    question=q,
                    item_type=qtype,
                    golds=[
                        GoldChunk(
                            file=frag.file,
                            chunk_index=frag.chunk_index,
                            grade=3 if qtype != "multi_hop" else 2,
                            snippet=_snippet_of(frag, 30),
                        )
                    ],
                    answer=ans,
                    note=f"{(frag.section + ' / ') if frag.section else ''}"
                    f"难度 {diff};由片段 {frag.label} 出题",
                    model=model,
                    today=_today(),
                )
            )

    negatives: list[EvalItem] = []
    if negatives_per_doc > 0 and jobs:
        def _one_neg(job: DocJob) -> list[tuple[str, str]]:
            r = llm.chat_json(
                _negative_messages(job),
                temperature=0.7,  # 反例要新鲜,温度高一点;但别高到逻辑乱
                max_tokens=n_tokens,
            )
            return parse_negatives(r)

        raw_neg = llm.map_batch(
            _one_neg, list(jobs), workers=workers, desc=f"出反例({len(jobs)} 篇)"
        )
        for i, (job, parsed) in enumerate(zip(jobs, raw_neg), 1):
            for j, (q, note) in enumerate((parsed or [])[:negatives_per_doc], 1):
                negatives.append(
                    _make_item(
                        item_id=f"gen-n{i:03d}-{j}",
                        question=q,
                        item_type="negative",
                        golds=[],
                        answer="",
                        note=(note or "本文档未提及")
                        + ";**反例必须人工核验** —— 它其实是答案题的话,"
                        "会永远伪装成检索失败",
                        model=model,
                        today=_today(),
                    )
                )

    return answerable, negatives


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _print_usage(llm: LLMClient) -> None:
    u = llm.usage
    print(
        f"\nLLM 用量:调用 {u.get('calls', 0)} 次,"
        f"prompt {u.get('prompt', 0)} tokens,"
        f"completion {u.get('completion', 0)} tokens,"
        f"reasoning {u.get('reasoning', 0)} tokens"
    )
    if u.get("reasoning"):
        print("  (reasoning 单独记一笔:推理模型的思维链也计费,但不出现在 content 里 —— 账单才对得上)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval_gen.py",
        description="给评估集出初稿(会花 LLM 配额)。产出 status=draft,人工筛完才进定稿。",
    )
    ap.add_argument("--corpus", default=None, help="语料目录(默认 eval/corpus/seed)")
    ap.add_argument("--out", default=None, help="输出文件(默认 eval/qa/<slug>.draft.jsonl)")
    ap.add_argument(
        "--per-doc",
        type=int,
        default=3,
        help=f"每篇文档抽几个片段出题(默认 3,上限 {MAX_Q_PER_CALL})",
    )
    ap.add_argument("--max", dest="max_items", type=int, default=40, help="总题数上限(默认 40)")
    ap.add_argument("--negatives-per-doc", type=int, default=1, help="每篇出几个反例,0=不出(默认 1)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇文档(0=不限)")
    ap.add_argument("--seed", type=int, default=17, help="抽样随机种子(默认 17,保证重跑同一批)")
    ap.add_argument("--workers", type=int, default=4, help="并发调用数(默认 4;调大容易撞 429)")
    ap.add_argument("--model", default=None, help="覆盖 LLM_MODEL")
    ap.add_argument("--yes", action="store_true", help=f"调用数超过 {CONFIRM_ABOVE_CALLS} 次时确认")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划,零调用")
    args = ap.parse_args(argv)

    # 一次调用要的题多了,模型会开始互相抄措辞,题与题撞车。
    # 而且 token 预算是按题量算的 —— 放开 per_doc 会让「预算按 n 道题给」
    # 与「实际要 n 道题」脱钩,又回到截断那条老路。
    if args.per_doc > MAX_Q_PER_CALL:
        print(
            f"--per-doc {args.per_doc} 超过单次调用的上限 {MAX_Q_PER_CALL}"
            f"(一次要太多题会让模型抄自己),按 {MAX_Q_PER_CALL} 处理。",
            file=sys.stderr,
        )
        args.per_doc = MAX_Q_PER_CALL

    settings = get_settings()
    if args.model:
        import dataclasses

        settings = dataclasses.replace(
            settings, llm=dataclasses.replace(settings.llm, model=args.model)
        )

    root = C.resolve_corpus(args.corpus)
    slug = C.corpus_slug(root)
    out = Path(args.out) if args.out else BASE_DIR / "eval" / "qa" / f"{slug}.draft.jsonl"

    jobs, errors, _index = plan(
        corpus=root,
        settings=settings,
        per_doc=args.per_doc,
        max_items=args.max_items,
        seed=args.seed,
        limit=args.limit,
    )

    n_frag = sum(len(j.fragments) for j in jobs)
    calls = estimate_calls(jobs, negatives_per_doc=args.negatives_per_doc)

    print("=" * 70)
    print("评估集出题计划")
    print("=" * 70)
    print(f"语料      {root}")
    print(f"输出      {out}")
    print(f"分块参数  chunk_size={settings.ingest.chunk_size} "
          f"chunk_overlap={settings.ingest.chunk_overlap}  (必须与导入时一致)")
    print(f"模型      {settings.llm.model}")
    print(f"文档      {len(jobs)} 篇参与出题"
          + (f",{len(errors)} 篇读失败" if errors else ""))
    print(f"片段      {n_frag} 个 → 预计 {n_frag} 道可答题")
    print(f"反例      每篇 {args.negatives_per_doc} 个 → 预计 {len(jobs) * args.negatives_per_doc} 道")
    print(f"调用      **{calls} 次**")
    for _p, err in errors[:5]:
        print(f"  ⚠ 读失败: {err}")

    if not jobs:
        print("\n没有抽到任何片段 —— 语料目录是不是空的?或者文档都短到每块不足 "
              f"{MIN_CHUNK_CHARS} 字?")
        return 1

    if args.dry_run:
        print("\n--dry-run:一次调用都没发。上面就是真实计划,去掉 --dry-run 即执行。")
        return 0

    if not settings.llm.configured:
        print("\nLLM 没配置(.env 里的 LLM_API_KEY)—— 出题要调模型,先配上。", file=sys.stderr)
        return 1

    if calls > CONFIRM_ABOVE_CALLS and not args.yes:
        print(
            f"\n预计 {calls} 次调用,超过 {CONFIRM_ABOVE_CALLS} 次的确认线。"
            "确要认真跑就加 --yes;想先小规模试就调小 --per-doc / --limit / --max。",
            file=sys.stderr,
        )
        return 1

    llm = get_llm(settings.llm)
    try:
        answerable, negatives = run_generation(
            jobs=jobs,
            llm=llm,
            settings=settings,
            negatives_per_doc=args.negatives_per_doc,
            workers=args.workers,
        )
    except LLMError as exc:
        print(f"\nLLM 调用失败:{exc}", file=sys.stderr)
        _print_usage(llm)
        return 1

    es = EvalSet(
        items=answerable + negatives,
        meta=make_meta(
            corpus=slug,
            chunk_size=settings.ingest.chunk_size,
            chunk_overlap=settings.ingest.chunk_overlap,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        ),
    )
    dump_qa(
        es,
        out,
        header=[
            "由 scripts/eval_gen.py 生成,**不要手改这个文件**(下次生成会覆盖)。",
            "筛题办法:把要保留的行复制进定稿文件(如 eval/qa/seed.jsonl),",
            "把 status 从 draft 改成 keep,补齐/收窄 expected,再改掉 id。",
            "反例必须人工核验 —— 一条其实答得上的「反例」会永远伪装成检索失败。",
        ],
    )

    print(f"\n写出 {len(es.items)} 条({len(answerable)} 可答 / {len(negatives)} 反例)→ {out}")
    for w in es.warnings()[:10]:
        print(f"  ⚠ {w}")
    _print_usage(llm)
    print(
        "\n下一步:人工筛。**不要**直接把 draft 当评估集用 ——\n"
        "  生成器只保证「它给的那个片段答得上」,不保证那是唯一答案;\n"
        "  两块都答得上的要把 expected 加宽。"
    )
    return 0


__all__ = [
    "DocJob",
    "Fragment",
    "estimate_calls",
    "main",
    "parse_negatives",
    "parse_questions",
    "plan",
    "run_generation",
    "sample_jobs",
]


if __name__ == "__main__":
    raise SystemExit(main())
