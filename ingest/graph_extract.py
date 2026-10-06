"""中文三元组抽取:提示词 + 解析器。

为什么这两样都得自己写
--------------------
LlamaIndex 的 `SimpleLLMPathExtractor` 默认提示词是英文,而且默认解析器
`default_parse_triplets_fn` 有两处对中文/工科语料**有害且静默**的行为:

1. `tokens = triplet_part.split(",")`,紧跟 `if len(tokens) != 3: continue`
   —— 实体名里只要含一个半角逗号,这条三元组就被丢掉,不报错、不留日志。
   「GB/T 14285, 5.1 条」这种写法在中文标准文档里到处都是。

2. `entity.strip('"').capitalize()` —— 把每个实体首字母大写、其余全部小写:

       110kV    -> 110kv
       GB/T     -> Gb/t
       DL/T 741 -> Dl/t 741

   设备型号和标准号被永久改坏,而且**不可逆** —— 此后实体对齐、图查询
   都对不上原文。注意 capitalize() 对中文是空操作,所以纯中文语料看不出来,
   只有中英混排(电力、制造、医疗文档的常态)才暴露,更难发现。

这里换成:优先认 JSON(按提示词输出结构化结果最稳),退化路径按
「第一个逗号和最后一个逗号」切分 —— 这样实体内部的逗号能保住。
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

from config import get_settings
from llm.client import extract_json

logger = logging.getLogger(__name__)

# 半角 + 全角逗号。中文标点必须一起认,否则「(甲,乙,丙)」解析不出来。
_COMMA_CHARS = ",，"


EXTRACT_PROMPT = """\
从下面的文本中抽取知识三元组,用于构建知识图谱。

要求:
1. 最多抽取 {max_knowledge_triplets} 条三元组。宁可少抽,不要凑数。
2. 每条三元组是 [头实体, 关系, 尾实体]。
3. 实体必须是原文中出现的表述,**照抄原文**,不要改写、不要翻译、不要补全简称。
   型号、编号、标准号要保留原始大小写和写法(如 110kV、GB/T 14285、DL/T 741)。
4. 关系用简短的动词或动词短语,例如:包含、位于、属于、连接、导致、检修、检测。
5. 实体名里可以含逗号、斜杠、括号等符号,照写即可。
6. 不要编造原文没有的信息。抽不到就返回空数组 []。
7. 只输出 JSON 数组,不要解释、不要 markdown 代码块。

输出格式示例:
[["1号杆塔", "包含", "绝缘子"], ["绝缘子", "存在缺陷", "破损"]]

文本:
{text}
"""


def _split_three(inner: str) -> Tuple[str, str, str] | None:
    """把 `甲, 乙, 丙` 切成三段,允许甲/丙内部含逗号。

    按**第一个**和**最后一个**逗号切,而不是 split(",") 后要求恰好 3 段 ——
    后者遇到实体名里的逗号就直接丢弃整条,正是上游解析器的毛病。

    **已知局限**:逗号多于两个时,多出来的那些都会落进中间那段。也就是说
    `(甲, 关系, 乙, 丙)` 会解析成关系="关系, 乙"、尾实体="丙"。
    这种输入本身就是有歧义的,靠标点无法还原 —— 所以 JSON 才是主路径,
    这条只是模型没按格式输出时的兜底。
    """
    idx = [i for i, ch in enumerate(inner) if ch in _COMMA_CHARS]
    if len(idx) < 2:
        return None
    return (
        inner[: idx[0]].strip(),
        inner[idx[0] + 1 : idx[-1]].strip(),
        inner[idx[-1] + 1 :].strip(),
    )


def _parse_json_triplets(response: str) -> List[Tuple[str, str, str]] | None:
    """按提示词的约定解析 JSON。解析不出来返回 None,交给退化路径。"""
    try:
        data = extract_json(response)
    except ValueError:
        return None

    # 允许模型套一层 {"triplets": [...]}
    if isinstance(data, dict):
        for key in ("triplets", "triples", "结果", "三元组"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            return None

    if not isinstance(data, list):
        return None

    out: List[Tuple[str, str, str]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            s, r, o = (str(x) for x in item)
        elif isinstance(item, dict):
            # 容忍模型换键名
            keys = ("head", "relation", "tail")
            vals = None
            for cand in (
                ("头实体", "关系", "尾实体"),
                ("subject", "predicate", "object"),
                ("实体1", "关系", "实体2"),
                keys,
            ):
                if all(k in item for k in cand):
                    vals = [item[k] for k in cand]
                    break
            if vals is None:
                continue
            s, r, o = (str(x) for x in vals)
        else:
            continue
        out.append((s, r, o))
    return out


def _parse_paren_triplets(response: str) -> List[Tuple[str, str, str]]:
    """退化路径:`(甲, 乙, 丙)` 一行一条。"""
    out: List[Tuple[str, str, str]] = []
    for line in (response or "").splitlines():
        open_i, close_i = line.find("("), line.find(")")
        # 全角括号也要认
        open_f, close_f = line.find("（"), line.find("）")
        if open_f != -1 and (open_i == -1 or open_f < open_i):
            open_i, close_i = open_f, close_f
        if open_i == -1 or close_i == -1 or close_i < open_i:
            continue
        parsed = _split_three(line[open_i + 1 : close_i])
        if parsed is None:
            continue
        # 去掉包裹的引号,但**不做 capitalize** —— 那会毁掉 110kV / GB/T
        out.append(tuple(x.strip().strip("\"'") for x in parsed))
    return out


def parse_triplets(response: str, max_length: int = 128) -> List[Tuple[str, str, str]]:
    """把模型输出解析成三元组列表。签名和上游的 parse_fn 一致。

    上游解析器在解析失败时是「返回空列表」,这里保持一致 ——
    抽取失败不该让整篇文档的导入崩掉。
    """
    if not response or not response.strip():
        return []

    triples = _parse_json_triplets(response)
    if triples is None:
        triples = _parse_paren_triplets(response)

    out: List[Tuple[str, str, str]] = []
    for s, r, o in triples:
        s, r, o = s.strip(), r.strip(), o.strip()
        # 残缺三元组丢掉:图里挂一个空名字的节点比少一条边更糟
        if not s or not r or not o:
            continue
        # 上游按 UTF-8 字节数设限,一个汉字 3 字节,这里保持同样的口径
        if any(len(x.encode("utf-8")) > max_length for x in (s, r, o)):
            logger.debug("跳过过长的三元组: %r / %r / %r", s[:20], r[:20], o[:20])
            continue
        out.append((s, r, o))
    return out


def build_extractor() -> Any:
    """造一个用中文提示词 + 自定义解析器的抽取器。

    用 SimpleLLMPathExtractor 而不是 SchemaLLMPathExtractor:
    后者的类型约束是通过 structured_predict 注入的,要端点支持结构化输出
    (function calling / json_schema)。而这套要靠 LLM_API_KEY 才能验证是否可用,
    失败时表现为「一条实体都抽不出来」,排查成本高。
    类型引导改成写在提示词里,可控且不依赖端点的额外能力。
    """
    from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

    from llm.llamaindex_adapter import get_llamaindex_llm

    s = get_settings()
    extractor = SimpleLLMPathExtractor(
        # 抽取用自己的 completion 预算,不跟全局 llm.max_tokens 走 ——
        # 推理模型的思维链会先把预算吃干净,一块抽挂就带走整篇文档
        # (LLMError 是 RuntimeError,raise_on_error=False 兜不住)。详见 config.py
        # 里 extract_max_tokens 的注释与实测数字。
        llm=get_llamaindex_llm(default_max_tokens=s.graphrag.extract_max_tokens),
        extract_prompt=EXTRACT_PROMPT,
        parse_fn=parse_triplets,
        max_paths_per_chunk=s.graphrag.max_paths_per_chunk,
        num_workers=s.graphrag.extraction_workers,
        # 单个 chunk 抽取失败(模型抽风/超时)不该让整批导入挂掉。
        # 代价是这一块的实体没了 —— 所以下面在入库统计里会体现出来。
        raise_on_error=False,
    )
    return extractor


__all__ = ["EXTRACT_PROMPT", "parse_triplets", "build_extractor"]
