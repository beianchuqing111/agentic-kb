"""检索质量评估的自检 —— **不花 LLM 配额**,大部分逻辑离线可跑。

跑法:
    set PYTHONIOENCODING=utf-8
    python scripts\\check_eval.py
    python scripts\\check_eval.py --no-qdrant    # 只跑纯计算部分(不连 Qdrant、不加载模型)
    python scripts\\check_eval.py --no-models    # 跳过需要 2.3G 重排器的断言

这个脚本要回答的不是「评估能不能跑通」,而是**「这套设施能不能区分好坏」**。
一个永远打印漂亮数字的评估集比没有评估集更糟 —— 它会让人放心地往错的方向调参。
所以下面每条断言都在试着把设施**弄坏**,看它会不会响:

  A1 指标对拍手算值                       —— 「我们的实现是对的」这条腿
  A2 与 LlamaIndex 的 HitRate / MRR / 二值 NDCG 交叉验证,
     并**现场演示** `Precision` 为什么虚高(而不是只在注释里声称)
  A3 加载器容错与报错定位:空行 / `#` 注释 / drop / 未知 type 能读,
     一行坏 JSON 要抛出**行号与原文**
  A4 锚点校验必须**响亮失败**:缺文件 / 越界 / 内容位移,
     且必须在评分之前中止(打印了聚合就说明拦晚了)
  A5 语料体检:种子集的 max(chunk_index) >= 2(否则锚点方案等于没测)、
     引用的文件都在
  A6 端到端(一次性小语料 + 真 runner):
       - 可答题与反例分族;只改反例的问题文本,可答题指标**逐位不变**
       - 基线不可比的两条路都要拦住(qa.sha1 与 配置变体)
       - `fusion_top_k` 是真的封顶:超过的 k 从报告里**消失并被announce**,
         而不是打成一个看着像「全错」的 0
  A7 防循环:种子集的 hit@1 落在宽区间内 —— 满 1.0 说明题太简单或 gold 自证
  A8 重排器在位时:开/关重排必须挪动 mrr,阈值必须挪动反例 leak
  A9 收尾:一次性 collection 与临时目录都删掉,并再次断言正式库未被创建

**为什么不删种子语料的 collection**:它是一次性数据之外的东西 ——
`agentic_kb_eval_seed_noctx_*` 是可复用的工作库,`--purge` 会顺手毁掉
你手上基线的配套数据。一次性小语料那份照 house style 全清。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BASE_DIR, get_settings  # noqa: E402
from eval import corpus as C  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import report as R  # noqa: E402
from eval import runner as RUN  # noqa: E402
from eval.schema import (  # noqa: E402
    EvalSet,
    EvalSetError,
    GoldChunk,
    load_qa,
)

# 一次性数据的落脚点。固定名字而不是 tempfile:自检中途崩掉时,
# 残留物要能被人一眼认出来该删什么。
SELFTEST_DIR = BASE_DIR / "eval" / "_selftest"
SEED_CORPUS = BASE_DIR / "eval" / "corpus" / "seed"
SEED_QA = BASE_DIR / "eval" / "qa" / "seed.jsonl"

SEED_COLLECTION_PREFIX = "agentic_kb_eval_seed_"


# --------------------------------------------------------------------- #
# 报告器
# --------------------------------------------------------------------- #


class Checker:
    """收集失败项,让整轮跑完再一起报 —— 一条坏了不影响看其余的。"""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.n_checks = 0

    def section(self, title: str) -> None:
        print(f"\n{'═' * 70}\n{title}\n{'═' * 70}")

    def check(self, cond: bool, label: str, detail: str = "") -> bool:
        self.n_checks += 1
        mark = "✅" if cond else "❌"
        line = f"  {mark} {label}"
        print(f"{line}\n       {detail}" if detail else line)
        if not cond:
            self.failures.append(f"{label}{'  —— ' + detail if detail else ''}")
        return bool(cond)

    def info(self, text: str) -> None:
        print(f"     {text}")

    def crash(self, label: str, exc: BaseException) -> None:
        self.n_checks += 1
        print(f"  ❌ {label} —— 抛异常了")
        print("       " + "".join(traceback.format_exception_only(type(exc), exc)).strip())
        self.failures.append(f"{label} 抛出异常:{exc!r}")


# --------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------- #


def collection_names(url: str) -> list[str]:
    """直接问 Qdrant 有哪些 collection。**只读**,不构造任何 store。

    刻意不走 `QdrantStore` —— 为了确认「正式库不存在」而先构造一个指向正式库的
    store,本身就是不该做的事。一个 GET 就够了。
    """
    with urllib.request.urlopen(f"{url.rstrip('/')}/collections", timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [c["name"] for c in payload.get("result", {}).get("collections", [])]


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    """在**本进程内**调 `runner.main` 并把两路输出收下来。

    不复用子进程:`get_embedder` / `get_reranker` 是模块单例,进程内跑能让
    bge-m3(2.2G)与重排器(2.3G)只加载一次。代价是重排器的单例语义在这里
    也是真的 —— 所以本脚本只切 `rerank_enabled` / `rerank_min_score`
    (前者是 `retrieve()` 的参数、后者由 `apply_rerank` 从检索器 cfg 读),
    **绝不切** `rerank_model`,那正是 `INERT_IN_PROCESS` 点名的坑。
    """
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = RUN.main(argv)
    return code, out.getvalue(), err.getvalue()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hit_of(payload: dict, spec: str, k: int) -> float:
    return float(payload["runs"][spec]["aggregates"][str(k)]["hit"])


def ranks_of(payload: dict, spec: str) -> dict[str, int | None]:
    return {d["item_id"]: d["rank"] for d in payload["runs"][spec]["per_item"]}


def _finite(rank: int | None) -> float:
    """None(没召回)→ 无穷大,好让「名次不得变差」这类比较能一行写完。"""
    return float("inf") if rank is None else rank


# --------------------------------------------------------------------- #
# 一次性语料
# --------------------------------------------------------------------- #

# 三篇**与种子语料不同领域**的文档,每篇约 1700 字。硬要求是每篇都得
# 分出 3 块以上:chunk_size=512 / overlap=64 时步长 448,不到 900 字就只有
# 2 块、`chunk_index` 最大是 1,`out_of_range` 那条断言就成了空转。
#
# 段末的 `【蓄-3】` 这类标记是给 `locate_text` 用的锚:断言里不写死
# chunk_index,而是开跑前现算 —— 写死的话,以后改一个字就会被判成
# 「检索退步」,而那只是夹具自己做旧了。
DOCS: dict[str, str] = {
    "01_直流系统蓄电池组运行维护.md": """# 直流系统蓄电池组运行维护

## 浮充运行

阀控式密封铅酸蓄电池组在变电站直流系统中承担事故放电任务,正常运行时长期处于浮充状态。浮充电压一般取每只 2.23 至 2.27 伏(25 摄氏度基准),温度偏离基准值时应当按照厂家给出的温度补偿曲线逐只修正,否则会出现欠充或者过充,这两种状态都会明显缩短电池寿命。【蓄-1】浮充状态下单体电压的离散度应当小于 50 毫伏,超过该值即判为一致性不合格,应当安排均充而不是直接更换。

## 均充

均充是针对单体电压离散度超标而进行的恢复性充电,均充电压一般取每只 2.30 至 2.35 伏,单次持续时间不超过十小时。【蓄-2】均充期间应当每小时记录一次单体电压与温度,任意单体温度超过 45 摄氏度时立即转回浮充并查明原因,不允许在无人监视的情况下长时间均充。均充结束后应当静置两小时再测电压,刚断电时的读数受极化影响偏虚。

## 核对性放电

核对性放电试验每两到三年进行一次,放电容量低于额定容量的百分之八十即判定为不合格。【蓄-3】判定不合格的蓄电池组应当整组更换,不允许只更换个别落后单体 —— 新旧单体的内阻差异会使新单体长期处于过充状态,往往在半年之内就被拖坏,反而比不换更花钱。

## 内阻测试

内阻测试是比端电压灵敏得多的劣化指标,单体内阻超过出厂基准值百分之五十时,即使当时的放电容量尚可,也应当列入更换计划。【蓄-4】内阻测试应当在电池充满并静置两小时后进行,测试夹要夹在极柱根部,接触电阻会直接叠加进测量结果,而它的量级和真实内阻相当。

## 直流母线绝缘

直流母线的绝缘监察装置应当每班检查一次,正负极对地绝缘电阻低于 0.5 兆欧时装置发出告警,值班人员应当立即查找接地点。【蓄-5】查找过程中不允许采取轮流拉路以外的办法,且拉路之前必须确认该回路不承载保护装置的直流电源,否则保护会在最需要它的时候失电。
""",
    "02_电缆沟与电缆通道巡视.md": """# 电缆沟与电缆通道巡视

## 积水与排水

电缆沟巡视的重点是积水、支架锈蚀和外护套破损。沟内积水深度超过 100 毫米时应当立即安排排水,长时间浸泡会使外护套上原本无害的微小缺陷发展为进水受潮。【缆-1】排水之后还应当检查沟底的排水坡度,坡度不足是反复积水的根本原因,只抽水不修坡度,等于每个雨季都要把同一件事重做一遍。

## 支架锈蚀

支架锈蚀以镀锌层的剥落面积为判据,剥落面积超过支架表面积的百分之三十时应当除锈并补涂防锈漆,超过百分之五十时更换支架。【缆-2】更换支架时应当先用临时支撑把该段电缆托住,不允许让电缆长时间悬空 —— 电缆自身的重量会造成金属护套疲劳,而护套一旦出现疲劳裂纹,进水就是时间问题。

## 外护套破损

外护套破损的定位通常先用绝缘电阻测试把范围缩小,再用跨步电压法或者声磁同步法精确定点。【缆-3】定点误差应当小于 0.5 米,因为开挖修复的成本与开挖长度直接相关,定位不准会让修复长度成倍增加,在城市道路上还要额外承担路面恢复的费用。

## 接头测温

电缆接头是整条线路最薄弱的环节,红外测温应当在负荷高峰期进行,接头表面温差超过 5 开尔文即列入重点监测。【缆-4】温差在同一接头的不同相之间比较,比在同一时刻的不同接头之间比较更灵敏,因为三相的负荷电流相同,这样就排除了负荷波动的干扰。

## 通道内动土

电缆通道内的动土作业必须办理工作票并全程有人旁站,机械开挖时应当在电缆两侧各 1 米范围内改为人工开挖。【缆-5】历史上多数电缆外力破坏事故都发生在旁站人员离岗的那几分钟内,所以旁站期间不允许兼任其他工作,哪怕只是回车里取一件工具。
""",
    "03_避雷器与过电压保护检测.md": """# 避雷器与过电压保护检测

## 泄漏电流

金属氧化物避雷器的核心参数是持续运行电压下的泄漏电流,其中阻性分量应当小于 500 微安。【雷-1】泄漏电流的阻性分量对内部受潮极其敏感,而全电流因为容性分量占绝大多数,往往要到受潮的后期才出现明显变化,所以只看全电流会漏掉早期缺陷。

## 红外测温

红外测温可以发现避雷器内部的受潮与阀片劣化,同一组三相避雷器中任意一相温度高于其他两相 1 开尔文以上时应当停电检查。【雷-2】正常运行中的避雷器温度略高于环境温度是正常的,判据是三相之间的相对温差,而不是与环境温度之差 —— 后者会被日照和风速带偏。

## 动作计数器

计数器动作次数应当每次巡视都抄录,动作之后应当检查避雷器本体有无放电痕迹和瓷套裂纹。【雷-3】计数器本身的故障率不低,长期不动不一定代表没有动作,也可能是计数器已经卡死,所以还要结合线路的落雷记录交叉核对,两者对不上就应当停电检查。

## 停电试验

停电试验项目包括绝缘电阻、直流 1 毫安参考电压和百分之七十五参考电压下的泄漏电流。【雷-4】上述三项中有任意一项不合格即判定该只避雷器退出运行,不允许降级使用 —— 避雷器是保护设备,它自身的劣化不会引起任何可见的运行异常,一旦失效就要等到出事故才会被发现。

## 安装要求

避雷器的安装高度和相间距离必须满足过电压保护规程的要求,引下线应当尽量短而直,弯曲半径过小会显著增大冲击阻抗。【雷-5】引下线与主接地网的连接点应当便于检查,且不应当被后期施工的硬化地面永久覆盖,否则下一次测量接地电阻时只能破坏地面。
""",
}

# 反例:语料里**完全没有**的实体。这三篇讲的是蓄电池、电缆、避雷器,
# 一个字都没提 GIS / SF6 / 微水 —— 但「微水含量」在别处的语料里很常见,
# 所以它是一个「听起来合理」的域外问题,而不是「今天天气怎么样」那种
# 一望即分的噪声。反例本身不可信的话,反例族的指标就全是噪声。
NEGATIVE_QUESTION = "GIS 组合电器 SF6 气室的微水含量限值是多少?"


def corrupt_snippet(src: Path, dst: Path) -> None:
    """把第一条可答题的 `snippet` 改成一段语料里不存在的文字。

    模拟**最阴的一种锚点失效**:文件还在、`chunk_index` 也没越界,
    只是在目标块前面增删了字符,块整体位移了 —— 检索器一点问题没有,
    但拿它算出来的分数全是错的。这种必须被拦住,否则人会去调一个没坏的参数。
    """
    out: list[str] = []
    done = False
    for line in src.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not done and s.startswith("{"):
            obj = json.loads(s)
            if obj.get("expected"):
                obj["expected"][0]["snippet"] = "这段文字在语料里根本不存在,块已经位移了"
                line = json.dumps(obj, ensure_ascii=False)
                done = True
        out.append(line)
    assert done, "夹具里没有可答题 —— corrupt_snippet 无从下手"
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_temp_corpus(root: Path) -> dict[str, list[str]]:
    """把 `DOCS` 写到临时目录,并返回 `{文件名: [标记是否落在该块]}` 的辅助数据。"""
    root.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (root / name).write_text(text, encoding="utf-8")
    return {name: [] for name in DOCS}


def build_temp_qa(
    root: Path,
    path: Path,
    *,
    negative_question: str = NEGATIVE_QUESTION,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, tuple[str, int]]:
    """写一份一次性问答集,锚点靠**现算**而不是写死。

    返回 `{标记: (文件, chunk_index)}` 供断言引用。

    只有 2 条可答题 + 1 条反例是刻意的:这个夹具要验证的是**设施的机制**
    (分族、封顶、不可比护栏),不是检索质量本身。题多了只会让跑得慢。
    """
    docs, errors = C.load_doc_chunks(
        root, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if errors:
        raise AssertionError(f"一次性语料加载就出错了:{errors}")

    wanted = {
        "【蓄-3】": ("整组更换,不允许只更换个别落后单体", "蓄电池组放电容量不合格之后应当怎么处理?"),
        "【缆-2】": ("支架锈蚀到什么程度必须更换", "电缆沟支架的镀锌层剥落到什么程度就要更换?"),
    }

    found: dict[str, tuple[str, int]] = {}
    lines: list[str] = [
        "# meta: " + json.dumps(
            {
                "schema_version": 1,
                "corpus": "selftest",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "generated_at": "self-check",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "# 由 scripts/check_eval.py 生成的一次性夹具,随时可删。",
    ]

    for i, (marker, (answer, question)) in enumerate(wanted.items(), 1):
        hits = C.locate_text(docs, marker)
        if not hits:
            raise AssertionError(f"夹具语料里找不到标记 {marker} —— DOCS 被改坏了")
        h = hits[0]
        # snippet 取目标块的**前 30 字**(与 schema.SNIPPET_CHARS 一致),
        # 不是取标记本身:标记在块的中部,而校验比的是目标块的整段正文。
        snippet = docs[h.file][h.chunk_index].text[:30]
        found[marker] = (h.file, h.chunk_index)
        lines.append(
            json.dumps(
                {
                    "id": f"self-{i:03d}",
                    "question": question,
                    "type": "semantic",
                    "status": "keep",
                    "expected": [
                        {
                            "file": h.file,
                            "chunk_index": h.chunk_index,
                            "grade": 3,
                            "snippet": snippet,
                        }
                    ],
                    "answer": answer,
                    "note": "自检夹具",
                    "added_by": "check_eval",
                    "added_at": "self-check",
                },
                ensure_ascii=False,
            )
        )

    lines.append(
        json.dumps(
            {
                "id": "self-901",
                "question": negative_question,
                "type": "negative",
                "status": "keep",
                "expected": [],
                "answer": "",
                "note": "反例:三篇语料里没有任何一个字提到 GIS / SF6 / 微水",
                "added_by": "check_eval",
                "added_at": "self-check",
            },
            ensure_ascii=False,
        )
    )

    dump_lines: list[str] = []
    for line in lines:
        dump_lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(dump_lines) + "\n", encoding="utf-8")
    return found


# --------------------------------------------------------------------- #
# A1 / A2 —— 指标
# --------------------------------------------------------------------- #


def check_metrics(ch: Checker) -> None:
    ch.section("A1 指标对拍手算值")

    ranked = [("d2", 0), ("d1", 1), ("d3", 2)]
    gold = {("d1", 1): 3}

    ch.check(M.hit_at_k(ranked, gold, 1) == 0.0, "hit@1 == 0(gold 在第 2)", "d1 排在第 2,窗口只到 1")
    ch.check(M.hit_at_k(ranked, gold, 3) == 1.0, "hit@3 == 1")
    ch.check(M.reciprocal_rank_at_k(ranked, gold, 3) == 0.5, "mrr@3 == 0.5", "1/2")
    # 除以 k=3 而不是返回条数 3 —— 这个用例里两者恰好相同,所以它不能证明
    # 「除的是 k」。真正的判别用例在 A2 的 Precision 演示里。
    ch.check(M.precision_at_k(ranked, gold, 3) == 1 / 3, "precision@3 == 1/3")
    ch.check(M.recall_at_k(ranked, gold, 3) == 1.0, "recall@3 == 1.0(唯一 gold 被找到)")

    # 多 gold + 分级:理想排序拿满分,把高相关的块压到第 3 要扣分
    multi = [("a", 0), ("b", 1), ("c", 2)]
    grades = {("a", 0): 1, ("c", 2): 3}
    ideal = M.ndcg_at_k([("c", 2), ("a", 0), ("b", 1)], grades, 3)
    worse = M.ndcg_at_k(multi, grades, 3)
    ch.check(ideal == 1.0, "理想排序的分级 nDCG == 1.0", f"实得 {ideal}")
    ch.check(
        0 < worse < ideal,
        "分级 nDCG 对「高相关的块被压后」扣分",
        f"{worse:.4f} < {ideal:.4f}",
    )

    # 分级 vs 二值的**唯一**干净对照:两个排序占用的位置完全相同
    # (第 1、第 2 位都是 gold),只有 grade 换了个座。位置没变 ⇒ 二值口径下
    # 两个排序连带折扣的那部分都一模一样 ⇒ 必然同分;分级口径才分得出。
    #
    # 注意别用「交换两个块的名次」来构造这个对照 —— 那会连**占用的位置**
    # 一起改掉,二值 nDCG 也会跟着动,得出的结论是错的。
    two = {("a", 0): 3, ("b", 1): 1}
    hi_first = [("a", 0), ("b", 1)]
    lo_first = [("b", 1), ("a", 0)]
    ch.check(
        M.ndcg_at_k(hi_first, two, 2) == 1.0
        and M.ndcg_at_k(lo_first, two, 2) < 1.0,
        "分级 nDCG 在乎「高低档谁在前」",
        f"{M.ndcg_at_k(hi_first, two, 2):.4f} vs {M.ndcg_at_k(lo_first, two, 2):.4f}",
    )
    ch.check(
        M.ndcg_at_k(hi_first, two, 2, binary=True)
        == M.ndcg_at_k(lo_first, two, 2, binary=True),
        "同一对在二值 nDCG 下**同分**",
        "所以 grade 没填也出得了有意义的数,填了才多这一档分辨力 —— 两个并排报的理由",
    )

    # 反例绝不能进 score_query —— 进了会算出一个没有意义的分数
    try:
        M.score_query(ranked, {}, 3, item_id="neg")
        ch.check(False, "score_query 拒绝空 gold", "它居然返回了")
    except ValueError:
        ch.check(True, "score_query 对空 gold 抛 ValueError(反例走 score_negative)")

    # 去重要真的生效:同一块出现两次不该顶掉一个真名次
    dup = [("d1", 1), ("d1", 1), ("d2", 0), ("d3", 2)]
    ch.check(
        M.hit_at_k(dup, gold, 2) == 1.0,
        "window 按首次出现去重",
        "重复项不该挤掉第 2 个位置",
    )

    # k <= 0 要抛,不能静默返回空窗口(那会让 k=0 看着像全错)
    try:
        M.unique_window(ranked, 0)
        ch.check(False, "k<=0 被拒绝", "它居然返回了")
    except ValueError:
        ch.check(True, "k<=0 抛 ValueError")

    # 反例族只认分数与阈值
    ns = M.score_negative([(("x", 0), 0.9), (("y", 1), 0.1)], 0.5)
    ch.check(ns.leak == 1.0 and ns.above_threshold_rate == 0.5, "反例 leak / above_threshold", "0.9≥0.5 而 0.1<0.5")
    ch.check(M.score_negative([], 0.5).leak == 0.0, "无返回时 leak == 0")


def check_llamaindex(ch: Checker) -> None:
    ch.section("A2 与 LlamaIndex 交叉验证")

    try:
        # 这几个在 `llama_index.core.evaluation` **顶层不存在**,只在
        # `...evaluation.retrieval.metrics` 里。`Precision` 同理 ——
        # 计划里写的 `from llama_index.core.evaluation import Precision` 会 ImportError。
        from llama_index.core.evaluation.retrieval.metrics import (
            HitRate,
            MRR,
            NDCG,
            Precision,
        )
    except ImportError as exc:
        ch.info(f"没装 llama-index,跳过交叉验证({exc})")
        return

    def li(metric, expected: list[str], retrieved: list[str]) -> float:
        return float(
            metric.compute(query="q", expected_ids=expected, retrieved_ids=retrieved).score
        )

    # 它们都**不接受 k**:自己先把 retrieved 截断到 k,这一点和 eval.metrics
    # 的约定一致(那边也是 k 由 metrics 截断,但交叉验证必须显式给出同一批)
    cases = [
        # (expected, retrieved, k)
        (["d1"], ["d2", "d1", "d3"], 3),
        (["d1"], ["d3", "d2", "d1"], 3),
        (["a", "c"], ["b", "a", "c"], 3),
    ]

    for expected, retrieved, k in cases:
        # gold 的键与 retrieved 的 id 对齐:把 "d1" 这种 id 映射成 (id, 0)
        gold_grades = {(e, 0): 1 for e in expected}
        ranked = [(r, 0) for r in retrieved]
        trunc_r = retrieved[:k]

        hr_li, hr_ours = li(HitRate(), expected, trunc_r), M.hit_at_k(ranked, gold_grades.keys(), k)
        mrr_li, mrr_ours = li(MRR(), expected, trunc_r), M.reciprocal_rank_at_k(ranked, gold_grades.keys(), k)
        nd_li, nd_ours = li(NDCG(), expected, trunc_r), M.ndcg_at_k(ranked, gold_grades, k, binary=True)
        ch.check(
            abs(hr_li - hr_ours) < 1e-9 and abs(mrr_li - mrr_ours) < 1e-9 and abs(nd_li - nd_ours) < 1e-9,
            f"HitRate/MRR/二值NDCG 一致  expected={expected} retrieved={retrieved} k={k}",
            f"hit {hr_li:.6f}/{hr_ours:.6f}  mrr {mrr_li:.6f}/{mrr_ours:.6f}  ndcg {nd_li:.6f}/{nd_ours:.6f}",
        )
    ch.info("注:上面每条都满足 |expected| <= k —— LlamaIndex 的 NDCG 不接受 k,")
    ch.info("    IDCG 按**全部** expected 算,|expected| > k 时与我们的截断口径不同。")

    # 反例根本喂不进去:expected 为空时它**抛异常**,不是得 0 分。
    # (实测:`compute` 开头就是 `if not retrieved_ids or not expected_ids: raise`)
    # 这不是抱怨 —— 是一条设计约束:反例必须走自己的指标族,
    # 因为它们在可答题的度量里既算不出分、也不该被算。
    try:
        li(HitRate(), [], ["x", "y"])
        ch.check(False, "expected 为空时 LlamaIndex 要拒绝", "它居然返回了")
    except ValueError as exc:
        ch.check(
            True,
            "expected 为空 → LlamaIndex 抛 ValueError(反例喂不进可答题那套指标)",
            f"「{exc}」—— 所以反例族只能走 leak / top1_score 那套自己的指标",
        )
    # --- Precision:不只是「排除掉」,而是把它的危害演出来 ---
    #
    # 它的定义是 |retrieved ∩ expected| / len(retrieved),**分母是返回条数而不是 k**,
    # 而且没有 k 参数。于是「阈值只放行 1 条、恰好是 gold」这种
    # **系统正在失败**的形态,它报满分。
    one_hit = ["d1"]
    gold_ids = ["d1"]
    p_li = li(Precision(), gold_ids, one_hit)
    p_ours = M.precision_at_k([("d1", 0)], {("d1", 0): 3}, 3)
    ch.check(
        p_li == 1.0 and abs(p_ours - 1 / 3) < 1e-9,
        "Precision 现场演示:只返回 1 条且正确时它报 1.0,我们报 1/3",
        f"llamaindex {p_li:.4f} / 我们 {p_ours:.4f} —— 返回得越少它越高兴,方向是反的",
    )
    ch.info("   这就是 eval/metrics.py 自己写 precision 并**除以 k** 的全部理由;")
    ch.info("   也是它被排除在交叉验证之外的原因(不是嫌它麻烦)。")

    # 空 expected 会让它直接崩,而不是得 0 分 —— 反例根本喂不进去
    try:
        li(Precision(), [], ["x"])
        ch.check(False, "Precision 对空 expected 抛异常", "它居然返回了")
    except Exception as exc:  # noqa: BLE001 - 具体异常类型随版本变,不写死
        ch.check(True, "Precision 对空 expected 直接抛异常(反例喂不进去)", type(exc).__name__)


# --------------------------------------------------------------------- #
# A3 —— 加载器
# --------------------------------------------------------------------- #


def check_loader(ch: Checker, tmp: Path) -> None:
    ch.section("A3 加载器容错与报错定位")

    good = tmp / "tolerance.jsonl"
    good.write_text(
        "\n".join(
            [
                "# meta: " + json.dumps({"schema_version": 1, "corpus": "tolerance"}, ensure_ascii=False),
                "# 一条普通注释,人工筛的时候会写「# 这条待确认」",
                "",
                "   ",
                json.dumps(
                    {
                        "id": "k-1",
                        "question": "一条正常的问题?",
                        "type": "lexical",
                        "status": "keep",
                        "expected": [{"file": "a.md", "chunk_index": 1}],
                        "answer": "答案",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    # drop 且 expected 为空、type 还不是 negative:明说了不参与
                    # 评分,不该让整份文件加载不了
                    {"id": "k-2", "question": "被丢弃的问题?", "type": "semantic", "status": "drop", "expected": []},
                    ensure_ascii=False,
                ),
                json.dumps(
                    # **没标完**的题:生成器出了题、锚点还没附上。这是 draft 的
                    # 正常形态,必须能读进来 —— 否则一行没标完,
                    # `eval_gen` 的整批产出都读不进来。它只该出现在 warnings() 里。
                    {"id": "k-4", "question": "一条还没标完的题?", "type": "semantic", "status": "draft", "expected": []},
                    ensure_ascii=False,
                ),
                json.dumps(
                    # 未知 type:**告警不报错**,让人能自己加类而不必改代码
                    {"id": "k-3", "question": "一个用了新题型的问题?", "type": "brand_new", "status": "keep", "expected": [{"file": "a.md", "chunk_index": 0}]},
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        es = load_qa(good)
        n = len(es.items)
        ch.check(n == 4, "空行 / 空白行 / # 注释 都被跳过", f"读到 {n} 条")
        ch.check(es.meta.get("corpus") == "tolerance", "`# meta:` 行被解析成文件级元信息")
        ch.check(
            [i.id for i in es.kept] == ["k-1", "k-3"],
            "drop / draft 不进 kept",
            f"kept={[i.id for i in es.kept]}",
        )
        warns = es.warnings()
        ch.check(
            any("brand_new" in w for w in warns),
            "未知 type 只告警",
            "不报错,所以加新题型不用改代码",
        )
        # 这条是自检**真的发现过一个 bug** 的地方:原实现不看 status,
        # 于是「drop 或 draft 且 expected 为空、type 还不是 negative」——
        # 也就是「没标完的正常形态」—— 会让**整份文件**加载不了,
        # eval_gen 出了一批题只要有一条没标完,全批都读不进来。
        ch.check(
            any("k-4" in w and "还没标完" in w for w in warns),
            "没标完的 draft 能读进来,但会出现在 warnings() 里",
            next((w for w in warns if w.startswith("k-4")), "没有提醒 —— 会永远躺在 draft 里"),
        )
        # 指纹只算 kept:把一条题标成 drop 之后指纹不该变,否则护栏变成噪音
        ch.check(
            load_qa(good).sha1() == es.sha1(),
            "sha1 稳定(同一份文件两次读一致)",
        )
    except EvalSetError as exc:
        ch.check(False, "容错加载", str(exc))

    # 但 status=keep 的那条护栏**必须还在** —— 放宽只能放到 draft/drop 为止
    not_yet = tmp / "keep_empty.jsonl"
    not_yet.write_text(
        json.dumps(
            {"id": "n-1", "question": "keep 但没有 gold 的题?", "type": "semantic", "status": "keep", "expected": []},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        load_qa(not_yet)
        ch.check(False, "status=keep 却没有 gold 仍要被拒", "它居然读过去了")
    except EvalSetError as exc:
        ch.check(
            "expected 为空但 type" in str(exc),
            "放宽之后 keep 的等价性护栏还在(没被顺手拆掉)",
            "否则「可答题但没 gold」会被静默当成反例吸进分母",
        )

    # 坏行必须给行号与原文
    bad = tmp / "broken.jsonl"
    bad.write_text(
        "# 注释\n" + '{"id": "b-1", "question": "好的?"}\n' + '{这不是 JSON\n',
        encoding="utf-8",
    )
    try:
        load_qa(bad)
        ch.check(False, "坏 JSON 要抛出", "它居然读过去了")
    except EvalSetError as exc:
        msg = str(exc)
        ch.check(
            ":3:" in msg and "这不是 JSON" in msg,
            "坏 JSON 抛出时带**行号与原文**",
            f"报错里含行号 :3: 与原文 → {msg.splitlines()[0][:70]}",
        )

    # type=negative 与 expected 为空必须严格等价 —— 否则「可答题但没 gold」
    # 会被当成反例静默吸收进分母,把平均分往下拽
    contradictory = tmp / "contradictory.jsonl"
    contradictory.write_text(
        json.dumps({"id": "c-1", "question": "说是反例却列了 gold?", "type": "negative", "expected": [{"file": "a.md", "chunk_index": 0}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        load_qa(contradictory)
        ch.check(False, "type=negative 却带 gold 要被拒", "它居然读过去了")
    except EvalSetError:
        ch.check(True, "type=negative 与 expected==[] 严格等价(矛盾数据被拒)")

    # grade 的边界:0 分是「列上去了但一点都不相关」,自相矛盾
    grade0 = tmp / "grade0.jsonl"
    grade0.write_text(
        json.dumps({"id": "g-1", "question": "grade 写 0 的问题?", "expected": [{"file": "a.md", "chunk_index": 0, "grade": 0}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        load_qa(grade0)
        ch.check(False, "grade=0 要被拒", "它居然读过去了")
    except EvalSetError:
        ch.check(True, "grade=0 被拒(要表示不相关就别列它)")

    # bool 是 int 的子类,不挡的话 "chunk_index": true 会静默变成 1
    booly = tmp / "booly.jsonl"
    booly.write_text(
        json.dumps({"id": "b-1", "question": "chunk_index 写成 true 的问题?", "expected": [{"file": "a.md", "chunk_index": True}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        load_qa(booly)
        ch.check(False, "chunk_index=true 要被拒", "bool 是 int 的子类,不挡就静默变成 1")
    except EvalSetError:
        ch.check(True, "chunk_index=true 被拒(不静默变成 1)")

    # `..` 能读到语料目录外面去 —— 那不是锚点画错,是越权读文件
    escape = tmp / "escape.jsonl"
    escape.write_text(
        json.dumps({"id": "e-1", "question": "锚点指到语料外面去的问题?", "expected": [{"file": "../secret.md", "chunk_index": 0}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        load_qa(escape)
        ch.check(False, "gold.file 含 .. 要被拒", "它居然读过去了")
    except EvalSetError:
        ch.check(True, "gold.file 含 .. 被拒(锚点不许逃出语料目录)")


# --------------------------------------------------------------------- #
# A4 —— 锚点
# --------------------------------------------------------------------- #


def check_anchors(ch: Checker, tmp: Path, *, chunk_size: int, chunk_overlap: int) -> None:
    ch.section("A4 锚点校验必须响亮失败")

    root = tmp / "anchor_corpus"
    root.mkdir(parents=True, exist_ok=True)
    (root / "real.md").write_text(
        "# 真文档\n\n" + "这一段用来把文档撑到两块以上,好让越界断言有意义。" * 30,
        encoding="utf-8",
    )
    docs, _ = C.load_doc_chunks(root, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    n_chunks = len(docs["real.md"])
    ch.check(n_chunks >= 2, "夹具文档确实分出了多块", f"real.md → {n_chunks} 块")

    es = EvalSet(
        items=[
            _item("a-missing", [GoldChunk(file="real_doc.md", chunk_index=0, snippet="x")]),
            _item("a-range", [GoldChunk(file="real.md", chunk_index=n_chunks + 5, snippet="y")]),
            _item("a-shift", [GoldChunk(file="real.md", chunk_index=0, snippet="一段根本不在这个文件里的文字")]),
            _item("a-ok", [GoldChunk(file="real.md", chunk_index=0, snippet=docs["real.md"][0].text[:20])]),
        ]
    )
    report = C.resolve_anchors(es, root, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    kinds = {i.kind for i in report.issues}
    ch.check(not report.ok, "有问题的锚点让 report.ok 为假", f"问题 {len(report.issues)} 条")
    ch.check("missing_file" in kinds, "缺文件被查出来")
    ch.check("out_of_range" in kinds, "chunk_index 越界被查出来")
    ch.check(
        "snippet_mismatch" in kinds,
        "**内容位移**被查出来",
        "块还在、序号也没越界,只有正文对不上 —— 这类最要命",
    )
    ch.check("a-ok" in report.resolved, "对的锚点照常通过")

    missing = [i for i in report.issues if i.kind == "missing_file"]
    # 拼错文件名时不能只说「没有」,要给出最接近的现有名字 ——
    # 把一次长排查变成五秒修复
    ch.check(
        bool(missing and "real.md" in missing[0].suggestion),
        "缺文件时给 difflib 近似名字建议",
        missing[0].suggestion[:60] if missing else "",
    )

    shift = [i for i in report.issues if i.kind == "snippet_mismatch"]
    ch.check(bool(shift and shift[0].suggestion), "位移时给出「这段文字现在在哪一块」的建议")

    text = C.format_issues(report.issues)
    ch.check(
        "拒绝出分" in text and "不是检索质量下降" in text,
        "失败文案明确写出「拒绝出分」且把锚点失效与检索退步分开",
    )

    # 上面测的是 `resolve_anchors` 判得对不对。**更要紧的一半**是
    # 「runner 会不会照常出分」—— 判对了但照常打印一张低分表,等于没拦。
    # 那只跑得动真 runner,所以放在 A6(要连 Qdrant)。这里留个记号。


def _item(item_id: str, golds: list[GoldChunk]) -> object:
    from eval.schema import EvalItem

    return EvalItem(id=item_id, question=f"{item_id} 的问题内容够长了吗?", expected=golds)


def check_seed_corpus(ch: Checker, *, chunk_size: int, chunk_overlap: int) -> None:
    ch.section("A5 种子语料体检")

    if not SEED_QA.is_file():
        ch.check(False, "种子问答集存在", str(SEED_QA))
        return

    es = load_qa(SEED_QA)
    ch.check(bool(es.kept), "有 status=keep 的题", f"keep {len(es.kept)} 条")
    ch.check(
        es.max_chunk_index() >= 2,
        "max(chunk_index) >= 2",
        f"实得 {es.max_chunk_index()} —— 为 0 说明每篇只出 1 块,锚点方案等于没测",
    )

    docs, errors = C.load_doc_chunks(
        SEED_CORPUS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    ch.check(not errors, "种子语料全部可加载", f"{len(errors)} 个文件读不出来")
    ch.check(len(docs) >= 6, "同领域文档 >= 6 篇", f"实得 {len(docs)} 篇(否则没有干扰项,1 篇语料人人 hit@1)")

    referenced = es.files()
    missing = [f for f in referenced if f not in docs]
    ch.check(not missing, "问答集引用的文件都在语料目录里", f"缺 {missing}" if missing else f"{len(referenced)} 个文件")

    types = es.type_counts()
    ch.check("negative" in types, "有反例族", f"题型分布 {types}")
    ch.check(
        all(t in {"lexical", "semantic", "multi_hop", "contextual", "negative", "ambiguous"} for t in types),
        "题型都在已知集合内",
    )

    # 锚点必须**现在**就能通过,否则这份基线是坏的
    report = C.resolve_anchors(
        es, SEED_CORPUS, chunk_size=chunk_size, chunk_overlap=chunk_overlap, docs=docs
    )
    ch.check(report.ok, "种子问答集的锚点全部通过", f"问题 {len(report.issues)} 条")


# --------------------------------------------------------------------- #
# A6 —— 端到端(一次性小语料)
# --------------------------------------------------------------------- #


def check_end_to_end(
    ch: Checker,
    tmp: Path,
    *,
    settings,
    models: bool,
) -> str | None:
    ch.section("A6 端到端:一次性小语料 + 真 runner")

    corpus = tmp / "corpus"
    qa = tmp / "qa.jsonl"
    qa_v2 = tmp / "qa_noedit.jsonl"

    build_temp_corpus(corpus)
    anchors = build_temp_qa(
        corpus,
        qa,
        chunk_size=settings.ingest.chunk_size,
        chunk_overlap=settings.ingest.chunk_overlap,
    )
    # 只改**反例**的问题文本。可答题一个字没动 —— 所以可答题的指标必须逐位相同。
    build_temp_qa(
        corpus,
        qa_v2,
        negative_question="变电站的消防水泵多久试运行一次?",
        chunk_size=settings.ingest.chunk_size,
        chunk_overlap=settings.ingest.chunk_overlap,
    )
    ch.info(f"夹具语料 {len(DOCS)} 篇;锚点 {anchors}")

    collection = C.collection_name(corpus, contextual=False)
    ch.check(
        collection.startswith("agentic_kb_eval_"),
        "collection 名走评估前缀",
        collection,
    )

    out_a, out_b, out_c = tmp / "a.json", tmp / "b.json", tmp / "c.json"

    # --- 基准跑 ---
    code, _o, err = run_cli(
        ["--corpus", str(corpus), "--qa", str(qa), "--no-contextual",
         "--preset", "no-rerank", "--ks", "1,3,5", "--top-k", "5", "--json", str(out_a)]
    )
    if not ch.check(code == 0, "基准跑通", f"退出码 {code}\n       {err.strip()[:400]}"):
        return collection
    pa = read_json(out_a)

    ch.check(
        pa["runs"]["no-rerank"]["n_answerable"] == 2 and pa["runs"]["no-rerank"]["n_negative"] == 1,
        "可答题与反例分族计数正确",
        f"可答 {pa['runs']['no-rerank']['n_answerable']} / 反例 {pa['runs']['no-rerank']['n_negative']}",
    )
    neg_keys = set(pa["runs"]["no-rerank"]["negatives"].get("1", {}))
    ch.check(
        "leak" in neg_keys and "top1_score_mean" in neg_keys,
        "反例族有自己的指标名",
        f"{sorted(neg_keys)}",
    )
    ch.check(
        not (neg_keys & {"hit", "mrr", "rr"}),
        "反例族与可答题的指标名不重叠(不会被误并进平均)",
    )

    # --- 反例隔离 ---
    code, _o, err = run_cli(
        ["--corpus", str(corpus), "--qa", str(qa_v2), "--no-contextual",
         "--preset", "no-rerank", "--ks", "1,3,5", "--top-k", "5", "--json", str(out_c)]
    )
    if ch.check(code == 0, "只改反例文本后重跑", f"退出码 {code}"):
        pc = read_json(out_c)
        a_agg = pa["runs"]["no-rerank"]["aggregates"]
        c_agg = pc["runs"]["no-rerank"]["aggregates"]
        a_score = {d["item_id"]: d["rank"] for d in pa["runs"]["no-rerank"]["per_item"]}
        c_score = {d["item_id"]: d["rank"] for d in pc["runs"]["no-rerank"]["per_item"]}
        ch.check(
            a_agg == c_agg and a_score == c_score,
            "改反例的问题文本后,可答题的聚合与逐题名次**逐位不变**",
            "反例确实被隔离在可答题之外" if a_agg == c_agg else f"聚合变了:{a_agg} → {c_agg}",
        )
        ch.check(
            pa["qa"]["sha1"] != pc["qa"]["sha1"],
            "但问答集指纹**变了**(所以基线对比会正确拒绝)",
            f"{pa['qa']['sha1']} vs {pc['qa']['sha1']}",
        )

    # --- fusion_top_k 真的封顶 ---
    code, _o, err = run_cli(
        ["--corpus", str(corpus), "--qa", str(qa), "--no-contextual",
         "--preset", "no-rerank", "--set", "fusion_top_k=2",
         "--ks", "1,5", "--top-k", "5", "--json", str(out_b)]
    )
    if ch.check(code == 0, "fusion_top_k=2 跑通", f"退出码 {code}"):
        pb = read_json(out_b)
        rb = pb["runs"]["no-rerank"]
        ch.check(
            rb["effective_k"] == 2,
            "effective_k 如实报 2",
            f"min(top_k=5, fusion_top_k=2, 100) = {rb['effective_k']}",
        )
        returned = [d["n_returned"] for d in rb["per_item"]]
        ch.check(
            all(n <= 2 for n in returned),
            "每条的返回条数都被封在 2 以内",
            f"n_returned = {returned}",
        )
        ch.check(
            "5" not in rb["aggregates"],
            "k=5 **从报告里消失**",
            "不是打成一个看着像「全错」的 0 —— 测不了和全错是两件事",
        )
        ch.check(
            "fusion_top_k=2" in err and "top_k=5" in err,
            "runner 大声报出了「top_k > fusion_top_k」并指名两个数字",
            next((ln.strip() for ln in err.splitlines() if "fusion_top_k=2" in ln), "")[:120],
        )
        # 封顶只会**移除**竞争者,所以名次不可能变差 —— 这条同时证明了
        # 「capped 跑的是同一个池子的前缀」,而不是换了个池子。
        ra = ranks_of(pa, "no-rerank")
        rcap = ranks_of(pb, "no-rerank")
        ch.check(
            all(_finite(rcap[i]) <= _finite(ra[i]) for i in rcap),
            "封顶后的名次不会变差(证明它是同一池子的前缀)",
            f"cap={rcap}  full={ra}",
        )

    # --- 锚点失效必须在评分之前中止 ---
    qa_bad = tmp / "qa_bad.jsonl"
    corrupt_snippet(qa, qa_bad)
    code, out_bad, err_bad = run_cli(
        ["--corpus", str(corpus), "--qa", str(qa_bad), "--no-contextual",
         "--preset", "no-rerank", "--ks", "1,3", "--top-k", "3"]
    )
    ch.check(code == 1, "锚点失效(块位移)→ 退出 1", f"退出码 {code}")
    ch.check(
        "锚点" in out_bad + err_bad,
        "报错里点名「锚点」",
        next((ln.strip() for ln in (out_bad + err_bad).splitlines() if "锚点" in ln), "")[:120],
    )
    # 关键:断的是「**没有**打印聚合表」。判得对但照常出分的实现也能退出 1,
    # 而那张低分表会被当成一次真实的检索退步记进脑子里。
    # 找 mrr/ndcg 而不是 "hit":后者的子串风险高,前者只在聚合表里出现。
    blob = (out_bad + err_bad).lower()
    ch.check(
        "mrr" not in blob and "ndcg" not in blob,
        "**没有打印任何聚合指标**(证明是在评分之前中止,不是被当成一次 miss 计进去)",
        "输出里出现了 mrr/ndcg 字样",
    )

    # --- 不可比护栏:qa.sha1 ---
    base_name = "_selftest_base"
    code, _o, err = run_cli(
        ["--corpus", str(corpus), "--qa", str(qa), "--no-contextual",
         "--preset", "no-rerank", "--ks", "1,3,5", "--top-k", "5",
         "--save-baseline", base_name]
    )
    if ch.check(code == 0, "存基线", f"退出码 {code}"):
        code, out, err = run_cli(
            ["--corpus", str(corpus), "--qa", str(qa_v2), "--no-contextual",
             "--preset", "no-rerank", "--ks", "1,3,5", "--top-k", "5",
             "--baseline", base_name]
        )
        ch.check(code == 1, "问答集被改过 → 拒绝出 diff(退出 1)", f"退出码 {code}")
        ch.check(
            "问答集 sha1" in err,
            "拒绝时**指名**是哪个字段不同",
            next((ln.strip() for ln in err.splitlines() if "sha1" in ln), "")[:120],
        )
        ch.check(
            "hit" not in out.split("对比")[-1] if "对比" in out else True,
            "被拒绝时不打印任何涨跌表",
        )

        # --- 不可比护栏:配置变体 ---
        # 这一步刻意用 `no-rerank+dense-rrf-k2` 而不是 `no-threshold`:
        # 两者名字都不同,但前者不需要加载 2.3G 重排器,
        # 所以 --no-models 下这条护栏照样能被验证。
        code, out, err = run_cli(
            ["--corpus", str(corpus), "--qa", str(qa), "--no-contextual",
             "--preset", "no-rerank+dense-rrf-k2", "--ks", "1,3,5", "--top-k", "5",
             "--baseline", base_name]
        )
        ch.check(code == 1, "配置变体不同 → 也拒绝出 diff", f"退出码 {code}")
        ch.check(
            "配置变体不同" in err,
            "拒绝时指名两个 spec",
            next((ln.strip() for ln in err.splitlines() if "配置变体" in ln), "")[:140],
        )
        ch.info("   ↑ 这条护栏也意味着:**跨 preset 的对比不能靠基线**。")
        ch.info("     要比重排开/关,就固定 spec 名、用 --set 挪一个旋钮。")

    return collection


# --------------------------------------------------------------------- #
# A7 / A8 —— 种子语料上的真实数字
# --------------------------------------------------------------------- #


def check_seed_run(ch: Checker, tmp: Path, *, models: bool) -> None:
    ch.section("A7 防循环 + A8 旋钮真的能挪动分数(种子语料)")

    presets = ["no-rerank"] + (["shipped", "no-threshold"] if models else [])
    out = tmp / "seed.json"
    argv = ["--corpus", str(SEED_CORPUS), "--qa", str(SEED_QA), "--no-contextual",
            "--ks", "1,3,5,10", "--top-k", "10", "--json", str(out)]
    for p in presets:
        argv += ["--preset", p]

    code, out_text, err = run_cli(argv)
    if not ch.check(code == 0, f"种子语料跑通({', '.join(presets)})", f"退出码 {code}\n       {err.strip()[:400]}"):
        return

    payload = read_json(out)
    runs = payload["runs"]

    # --- A7 防循环 ---
    h1 = hit_of(payload, "no-rerank", 1)
    ch.check(
        0.30 <= h1 <= 0.95,
        f"no-rerank 的 hit@1 = {h1:.3f} 落在 0.30~0.95 的宽区间内",
        "满 1.0 说明题太简单或 gold 是照检索器自己的输出标的(自证循环);~0 说明锚点或导入坏了",
    )

    # 报告里的警告要与这里的判据**同源**,不然「脚本绿」和「输出在喊」会打架。
    # 警告是**逐 spec** 的(`⚠ {name}:k={head} 的 hit = ...`),所以只能按 spec 找,
    # 不能拿整段 stdout 一刀切 —— 那会把「另一个配置饱和了」误判成误报。
    warn_lines = [ln.strip() for ln in out_text.splitlines() if "几乎满分" in ln]
    flagged = {ln.split(":")[0].lstrip("⚠ ") for ln in warn_lines}
    ch.check(
        "no-rerank" not in flagged,
        f"hit@1={h1:.3f} 未饱和,runner 没有对它误报防循环警告",
        f"被点名的配置:{sorted(flagged)}",
    )
    if models:
        # 开重排 + 阈值 0 时种子集**确实**全对(hit@1 = 1.000),这时警告必须响。
        # 它是在如实说「这套题对这个配置已经测不出差别了」—— 不是误报,
        # 也正是「头条数字要在阈值中性配置下算」的实际理由。
        ch.check(
            "no-threshold" in flagged or "shipped" in flagged,
            "重排打开后 hit@1 饱和,警告**确实**响了(该喊的时候会喊)",
            f"被点名的配置:{sorted(flagged)}",
        )
        ch.info("   ↑ 这正是种子集在重排面前饱和的证据;no-rerank 才是判别性配置。")
    else:
        ch.info("   (--no-models:shipped/no-threshold 没跑,不检查饱和告警)")

    # 并列定序:同分时 Qdrant 的返回顺序不稳定,名次是抛硬币的结果。
    # 报告必须**如实标出来**,否则一次 `hit@1 0.800→0.867` 会被读成进步,
    # 而那只是这条题换了一枚硬币的正反面。
    tie = runs["no-rerank"].get("tie_ambiguous") or []
    ch.info(f"并列区间内定序的题:{tie or '无'}")
    by_item = {d["item_id"]: d for d in runs["no-rerank"]["per_item"]}
    unearned = [
        i for i in tie
        if i in by_item and by_item[i]["rank_lo"] == by_item[i]["rank_hi"]
    ]
    ch.check(
        not unearned,
        "被标成「并列」的题,其名次区间 lo != hi",
        "标记是挣来的:lo == hi 说明那题其实不并列" if not unearned else f"误标:{unearned}",
    )
    ch.check(
        "tie_ambiguous" in runs["no-rerank"],
        "并列信息进了基线(将来逐题 diff 才看得见「这条变化发生在抛硬币上」)",
    )

    if not models:
        ch.info("--no-models:跳过重排器相关断言(shipped / no-threshold 没跑)")
        return

    # --- A8 重排开关 ---
    m_off = float(runs["no-rerank"]["aggregates"]["1"]["rr"])
    m_on = float(runs["shipped"]["aggregates"]["1"]["rr"])
    ch.check(
        m_on > m_off,
        f"开重排后 mrr@1 上升:{m_off:.3f} → {m_on:.3f}",
        "这是重排器**值不值 2.3G** 的第一个数字",
    )
    ch.check(
        runs["no-rerank"]["aggregates"] != runs["shipped"]["aggregates"],
        "两个配置的聚合**不是**逐位相同(旋钮接了线)",
        "逐位相同就该怀疑参数静默无效",
    )

    # --- A8 阈值挪动反例 ---
    leak_shipped = float(runs["shipped"]["negatives"]["1"]["leak"])
    leak_nothresh = float(runs["no-threshold"]["negatives"]["1"]["leak"])
    ch.check(
        leak_nothresh > leak_shipped,
        f"阈值 0.05→0 让反例 leak 上升:{leak_shipped:.3f} → {leak_nothresh:.3f}",
        "leak 是阈值参数**唯一**的判别性指标,别的指标对它几乎不敏感",
    )
    thr = runs["no-threshold"]["retrieval"]["rerank_min_score"]
    ch.check(
        float(thr) == 0.0,
        "no-threshold 的阈值确实是 0",
        f"rerank_min_score={thr}",
    )

    # `rrf_k` 走的是 store 侧那条路(query_hybrid 从 self.retrieval 读),
    # 曾经**静默无效**过。这里拿反例的分数底当地标:改 rrf_k 会改分数刻度。
    ch.info(
        f"分数刻度参考:no-rerank 的 top1_score_mean="
        f"{runs['no-rerank']['negatives']['1']['top1_score_mean']:.4f}"
    )


# --------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="check_eval.py",
        description="检索质量评估设施自检 —— 证明它能区分好坏,而不是永远打印好看的数字。",
    )
    ap.add_argument("--no-qdrant", action="store_true", help="只跑纯计算部分(不连 Qdrant、不加载模型)")
    ap.add_argument("--no-models", action="store_true", help="跳过需要 2.3G 重排器的断言")
    args = ap.parse_args(argv)

    ch = Checker()
    offline = args.no_qdrant

    print("=" * 70)
    print("检索质量评估自检")
    print("=" * 70)
    print("这个脚本**不花 LLM 配额**;但连 Qdrant 的部分会加载 bge-m3(约 2.2G)。")

    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        print(f"读不到配置:{exc}", file=sys.stderr)
        return 1

    tmp = SELFTEST_DIR
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    created_collection: str | None = None
    v1 = v2 = None
    try:
        # 离线部分
        check_metrics(ch)
        check_llamaindex(ch)
        check_loader(ch, tmp)
        check_anchors(ch, tmp, chunk_size=settings.ingest.chunk_size, chunk_overlap=settings.ingest.chunk_overlap)
        check_seed_corpus(ch, chunk_size=settings.ingest.chunk_size, chunk_overlap=settings.ingest.chunk_overlap)

        if offline:
            ch.section("跳过:Qdrant 相关部分(--no-qdrant)")
            ch.info("A6 端到端 / A7 防循环 / A8 旋钮 都没跑 —— 上面全绿不代表评估能出数。")
        else:
            url = settings.qdrant.url
            try:
                before = collection_names(url)
            except Exception as exc:  # noqa: BLE001
                ch.section("跳过:连不上 Qdrant")
                ch.info(f"{exc}")
                ch.info("先跑 scripts\\start_qdrant.bat")
                before = None

            if before is not None:
                ch.section("A9-0 前置:正式库必须不存在")
                # 计划里的验收项:curl /collections 只有 eval 库,没有 agentic_kb
                prod = settings.qdrant.collection
                ch.check(
                    prod not in before,
                    f"正式 collection {prod!r} 不存在",
                    f"当前:{before}",
                )
                ch.info("评估全程只写 agentic_kb_eval_*;`_open_eval_store` 里还有代码级护栏。")

                created_collection = check_end_to_end(
                    ch, tmp, settings=settings, models=not args.no_models
                )
                check_seed_run(ch, tmp, models=not args.no_models)

                ch.section("A9 收尾:正式库仍然不存在")
                after = collection_names(url)
                ch.check(prod not in after, f"跑完之后 {prod!r} 仍然不存在", f"当前:{after}")
                ch.check(
                    "agentic_kb" not in after,
                    "没有任何名字恰好是 agentic_kb 的库",
                    f"当前:{after}",
                )

    finally:
        # 一次性 collection 与临时目录照 house style 全清;
        # 种子语料的那个**留着**(见模块头:它是可复用的工作库)。
        if created_collection:
            try:
                from store import QdrantStore
                import dataclasses as _dc

                st = QdrantStore(cfg=_dc.replace(settings.qdrant, collection=created_collection))
                if st.exists():
                    st.client.delete_collection(created_collection)
                    print(f"\n(已清理一次性 collection {created_collection})")
                st.close()
            except Exception as exc:  # noqa: BLE001
                print(f"\n(清理 collection 失败:{exc})")

        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"(已删除临时目录 {tmp})")

        # 顺手清掉自检写的基线 —— 它落的是 eval/baselines/ 这个真实目录
        for name in ("_selftest_base.json",):
            p = R.baseline_dir() / name
            if p.exists():
                p.unlink()
                print(f"(已删除自检基线 {p.name})")

    print(f"\n{'=' * 70}")
    if ch.failures:
        print(f"失败 {len(ch.failures)} 项 / 共 {ch.n_checks} 项:")
        for f in ch.failures:
            print(f"  ❌ {f}")
        return 1
    print(f"全部通过 ✅  ({ch.n_checks} 项断言)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
