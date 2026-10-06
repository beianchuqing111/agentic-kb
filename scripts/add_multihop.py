"""补 10 条**真跨文档**的 multi_hop 用例,并把 2 条标错的降级。

为什么要动评估集(待补实证清单 #6)
----------------------------------
`by_type.multi_hop` 原有 3 条,但逐条查 gold 落在哪几篇:

    seed-009  09#2 + 03#2 + 02#1   真跨三篇   ✅
    seed-010  04#2 + 04#5          同一篇     ❌ 单文档多块,分类标错
    seed-011  06#3 + 06#2          同一篇     ❌ 同上

也就是说,「多跳」这个能力**在评估集里其实没被考过**:剩下那两条只要召回
一篇文档就能拿满。所以简历上任何关于多跳的说法,现有评测都支撑不了。

这个脚本补的是**必须跨篇才能作答**的题:每条 gold 至少落在两个不同文件上,
且任一篇单独拿出来都答不全(一部分给「判据 → 等级」,另一部分给
「等级 → 时限」这类接力关系)。

snippet 为什么是**抽**的而不是**抄**的
--------------------------------------
`snippet` 的作用是锚点失效检测(`eval/corpus.py`:比对用
`snippet in chunk.text`)。手抄一段 30 字的中文,抄错一个字,这块就被判成
「锚点失效」——而失效的报错长得跟「检索变差」一模一样,排查方向直接跑偏。
所以这里只写一个**定位用的标记串**(必然存在于目标块的原文片段),
snippet 由程序从真实块文本里切出来,保证字字属实。

三条自检,任一不过就整体不落盘:
  1. 分块参数必须与 `eval/qa/seed.jsonl` 头部 meta 一致(参数一变换,锚点整体位移)
  2. 标记串必须能在目标块里找到
  3. 切出的 snippet 在**全部 61 块**里只能命中目标块这一处
     (命中两处时锚点坏了也发现不了 —— 那是假的安全感)

用法:
    python scripts/add_multihop.py --dry-run   # 只自检,不改文件
    python scripts/add_multihop.py             # 通过自检后写回
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from eval.schema import (  # noqa: E402
    GRADE_ANSWER,
    GRADE_SUPPORT,
    EvalItem,
    EvalSet,
    GoldChunk,
    SNIPPET_CHARS,
    dump_qa,
    load_qa,
)
from ingest.chunker import chunk_text  # noqa: E402
from ingest.loader import discover, load_many  # noqa: E402

QA = ROOT / "eval" / "qa" / "seed.jsonl"
CORPUS = ROOT / "eval" / "corpus" / "seed"

TODAY = "2026-10-06"

# ---------------------------------------------------------------------- #
# 新增用例。每条 gold = (文件, 块序号, grade, 定位标记串)
# 标记串只是用来在块里定位,snippet 由它切出来 —— 见模块开头。
# ---------------------------------------------------------------------- #

NEW = [
    dict(
        id="seed-036",
        question="巡视发现水泥杆倾斜超过千分之十五,应当定为什么等级的缺陷,最晚多久消缺?",
        answer="属于严重缺陷(01 篇的倾斜度判据);严重缺陷应当立即上报并在七日内消缺,"
               "七日内无法消缺的要制定临时安全措施并报上级批准(06 篇的时限)。",
        note="跨文档接力:01#4 给「倾斜度判据 → 等级」,06#3 给「等级 → 时限」。"
             "只召回 06 会不知道这个现象算哪级,只召回 01 会不知道七日的期限。",
        golds=[
            ("01_架空输电线路巡视作业指导书.md", 4, GRADE_ANSWER,
             "水泥杆倾斜度超过千分之十五即应列入严重缺陷"),
            ("06_缺陷定级与消缺管理细则.md", 3, GRADE_ANSWER,
             "严重缺陷应当立即上报并在七日内消缺"),
        ],
    ),
    dict(
        id="seed-037",
        question="从巡视当天算起,发现的一条严重缺陷最晚第几天必须消缺完毕?",
        answer="约第 8 天。巡视结束后二十四小时内要完成缺陷录入(01 篇);"
               "而消缺时限是从「录入系统之日」起算、不是从发现之日起算,"
               "严重缺陷的期限是七日(06 篇)。24 小时 + 7 天。",
        note="这是真需要两篇一起算的题:01#5 给 24 小时录入窗口,"
             "06#3 明确「从录入之日起算,不是从发现之日」并给七日。"
             "只看 06 会答成「发现后 7 天」,少算一天。",
        golds=[
            ("01_架空输电线路巡视作业指导书.md", 5, GRADE_ANSWER,
             "巡视人员应当在二十四小时内完成缺陷录入并上传现场影像资料"),
            ("06_缺陷定级与消缺管理细则.md", 3, GRADE_ANSWER,
             "消缺时限从缺陷录入系统之日起计算,不是从发现之日起计算"),
        ],
    ),
    dict(
        id="seed-038",
        question="主变油温的抄录频次平时是多少,什么情况下要加密到每小时一次?",
        answer="平时每四小时抄录一次(03 篇的值班监视要求);"
               "迎峰度夏期间主变负载率超过百分之八十时,改为每小时记录一次油温和负载电流(09 篇)。",
        note="跨文档接力:09#2 的原文自己就写了「低于该值时按正常周期抄录即可」——"
             "「正常周期」并不在 09 篇里,而在 03#2。这是语料自带的显式跨篇引用。",
        golds=[
            ("03_变电站值班与交接班制度.md", 2, GRADE_ANSWER,
             "值班人员每四小时抄录一次主变油温"),
            ("09_迎峰度夏专项运行措施.md", 2, GRADE_ANSWER,
             "主变负载率超过百分之八十时应当每小时记录一次油温和负载电流"),
        ],
    ),
    dict(
        id="seed-039",
        question="导线断股按截面比例怎么定级,对应的处理方式是什么?",
        answer="断股在百分之七以内属严重缺陷,集中在表层时可用预绞式修补条补修、不切断导线;"
               "超过百分之七属危急缺陷,应当补修或更换并立即处理。",
        note="跨文档:06#2 只给分级标准(7% 这条线划在严重/危急之间),"
             "05#2 只给处理方式(补齐还是补修)。两篇合起来才是「定级 + 怎么办」。",
        golds=[
            ("06_缺陷定级与消缺管理细则.md", 2, GRADE_ANSWER,
             "严重缺陷是指可能危及设备安全运行、需要尽快处理的缺陷"),
            ("05_金具与导地线检查作业指导书.md", 2, GRADE_ANSWER,
             "断股数量在百分之七以内且集中在表层时,可采用预绞式修补条进行补修"),
        ],
    ),
    dict(
        id="seed-040",
        question="事故记录到底要保存多久?相关规定之间有没有不一致?",
        answer="03 篇说值班日志、操作票和事故记录的保存期限不少于三年;"
               "08 篇说涉及事故和设备损坏的资料应当永久保存、不受三年期限的限制。"
               "两者不一致,执行时应按更严的来:事故类资料永久保存。",
        note="跨文档**冲突**题:两篇各给一个期限,必须并列召回才能看出「三年」被"
             "「永久」覆盖。这类题只有多跳检索能答,单篇召回会给出不完整的答案。",
        golds=[
            ("03_变电站值班与交接班制度.md", 5, GRADE_SUPPORT,
             "值班日志、操作票和事故记录都应当归档保存"),
            ("08_设备台账与运行资料归档管理.md", 3, GRADE_ANSWER,
             "涉及事故和设备损坏的资料应当永久保存,不受三年期限的限制"),
        ],
    ),
    dict(
        id="seed-041",
        question="迎峰度夏前备品备件清查重点查哪几类,储备数量依据什么确定?",
        answer="重点核查导线、线夹、绝缘子和熔断器四类易损件(09 篇);"
               "数量的依据来自消缺记录——06 篇要求在系统中回填处理方式和更换部件的型号,"
               "这些信息是备品备件计划的输入。",
        note="跨文档:09#4 给清查范围(四类),06#4 给数量依据(消缺回填的部件型号)。"
             "「查哪几类」和「按什么定数」分处两篇。",
        golds=[
            ("09_迎峰度夏专项运行措施.md", 4, GRADE_ANSWER,
             "重点核查导线、线夹、绝缘子和熔断器四类易损件的储备数量"),
            ("06_缺陷定级与消缺管理细则.md", 4, GRADE_SUPPORT,
             "消缺结果应当在系统中回填处理方式和更换部件的型号"),
        ],
    ),
    dict(
        id="seed-042",
        question="带电作业和红外测温各自的风速上限是多少?",
        answer="带电作业的风速上限是十米每秒,超过即应当停止作业(05 篇);"
               "红外测温更严,风速大于五米每秒即不宜开展(07 篇)。两者不能混用。",
        note="跨文档**易混**题:两个数字分处两篇且量级接近(10 与 5),"
             "只召回一篇会把另一个数字张冠李戴。",
        golds=[
            ("05_金具与导地线检查作业指导书.md", 6, GRADE_ANSWER,
             "风速超过十米每秒时应当停止作业"),
            ("07_红外测温作业指导书.md", 1, GRADE_ANSWER,
             "雨雪天气和风速大于五米每秒时不宜开展测温"),
        ],
    ),
    dict(
        id="seed-043",
        question="复合绝缘子内部受潮这类肉眼看不见的缺陷用什么手段发现?判据和线路接头的温升判据一样吗?",
        answer="都用红外测温:复合绝缘子的导通性缺陷可见光相机完全看不出来,只能用红外测温发现(04 篇);"
               "但判据量级不同——复合绝缘子温差超过零点五开尔文就应列入重点观察,"
               "而线路接头要温升超过十开尔文或相对温差超过百分之八十才算严重缺陷。",
        note="跨文档**对比**题:04#3 给手段与 0.5K 判据,05#2 给接头的 10K / 80% 判据。"
             "回答「一不一样」必须同时拿到两篇。",
        golds=[
            ("04_绝缘子检测作业指导书.md", 3, GRADE_ANSWER,
             "复合绝缘子的异常发热通常表现为局部温升,温差超过零点五开尔文"),
            ("05_金具与导地线检查作业指导书.md", 2, GRADE_ANSWER,
             "接头温升超过环境温度十开尔文,或者相对温差超过百分之八十"),
        ],
    ),
    dict(
        id="seed-044",
        question="迎峰度夏期间重点站所的值班人数怎么调整?这个「两人」的基线是在哪里规定的?",
        answer="重点站所由两人值班调整为三人值班(09 篇);"
               "「两人」的基线出自 03 篇:每班设值班长一名、值班员不少于一名,不得单人值班。",
        note="跨文档接力:09#6 直接拿「两人」当已知量用,却没定义它;"
             "定义在 03#1。只召回 09 答不出基线出处。",
        golds=[
            ("09_迎峰度夏专项运行措施.md", 6, GRADE_ANSWER,
             "重点站所由两人值班调整为三人值班"),
            ("03_变电站值班与交接班制度.md", 1, GRADE_SUPPORT,
             "每班设值班长一名、值班员不少于一名,不得单人值班"),
        ],
    ),
    dict(
        id="seed-045",
        question="测温数据要记录哪些量才能用于后续比对?巡视中的临时测温和正式测温要求一样吗?",
        answer="巡视中红外测温属普查手段,发现异常发热点后至少要记录当时的负荷电流和环境温度,"
               "否则温度数据无法比对(01 篇);正式测温更严,每一次都要记录环境温度、湿度、风速、"
               "负荷电流和测量距离五项,缺一项温升就无法复现(07 篇)。",
        note="跨文档:01#6 给巡视普查的最低要求(两项),07#5 给正式测温的完整要求(五项),"
             "两者是「至少」与「完整」的关系,需并列召回。",
        golds=[
            ("01_架空输电线路巡视作业指导书.md", 6, GRADE_ANSWER,
             "发现异常发热点后应当记录当时的负荷电流和环境温度,否则温度数据无法比对"),
            ("07_红外测温作业指导书.md", 5, GRADE_ANSWER,
             "每一次测温都应当记录环境温度、湿度、风速、负荷电流和测量距离"),
        ],
    ),
]

# 两条标错类型的:gold 全在同一篇文档内,不是多跳。
# 按清单「不要留着充数」的要求降级,并把降级理由写进 note 留痕。
DEMOTE = {
    "seed-010": ("single_hop",
                 "【2026-10-06 降级】原标 multi_hop,但两个 gold(04#2、04#5)都在"
                 "04 篇内,属单文档多块,不是跨文档多跳。降为 single_hop。"),
    "seed-011": ("single_hop",
                 "【2026-10-06 降级】原标 multi_hop,但两个 gold(06#3、06#2)都在"
                 "06 篇内。降为 single_hop。"),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="补真跨文档 multi_hop 用例")
    ap.add_argument("--dry-run", action="store_true", help="只自检,不写文件")
    args = ap.parse_args(argv)

    es = load_qa(QA)
    cfg = get_settings().ingest

    # --- 自检 1:分块参数必须与 meta 一致,否则锚点整体位移 ---
    meta_cs, meta_co = es.meta.get("chunk_size"), es.meta.get("chunk_overlap")
    if (meta_cs, meta_co) != (cfg.chunk_size, cfg.chunk_overlap):
        print(f"!! 分块参数不一致:评估集 meta={meta_cs}/{meta_co},"
              f"当前 ingest={cfg.chunk_size}/{cfg.chunk_overlap}")
        print("   锚点会整体位移,先对齐参数再补题。")
        return 1
    print(f"分块参数 {meta_cs}/{meta_co} 与 ingest 一致 [OK]")

    docs, errs = load_many(discover(CORPUS, recursive=True), on_error="raise")
    if errs:
        print("!! 读取语料出错:", errs)
        return 1
    by_name = {Path(d.source).name: d for d in docs}
    chunks = {name: chunk_text(d.text, cfg.chunk_size, cfg.chunk_overlap)
              for name, d in by_name.items()}
    n_chunks = sum(len(c) for c in chunks.values())
    print(f"语料 {len(docs)} 篇 / {n_chunks} 块")

    # --- 定位标记串 → snippet ---
    all_texts = [(name, i, c.text) for name, cs in chunks.items()
                 for i, c in enumerate(cs)]

    def snippet_for(fname: str, idx: int, marker: str, where: str) -> str:
        if fname not in chunks:
            raise SystemExit(f"!! {where}: 语料里没有 {fname}")
        cs = chunks[fname]
        if not 0 <= idx < len(cs):
            raise SystemExit(f"!! {where}: {fname} 只有 {len(cs)} 块,取不到 #{idx}")
        text = cs[idx].text
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit(f"!! {where}: 标记串不在 {fname}#{idx} 里 —— {marker!r}")
        snip = text[pos:pos + SNIPPET_CHARS]
        if len(snip) < 20:
            raise SystemExit(f"!! {where}: 切出的 snippet 太短({len(snip)} 字),"
                             "标记串太靠近块尾,换一个")
        hits = [(n, i) for n, i, t in all_texts if snip in t]
        if len(hits) != 1 or hits[0] != (fname, idx):
            raise SystemExit(f"!! {where}: snippet 在 {len(hits)} 处命中 {hits} —— "
                             "锚点无法唯一确定目标块,换一个标记串")
        return snip

    # --- 组装新题 ---
    have = {it.id for it in es.items}
    new_items: list[EvalItem] = []
    for spec in NEW:
        if spec["id"] in have:
            raise SystemExit(f"!! id 已存在: {spec['id']}(这个脚本只该跑一次)")
        golds = [
            GoldChunk(file=f, chunk_index=i, grade=g,
                      snippet=snippet_for(f, i, m, f"{spec['id']} {f}#{i}"))
            for f, i, g, m in spec["golds"]
        ]
        files = {g.file for g in golds}
        if len(files) < 2:
            raise SystemExit(f"!! {spec['id']} 的 gold 只落在 {files} —— "
                             "跨文档多跳至少要两个不同文件")
        assert len({(g.file, g.chunk_index) for g in golds}) == len(golds)
        new_items.append(EvalItem(
            id=spec["id"], question=spec["question"], type="multi_hop",
            status="keep", expected=golds, answer=spec["answer"], note=spec["note"],
            added_by="claude", added_at=TODAY,
        ))

    # --- 降级标错的 ---
    demoted = []
    for it in es.items:
        if it.id in DEMOTE:
            new_type, note = DEMOTE[it.id]
            demoted.append(f"{it.id}: {it.type} → {new_type}")
            it.type = new_type
            it.note = (it.note + " " if it.note else "") + note

    es.items.extend(new_items)

    # --- 展示 ---
    print()
    print("新增 multi_hop:")
    for it in new_items:
        print(f"  {it.id}  {it.question}")
        for g in it.expected:
            print(f"      g{g.grade}  {g.file}#{g.chunk_index}  {g.snippet!r}")
    print()
    print("降级:", demoted if demoted else "(无)")
    print()

    errs = es.validate()
    if errs:
        print("!! 校验未通过:")
        for e in errs:
            print("   -", e)
        return 1
    for w in es.warnings():
        print("告警:", w)

    types: dict[str, int] = {}
    for it in es.kept:
        types[it.type] = types.get(it.type, 0) + 1
    print()
    print("题型分布:", json.dumps(types, ensure_ascii=False, sort_keys=True))
    print(f"新 qa.sha1 = {es.sha1()}   ← 与旧基线不再可比(本次是有意改题集)")

    if args.dry_run:
        print("\n(--dry-run,未写文件)")
        return 0

    es.meta = dict(es.meta, generated_at=TODAY)
    header = [
        "人工定稿的种子问答集。一行一条,改 status 即可增删:",
        "  keep  = 参与打分(默认)   draft = 还没看   drop = 明确不要   skip = 暂时跳过",
        "锚点 = 语料目录下的相对文件名 + chunk_index;snippet 是目标块正文里的一段原样文字,",
        "开跑前会逐条比对,对不上就报「锚点失效」并拒绝出分(而不是当成检索失败)。",
        f"{TODAY}: 新增 seed-036~045 十条**跨文档** multi_hop;snippet 由",
        "  scripts/add_multihop.py 从真实块文本切出(非手抄),并校验全库唯一。",
        "  同批把原 seed-010/011 从 multi_hop 降为 single_hop —— 它们的 gold 都在同一篇内,",
        "  不是多跳,留着会让 by_type.multi_hop 虚高。",
    ]
    dump_qa(es, QA, header=header)
    print(f"\n已写 {QA}  ({len(es.items)} 条,其中 keep {len(es.kept)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
