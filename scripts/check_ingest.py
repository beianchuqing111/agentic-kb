"""导入层自检:真建文件、真编码、真落库,跑完清理。

用独立 collection(agentic_kb_selftest_ingest),不碰正式库。
LLM 没配也能跑 —— 上下文增强会优雅降级,只是没有定位语。

跑法:
    set PYTHONIOENCODING=utf-8
    python scripts\\check_ingest.py

重点验证:
  1. **GBK 中文 txt 能读对** —— 这是 Windows 上最容易静默出错的一环。
     读错的表现不是报错,是中文全丢、检索永远为空。
  2. UTF-8 / GBK / 子目录 / 不支持格式,四种情况一次覆盖
  3. 增量:重跑一遍应当「一篇都不重写」
  4. 只改一篇 → 只有那一篇重写,其余跳过(且库里**不出现重复块**)
  5. 分块结果、章节识别、payload 字段落库正确
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from embed import get_embedder  # noqa: E402
from ingest import IngestPipeline, load_file  # noqa: E402
from llm import get_llm  # noqa: E402
from store import QdrantStore  # noqa: E402

TEST_COLLECTION = "agentic_kb_selftest_ingest"

# 故意用 GBK 存这段中文。用 UTF-8 写进去就测不出问题了。
GBK_SENTENCE = "变压器油温超过告警阈值时,应立即启动备用冷却系统,并通知运维人员到现场检查。"

# ⚠️ 这份语料必须**明显超过 chunk_size(默认 512 字符)**,否则整篇只会分出
#    一块,「章节识别」「跨块 section 传递」这些断言等于没测。
#    第一版就是 237 字配 512 的上限,分块数恒为 1,断言全绿得毫无意义。
MD_UTF8 = """# 输电线路巡检作业指导书

## 一、适用范围

本指导书适用于 110kV 及以上架空输电线路的日常巡视与缺陷判定工作,由运维班组按季度组织实施。
巡视周期根据线路所处区域的气象条件、通道环境和历史缺陷率综合确定,一般不短于每季度一次。
对于跨越高速公路、铁路、通航河流的重要跨越段,应当缩短至每月一次并单独建立缺陷台账。
台风、暴雨、覆冰等极端天气前后必须开展特殊巡视,重点检查杆塔基础、导地线弧垂和金具锈蚀情况。

## 二、变压器检查要点

变压器油温超过告警阈值时,应立即启动备用冷却系统,并通知运维人员到现场检查。
冷却系统由油泵、散热器和风扇三部分组成,任一部件故障都会导致油温上升。
主变正常运行时上层油温一般不应超过八十五摄氏度,超过七十五摄氏度即应加强监视。
红外测温是发现变压器局部过热的有效手段,可定位到具体套管或接头位置。
检查呼吸器硅胶颜色变化,受潮变色超过三分之二时应当及时更换,否则变压器油会加速劣化。

## 三、绝缘子检查要点

绝缘子破损是输电线路最常见的缺陷,通常通过无人机航拍图像识别。
复合绝缘子还应重点检查伞裙是否老化开裂、芯棒是否受潮以及端部金具的密封情况。
红外测温可以发现复合绝缘子内部的导通性缺陷,这类缺陷用可见光相机完全看不出来。
零值绝缘子需用专用检测仪逐片测量,发现零值或低值绝缘子应当整串更换,不允许单片替换。

## 四、金具与导线检查要点

金具锈蚀、磨损和变形是长期运行后的常见问题,重点检查耐张线夹、悬垂线夹和防振锤。
导线断股截面超过总截面百分之七时,应当进行补修或更换处理。
弧垂异常往往意味着基础沉降或杆塔倾斜,需要结合基础检查一并判断,不能只看导线。
导线接头处的发热在负荷高峰期最为明显,红外测温应当安排在用电高峰时段进行。

## 五、记录与归档要求

每次巡视结束后,巡视人员应当在二十四小时内完成缺陷录入并上传现场影像资料。
一般缺陷纳入年度检修计划处理,严重缺陷应当立即上报并在七日内消缺。
所有巡视记录、影像和消缺结果保存期限不少于三年,以备事故追溯和运行分析。
"""


SUB_TXT = """变电站值班制度

值班人员每四小时抄录一次主变油温,并登记在运行日志中。
发现油温异常升高时,应当在十五分钟内报告值班长。
"""


def build_tree(root: Path) -> None:
    (root / "a_utf8.md").write_text(MD_UTF8, encoding="utf-8")
    # GBK 存中文 —— 记事本在中文 Windows 上的默认行为
    (root / "b_gbk.txt").write_text(GBK_SENTENCE + "\n第二行也是中文,用来看编码有没有整体错。\n",
                                    encoding="gbk")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c_sub.txt").write_text(SUB_TXT, encoding="utf-8")
    # 不支持的格式:应当被 discover 忽略,不报错
    (root / "d_ignored.csv").write_text("a,b\n1,2\n", encoding="utf-8")


def main() -> int:
    s = get_settings()
    cfg = dataclasses.replace(s.qdrant, collection=TEST_COLLECTION)
    store = QdrantStore(cfg=cfg, retrieval=s.retrieval)

    print("=" * 70)
    h = store.health()
    if not h["ok"]:
        print(f"❌ 连不上 Qdrant: {h['error']}")
        print("   先跑 scripts\\start_qdrant.bat")
        return 1
    print(f"Qdrant ok | LLM configured: {s.llm.configured} "
          f"| 上下文增强开关: {s.ingest.contextual_enabled}")
    print("=" * 70)

    root = Path(tempfile.mkdtemp(prefix="kb_ingest_"))
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"    ✅ {msg}")
        else:
            print(f"    ❌ {msg}")
            failures.append(msg)

    try:
        build_tree(root)
        print(f"\n[0] 测试语料建在 {root}")
        for p in sorted(root.rglob("*")):
            if p.is_file():
                print(f"      {p.relative_to(root)}  ({p.stat().st_size} B)")

        # --- 1. 编码:GBK 文件必须读成正确中文 ---
        print("\n[1] GBK 编码读取")
        gbk_doc = load_file(root / "b_gbk.txt")
        check(gbk_doc.metadata.get("encoding") in ("gb18030", "gbk"),
              f"识别出编码 = {gbk_doc.metadata.get('encoding')}")
        check(GBK_SENTENCE in gbk_doc.text,
              "GBK 中文完整读出(没静默丢字)")
        check("�" not in gbk_doc.text, "无替换字符")

        # --- 2. 分块与章节识别 ---
        print("\n[2] 分块与章节识别")
        md_doc = load_file(root / "a_utf8.md")
        from ingest import chunk_text  # noqa: E402

        chunks = chunk_text(md_doc.text, s.ingest.chunk_size, s.ingest.chunk_overlap)
        check(len(chunks) >= 2,
              f"markdown {md_doc.char_count} 字分出 {len(chunks)} 块 "
              f"(各块 {[c.char_count for c in chunks]})")
        check(all(c.char_count <= s.ingest.chunk_size for c in chunks),
              f"每块都不超 chunk_size={s.ingest.chunk_size}")
        check(all(c.section for c in chunks), "每块都有 section")

        # 核心不变量:**块不跨章节**。一个标题如果在某块里出现,
        # 它必须是那一块的**第一行** —— 否则说明这块横跨了两个标题,
        # section 标签对后半段就是错的。
        import re as _re  # noqa: E402

        spanning = []
        for c in chunks:
            heads = _re.findall(r"^\s{0,3}#{1,6}\s+\S.*$", c.text, _re.MULTILINE)
            for h in heads:
                if not c.text.lstrip().startswith(h.strip()):
                    spanning.append((c.index, h.strip()))
        check(not spanning, f"没有块横跨两个标题(违例: {spanning})")

        # 标题在块内时,section 必须等于那个标题
        matched = [c for c in chunks if c.text.lstrip().startswith("##")]
        check(all(c.section == c.text.lstrip().split("\n")[0].strip() for c in matched),
              f"以标题开头的块,section 就是该标题 "
              f"({[c.section for c in matched]})")

        # 「变压器检查要点」那一节的内容,其 section 必须是变压器那节
        idx = next((i for i, c in enumerate(chunks) if "呼吸器硅胶" in c.text), None)
        if idx is None:
            check(False, "找不到含「呼吸器硅胶」的块")
        else:
            check("变压器" in chunks[idx].section,
                  f"含「呼吸器硅胶」的块 section = {chunks[idx].section!r}")
        # 反例:绝缘子那节不该被标成变压器
        idx2 = next((i for i, c in enumerate(chunks) if "零值绝缘子" in c.text), None)
        if idx2 is not None:
            check("变压器" not in chunks[idx2].section,
                  f"含「零值绝缘子」的块 section = {chunks[idx2].section!r}")

        # --- 3. 首次导入 ---
        print("\n[3] 首次导入")
        store.ensure_collection(dense_dim=get_embedder().dense_dim, recreate=True)
        pipe = IngestPipeline(store=store, cfg=s.ingest, llm=get_llm())
        st1 = pipe.ingest_path(root)
        print(f"    {st1.summary()}")
        check(st1.files_seen == 3, f"扫到 3 个支持的文件(跳过 csv),实得 {st1.files_seen}")
        check(st1.docs_indexed == 3, f"3 篇都导入了,实得 {st1.docs_indexed}")
        check(not st1.errors, f"无错误,实得 {st1.errors}")
        n1 = store.count()
        check(n1 == st1.chunks_written, f"库内 {n1} 条 == 写入 {st1.chunks_written} 条")

        # --- 4. 增量:再跑一遍应当全跳过 ---
        print("\n[4] 增量:原样重跑")
        st2 = pipe.ingest_path(root)
        print(f"    {st2.summary()}")
        check(st2.docs_unchanged == 3, f"3 篇全部判定未变化,实得 {st2.docs_unchanged}")
        check(st2.chunks_written == 0, f"没写任何块,实得 {st2.chunks_written}")
        check(store.count() == n1, f"库内数量不变 ({n1})")
        print("    ↑ 这一条最关键:重复导入既不重复写、也不删数据")

        # --- 5. 只改一篇 ---
        print("\n[5] 增量:只改 sub/c_sub.txt")
        target = root / "sub" / "c_sub.txt"
        target.write_text(
            SUB_TXT + "\n新增条款:交接班时必须核对上一班的油温记录。\n",
            encoding="utf-8",
        )
        st3 = pipe.ingest_path(root)
        print(f"    {st3.summary()}")
        check(st3.docs_indexed == 1, f"只有 1 篇被重写,实得 {st3.docs_indexed}")
        check(st3.docs_unchanged == 2, f"其余 2 篇跳过,实得 {st3.docs_unchanged}")

        # 关键:**不能出现重复块**。改了内容但 doc_id 没变,
        # 走的是「先 delete_by_doc 再重写」,所以总数应当只增加新块那点
        n3 = store.count()
        print(f"    库内 {n1} → {n3} 条")
        docs = store.list_docs()
        for d in docs:
            print(f"      {d['doc_id']}  {d['chunks']} 块  {Path(d['source']).name}")
        check(len(docs) == 3, f"仍然是 3 篇文档,没有分裂,实得 {len(docs)}")
        # 每篇文档的块数应当等于它自己的分块数
        target_doc = load_file(target)
        expect_chunks = len(chunk_text(target_doc.text, s.ingest.chunk_size,
                                       s.ingest.chunk_overlap))
        got = next(d["chunks"] for d in docs if d["doc_id"] == target_doc.doc_id)
        check(got == expect_chunks,
              f"改动那篇 {got} 块 == 重新分块 {expect_chunks} 块(旧块已清掉)")
        check(any("交接班" in (p.payload or {}).get("text", "")
                  for p in store.client.scroll(TEST_COLLECTION, limit=200,
                                               with_payload=True)[0]),
              "新增内容已入库")

        # --- 6. payload 完整性 ---
        # ⚠️ 必须**指定文档**再取点。第一版直接 scroll(limit=1) 抓第一个点,
        #    而 b_gbk.txt 按设计就没有标题、section 本来就是空的 ——
        #    断言随机挑中的是哪篇文档,等于抛硬币。
        print("\n[6] payload 字段")
        md_doc_id = load_file(root / "a_utf8.md").doc_id
        from qdrant_client import models as qm  # noqa: E402

        pts, _ = store.client.scroll(
            TEST_COLLECTION,
            scroll_filter=qm.Filter(must=[qm.FieldCondition(
                key="doc_id", match=qm.MatchValue(value=md_doc_id))]),
            limit=100, with_payload=True,
        )
        check(len(pts) == len(chunks),
              f"markdown 那篇在库里有 {len(pts)} 个点 == 分块数 {len(chunks)}")
        pl = pts[0].payload or {}
        for k in ("text", "doc_id", "chunk_index", "source", "title",
                  "content_hash", "char_count", "ingested_at", "section"):
            check(k in pl, f"payload 有 {k}")
        # context 的**正确行为取决于本机配没配 LLM**,所以断言必须两边都成立,
        # 不能像原来那样写死「为空」。
        # 原来那句在 .env 配上 LLM_API_KEY 之后必然失败,而且失败文案是
        # 「context 为空」—— 说反了,人会以为产品坏了,其实只是机器状态变了。
        # 一个随机器状态翻脸的断言比没有断言更糟:它教人忽略失败。
        # 真正的不变式是「context 为空 ⟺ 没配 LLM」,下面按这个判。
        llm_on = get_settings().llm.configured
        ctx = pl.get("context", "")
        if llm_on:
            check(bool(ctx),
                  f"配了 LLM,context 应有定位语(实际 {ctx[:40]!r}…)")
        else:
            check(ctx == "",
                  "没配 LLM,context 应为空(优雅降级而不是报错)")
        check(all((p.payload or {}).get("section") for p in pts),
              f"该篇每块都带 section,例如 {(pts[0].payload or {}).get('section')!r}")
        check(len({(p.payload or {}).get("chunk_index") for p in pts}) == len(pts),
              "chunk_index 无重复")

        # --- 7. 向量真的写进去了 ---
        print("\n[7] 向量")
        pts, _ = store.client.scroll(TEST_COLLECTION, limit=1, with_vectors=True)
        vec = pts[0].vector
        check(isinstance(vec, dict) and "dense" in vec and "sparse" in vec,
              f"单点同时带稠密+稀疏: {sorted(vec) if isinstance(vec, dict) else type(vec)}")
        if isinstance(vec, dict):
            check(len(vec["dense"]) == 1024, f"稠密维度 {len(vec['dense'])}")
            check(len(vec["sparse"].indices) > 0, f"稀疏非空({len(vec['sparse'].indices)} 个 token)")

        print("\n" + "=" * 70)
        if failures:
            print(f"失败 {len(failures)} 项:")
            for f in failures:
                print(f"  ❌ {f}")
            return 1
        print("全部通过 ✅")
        return 0

    finally:
        try:
            store.client.delete_collection(TEST_COLLECTION)
            print(f"\n(已清理测试 collection {TEST_COLLECTION})")
        except Exception as exc:  # noqa: BLE001
            print(f"\n(清理 collection 失败: {exc})")
        store.close()
        shutil.rmtree(root, ignore_errors=True)
        print(f"(已删除测试语料 {root})")


if __name__ == "__main__":
    raise SystemExit(main())
