# 检索质量评估(`eval/`)

前 6 个 `check_*.py` 验证的是**流程对不对**,验证不了**召回质量好不好**。
改分块大小、换 reranker、调 `rerank_min_score`、改 RRF 的 `k`、加/去定位语 ——
在评估集出现之前,判断好坏的唯一依据是打开界面看几条。有了它,
这些决策从"双方讲道理"变成"改前后各跑一次,看 `hit@5` 和 `by_type` 里的数字"。

**先说钱。这两个脚本一个花钱一个不花:**

| 脚本 | 花钱? | 干什么 |
|---|---|---|
| `scripts\eval_run.py` | **不花** | 跑已有问答集,算指标,存/比基线 |
| `scripts\eval_gen.py` | **花**(调 LLM 出题) | 从语料生成待筛的问答初稿 |

`--dry-run` **一次 API 都不发**,只读语料、算抽样、打印调用数。超过 20 次调用
要求显式 `--yes`。

> 注意 `check_*.py` 那批自检脚本的性质是"不花钱",`eval_gen.py` **不是**自检脚本
> —— 别照着那节的习惯随手跑。

---

## 1. 三条用法

```bat
REM 一、多个 spec 并排跑 —— 最快的用法,一次进程加载模型
REM   注意 preset 是**用 + 拼成一个 spec**,不是两个独立的 flag
python scripts\eval_run.py --preset no-rerank+no-threshold+wide --preset no-threshold+wide

REM 二、存基线再改一个旋钮,看涨跌
python scripts\eval_run.py --preset no-threshold+wide --save-baseline before
python scripts\eval_run.py --preset no-rerank+no-threshold+wide --baseline before

REM 三、回答"定位语到底值多少召回" —— 独立 collection,两种都要重导一次
python scripts\eval_run.py --no-contextual --save-baseline noctx

REM 四、换语料:先出题(花钱),人工筛,再跑(不花钱)
python scripts\eval_gen.py --corpus D:\some\docs --limit 30 --dry-run
python scripts\eval_gen.py --corpus D:\some\docs --limit 30 --yes
REM   ↑ 产物是 eval\qa\<slug>.draft.jsonl,全是 status="draft",**不是评估集**
REM     人工把要留的行复制进 eval\qa\<slug>.jsonl 并改成 status="keep"
python scripts\eval_run.py --corpus D:\some\docs
```

## 2. 挑旋钮挑**真会动**的那个(实测)

种子上"重排开/关"的实测差(其余全同):

| @1 | 重排关 | 重排开 |
|---|---|---|
| hit / mrr | 0.867 | **1.000** |
| ndcg | 0.790 | 0.962 |
| recall | 0.678 | 0.811 |
| **contextual 的 hit@1** | **0.500** | **1.000** |

那一行 `contextual` 就是"定位语 + 重排"合起来值多少钱的具体证据 ——
代词密集段(「该系统」「上述参数」)本身就是靠重排救回来的。

**反面教材:`rrf_k` 在这个语料上根本不动。** 计划里原以为它是"保证会动的
干净演示"(依据是 `store/qdrant_store.py` 那段 k=2 赢者通吃的说明),实测
`--set rrf_k=2` 与 k=60 **逐题完全一致**。原因不是旋钮没接线 ——
`check_store.py` 把两条路的分数打出来后一眼可辨:

```
k=60:  d1 = 1/60+1/60 = .0333    d2 = 1/61+1/61 = .0328
k=2:   d1 = 1/2 +1/2  = 1.0      d2 = 1/3 +1/3  = .667     ← 顺序完全一样
```

RRF 的 k **只改分数的刻度,不改名次**,除非稠密与稀疏两路**互相不同意**。
这个语料的词法和语义排序高度一致,所以 k 怎么调名次都不动。要看它的效果,
得找一个两路分道扬镳的查询 —— 或直接看 `top1_score_mean`(它确实变了:
无阈值下 0.033 是 RRF 分的量级)。

> 这条正是这套设施的价值所在:它把一个"想当然会动"的旋钮变成了
> "拿数字证明它不动"。报告里那句 `(逐题无变化 —— 若聚合也没动,
> 先怀疑配置没生效)` 就是逼你去分辨"真没动"还是"没接线"。

## 3. 四条规矩(改这块之前必须知道)

1. **`k` 有天花板。** `effective_k = min(top_k, fusion_top_k, dense_top_k,
   sparse_top_k)`,不是 `top_k` 想多大就多大。`--top-k 10` 配默认
   `fusion_top_k=5` 时 `hit@10` 是**幻觉数**。runner 会告警并指名两个数字,
   `effective_k` 也记进基线;超过它的 k **从报告里删掉**,不打成 0。

2. **头条数字在"阈值中性"配置下算。** 开着 `rerank_min_score` 时
   `recall@10` 测的**根本不是排序器**,是阈值。两个数字印在一起,人就会得出
   "新重排器更差了"而其实只动了阈值。所以头条用 `no-threshold`
   (`rerank_min_score=0.0, rerank_min_keep=0`),`shipped` 并排报。

3. **反例独立成族,绝不进入任何平均。** 反例的 `expected` 是空的,
   混进平均会给每个数贡献恒定的 0 —— 既稀释,又**对参数变化完全不可见**。
   它们单独按分数分布评(`leak@k` / `top1_score_mean`)。另外
   LlamaIndex 那几个 metric 遇到空 `expected` 会直接 `ValueError`,
   想混也混不进去。

4. **比对模式拒绝跨不可比来源做 diff。** `qa.sha1`、`chunk_size`、
   `contextual_enabled`、`embed_model`、`rerank_model` 任一不同就拒绝并
   指名哪个字段不同。这不是洁癖 —— "改了一行问题、重跑、然后得出
   '检索器变了'的结论"是**必然会发生**的失败,`qa.sha1` 存在的全部理由
   就是拦它。同理 `by_type` 那种 3 题一类的均值是噪声,报告里每类都打 `n`。

## 4. 指标口径

可答题:`hit@k` / `mrr@k` / `ndcg@k`(分级 + 二值并排)/ `precision@k` /
`recall@k`,**按题宏平均**。`--ks 1,3,5,10`,其中 **5 = 出厂
`rerank_top_n`,也就是你今天实际拿到的量**。

指标全部**自己写**(`eval/metrics.py`,纯函数,无模型无网络),不 import
LlamaIndex 的。原因见 [gotchas.md](gotchas.md) 的 `Precision` 那条 ——
它的分母是"返回条数"而不是 `k`,方向是反的。LlamaIndex 的实现只在
`check_eval.py` 里当交叉验证的参照。

`no-threshold` 那档还有个诊断项 `nonempty_rate`,**它是 `rerank_min_keep=1`
的产物,不是质量**(见 [usage.md 的阈值说明](usage.md#阈值rerank_min_keep-默认-1所以检索不到几乎不会发生))
—— 打出来是为了让那个 1.0 有出处。

## 5. 语料与题集规模

| | |
|---|---|
| 种子语料 | `eval/corpus/seed/` 9 篇电力规程,61 个块 |
| 题集 | `eval/qa/seed.jsonl`,20 道(15 道可答 + 5 道反例),`sha1 = cff7b1451a4dd5b3` |
| 评测的 k | `1, 3, 5, 10` |

**这是自建的小评测集,不是公开 benchmark。** 15 道的规模意味着单题涨跌
就能让聚合值动 0.067 —— 也正是为此,报告里每个数字都带 `n`,并且强调
"`by_type` 那种 3 题一类的均值是噪声"。

`eval/baselines/` 下并存**两套题集**的基线:较新的是 **n=25**
(`b1b2_graph`、`b1b2_graph_prefix`、`rerank_off_clean`),较早的是 **n=15**
(`before`、`ct_on`、`ct_off`、`rerank_off`、`rrf60`)。**引用数字前先说清是
哪一套** —— 两套的题不一样,数字不可直接比,`comparability_problems` 也会
拒绝跨套 diff。
