"""
agentic-kb 集中配置。

约定:
- 可调参数一律收在 dataclass 里,业务代码不出现魔法值
- 路径相对 BASE_DIR,不依赖 cwd
- 密钥只从 .env 读,源码里不写任何默认值
  (内联默认密钥是最常见的泄漏路径 —— 文件被复制/提交时没人会注意到)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------------------------------- #
# env 读取小工具
# --------------------------------------------------------------------------- #

def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def detect_device() -> str:
    """torch 在不在、CUDA 能不能用。编码 1.2G 的两个模型,CPU 上慢十倍以上。"""
    override = _env("KB_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# HuggingFace 离线
# --------------------------------------------------------------------------- #
# 必须**在 huggingface_hub 被导入之前**设置,所以放在 config 模块的导入期。
# 项目里任何模块都先 import config,huggingface_hub 则是后面由 FlagEmbedding
# 惰性导入的,顺序天然满足。
#
# 为什么非关不可:模型已经完整缓存在本地时,BGEM3FlagModel 加载**仍然**会向
# huggingface.co 发一次 HEAD 去查最新版本。这一次请求的成败和模型能不能加载
# **毫无关系**,但它失败时异常会一路冒出来,把整篇文档的导入搞挂。
# 实测代价:一次代理抖动 → 5 分钟重试 → a_utf8.md 直接没进库,
# 只在 stats.errors 里留一行。这种故障最难查 —— 网络是别人的,数据是你的。
#
# 首次下载模型时把 KB_HF_ONLINE=1 打开,下完再关掉。
_HF_ONLINE = _env_bool("KB_HF_ONLINE", False)
if not _HF_ONLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # 别让 datasets/accelerate 之类的顺带去做版本检查
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def hf_offline() -> bool:
    return not _HF_ONLINE


def assert_hf_offline_env() -> None:
    """在构造 HF 模型前再确认一次环境变量。

    config 通常是最先被导入的,但万一有别的调用路径先导入了
    huggingface_hub,那时它的常量已经定型,env 再设也不生效 ——
    这里显式报出来,好过让人对着一个网络错误查半天。
    """
    if hf_offline() and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError(
            "HF_HUB_OFFLINE 没设上。多半是 huggingface_hub 在 config 之前就被导入了,"
            "它把离线常量在导入期定死了。\n"
            "解决:确保任何 HF 相关导入之前先 `import config`,"
            "或在进程启动时就设好环境变量 HF_HUB_OFFLINE=1。"
        )


# --------------------------------------------------------------------------- #
# 后端切换
# --------------------------------------------------------------------------- #

class BackendType(str, Enum):
    """知识库后端。两条路都在 LlamaIndex 里,配置里一个字段切换。"""

    HYBRID = "hybrid"      # bge-m3 稠密+稀疏 → Qdrant → RRF → bge-reranker-v2
    GRAPHRAG = "graphrag"  # LlamaIndex PropertyGraphStore → Neo4j (+ Qdrant 存向量)


# --------------------------------------------------------------------------- #
# 各子系统配置
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LLMConfig:
    """OpenAI 兼容接口。中文语料务必选中文强的模型。"""

    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.moonshot.cn/v1"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "kimi-k3"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))
    timeout: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 120.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 4))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class EmbedConfig:
    """bge-m3:BGE-M3 - Multi-Linguality, Multi-Functionality, Multi-Granularity。

    一次前向同时出稠密向量和稀疏权重(lexical_weights),稀疏不是第二次推理。
    """

    model_name: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-m3"))
    device: str = field(default_factory=detect_device)
    use_fp16: bool = field(default_factory=lambda: _env_bool("EMBED_FP16", True))
    batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH", 12))
    max_length: int = field(default_factory=lambda: _env_int("EMBED_MAX_LEN", 8192))
    dense_dim: int = field(default_factory=lambda: _env_int("EMBED_DENSE_DIM", 1024))

    # 稀疏来源:"m3" = bge-m3 学到的权重;"bm25" = 自建中文 BM25
    sparse_source: str = field(default_factory=lambda: _env("SPARSE_SOURCE", "m3"))


@dataclass(frozen=True)
class BM25Config:
    """自建中文 BM25 稀疏向量。

    关键设计:IDF 放查询侧、tf 饱和放文档侧,于是
        score(q,d) = Σ_t IDF(t)·tf_sat(t,d) = q·d
    这就是一个点积 —— 所以 BM25 能当稀疏向量存进 Qdrant。
    IDF 在查询侧意味着语料增长不需要重建索引,只需维护 document_frequency。

    avgdl 必须冻结:一旦文档向量按旧的 avgdl 算好存进库,改 avgdl 就作废了。
    """

    avgdl: float = field(default_factory=lambda: _env_float("BM25_AVGDL", 512.0))
    k1: float = field(default_factory=lambda: _env_float("BM25_K1", 1.5))
    b: float = field(default_factory=lambda: _env_float("BM25_B", 0.75))
    # 词→id 映射的持久化路径。索引和查询必须用同一份映射,
    # 不一致会静默失效 —— 不报错,只是所有稀疏匹配都落空。
    vocab_path: Path = field(default_factory=lambda: CACHE_DIR / "bm25_vocab.json")


@dataclass(frozen=True)
class QdrantConfig:
    """独立实例:独立端口 + 独立 storage 目录。

    这台机器上 imgsearch 也用 qdrant.exe(默认 6333,storage 在 imgsearch 目录下)。
    共用实例会互相污染数据、共占端口,所以这里另起一个。
    """

    url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://127.0.0.1:6343"))
    api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "agentic_kb"))
    # 单 collection 双字段:稠密走 named vector,稀疏走 sparse vector
    dense_field: str = "dense"
    sparse_field: str = "sparse"
    timeout: int = field(default_factory=lambda: _env_int("QDRANT_TIMEOUT", 30))

    # 下面两个只给 scripts/start_qdrant.bat 用
    exe: str = field(
        default_factory=lambda: _env("QDRANT_EXE", r"D:\imgsearch\qdrant_server\qdrant.exe")
    )
    storage: Path = field(
        default_factory=lambda: Path(_env("QDRANT_STORAGE", str(BASE_DIR / "qdrant_storage")))
    )
    http_port: int = field(default_factory=lambda: _env_int("QDRANT_HTTP_PORT", 6343))
    grpc_port: int = field(default_factory=lambda: _env_int("QDRANT_GRPC_PORT", 6344))


@dataclass(frozen=True)
class Neo4jConfig:
    """LlamaIndex PropertyGraphStore 的图存储。

    中文注意:Neo4j 全文索引底层是 Lucene,中文分词要显式配 analyzer,
    可选 cjk(Lucene 的 CJKAnalyzer,按双字切分,中文最稳)。
    """

    uri: str = field(default_factory=lambda: _env("NEO4J_URI"))
    username: str = field(default_factory=lambda: _env("NEO4J_USERNAME"))
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD"))
    database: str = field(default_factory=lambda: _env("NEO4J_DATABASE"))
    max_connection_lifetime: int = 300

    @property
    def configured(self) -> bool:
        return bool(self.uri and self.username and self.password)


@dataclass(frozen=True)
class IngestConfig:
    """导入。Anthropic 上下文感知检索是这一层的事,跟检索算法无关。"""

    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 64))
    # 上下文感知:每块先让 LLM 写一句定位语,再拼到块前面一起编码
    contextual_enabled: bool = field(default_factory=lambda: _env_bool("CONTEXTUAL_ENABLED", True))
    # 上下文字数上限 —— 太长会淹没原文
    contextual_max_chars: int = field(default_factory=lambda: _env_int("CONTEXTUAL_MAX_CHARS", 200))
    # 上下文增强的并发度
    contextual_workers: int = field(default_factory=lambda: _env_int("CONTEXTUAL_WORKERS", 4))
    # 分块时向前看多少字判断边界
    paragraph_sep: str = "\n\n"


@dataclass(frozen=True)
class RetrievalConfig:
    """混合检索:双路召回 → RRF 融合 → 交叉编码器重排。

    RRF 是纯排名融合 1/(k+rank),与分数量纲无关;
    LlamaIndex 的 Qdrant hybrid 默认是 Relative Score Fusion,不是 RRF ——
    那个要把稠密和稀疏两套不可比的分数归一化后加权,坑在这里。
    """

    dense_top_k: int = field(default_factory=lambda: _env_int("DENSE_TOP_K", 50))
    sparse_top_k: int = field(default_factory=lambda: _env_int("SPARSE_TOP_K", 50))
    fusion_top_k: int = field(default_factory=lambda: _env_int("FUSION_TOP_K", 20))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # 重排:交叉编码器必须客户端做 —— Qdrant 服务端只能做 RRF,跑不了 ONNX 交叉编码器
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("RERANK_ENABLED", True))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 5))
    rerank_max_length: int = field(default_factory=lambda: _env_int("RERANK_MAX_LEN", 1024))

    # 分数下限。bge-reranker 过 sigmoid 后的分布**极其尖锐** ——
    # 实测相关文档 0.91~0.98,不相关的全在 0.04 以下。固定取 top-5
    # 会把 4 条近乎零分的垃圾一起塞给 LLM,既浪费上下文又干扰判断。
    # 低于这个分数的不返回;但至少保留 rerank_min_keep 条,避免一条都不剩。
    rerank_min_score: float = field(default_factory=lambda: _env_float("RERANK_MIN_SCORE", 0.05))
    rerank_min_keep: int = field(default_factory=lambda: _env_int("RERANK_MIN_KEEP", 1))


@dataclass(frozen=True)
class GraphRAGConfig:
    """GraphRAG 后端配置。用的是 LlamaIndex 的 `SimpleLLMPathExtractor`
    (实体抽取)+ `PropertyGraphStore`(存储),**不是** `PropertyGraphIndex`
    —— 写入流水线与多跳检索都是自研的,理由见 `ingest/graph_pipeline.py` 开头。

    中文支持的四个着力点:
      1. extract_prompt  —— 默认提示词是英文,这里换中文
      2. LLM 本身        —— 中文强的模型比提示词更重要
      3. 图库 analyzer   —— Neo4j 全文索引配 cjk
      4. 实体对齐        —— 「北京」/「北京市」/「首都」得合并,这层最容易被漏掉
    """

    # 实体类型白名单。留空 = 让 LLM 自己定(更灵活,质量更不稳)
    entity_types: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            t.strip() for t in _env("GRAPH_ENTITY_TYPES").split(",") if t.strip()
        )
    )
    relation_types: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            t.strip() for t in _env("GRAPH_RELATION_TYPES").split(",") if t.strip()
        )
    )
    max_paths_per_chunk: int = field(default_factory=lambda: _env_int("GRAPH_MAX_PATHS", 10))
    extraction_workers: int = field(default_factory=lambda: _env_int("GRAPH_WORKERS", 4))

    # 实体对齐:同一个实体被写成不同名字时,靠向量相似度合并
    entity_merge_enabled: bool = field(default_factory=lambda: _env_bool("GRAPH_MERGE_ENTITIES", True))
    entity_merge_threshold: float = field(
        default_factory=lambda: _env_float("GRAPH_MERGE_THRESHOLD", 0.92)
    )

    # 向量检索器指向哪个 collection(复用混合检索那套 Qdrant)
    vector_top_k: int = field(default_factory=lambda: _env_int("GRAPH_VECTOR_TOP_K", 10))

    # 图查询走多跳时的跳数上限 —— 没有 visited 集合的多跳在带环图上会路径爆炸
    max_hops: int = field(default_factory=lambda: _env_int("GRAPH_MAX_HOPS", 2))


@dataclass(frozen=True)
class WebSearchConfig:
    api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    max_results: int = field(default_factory=lambda: _env_int("WEB_MAX_RESULTS", 5))
    timeout: float = field(default_factory=lambda: _env_float("WEB_TIMEOUT", 30.0))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class AgentConfig:
    """ReAct 循环。"""

    max_iterations: int = field(default_factory=lambda: _env_int("AGENT_MAX_ITER", 8))
    # 单条工具结果塞进历史的字符上限。
    # 不截断的后果是灾难性的:一次误操作读了个大文件,历史涨到几十 MB,
    # 下一轮请求直接 400(超模型上限),而且日志里看不出来。
    tool_result_max_chars: int = field(default_factory=lambda: _env_int("AGENT_TOOL_MAX_CHARS", 8000))
    # 工具输出保留首尾 —— 报错通常在末尾
    tool_result_head_ratio: float = 0.6
    # 输出不符合 ReAct 格式时,最多纠正几次。超过之后不再空转,把模型
    # 最后那段文本原样当答案返回(见 react.ReActAgent.run)。
    max_format_retries: int = field(default_factory=lambda: _env_int("AGENT_FORMAT_RETRIES", 2))
    # ReAct 要的是格式稳定,不是文采。默认 0,别去调高。
    temperature: float = field(default_factory=lambda: _env_float("AGENT_TEMPERATURE", 0.0))


@dataclass(frozen=True)
class Settings:
    backend: BackendType = field(
        default_factory=lambda: BackendType(_env("KB_BACKEND", BackendType.HYBRID.value))
    )
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    web: WebSearchConfig = field(default_factory=WebSearchConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def ensure_dirs(self) -> None:
        for path in (DATA_DIR, LOG_DIR, CACHE_DIR):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()


def get_settings() -> Settings:
    return settings
