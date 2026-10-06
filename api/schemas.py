"""请求体的 pydantic 模型。

为什么用一个显式的模型,而不是让路由收 `dict`:
收 `dict` 的话,`{"qurey": "..."}` 这种拼错的字段会被**静默忽略**,
然后请求以一个空查询跑下去 —— 前端拿到的是一个看起来正常的结果
(或许还是空列表),而真正的问题是字段名拼错了。模型会把未知字段和
类型错误都变成 400。

响应体**不**做模型。原因在 `serialize.py` 的模块头:响应形状是从领域
对象推出来的,再拿一份 pydantic 副本去镜像它,就有两个地方要同步改,
而 pydantic 副本不会因为你改了 `RetrievedChunk` 而报错。请求体则相反 ——
它是外部输入,必须有闸。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["SearchRequest", "AskRequest", "IngestPathRequest"]


def _nonblank(v: str, what: str) -> str:
    """去空白后非空。

    `Field(min_length=1)` 挡不住 `"   "` —— 那个校验看的是长度,而全空格的
    查询在检索里等价于空查询(会走一条没有意义的路径,而不是干脆报错)。
    """
    v = (v or "").strip()
    if not v:
        raise ValueError(f"{what}不能为空")
    return v


class _Strict(BaseModel):
    """拒绝未知字段。

    **默认不拒绝**是 pydantic 的行为(`extra="ignore"`),而在这里恰好是
    最坏的默认值:前端带了拼错的 `top_k_` 时,后端不报错、按默认值跑,
    两边对同一个请求的理解不一致却不报错 —— 这正是最该被拒的一类请求。
    """

    model_config = ConfigDict(extra="forbid")


class SearchRequest(_Strict):
    query: str = Field(..., description="查询文本")
    top_k: int | None = Field(
        default=None, ge=1, le=100, description="返回条数上限,不传取配置"
    )
    backend: str | None = Field(
        default=None, description="hybrid / graphrag;不传 = 用服务端当前后端"
    )
    include_superseded: bool | None = Field(
        default=None, description="是否放行已标记失效的版本;不传 = 服务端默认口径"
    )
    explain: bool = Field(
        default=False,
        description=(
            "true = 把「这条被哪一路召回、在那一路排第几」填进 scores."
            "dense_rank/sparse_rank。代价是额外的两次 Qdrant 查询,默认关"
        ),
    )

    @field_validator("query")
    @classmethod
    def _v_query(cls, v: str) -> str:
        return _nonblank(v, "查询")


class AskRequest(_Strict):
    question: str = Field(..., description="问题")
    backend: str | None = None
    include_superseded: bool | None = None
    allow_write: bool = Field(
        default=False, description="闸 1:请求级写意图。默认 false"
    )
    confirm_write_tools: list[str] = Field(
        default_factory=list,
        description="闸 2:逐个点名的写工具。空列表 = 任何写工具都不批准",
    )
    stream: bool = Field(default=True, description="true=SSE,false=等完整结果")

    @field_validator("question")
    @classmethod
    def _v_question(cls, v: str) -> str:
        return _nonblank(v, "问题")


class IngestPathRequest(_Strict):
    path: str = Field(..., description="服务器上的文件或目录")
    recursive: bool = True
    force: bool = Field(
        default=False,
        description="忽略 content_hash 全部重导。费时,但能补齐缺失的定位语",
    )
    backend: str | None = None

    @field_validator("path")
    @classmethod
    def _v_path(cls, v: str) -> str:
        # webui 那边用户常把带引号的路径粘进来(`"D:\docs"`),这里同样剥掉
        return _nonblank(v, "路径").strip('"')
