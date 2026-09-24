"""Strict version-two model parameters without importing tensor runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class V2Architecture:
    width: int = 256
    heads: int = 4
    graph_layers: int = 2
    latent_rank: int = 64
    streams: int = 4
    dropout: float = 0.1
    attention: str = "hybrid"
    ffn_type: str = "gelu"
    sparse_topk: int = 500
    sparse_index_dim: int = 64
    sparse_query_chunk: int = 8
    kda_layers: int = 3
    checkpoint_layers: bool = True
    projector_hidden: int = 2048
    projector_bottleneck: int = 256
    prototypes: int = 16384

    def __post_init__(self) -> None:
        for name in (
            "width",
            "heads",
            "graph_layers",
            "latent_rank",
            "streams",
            "projector_hidden",
            "projector_bottleneck",
            "prototypes",
            "sparse_topk",
            "sparse_index_dim",
            "sparse_query_chunk",
            "kda_layers",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("invalid head width or dropout")
        if self.attention not in (
            "hybrid",
            "hybrid_sparse",
            "full_latent",
            "delta_full",
            "full",
            "per_gene",
        ):
            raise ValueError("unknown attention variant")
        if self.ffn_type not in ("gelu", "swiglu"):
            raise ValueError("unknown FFN type")
        if self.sparse_topk < 2:
            raise ValueError("sparse_topk must leave slots for self and CLS")
        if type(self.checkpoint_layers) is not bool:
            raise ValueError("checkpoint_layers must be boolean")

    @classmethod
    def parse(cls, values: dict[str, Any]) -> V2Architecture:
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown v2 architecture fields: {sorted(unknown)}")
        return cls(**values)

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V2Options:
    query_count: int
    microbatch: int
    accumulation: int
    world_size: int
    eval_query_count: int
    graph_expander_degree: int
    graph_expander_seed: int
    lambda1: float
    lambda2: float
    ssl1_condition: float
    ssl1_node: float
    ssl1_spread: float
    ssl2_dino: float
    ssl2_ibot: float
    ssl2_koleo: float
    ssl1_reduction: str
    local_views: int
    global_min_ratio: float
    global_max_ratio: float
    local_min_ratio: float
    local_max_ratio: float
    mask_probability: float
    mask_min_ratio: float
    mask_max_ratio: float
    graph_mask_ratio: float
    graph_edge_dropout: float
    max_conditions: int
    teacher_start: float
    teacher_end: float
    genept_artifact_path: str
    genept_sha256: str
    graph_manifest_path: str
    graph_manifest_sha256: str
    gene_initialization: str
    prediction_loss: str
    loss_reduction: str = "row_mean"
    expression_holdout_path: str = ""
    expression_holdout_sha256: str = ""

    @classmethod
    def parse_parameters(cls, values: dict[str, Any]) -> tuple[V2Architecture, V2Options]:
        arch_names = {f.name for f in fields(V2Architecture)}
        names = {f.name for f in fields(cls)}
        optional = {
            "expression_holdout_path",
            "expression_holdout_sha256",
            "loss_reduction",
            "ffn_type",
            "sparse_topk",
            "sparse_index_dim",
            "sparse_query_chunk",
            "kda_layers",
        }
        required = (arch_names | names) - optional
        if not required <= set(values) or set(values) - (arch_names | names):
            raise ValueError(
                f"v2 missing fields={sorted(required - set(values))}; "
                f"unknown={sorted(set(values) - (arch_names | names))}"
            )
        plain = {name: value.value for name, value in values.items()}
        arch = V2Architecture.parse({name: plain[name] for name in arch_names if name in plain})
        return arch, cls(**{name: plain[name] for name in names if name in plain})

    def __post_init__(self) -> None:
        if self.loss_reduction not in ("row_mean", "condition_mean"):
            raise ValueError("unknown unified loss reduction")
        if bool(self.expression_holdout_path) != bool(self.expression_holdout_sha256):
            raise ValueError("expression holdout manifest path and hash must be supplied together")
        if self.expression_holdout_sha256 and (
            len(self.expression_holdout_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.expression_holdout_sha256)
        ):
            raise ValueError("expression holdout requires a lowercase SHA256")

        for name in (
            "query_count",
            "microbatch",
            "accumulation",
            "world_size",
            "eval_query_count",
            "local_views",
            "max_conditions",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "lambda1",
            "lambda2",
            "ssl1_condition",
            "ssl1_node",
            "ssl1_spread",
            "ssl2_dino",
            "ssl2_ibot",
            "ssl2_koleo",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= value < float("inf")
            ):
                raise ValueError(f"invalid loss weight: {name}")
        for name in ("global", "local", "mask"):
            lo, hi = getattr(self, name + "_min_ratio"), getattr(self, name + "_max_ratio")
            if not 0 < lo <= hi <= 1:
                raise ValueError(f"invalid {name} ratio interval")
        if not 0 <= self.teacher_start <= self.teacher_end <= 1:
            raise ValueError("invalid EMA schedule")
        if self.ssl1_reduction not in ("condition_mean", "row_mean"):
            raise ValueError("invalid SSL1 reduction")
        if self.prediction_loss not in ("cell_mean", "condition_mean"):
            raise ValueError("invalid prediction loss")
        if self.gene_initialization not in ("genept", "random"):
            raise ValueError("invalid gene initialization")
        if self.max_conditions == 1 and self.ssl2_koleo != 0:
            raise ValueError("single-condition batches require SSL2 KoLeo disabled")
        for name in ("mask_probability", "graph_mask_ratio", "graph_edge_dropout"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"invalid probability: {name}")
        for name in ("genept_sha256", "graph_manifest_sha256"):
            digest = getattr(self, name)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name} must be an exact SHA256")
