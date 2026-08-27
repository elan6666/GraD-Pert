"""Strict native architecture options shared by every GraD-Pert entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal, cast

from gradpert.features import GENEPT_EMB_B_SHA256
from gradpert.hashing import sha256_json

GraphAxisPolicy = Literal[
    "canonical_full",
    "recomputed_top500_union_candidate_targets",
    "recomputed_hvg_union_candidate_targets",
]
GraphEncoderFamily = Literal[
    "adaptive_relation_gat",
    "single_source_gat",
    "single_source_sparse_transformer",
    "multi_source_sparse_transformer",
    "adaptive_source_gat_fusion",
]
StringWeightMode = Literal[
    "selection_only",
    "edge_feature",
    "fixed_prior",
    "prior_residual",
    "shuffled_edge_feature",
]
LocalViewBuilder = Literal["ring_induced", "fanout"]
GeneFeatureMode = Literal[
    "learned_id",
    "frozen_genept_projection",
    "genept_id_residual",
    "genept_initialized",
    "genept_shuffled",
]
DecoderMode = Literal["additive", "parameter_matched_mlp", "control_condition_transformer"]


def _unpack(value: object) -> object:
    return getattr(value, "value", value)


def _value(parameters: Mapping[str, object], name: str, default: object) -> object:
    return _unpack(parameters[name]) if name in parameters else default


def _string(parameters: Mapping[str, object], name: str, default: str) -> str:
    value = _value(parameters, name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"native architecture parameter {name} must be a non-empty string")
    return value


def _integer(parameters: Mapping[str, object], name: str, default: int) -> int:
    value = _value(parameters, name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"native architecture parameter {name} must be an integer")
    return value


def _boolean(parameters: Mapping[str, object], name: str, default: bool) -> bool:
    value = _value(parameters, name, default)
    if not isinstance(value, bool):
        raise ValueError(f"native architecture parameter {name} must be a boolean")
    return value


def _float(parameters: Mapping[str, object], name: str, default: float) -> float:
    value = _value(parameters, name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"native architecture parameter {name} must be numeric")
    return float(value)


def _graph_sources(label: str) -> tuple[str, ...]:
    labels = {
        "string": ("string",),
        "string_go": ("string", "go"),
    }
    try:
        return labels[label]
    except KeyError as error:
        raise ValueError("graph_sources must be string or ordered string_go") from error


def _fanout(label: str) -> tuple[int, int, int, int]:
    parts = label.split("_")
    if len(parts) != 4 or any(not part.isdigit() for part in parts):
        raise ValueError("local_view_fanout must contain four positive integers")
    fanout = cast(tuple[int, int, int, int], tuple(int(part) for part in parts))
    if any(value <= 0 for value in fanout):
        raise ValueError("local_view_fanout must contain four positive integers")
    return fanout


@dataclass(frozen=True)
class NativeArchitectureOptions:
    """One immutable architecture identity resolved before data or model construction."""

    graph_axis_policy: GraphAxisPolicy
    graph_hvg_count: int
    graph_sources: tuple[str, ...]
    graph_encoder_family: GraphEncoderFamily
    string_weight_mode: StringWeightMode
    graph_input_dim: int
    graph_layer_count: int
    graph_head_count: int
    graph_hidden_dim: int
    graph_output_dim: int
    graph_dropout: float
    graph_expander_degree: int
    graph_add_reverse_edges: bool
    graph_add_self_loops: bool
    graph_first_source_local_branch: bool
    global_view_count: int
    local_view_builder: LocalViewBuilder
    local_view_count: int
    local_view_node_budget: int
    local_view_fanout: tuple[int, int, int, int]
    local_anchor_mask_count: int
    gene_feature_mode: GeneFeatureMode
    decoder_mode: DecoderMode
    genept_expected_sha256: str | None

    def __post_init__(self) -> None:
        if self.graph_axis_policy == "recomputed_hvg_union_candidate_targets":
            if self.graph_hvg_count != 512:
                raise ValueError("B2-vNext reduced graph requires exactly 512 HVGs")
        elif self.graph_axis_policy == "recomputed_top500_union_candidate_targets":
            if self.graph_hvg_count != 500:
                raise ValueError("sealed Top500 pilot requires exactly 500 HVGs")
        elif self.graph_hvg_count != 5000:
            raise ValueError("canonical_full requires graph_hvg_count=5000")

        if self.global_view_count != 2 or self.local_view_count != 8:
            raise ValueError("GraD-Pert requires two global and eight local views")
        if self.local_view_node_budget not in {256, 512}:
            raise ValueError("local view node budget must be 256 or 512")
        if self.local_anchor_mask_count not in {0, 4}:
            raise ValueError("local anchor mask count must be 0 or 4")
        if self.local_anchor_mask_count > self.local_view_count:
            raise ValueError("local anchor mask count exceeds local view count")
        if self.local_view_builder == "fanout" and self.local_view_fanout != (20, 10, 5, 5):
            raise ValueError("B2-vNext Fanout requires the frozen 20_10_5_5 schedule")

        expected_dimensions = (128, 4, 2, 128, 64)
        observed_dimensions = (
            self.graph_input_dim,
            self.graph_layer_count,
            self.graph_head_count,
            self.graph_hidden_dim,
            self.graph_output_dim,
        )
        if observed_dimensions != expected_dimensions:
            raise ValueError("native graph encoder dimensions are frozen")
        expected_dropout = {
            "adaptive_relation_gat": 0.1,
            "single_source_gat": 0.2,
            "single_source_sparse_transformer": 0.1,
            "multi_source_sparse_transformer": 0.1,
            "adaptive_source_gat_fusion": 0.2,
        }[self.graph_encoder_family]
        if self.graph_dropout != expected_dropout:
            raise ValueError(
                f"{self.graph_encoder_family} requires graph_encoder_dropout={expected_dropout}"
            )
        if (
            self.graph_expander_degree != 3
            or not self.graph_add_reverse_edges
            or not self.graph_add_self_loops
            or not self.graph_first_source_local_branch
        ):
            raise ValueError("sparse graph Transformer edge/local contracts are frozen")

        single = self.graph_sources == ("string",)
        multi = self.graph_sources == ("string", "go")
        if (
            self.graph_encoder_family
            in {
                "single_source_gat",
                "single_source_sparse_transformer",
            }
            and not single
        ):
            raise ValueError("single-source encoders require graph_sources=string")
        if (
            self.graph_encoder_family
            in {
                "adaptive_relation_gat",
                "multi_source_sparse_transformer",
                "adaptive_source_gat_fusion",
            }
            and not multi
        ):
            raise ValueError("multi-source encoders require ordered graph_sources=string_go")
        if (
            self.graph_encoder_family != "single_source_gat"
            and self.string_weight_mode != "selection_only"
        ):
            raise ValueError("STRING numerical weight modes are supported only by single GAT")

        uses_genept = self.gene_feature_mode != "learned_id"
        if uses_genept:
            if self.genept_expected_sha256 != GENEPT_EMB_B_SHA256:
                raise ValueError("GenePT modes require the exact frozen emb_b SHA-256")
        elif self.genept_expected_sha256 is not None:
            raise ValueError("learned_id mode must not bind an unused GenePT artifact")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, object]) -> NativeArchitectureOptions:
        """Resolve v1-compatible defaults or an explicit vNext architecture."""

        graph_axis_policy = _string(parameters, "graph_axis_policy", "canonical_full")
        default_hvg_count = {
            "canonical_full": 5000,
            "recomputed_top500_union_candidate_targets": 500,
            "recomputed_hvg_union_candidate_targets": 512,
        }.get(graph_axis_policy)
        if default_hvg_count is None:
            raise ValueError(f"unsupported graph_axis_policy: {graph_axis_policy}")
        sources = _graph_sources(_string(parameters, "graph_sources", "string_go"))
        family = _string(parameters, "graph_encoder_family", "adaptive_relation_gat")
        weight_mode = _string(parameters, "string_weight_mode", "selection_only")
        local_builder = _string(parameters, "local_view_builder", "ring_induced")
        gene_feature_mode = _string(parameters, "gene_feature_mode", "learned_id")
        decoder_mode = _string(parameters, "decoder_mode", "additive")
        if family not in {
            "adaptive_relation_gat",
            "single_source_gat",
            "single_source_sparse_transformer",
            "multi_source_sparse_transformer",
            "adaptive_source_gat_fusion",
        }:
            raise ValueError(f"unsupported graph_encoder_family: {family}")
        if weight_mode not in {
            "selection_only",
            "edge_feature",
            "fixed_prior",
            "prior_residual",
            "shuffled_edge_feature",
        }:
            raise ValueError(f"unsupported string_weight_mode: {weight_mode}")
        if local_builder not in {"ring_induced", "fanout"}:
            raise ValueError(f"unsupported local_view_builder: {local_builder}")
        if gene_feature_mode not in {
            "learned_id",
            "frozen_genept_projection",
            "genept_id_residual",
            "genept_initialized",
            "genept_shuffled",
        }:
            raise ValueError(f"unsupported gene_feature_mode: {gene_feature_mode}")
        if decoder_mode not in {
            "additive",
            "parameter_matched_mlp",
            "control_condition_transformer",
        }:
            raise ValueError(f"unsupported decoder_mode: {decoder_mode}")
        genept_sha = _value(parameters, "genept_expected_sha256", None)
        if genept_sha is not None and not isinstance(genept_sha, str):
            raise ValueError("genept_expected_sha256 must be a string when declared")

        return cls(
            graph_axis_policy=cast(GraphAxisPolicy, graph_axis_policy),
            graph_hvg_count=_integer(parameters, "graph_hvg_count", default_hvg_count),
            graph_sources=sources,
            graph_encoder_family=cast(GraphEncoderFamily, family),
            string_weight_mode=cast(StringWeightMode, weight_mode),
            graph_input_dim=_integer(parameters, "gene_embedding_dim", 128),
            graph_layer_count=_integer(parameters, "graph_tower_layers", 4),
            graph_head_count=_integer(parameters, "graph_tower_heads", 2),
            graph_hidden_dim=_integer(parameters, "graph_head_dim", 128),
            graph_output_dim=_integer(parameters, "graph_tower_output_dim", 64),
            graph_dropout=_float(parameters, "graph_encoder_dropout", 0.1),
            graph_expander_degree=_integer(parameters, "graph_expander_degree", 3),
            graph_add_reverse_edges=_boolean(parameters, "graph_add_reverse_edges", True),
            graph_add_self_loops=_boolean(parameters, "graph_add_self_loops", True),
            graph_first_source_local_branch=_boolean(
                parameters, "graph_first_source_local_branch", True
            ),
            global_view_count=_integer(parameters, "global_view_count", 2),
            local_view_builder=cast(LocalViewBuilder, local_builder),
            local_view_count=_integer(parameters, "local_view_count", 8),
            local_view_node_budget=_integer(parameters, "local_view_node_budget", 512),
            local_view_fanout=_fanout(_string(parameters, "local_view_fanout", "20_10_5_5")),
            local_anchor_mask_count=_integer(parameters, "local_anchor_mask_count", 4),
            gene_feature_mode=cast(GeneFeatureMode, gene_feature_mode),
            decoder_mode=cast(DecoderMode, decoder_mode),
            genept_expected_sha256=genept_sha,
        )

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = "native-architecture-vnext-1"
        payload["graph_sources"] = list(self.graph_sources)
        payload["local_view_fanout"] = list(self.local_view_fanout)
        return payload

    @property
    def payload_sha256(self) -> str:
        return sha256_json(self.payload())
