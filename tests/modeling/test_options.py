from __future__ import annotations

import pytest

from gradpert.config.native import NativeArchitectureOptions


def _vnext(**overrides: object) -> dict[str, object]:
    parameters: dict[str, object] = {
        "graph_axis_policy": "recomputed_hvg_union_candidate_targets",
        "graph_hvg_count": 512,
        "graph_sources": "string_go",
        "graph_encoder_family": "multi_source_sparse_transformer",
        "string_weight_mode": "selection_only",
        "local_view_builder": "fanout",
        "local_view_count": 8,
        "local_view_node_budget": 256,
        "local_view_fanout": "20_10_5_5",
        "local_anchor_mask_count": 0,
        "gene_feature_mode": "learned_id",
        "decoder_mode": "additive",
    }
    parameters.update(overrides)
    return parameters


def test_vnext_default_resolves_one_hashable_architecture() -> None:
    options = NativeArchitectureOptions.from_parameters(_vnext())
    assert options.graph_sources == ("string", "go")
    assert options.local_view_fanout == (20, 10, 5, 5)
    assert options.local_anchor_mask_count == 0
    assert len(options.payload_sha256) == 64
    assert options.payload()["schema_version"] == "native-architecture-vnext-1"


def test_historical_parameters_keep_v1_behavior() -> None:
    options = NativeArchitectureOptions.from_parameters(
        {"graph_axis_policy": "canonical_full", "graph_sources": "string_go"}
    )
    assert options.graph_encoder_family == "adaptive_relation_gat"
    assert options.local_view_builder == "ring_induced"
    assert options.local_view_node_budget == 512
    assert options.local_anchor_mask_count == 4


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"graph_hvg_count": 511}, "exactly 512"),
        ({"local_view_fanout": "20_10_5"}, "four positive"),
        ({"local_anchor_mask_count": 2}, "must be 0 or 4"),
        ({"string_weight_mode": "edge_feature"}, "only by single GAT"),
        ({"graph_sources": "string"}, "multi-source encoders"),
    ],
)
def test_invalid_vnext_combinations_fail_closed(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NativeArchitectureOptions.from_parameters(_vnext(**override))


def test_genept_requires_exact_artifact_identity() -> None:
    with pytest.raises(ValueError, match="exact frozen emb_b"):
        NativeArchitectureOptions.from_parameters(
            _vnext(gene_feature_mode="frozen_genept_projection")
        )


def test_single_gat_accepts_registered_weight_ablation() -> None:
    options = NativeArchitectureOptions.from_parameters(
        _vnext(
            graph_sources="string",
            graph_encoder_family="single_source_gat",
            graph_encoder_dropout=0.2,
            string_weight_mode="prior_residual",
        )
    )
    assert options.graph_sources == ("string",)
    assert options.string_weight_mode == "prior_residual"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("graph_encoder_family", "invented_encoder"),
        ("string_weight_mode", "invented_weight_mode"),
        ("local_view_builder", "invented_view"),
        ("gene_feature_mode", "invented_feature"),
        ("decoder_mode", "invented_decoder"),
    ],
)
def test_options_reject_unknown_config_selected_routes(name: str, value: str) -> None:
    parameters = _vnext()
    parameters[name] = value
    with pytest.raises(ValueError, match="unsupported"):
        NativeArchitectureOptions.from_parameters(parameters)
