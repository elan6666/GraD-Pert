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
        "local_view_node_budget_ratio": "1/2",
        "local_view_fanout": "20_10_5_5",
        "local_anchor_mask_view_ratio": "0/1",
        "gene_feature_mode": "learned_id",
        "decoder_mode": "additive",
    }
    parameters.update(overrides)
    return parameters


def test_vnext_default_resolves_one_hashable_architecture() -> None:
    options = NativeArchitectureOptions.from_parameters(_vnext())
    assert options.graph_sources == ("string", "go")
    assert options.local_view_fanout == (20, 10, 5, 5)
    assert options.local_view_node_budget_ratio_numerator == 1
    assert options.local_view_node_budget_ratio_denominator == 2
    assert options.local_anchor_mask_view_ratio_numerator == 0
    assert options.local_anchor_mask_view_ratio_denominator == 1
    assert len(options.payload_sha256) == 64
    assert options.payload()["schema_version"] == "native-architecture-vnext-2"


def test_minimal_parameters_resolve_ratio_based_defaults() -> None:
    options = NativeArchitectureOptions.from_parameters(
        {"graph_axis_policy": "canonical_full", "graph_sources": "string_go"}
    )
    assert options.graph_encoder_family == "adaptive_relation_gat"
    assert options.local_view_builder == "ring_induced"
    assert options.local_view_node_budget_ratio_numerator == 1
    assert options.local_view_node_budget_ratio_denominator == 2
    assert options.local_anchor_mask_view_ratio_numerator == 0
    assert options.local_anchor_mask_view_ratio_denominator == 1


@pytest.mark.parametrize("graph_hvg_count", [512, 1024, 2048, 5000])
def test_hvg_graph_scales_are_explicitly_supported(graph_hvg_count: int) -> None:
    options = NativeArchitectureOptions.from_parameters(_vnext(graph_hvg_count=graph_hvg_count))
    assert options.graph_hvg_count == graph_hvg_count


@pytest.mark.parametrize(
    ("local_count", "mask_ratio", "expected_mask_ratio"),
    [
        (4, "0", (0, 1)),
        (4, "1/4", (1, 4)),
        (4, "1/2", (1, 2)),
        (8, 0.25, (1, 4)),
        (8, 0.5, (1, 2)),
    ],
)
def test_local_count_and_exact_mask_ratios_are_supported(
    local_count: int,
    mask_ratio: object,
    expected_mask_ratio: tuple[int, int],
) -> None:
    options = NativeArchitectureOptions.from_parameters(
        _vnext(
            local_view_count=local_count,
            local_anchor_mask_view_ratio=mask_ratio,
        )
    )
    assert (
        options.local_anchor_mask_view_ratio_numerator,
        options.local_anchor_mask_view_ratio_denominator,
    ) == expected_mask_ratio


def test_ratio_inputs_are_canonicalized_without_float_arithmetic() -> None:
    options = NativeArchitectureOptions.from_parameters(_vnext(local_view_node_budget_ratio=0.5))
    assert (
        options.local_view_node_budget_ratio_numerator,
        options.local_view_node_budget_ratio_denominator,
    ) == (1, 2)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"graph_hvg_count": 511}, "512, 1024, 2048, or 5000"),
        ({"local_view_fanout": "20_10_5"}, "four positive"),
        (
            {"local_anchor_mask_view_ratio": "1/3"},
            "local_count.*ratio must be an integer",
        ),
        ({"local_view_node_budget_ratio": "0/1"}, r"\(0, 1\]"),
        ({"string_weight_mode": "edge_feature"}, "only by single GAT"),
        ({"graph_sources": "string"}, "multi-source encoders"),
    ],
)
def test_invalid_vnext_combinations_fail_closed(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NativeArchitectureOptions.from_parameters(_vnext(**override))


@pytest.mark.parametrize("legacy_name", ["local_view_node_budget", "local_anchor_mask_count"])
def test_fixed_local_view_factors_fail_closed(legacy_name: str) -> None:
    with pytest.raises(ValueError, match="fixed local-view factors are unsupported"):
        NativeArchitectureOptions.from_parameters(_vnext(**{legacy_name: 4}))


def test_genept_requires_artifact_identity_for_every_route() -> None:
    with pytest.raises(ValueError, match="artifact SHA-256"):
        NativeArchitectureOptions.from_parameters(
            _vnext(gene_feature_mode="frozen_genept_projection")
        )


@pytest.mark.parametrize(
    "feature_mode",
    [
        "frozen_genept_projection",
        "genept_id_residual",
        "genept_initialized",
        "genept_shuffled",
    ],
)
def test_genept_routes_accept_one_hash_pinned_artifact(feature_mode: str) -> None:
    options = NativeArchitectureOptions.from_parameters(
        _vnext(
            gene_feature_mode=feature_mode,
            genept_expected_sha256="a" * 64,
        )
    )
    assert options.gene_feature_mode == feature_mode
    assert options.genept_expected_sha256 == "a" * 64


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
