from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gradpert.config import NativeArchitectureOptions, load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/ablations/nadig_jurkat"
SUCCESSOR_A0 = "a0_ratio_ring_half"
LEGACY_FIXED_BUDGET_VARIANTS = {
    "a0_default",
    "l1_ring_256",
    "l2_fanout_512",
    "l3_ring_512",
    "l4_anchor_mask_4",
    "g1_canonical_full",
}


def config_paths() -> tuple[Path, ...]:
    return tuple(sorted(CONFIG_ROOT.glob("*/gradpert_b2/nadig_jurkat.yaml")))


def test_vnext_ablation_matrix_is_self_contained_and_frozen() -> None:
    paths = config_paths()
    assert len(paths) == 25
    for path in paths:
        config = load_experiment_config(path)
        assert config.dataset_id == "nadig_jurkat"
        assert config.training.formal_run_policy == "fixed_epoch_pilot"
        assert config.training.max_epochs.value == 10
        assert config.training.run_seeds == [1]
        assert not config.training.early_stopping
        assert config.artifacts.result_mode == "metrics_only"
        options = NativeArchitectureOptions.from_parameters(config.model.parameters)
        assert options.global_view_count == 2
        assert options.local_view_count in {4, 8}


def test_vnext_default_is_the_successor_ratio_a0() -> None:
    config = load_experiment_config(CONFIG_ROOT / SUCCESSOR_A0 / "gradpert_b2/nadig_jurkat.yaml")
    options = NativeArchitectureOptions.from_parameters(config.model.parameters)
    assert options.graph_hvg_count == 512
    assert options.graph_sources == ("string", "go")
    assert options.graph_encoder_family == "multi_source_sparse_transformer"
    assert options.local_view_builder == "ring_induced"
    assert options.local_view_count == 4
    assert config.model.parameters["local_view_node_budget_ratio"].value == "1/2"
    assert config.model.parameters["local_anchor_mask_view_ratio"].value == "0/1"
    assert "local_view_node_budget" not in config.model.parameters
    assert "local_anchor_mask_count" not in config.model.parameters
    assert options.gene_feature_mode == "learned_id"
    assert options.decoder_mode == "additive"


def test_successor_h_rows_change_only_graph_scale_and_keep_recomputed_axes() -> None:
    expected = {
        "h1_hvg1024_ratio_half": (1024, "hvg1024_plus_targets"),
        "h2_hvg2048_ratio_half": (2048, "hvg2048_plus_targets"),
        "h3_hvg5000_ratio_half": (5000, "hvg5000_plus_targets"),
    }
    for variant_id, (hvg_count, graph_suffix) in expected.items():
        config = load_experiment_config(CONFIG_ROOT / variant_id / "gradpert_b2/nadig_jurkat.yaml")
        parameters = config.model.parameters
        assert parameters["graph_axis_policy"].value == ("recomputed_hvg_union_candidate_targets")
        assert parameters["graph_hvg_count"].value == hvg_count
        assert parameters["runtime_graph_root"].value.endswith(graph_suffix)
        assert parameters["local_view_builder"].value == "ring_induced"
        assert parameters["local_view_node_budget_ratio"].value == "1/2"
        assert parameters["local_view_count"].value == 4
        assert parameters["local_anchor_mask_view_ratio"].value == "0/1"


def test_successor_l_rows_are_exact_one_factor_local_variants() -> None:
    expected = {
        "l1_fanout_ratio_half": ("fanout", "1/2", 4, "0/1"),
        "l2_ring_half_count8": ("ring_induced", "1/2", 8, "0/1"),
        "l3_ring_quarter": ("ring_induced", "1/4", 4, "0/1"),
        "l4_ring_half_mask_half": ("ring_induced", "1/2", 4, "1/2"),
        "l5_ring_half_mask_quarter": ("ring_induced", "1/2", 4, "1/4"),
    }
    for variant_id, values in expected.items():
        config = load_experiment_config(CONFIG_ROOT / variant_id / "gradpert_b2/nadig_jurkat.yaml")
        parameters = config.model.parameters
        assert parameters["graph_hvg_count"].value == 512
        assert (
            parameters["local_view_builder"].value,
            parameters["local_view_node_budget_ratio"].value,
            parameters["local_view_count"].value,
            parameters["local_anchor_mask_view_ratio"].value,
        ) == values


def test_genept_rows_bind_seed_go_protein_pathway_master_and_unfiltered_graph() -> None:
    genept_paths = tuple(
        path
        for path in config_paths()
        if path.parents[1].name.startswith(("e1_", "e2_", "e3_", "es_"))
    )
    assert len(genept_paths) == 4
    for path in genept_paths:
        config = load_experiment_config(path)
        parameters = config.model.parameters
        assert parameters["genept_expected_sha256"].value == (
            "34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318"
        )
        assert parameters["genept_artifact_path"].value == (
            "/data/yilangliu/GenePT-Seed/data/embeddings/seed-go-protein-pathway-master-aligned.npz"
        )
        assert parameters["runtime_graph_root"].value.endswith("hvg512_plus_targets")


def test_vnext_matrix_hash_pins_every_config_before_results() -> None:
    matrix = json.loads((CONFIG_ROOT / "matrix.json").read_text(encoding="utf-8"))
    assert matrix["schema_version"] == "2"
    assert matrix["matrix_id"] == "nadig_jurkat_vnext_ratio_graph_v3"
    assert matrix["row_count"] == 25
    assert matrix["run_seeds"] == [1]
    assert matrix["max_epochs"] == 10
    assert len(matrix["rows"]) == 25
    assert len({row["variant_id"] for row in matrix["rows"]}) == 25
    assert not ({row["variant_id"] for row in matrix["rows"]} & LEGACY_FIXED_BUDGET_VARIANTS)
    for row in matrix["rows"]:
        path = ROOT / row["config_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["config_sha256"]
        assert isinstance(row["semantic_factor"], str)
        assert row["declared_parameter_diffs"] == sorted(row["declared_parameter_diffs"])


def test_each_ablation_differs_from_a0_only_by_its_declared_factor() -> None:
    a0 = load_experiment_config(CONFIG_ROOT / SUCCESSOR_A0 / "gradpert_b2/nadig_jurkat.yaml")
    base = {name: parameter.value for name, parameter in a0.model.parameters.items()}
    expected = {
        SUCCESSOR_A0: set(),
        "h1_hvg1024_ratio_half": {"graph_hvg_count", "runtime_graph_root"},
        "h2_hvg2048_ratio_half": {"graph_hvg_count", "runtime_graph_root"},
        "h3_hvg5000_ratio_half": {"graph_hvg_count", "runtime_graph_root"},
        "l1_fanout_ratio_half": {"local_view_builder"},
        "l2_ring_half_count8": {"local_view_count"},
        "l3_ring_quarter": {"local_view_node_budget_ratio"},
        "l4_ring_half_mask_half": {"local_anchor_mask_view_ratio"},
        "l5_ring_half_mask_quarter": {"local_anchor_mask_view_ratio"},
        "m1_single_string_gat": {
            "graph_sources",
            "graph_encoder_family",
            "graph_encoder_dropout",
        },
        "m2_single_string_transformer": {"graph_sources", "graph_encoder_family"},
        "m4_adaptive_source_gat": {"graph_encoder_family", "graph_encoder_dropout"},
        "w1_string_edge_feature": {
            "graph_sources",
            "graph_encoder_family",
            "graph_encoder_dropout",
            "string_weight_mode",
        },
        "w2_string_fixed_prior": {
            "graph_sources",
            "graph_encoder_family",
            "graph_encoder_dropout",
            "string_weight_mode",
        },
        "w3_string_prior_residual": {
            "graph_sources",
            "graph_encoder_family",
            "graph_encoder_dropout",
            "string_weight_mode",
        },
        "ws_string_weight_shuffle": {
            "graph_sources",
            "graph_encoder_family",
            "graph_encoder_dropout",
            "string_weight_mode",
        },
        "d1_control_mlp": {"decoder_mode"},
        "d2_control_transformer": {"decoder_mode"},
        "e1_frozen_genept": {
            "gene_feature_mode",
            "genept_artifact_path",
            "genept_expected_sha256",
        },
        "e2_genept_id_residual": {
            "gene_feature_mode",
            "genept_artifact_path",
            "genept_expected_sha256",
        },
        "e3_genept_initialized": {
            "gene_feature_mode",
            "genept_artifact_path",
            "genept_expected_sha256",
        },
        "es_genept_shuffle": {
            "gene_feature_mode",
            "genept_artifact_path",
            "genept_expected_sha256",
        },
        "o1_no_condition": {"condition_consistency_loss_weight"},
        "o2_no_masked_node": {"masked_node_loss_weight"},
        "o3_no_spread": {"spread_loss_weight"},
    }
    assert set(expected) == {path.parents[1].name for path in config_paths()}
    for path in config_paths():
        variant_id = path.parents[1].name
        config = load_experiment_config(path)
        observed = {name: parameter.value for name, parameter in config.model.parameters.items()}
        changed = {
            name
            for name in set(base) | set(observed)
            if base.get(name, "<missing>") != observed.get(name, "<missing>")
        }
        changed.discard("performance_pilot_variant")
        assert changed == expected[variant_id]

        row = next(
            item
            for item in json.loads((CONFIG_ROOT / "matrix.json").read_text(encoding="utf-8"))[
                "rows"
            ]
            if item["variant_id"] == variant_id
        )
        assert set(row["declared_parameter_diffs"]) == changed
