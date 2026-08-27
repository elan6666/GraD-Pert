from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gradpert.config import NativeArchitectureOptions, load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/ablations/nadig_jurkat"


def config_paths() -> tuple[Path, ...]:
    return tuple(sorted(CONFIG_ROOT.glob("*/gradpert_b2/nadig_jurkat.yaml")))


def test_vnext_ablation_matrix_is_self_contained_and_frozen() -> None:
    paths = config_paths()
    assert len(paths) == 22
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
        assert options.local_view_count == 8


def test_vnext_default_is_the_user_confirmed_a0() -> None:
    config = load_experiment_config(CONFIG_ROOT / "a0_default/gradpert_b2/nadig_jurkat.yaml")
    options = NativeArchitectureOptions.from_parameters(config.model.parameters)
    assert options.graph_hvg_count == 512
    assert options.graph_sources == ("string", "go")
    assert options.graph_encoder_family == "multi_source_sparse_transformer"
    assert options.local_view_builder == "fanout"
    assert options.local_view_node_budget == 256
    assert options.local_anchor_mask_count == 0
    assert options.gene_feature_mode == "learned_id"
    assert options.decoder_mode == "additive"


def test_genept_rows_bind_exact_artifact_and_filtered_graph() -> None:
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
            "fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862"
        )
        assert parameters["runtime_graph_root"].value.endswith("hvg512_genept_exact")


def test_vnext_matrix_hash_pins_every_config_before_results() -> None:
    matrix = json.loads((CONFIG_ROOT / "matrix.json").read_text(encoding="utf-8"))
    assert matrix["row_count"] == 22
    assert matrix["run_seeds"] == [1]
    assert matrix["max_epochs"] == 10
    assert len(matrix["rows"]) == 22
    assert len({row["variant_id"] for row in matrix["rows"]}) == 22
    for row in matrix["rows"]:
        path = ROOT / row["config_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["config_sha256"]


def test_each_ablation_differs_from_a0_only_by_its_declared_factor() -> None:
    a0 = load_experiment_config(CONFIG_ROOT / "a0_default/gradpert_b2/nadig_jurkat.yaml")
    base = {name: parameter.value for name, parameter in a0.model.parameters.items()}
    expected = {
        "a0_default": set(),
        "l1_ring_256": {"local_view_builder"},
        "l2_fanout_512": {"local_view_node_budget"},
        "l3_ring_512": {"local_view_builder", "local_view_node_budget"},
        "l4_anchor_mask_4": {"local_anchor_mask_count"},
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
            "runtime_graph_root",
        },
        "e2_genept_id_residual": {
            "gene_feature_mode",
            "genept_artifact_path",
            "genept_expected_sha256",
            "runtime_graph_root",
        },
        "e3_genept_initialized": {
            "gene_feature_mode",
            "genept_artifact_path",
            "genept_expected_sha256",
            "runtime_graph_root",
        },
        "es_genept_shuffle": {
            "gene_feature_mode",
            "genept_artifact_path",
            "genept_expected_sha256",
            "runtime_graph_root",
        },
        "o1_no_condition": {"condition_consistency_loss_weight"},
        "o2_no_masked_node": {"masked_node_loss_weight"},
        "o3_no_spread": {"spread_loss_weight"},
        "g1_canonical_full": {"graph_axis_policy", "graph_hvg_count", "runtime_graph_root"},
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
