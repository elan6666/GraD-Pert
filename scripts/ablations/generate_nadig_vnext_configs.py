#!/usr/bin/env python3
"""Generate the frozen, self-contained Nadig Jurkat B2-vNext config matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from gradpert.features import GENEPT_EMB_B_SHA256

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/experiments/gradpert_b2/nadig_jurkat.yaml"
OUTPUT = ROOT / "configs/ablations/nadig_jurkat"
REFERENCE = "docs/design/GRADPERT_VNEXT_ABLATIONS.md"
GENEPT_PATH = (
    "/data/yilangliu/trishift/src/data/Data_GeneEmbd/GenePT_gene_embedding_ada_text.pickle"
)


def sourced(value: object, *, source: str = "user_locked") -> dict[str, object]:
    return {"value": value, "source": source, "reference": REFERENCE}


def base_config() -> dict[str, object]:
    payload = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    parameters = payload["model"]["parameters"]
    parameters.update(
        {
            "performance_pilot_variant": sourced("vnext_a0"),
            "graph_axis_policy": sourced("recomputed_hvg_union_candidate_targets"),
            "graph_hvg_count": sourced(512),
            "runtime_graph_root": sourced("vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"),
            "graph_sources": sourced("string_go"),
            "graph_encoder_family": sourced("multi_source_sparse_transformer"),
            "string_weight_mode": sourced("selection_only"),
            "graph_encoder_dropout": sourced(0.1),
            "graph_expander_degree": sourced(3),
            "graph_add_reverse_edges": sourced(True),
            "graph_add_self_loops": sourced(True),
            "graph_first_source_local_branch": sourced(True),
            "local_view_builder": sourced("fanout"),
            "local_view_count": sourced(8),
            "local_view_node_budget": sourced(256),
            "local_view_fanout": sourced("20_10_5_5"),
            "local_anchor_mask_count": sourced(0),
            "gene_feature_mode": sourced("learned_id"),
            "decoder_mode": sourced("additive"),
        }
    )
    training = payload["training"]
    training.update(
        {
            "formal_run_policy": "fixed_epoch_pilot",
            "max_epochs": sourced(10),
            "early_stopping": False,
            "early_stopping_patience": sourced(10),
            "monitor": "val/txpert_macro_pearson_delta",
            "monitor_mode": "max",
            "min_delta": 0.0,
            "run_seeds": [1],
        }
    )
    payload["artifacts"]["root"] = "runs/ablations/nadig_jurkat/vnext_a0"
    payload["artifacts"]["result_mode"] = "metrics_only"
    return payload


def variants() -> dict[str, dict[str, object]]:
    return {
        "a0_default": {},
        "l1_ring_256": {"local_view_builder": "ring_induced"},
        "l2_fanout_512": {"local_view_node_budget": 512},
        "l3_ring_512": {
            "local_view_builder": "ring_induced",
            "local_view_node_budget": 512,
        },
        "l4_anchor_mask_4": {"local_anchor_mask_count": 4},
        "m1_single_string_gat": {
            "graph_sources": "string",
            "graph_encoder_family": "single_source_gat",
            "graph_encoder_dropout": 0.2,
        },
        "m2_single_string_transformer": {
            "graph_sources": "string",
            "graph_encoder_family": "single_source_sparse_transformer",
        },
        "m4_adaptive_source_gat": {
            "graph_encoder_family": "adaptive_source_gat_fusion",
            "graph_encoder_dropout": 0.2,
        },
        "w1_string_edge_feature": {
            "graph_sources": "string",
            "graph_encoder_family": "single_source_gat",
            "graph_encoder_dropout": 0.2,
            "string_weight_mode": "edge_feature",
        },
        "w2_string_fixed_prior": {
            "graph_sources": "string",
            "graph_encoder_family": "single_source_gat",
            "graph_encoder_dropout": 0.2,
            "string_weight_mode": "fixed_prior",
        },
        "w3_string_prior_residual": {
            "graph_sources": "string",
            "graph_encoder_family": "single_source_gat",
            "graph_encoder_dropout": 0.2,
            "string_weight_mode": "prior_residual",
        },
        "ws_string_weight_shuffle": {
            "graph_sources": "string",
            "graph_encoder_family": "single_source_gat",
            "graph_encoder_dropout": 0.2,
            "string_weight_mode": "shuffled_edge_feature",
        },
        "d1_control_mlp": {"decoder_mode": "parameter_matched_mlp"},
        "d2_control_transformer": {"decoder_mode": "control_condition_transformer"},
        "e1_frozen_genept": {"gene_feature_mode": "frozen_genept_projection"},
        "e2_genept_id_residual": {"gene_feature_mode": "genept_id_residual"},
        "e3_genept_initialized": {"gene_feature_mode": "genept_initialized"},
        "es_genept_shuffle": {"gene_feature_mode": "genept_shuffled"},
        "o1_no_condition": {"condition_consistency_loss_weight": 0.0},
        "o2_no_masked_node": {"masked_node_loss_weight": 0.0},
        "o3_no_spread": {"spread_loss_weight": 0.0},
        "g1_canonical_full": {
            "graph_axis_policy": "canonical_full",
            "graph_hvg_count": 5000,
        },
    }


def render() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()
    rows: list[dict[str, object]] = []
    for name, changes in variants().items():
        payload = copy.deepcopy(base_config())
        parameters = payload["model"]["parameters"]
        parameters["performance_pilot_variant"] = sourced(f"vnext_{name}")
        for key, value in changes.items():
            parameters[key] = sourced(value)
        if name.startswith(("e1_", "e2_", "e3_", "es_")):
            parameters["runtime_graph_root"] = sourced(
                "vnext/graph_axes/nadig_jurkat/hvg512_genept_exact"
            )
            parameters["genept_expected_sha256"] = sourced(GENEPT_EMB_B_SHA256)
            parameters["genept_artifact_path"] = sourced(GENEPT_PATH)
        if name == "g1_canonical_full":
            parameters.pop("runtime_graph_root")
        payload["artifacts"]["root"] = f"runs/ablations/nadig_jurkat/{name}"
        destination = OUTPUT / name / "gradpert_b2" / "nadig_jurkat.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected.add(destination)
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        config_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        rows.append(
            {
                "variant_id": name,
                "config_path": str(destination.relative_to(ROOT)),
                "config_sha256": config_sha256,
                "dataset_id": "nadig_jurkat",
                "split_policy": "frozen_canonical",
                "run_seed": 1,
                "max_epochs": 10,
                "result_mode": "metrics_only",
                "genept_preflight_required": name.startswith(("e1_", "e2_", "e3_", "es_")),
            }
        )
    stale = set(OUTPUT.glob("**/*.yaml")) - expected
    if stale:
        raise RuntimeError(
            "refusing to leave stale ablation configs: "
            f"{sorted(str(path.relative_to(ROOT)) for path in stale)}"
        )
    matrix = {
        "schema_version": "1",
        "design_reference": REFERENCE,
        "dataset_id": "nadig_jurkat",
        "canonical_split_count": 1,
        "run_seeds": [1],
        "max_epochs": 10,
        "row_count": len(rows),
        "rows": rows,
    }
    (OUTPUT / "matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render()
