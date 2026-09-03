#!/usr/bin/env python3
"""Generate the frozen, self-contained Nadig Jurkat B2-vNext config matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/experiments/gradpert_b2/nadig_jurkat.yaml"
OUTPUT = ROOT / "configs/ablations/nadig_jurkat"
REFERENCE = "docs/design/GRADPERT_VNEXT_ABLATIONS.md"
SUCCESSOR_REFERENCE = "docs/experiments/VNEXT_GRAPH_SCALE_AND_LOCAL_ABLATIONS.md"
DECODER_FACTORIAL_REFERENCE = "docs/experiments/VNEXT_DECODER_FUSION_WIDTH.md"
# GenePT-Seed names this exact artifact ``Seed-GO-ProteinPathway``.  Its
# scientific factor is Protein+Reactome+SIGNOR; no second file is implied.
GENEPT_PROTEIN_REACTOME_SIGNOR_PATH = (
    "/data/yilangliu/GenePT-Seed/data/embeddings/seed-go-protein-pathway-master-aligned.npz"
)
GENEPT_PROTEIN_REACTOME_SIGNOR_SHA256 = (
    "34d4c81b311f567304d299800eb07c8847641f26e82e573f5a1acfe77c202318"
)
TXPERT_CANDIDATE_GENE_SET_SHA256 = (
    "7e2be69a204b72349b793cc6723a5f88419f1ca6472ea5e28c5f7d623ee8e23d"
)

SUCCESSOR_A0 = "a0_ratio_ring_half"
LEGACY_FIXED_BUDGET_VARIANTS = frozenset(
    {
        "a0_default",
        "l1_ring_256",
        "l2_fanout_512",
        "l3_ring_512",
        "l4_anchor_mask_4",
        "g1_canonical_full",
    }
)


@dataclass(frozen=True)
class VariantSpec:
    semantic_factor: str
    changes: dict[str, object]
    declared_parameter_diffs: frozenset[str]


def variant(
    semantic_factor: str,
    changes: dict[str, object] | None = None,
    *,
    derived_diffs: frozenset[str] = frozenset(),
) -> VariantSpec:
    direct = {} if changes is None else changes
    return VariantSpec(
        semantic_factor=semantic_factor,
        changes=direct,
        declared_parameter_diffs=frozenset(direct) | derived_diffs,
    )


def sourced(
    value: object,
    *,
    source: str = "user_locked",
    reference: str = REFERENCE,
) -> dict[str, object]:
    return {"value": value, "source": source, "reference": reference}


def base_config() -> dict[str, object]:
    payload = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    parameters = payload["model"]["parameters"]
    parameters.pop("local_view_node_budget", None)
    parameters.pop("local_anchor_mask_count", None)
    parameters.update(
        {
            "systems_local_activation_checkpointing": sourced(
                True,
                source="project_preregistered",
                reference="docs/experiments/VNEXT_EXACT_EFFECT_PERFORMANCE.md",
            ),
            "performance_pilot_variant": sourced(
                f"vnext_{SUCCESSOR_A0}", reference=SUCCESSOR_REFERENCE
            ),
            "graph_axis_policy": sourced(
                "recomputed_hvg_union_candidate_targets", reference=SUCCESSOR_REFERENCE
            ),
            "graph_hvg_count": sourced(512, reference=SUCCESSOR_REFERENCE),
            "runtime_graph_root": sourced(
                "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets",
                reference=SUCCESSOR_REFERENCE,
            ),
            "graph_sources": sourced("string_go"),
            "graph_encoder_family": sourced("multi_source_sparse_transformer"),
            "string_weight_mode": sourced("selection_only"),
            "graph_encoder_dropout": sourced(0.1),
            "graph_expander_degree": sourced(3),
            "graph_add_reverse_edges": sourced(True),
            "graph_add_self_loops": sourced(True),
            "graph_first_source_local_branch": sourced(True),
            "local_view_builder": sourced("ring_induced", reference=SUCCESSOR_REFERENCE),
            "local_view_count": sourced(4, reference=SUCCESSOR_REFERENCE),
            "local_view_node_budget_ratio": sourced("1/2", reference=SUCCESSOR_REFERENCE),
            "local_view_fanout": sourced("20_10_5_5"),
            "local_anchor_mask_view_ratio": sourced("0/1", reference=SUCCESSOR_REFERENCE),
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
    payload["artifacts"]["root"] = f"runs/ablations/nadig_jurkat/{SUCCESSOR_A0}"
    payload["artifacts"]["result_mode"] = "metrics_only"
    return payload


def variants() -> dict[str, VariantSpec]:
    return {
        SUCCESSOR_A0: variant("reference"),
        "h1_hvg1024_ratio_half": variant(
            "graph_hvg_count",
            {
                "graph_hvg_count": 1024,
                "runtime_graph_root": "vnext/graph_axes/nadig_jurkat/hvg1024_plus_targets",
            },
        ),
        "h2_hvg2048_ratio_half": variant(
            "graph_hvg_count",
            {
                "graph_hvg_count": 2048,
                "runtime_graph_root": "vnext/graph_axes/nadig_jurkat/hvg2048_plus_targets",
            },
        ),
        "h3_hvg5000_ratio_half": variant(
            "graph_hvg_count",
            {
                "graph_hvg_count": 5000,
                "runtime_graph_root": "vnext/graph_axes/nadig_jurkat/hvg5000_plus_targets",
            },
        ),
        "h4_txpert_candidate_ratio_half": variant(
            "graph_gene_universe",
            {
                "graph_axis_policy": "txpert_candidate_gene_universe",
                "graph_hvg_count": 9853,
                "graph_axis_source_sha256": TXPERT_CANDIDATE_GENE_SET_SHA256,
                "runtime_graph_root": ("vnext/graph_axes/nadig_jurkat/txpert_candidate_9853"),
            },
        ),
        "l1_fanout_ratio_half": variant("local_view_builder", {"local_view_builder": "fanout"}),
        "l2_ring_half_count8": variant("local_view_count", {"local_view_count": 8}),
        "l3_ring_quarter": variant(
            "local_view_node_budget_ratio",
            {"local_view_node_budget_ratio": "1/4"},
        ),
        "l4_ring_half_mask_half": variant(
            "local_anchor_mask_view_ratio",
            {"local_anchor_mask_view_ratio": "1/2"},
        ),
        "l5_ring_half_mask_quarter": variant(
            "local_anchor_mask_view_ratio",
            {"local_anchor_mask_view_ratio": "1/4"},
        ),
        "m1_single_string_gat": variant(
            "graph_encoder_family",
            {
                "graph_sources": "string",
                "graph_encoder_family": "single_source_gat",
                "graph_encoder_dropout": 0.2,
            },
        ),
        "m2_single_string_transformer": variant(
            "graph_encoder_family",
            {
                "graph_sources": "string",
                "graph_encoder_family": "single_source_sparse_transformer",
            },
        ),
        "m4_adaptive_source_gat": variant(
            "graph_encoder_family",
            {
                "graph_encoder_family": "adaptive_source_gat_fusion",
                "graph_encoder_dropout": 0.2,
            },
        ),
        "w1_string_edge_feature": variant(
            "string_weight_mode",
            {
                "graph_sources": "string",
                "graph_encoder_family": "single_source_gat",
                "graph_encoder_dropout": 0.2,
                "string_weight_mode": "edge_feature",
            },
        ),
        "w2_string_fixed_prior": variant(
            "string_weight_mode",
            {
                "graph_sources": "string",
                "graph_encoder_family": "single_source_gat",
                "graph_encoder_dropout": 0.2,
                "string_weight_mode": "fixed_prior",
            },
        ),
        "w3_string_prior_residual": variant(
            "string_weight_mode",
            {
                "graph_sources": "string",
                "graph_encoder_family": "single_source_gat",
                "graph_encoder_dropout": 0.2,
                "string_weight_mode": "prior_residual",
            },
        ),
        "ws_string_weight_shuffle": variant(
            "string_weight_mode",
            {
                "graph_sources": "string",
                "graph_encoder_family": "single_source_gat",
                "graph_encoder_dropout": 0.2,
                "string_weight_mode": "shuffled_edge_feature",
            },
        ),
        "d1_control_mlp": variant("decoder_mode", {"decoder_mode": "parameter_matched_mlp"}),
        "d2_control_transformer": variant(
            "decoder_mode", {"decoder_mode": "control_condition_transformer"}
        ),
        "d3_concat_p64": variant("decoder_fusion", {"decoder_mode": "concat"}),
        "d4_concat_transformer_p64": variant(
            "decoder_fusion",
            {"decoder_mode": "concat_transformer"},
        ),
        "d5_concat_p256": variant(
            "decoder_fusion_x_perturbation_width",
            {"decoder_mode": "concat", "graph_tower_output_dim": 256},
        ),
        "d6_concat_transformer_p256": variant(
            "decoder_fusion_x_perturbation_width",
            {"decoder_mode": "concat_transformer", "graph_tower_output_dim": 256},
        ),
        "e1_frozen_genept": variant(
            "gene_feature_mode",
            {"gene_feature_mode": "frozen_genept_projection"},
            derived_diffs=frozenset({"genept_artifact_path", "genept_expected_sha256"}),
        ),
        "e2_genept_id_residual": variant(
            "gene_feature_mode",
            {"gene_feature_mode": "genept_id_residual"},
            derived_diffs=frozenset({"genept_artifact_path", "genept_expected_sha256"}),
        ),
        "e3_genept_initialized": variant(
            "gene_feature_mode",
            {"gene_feature_mode": "genept_initialized"},
            derived_diffs=frozenset({"genept_artifact_path", "genept_expected_sha256"}),
        ),
        "es_genept_shuffle": variant(
            "gene_feature_mode",
            {"gene_feature_mode": "genept_shuffled"},
            derived_diffs=frozenset({"genept_artifact_path", "genept_expected_sha256"}),
        ),
        "o1_no_condition": variant(
            "condition_consistency_loss_weight", {"condition_consistency_loss_weight": 0.0}
        ),
        "o2_no_masked_node": variant("masked_node_loss_weight", {"masked_node_loss_weight": 0.0}),
        "o3_no_spread": variant("spread_loss_weight", {"spread_loss_weight": 0.0}),
    }


def parameter_values(payload: dict[str, object]) -> dict[str, object]:
    parameters = payload["model"]["parameters"]
    return {name: value["value"] for name, value in parameters.items()}


def require_declared_parameter_diff(
    *,
    variant_id: str,
    payload: dict[str, object],
    spec: VariantSpec,
) -> None:
    baseline = parameter_values(base_config())
    observed = parameter_values(payload)
    changed = {
        name
        for name in set(baseline) | set(observed)
        if baseline.get(name, "<missing>") != observed.get(name, "<missing>")
    }
    changed.discard("performance_pilot_variant")
    if changed != spec.declared_parameter_diffs:
        raise RuntimeError(
            f"{variant_id} scientific diff differs: "
            f"expected {sorted(spec.declared_parameter_diffs)}, observed {sorted(changed)}"
        )


def render() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specs = variants()
    collisions = set(specs) & LEGACY_FIXED_BUDGET_VARIANTS
    if collisions:
        raise RuntimeError(
            f"successor IDs collide with fixed-budget lineages: {sorted(collisions)}"
        )
    expected: set[Path] = set()
    rows: list[dict[str, object]] = []
    for name, spec in specs.items():
        payload = copy.deepcopy(base_config())
        parameters = payload["model"]["parameters"]
        successor_row = name == SUCCESSOR_A0 or name.startswith(("h", "l"))
        decoder_factorial_row = name.startswith(("d3_", "d4_", "d5_", "d6_"))
        change_reference = (
            SUCCESSOR_REFERENCE
            if successor_row
            else DECODER_FACTORIAL_REFERENCE
            if decoder_factorial_row
            else REFERENCE
        )
        parameters["performance_pilot_variant"] = sourced(
            f"vnext_{name}", reference=change_reference
        )
        for key, value in spec.changes.items():
            parameters[key] = sourced(value, reference=change_reference)
        if name.startswith(("e1_", "e2_", "e3_", "es_")):
            parameters["genept_expected_sha256"] = sourced(GENEPT_PROTEIN_REACTOME_SIGNOR_SHA256)
            parameters["genept_artifact_path"] = sourced(GENEPT_PROTEIN_REACTOME_SIGNOR_PATH)
        require_declared_parameter_diff(variant_id=name, payload=payload, spec=spec)
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
                "semantic_factor": spec.semantic_factor,
                "declared_parameter_diffs": sorted(spec.declared_parameter_diffs),
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
        "schema_version": "2",
        "matrix_id": "nadig_jurkat_vnext_ratio_graph_v5",
        "design_reference": SUCCESSOR_REFERENCE,
        "decoder_factorial_reference": DECODER_FACTORIAL_REFERENCE,
        "architecture_reference": REFERENCE,
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
