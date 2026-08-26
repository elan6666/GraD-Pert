#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/data/yilangliu/GraD-Pert")
RUNS = {
    "B0": ROOT / "runs/pilot-b0-metrics-only-7bed/small_results",
    "B1": ROOT / "runs/pilot-b1-graph-only-0a4d/small_results",
    "B2": ROOT / "runs/pilot-b2-systems-only-2e30/small_results",
    "B3": ROOT / "runs/pilot-b3-combined-44ae-r2/small_results",
}
OUT = ROOT / "contracts/pilot-b0-7bed/nadig-jurkat-speed-comparison.json"
HISTORICAL_B0_MANIFEST = (
    ROOT / "runs/formal-v2/smoke/gradpert_b2/nadig_jurkat/seed-1/small_results/run_manifest.json"
)
B0_VALIDATION = ROOT / "contracts/pilot-b0-7bed/b0-metrics-only-strict-validation.json"
B3_VALIDATION = ROOT / "contracts/pilot-b3-44ae-r2/b3-strict-validation.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant(label: str, root: Path) -> dict[str, Any]:
    perf_path = root / "performance_receipt.json"
    perf = load(perf_path)
    manifest_path = root / "run_manifest.json"
    manifest = load(manifest_path)
    with (root / "train_steps.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cells = sum(int(row["batch_cell_count"]) for row in rows)
    wall_s = float(perf["one_epoch_training_wall_ms"]) / 1000.0
    return {
        "label": label,
        "source_commit": manifest["source_commit"],
        "run_id": manifest["run_id"],
        "run_manifest_sha256": sha256(manifest_path),
        "performance_receipt_sha256": sha256(perf_path),
        "checkpoint_sha256": manifest["best_checkpoint_sha256"],
        "graph_axis_policy": perf["graph_axis_policy"],
        "graph_node_count": perf["graph_node_count"],
        "graph_nonself_edge_count": perf["graph_nonself_edge_count"],
        "expression_gene_count": perf["expression_gene_count"],
        "output_gene_count": perf["output_gene_count"],
        "evaluation_gene_count": perf["evaluation_gene_count"],
        "systems_optimizations": perf.get("systems_optimizations", {}),
        "cold_start_ms": perf["cold_start_ms"],
        "cache_build_ms": perf["cache_build_ms"],
        "one_epoch_fit_wall_ms": perf["one_epoch_fit_wall_ms"],
        "one_epoch_training_wall_ms": perf["one_epoch_training_wall_ms"],
        "actual_full_epoch_steps": len(rows),
        "actual_full_epoch_cells": cells,
        "actual_full_epoch_steps_per_second": len(rows) / wall_s,
        "actual_full_epoch_cells_per_second": cells / wall_s,
        "receipt_warmup_steps": perf["warmup_steps"],
        "receipt_measured_steps": perf["measured_steps"],
        "receipt_serial_stage_sum_steps_per_second": perf["steps_per_second"],
        "receipt_serial_stage_sum_cells_per_second": perf["cells_per_second"],
        "validation_ms": perf["validation_ms"],
        "checkpoint_ms": perf["checkpoint_ms"],
        "logging_ms": perf["logging_ms"],
        "mean_stage_ms": perf["mean_stage_ms"],
        "peak_allocated_gpu_bytes": perf["peak_allocated_gpu_bytes"],
        "peak_reserved_gpu_bytes": perf["peak_reserved_gpu_bytes"],
        "peak_cpu_ram_bytes": perf["peak_cpu_ram_bytes"],
        "metrics_non_decisional": {
            row["metric_id"]: row["macro_mean"] for row in perf["headline_metrics_non_decisional"]
        },
    }


def comparison(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    base_wall = float(base["one_epoch_training_wall_ms"])
    candidate_wall = float(candidate["one_epoch_training_wall_ms"])
    return {
        "speedup_x": base_wall / candidate_wall,
        "wall_time_reduction_fraction": 1.0 - candidate_wall / base_wall,
        "cells_per_second_gain_fraction": float(candidate["actual_full_epoch_cells_per_second"])
        / float(base["actual_full_epoch_cells_per_second"])
        - 1.0,
    }


def main() -> None:
    variants = {label: variant(label, root) for label, root in RUNS.items()}
    b0_validation = load(B0_VALIDATION)
    b3_validation = load(B3_VALIDATION)
    if b0_validation["status"] != "PASS" or b3_validation["status"] != "PASS":
        raise RuntimeError("strict validation receipt is not PASS")
    selected_variant = min(
        variants,
        key=lambda label: float(variants[label]["one_epoch_training_wall_ms"]),
    )
    payload = {
        "schema_version": "nadig-jurkat-speed-comparison-v2",
        "status": "PASS",
        "selection_policy": "speed_only_one_epoch_metrics_non_decisional",
        "selected_variant": selected_variant,
        "selected_reason": (
            "lowest actual one-epoch training wall time under the frozen pilot contract"
        ),
        "historical_b0": {
            "rerun": False,
            "role": "immutable historical result; not used as the performance timing baseline",
            "run_manifest_sha256": sha256(HISTORICAL_B0_MANIFEST),
        },
        "metrics_only_b0_rerun": {
            "rerun": True,
            "role": (
                "timing baseline with the same metrics-only artifact and timing protocol as B1-B3"
            ),
            "strict_validation_sha256": sha256(B0_VALIDATION),
        },
        "shared_contract": {
            "dataset_id": "nadig_jurkat",
            "seed": 1,
            "batch_size": 256,
            "prototype_count": 16384,
            "epochs": 1,
            "pytorch_alloc_conf": "expandable_segments:True",
            "expression_output_evaluation_genes": 5000,
            "persistent_pkl_count": 0,
        },
        "variants": variants,
        "direct_factor_comparisons": {
            "B0_to_B1_graph_reduction_without_systems": comparison(variants["B0"], variants["B1"]),
            "B0_to_B2_systems_on_full_graph": comparison(variants["B0"], variants["B2"]),
            "B1_to_B3_systems_on_reduced_graph": comparison(variants["B1"], variants["B3"]),
            "B2_to_B3_graph_reduction_with_systems": comparison(variants["B2"], variants["B3"]),
        },
        "context_only_comparison": {
            "B0_to_B3_combined": comparison(variants["B0"], variants["B3"]),
        },
        "timing_interpretation": {
            "primary": (
                "one_epoch_training_wall_ms and throughput recomputed from the same monotonic "
                "full-epoch wall"
            ),
            "receipt_field_limitation": (
                "For prefetch-enabled B2/B3, measured_end_to_end_wall_ms is a serial sum of "
                "data_read_ms, host_to_device_ms, and step_wall_ms even though data preparation "
                "overlaps GPU computation. The original fields are retained but not used as "
                "actual wall throughput."
            ),
        },
        "effect_interpretation": (
            "The three metrics are recorded only as non-decisional evidence. One epoch cannot "
            "establish unchanged predictive effect, and no effect-equivalence claim is made."
        ),
        "b0_strict_validation_sha256": sha256(B0_VALIDATION),
        "b3_strict_validation_sha256": sha256(B3_VALIDATION),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=OUT.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, OUT)
    print(
        json.dumps(
            {
                "output": str(OUT),
                "sha256": sha256(OUT),
                "selected_variant": selected_variant,
                "direct_factor_comparisons": payload["direct_factor_comparisons"],
                "context_only_comparison": payload["context_only_comparison"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
