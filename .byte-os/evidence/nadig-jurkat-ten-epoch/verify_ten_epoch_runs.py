#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

ROOT = Path("/data/yilangliu/GraD-Pert")
SOURCE = ROOT / "source"
OUTPUT = ROOT / "contracts/pilot-ten-epoch-ddf4/ten-epoch-strict-validation.json"
EXPECTED_COMMIT = "ddf40fd14db8c07da1e03ddf381508a2012ac632"
METRIC_IDS = {
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
}
SYSTEMS_REQUESTED = {
    "background_prefetch": True,
    "buffered_training_logs": True,
    "control_expression_cache": True,
    "log_buffer_steps": 64,
    "merged_hdf5_reads": True,
    "nonblocking_transfer": True,
    "pin_memory": True,
    "prefetch_depth": 2,
    "resident_graph_tensors": True,
    "single_checkpoint_serialization": True,
    "validation_expression_cache": True,
}

VARIANTS = {
    "B2": {
        "contract": ROOT / "contracts/pilot-b2-10epoch-ddf4/b2-ten-epoch-contract.json",
        "contract_sha256": "1f787d7b36c3cba7ed90aa1fb45b17d4c1febb109941fe2a8ed1b78c52051f4b",
        "run": ROOT / "runs/pilot-b2-systems-only-10epoch-ddf4",
        "reference": ROOT / "runs/pilot-b2-systems-only-2e30",
        "graph_nodes": 6506,
        "graph_nonself_edges": 222654,
    },
    "B3": {
        "contract": ROOT / "contracts/pilot-b3-10epoch-ddf4-r2/b3-ten-epoch-contract.json",
        "contract_sha256": "54453922eb6a2875ecf4842de02aff5ae12b7b062cdba0a5853637ea8a7f87ca",
        "run": ROOT / "runs/pilot-b3-combined-10epoch-ddf4-r2",
        "reference": ROOT / "runs/pilot-b3-combined-44ae-r2",
        "graph_nodes": 2798,
        "graph_nonself_edges": 89561,
    },
}


def load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def command(*args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(args, cwd=SOURCE, text=True, env=env).strip()


def validate_variant(label: str, spec: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    contract_path = Path(spec["contract"])
    run_root = Path(spec["run"])
    small = run_root / "small_results"
    reference = Path(spec["reference"]) / "small_results"

    contract = load(contract_path)
    manifest = load(small / "run_manifest.json")
    training = load(small / "training_receipt.json")
    training_data = load(small / "training_data.json")
    performance = load(small / "performance_receipt.json")
    systems = load(small / "systems_runtime.json")
    first = load(small / "first_step_equivalence.json")
    retention = load(small / "checkpoint_retention.json")
    recipe = load(small / "inference_recipe.json")
    reference_recipe = load(reference / "inference_recipe.json")
    prediction = load(small / "prediction_manifest.json")
    evaluation = load(small / "evaluation_manifest.json")
    source_identity = load(small / "source_identity.json")
    run_identity = load(small / "run_identity.json")
    test_once = load(small / "test_once.json")

    checkpoint = run_root / "checkpoints/best.pt"
    checkpoint_files = sorted(
        str(path.relative_to(run_root))
        for path in run_root.glob("checkpoints/**/*")
        if path.is_file()
    )
    pkl_files = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*.pkl"))
    work_dirs = sorted(path.name for path in run_root.glob(".result-work-*") if path.is_dir())
    validation_files = sorted(small.glob("validation.epoch-*.json"))

    with (small / "train_steps.csv").open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    global_steps = [int(row["global_step"]) for row in train_rows]
    epochs = [int(row["epoch"]) for row in train_rows]
    total_cells = sum(int(row["batch_cell_count"]) for row in train_rows)

    metric_rows = performance["headline_metrics_non_decisional"]
    metric_values = {row["metric_id"]: row["macro_mean"] for row in metric_rows}
    requested = systems["requested"]
    pipeline = systems["training_pipeline"]
    validation_cache = systems["validation_cache"]
    resident = systems["resident_graph_tensors"]
    ordered_controls = recipe["condition_input_control_row_ids"]
    ordered_truth = recipe["condition_truth_row_ids"]
    reference_controls = reference_recipe["condition_input_control_row_ids"]
    reference_truth = reference_recipe["condition_truth_row_ids"]
    fairness_keys = {
        "canonical_data_sha256",
        "control_manifest_sha256",
        "split_content_sha256",
        "gene_order_sha256",
        "graph_gene_order_sha256",
        "graph_manifest_sha256",
        "graph_topology_content_sha256",
    }

    checks = {
        "contract_hash_exact": sha256(contract_path) == spec["contract_sha256"],
        "source_commit_exact": (
            manifest["source_commit"]
            == source_identity["commit"]
            == contract["source_commit"]
            == EXPECTED_COMMIT
        ),
        "source_clean_in_receipts": not manifest["source_dirty"] and not source_identity["dirty"],
        "config_hash_exact": manifest["config_sha256"] == contract["config_sha256"],
        "run_identity_exact": (
            manifest["run_id"] == contract["run_id"]
            and manifest["model_id"] == contract["model_id"] == "gradpert_b2"
            and manifest["dataset_id"] == contract["dataset_id"] == "nadig_jurkat"
            and manifest["run_seed"] == contract["seed"] == 1
            and manifest["status"] == "evaluated"
            and run_identity["formal_eligible"]
        ),
        "ten_epochs_exact": training["epochs_requested"] == training["epochs_completed"] == 10,
        "optimizer_steps_exact": training["optimizer_steps"] == 5820,
        "train_step_order_exact": (
            len(train_rows) == 5820
            and global_steps == list(range(5820))
            and all(epochs.count(epoch) == 582 for epoch in range(10))
        ),
        "ten_validations_exact": (
            len(validation_files) == 10
            and [path.name for path in validation_files]
            == [f"validation.epoch-{epoch:03d}.json" for epoch in range(10)]
        ),
        "checkpoint_hash_exact": (
            checkpoint.is_file()
            and sha256(checkpoint)
            == manifest["best_checkpoint_sha256"]
            == training["checkpoint_sha256"]
            == recipe["checkpoint_sha256"]
            == retention["best_checkpoint_sha256"]
        ),
        "checkpoint_retention_best_only": (
            checkpoint_files == ["checkpoints/best.pt"] and retention["last_checkpoint_removed"]
        ),
        "no_test_truth_during_fit": training["canonical_test_truth_present_during_fit"] is False,
        "one_test_evaluation": manifest["test_evaluations"] == 1
        and test_once["state"] == "completed",
        "three_metric_schema_exact": (
            {row["metric_id"] for row in metric_rows} == METRIC_IDS and len(metric_rows) == 3
        ),
        "metrics_finite": all(math.isfinite(float(value)) for value in metric_values.values()),
        "expression_output_evaluation_5000": (
            performance["expression_gene_count"]
            == performance["output_gene_count"]
            == performance["evaluation_gene_count"]
            == 5000
        ),
        "graph_axis_exact": (
            performance["graph_axis_policy"] == contract["graph_axis_policy"]
            and performance["graph_node_count"] == spec["graph_nodes"]
            and performance["graph_nonself_edge_count"] == spec["graph_nonself_edges"]
            and training_data["runtime_graph_gene_count"] == spec["graph_nodes"]
        ),
        "fairness_hashes_match_one_epoch": all(
            recipe.get(key) == reference_recipe.get(key) for key in fairness_keys
        ),
        "ordered_controls_match_one_epoch": ordered_controls == reference_controls,
        "ordered_truth_match_one_epoch": ordered_truth == reference_truth,
        "evaluation_conditions_exact": (
            len(evaluation["conditions"])
            == len(prediction["conditions"])
            == len(ordered_controls)
            == len(ordered_truth)
            == 592
        ),
        "prediction_truth_excluded": prediction["truth_included"] is False,
        "metrics_only_recipe": (
            recipe["result_mode"] == "metrics_only"
            and recipe["result_pkl_path"] is None
            and recipe["result_pkl_sha256"] is None
        ),
        "zero_pkl_and_no_work_dir": not pkl_files and not work_dirs,
        "timing_protocol_exact": performance["warmup_steps"] == 10
        and performance["measured_steps"] == 5810,
        "resource_metrics_present": (
            performance["peak_allocated_gpu_bytes"] > 0
            and performance["peak_reserved_gpu_bytes"] > 0
            and performance["peak_cpu_ram_bytes"] > 0
        ),
        "systems_requested_all": requested == SYSTEMS_REQUESTED
        and performance["systems_optimizations"] == SYSTEMS_REQUESTED,
        "control_cache_active": pipeline["control_cache_active"]
        and pipeline["cached_control_batches"] == pipeline["yielded_batches"] == 5820,
        "prefetch_pin_nonblock_active": pipeline["prefetch_active"]
        and pipeline["pin_memory_active"]
        and pipeline["nonblocking_transfer_active"],
        "resident_graph_active": (
            resident["student"]["active"]
            and resident["teacher"]["active"]
            and resident["student"]["node_count"]
            == resident["teacher"]["node_count"]
            == spec["graph_nodes"]
        ),
        "validation_cache_active": validation_cache["active"]
        and validation_cache["cache_hits"] > 0,
        "single_checkpoint_serialization": (
            systems["checkpoint"]["single_serialization_per_epoch"]
            and systems["checkpoint"]["peer_method"] in {"reflink", "copy"}
        ),
        "first_step_evidence_complete": (
            first["update_order"] == ["optimizer_step", "teacher_ema", "center_update"]
            and all(math.isfinite(float(value)) for value in first["losses"].values())
        ),
        "manifest_hash_chain_exact": (
            run_identity["run_manifest_sha256"] == canonical_sha(manifest)
            and recipe["prediction_manifest_sha256"] == sha256(small / "prediction_manifest.json")
            and recipe["evaluation_manifest_sha256"] == sha256(small / "evaluation_manifest.json")
        ),
    }

    wall_seconds = float(performance["one_epoch_training_wall_ms"]) / 1000.0
    evidence = {
        "contract_path": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "run_root": str(run_root),
        "run_manifest_sha256": sha256(small / "run_manifest.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_files": checkpoint_files,
        "pkl_files": pkl_files,
        "work_dirs": work_dirs,
        "epochs": {
            "requested": training["epochs_requested"],
            "completed": training["epochs_completed"],
        },
        "optimizer_steps": training["optimizer_steps"],
        "validation_count": len(validation_files),
        "axes": {
            "expression": performance["expression_gene_count"],
            "output": performance["output_gene_count"],
            "evaluation": performance["evaluation_gene_count"],
            "graph_nodes": performance["graph_node_count"],
            "graph_nonself_edges": performance["graph_nonself_edge_count"],
        },
        "metrics": metric_values,
        "ordered_control_ids_sha256": canonical_sha(ordered_controls),
        "ordered_truth_ids_sha256": canonical_sha(ordered_truth),
        "performance": {
            "training_wall_seconds": wall_seconds,
            "steps_per_second_from_training_wall": len(train_rows) / wall_seconds,
            "cells_per_second_from_training_wall": total_cells / wall_seconds,
            "total_cells": total_cells,
            "peak_allocated_gpu_bytes": performance["peak_allocated_gpu_bytes"],
            "peak_reserved_gpu_bytes": performance["peak_reserved_gpu_bytes"],
            "peak_cpu_ram_bytes": performance["peak_cpu_ram_bytes"],
        },
        "receipt_field_name_limitation": (
            "one_epoch_training_wall_ms and one_epoch_fit_wall_ms are legacy schema field names; "
            "for this fixed run they cover all 10 epochs"
        ),
    }
    return checks, evidence


def main() -> int:
    git_head = command("git", "rev-parse", "HEAD")
    git_clean = not bool(command("git", "status", "--porcelain"))
    proxy_env = dict(os.environ)
    proxy_env.update(
        {
            "ALL_PROXY": "socks5h://127.0.0.1:17897",
            "HTTPS_PROXY": "socks5h://127.0.0.1:17897",
        }
    )
    public_line = command(
        "git",
        "ls-remote",
        "https://github.com/elan6666/GraD-Pert.git",
        "refs/heads/main",
        env=proxy_env,
    )
    public_main = public_line.split()[0]

    checks: dict[str, bool] = {
        "server_source_exact": git_head == EXPECTED_COMMIT,
        "server_source_clean": git_clean,
        "public_main_exact": public_main == EXPECTED_COMMIT,
    }
    variants: dict[str, Any] = {}
    for label, spec in VARIANTS.items():
        variant_checks, evidence = validate_variant(label, spec)
        variants[label] = {"checks": variant_checks, "evidence": evidence}
        checks.update({f"{label}.{key}": value for key, value in variant_checks.items()})

    b2 = variants["B2"]["evidence"]
    b3 = variants["B3"]["evidence"]
    b2_wall = float(b2["performance"]["training_wall_seconds"])
    b3_wall = float(b3["performance"]["training_wall_seconds"])
    metric_deltas = {
        metric_id: float(b3["metrics"][metric_id]) - float(b2["metrics"][metric_id])
        for metric_id in sorted(METRIC_IDS)
    }
    comparison = {
        "selection_policy": "speed_comparison_only_no_effect_equivalence_claim",
        "b3_vs_b2_speedup_x": b2_wall / b3_wall,
        "b3_vs_b2_wall_reduction_fraction": 1.0 - b3_wall / b2_wall,
        "b3_minus_b2_metric_deltas_non_decisional": metric_deltas,
        "concurrency_limitation": (
            "B2 and B3 ran concurrently on separate GPUs and shared host CPU, RAM, and storage; "
            "absolute timing should not be mixed with the earlier sequential one-epoch pilot"
        ),
    }

    failed = sorted(key for key, value in checks.items() if not value)
    receipt = {
        "schema_version": "nadig-jurkat-ten-epoch-validation-v1",
        "status": "PASS" if not failed else "FAIL",
        "source_commit": EXPECTED_COMMIT,
        "checks": checks,
        "failed_checks": failed,
        "variants": variants,
        "comparison": comparison,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=OUTPUT.parent, delete=False
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, OUTPUT)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checks": len(checks),
                "failed": failed,
                "output": str(OUTPUT),
            }
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
