#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path("/data/yilangliu/GraD-Pert")
SOURCE = ROOT / "source"
CONTRACT = ROOT / "contracts/pilot-b3-44ae-r2/gradpert-b3-44ae-r2-contract.json"
OUTPUT = ROOT / "contracts/pilot-b3-44ae-r2/b3-strict-validation.json"
B0 = ROOT / "runs/formal-v2/smoke/gradpert_b2/nadig_jurkat/seed-1/small_results"
B1 = ROOT / "runs/pilot-b1-graph-only-0a4d/small_results"
B2 = ROOT / "runs/pilot-b2-systems-only-2e30/small_results"
B3_ROOT = ROOT / "runs/pilot-b3-combined-44ae-r2"
B3 = B3_ROOT / "small_results"
B2_VALIDATION = ROOT / "contracts/pilot-b2-2e30/b2-strict-validation.json"
EXPECTED_COMMIT = "44ae7ff7ec2df7a91af8294ff7207e9795437d48"
EXPECTED_CONFIG_SHA = "4c63dac6afd48d25680d8354e9d554b01c406e57e25790d023d93e4552b23a1c"
EXPECTED_RUN_ID = "pilot-b3-combined__nadig_jurkat__seed-1__44ae7ff-r2"
METRIC_IDS = {
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def full_epoch_throughput(small_root: Path) -> dict[str, float | int]:
    perf = load(small_root / "performance_receipt.json")
    with (small_root / "train_steps.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cells = sum(int(row["batch_cell_count"]) for row in rows)
    wall_seconds = float(perf["one_epoch_training_wall_ms"]) / 1000.0
    return {
        "steps": len(rows),
        "cells": cells,
        "wall_seconds": wall_seconds,
        "steps_per_second": len(rows) / wall_seconds,
        "cells_per_second": cells / wall_seconds,
    }


def main() -> int:
    contract = load(CONTRACT)
    b0_manifest = load(B0 / "run_manifest.json")
    b1_manifest = load(B1 / "run_manifest.json")
    b1_recipe = load(B1 / "inference_recipe.json")
    b2_recipe = load(B2 / "inference_recipe.json")
    b2_validation = load(B2_VALIDATION)
    manifest = load(B3 / "run_manifest.json")
    training = load(B3 / "training_receipt.json")
    training_data = load(B3 / "training_data.json")
    performance = load(B3 / "performance_receipt.json")
    systems = load(B3 / "systems_runtime.json")
    first = load(B3 / "first_step_equivalence.json")
    retention = load(B3 / "checkpoint_retention.json")
    recipe = load(B3 / "inference_recipe.json")
    prediction = load(B3 / "prediction_manifest.json")
    evaluation = load(B3 / "evaluation_manifest.json")
    metrics = load(B3 / "metrics_summary.json")
    source_identity = load(B3 / "source_identity.json")
    run_identity = load(B3 / "run_identity.json")
    test_once = load(B3 / "test_once.json")

    checkpoint = B3_ROOT / "checkpoints/best.pt"
    checkpoint_files = sorted(
        str(path.relative_to(B3_ROOT)) for path in B3_ROOT.glob("checkpoints/**/*") if path.is_file()
    )
    pkl_files = sorted(str(path.relative_to(B3_ROOT)) for path in B3_ROOT.rglob("*.pkl"))
    work_dirs = sorted(path.name for path in B3_ROOT.glob(".result-work-*") if path.is_dir())

    git_head = command("git", "rev-parse", "HEAD")
    git_dirty = bool(command("git", "status", "--porcelain"))
    proxy_env = dict(os.environ)
    proxy_env.update(
        {
            "ALL_PROXY": "socks5h://127.0.0.1:17897",
            "HTTPS_PROXY": "socks5h://127.0.0.1:17897",
        }
    )
    public_line = command(
        "git", "ls-remote", "https://github.com/elan6666/GraD-Pert.git", "refs/heads/main", env=proxy_env
    )
    public_main = public_line.split()[0]

    metric_rows = performance["headline_metrics_non_decisional"]
    metric_values = {row["metric_id"]: row["macro_mean"] for row in metric_rows}
    stage_fields = {
        "data_read_ms",
        "host_to_device_ms",
        "view_build_ms",
        "teacher_forward_ms",
        "student_global_ms",
        "student_local_ms",
        "prediction_ms",
        "backward_update_ms",
        "step_wall_ms",
    }
    requested = systems["requested"]
    pipeline = systems["training_pipeline"]
    validation_cache = systems["validation_cache"]
    resident = systems["resident_graph_tensors"]
    expected_requested = {
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

    ordered_controls = recipe["condition_input_control_row_ids"]
    ordered_truth = recipe["condition_truth_row_ids"]
    b2_pipeline = load(B2 / "systems_runtime.json")["training_pipeline"]
    checks: dict[str, bool] = {
        "contract_sha256": sha256(CONTRACT) == "512f08c96ae0b4bac84afca96c88e89c05f0d802407e0831ad6b7d94cddbceb7",
        "immutable_b0_manifest_exact": sha256(B0 / "run_manifest.json") == contract["immutable_b0_manifest_sha256"],
        "b1_manifest_exact": sha256(B1 / "run_manifest.json") == contract["b1_run_manifest_sha256"],
        "b2_validation_exact": sha256(B2_VALIDATION) == contract["b2_validation_receipt_sha256"] and b2_validation["status"] == "PASS",
        "source_commit_exact": git_head == EXPECTED_COMMIT == manifest["source_commit"] == source_identity["commit"],
        "source_clean": not git_dirty and not manifest["source_dirty"] and not source_identity["dirty"],
        "public_main_exact": public_main == EXPECTED_COMMIT == source_identity["published_commit"],
        "config_hash_exact": manifest["config_sha256"] == EXPECTED_CONFIG_SHA == contract["config_sha256"],
        "run_identity_exact": manifest["run_id"] == EXPECTED_RUN_ID == contract["run_id"] and run_identity["formal_eligible"],
        "model_dataset_seed_exact": manifest["model_id"] == "gradpert_b2" and manifest["dataset_id"] == "nadig_jurkat" and manifest["run_seed"] == 1,
        "one_epoch_exact": training["epochs_requested"] == training["epochs_completed"] == 1,
        "optimizer_steps_exact": training["optimizer_steps"] == 582,
        "checkpoint_hash_exact": checkpoint.is_file() and sha256(checkpoint) == manifest["best_checkpoint_sha256"] == training["checkpoint_sha256"] == recipe["checkpoint_sha256"] == retention["best_checkpoint_sha256"],
        "checkpoint_retention_best_only": checkpoint_files == ["checkpoints/best.pt"] and retention["last_checkpoint_removed"],
        "no_test_truth_during_fit": training["canonical_test_truth_present_during_fit"] is False,
        "one_test_evaluation": manifest["test_evaluations"] == 1 and test_once["state"] == "completed",
        "three_metric_schema_exact": {row["metric_id"] for row in metric_rows} == METRIC_IDS and len(metric_rows) == 3,
        "metrics_finite": all(math.isfinite(float(value)) for value in metric_values.values()),
        "speed_only_nondecisional": performance["selection_policy"] == "speed_only_one_epoch_metrics_non_decisional",
        "expression_output_evaluation_5000": performance["expression_gene_count"] == performance["output_gene_count"] == performance["evaluation_gene_count"] == 5000,
        "reduced_graph_axis_exact": performance["graph_axis_policy"] == contract["graph_axis_policy"] and performance["graph_node_count"] == 2798 and performance["graph_nonself_edge_count"] == 89561,
        "runtime_graph_order_exact": training_data["runtime_graph_gene_count"] == 2798 and training_data["runtime_graph_gene_order_sha256"] == contract["expected_graph_gene_order_sha256"],
        "graph_content_exact": recipe["graph_topology_content_sha256"] == contract["expected_graph_topology_content_sha256"] and recipe["graph_manifest_sha256"] == contract["graph_manifest_sha256"],
        "canonical_hash_matches_b0": manifest["canonical_data_sha256"] == b0_manifest["canonical_data_sha256"] == contract["canonical_data_sha256"],
        "split_hash_matches_b0": manifest["split_content_sha256"] == b0_manifest["split_content_sha256"] == contract["split_content_sha256"],
        "control_manifest_matches_b0": manifest["control_manifest_sha256"] == b0_manifest["control_manifest_sha256"],
        "ordered_control_ids_match_b1_b2": ordered_controls == b1_recipe["condition_input_control_row_ids"] == b2_recipe["condition_input_control_row_ids"],
        "ordered_truth_ids_match_b1_b2": ordered_truth == b1_recipe["condition_truth_row_ids"] == b2_recipe["condition_truth_row_ids"],
        "evaluation_conditions_exact": len(evaluation["conditions"]) == len(prediction["conditions"]) == len(ordered_controls) == len(ordered_truth) == 592,
        "prediction_truth_excluded": prediction["truth_included"] is False,
        "metrics_only_recipe": recipe["result_mode"] == "metrics_only" and recipe["result_pkl_path"] is None and recipe["result_pkl_sha256"] is None,
        "zero_pkl": not pkl_files and not work_dirs,
        "timing_protocol_exact": performance["warmup_steps"] == 10 and performance["measured_steps"] == 572,
        "stage_timings_complete": set(performance["mean_stage_ms"]) == stage_fields and all(float(performance["mean_stage_ms"][key]) >= 0 for key in stage_fields),
        "resource_metrics_present": performance["peak_allocated_gpu_bytes"] > 0 and performance["peak_reserved_gpu_bytes"] > 0 and performance["peak_cpu_ram_bytes"] > 0,
        "systems_requested_all": requested == expected_requested and performance["systems_optimizations"] == expected_requested,
        "control_cache_active": pipeline["control_cache_active"] and pipeline["cached_control_batches"] == pipeline["yielded_batches"] == 582,
        "prefetch_pin_nonblock_active": pipeline["prefetch_active"] and pipeline["pin_memory_active"] and pipeline["nonblocking_transfer_active"],
        "resident_graph_active": resident["student"]["active"] and resident["teacher"]["active"] and resident["student"]["node_count"] == resident["teacher"]["node_count"] == 2798,
        "validation_cache_active": validation_cache["active"] and validation_cache["cache_hits"] > 0 and validation_cache["cached_rows"] > 0,
        "buffered_logging_active": requested["buffered_training_logs"] and systems["log_buffer_steps"] == 64,
        "single_checkpoint_serialization": systems["checkpoint"]["single_serialization_per_epoch"] and systems["checkpoint"]["peer_method"] in {"reflink", "copy"},
        "merged_read_enabled_and_cache_subsumed": requested["merged_hdf5_reads"] and pipeline["merged_read_batches"] == 0 and pipeline["cached_control_batches"] == 582,
        "first_step_hashes_complete": all(isinstance(first.get(key), str) and len(first[key]) == 64 for key in ["control_row_ids_sha256","perturbed_row_ids_sha256","pretransfer_control_sha256","pretransfer_target_sha256","view_structure_sha256","rng_state_before_sha256","rng_state_after_sha256","parameter_state_before_sha256","parameter_state_after_sha256"]),
        "first_step_runtime_matches_pipeline": first["control_row_ids_sha256"] == pipeline["first_control_row_ids_sha256"] and first["perturbed_row_ids_sha256"] == pipeline["first_perturbed_row_ids_sha256"] and first["pretransfer_control_sha256"] == pipeline["first_pretransfer_control_sha256"] and first["pretransfer_target_sha256"] == pipeline["first_pretransfer_target_sha256"],
        "batch_and_pretransfer_match_b2": first["control_row_ids_sha256"] == b2_pipeline["first_control_row_ids_sha256"] and first["perturbed_row_ids_sha256"] == b2_pipeline["first_perturbed_row_ids_sha256"] and first["pretransfer_control_sha256"] == b2_pipeline["first_pretransfer_control_sha256"] and first["pretransfer_target_sha256"] == b2_pipeline["first_pretransfer_target_sha256"] and pipeline["epoch_batch_identity_sha256"] == b2_pipeline["epoch_batch_identity_sha256"],
        "update_order_exact": first["update_order"] == contract["required_update_order"],
        "first_step_losses_finite": all(math.isfinite(float(value)) for value in first["losses"].values()),
        "manifest_hash_chain_exact": run_identity["run_manifest_sha256"] == canonical_sha(manifest) and recipe["prediction_manifest_sha256"] == sha256(B3 / "prediction_manifest.json") and recipe["evaluation_manifest_sha256"] == sha256(B3 / "evaluation_manifest.json"),
    }

    throughput = {label: full_epoch_throughput(root) for label, root in {"B1": B1, "B2": B2, "B3": B3}.items()}
    b1_wall = float(throughput["B1"]["wall_seconds"])
    b2_wall = float(throughput["B2"]["wall_seconds"])
    b3_wall = float(throughput["B3"]["wall_seconds"])
    comparisons = {
        "primary_metric": "one_epoch_training_wall_ms",
        "primary_reason": "actual monotonic wall clock; prefetch overlaps data reads with GPU work",
        "B1_to_B2": {"speedup_x": b1_wall / b2_wall, "wall_time_reduction_fraction": 1 - b2_wall / b1_wall},
        "B1_to_B3": {"speedup_x": b1_wall / b3_wall, "wall_time_reduction_fraction": 1 - b3_wall / b1_wall},
        "B2_to_B3": {"speedup_x": b2_wall / b3_wall, "wall_time_reduction_fraction": 1 - b3_wall / b2_wall},
        "full_epoch_wall_throughput": throughput,
        "receipt_measured_throughput_limitation": {
            "affected_variants": ["B2", "B3"],
            "reason": "measured_end_to_end_wall_ms sums data_read_ms plus step_wall_ms even when background prefetch overlaps them; retain the receipt field but do not use it as actual wall throughput",
        },
    }
    failed = sorted(key for key, value in checks.items() if not value)
    evidence = {
        "contract_path": str(CONTRACT),
        "contract_sha256": sha256(CONTRACT),
        "run_root": str(B3_ROOT),
        "source_commit": EXPECTED_COMMIT,
        "run_manifest_sha256": sha256(B3 / "run_manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_files": checkpoint_files,
        "pkl_files": pkl_files,
        "work_dirs": work_dirs,
        "axes": {
            "expression": performance["expression_gene_count"],
            "output": performance["output_gene_count"],
            "evaluation": performance["evaluation_gene_count"],
            "graph_nodes": performance["graph_node_count"],
            "graph_nonself_edges": performance["graph_nonself_edge_count"],
        },
        "epochs": {"requested": training["epochs_requested"], "completed": training["epochs_completed"]},
        "optimizer_steps": training["optimizer_steps"],
        "metric_ids": sorted(metric_values),
        "metrics_non_decisional": metric_values,
        "ordered_control_ids_sha256": canonical_sha(ordered_controls),
        "ordered_truth_ids_sha256": canonical_sha(ordered_truth),
        "first_step_equivalence": first,
        "systems_runtime": systems,
        "performance_receipt": performance,
        "comparisons": comparisons,
    }
    receipt = {
        "schema_version": "pilot-strict-validation-v2",
        "pilot_id": "B3_combined",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "evidence": evidence,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=OUTPUT.parent, delete=False) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, OUTPUT)
    print(json.dumps({"status": receipt["status"], "check_count": len(checks), "failed_checks": failed, "output": str(OUTPUT), "output_sha256": sha256(OUTPUT), "comparisons": comparisons}, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
