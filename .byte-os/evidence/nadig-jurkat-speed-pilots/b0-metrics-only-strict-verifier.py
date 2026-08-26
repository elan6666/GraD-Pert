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
CONTRACT = ROOT / "contracts/pilot-b0-7bed/gradpert-b0-7bed-contract.json"
OUTPUT = ROOT / "contracts/pilot-b0-7bed/b0-metrics-only-strict-validation.json"
LOG = ROOT / "runs/pilot-b0-metrics-only-7bed-launch.log"
RUN_ROOT = ROOT / "runs/pilot-b0-metrics-only-7bed"
SMALL = RUN_ROOT / "small_results"
HISTORICAL_B0 = ROOT / "runs/formal-v2/smoke/gradpert_b2/nadig_jurkat/seed-1/small_results"
B1 = ROOT / "runs/pilot-b1-graph-only-0a4d/small_results"
B2 = ROOT / "runs/pilot-b2-systems-only-2e30/small_results"
B3 = ROOT / "runs/pilot-b3-combined-44ae-r2/small_results"
B2_VALIDATION = ROOT / "contracts/pilot-b2-2e30/b2-strict-validation.json"
B3_VALIDATION = ROOT / "contracts/pilot-b3-44ae-r2/b3-strict-validation.json"
EXPECTED_CONTRACT_SHA = "61efdb8fd51769914f60dbbe3883860282934027fd0d2037fd7cf86951df3244"
EXPECTED_COMMIT = "7bed1f0cfbfa0be17c16ac0000d738510b87e96f"
EXPECTED_CONFIG_SHA = "e3c0ada7778a2712b58828b048a8eae5a2478ffdae48666c781f7e7565eab399"
EXPECTED_RUN_ID = "pilot-b0-metrics-only__nadig_jurkat__seed-1__7bed1f0"
METRIC_IDS = {
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
}
STAGE_FIELDS = {
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


def main() -> int:
    contract = load(CONTRACT)
    manifest = load(SMALL / "run_manifest.json")
    training = load(SMALL / "training_receipt.json")
    training_data = load(SMALL / "training_data.json")
    performance = load(SMALL / "performance_receipt.json")
    systems = load(SMALL / "systems_runtime.json")
    retention = load(SMALL / "checkpoint_retention.json")
    recipe = load(SMALL / "inference_recipe.json")
    prediction = load(SMALL / "prediction_manifest.json")
    evaluation = load(SMALL / "evaluation_manifest.json")
    source_identity = load(SMALL / "source_identity.json")
    run_identity = load(SMALL / "run_identity.json")
    test_once = load(SMALL / "test_once.json")
    historical_manifest = load(HISTORICAL_B0 / "run_manifest.json")
    b1_manifest = load(B1 / "run_manifest.json")
    recipes = {
        label: load(path / "inference_recipe.json")
        for label, path in {"B1": B1, "B2": B2, "B3": B3}.items()
    }
    b2_systems = load(B2 / "systems_runtime.json")
    b2_validation = load(B2_VALIDATION)
    b3_validation = load(B3_VALIDATION)

    with (SMALL / "train_steps.csv").open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    step_ids = [int(row["global_step"]) for row in train_rows]
    epoch_ids = [int(row["epoch"]) for row in train_rows]
    train_cells = sum(int(row["batch_cell_count"]) for row in train_rows)
    wall_seconds = float(performance["one_epoch_training_wall_ms"]) / 1000.0

    checkpoint = RUN_ROOT / "checkpoints/best.pt"
    checkpoint_files = sorted(
        str(path.relative_to(RUN_ROOT))
        for path in RUN_ROOT.glob("checkpoints/**/*")
        if path.is_file()
    )
    pkl_files = sorted(str(path.relative_to(RUN_ROOT)) for path in RUN_ROOT.rglob("*.pkl"))
    work_dirs = sorted(path.name for path in RUN_ROOT.glob(".result-work-*") if path.is_dir())
    matching_run_roots = sorted(
        path.name for path in (ROOT / "runs").glob("pilot-b0-metrics-only-7bed*") if path.is_dir()
    )

    git_head = command("git", "rev-parse", "HEAD")
    git_dirty = bool(command("git", "status", "--porcelain"))
    proxy_env = dict(os.environ)
    proxy_env.update(
        {
            "ALL_PROXY": "socks5h://127.0.0.1:17897",
            "HTTPS_PROXY": "socks5h://127.0.0.1:17897",
        }
    )
    public_main = command(
        "git",
        "ls-remote",
        "https://github.com/elan6666/GraD-Pert.git",
        "refs/heads/main",
        env=proxy_env,
    ).split()[0]

    metric_rows = performance["headline_metrics_non_decisional"]
    metric_values = {row["metric_id"]: row["macro_mean"] for row in metric_rows}
    ordered_controls = recipe["condition_input_control_row_ids"]
    ordered_truth = recipe["condition_truth_row_ids"]
    pipeline = systems["training_pipeline"]
    b2_pipeline = b2_systems["training_pipeline"]
    requested_without_label = dict(contract["systems_optimizations"])
    requested_without_label.pop("label")
    expected_runtime_requested = {
        "background_prefetch": False,
        "buffered_training_logs": False,
        "control_expression_cache": False,
        "log_buffer_steps": 64,
        "merged_hdf5_reads": False,
        "nonblocking_transfer": False,
        "pin_memory": False,
        "prefetch_depth": 1,
        "resident_graph_tensors": False,
        "single_checkpoint_serialization": False,
        "validation_expression_cache": False,
    }
    log_text = LOG.read_text(encoding="utf-8")

    checks: dict[str, bool] = {
        "contract_hash_exact": sha256(CONTRACT) == EXPECTED_CONTRACT_SHA,
        "single_fresh_run_root": matching_run_roots == [RUN_ROOT.name],
        "completion_marker_once": log_text.count("B0_METRICS_ONLY_COMPLETE") == 1,
        "launch_log_no_failure": not any(
            token in log_text for token in ("Traceback", "No space left", "CUDA out of memory")
        ),
        "historical_b0_manifest_immutable": sha256(HISTORICAL_B0 / "run_manifest.json")
        == contract["historical_b0_manifest_sha256"],
        "comparison_inputs_exact": sha256(B1 / "run_manifest.json")
        == contract["comparison_inputs"]["b1_run_manifest_sha256"]
        and sha256(B2_VALIDATION) == contract["comparison_inputs"]["b2_validation_sha256"]
        and sha256(B3_VALIDATION) == contract["comparison_inputs"]["b3_validation_sha256"]
        and b2_validation["status"] == b3_validation["status"] == "PASS",
        "source_commit_exact": git_head
        == EXPECTED_COMMIT
        == manifest["source_commit"]
        == source_identity["commit"]
        == contract["source_commit"],
        "source_clean": not git_dirty
        and not manifest["source_dirty"]
        and not source_identity["dirty"],
        "public_main_exact": public_main == EXPECTED_COMMIT == source_identity["published_commit"],
        "config_hash_exact": manifest["config_sha256"]
        == EXPECTED_CONFIG_SHA
        == contract["config_sha256"],
        "run_identity_exact": manifest["run_id"] == EXPECTED_RUN_ID == contract["run_id"]
        and run_identity["formal_eligible"],
        "model_dataset_seed_exact": manifest["model_id"] == "gradpert_b2"
        and manifest["dataset_id"] == "nadig_jurkat"
        and manifest["run_seed"] == contract["seed"] == 1,
        "one_epoch_exact": training["epochs_requested"]
        == training["epochs_completed"]
        == contract["expected_epochs"]
        == 1,
        "optimizer_steps_exact": training["optimizer_steps"]
        == contract["expected_optimizer_steps"]
        == 582,
        "train_step_sequence_exact": len(train_rows) == 582
        and step_ids == list(range(582))
        and epoch_ids == [0] * 582,
        "checkpoint_hash_chain_exact": checkpoint.is_file()
        and sha256(checkpoint)
        == manifest["best_checkpoint_sha256"]
        == training["checkpoint_sha256"]
        == recipe["checkpoint_sha256"]
        == retention["best_checkpoint_sha256"],
        "checkpoint_retention_best_only": checkpoint_files == ["checkpoints/best.pt"]
        and retention["last_checkpoint_removed"]
        and retention["policy"] == "best_only_after_successful_evaluation",
        "no_test_truth_during_fit": training["canonical_test_truth_present_during_fit"] is False,
        "one_test_evaluation": manifest["test_evaluations"] == 1
        and test_once["state"] == "completed",
        "three_metric_schema_exact": len(metric_rows) == 3
        and {row["metric_id"] for row in metric_rows} == METRIC_IDS,
        "metrics_finite_and_nondecisional": all(
            math.isfinite(float(value)) for value in metric_values.values()
        )
        and performance["selection_policy"] == "speed_only_one_epoch_metrics_non_decisional",
        "expression_output_evaluation_axes_exact": performance["expression_gene_count"]
        == contract["expected_expression_gene_count"]
        == 5000
        and performance["output_gene_count"] == contract["expected_output_gene_count"] == 5000
        and performance["evaluation_gene_count"]
        == contract["expected_evaluation_gene_count"]
        == 5000
        and training_data["expression_gene_order_sha256"]
        == contract["expected_expression_gene_order_sha256"],
        "canonical_full_graph_exact": performance["graph_axis_policy"]
        == contract["graph_axis_policy"]
        == "canonical_full"
        and performance["graph_node_count"]
        == training_data["runtime_graph_gene_count"]
        == contract["expected_graph_gene_count"]
        == 6506
        and performance["graph_nonself_edge_count"]
        == contract["expected_graph_nonself_edge_count"]
        == 222654
        and training_data["runtime_graph_gene_order_sha256"]
        == contract["expected_graph_gene_order_sha256"],
        "canonical_hash_matches_historical_and_variants": manifest["canonical_data_sha256"]
        == historical_manifest["canonical_data_sha256"]
        == b1_manifest["canonical_data_sha256"]
        == contract["canonical_data_sha256"],
        "split_hash_matches_historical_and_variants": manifest["split_content_sha256"]
        == historical_manifest["split_content_sha256"]
        == b1_manifest["split_content_sha256"]
        == contract["split_content_sha256"],
        "control_manifest_matches_historical_and_variants": manifest["control_manifest_sha256"]
        == historical_manifest["control_manifest_sha256"]
        == b1_manifest["control_manifest_sha256"]
        == contract["control_manifest_sha256"],
        "ordered_control_ids_match_b1_b2_b3": all(
            ordered_controls == other["condition_input_control_row_ids"]
            for other in recipes.values()
        ),
        "ordered_truth_ids_match_b1_b2_b3": all(
            ordered_truth == other["condition_truth_row_ids"] for other in recipes.values()
        ),
        "evaluation_conditions_exact": len(evaluation["conditions"])
        == len(prediction["conditions"])
        == len(ordered_controls)
        == len(ordered_truth)
        == 592,
        "prediction_truth_excluded": prediction["truth_included"] is False,
        "metrics_only_recipe": recipe["result_mode"] == contract["result_mode"] == "metrics_only"
        and recipe["result_pkl_path"] is None
        and recipe["result_pkl_sha256"] is None,
        "zero_persistent_pkl_and_workdirs": not pkl_files
        and not work_dirs
        and contract["persistent_pkl_count"] == 0,
        "timing_protocol_exact": performance["warmup_steps"] == 10
        and performance["measured_steps"] == 572,
        "timing_and_throughput_positive": wall_seconds > 0
        and train_cells > 0
        and len(train_rows) / wall_seconds > 0
        and train_cells / wall_seconds > 0,
        "stage_timings_complete": set(performance["mean_stage_ms"]) == STAGE_FIELDS
        and all(float(performance["mean_stage_ms"][field]) >= 0 for field in STAGE_FIELDS),
        "resource_metrics_present": performance["peak_allocated_gpu_bytes"] > 0
        and performance["peak_reserved_gpu_bytes"] > 0
        and performance["peak_cpu_ram_bytes"] > 0,
        "all_systems_requested_disabled": systems["requested"]
        == performance["systems_optimizations"]
        == expected_runtime_requested
        == requested_without_label,
        "training_pipeline_disabled": pipeline["yielded_batches"] == 582
        and not pipeline["control_cache_active"]
        and pipeline["cached_control_batches"] == 0
        and not pipeline["prefetch_active"]
        and not pipeline["pin_memory_active"]
        and not pipeline["nonblocking_transfer_active"]
        and pipeline["merged_read_batches"] == 0,
        "resident_graph_disabled": not systems["resident_graph_tensors"]["student"]["active"]
        and not systems["resident_graph_tensors"]["teacher"]["active"],
        "validation_cache_disabled": not systems["validation_cache"]["active"]
        and systems["validation_cache"]["cache_hits"] == 0,
        "buffered_logging_disabled": systems["log_buffer_steps"] == 1,
        "single_checkpoint_serialization_disabled": not systems["checkpoint"][
            "single_serialization_per_epoch"
        ]
        and systems["checkpoint"]["peer_method"] is None,
        "batch_identity_matches_b2": pipeline["epoch_batch_identity_sha256"]
        == b2_pipeline["epoch_batch_identity_sha256"]
        and pipeline["first_control_row_ids_sha256"] == b2_pipeline["first_control_row_ids_sha256"]
        and pipeline["first_perturbed_row_ids_sha256"]
        == b2_pipeline["first_perturbed_row_ids_sha256"]
        and pipeline["first_pretransfer_control_sha256"]
        == b2_pipeline["first_pretransfer_control_sha256"]
        and pipeline["first_pretransfer_target_sha256"]
        == b2_pipeline["first_pretransfer_target_sha256"],
        "manifest_hash_chain_exact": run_identity["run_manifest_sha256"] == canonical_sha(manifest)
        and recipe["prediction_manifest_sha256"] == sha256(SMALL / "prediction_manifest.json")
        and recipe["evaluation_manifest_sha256"] == sha256(SMALL / "evaluation_manifest.json"),
    }

    failed = sorted(key for key, passed in checks.items() if not passed)
    evidence = {
        "contract_path": str(CONTRACT),
        "contract_sha256": sha256(CONTRACT),
        "run_root": str(RUN_ROOT),
        "run_manifest_sha256": sha256(SMALL / "run_manifest.json"),
        "source_commit": EXPECTED_COMMIT,
        "config_sha256": manifest["config_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_files": checkpoint_files,
        "pkl_files": pkl_files,
        "work_dirs": work_dirs,
        "matching_run_roots": matching_run_roots,
        "historical_b0_manifest_sha256": sha256(HISTORICAL_B0 / "run_manifest.json"),
        "axes": {
            "expression": performance["expression_gene_count"],
            "output": performance["output_gene_count"],
            "evaluation": performance["evaluation_gene_count"],
            "graph_nodes": performance["graph_node_count"],
            "graph_nonself_edges": performance["graph_nonself_edge_count"],
        },
        "epochs": {
            "requested": training["epochs_requested"],
            "completed": training["epochs_completed"],
        },
        "optimizer_steps": training["optimizer_steps"],
        "full_epoch": {
            "steps": len(train_rows),
            "cells": train_cells,
            "wall_seconds": wall_seconds,
            "steps_per_second": len(train_rows) / wall_seconds,
            "cells_per_second": train_cells / wall_seconds,
        },
        "metric_ids": sorted(metric_values),
        "metrics_non_decisional": metric_values,
        "ordered_control_ids_sha256": canonical_sha(ordered_controls),
        "ordered_truth_ids_sha256": canonical_sha(ordered_truth),
        "systems_runtime": systems,
        "performance_receipt": performance,
    }
    receipt = {
        "schema_version": "pilot-strict-validation-v3",
        "pilot_id": "B0_metrics_only_rerun",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "evidence": evidence,
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
                "check_count": len(checks),
                "failed_checks": failed,
                "output": str(OUTPUT),
                "output_sha256": sha256(OUTPUT),
                "full_epoch": evidence["full_epoch"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
