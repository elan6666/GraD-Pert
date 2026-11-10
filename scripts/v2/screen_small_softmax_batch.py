"""Bounded dual-GPU batch screen using the native capacity probe."""

from __future__ import annotations

import argparse
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import yaml

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE), str(SOURCE / "scripts/v2")]

from gradpert.config import load_experiment_config  # noqa: E402
from gradpert.config.v2 import V2Options  # noqa: E402
from gradpert.data._io import atomic_json, read_json  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.hashing import sha256_file  # noqa: E402
from scripts.v2.run_dataset_transfer_300 import native_run  # noqa: E402
from scripts.v2.run_group import lock  # noqa: E402
from scripts.v2.run_small_softmax import admit_selected_devices  # noqa: E402
from scripts.v2.shared_gpu import process_info, snapshot  # noqa: E402

REFERENCE = "docs/design/GRADPERT_SMALL_SOFTMAX_FOLLOWUPS_20261110.md"
BASE = "configs/v2/small_softmax_mhc_no_distill/gradpert_v2/nadig_jurkat.yaml"
GRID = (80, 96, 128, 160, 224, 320, 448, 640)
STRIDE = 4
DEEP_BASE = "configs/v2/small_softmax_mhc_deep_no_distill/gradpert_v2/nadig_jurkat.yaml"
DEEP_GRID = (64, 96, 128, 160, 224, 320, 448, 640)
SCREEN_VARIANTS = {"original": (BASE, GRID, STRIDE), "deep": (DEEP_BASE, DEEP_GRID, 1)}
ROWS = 47_836
FULL_ROWS = 128_266
FULL_GRID = {
    "original": (64, 96, 128, 132, 133, 144, 160, 224, 320, 448, 640),
    "deep": (64, 96, 128, 131, 132, 144, 160, 224, 320, 448, 640),
}


def screen_grid(variant: str, training_scope: str) -> tuple[str, tuple[int, ...], int]:
    base, grid, stride = SCREEN_VARIANTS[variant]
    if training_scope == "full":
        return base, FULL_GRID[variant], 1
    if training_scope != "cap40":
        raise ValueError("unknown batch screen training scope")
    return base, grid, stride


def full_training_recipe(base: str) -> dict[str, Any]:
    payload = yaml.safe_load((SOURCE / base).read_text())
    params = payload["model"]["parameters"]
    for key in ("train_selection_path", "train_selection_sha256"):
        params[key] = dict(value="", source="user_locked", reference=REFERENCE)
    return payload


def recipe(
    microbatch: int, variant: str = "original", training_scope: str = "cap40"
) -> dict[str, Any]:
    base, grid, stride = screen_grid(variant, training_scope)
    if microbatch < grid[0] or microbatch > grid[-1] or microbatch % stride:
        raise ValueError("candidate outside registered microbatch grid")
    payload = (
        full_training_recipe(base)
        if training_scope == "full"
        else yaml.safe_load((SOURCE / base).read_text())
    )
    provenance = dict(source="user_locked", reference=REFERENCE)
    params = payload["model"]["parameters"]
    for key, value in {
        "microbatch": microbatch,
        "world_size": 2,
        "accumulation": 1,
        "distillation_enabled": False,
    }.items():
        params[key] = dict(value=value, **provenance)
    payload["training"]["train_batch_size"] = dict(value=2 * microbatch, **provenance)
    return payload


def screen_evidence(directory: Path, microbatch: int, plan: dict, config_sha256: str) -> dict:
    def matches(value, rank):
        screen = value["batch_capacity_screen"]
        return (
            value["source"]["commit"] == plan["source_commit"]
            and value["source"]["formal_eligible"]
            and value["config_sha256"] == config_sha256
            and value["gpu"] == "0,1"
            and value["data_root"] == plan["data_root"]
            and screen["rank"] == rank
            and screen["microbatch"] == microbatch
            and screen["global_batch"] == 2 * microbatch
            and screen["world_size"] == 2
            and screen["accumulation"] == 1
            and screen["queries"] == 1000
            and screen["cuda_memory_fraction"] == 0.9
            and (
                plan.get("training_scope", "cap40") != "full"
                or (
                    value.get("training_rows") == FULL_ROWS
                    and "training_row_selection" not in value["data"]
                )
            )
        )

    receipt = directory / "receipt.json"
    if receipt.exists():
        value = read_json(receipt)
        if (
            matches(value, 0)
            and value["status"] == "passed"
            and value["steps_completed"] == value["steps_requested"] == 3
            and value["inference_exercised"] is False
            and value["batch_capacity_screen"]["phase"] == "screen_complete"
        ):
            ranks = value["rank_measurements"]
            if (
                {r["rank"]: r["physical_gpu"] for r in ranks} != {0: "0", 1: "1"}
                or any(
                    len(r["all_update_seconds"]) != 3
                    or not 0 < r["peak_allocated_bytes"] <= r["peak_reserved_bytes"]
                    for r in ranks
                )
                or not value.get("initial_loss_diagnostic")
                or not math.isfinite(value["measured_cells_per_second_including_data_wait"])
            ):
                raise ValueError("incomplete rank memory, training or diagnostic evidence")
            return dict(
                classification="passed",
                config_sha256=config_sha256,
                receipt=str(receipt),
                receipt_sha256=sha256_file(receipt),
                cells_per_second=value["measured_cells_per_second_including_data_wait"],
                rank_memory=[
                    {k: r[k] for k in ("rank", "peak_allocated_bytes", "peak_reserved_bytes")}
                    for r in ranks
                ],
            )
    failures = []
    for rank in (0, 1):
        file = directory / f"rank-{rank}-failure.json"
        if not file.exists():
            continue
        value = read_json(file)
        if matches(value, rank) and (
            value["status"] == "failed"
            and value["batch_capacity_screen"]["failure_class"] == "cuda_update_oom"
        ):
            failures.append(dict(rank=rank, receipt=str(file), sha256=sha256_file(file)))
    if failures:
        return dict(
            classification="cuda_update_oom", config_sha256=config_sha256, failures=failures
        )
    raise ValueError("no qualified two-rank pass or update CUDA OOM")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("sealed batch screen plan differs")
    root = args.plan.resolve().parent
    if not root.is_relative_to("/data/yilangliu") or not root.is_dir():
        raise ValueError("batch screen is server-only")
    plan = read_json(args.plan)
    variant = plan.get("variant", "original")
    if variant not in SCREEN_VARIANTS:
        raise ValueError("unknown batch screen variant")
    training_scope = plan.get("training_scope", "cap40")
    base, grid, stride = screen_grid(variant, training_scope)
    training_rows = FULL_ROWS if training_scope == "full" else ROWS
    if (
        plan["gpu"] != "0,1"
        or plan["gpu_uuids"]
        != ["GPU-23bb1466-b82b-982d-8eaf-9a0002c6a4e7", "GPU-edd7484e-c71d-1938-4831-76c233466ccb"]
        or plan["candidate_grid"] != list(grid)
        or plan["stride"] != stride
        or plan["baseline_recipe_sha256"] != sha256_file(SOURCE / base)
        or plan.get("training_rows", ROWS) != training_rows
    ):
        raise ValueError("batch screen scope differs")
    for name, expected in (
        ("CUDA_VISIBLE_DEVICES", "0,1"),
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
    ):
        if os.environ.get(name) != expected:
            raise ValueError(f"batch screen requires {name}={expected}")
    identity = inspect_source_identity(
        SOURCE,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=plan["publication"],
        expected_publication_receipt_sha256=plan["publication_sha256"],
    )
    if identity.commit != plan["source_commit"]:
        raise ValueError("batch screen source commit differs")
    selected = dict(enumerate(plan["gpu_uuids"]))
    owner = process_info(os.getpid())
    if owner is None:
        raise RuntimeError("batch screen owner not identifiable")

    def guard():
        state = snapshot(os.getpid())
        admit_selected_devices(state, selected)
        atomic_json(root / "RESOURCE_CURRENT.json", state)

    with ExitStack() as stack:
        for gpu in selected:
            stack.enter_context(lock(Path(f"/data/yilangliu/GraD-Pert/runtime/v2-gpu-{gpu}.lock")))
        guard()
        with (root / "capacity-owner.json").open("x") as file:
            import json

            json.dump(
                {"owner": owner, "plan_sha256": args.plan_sha256, "gpu_uuids": plan["gpu_uuids"]},
                file,
            )
        rows = []
        try:

            def check(microbatch):
                attempt = root / f"micro-{microbatch:04d}"
                attempt.mkdir()
                path = attempt / "gradpert_v2/nadig_jurkat.yaml"
                path.parent.mkdir()
                path.write_text(
                    yaml.safe_dump(recipe(microbatch, variant, training_scope), sort_keys=False)
                )
                cfg = load_experiment_config(path)
                _, opt = V2Options.parse_parameters(cfg.model.parameters)
                if int(cfg.training.train_batch_size.value) != 2 * opt.microbatch:
                    raise ValueError("candidate effective batch does not match rank-local batch")
                digest = sha256_file(path)
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    "2",
                    str(SOURCE / "scripts/v2/capacity_probe.py"),
                    "--config",
                    str(path),
                    "--data-root",
                    plan["data_root"],
                    "--output",
                    str(attempt / "probe"),
                    "--gpu",
                    "0,1",
                    "--publication",
                    plan["publication"],
                    "--publication-sha256",
                    plan["publication_sha256"],
                    "--benchmark-only",
                    "--batch-capacity-only",
                    "--steps",
                    "3",
                ]
                guard()
                try:
                    native_run(command, attempt / "screen.log", {}, admission_check=guard)
                except RuntimeError as exc:
                    if not str(exc).startswith("native child exited "):
                        raise
                    evidence = screen_evidence(attempt / "probe", microbatch, plan, digest)
                    if evidence["classification"] != "cuda_update_oom":
                        raise
                else:
                    evidence = screen_evidence(attempt / "probe", microbatch, plan, digest)
                    if evidence["classification"] != "passed":
                        raise ValueError("native exit status differs from screening receipt")
                rows.append(dict(microbatch=microbatch, global_batch=2 * microbatch, **evidence))
                atomic_json(
                    root / "capacity-progress.json", dict(plan_sha256=args.plan_sha256, rows=rows)
                )
                return evidence["classification"] == "passed"

            best, oom = None, None
            for micro in grid:
                if check(micro):
                    best = micro
                else:
                    oom = micro
                    break
            if best is None:
                raise ValueError(f"microbatch {grid[0]} did not pass two-rank capacity screening")
            while oom is not None and oom - best > stride:
                midpoint = ((best + oom) // (2 * stride)) * stride
                if check(midpoint):
                    best = midpoint
                else:
                    oom = midpoint
            atomic_json(
                root / "capacity-complete.json",
                dict(
                    status="screen_complete",
                    plan_sha256=args.plan_sha256,
                    source_commit=plan["source_commit"],
                    variant=variant,
                    training_scope=training_scope,
                    training_rows=training_rows,
                    highest_verified_microbatch=best,
                    global_batch=2 * best,
                    accumulation=1,
                    world_size=2,
                    lowest_verified_cuda_update_oom=oom,
                    search_upper_microbatch=grid[-1],
                    resolution=stride,
                    steps_per_epoch=math.ceil(training_rows / (2 * best)),
                    last_global_batch=training_rows % (2 * best) or 2 * best,
                    rows=rows,
                ),
            )
        except Exception as exc:
            atomic_json(
                root / "capacity-failure.json",
                dict(
                    error_type=type(exc).__name__,
                    error=str(exc),
                    plan_sha256=args.plan_sha256,
                    rows=rows,
                ),
            )
            raise


if __name__ == "__main__":
    main()
