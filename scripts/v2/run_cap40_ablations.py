"""Run the three approved cap40 ablations after exact preflight and idle-GPU gates."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from collect_results import collect_run
from run_group import lock, next_action, validate_preflight

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.train_entry import resolve_plan
from gradpert.hashing import sha256_file

ROWS = (
    ("E1_no_mhc", "5718b2c298aaed31ae796205be5f8597a060da4106e4b0017e35f605dff68686"),
    ("E2_prototypes16384", "5197a3e3d7bfe57a7545b4cc1dad8ff7b5c666dd521bb298cec1ed2d9dabe64f"),
    (
        "E3_unit_distillation_no_spread_koleo",
        "6982785ffac1b5ac3f681956ddf8ad51af2aa505e3f02e22572876cb7d34cb2a",
    ),
)


COMBINED_ROWS = (
    (
        "E23_prototypes16384_unit_distillation",
        "7214e3e382232ae8856998544514c54aecd048aeec2ab3cb8f329c7157c08df8",
    ),
)


def idle_devices(output: str) -> bool:
    usage = {
        p[0].strip(): int(p[1])
        for line in output.splitlines()
        if line.strip() and (p := line.split(","))
    }
    return set(usage) >= {"0", "1"} and all(usage[g] <= 512 for g in ("0", "1"))


def require_fresh_formal(plan: dict) -> None:
    if next_action(plan) != "launch":
        raise ValueError("fresh queue cannot resume or replace an existing experiment")


def validate_completed_e1(root: Path, baseline: Path) -> dict:
    """Accept a terminal E1 without training or rewriting its historical identity."""
    source = Path(__file__).resolve().parents[2]
    config = source / "configs/v2/cap40_ablations_jurkat/E1_no_mhc/gradpert_v2/nadig_jurkat.yaml"
    rows = collect_run(root)
    if (
        len(rows) != 2
        or {row["role"] for row in rows} != {"best", "last"}
        or any(row["status"] != "complete" or row["config_sha256"] != ROWS[0][1] for row in rows)
        or (root / "FAILURE.json").exists()
    ):
        raise ValueError("completed E1 lacks the exact approved terminal contract")
    if json.loads((root / "resolved_config.json").read_text()) != load_experiment_config(
        config
    ).model_dump(mode="json"):
        raise ValueError("completed E1 resolved configuration differs from approved config")
    manifest = json.loads((root / "run_manifest.json").read_text())
    parent = json.loads((baseline / "run_manifest.json").read_text())
    if manifest["data"] != parent["data"]:
        raise ValueError("completed E1 data/selection/seed differs from baseline")
    prior_tree = subprocess.check_output(
        ["git", "rev-parse", manifest["source"]["commit"] + ":src"], cwd=source, text=True
    ).strip()
    current_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:src"], cwd=source, text=True
    ).strip()
    if prior_tree != current_tree:
        raise ValueError("completed E1 native source differs from the remaining queue")
    return {
        "root": str(root),
        "complete_sha256": sha256_file(root / "COMPLETE.json"),
        "native_src_tree": prior_tree,
        "collected_rows": rows,
    }


def prepare(
    runtime: Path,
    baseline: Path,
    completed_e1: Path | None = None,
    *,
    combined_twenty: bool = False,
) -> dict:
    if combined_twenty and completed_e1 is not None:
        raise ValueError("combined twenty-epoch run does not import an E1")
    complete = json.loads((baseline / "COMPLETE.json").read_text())
    journal = json.loads((baseline / "fit/epoch_state.json").read_text())
    if complete["epoch"] != 6 or journal["epoch"] != 6 or not complete["zero_pkl"]:
        raise ValueError("cap40 B0 terminal prerequisite is incomplete")
    source = Path(__file__).resolve().parents[2]
    imported = validate_completed_e1(completed_e1, baseline) if completed_e1 else None
    rows = []
    family = "cap40_combined_jurkat" if combined_twenty else "cap40_ablations_jurkat"
    approved = COMBINED_ROWS if combined_twenty else ROWS
    for name, digest in approved:
        config = source / "configs/v2" / family / name / "gradpert_v2/nadig_jurkat.yaml"
        if sha256_file(config) != digest:
            raise ValueError("approved cap40 configuration changed")
        if imported and name == "E1_no_mhc":
            continue
        plan = resolve_plan(
            argparse.Namespace(config=config, runtime=runtime, data_root=None, gpu="0,1", seed=1)
        )
        rows.append(
            {
                "name": name,
                "plan": plan,
                "probe_kind": "capacity_only" if name.startswith("E2") else "integration_only",
            }
        )
    for role in ("best", "last"):
        result = json.loads((baseline / "fit" / f"{role}-test.json").read_text())
        if result["identity"]["role"] != role or result["result"]["split"] != "test":
            raise ValueError("baseline frozen test role is missing")
    return {
        "schema": (
            "cap40-combined-twenty-queue-v1"
            if combined_twenty
            else "cap40-three-ablations-queue-v1"
        ),
        "baseline": str(baseline),
        "baseline_complete_sha256": sha256_file(baseline / "COMPLETE.json"),
        "completed_e1": imported,
        "rows": rows,
    }


def execute(queue: dict, directory: Path) -> None:
    if not directory.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("queue outputs stay on server")
    state = {
        "schema": queue["schema"],
        "pid": os.getpid(),
        "status": "running",
        "phase": "preflight",
        "row": None,
    }

    def record(**values):
        state.update(values, updated_unix=time.time())
        atomic_json(directory / "state.json", state)

    def run_child(command: list[str], row: dict, stage: str) -> None:
        record(
            phase="waiting_for_idle_gpus",
            pending_stage=stage,
            row=row["name"],
            run_id=row["plan"]["run_id"],
            child_pid=None,
        )
        while True:
            usage = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            record(gpu_memory_snapshot=usage.strip())
            if idle_devices(usage):
                break
            time.sleep(60)
        with contextlib.ExitStack() as stack:
            descriptors = [
                stack.enter_context(
                    lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{g}.lock")
                )
                for g in ("0", "1")
            ]
            plan = row["plan"]
            lease = f"v2-row-{plan['config_sha256']}-{plan['source_commit']}-{plan['seed']}.lock"
            descriptors.append(
                stack.enter_context(lock(Path("/data/yilangliu/GraD-Pert/runtime") / lease))
            )
            if stage == "formal":
                require_fresh_formal(plan)
            # Recheck after acquiring the project leases; never kill unrelated work.
            usage = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            if not idle_devices(usage):
                raise RuntimeError("GPU resource changed after idle check; preserve other jobs")
            with (directory / f"{row['name']}-{stage}.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=row["plan"]["repository_root"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=tuple(descriptors),
                )
                record(phase=stage, child_pid=child.pid)
                code = child.wait()
            atomic_json(
                directory / f"{row['name']}-{stage}.exit.json",
                {
                    "exit_code": code,
                    "finished_unix": time.time(),
                    "run_id": row["plan"]["run_id"],
                    "stage": stage,
                },
            )
            if code:
                raise RuntimeError(
                    f"{row['name']} {stage} failed with exit {code}; explicit repair required"
                )

    try:
        with lock(directory / "queue.lock"):
            baseline = Path(queue["baseline"])
            if sha256_file(baseline / "COMPLETE.json") != queue["baseline_complete_sha256"]:
                raise ValueError("baseline terminal receipt changed")
            if queue.get("completed_e1"):
                imported = queue["completed_e1"]
                if validate_completed_e1(Path(imported["root"]), baseline) != imported:
                    raise ValueError("imported E1 terminal evidence changed")
            for row in queue["rows"]:
                plan = row["plan"]
                probe = directory / (row["name"] + "-preflight")
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    "2",
                    str(Path(plan["repository_root"]) / "scripts/v2/capacity_probe.py"),
                    "--config",
                    plan["config"],
                    "--data-root",
                    plan["data_root"],
                    "--output",
                    str(probe),
                    "--gpu",
                    "0,1",
                    "--publication",
                    plan["publication"],
                    "--publication-sha256",
                    plan["publication_sha256"],
                ]
                if row["probe_kind"] == "integration_only":
                    command.append("--integration-only")
                else:
                    command.extend(["--steps", "128"])
                run_child(command, row, "preflight")
                entry = {
                    "receipt": str(probe / "receipt.json"),
                    "sha256": sha256_file(probe / "receipt.json"),
                }
                validate_preflight(plan, entry)
                row["preflight"] = entry
                atomic_json(directory / "preflight-index.json", {"rows": queue["rows"]})
            for row in queue["rows"]:
                plan = row["plan"]
                validate_preflight(plan, row["preflight"])
                require_fresh_formal(plan)
                plan_path = directory / f"{row['name']}.launch-plan.json"
                atomic_json(plan_path, plan)
                command = [
                    sys.executable,
                    "-c",
                    "import json,sys;from gradpert.execution.train_entry import execute_plan;"
                    "execute_plan(json.load(open(sys.argv[1])))",
                    str(plan_path),
                ]
                run_child(command, row, "formal")
                if next_action(plan) != "skip_complete":
                    raise RuntimeError("formal child exited without matching best/last completion")
            if queue.get("completed_e1"):
                imported = queue["completed_e1"]
                if validate_completed_e1(Path(imported["root"]), baseline) != imported:
                    raise ValueError("imported E1 terminal evidence changed")
            record(status="complete", phase="complete", child_pid=None)
            atomic_json(directory / "COMPLETE.json", state)
    except BaseException as error:
        record(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(directory / "FAILURE.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--completed-e1", type=Path)
    parser.add_argument("--combined-twenty", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("required allocator contract is missing")
    if not args.queue_root.resolve().is_relative_to("/data/yilangliu"):
        parser.error("queue outputs stay on server")
    if args.completed_e1 and not args.completed_e1.resolve().is_relative_to("/data/yilangliu"):
        parser.error("completed E1 evidence stays on server")
    queue = prepare(
        args.runtime, args.baseline, args.completed_e1, combined_twenty=args.combined_twenty
    )
    args.queue_root.mkdir(parents=True, exist_ok=False)
    atomic_json(args.queue_root / "queue.json", queue)
    if args.execute:
        execute(queue, args.queue_root)
    else:
        print(
            json.dumps(
                {
                    "queue_root": str(args.queue_root),
                    "rows": len(queue["rows"]),
                    "status": "planned_not_launched",
                }
            )
        )


if __name__ == "__main__":
    main()
