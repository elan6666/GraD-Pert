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

from run_group import lock, next_action, validate_preflight

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


def prepare(runtime: Path, baseline: Path) -> dict:
    complete = json.loads((baseline / "COMPLETE.json").read_text())
    journal = json.loads((baseline / "fit/epoch_state.json").read_text())
    if complete["epoch"] != 6 or journal["epoch"] != 6 or not complete["zero_pkl"]:
        raise ValueError("cap40 B0 terminal prerequisite is incomplete")
    source = Path(__file__).resolve().parents[2]
    rows = []
    for name, digest in ROWS:
        config = (
            source / "configs/v2/cap40_ablations_jurkat" / name / "gradpert_v2/nadig_jurkat.yaml"
        )
        if sha256_file(config) != digest:
            raise ValueError("approved cap40 configuration changed")
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
        "schema": "cap40-three-ablations-queue-v1",
        "baseline": str(baseline),
        "baseline_complete_sha256": sha256_file(baseline / "COMPLETE.json"),
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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("required allocator contract is missing")
    if not args.queue_root.resolve().is_relative_to("/data/yilangliu"):
        parser.error("queue outputs stay on server")
    queue = prepare(args.runtime, args.baseline)
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
