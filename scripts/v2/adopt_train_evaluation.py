"""Adopt an existing fit, evaluate the preceding checkpoint, never relaunch fit."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from run_group import lock

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.v2_deferred_postfit import run_deferred_postfit
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file


def verified_process(pid: int, expected: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    if not command:
        return False
    if hashlib.sha256(command).hexdigest() != expected:
        raise ValueError("adopted PID was reused; do not signal or wait for another process")
    return True


def execute(schedule: dict) -> None:
    directory = Path(schedule["output"])
    if not directory.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("schedule stays on the server")
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "schedule.json", schedule)
    state = {"pid": os.getpid(), "status": "running", "phase": "verify"}

    def record(**values):
        state.update(values, observed_unix=time.time())
        atomic_json(directory / "state.json", state)

    try:
        if sha256_file(Path(schedule["old_queue"]) / "queue.json") != schedule["old_queue_sha256"]:
            raise ValueError("original queue identity changed")
        if verified_process(
            schedule["retired_controller_pid"], schedule["controller_cmdline_sha256"]
        ):
            raise ValueError("original controller must be retired before adoption")
        plans = []
        for item in schedule["rows"]:
            path = Path(item["launch"])
            if sha256_file(path) != item["launch_sha256"]:
                raise ValueError("training launch changed")
            plans.append(read_json(path))
        validate_training_stage(plans[0])
        # The user-scoped shared admission names one fit and a capped evaluator;
        # it does not weaken normal GPU resource gates for other queues.
        record(phase="U4_eval_with_U3_fit", adopted_fit_pid=schedule["fit_pid"])
        run_deferred_postfit(
            plans[0],
            gpu=schedule["shared_evaluation_gpu"],
            cuda_memory_fraction=schedule["evaluation_memory_fraction"],
            evaluation_runtime=Path(schedule["evaluation_runtime"]),
        )
        record(phase="U4_evaluation_complete_wait_U3_fit")
        while verified_process(schedule["fit_pid"], schedule["fit_cmdline_sha256"]):
            if (Path(plans[1]["run_root"]) / "FAILURE.json").exists():
                raise RuntimeError("adopted fit failed; preserve checkpoint for explicit repair")
            time.sleep(30)
        validate_training_stage(plans[1])
        record(phase="U3_eval_wait_idle")
        while True:
            usage = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            memory = {x.split(",")[0].strip(): int(x.split(",")[1]) for x in usage.splitlines()}
            if all(memory.get(g, 2**63) <= 512 for g in ("0", "1")):
                break
            time.sleep(30)
        with contextlib.ExitStack() as stack:
            for g in ("0", "1"):
                stack.enter_context(
                    lock(Path("/data/yilangliu/GraD-Pert/runtime") / f"v2-gpu-{g}.lock")
                )
            record(phase="U3_evaluation")
            run_deferred_postfit(
                plans[1], gpu="0,1", evaluation_runtime=Path(schedule["evaluation_runtime"])
            )
        record(status="complete", phase="complete")
        atomic_json(directory / "COMPLETE.json", state)
    except BaseException as error:
        record(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(directory / "FAILURE.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    execute(json.loads(args.schedule.read_text()))


if __name__ == "__main__":
    main()
