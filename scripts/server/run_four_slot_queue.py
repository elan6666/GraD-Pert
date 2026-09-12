"""Execute a hash-pinned queue with two slots per GPU, including postfit tests."""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import threading
from pathlib import Path

from scripts.server.run_r50_selection import sha
from scripts.server.run_r1024_steps import process_environment


def run_slots(rows, gpus, execute):
    """Event-driven refill: a task owns its slot until fit and tests terminate."""
    pending = iter(rows)
    lock = threading.Lock()
    fatal = threading.Event()
    results = {}

    def lane(slot):
        while not fatal.is_set():
            with lock:
                row = next(pending, None)
            if row is None:
                return
            try:
                result = execute(row, gpus[slot // 2], slot)
            except Exception:
                fatal.set()
                raise
            with lock:
                results[row] = result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lane, slot) for slot in range(4)]
        for future in futures:
            future.result()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    a = parser.parse_args()
    if sha(a.plan) != a.plan_sha:
        raise ValueError("queue plan hash mismatch")
    p = json.loads(a.plan.read_text())
    root, source = Path(p["queue_root"]), Path(p["source"])
    if root.exists() or not root.resolve().is_relative_to("/data/yilangliu/GraD-Pert"):
        raise ValueError("fresh server queue root required")
    if len(p["gpus"]) != 2 or len(set(p["gpus"])) != 2:
        raise ValueError("two distinct physical GPUs required")
    if len(set(p["rows"])) != len(p["rows"]):
        raise ValueError("duplicate row")

    def identity():
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            != p["commit"]
        ):
            raise ValueError("source drift")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
            raise ValueError("dirty source")
        if any(sha(Path(f)) != h for f, h in p["inputs"].items()):
            raise ValueError("input drift")

    identity()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(gpu in apps for gpu in p["gpus"]):
        raise ValueError("queue initial GPUs must be idle")
    root.mkdir(parents=True)
    process_lock = threading.Lock()
    active = {}
    aborted = threading.Event()

    def abort_owned():
        with process_lock:
            aborted.set()
            for process in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)

    def execute(row, gpu, slot):
        try:
            identity()
            with (root / (row + ".log")).open("x") as log:
                with process_lock:
                    if aborted.is_set():
                        raise RuntimeError("queue halted; refuse new dispatch")
                    identity()
                    process = subprocess.Popen(
                        p["commands"][row],
                        cwd=source,
                        env=process_environment(source, gpu),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[row] = process
                    (root / (row + "-dispatch.json")).write_text(
                        json.dumps(
                            {
                                "pid": process.pid,
                                "pgid": process.pid,
                                "gpu": gpu,
                                "slot": slot,
                                "commit": p["commit"],
                                "command": p["commands"][row],
                            }
                        )
                    )
                rc = process.wait()
                with process_lock:
                    active.pop(row)
            identity()
        except Exception as error:
            (root / (row + "-FATAL.json")).write_text(json.dumps({"error": str(error)}))
            abort_owned()
            raise
        result = {"row": row, "gpu": gpu, "slot": slot, "rc": rc, "commit": p["commit"]}
        (root / (row + ".json")).write_text(json.dumps(result))
        # Row commands are the full fit + postfit runner; a nonzero exit is
        # preserved, never retried. Other independent rows may use its slot.
        return result

    results = run_slots(p["rows"], p["gpus"], execute)
    (root / "TERMINAL.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
