"""Prioritize PO after adopting an existing immutable DP worker on GPU0.

Scheduling only. Workers, reports and comparison use the frozen analysis source.
Never restarts an adopted worker or changes scientific inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def proc(pid, proc_root=Path("/proc")):
    root = proc_root / str(pid)
    try:
        stat = (root / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "command_sha256": sha(root / "cmdline"),
        }
    except FileNotFoundError:
        return None


def same_process(expected):
    actual = proc(expected["pid"])
    if actual is not None and actual != expected:
        raise ValueError("adopted PID identity changed")
    return actual is not None


def validate_complete(row, q):
    folder = Path(row["formal"])
    complete = read(folder / "COMPLETE.json")
    if (
        complete["status"] != "completed"
        or complete["scope"] != "full_cap40_train"
        or complete["manifest_sha256"] != q["manifest_sha256"]
        or complete["checkpoint_sha256"] != row["checkpoint_sha256"]
        or not complete["state_unchanged"]
    ):
        raise ValueError("adopted analysis completion identity differs")
    for name, entry in complete["outputs"].items():
        if Path(name).name != name or sha(folder / name) != entry["sha256"]:
            raise ValueError("analysis output hash differs")
    summary = read(folder / "summary.json")
    if (
        summary["arm"] != row["arm"]
        or summary["rows"] != q["expected_rows"]
        or summary["conditions"] != q["expected_conditions"]
        or summary["deep_rows"] != q["expected_deep_rows"]
        or summary["updates"] != 0
        or summary["analysis"]["commit"] != q["analysis_commit"]
        or summary["manifest_sha256"] != q["manifest_sha256"]
        or summary["checkpoint_sha256"] != row["checkpoint_sha256"]
    ):
        raise ValueError("analysis population or source differs")
    return sha(folder / "COMPLETE.json")


def validate_plan(q):
    own_source = Path(__file__).resolve().parents[2]
    own_commit = subprocess.check_output(
        ["git", "-C", str(own_source), "rev-parse", "HEAD"], text=True
    ).strip()
    own_dirty = subprocess.check_output(["git", "-C", str(own_source), "status", "--porcelain"])
    if own_commit != q["orchestration_commit"] or own_dirty:
        raise ValueError("orchestration source differs")
    if [r["arm"] for r in q["rows"]] != ["DP", "FN", "FG", "FB", "PO"]:
        raise ValueError("wrong family")
    if q["priority_arm"] != "PO" or q["adopted_arm"] != "DP":
        raise ValueError("unsupported priority/adoption")
    if sha(q["manifest"]) != q["manifest_sha256"]:
        raise ValueError("manifest differs")
    for row in q["rows"]:
        gate = read(Path(row["gate"]) / "COMPLETE.json")
        if (
            gate["scope"] != "gate_subset"
            or not gate["state_unchanged"]
            or gate["manifest_sha256"] != q["manifest_sha256"]
            or gate["checkpoint_sha256"] != row["checkpoint_sha256"]
        ):
            raise ValueError("missing frozen gate")
    source = Path(q["worker_source"])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True)
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"])
    if commit.strip() != q["analysis_commit"] or dirty:
        raise ValueError("worker source differs or dirty")
    for filename, expected in q["worker_files"].items():
        if sha(source / filename) != expected:
            raise ValueError("worker script differs")


def command(q, row):
    return [
        sys.executable,
        str(Path(q["worker_source"]) / "scripts/v2/cap40_posthoc.py"),
        "run",
        "--arm",
        row["arm"],
        "--run-root",
        row["run_root"],
        "--config",
        row["config"],
        "--data-root",
        q["data_root"],
        "--manifest",
        q["manifest"],
        "--manifest-sha256",
        q["manifest_sha256"],
        "--resource-queue",
        q["resource_queue"],
        "--output",
        row["formal"],
    ]


def environment(q, gpu):
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=q["method_source"] + "/src",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )


def execute(args):
    if sha(args.queue) != args.queue_sha256:
        raise ValueError("priority queue hash differs")
    q = read(args.queue)
    root = args.queue.parent
    state_path = root / "state.json"
    if state_path.exists():
        raise ValueError("fresh scheduling ID required")
    validate_plan(q)
    if not same_process(q["adopted_worker"]):
        raise ValueError("adopted worker absent before priority launch")
    # Main must retire only the old parent; no two controllers may dispatch peers.
    if proc(q["retired_controller"]["pid"]) is not None:
        raise ValueError("old dispatch controller is still alive")
    rows = {r["arm"]: r for r in q["rows"]}
    state = {"status": "running", "priority_arm": "PO", "adopted_arm": "DP"}
    state["queue_sha256"] = args.queue_sha256

    def report(arm):
        row = rows[arm]
        validate_complete(row, q)
        state["phase"] = "report_" + arm
        write(state_path, state)
        with (root / f"report-{arm}.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(q["worker_source"]) / "scripts/v2/cap40_posthoc_report.py"),
                    "--input",
                    row["formal"],
                    "--original",
                    q["original_analysis"] + "/" + arm,
                    "--output",
                    row["report"],
                ],
                env=environment(q, 0),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        state.setdefault("completed_reports", []).append(arm)
        write(state_path, state)

    def start(arm, gpu):
        row = rows[arm]
        if Path(row["formal"]).exists():
            raise ValueError("refuse duplicate analysis destination")
        with (root / f"formal-{arm}.log").open("x") as log:
            child = subprocess.Popen(
                command(q, row),
                env=environment(q, gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        identity = proc(child.pid)
        state.setdefault("workers", {})[arm] = {"identity": identity, "gpu": gpu}
        state["phase"] = "formal_" + arm
        write(state_path, state)
        return child

    try:
        # The frozen per-worker guard treats independent GPU peers as foreign.
        # Preserve DP instead of weakening that guard or launching a peer early.
        state["phase"] = "await_adopted_DP"
        write(state_path, state)
        while same_process(q["adopted_worker"]):
            state["observed_unix"] = time.time()
            write(state_path, state)
            time.sleep(10)
        validate_complete(rows["DP"], q)
        priority = start("PO", 1)
        report("DP")  # CPU-only report; no second GPU model or guard bypass.
        state["phase"] = "formal_PO"
        write(state_path, state)
        while priority.poll() is None:
            state["observed_unix"] = time.time()
            write(state_path, state)
            time.sleep(10)
        if priority.returncode:
            raise RuntimeError(f"PO worker exited {priority.returncode}")
        report("PO")
        for arm in ("FN", "FG", "FB"):
            child = start(arm, 0)
            if child.wait():
                raise RuntimeError(f"{arm} worker failed")
            report(arm)
        state["phase"] = "comparison"
        write(state_path, state)
        with (root / "comparison.log").open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(q["worker_source"]) / "scripts/v2/cap40_posthoc_compare.py"),
                    "--queue",
                    str(args.queue),
                    "--output",
                    q["comparison"],
                ],
                env=environment(q, 0),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        state.update(status="completed", phase="completed")
        write(state_path, state)
    except BaseException as error:
        # Preserve all independent/adopted processes and receipts for main repair.
        state.update(status="failed", error=repr(error))
        write(state_path, state)
        write(root / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    execute(parser.parse_args())
