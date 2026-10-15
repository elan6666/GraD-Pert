"""Two one-GPU analysis slots, adopting exact live/completed immutable work."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from cap40_posthoc_priority import proc, read, same_process, sha, write


def complete(row, q):
    folder = Path(row["formal"])
    receipt = read(folder / "COMPLETE.json")
    if (
        receipt["status"] != "completed"
        or receipt["scope"] != "medium_cap40_stratified"
        or not receipt["state_unchanged"]
        or receipt["manifest_sha256"] != q["manifest_sha256"]
        or receipt["checkpoint_sha256"] != row["checkpoint_sha256"]
    ):
        raise ValueError("analysis completion identity differs")
    for name, entry in receipt["outputs"].items():
        if Path(name).name != name or sha(folder / name) != entry["sha256"]:
            raise ValueError("analysis output hash differs")
    s = read(folder / "summary.json")
    counts = {
        "conditions": 1335,
        "rows": 5317,
        "deep_conditions": 128,
        "deep_rows": 506,
        "gradient_conditions": 32,
        "gradient_rows": 125,
    }
    if (
        any(s[k] != v for k, v in counts.items())
        or s["updates"] != 0
        or s["arm"] != row["arm"]
        or s["analysis"]["commit"] != row["analysis_commit"]
        or s["manifest_sha256"] != q["manifest_sha256"]
        or s["checkpoint_sha256"] != row["checkpoint_sha256"]
    ):
        raise ValueError("analysis population/source differs")
    return sha(folder / "COMPLETE.json")


def validate(q):
    root = Path(__file__).resolve().parents[2]
    if Path(q["worker_source"]).resolve() != root:
        raise ValueError("worker and dispatcher checkout differ")
    if (
        subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        != q["analysis_commit"]
    ):
        raise ValueError("dispatcher source differs")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]):
        raise ValueError("dispatcher source dirty")
    if sha(q["manifest"]) != q["manifest_sha256"]:
        raise ValueError("manifest differs")
    if [r["arm"] for r in q["rows"]] != ["DP", "FN", "FG", "FB", "PO"]:
        raise ValueError("wrong family")
    if proc(q["retired_controller"]["pid"]) is not None:
        raise ValueError("old dispatcher still alive")
    if not set(q["pending"]).issubset({"FN", "FG", "FB"}):
        raise ValueError("pending scope differs")
    expected = set(q["pending"]) | set(q["completed"]) | set(q.get("adopted", {}))
    if len(q["pending"]) + len(q["completed"]) + len(q.get("adopted", {})) != 5 or expected != {
        "DP",
        "FN",
        "FG",
        "FB",
        "PO",
    }:
        raise ValueError("every arm needs exactly one execution owner")
    free_slots(q.get("adopted", {}))
    for r in q["rows"]:
        g = read(Path(r["gate"]) / "COMPLETE.json")
        if (
            g["scope"] != "gate_subset"
            or not g["state_unchanged"]
            or g["manifest_sha256"] != q["manifest_sha256"]
            or g["checkpoint_sha256"] != r["checkpoint_sha256"]
        ):
            raise ValueError("frozen gate mismatch")


def free_slots(active):
    devices = [v["gpu"] for v in active.values()]
    if len(set(devices)) != len(devices) or any(g not in (0, 1) for g in devices):
        raise ValueError("one analysis per physical GPU")
    return [g for g in (0, 1) if g not in devices]


def execute(args):
    if sha(args.queue) != args.queue_sha256:
        raise ValueError("queue hash differs")
    q = read(args.queue)
    validate(q)
    root = args.queue.parent
    state_path = root / ("gate-state.json" if args.gate_only else "state.json")
    if state_path.exists():
        raise ValueError("fresh dispatcher state required")
    rows = {r["arm"]: r for r in q["rows"]}
    active = {a: {**v, "child": None} for a, v in q.get("adopted", {}).items()}
    state = {
        "status": "running",
        "phase": "gate" if args.gate_only else "formal",
        "completed": {},
        "active": {},
        "queue_sha256": args.queue_sha256,
    }

    def report(a):
        row = rows[a]
        complete(row, q)
        destination = Path(row["report"])
        if not destination.exists():
            with (root / f"report-{a}.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(q["worker_source"]) / "scripts/v2/cap40_posthoc_report.py"),
                        "--input",
                        row["formal"],
                        "--original",
                        q["original_analysis"] + "/" + a,
                        "--output",
                        row["report"],
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        receipt = read(destination / "receipt.json")
        if (
            receipt["manifest_sha256"] != q["manifest_sha256"]
            or receipt["checkpoint_sha256"] != row["checkpoint_sha256"]
        ):
            raise ValueError("report binding differs")
        for name, e in receipt["outputs"].items():
            if sha(destination / name) != e["sha256"]:
                raise ValueError("report bytes differ")
        state["completed"][a] = sha(destination / "receipt.json")

    def start(a, gpu, gate=False):
        row = rows[a]
        destination = row["dual_gate"] if gate else row["formal"]
        if Path(destination).exists():
            raise ValueError("refuse duplicate output")
        cmd = [
            sys.executable,
            str(Path(q["worker_source"]) / "scripts/v2/cap40_posthoc.py"),
            "run",
            "--arm",
            a,
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
            destination,
        ]
        for peer_arm, v in active.items():
            if v["gpu"] != gpu and same_process(v["process"]):
                peer = {
                    "schema": "cap40-exact-analysis-peer-1",
                    "arm": peer_arm,
                    "gpu": v["gpu"],
                    "uid": os.getuid(),
                    "process": v["process"],
                    "manifest_sha256": q["manifest_sha256"],
                }
                path = root / f"{a}-{'gate' if gate else 'formal'}-peer.json"
                write(path, peer)
                cmd += ["--peer-receipt", str(path), "--peer-receipt-sha256", sha(path)]
        if gate:
            cmd += ["--limit", "1", "--gate-indices", "0,1,15"]
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTORCH_ALLOC_CONF="expandable_segments:True",
            PYTHONPATH=q["method_source"] + "/src",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        with (root / f"{'gate' if gate else 'formal'}-{a}.log").open("x") as log:
            child = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        identity = proc(child.pid)
        if identity is None:
            raise RuntimeError("worker exited before identity registration")
        active[a] = {"gpu": gpu, "process": identity, "child": child}
        # Admission precedes the first CUDA allocation. Do not race the first
        # worker's ownership snapshot with another newly allocated peer.
        deadline = time.monotonic() + 180
        while not (Path(destination) / "ADMISSION.json").exists():
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("worker admission failed or timed out")
            time.sleep(1)

    try:
        for a in q["completed"]:
            report(a)
        if args.gate_only:
            for a in q["pending"]:
                # The adopted worker remains untouched. Probe the other GPU.
                busy = {v["gpu"] for v in active.values() if same_process(v["process"])}
                gpu = next(g for g in (1, 0) if g not in busy)
                start(a, gpu, True)
                if active[a]["child"].wait():
                    raise RuntimeError(f"{a} gate failed")
                del active[a]
                gate = read(Path(rows[a]["dual_gate"]) / "COMPLETE.json")
                if (
                    gate["scope"] != "gate_subset"
                    or not gate["state_unchanged"]
                    or gate["manifest_sha256"] != q["manifest_sha256"]
                    or gate["checkpoint_sha256"] != rows[a]["checkpoint_sha256"]
                ):
                    raise ValueError("new GPU gate failed")
            state.update(status="gate_completed", phase="gate")
            write(root / "PREFLIGHT_COMPLETE.json", state)
        else:
            acceptance = read(root / "MAIN_ACCEPTANCE.json")
            if acceptance["queue_sha256"] != args.queue_sha256 or not acceptance["accepted"]:
                raise ValueError("main acceptance missing")
            pending = list(q["pending"])
            while active or pending:
                for a, v in list(active.items()):
                    child = v["child"]
                    alive = (
                        child.poll() is None if child is not None else same_process(v["process"])
                    )
                    if not alive:
                        if child is not None and child.returncode:
                            raise RuntimeError(f"{a} failed: {child.returncode}")
                        report(a)
                        del active[a]
                for gpu in free_slots(active):
                    if pending:
                        start(pending.pop(0), gpu)
                state["active"] = {
                    a: {k: v[k] for k in ("gpu", "process")} for a, v in active.items()
                }
                state["pending"] = pending
                state["observed_unix"] = time.time()
                write(state_path, state)
                if active:
                    time.sleep(5)
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
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            state.update(status="completed", phase="completed")
        write(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        write(root / ("GATE_FAILURE.json" if args.gate_only else "FAILURE.json"), state)
        write(state_path, state)
        # Preserve every still-running adopted/new worker; main owns repairs.
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--gate-only", action="store_true")
    execute(parser.parse_args())
