"""Paired observer benchmark followed by unchanged full-medium acceptance.

Scientific source stays frozen at 420d3a1. This driver only runs read-only
diagnostics, never launches training and never reduces the medium population.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

from cap40_posthoc import METHOD_SHA, read, server_path, sha, source_identity, write

SKIP = {"observer_timings", "elapsed_seconds", "peak_allocated_mib"}


def compare(reference, candidate, *, tolerance=1e-8):
    """Validate scientific values, never performance timestamps or source labels."""
    maximum, compared = 0.0, 0

    def visit(a, b, path):
        nonlocal maximum, compared
        if isinstance(a, dict):
            ak, bk = set(a) - SKIP, set(b) - SKIP
            if ak != bk:
                raise ValueError(f"statistic keys differ at {path}: {ak ^ bk}")
            for k in sorted(ak):
                visit(a[k], b[k], path + "/" + k)
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f"row count differs at {path}")
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, path + "/" + str(i))
        elif isinstance(a, float):
            difference = abs(a - b)
            if not math.isfinite(difference) or difference > tolerance:
                raise ValueError(f"numeric difference {difference} at {path}")
            maximum = max(maximum, difference)
            compared += 1
        elif a != b:
            raise ValueError(f"identity/count differs at {path}: {a!r} != {b!r}")

    for name in ("deep.json", "gradients.json"):
        visit(read(reference / name), read(candidate / name), name)
    a, b = read(reference / "summary.json"), read(candidate / "summary.json")
    for key in (
        "manifest_sha256",
        "checkpoint_sha256",
        "conditions",
        "rows",
        "deep_conditions",
        "deep_rows",
        "gradient_conditions",
        "gradient_rows",
        "updates",
        "global_heads",
        "global_features",
        "cls_decomposition",
        "metrics",
    ):
        visit(a[key], b[key], "summary/" + key)
    for folder in (reference, candidate):
        complete = read(folder / "COMPLETE.json")
        s = read(folder / "summary.json")
        if not complete["state_unchanged"] or s["updates"] or s["state_before"] != s["state_after"]:
            raise ValueError("checkpoint/state changed")
        for name, meta in complete["outputs"].items():
            if sha(folder / name) != meta["sha256"]:
                raise ValueError("sealed output hash changed")
    for name in ("outputs_full.csv", "cls_within_condition.csv"):
        with (reference / name).open() as f:
            left = list(csv.DictReader(f))
        with (candidate / name).open() as f:
            right = list(csv.DictReader(f))
        # Native prediction values and basic CLS data are unaffected by observers.
        if left != right:
            raise ValueError(f"native predictions or basic identities changed: {name}")
    return {"passed": True, "max_abs": maximum, "compared_floats": compared, "atol": tolerance}


def execute(args):
    server_path(args.output)
    if args.output.exists():
        raise ValueError("fresh benchmark root required")
    args.output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2]
    identity = source_identity(source)
    q = read(args.parent_queue)
    if sha(args.parent_queue) != args.parent_queue_sha256:
        raise ValueError("parent queue hash mismatch")
    row = next(r for r in q["rows"] if r["arm"] == "DP")
    worker = source / "scripts/v2/cap40_posthoc.py"
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(Path(q["method_source"]) / "src"),
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="1",
    )
    state = {"status": "running", "source": identity, "training_sha": METHOD_SHA}
    write(args.output / "state.json", state)

    def run(label, backend, *, full=False):
        output = args.output / label
        command = [
            sys.executable,
            str(worker),
            "run",
            "--run-root",
            row["run_root"],
            "--config",
            row["config"],
            "--data-root",
            q["data_root"],
            "--output",
            str(output),
            "--manifest",
            q["manifest"],
            "--manifest-sha256",
            q["manifest_sha256"],
            "--resource-queue",
            q["resource_queue"],
            "--arm",
            "DP",
            "--stats-backend",
            backend,
        ]
        if not full:
            command += ["--limit", "3", "--gate-indices", "0,1,15", "--profile"]
        state.update(phase=label, command=command)
        write(args.output / "state.json", state)
        with (args.output / (label + ".log")).open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        s = read(output / "summary.json")
        if not read(output / "COMPLETE.json")["state_unchanged"]:
            raise ValueError("analysis mutated state")
        return s

    try:
        reference = run("reference-probe", "reference")
        fast = run("fast-probe", "fast")
        parity = compare(args.output / "reference-probe", args.output / "fast-probe")
        probe = {
            "parity": parity,
            "reference_seconds": reference["elapsed_seconds"],
            "fast_seconds": fast["elapsed_seconds"],
            "reference_regions": reference["observer_timings"],
            "fast_regions": fast["observer_timings"],
        }
        write(args.output / "PROBE_COMPLETE.json", probe)
        if fast["elapsed_seconds"] >= reference["elapsed_seconds"]:
            raise ValueError("candidate has no end-to-end probe gain; do not expand")
        full = run("fast-medium", "fast", full=True)
        parity_full = compare(Path(row["formal"]), args.output / "fast-medium")
        old = read(Path(row["formal"]) / "summary.json")
        write(
            args.output / "COMPLETE.json",
            {
                "status": "completed",
                "source": identity,
                "training_sha": METHOD_SHA,
                "manifest_sha256": q["manifest_sha256"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "probe": probe,
                "full_parity": parity_full,
                "full_reference_seconds": old["elapsed_seconds"],
                "full_fast_seconds": full["elapsed_seconds"],
                "full_peak_allocated_mib": full["peak_allocated_mib"],
                "reference_measurement": "previous sealed medium run; short profiles paired currently",
                "same_sample_scope": True,
                "model_updates": 0,
            },
        )
        state.update(status="completed", phase="completed")
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write(args.output / "FAILURE.json", state)
        raise
    finally:
        write(args.output / "state.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent-queue", type=Path, required=True)
    p.add_argument("--parent-queue-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    execute(p.parse_args())
