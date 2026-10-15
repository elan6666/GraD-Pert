"""Sequential immutable analysis queue; never fits or updates a model."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cap40_posthoc import read, sha, source_identity, write


def execute(args):
    q = read(args.queue)
    if sha(args.queue) != args.queue_sha256:
        raise ValueError("queue hash differs")
    root = Path(__file__).resolve().parents[2]
    if source_identity(root)["commit"] != q["analysis_commit"]:
        raise ValueError("analysis version differs")
    state = args.queue.parent / "state.json"
    if state.exists():
        raise ValueError("queue cannot be launched twice; use a fresh ID after failure")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=q["method_source"] + "/src",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    try:
        phases = ("gate",) if args.gate_only else ("formal",)
        for phase in phases:
            for row in q["rows"]:
                arm = row["arm"]
                if phase == "formal":
                    for peer in q["rows"]:
                        gate = read(Path(peer["gate"]) / "COMPLETE.json")
                        if (
                            gate["scope"] != "gate_subset"
                            or not gate["state_unchanged"]
                            or gate["manifest_sha256"] != q["manifest_sha256"]
                            or gate["checkpoint_sha256"] != peer["checkpoint_sha256"]
                        ):
                            raise ValueError("all five read-only parity gates must pass")
                destination = row[phase]
                write(
                    state, {"status": "running", "phase": phase, "arm": arm, "output": destination}
                )
                command = [
                    sys.executable,
                    str(root / "scripts/v2/cap40_posthoc.py"),
                    "run",
                    "--arm",
                    arm,
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
                if phase == "gate":
                    command += [
                        "--limit",
                        "1",
                        "--gate-indices",
                        ",".join(map(str, q["gate_indices"])),
                    ]
                with (args.queue.parent / f"{phase}-{arm}.log").open("x") as log:
                    subprocess.run(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                if phase == "formal":
                    with (args.queue.parent / f"report-{arm}.log").open("x") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(root / "scripts/v2/cap40_posthoc_report.py"),
                                "--input",
                                destination,
                                "--original",
                                q["original_analysis"] + "/" + arm,
                                "--output",
                                row["report"],
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
        if not args.gate_only:
            with (args.queue.parent / "comparison.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(root / "scripts/v2/cap40_posthoc_compare.py"),
                        "--queue",
                        str(args.queue),
                        "--output",
                        q["comparison"],
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        write(
            state, {"status": "completed", "phase": phases[-1], "queue_sha256": args.queue_sha256}
        )
    except BaseException as error:
        write(
            args.queue.parent / "FAILURE.json",
            {"error": repr(error), "queue_sha256": args.queue_sha256},
        )
        write(state, {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--queue-sha256", required=True)
    p.add_argument("--gate-only", action="store_true")
    execute(p.parse_args())
