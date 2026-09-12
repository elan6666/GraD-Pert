"""Bounded eleven-row one-step queue, two independent processes per idle GPU."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.server.run_r50_selection import sha
from scripts.server.run_r1024_full import ROWS


def process_environment(source: Path, gpu: str) -> dict[str, str]:
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))),
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        GRADPERT_SPARSE_UNION_IMPL="cpu_array",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha:
        raise ValueError("plan hash mismatch")
    plan = json.loads(args.plan.read_text())
    source, root = Path(plan["source"]), Path(plan["root"])
    if root.exists() or not root.is_relative_to("/data/yilangliu/GraD-Pert/runs"):
        raise ValueError("fresh server root required")
    if tuple(plan["rows"]) != ROWS or len(set(plan["gpus"])) != 2:
        raise ValueError("exact eleven rows and two physical GPUs required")

    def identity():
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            != plan["commit"]
        ):
            raise ValueError("source commit differs")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
            raise ValueError("source dirty")
        for path, digest in plan["inputs"].items():
            if sha(Path(path)) != digest:
                raise ValueError("immutable input drift")

    identity()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if any(gpu in apps for gpu in plan["gpus"]):
        raise ValueError("initial capacity test requires both GPUs idle; preserve peers")
    root.mkdir(parents=True)

    def lane(slot):
        gpu = plan["gpus"][slot // 2]
        env = process_environment(source, gpu)
        for row in ROWS[slot::4]:
            identity()
            started = time.time()
            cmd = [
                sys.executable,
                "-m",
                "scripts.server.native_step_smoke",
                "--config-path",
                str(source / f"configs/r50/{row}/gradpert_b2/nadig_jurkat.yaml"),
                "--repository-root",
                str(source),
                "--run-root",
                str(root / row),
                "--data-root",
                plan["data_root"],
                "--run-id",
                f"{root.name}/{row}",
                "--source-publication-receipt",
                plan["publication"],
                "--source-publication-receipt-sha256",
                plan["inputs"][plan["publication"]],
                "--genept-preflight-receipt",
                plan["prior"],
                "--genept-preflight-receipt-sha256",
                plan["inputs"][plan["prior"]],
                "--memory-fraction",
                "0.4",
            ]
            with (root / (row + ".log")).open("x") as log:
                rc = subprocess.run(
                    cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            receipt = root / row / "small_results/one_step_smoke.json"
            if rc == 0 and not receipt.is_file():
                rc = 90
            result = dict(
                row=row,
                slot=slot,
                gpu=gpu,
                pid_scope="child_in_log",
                started=started,
                ended=time.time(),
                rc=rc,
                commit=plan["commit"],
                config_sha256=plan["inputs"][cmd[4]],
                receipt_sha256=sha(receipt) if receipt.exists() else None,
            )
            (root / (row + "-exit.json")).write_text(json.dumps(result))
            # Fail this lane closed; no retry and no formal work from this script.
            if rc:
                break

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lane, range(4)))
    identity()
    (root / "QUEUE_TERMINAL.json").write_text(
        json.dumps(
            {
                "commit": plan["commit"],
                "scientific_completion": False,
                "exits": [json.loads(p.read_text()) for p in sorted(root.glob("*-exit.json"))],
            }
        )
    )


if __name__ == "__main__":
    main()
