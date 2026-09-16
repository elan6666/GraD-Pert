"""Seven pinned native rows: all one-step gates, then two exclusive GPU lanes."""

import argparse
import concurrent.futures
import fcntl
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from scripts.server.run_r50_selection import sha
from scripts.server.run_r1024_full import NEW_ROWS
from scripts.server.run_r1024_steps import process_environment


def step_valid(receipt, config_sha, commit):
    p = json.loads(receipt.read_text())
    identity_path = receipt.parent / "source_identity.json"
    identity = json.loads(identity_path.read_text())
    return (
        p["status"] == "complete"
        and p["config_sha256"] == config_sha
        and p["source_identity_sha256"] == sha(identity_path)
        and identity["commit"] == commit
        and not identity["dirty"]
        and p["test_evaluations"] == 0
        and len(p["steps"]) == 1
        and p["steps"][0]["completed_steps"] == 1
        and p["steps"][0]["checkpoint_roundtrip_exact"]
        and p["steps"][0]["resources"]["cuda_acceptance"]
    )


def gpu_idle(gpu):
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if gpu in apps:
        raise RuntimeError(f"GPU {gpu} occupied; preserve peers and defer lane")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    a = parser.parse_args()
    if sha(a.plan) != a.plan_sha:
        raise ValueError("plan changed")
    plan = json.loads(a.plan.read_text())
    source, root = Path(plan["source"]), Path(plan["root"])
    if tuple(plan["rows"]) != NEW_ROWS or len(set(plan["gpus"])) != 2:
        raise ValueError("exact authorized rows and two distinct GPU UUIDs required")
    if root.exists() or not root.resolve().is_relative_to("/data/yilangliu/GraD-Pert/runs"):
        raise ValueError("fresh server root required")

    def identity():
        if sha(a.plan) != a.plan_sha:
            raise ValueError("plan changed during queue")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
        if head != plan["commit"] or dirty.strip():
            raise ValueError("source identity changed")
        for path, expected in plan["inputs"].items():
            if sha(Path(path)) != expected:
                raise ValueError(f"pinned input changed: {path}")

    identity()
    # Locks live outside individual run roots, and survive for both phases.
    locks = ExitStack()
    for gpu in sorted(plan["gpus"]):
        lock = locks.enter_context(
            open(f"/data/yilangliu/GraD-Pert/contracts/exclusive-{gpu}.lock", "a")  # noqa: SIM115
        )
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_idle(gpu)
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan, indent=2))

    def run(row, slot, phase):
        identity()
        gpu = plan["gpus"][slot]
        gpu_idle(gpu)
        cfg = source / f"configs/r50/{row}/gradpert_b2/nadig_jurkat.yaml"
        step = root / "steps" / row / "small_results/one_step_smoke.json"
        common = [
            "--source-publication-receipt",
            plan["publication"],
            "--source-publication-receipt-sha256",
            plan["inputs"][plan["publication"]],
            "--genept-preflight-receipt",
            plan["prior"],
            "--genept-preflight-receipt-sha256",
            plan["inputs"][plan["prior"]],
        ]
        if phase == "step":
            cmd = [
                sys.executable,
                "-m",
                "scripts.server.native_step_smoke",
                "--config-path",
                str(cfg),
                "--repository-root",
                str(source),
                "--run-root",
                str(root / "steps" / row),
                "--data-root",
                plan["data_root"],
                "--run-id",
                f"{root.name}/steps/{row}",
                "--memory-fraction",
                "0.4",
                *common,
            ]
        else:
            if not step_valid(step, plan["inputs"][str(cfg)], plan["commit"]):
                raise ValueError("full run requires successful matching step")
            cmd = [
                sys.executable,
                "-m",
                "scripts.server.run_r1024_full",
                "--row",
                row,
                "--source",
                str(source),
                "--root",
                str(root / "full" / row),
                "--data-root",
                plan["data_root"],
                "--publication",
                plan["publication"],
                "--publication-sha",
                plan["inputs"][plan["publication"]],
                "--genept-receipt",
                plan["prior"],
                "--genept-sha",
                plan["inputs"][plan["prior"]],
                "--step-receipt",
                str(step),
                "--step-sha",
                sha(step),
                "--commit",
                plan["commit"],
                "--config-sha",
                plan["inputs"][str(cfg)],
                "--memory-fraction",
                "0.4",
            ]
        record = dict(
            row=row,
            phase=phase,
            gpu=gpu,
            source_commit=plan["commit"],
            started_unix=time.time(),
            config_sha256=sha(cfg),
        )
        (root / f"{row}-{phase}-started.json").write_text(json.dumps(record))
        with (root / f"{row}-{phase}.log").open("x") as log:
            rc = subprocess.run(
                cmd,
                cwd=source,
                env=process_environment(source, gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        if rc == 0 and phase == "step" and not step_valid(step, sha(cfg), plan["commit"]):
            rc = 90
        if rc == 0 and phase == "full" and not (root / "full" / row / "COMPLETE.json").exists():
            rc = 91
        record.update(rc=rc, ended_unix=time.time())
        (root / f"{row}-{phase}-exit.json").write_text(json.dumps(record))
        return rc

    def steps(slot):
        passed = []
        for row in NEW_ROWS[slot::2]:
            if run(row, slot, "step") == 0:
                passed.append(row)
        return passed

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        passed = list(pool.map(steps, range(2)))
    (root / "STEP_PHASE_COMPLETE.json").write_text(json.dumps({"passed_by_lane": passed}))

    def full(slot):
        for row in passed[slot]:
            # No retry after failure. Independent next rows remain authorized.
            run(row, slot, "full")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(full, range(2)))
    (root / "QUEUE_TERMINAL.json").write_text(
        json.dumps(
            {
                "source_commit": plan["commit"],
                "ended_unix": time.time(),
                "exits": [json.loads(p.read_text()) for p in sorted(root.glob("*-exit.json"))],
                "blocked_designs": [],
                "note": "queue terminal does not mean every scientific row succeeded",
            }
        )
    )


if __name__ == "__main__":
    main()
