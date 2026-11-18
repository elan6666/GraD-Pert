"""Advance a frozen queue of official-runner experiments on the research server."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

MODELS = ("txpert", "scdfm", "scfoundation", "scbig", "perturbdiff", "prescribe")
ORDER = tuple(f"{model}-{scenario}" for model in MODELS for scenario in ("j2", "c2"))


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def check_receipt(item):
    receipt = Path(item["receipt"])
    if not receipt.is_file():
        return False
    data = read(receipt)
    if any(data.get(key) != expected for key, expected in item["expected"].items()):
        raise ValueError(f"completion receipt disagrees with plan: {receipt}")
    return True


def load_job(root, arm, plan):
    ready = root / "ready" / f"{arm}.json"
    if not ready.exists():
        return None
    job = read(ready)
    if (
        job["arm_id"] != arm
        or job["cohort_sha256"] != plan["cohort_sha256"]
        or [stage["name"] for stage in job["stages"]] != ["smoke", "formal"]
    ):
        raise ValueError(f"unrecognized queue descriptor: {ready}")
    repository = Path(job["repository"]).resolve(strict=True)
    for spec in job["stages"]:
        for key in ("log", "receipt", "run_root"):
            if not Path(spec[key]).resolve().is_relative_to(root.parent):
                raise ValueError(f"experiment output must stay under the study root: {key}")
        if len(spec["command"]) < 2 or not Path(spec["command"][0]).is_absolute():
            raise ValueError("an absolute runner executable is required")
    observed_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    if observed_commit != job["source_commit"] or dirty:
        raise ValueError(f"queue runner differs from its sealed clean source: {repository}")
    return job, digest(ready)


def run(root, expected_plan_sha256, poll_seconds):
    plan_path = root / "plan.json"
    if digest(plan_path) != expected_plan_sha256:
        raise ValueError("queue plan hash mismatch")
    plan = read(plan_path)
    if tuple(plan["order"]) != ORDER:
        raise ValueError("the queue must contain each of the twelve authorized J2/C2 arms once")
    state_path = root / "state.json"
    state = {"plan_sha256": expected_plan_sha256, "order": list(ORDER), "arms": {}}

    def set_arm(arm, status, **fields):
        state["status"] = status
        state["active_arm"] = arm
        state["arms"][arm] = {
            "status": status,
            "updated_unix": time.time(),
            **fields,
        }
        write(state_path, state)

    if state_path.exists():
        state = read(state_path)
        if state["plan_sha256"] != expected_plan_sha256:
            raise ValueError("existing queue state belongs to a different plan")
    for arm in ORDER:
        record = state["arms"].get(arm, {})
        if record.get("status") == "completed":
            continue
        while True:
            prepared = load_job(root, arm, plan)
            if prepared is not None:
                break
            if record.get("status") != "awaiting_verified_runner":
                set_arm(arm, "awaiting_verified_runner")
                record = state["arms"][arm]
            time.sleep(poll_seconds)
        job, descriptor_sha256 = prepared
        for stage in job["stages"]:
            try:
                if check_receipt(stage):
                    continue
                log_path = Path(stage["log"])
                if log_path.exists() or Path(stage["run_root"]).exists():
                    raise RuntimeError("attempt exists without an accepted terminal receipt")
                with log_path.open("xb", buffering=0) as output:
                    print(shlex.join(stage["command"]), flush=True)
                    with subprocess.Popen(
                        stage["command"],
                        cwd=job["repository"],
                        env={**os.environ, **job["env"]},
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                    ) as process:
                        set_arm(
                            arm,
                            "running",
                            stage=stage["name"],
                            pid=process.pid,
                            run_root=stage["run_root"],
                            log=stage["log"],
                            descriptor_sha256=descriptor_sha256,
                        )
                        code = process.wait()
                if code != 0 or not check_receipt(stage):
                    raise RuntimeError(f"runner exited with code {code} without a valid receipt")
            except (OSError, ValueError, RuntimeError) as error:
                set_arm(
                    arm,
                    "blocked",
                    stage=stage["name"],
                    log=stage["log"],
                    run_root=stage["run_root"],
                    error=str(error),
                    descriptor_sha256=descriptor_sha256,
                )
                raise
        set_arm(
            arm,
            "completed",
            run_root=job["stages"][-1]["run_root"],
            receipt=job["stages"][-1]["receipt"],
            descriptor_sha256=descriptor_sha256,
        )
    state.update(status="completed", active_arm=None)
    write(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("plan_sha256")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    with (root / "controller.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, args.plan_sha256, 30)


if __name__ == "__main__":
    main()
