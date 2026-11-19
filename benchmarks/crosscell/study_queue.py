"""Advance a frozen queue of official-runner experiments on the research server."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

MODELS = ("txpert", "scdfm", "scfoundation", "scbig", "perturbdiff", "prescribe")
ORDER = tuple(f"{model}-{scenario}" for model in MODELS for scenario in ("j2", "c2"))
CROSS_ONLY_ORDER = tuple(f"{model}-c2" for model in MODELS[1:])


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    support = plan.get("shared_fit_support_sha256")
    if support is not None and (
        job.get("shared_fit_support_sha256") != support
        or job["stages"][-1]["expected"].get("shared_fit_support_file_sha256") != support
    ):
        raise ValueError("C2 runner is not bound to the common fitting support")
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


def wait_for_txpert_c2(root, plan, state, state_path, poll_seconds):
    """Continue after the active C2 result, without restarting or overlapping it."""
    handoff_path = root / "predecessor_handoff.json"
    if handoff_path.exists():
        handoff = read(handoff_path)
        if digest(Path(handoff["c2_result"])) != handoff["sha256"]:
            raise ValueError("accepted TxPert C2 result changed")
        return
    predecessor = root.parent / "double-holdout-queue-a1"
    old_plan_sha = plan["predecessor_plan_sha256"]
    if digest(predecessor / "plan.json") != old_plan_sha:
        raise ValueError("predecessor plan identity changed")
    state.update(status="awaiting_txpert_c2", active_arm=None)
    write(state_path, state)
    while True:
        prior = read(predecessor / "state.json")
        if prior["plan_sha256"] != old_plan_sha:
            raise ValueError("predecessor state belongs to another plan")
        c2 = prior["arms"].get("txpert-c2", {})
        if (
            c2.get("status") == "completed"
            and prior["active_arm"] == "scdfm-j2"
            and prior["status"] == "awaiting_verified_runner"
        ):
            break
        time.sleep(poll_seconds)
    result = Path(c2["receipt"]).resolve(strict=True)
    if result.parent.parent != root.parent:
        raise ValueError("predecessor C2 receipt is outside the study")
    receipt = read(result)
    if (
        receipt.get("scenario") != "C2"
        or receipt.get("status") != "evaluated"
        or receipt.get("formal_eligible") is not True
        or receipt.get("conditions") != 128
        or receipt.get("shared_fit_support_file_sha256") != plan["shared_fit_support_sha256"]
    ):
        raise ValueError("predecessor C2 did not seal the shared frozen cohort")
    if (predecessor / "ready/scdfm-j2.json").exists():
        raise ValueError("retired Jurkat arm must not have an executable descriptor")
    pid = int((predecessor / "controller.pid").read_text())
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        pidfd = None
    if pidfd is not None:
        try:
            process = Path(f"/proc/{pid}")
            argv = os.fsdecode((process / "cmdline").read_bytes()).rstrip("\0").split("\0")
            children = (process / f"task/{pid}/children").read_text().strip()
            if argv != [""] and (
                argv[2:] != [str(predecessor / "controller.py"), str(predecessor), old_plan_sha]
                or process.stat().st_uid != os.getuid()
                or children
            ):
                raise ValueError("predecessor is not the owned, idle queue controller")
            if argv != [""]:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        finally:
            os.close(pidfd)
    write(
        handoff_path, {"c2_result": str(result), "sha256": digest(result), "predecessor_pid": pid}
    )


def deferred_run(root, arm, record):
    path = root / "deferrals" / f"{arm}.json"
    if not path.is_file():
        return None
    spec = read(path)
    run = Path(record["run_root"]).resolve(strict=True)
    marker = run / "SCOPE_DEFER.json"
    if (
        record.get("status") not in ("blocked", "deferred_by_user")
        or spec["arm_id"] != arm
        or run.parent != root.parent
        or spec["marker_sha256"] != digest(marker)
        or (run / "result.json").exists()
    ):
        raise ValueError("deferred arm must match its stopped run and immutable receipt")
    note = read(marker)
    if (
        note["reason"] != "deferred_by_user"
        or note["run_id"] != run.name
        or note["descriptor_sha256"] != record["descriptor_sha256"]
    ):
        raise ValueError("user deferral belongs to another descriptor or run")
    return {
        "run_root": str(run),
        "descriptor_sha256": record["descriptor_sha256"],
        "deferral_receipt": str(marker),
        "deferral_receipt_sha256": spec["marker_sha256"],
    }


def run(root, expected_plan_sha256, poll_seconds):
    plan_path = root / "plan.json"
    if digest(plan_path) != expected_plan_sha256:
        raise ValueError("queue plan hash mismatch")
    plan = read(plan_path)
    order = tuple(plan["order"])
    if order not in (ORDER, CROSS_ONLY_ORDER):
        raise ValueError("the queue must match an authorized study order")
    state_path = root / "state.json"
    state = {"plan_sha256": expected_plan_sha256, "order": list(order), "arms": {}}

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
    if order == CROSS_ONLY_ORDER:
        wait_for_txpert_c2(root, plan, state, state_path, poll_seconds)
    for arm in order:
        record = state["arms"].get(arm, {})
        if record.get("status") == "completed":
            continue
        deferred = deferred_run(root, arm, record)
        if deferred is not None:
            if record["status"] != "deferred_by_user":
                set_arm(arm, "deferred_by_user", **deferred)
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
