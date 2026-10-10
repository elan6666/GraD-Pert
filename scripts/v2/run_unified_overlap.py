"""Migrate the first ten arms to bounded, independent fit/postfit GPU lanes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_group import lock
from run_unified_group import (
    ALLOCATOR,
    ARMS,
    LEASE_ROOT,
    _preflight_entries,
    child_environment,
    gpu_idle,
    validate_completed,
    verify_seal,
)

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.train_entry import resolve_plan
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file

SCHEMA = "gradpert-v2-unified-overlap-1"
JSON = dict[str, Any]
FIT_FRACTION, EVAL_FRACTION = 0.60, 0.25
SERVER_ROOT = Path("/data/yilangliu")


def process_command(pid: int) -> bytes:
    return Path(f"/proc/{pid}/cmdline").read_bytes()


def process_identity(pid: int) -> JSON | None:
    """Bind both command and kernel start time; a recycled PID is not the child."""
    root = Path(f"/proc/{pid}")
    try:
        command = process_command(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if not command or fields[0] == "Z":
        return None
    return {
        "pid": pid,
        "command_sha256": hashlib.sha256(command).hexdigest(),
        "start_ticks": fields[19],
    }


def process_alive(expected: JSON) -> bool:
    actual = process_identity(expected["pid"])
    if actual is not None and actual != expected:
        raise ValueError("PID identity changed; preserve the unrelated process")
    return actual is not None


def method_tree(source: Path) -> str:
    """All native source/config and scientific launch/probe blobs must match exactly."""
    paths = ["src", "configs", "scripts/v2/capacity_probe.py", "scripts/v2/distributed_train.py"]
    blobs = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "HEAD", "--", *paths]
    )
    return hashlib.sha256(blobs).hexdigest()


def prepare_schedule(previous: Path, output: Path, runtime: Path, source: Path) -> JSON:
    """Retire only the legacy controller; preserve its sealed queue and live children."""
    for path in (previous, output, runtime):
        if not path.resolve().is_relative_to(SERVER_ROOT):
            raise ValueError("queue and scientific evidence stay on the server")
    if output.exists():
        raise FileExistsError("migration requires a new schedule root")
    old = read_json(previous / "queue.json")
    verify_seal(old)
    preflights = _preflight_entries(old, previous)
    if method_tree(source) != method_tree(Path(old["source"])):
        raise ValueError("preflight reuse requires identical native source/config/probe trees")
    current = read_json(previous / "state.json")
    if current["status"] != "running" or current["phase"] != "formal":
        raise ValueError("migration only adopts the live formal queue")
    controller = process_identity(current["pid"])
    if controller is None:
        raise ValueError("legacy controller missing; reconcile explicitly")
    command = process_command(controller["pid"]).split(b"\0")
    if str(previous).encode() not in command or not any(
        c.endswith(b"/scripts/v2/run_unified_group.py") for c in command
    ):
        raise ValueError("controller command is not the exact legacy queue")
    # Freeze dispatch only; no process-group signals and no child signals.
    os.kill(controller["pid"], signal.SIGSTOP)
    retired = False
    try:
        if process_identity(controller["pid"]) != controller:
            raise ValueError("controller changed before retirement")
        current = read_json(previous / "state.json")
        rows = []
        for row in old["rows"]:
            original = row["plan"]
            root = Path(original["run_root"])
            lane = current["lanes"][row["lane"]]
            item = {"name": row["name"], "lane": row["lane"], "parent_plan": original}
            if root.exists():
                if read_json(root / "launch.json") != original:
                    raise ValueError("existing run differs from legacy launch")
                if (root / "COMPLETE.json").exists():
                    validate_completed(original)
                    item.update(plan=original, mode="complete")
                elif lane["row"] == row["name"] and lane.get("child_pid"):
                    child = process_identity(lane["child_pid"])
                    if child is None:
                        raise ValueError("adopted child missing without scientific completion")
                    item.update(plan=original, mode="adopt_inline", process=child)
                else:
                    raise ValueError("started row has no matching active child or completion")
            else:
                if Path(original["test_root"]).exists():
                    raise ValueError("unused row has unexpected evaluation artifacts")
                config = source / Path(original["config"]).relative_to(old["source"])
                plan = resolve_plan(
                    argparse.Namespace(
                        config=config,
                        runtime=runtime,
                        data_root=None,
                        gpu=row["lane"],
                        seed=1,
                        defer_test=True,
                    )
                )
                if plan["resolved_config"] != original["resolved_config"] or (
                    plan["data_root"],
                    plan["genept_sha256"],
                ) != (original["data_root"], original["genept_sha256"]):
                    raise ValueError("migration changed scientific configuration/data/prior")
                item.update(plan=plan, mode="deferred")
            item["preflight"] = preflights[row["name"]]
            rows.append(item)
        schedule = {
            "schema": SCHEMA,
            "previous": str(previous),
            "previous_queue_sha256": sha256_file(previous / "queue.json"),
            "preflight_seal_sha256": sha256_file(previous / "PREFLIGHT_COMPLETE.json"),
            "method_tree_sha256": method_tree(source),
            "controller_source": str(source),
            "controller_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "retired_controller": controller,
            "rows": rows,
            "fit_fraction": FIT_FRACTION,
            "evaluation_fraction": EVAL_FRACTION,
            "max_processes_per_gpu": 2,
            "created_unix": time.time(),
        }
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "schedule.json", schedule)
        for row in rows:
            atomic_json(output / f"{row['name']}.launch-plan.json", row["plan"])
        atomic_json(
            previous / "SCHEDULING_SUPERSEDED.json",
            {
                "new_schedule": str(output),
                "new_schedule_sha256": sha256_file(output / "schedule.json"),
                "retired_controller": controller,
                "preserved_children": [r["process"] for r in rows if r["mode"] == "adopt_inline"],
                "reason": "User requested concurrent postfit; old queue remains immutable.",
            },
        )
        os.kill(controller["pid"], signal.SIGTERM)
        os.kill(controller["pid"], signal.SIGCONT)
        retired = True
        for _ in range(30):
            if not process_alive(controller):
                break
            time.sleep(0.1)
        if process_alive(controller):
            raise RuntimeError("legacy controller did not retire; do not start a second dispatcher")
        atomic_json(
            output / "MIGRATION.json",
            {
                "controller_retired": True,
                "retired_controller": controller,
                "preserved_children": [r["process"] for r in rows if r["mode"] == "adopt_inline"],
                "schedule_sha256": sha256_file(output / "schedule.json"),
            },
        )
        return schedule
    finally:
        if not retired and process_alive(controller):
            os.kill(controller["pid"], signal.SIGCONT)


def verify_schedule(schedule: JSON, directory: Path) -> None:
    if read_json(directory / "schedule.json") != schedule:
        raise ValueError("schedule differs from its immutable file")
    if schedule["schema"] != SCHEMA or [r["name"] for r in schedule["rows"]] != list(ARMS):
        raise ValueError("unknown first-batch overlap contract")
    if schedule["fit_fraction"] != FIT_FRACTION or schedule["evaluation_fraction"] != EVAL_FRACTION:
        raise ValueError("allocation budgets changed")
    previous = Path(schedule["previous"])
    if sha256_file(previous / "queue.json") != schedule["previous_queue_sha256"] or (
        sha256_file(previous / "PREFLIGHT_COMPLETE.json") != schedule["preflight_seal_sha256"]
    ):
        raise ValueError("legacy scientific/preflight seal changed")
    old = read_json(previous / "queue.json")
    verify_seal(old)
    _preflight_entries(old, previous)
    source = Path(schedule["controller_source"])
    if method_tree(source) != schedule["method_tree_sha256"] or (
        method_tree(Path(old["source"])) != schedule["method_tree_sha256"]
    ):
        raise ValueError("model/evaluation/probe trees changed after migration")
    runtime = read_json(schedule["runtime"])
    if sha256_file(Path(schedule["runtime"])) != schedule["runtime_sha256"]:
        raise ValueError("controller runtime changed")
    identity = inspect_source_identity(
        source,
        formal=True,
        expected_repository=old["rows"][0]["plan"]["resolved_config"]["source_code"]["repository"],
        publication_receipt=runtime["publication_receipt"],
        expected_publication_receipt_sha256=runtime["publication_sha256"],
    )
    if identity.commit != schedule["controller_commit"] or identity.dirty:
        raise ValueError("controller is not the clean published source")
    if process_alive(schedule["retired_controller"]):
        raise ValueError("legacy dispatcher still active")
    for row, prior in zip(schedule["rows"], old["rows"], strict=True):
        plan = row["plan"]
        if row["parent_plan"] != prior["plan"] or row["lane"] != prior["lane"]:
            raise ValueError("parent/lane identity changed")
        if read_json(directory / f"{row['name']}.launch-plan.json") != plan or (
            sha256_file(Path(plan["config"])) != plan["config_sha256"]
        ):
            raise ValueError("saved launch/config changed")
        if row["mode"] == "deferred":
            if plan["source_commit"] != identity.commit or plan.get("postfit_policy") != "deferred":
                raise ValueError("fresh fit must use published source and independent postfit")
            if plan["resolved_config"] != prior["plan"]["resolved_config"]:
                raise ValueError("model or training configuration changed")
        elif row["mode"] not in ("adopt_inline", "complete") or plan != prior["plan"]:
            raise ValueError("adopted launch changed")


def stage_command(directory: Path, row: JSON, stage: str, attempt: int = 1) -> list[str]:
    if stage == "fit":
        return [
            sys.executable,
            "-c",
            (
                "import json,sys,torch;"
                "torch.cuda.set_per_process_memory_fraction(float(sys.argv[2]),0);"
                "from gradpert.execution.train_entry import execute_plan;"
                "execute_plan(json.load(open(sys.argv[1])))"
            ),
            str(directory / f"{row['name']}.launch-plan.json"),
            str(FIT_FRACTION),
        ]
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--evaluate",
        str(directory / f"{row['name']}.launch-plan.json"),
        "--attempt",
        str(attempt),
        "--memory-fraction",
        str(EVAL_FRACTION if attempt == 1 else 1.0),
    ]


def evaluate_attempt(
    plan: JSON, attempt: int, fraction: float, *, gpu: str | None = None, full_first: bool = False
) -> None:
    """Same frozen inference, unique attempt roots; OOM cannot overwrite a receipt."""
    from gradpert.evaluation.state import prepare_evaluation_state
    from gradpert.execution.v2_checkpoint_eval import (
        execute_evaluation_plan,
        resolve_evaluation_plan,
    )
    from gradpert.execution.v2_deferred_postfit import finalize_deferred_run

    training = validate_training_stage(plan)
    root = Path(plan["run_root"])
    config = plan["resolved_config"]
    output = root / f"postfit-last-attempt{attempt}"
    if attempt not in (1, 2) or output.exists():
        raise ValueError("postfit attempt is new, bounded and never overwritten")
    if fraction != (EVAL_FRACTION if attempt == 1 and not full_first else 1.0):
        raise ValueError("postfit attempt allocation budget changed")
    try:
        prepare_evaluation_state(
            dataset_id=config["dataset_id"],
            protocol_id=config["data"]["protocol_id"],
            data_root=Path(plan["data_root"]),
            evaluation_protocol="v2",
        )
        selected = training["last"]
        request = argparse.Namespace(
            config=Path(plan["config"]),
            runtime=Path(plan["runtime"]),
            training_run_root=root,
            checkpoint=root / "fit" / selected["file"],
            checkpoint_sha256=selected["sha256"],
            checkpoint_role="last",
            output_root=output,
            gpu=plan["gpu"] if gpu is None else gpu,
            split="test",
        )
        evaluation = resolve_evaluation_plan(request)
        evaluation.update(cpu_training_state=True, cuda_memory_fraction=fraction)
        result = execute_evaluation_plan(evaluation)
        atomic_json(
            root / "fit/last-test.json",
            {
                "identity": {
                    "training": training["identity"],
                    "checkpoint": selected,
                    "role": "last",
                    "evaluation": {
                        **training["identity"],
                        "source": evaluation["evaluation_source"],
                        "environment": result["evaluation_environment"],
                    },
                },
                "result": result["result"],
                "independent_evaluation_receipt": str(output / "COMPLETE.json"),
                "independent_evaluation_sha256": sha256_file(output / "COMPLETE.json"),
                "worker_runtime_measurements": result["worker_runtime_measurements"],
            },
        )
        finalize_deferred_run(plan, training)
    except BaseException as error:
        atomic_json(
            root / f"POSTFIT_ATTEMPT{attempt}_FAILURE.json",
            {
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": str(error),
                "oom": "out of memory" in str(error).lower(),
                "scientific_complete": False,
            },
        )
        raise


def eligible_stages(
    lane: JSON, pending: list[JSON], evaluations: list[JSON]
) -> list[tuple[str, JSON]]:
    """One fit and one evaluator at most; a failed capped eval gets one idle-only retry."""
    stages: list[tuple[str, JSON]] = []
    if lane.get("adopted"):
        return stages
    if not lane.get("eval") and evaluations:
        first = evaluations[0]
        if first.get("attempt", 1) == 1 or not lane.get("fit"):
            stages.append(("eval", first))
    serialized = lane.get("serial_retry") or any(r.get("attempt", 1) == 2 for r in evaluations)
    if not lane.get("fit") and pending and not serialized:
        stages.append(("fit", pending[0]))
    return stages


def gpu_processes_owned(gpu: str, lane: JSON) -> bool:
    """An exclusive project lease does not authorize sharing with another user's process."""
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    uuids = {
        line.split(",")[0].strip(): line.split(",")[1].strip() for line in devices.splitlines()
    }
    if gpu not in uuids:
        raise ValueError("physical GPU disappeared")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    allowed = {active["pid"] for role in ("fit", "eval") if (active := lane[role])}
    for line in processes.splitlines():
        uuid, pid = (part.strip() for part in line.split(","))
        if uuid == uuids[gpu] and int(pid) not in allowed:
            return False
    return True


def execute_schedule(schedule: JSON, directory: Path, *, tick_seconds: float = 10) -> None:
    if not directory.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("execution stays on the server")
    if os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR:
        raise ValueError("allocator contract missing")
    if (directory / "state.json").exists():
        raise ValueError("controller cannot implicitly resume or duplicate an existing schedule")
    verify_schedule(schedule, directory)
    state: JSON = {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "status": "running",
        "phase": "adopting",
        "schedule_sha256": sha256_file(directory / "schedule.json"),
        "lanes": {},
        "completed": {},
        "failures": [],
        "events": [],
        "updated_unix": time.time(),
    }
    lanes: dict[str, JSON] = {}
    for gpu in ("0", "1"):
        rows = [r for r in schedule["rows"] if r["lane"] == gpu]
        adopted = [r for r in rows if r["mode"] == "adopt_inline"]
        if len(adopted) > 1:
            raise ValueError("only one adopted inline child per GPU")
        lanes[gpu] = {
            "adopted": adopted[0] if adopted else None,
            "pending": [r for r in rows if r["mode"] == "deferred"],
            "evaluations": [],
            "fit": None,
            "eval": None,
            "serial_retry": False,
            "gpu_lease": None,
        }
        for row in rows:
            if row["mode"] == "complete":
                state["completed"][row["name"]] = validate_completed(row["plan"])

    def record() -> None:
        state["updated_unix"] = time.time()
        state["lanes"] = {
            gpu: {
                "adopted": lane["adopted"],
                "pending": [r["name"] for r in lane["pending"]],
                "evaluation_pending": [r["name"] for r in lane["evaluations"]],
                "fit": {k: v for k, v in (lane["fit"] or {}).items() if k != "process"},
                "eval": {k: v for k, v in (lane["eval"] or {}).items() if k != "process"},
                "serial_retry": lane["serial_retry"],
            }
            for gpu, lane in lanes.items()
        }
        atomic_json(directory / "state.json", state)

    def start(gpu: str, lane: JSON, row: JSON, stage: str, stack: contextlib.ExitStack) -> None:
        verify_schedule(schedule, directory)
        plan, attempt = row["plan"], row.get("attempt", 1)
        if stage == "fit" and (Path(plan["run_root"]).exists() or Path(plan["test_root"]).exists()):
            raise ValueError("fresh fit cannot replace an existing run")
        tag = f"{row['name']}-{stage}-attempt{attempt}"
        log = stack.enter_context((directory / f"{tag}.log").open("x"))
        row_lease = (
            stack.enter_context(
                lock(
                    LEASE_ROOT
                    / (
                        f"v2-row-{plan['config_sha256']}-{plan['source_commit']}-{plan['seed']}.lock"
                    )
                )
            )
            if stage == "fit"
            else None
        )
        child = subprocess.Popen(
            stage_command(directory, row, stage, attempt),
            cwd=plan["repository_root"],
            env=child_environment(plan["repository_root"], gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(lane["gpu_lease"],) + ((row_lease,) if row_lease is not None else ()),
        )
        lane[stage] = {
            "name": row["name"],
            "row": row,
            "pid": child.pid,
            "process": child,
            "started_unix": time.time(),
            "log": str(directory / f"{tag}.log"),
        }
        lane["serial_retry"] = stage == "eval" and attempt == 2
        state["events"].append(
            {
                "event": "start",
                "stage": stage,
                "row": row["name"],
                "gpu": gpu,
                "attempt": attempt,
                "unix": time.time(),
            }
        )

    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(directory / "controller.lock"))
        record()
        while len(state["completed"]) < len(ARMS):
            for gpu, lane in lanes.items():
                try:
                    if lane["adopted"]:
                        row = lane["adopted"]
                        if not process_alive(row["process"]):
                            lane["adopted"] = None
                            state["completed"][row["name"]] = validate_completed(row["plan"])
                    if lane["adopted"]:
                        continue
                    for stage in ("fit", "eval"):
                        active = lane[stage]
                        if not active or active["process"].poll() is None:
                            continue
                        code = active["process"].returncode
                        row = active["row"]
                        atomic_json(
                            directory
                            / f"{row['name']}-{stage}-attempt{row.get('attempt', 1)}.exit.json",
                            {
                                "exit_code": code,
                                "started_unix": active["started_unix"],
                                "finished_unix": time.time(),
                                "pid": active["pid"],
                                "stage": stage,
                                "run_id": row["plan"]["run_id"],
                            },
                        )
                        lane[stage] = None
                        if code:
                            failed = Path(row["plan"]["run_root"]) / "POSTFIT_ATTEMPT1_FAILURE.json"
                            if (
                                stage == "eval"
                                and row.get("attempt", 1) == 1
                                and failed.exists()
                                and read_json(failed)["oom"]
                            ):
                                lane["evaluations"].insert(0, {**row, "attempt": 2})
                                lane["serial_retry"] = True
                                state["events"].append(
                                    {
                                        "event": "oom_idle_retry_queued",
                                        "row": row["name"],
                                        "gpu": gpu,
                                    }
                                )
                                continue
                            raise RuntimeError(
                                f"{row['name']} {stage} exited {code}; explicit repair required"
                            )
                        if stage == "fit":
                            validate_training_stage(row["plan"])
                            lane["evaluations"].append(row)
                        else:
                            state["completed"][row["name"]] = validate_completed(row["plan"])
                            lane["serial_retry"] = False
                    if state["failures"]:
                        continue
                    if lane["gpu_lease"] is None:
                        usage = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=index,memory.used",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        )
                        if not gpu_idle(usage, gpu):
                            continue
                        lane["gpu_lease"] = stack.enter_context(
                            lock(LEASE_ROOT / f"v2-gpu-{gpu}.lock")
                        )
                        if not gpu_idle(
                            subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=index,memory.used",
                                    "--format=csv,noheader,nounits",
                                ],
                                text=True,
                            ),
                            gpu,
                        ):
                            raise RuntimeError(
                                "GPU changed during admission; preserve unrelated work"
                            )
                    for stage, row in eligible_stages(lane, lane["pending"], lane["evaluations"]):
                        if not gpu_processes_owned(gpu, lane):
                            break
                        start(gpu, lane, row, stage, stack)
                        lane["pending" if stage == "fit" else "evaluations"].pop(0)
                except BaseException as error:
                    state["failures"].append(
                        {"gpu": gpu, "error_type": type(error).__name__, "error": str(error)}
                    )
            state["phase"] = (
                "draining_active_peers" if state["failures"] else "fit_postfit_pipeline"
            )
            if state["failures"]:
                state["status"] = "failed_draining"
            record()
            if state["failures"]:
                atomic_json(directory / "FAILURE.json", state)
            if state["failures"] and not any(
                lane["adopted"] or lane["fit"] or lane["eval"] for lane in lanes.values()
            ):
                state["status"] = "failed"
                atomic_json(directory / "FAILURE.json", state)
                raise RuntimeError("pipeline failed; active peers preserved, no new launches")
            time.sleep(tick_seconds)
        state.update(status="complete", phase="complete")
        record()
        atomic_json(directory / "COMPLETE.json", state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=EVAL_FRACTION)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR:
        parser.error("allocator contract missing")
    if args.evaluate:
        evaluate_attempt(read_json(args.evaluate), args.attempt, args.memory_fraction)
    elif args.schedule:
        execute_schedule(read_json(args.schedule), args.schedule.parent)
    else:
        if not all((args.previous, args.output, args.runtime)):
            parser.error("migration requires previous/output/runtime")
        prepare_schedule(
            args.previous, args.output, args.runtime, Path(__file__).resolve().parents[2]
        )


if __name__ == "__main__":
    main()
