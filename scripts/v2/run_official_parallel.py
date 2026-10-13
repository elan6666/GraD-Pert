"""Adopt an active scDFM worker and dispatch the sealed GEARS row on GPU1.

Only scheduling changes. Both scientific runners keep the original clean source,
config, runtime, run IDs and evaluation protocol. The original dependency queue
receives a validated completion forwarder, so its native dependent is unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from run_group import lock
from shared_gpu import admit, descendant, parent_map, process_info, snapshot

from benchmarks.expression_holdout.config import (
    load_config,
    validate_cpu_gate,
    validate_preflight,
    validate_terminal,
)
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SERVER = Path("/data/yilangliu")
SCHEMA = "official-two-gpu-adoption-1"
REPOSITORY = "https://github.com/elan6666/GraD-Pert.git"
IDENTITY_FIELDS = ("pid", "uid", "start_ticks", "command_sha256")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def same_process(expected: dict, actual: dict | None) -> bool:
    # Adoption reparents the worker; ppid is deliberately not its identity.
    return actual is not None and all(actual[k] == expected[k] for k in IDENTITY_FIELDS)


def scientific_tree(source: Path) -> dict:
    return {
        str(path.relative_to(source)): sha256_file(path)
        for directory in ("src", "benchmarks")
        for path in (source / directory).rglob("*.py")
    }


def verify_parent(previous: Path, seal: str) -> dict:
    if sha256_file(previous / "queue.json") != seal:
        raise ValueError("original official queue seal changed")
    queue = read(previous / "queue.json")
    if queue["schema"] != "official-expression-holdout-queue-v1" or [
        r["name"] for r in queue["rows"]
    ] != ["scdfm", "gears"]:
        raise ValueError("only the exact scDFM/GEARS pair can be adopted")
    for row in queue["rows"]:
        for name in ("config", "runtime"):
            if sha256_file(Path(row[name])) != row[f"{name}_sha256"]:
                raise ValueError("original scientific inputs changed")
    source = inspect_source_identity(queue["source"], formal=False, expected_repository=REPOSITORY)
    if source.dirty or source.commit != queue["source_commit"]:
        raise ValueError("original scientific source changed")
    if scientific_tree(ROOT) != scientific_tree(Path(queue["source"])):
        raise ValueError("parallel scheduling must not change scientific code")
    return queue


def owned_snapshot(worker: dict) -> dict:
    observed = snapshot(os.getpid())
    current = process_info(worker["pid"])
    if current is not None and not same_process(worker, current):
        raise ValueError("adopted worker PID was reused or its identity changed")
    parents = parent_map()
    for entry in observed["processes"]:
        if current is not None and descendant(entry["pid"], worker["pid"], parents):
            if entry["identity"] is None or entry["identity"]["uid"] != worker["uid"]:
                raise ValueError("adopted GPU process identity unavailable")
            entry["owned"] = True
    return observed


def validate_idle_gpu1(policy: dict, observed: dict, worker: dict) -> None:
    uuids = {d["gpu_index"]: d["gpu_uuid"] for d in policy["devices"]}
    if any(p["gpu_uuid"] == uuids[1] for p in observed["processes"]):
        raise RuntimeError("GPU1 is occupied; parallel GEARS is not dispatched")
    for p in observed["processes"]:
        if p["pid"] == worker["pid"] and same_process(worker, p["identity"]):
            if p["gpu_uuid"] != uuids[0]:
                raise ValueError("adopted scDFM is not on GPU0")
            p["owned"] = True
    admit(policy, observed)


def migrate(previous: Path, seal: str, output: Path, publication: Path, pub_sha: str) -> None:
    previous, output = previous.resolve(), output.resolve()
    if output.exists() or not all(p.is_relative_to(SERVER) for p in (previous, output)):
        raise ValueError("new server-only adoption root required")
    identity = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository=REPOSITORY,
        publication_receipt=publication,
        expected_publication_receipt_sha256=pub_sha,
    )
    queue = verify_parent(previous, seal)
    current = read(previous / "state.json")
    if current["status"] != "running" or current["phase"] != "scdfm_full":
        raise ValueError("only a live scDFM full stage may be adopted")
    active = current["active"]
    if active["name"] != "scdfm" or active["mode"] != "full":
        raise ValueError("wrong active worker")
    controller = process_info(current["pid"])
    worker = process_info(active["pid"])
    if controller is None or worker is None or controller["uid"] != os.getuid():
        raise ValueError("exact owned dispatcher and worker must be live")
    if worker["uid"] != os.getuid() or worker["ppid"] != controller["pid"]:
        raise ValueError("worker is not the original owned child")
    command = Path(f"/proc/{controller['pid']}/cmdline").read_bytes().split(b"\0")
    if str(previous / "queue.json").encode() not in command or not any(
        x.endswith(b"run_official_expression_group.py") for x in command
    ):
        raise ValueError("wrong dispatcher command")
    # A custom SIGTERM handler could terminate its children; refuse that source.
    if (
        "signal.signal("
        in (Path(queue["source"]) / "scripts/v2/run_official_expression_group.py").read_text()
    ):
        raise ValueError("dispatcher signal behavior requires explicit review")
    if os.getpgid(worker["pid"]) != worker["pid"]:
        raise ValueError("worker must have its own session/process group")
    gears = queue["rows"][1]
    if any(Path(gears[k]).exists() for k in ("run_root", "preflight_root")):
        raise FileExistsError("GEARS already started; do not duplicate it")
    validate_cpu_gate(read(Path(gears["runtime"])), Path(gears["config"]), queue["source_commit"])
    validate_idle_gpu1(queue["shared_gpu_policy"], snapshot(os.getpid()), worker)
    # Freeze only dispatch. Never signal the worker or a process group.
    os.kill(controller["pid"], signal.SIGSTOP)
    retired = False
    try:
        if not same_process(controller, process_info(controller["pid"])):
            raise ValueError("dispatcher identity changed before retirement")
        if read(previous / "state.json") != current:
            raise ValueError("dispatcher advanced before it was frozen")
        if not same_process(worker, process_info(worker["pid"])):
            raise ValueError("worker changed before adoption")
        validate_idle_gpu1(queue["shared_gpu_policy"], snapshot(os.getpid()), worker)
        output.mkdir()
        schedule = {
            "schema": SCHEMA,
            "previous": str(previous),
            "previous_queue_sha256": seal,
            "controller_source": str(ROOT),
            "controller_commit": identity.commit,
            "publication": str(publication),
            "publication_sha256": pub_sha,
            "source_commit": queue["source_commit"],
            "retired_controller": controller,
            "adopted_worker": worker,
            "rows": queue["rows"],
            "gears_physical_gpu": "1",
            "scientific_inputs_unchanged": True,
            "shared_gpu_policy": queue["shared_gpu_policy"],
        }
        atomic_json(output / "schedule.json", schedule)
        atomic_json(
            previous / "SCHEDULING_SUPERSEDED.json",
            {
                "schedule": str(output),
                "schedule_sha256": sha256_file(output / "schedule.json"),
                "retired_controller": controller,
                "preserved_worker": worker,
            },
        )
        os.kill(controller["pid"], signal.SIGTERM)
        os.kill(controller["pid"], signal.SIGCONT)
        retired = True
        for _ in range(30):
            if process_info(controller["pid"]) is None:
                break
            time.sleep(0.1)
        if process_info(controller["pid"]) is not None:
            raise RuntimeError("dispatcher has not retired; do not start GEARS")
        if not same_process(worker, process_info(worker["pid"])):
            raise RuntimeError("adopted worker missing; reconcile before dispatch")
        atomic_json(
            output / "MIGRATION.json",
            {
                "schedule_sha256": sha256_file(output / "schedule.json"),
                "controller_retired": True,
                "worker_preserved": True,
            },
        )
    finally:
        if not retired and same_process(controller, process_info(controller["pid"])):
            os.kill(controller["pid"], signal.SIGCONT)


def terminal_entry(row: dict, source_commit: str) -> dict:
    root = Path(row["run_root"])
    receipt = validate_terminal(root, row, source_commit)
    return {
        "root": str(root),
        "receipt_sha256": sha256_file(root / "receipt.json"),
        "last_test_sha256": receipt["last_test_sha256"],
    }


def forward_completion(schedule: dict, directory: Path, completed: dict) -> dict:
    """Forward only two independently validated terminals into the old dependency."""
    previous = Path(schedule["previous"])
    queue = verify_parent(previous, schedule["previous_queue_sha256"])
    if set(completed) != {"scdfm", "gears"}:
        raise ValueError("both scientific terminals required before forwarding")
    expected = {row["name"]: terminal_entry(row, queue["source_commit"]) for row in queue["rows"]}
    if completed != expected:
        raise ValueError("scientific terminal receipt identities changed")
    if (previous / "COMPLETE.json").exists() or (previous / "FAILURE.json").exists():
        raise FileExistsError("original terminal already exists; never overwrite it")
    forward = {
        **read(previous / "state.json"),
        "status": "complete",
        "phase": "complete",
        "active": None,
        "completed": completed,
        "scientific_complete": True,
        "completion_controller_commit": schedule["controller_commit"],
        "completion_schedule": str(directory),
        "completion_schedule_sha256": sha256_file(directory / "schedule.json"),
    }
    atomic_json(previous / "state.json", forward)
    atomic_json(previous / "COMPLETE.json", forward)
    return forward


def execute(directory: Path) -> None:
    if not directory.resolve().is_relative_to(SERVER):
        raise ValueError("server execution only")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    schedule_path = directory / "schedule.json"
    schedule = read(schedule_path)
    seal = sha256_file(schedule_path)
    identity = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository=REPOSITORY,
        publication_receipt=schedule["publication"],
        expected_publication_receipt_sha256=schedule["publication_sha256"],
    )
    if identity.commit != schedule["controller_commit"] or schedule["schema"] != SCHEMA:
        raise ValueError("adoption controller source differs")
    previous = Path(schedule["previous"])
    queue = verify_parent(previous, schedule["previous_queue_sha256"])
    if (
        schedule["rows"] != queue["rows"]
        or schedule["shared_gpu_policy"] != queue["shared_gpu_policy"]
    ):
        raise ValueError("adoption scientific inputs or resource policy changed")
    if schedule["gears_physical_gpu"] != "1" or schedule["source_commit"] != queue["source_commit"]:
        raise ValueError("adoption placement/source differs")
    migration = read(directory / "MIGRATION.json")
    if migration["schedule_sha256"] != seal or migration["worker_preserved"] is not True:
        raise ValueError("migration receipt differs")
    if process_info(schedule["retired_controller"]["pid"]) is not None:
        raise ValueError("old dispatcher still exists; no duplicate dispatcher")
    if (directory / "state.json").exists():
        raise FileExistsError("do not restart an adoption attempt")
    scdfm, gears = schedule["rows"]
    state = {
        "schema": SCHEMA,
        "status": "running",
        "pid": os.getpid(),
        "schedule_sha256": seal,
        "source_commit": queue["source_commit"],
        "controller_commit": identity.commit,
        "phase": "parallel",
        "adopted_worker": schedule["adopted_worker"],
        "completed": {},
        "scientific_complete": False,
        "active": None,
    }
    child = None
    stage = "preflight"
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(lock(directory / "controller.lock"))
            stack.enter_context(lock(previous / "controller.lock"))
            for gpu in ("0", "1"):
                stack.enter_context(lock(SERVER / "GraD-Pert/runtime" / f"v2-gpu-{gpu}.lock"))
            while len(state["completed"]) < 2:
                if sha256_file(schedule_path) != seal:
                    raise ValueError("sealed adoption schedule changed")
                verify_parent(previous, schedule["previous_queue_sha256"])
                worker_live = same_process(
                    schedule["adopted_worker"], process_info(schedule["adopted_worker"]["pid"])
                )
                if not worker_live and "scdfm" not in state["completed"]:
                    state["completed"]["scdfm"] = terminal_entry(scdfm, queue["source_commit"])
                observed = owned_snapshot(schedule["adopted_worker"])
                admit(queue["shared_gpu_policy"], observed)
                if child is None and "gears" not in state["completed"]:
                    destination = Path(
                        gears["preflight_root" if stage == "preflight" else "run_root"]
                    )
                    if destination.exists():
                        raise FileExistsError("GEARS destination exists before dispatch")
                    runtime = read(Path(gears["runtime"]))
                    argv = [
                        runtime["python"],
                        "-m",
                        "benchmarks.expression_holdout.runner",
                        "--config",
                        gears["config"],
                        "--runtime",
                        gears["runtime"],
                        "--output",
                        str(destination),
                        "--mode",
                        stage,
                    ]
                    if stage == "full":
                        argv += ["--preflight", str(Path(gears["preflight_root"]) / "receipt.json")]
                    environment = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "1",
                        "PYTHONPATH": str(Path(queue["source"]) / "src")
                        + os.pathsep
                        + queue["source"],
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                    }
                    log = stack.enter_context((directory / f"gears-{stage}.log").open("x"))
                    child = subprocess.Popen(
                        argv,
                        cwd=queue["source"],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    state["active"] = {
                        "name": "gears",
                        "mode": stage,
                        "pid": child.pid,
                        "physical_gpu": "1",
                        "run_root": str(destination),
                    }
                if child is not None and child.poll() is not None:
                    if child.returncode:
                        raise RuntimeError(f"GEARS {stage} exited {child.returncode}")
                    if stage == "preflight":
                        config = load_config(Path(gears["config"]))
                        validate_preflight(
                            read(Path(gears["preflight_root"]) / "receipt.json"),
                            config=Path(gears["config"]),
                            source_commit=queue["source_commit"],
                            official_commit=config["source_code"]["commit"],
                            runtime_sha256=gears["runtime_sha256"],
                        )
                        stage = "full"
                    else:
                        state["completed"]["gears"] = terminal_entry(gears, queue["source_commit"])
                    child = None
                    state["active"] = None
                state["updated_unix"] = time.time()
                atomic_json(directory / "state.json", state)
                if len(state["completed"]) < 2:
                    time.sleep(10)
            forward_completion(schedule, directory, state["completed"])
            state.update(status="complete", phase="complete", scientific_complete=True)
            atomic_json(directory / "state.json", state)
            atomic_json(directory / "COMPLETE.json", state)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=30)
        # Never signal the adopted scDFM worker during a GEARS/controller failure.
        state.update(
            status="failed",
            phase="failed",
            error=str(error),
            adopted_worker_preserved=True,
            scientific_complete=False,
        )
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "FAILURE.json", state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate", type=Path)
    parser.add_argument("--previous-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--publication-sha256")
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if (
        args.migrate
        and all((args.previous_sha256, args.output, args.publication, args.publication_sha256))
        and not args.execute
    ):
        migrate(
            args.migrate,
            args.previous_sha256,
            args.output,
            args.publication,
            args.publication_sha256,
        )
    elif args.execute and not args.migrate:
        execute(args.execute)
    else:
        parser.error("choose fully specified --migrate or --execute")


if __name__ == "__main__":
    main()
