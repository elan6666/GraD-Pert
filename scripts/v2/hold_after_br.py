"""Independently guard BR while retiring CPU dispatchers before later arms."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import time
from pathlib import Path

from run_group import lock
from run_unified_ddp import verify as verify_queue
from shared_gpu import admit, descendant, parent_map, process_info, snapshot, terminate_tree

from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file

SCHEMA = "gradpert-v2-post-br-holder-1"
SERVER_ROOT = Path("/data/yilangliu")
POLL_SECONDS = 5
MAX_SECONDS = 10 * 3600


def sealed(entry: dict) -> dict:
    path = Path(entry["path"])
    if not path.is_absolute() or not path.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("holder evidence must remain on the server")
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("holder sealed input changed")
    return read_json(path)


def live(identity: dict) -> dict | None:
    if type(identity.get("pid")) is not int or identity["pid"] <= 0:
        raise ValueError("invalid process identity")
    actual = process_info(identity["pid"])
    if actual is not None and (
        actual["uid"] != os.getuid()
        or any(
            actual.get(key) != identity.get(key) for key in ("pid", "start_ticks", "command_sha256")
        )
    ):
        raise ValueError("process identity changed; preserve the unrelated process")
    return actual


def direct(identity: dict, sig: int) -> None:
    if live(identity) is not None:
        os.kill(identity["pid"], sig)


def process_state(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


def pause(identity: dict) -> None:
    if live(identity) is None:
        raise ValueError("CPU dispatcher disappeared before pause")
    direct(identity, signal.SIGSTOP)
    deadline = time.monotonic() + 1
    while True:
        if live(identity) is None:
            raise ValueError("CPU dispatcher disappeared during pause")
        if process_state(identity["pid"]) in ("T", "t"):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("CPU dispatcher did not stop")
        time.sleep(0.01)


def retire(identity: dict) -> None:
    """A stopped parent must never CONT merely to process pending termination."""
    direct(identity, signal.SIGTERM)
    if live(identity) is not None:
        direct(identity, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while live(identity) is not None:
        if time.monotonic() >= deadline:
            raise RuntimeError("CPU dispatcher did not retire")
        time.sleep(0.05)


def session_members(session: int) -> list[int]:
    members = []
    for path in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(path.name)
            info = process_info(pid)
            if info is not None and info["uid"] == os.getuid() and os.getsid(pid) == session:
                members.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def fit_exit_code(plan: dict) -> int:
    """A paused parent cannot reap its zombie; require actual exit-status evidence."""
    stat = Path(f"/proc/{plan['fit']['pid']}/stat")
    if stat.exists():
        fields = stat.read_text().rsplit(")", 1)[1].split()
        if fields[0] != "Z" or fields[19] != plan["fit"]["start_ticks"]:
            raise ValueError("BR has no exact zombie exit-status identity")
        return os.waitstatus_to_exitcode(int(fields[49]))
    receipt = read_json(Path(plan["queue"]["path"]).parent / "BR-fit-attempt1.exit.json")
    if receipt["identity"] != plan["fit"] or receipt["name"] != "BR" or receipt["stage"] != "fit":
        raise ValueError("BR exit receipt differs from the sealed process")
    return receipt["exit_code"]


def validate_setup(plan: dict) -> tuple[dict, dict, dict, dict]:
    if plan["schema"] != SCHEMA or not 0 < plan.get("max_seconds", MAX_SECONDS) <= MAX_SECONDS:
        raise ValueError("unknown or unbounded holder plan")
    source = plan["holder_source"]
    publication = sealed(source["publication"])
    identity = inspect_source_identity(
        source["repository_root"],
        formal=True,
        expected_repository=publication["repository"],
        publication_receipt=source["publication"]["path"],
        expected_publication_receipt_sha256=source["publication"]["sha256"],
    )
    if (
        identity.commit != source["commit"]
        or identity.dirty
        or (Path(source["repository_root"]).resolve() != Path(__file__).resolve().parents[2])
    ):
        raise ValueError("holder source must be its clean published checkout")
    queue = sealed(plan["queue"])
    verify_queue(queue)
    if [row["name"] for row in queue["rows"]] != ["BR", "F1", "N1"]:
        raise ValueError("holder requires exactly BR/F1/N1")
    root = Path(plan["queue"]["path"]).parent
    state = read_json(root / "state.json")
    if (
        state.get("status") != "running"
        or set(state.get("active", {})) != {"fit"}
        or (
            state["active"]["fit"].get("name") != "BR"
            or state["active"]["fit"].get("identity") != plan["fit"]
            or state.get("pid") != plan["controller"]["pid"]
            or state.get("queue_sha256") != plan["queue"]["sha256"]
            or state.get("completed")
            or [
                e["name"]
                for e in state.get("events", [])
                if e.get("event") == "start" and e.get("stage") == "fit"
            ]
            != ["BR"]
        )
    ):
        raise ValueError("controller moved beyond BR-only fit; do not pause it")
    br_path = root / "BR.launch-plan.json"
    launch = read_json(br_path)
    if launch["run_id"] != queue["rows"][0]["plan"]["run_id"] or (
        launch["postfit_policy"] != "deferred"
        or launch["resolved_config"]["training"]["max_epochs"]["value"] != 4
    ):
        raise ValueError("BR launch is not the unchanged deferred four-epoch plan")
    gm_queue = sealed(plan["gm1"]["queue"])
    verify_queue(gm_queue)
    gm_state = read_json(Path(plan["gm1"]["queue"]["path"]).parent / "state.json")
    if (
        gm_state.get("status") != "running"
        or gm_state.get("phase") != "waiting_dependency"
        or (
            gm_state.get("active")
            or gm_state.get("preflights")
            or gm_state.get("pid") != plan["gm1"]["controller"]["pid"]
            or gm_state.get("queue_sha256") != plan["gm1"]["queue"]["sha256"]
            or [row["name"] for row in gm_queue["rows"]] != ["GM1"]
            or gm_queue["dependency"]["queue"] != plan["queue"]["path"]
            or gm_queue["dependency"]["queue_sha256"] != plan["queue"]["sha256"]
        )
    ):
        raise ValueError("GM1 is not the exact CPU dependency waiter")
    for process in (plan["controller"], plan["fit"], plan["gm1"]["controller"]):
        if live(process) is None:
            raise ValueError("sealed live process missing")
    if not descendant(plan["fit"]["pid"], plan["controller"]["pid"], parent_map()):
        raise ValueError("BR is not a descendant covered by the exact controller guard")
    if os.getpgid(plan["fit"]["pid"]) == os.getpgid(plan["controller"]["pid"]) or (
        os.getsid(plan["fit"]["pid"]) == os.getsid(plan["controller"]["pid"])
    ):
        raise ValueError("BR must run in an independent process group and session")
    return queue, launch, state, gm_state


def execute(plan: dict) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("holder must run with CUDA_VISIBLE_DEVICES='' before imports")
    output = Path(plan["output"])
    if (
        not output.is_absolute()
        or not output.resolve().is_relative_to(SERVER_ROOT)
        or output.exists()
    ):
        raise ValueError("holder output must be a fresh server root")
    output.mkdir()
    atomic_json(output / "plan.json", plan)
    state = {
        "schema": SCHEMA,
        "status": "pending",
        "pid": os.getpid(),
        "signals": [],
        "scientific_complete": False,
    }
    stopped, resource_failure = [], False
    queue = None
    setup_validated = False
    br_descendants = []

    def record():
        atomic_json(output / "state.json", state)

    def send(identity, sig):
        direct(identity, sig)
        state["signals"].append({"identity": identity, "signal": int(sig)})
        record()

    def guard():
        nonlocal resource_failure
        try:
            observed = snapshot(plan["controller"]["pid"])
            with (output / "telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(observed) + "\n")
            admit(queue["shared_gpu_policy"], observed)
        except KeyboardInterrupt:
            raise
        except BaseException:
            resource_failure = True
            raise

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt("holder interrupted")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    record()
    with lock(output / "holder.lock"):
        for sig in previous:
            signal.signal(sig, interrupted)
        try:
            queue, launch, frozen, gm_frozen = validate_setup(plan)
            setup_validated = True
            parents = parent_map()
            for pid in parents:
                if pid != plan["fit"]["pid"] and descendant(pid, plan["fit"]["pid"], parents):
                    info = process_info(pid)
                    if info is not None and info["uid"] == os.getuid():
                        br_descendants.append(
                            {key: info[key] for key in ("pid", "start_ticks", "command_sha256")}
                        )
            state["br_descendants"] = br_descendants
            record()
            guard()
            root = Path(plan["queue"]["path"]).parent
            gm_root = Path(plan["gm1"]["queue"]["path"]).parent
            if (
                read_json(root / "state.json") != frozen
                or read_json(gm_root / "state.json") != gm_frozen
            ):
                raise ValueError("dispatcher state raced before pause")
            for identity in (plan["controller"], plan["gm1"]["controller"]):
                stopped.append(identity)
                pause(identity)
                state["signals"].append({"identity": identity, "signal": int(signal.SIGSTOP)})
                record()
            root = Path(plan["queue"]["path"]).parent
            gm_root = Path(plan["gm1"]["queue"]["path"]).parent
            if (
                read_json(root / "state.json") != frozen
                or read_json(gm_root / "state.json") != gm_frozen
            ):
                raise ValueError("dispatcher state raced during pause")
            state.update(status="running", br_training_complete=False)
            record()
            deadline = time.monotonic() + plan.get("max_seconds", MAX_SECONDS)
            while True:
                guard()
                sealed(plan["queue"])
                sealed(plan["gm1"]["queue"])
                if live(plan["fit"]) is None:
                    if any(live(identity) is not None for identity in br_descendants):
                        raise RuntimeError("BR launcher exited with live sealed descendants")
                    if fit_exit_code(plan) != 0:
                        raise RuntimeError("BR launcher did not exit successfully")
                    training = validate_training_stage(launch)
                    if (
                        training["epoch"] != 4
                        or training["last"]["epoch"] != 4
                        or (
                            sha256_file(Path(launch["run_root"]) / "fit/last.pt")
                            != training["last"]["sha256"]
                        )
                    ):
                        raise ValueError("BR terminal epoch/checkpoint hash differs")
                    for identity in stopped:
                        retire(identity)
                    stopped.clear()
                    state.update(
                        status="ready",
                        br_training_complete=True,
                        training=training,
                        evaluation_pending=True,
                        retired_cpu_dispatchers=True,
                    )
                    record()
                    atomic_json(output / "READY.json", state)
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded BR holder expired")
                time.sleep(POLL_SECONDS)
        except BaseException as error:
            if not setup_validated:
                state.update(
                    status="failed",
                    error_type=type(error).__name__,
                    error=str(error),
                    cleanup_errors=[],
                )
                record()
                atomic_json(output / "FAILURE.json", state)
                raise
            cleanup = []
            try:
                raced = read_json(Path(plan["queue"]["path"]).parent / "state.json")
                other_active = any(
                    item.get("name") != "BR" or stage != "fit"
                    for stage, item in raced.get("active", {}).items()
                    if isinstance(item, dict)
                )
                remaining = [identity for identity in br_descendants if live(identity) is not None]
                if stopped and other_active:
                    state["handoff_raced_active_children"] = raced["active"]
                    if resource_failure or live(plan["fit"]) is None:
                        if resource_failure and live(plan["fit"]) is not None:
                            terminate_tree(plan["fit"]["pid"], plan["fit"])
                        for identity in remaining:
                            if live(identity) is not None:
                                terminate_tree(identity["pid"], identity)
                    for identity in stopped:
                        send(identity, signal.SIGCONT)
                elif resource_failure or (live(plan["fit"]) is None and remaining):
                    if live(plan["fit"]) is not None:
                        terminate_tree(plan["fit"]["pid"], plan["fit"])
                    for identity in remaining:
                        if live(identity) is not None:
                            terminate_tree(identity["pid"], identity)
                    for identity in stopped:
                        retire(identity)
                elif stopped:
                    if live(plan["fit"]) is not None:
                        guard()
                        for identity in stopped:
                            send(identity, signal.SIGCONT)
                    else:
                        for identity in stopped:
                            retire(identity)
            except BaseException as cleanup_error:
                cleanup.append(str(cleanup_error))
                # If returning the guard is unsafe, never leave an unguarded STOP parent.
                with contextlib.suppress(BaseException):
                    if live(plan["fit"]) is not None:
                        terminate_tree(plan["fit"]["pid"], plan["fit"])
                for identity in br_descendants:
                    with contextlib.suppress(BaseException):
                        if live(identity) is not None:
                            terminate_tree(identity["pid"], identity)
                for identity in stopped:
                    with contextlib.suppress(BaseException):
                        retire(identity)
            state.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                cleanup_errors=cleanup,
            )
            record()
            atomic_json(output / "FAILURE.json", state)
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = read_json(args.plan)
    if args.execute:
        execute(plan)
    else:
        validate_setup(plan)
        print(json.dumps({"schema": SCHEMA, "status": "validated", "scientific_complete": False}))


if __name__ == "__main__":
    main()
