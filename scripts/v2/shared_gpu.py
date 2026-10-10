"""Explicit bounded admission for one low-memory foreign job on physical GPU1."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path

SCHEMA = "shared-gpu-admission-1"


def validate_policy(policy: dict) -> None:
    if (
        policy.get("schema") != SCHEMA
        or policy.get("gpu_index") != 1
        or not isinstance(policy.get("foreign_uid"), int)
        or policy["foreign_uid"] <= 0
        or policy.get("command_basename") != "run_scbutterfly_archived_predictions.py"
        or policy.get("max_foreign_memory_mib") != 1024
        or policy.get("min_free_memory_mib") != 2048
        or policy.get("max_own_memory_fraction") != 0.85
        or not isinstance(policy.get("gpu_uuid"), str)
        or not policy["gpu_uuid"].startswith("GPU-")
    ):
        raise ValueError("shared-GPU policy differs from its bounded authorization")


def validate_probe_binding(queue: dict, args, source_commit: str, kind: str, world: int) -> None:
    """The shared exception applies only to an exact queued ten-update preflight."""
    rows = [
        r for r in queue["rows"] if Path(r["plan"]["config"]).resolve() == args.config.resolve()
    ]
    if len(rows) != 1 or kind != "preflight_only" or args.steps != 10 or world != 2:
        raise ValueError("shared admission requires one sealed dual-GPU ten-update preflight")
    row = rows[0]
    plan = row["plan"]
    if (
        queue["schema"] != "unified-ddp-queue-1"
        or queue["source_commit"] != source_commit
        or args.gpu != "0,1"
        or hashlib.sha256(args.config.read_bytes()).hexdigest() != plan["config_sha256"]
        or args.data_root.resolve() != Path(plan["data_root"]).resolve()
        or args.publication.resolve() != Path(plan["publication"]).resolve()
        or args.publication_sha256 != plan["publication_sha256"]
        or args.output.resolve() != args.shared_queue.parent.resolve() / f"{row['name']}-preflight"
    ):
        raise ValueError("shared preflight differs from its sealed queue")


def process_info(pid: int) -> dict | None:
    root = Path(f"/proc/{pid}")
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        command = (root / "cmdline").read_bytes()
        uid = root.stat().st_uid
    except FileNotFoundError:
        return None
    if fields[0] == "Z" or not command:
        return None
    arguments = [s.decode() for s in command.split(b"\0") if s]
    return {
        "pid": pid,
        "uid": uid,
        "ppid": int(fields[1]),
        "start_ticks": fields[19],
        "command_sha256": hashlib.sha256(command).hexdigest(),
        "script_basename": next((Path(s).name for s in arguments if s.endswith(".py")), None),
    }


def parent_map() -> dict[int, int]:
    parents = {}
    for root in Path("/proc").glob("[0-9]*"):
        try:
            fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(root.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return parents


def descendant(pid: int, owner_pid: int, parents: dict[int, int]) -> bool:
    visited = set()
    while pid != owner_pid and pid in parents and pid not in visited:
        visited.add(pid)
        pid = parents[pid]
    return pid == owner_pid


def snapshot(owner_pid: int) -> dict:
    def query(flag):
        output = subprocess.check_output(
            ["nvidia-smi", flag, "--format=csv,noheader,nounits"], text=True, timeout=10
        )
        return list(csv.reader(output.splitlines(), skipinitialspace=True))

    gpus = {}
    for index, uuid, total, used, free in query(
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free"
    ):
        if int(index) in (0, 1):
            gpus[int(index)] = {
                "uuid": uuid.strip(),
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
            }
    parents = parent_map()
    wanted = {g["uuid"] for g in gpus.values()}
    processes = []
    for uuid, pid, memory in query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory"):
        if uuid.strip() not in wanted:
            continue
        number = int(pid)
        processes.append(
            {
                "gpu_uuid": uuid.strip(),
                "pid": number,
                "memory_mib": int(memory),
                "owned": descendant(number, owner_pid, parents),
                "identity": process_info(number),
            }
        )
    return {"observed_unix": time.time(), "gpus": gpus, "processes": processes}


def admit(policy: dict, observed: dict) -> None:
    """Fail closed on an unknown job, missing telemetry, or exhausted headroom."""
    validate_policy(policy)
    gpus = {int(k): v for k, v in observed["gpus"].items()}
    if set(gpus) != {0, 1} or gpus[1]["uuid"] != policy["gpu_uuid"]:
        raise RuntimeError("shared-GPU physical device identity changed")
    foreign_memory = 0
    foreign_pids = set()
    for entry in observed["processes"]:
        if entry["owned"]:
            continue
        info = entry["identity"]
        if (
            entry["gpu_uuid"] != policy["gpu_uuid"]
            or info is None
            or info["uid"] != policy["foreign_uid"]
            or info.get("script_basename") != policy["command_basename"]
            or entry["memory_mib"] < 0
        ):
            raise RuntimeError("unapproved foreign GPU process; preserve it and stop our dispatch")
        foreign_pids.add(entry["pid"])
        foreign_memory += entry["memory_mib"]
    if len(foreign_pids) > 1 or foreign_memory > policy["max_foreign_memory_mib"]:
        raise RuntimeError("foreign GPU workload exceeds the one-job memory allowance")
    if any(g["free_mib"] < policy["min_free_memory_mib"] for g in gpus.values()):
        raise RuntimeError("shared-GPU physical free-memory reserve exhausted")


def terminate_tree(pid: int, expected: dict) -> None:
    """Signal only identity-checked descendants owned by the current Unix user."""
    current = process_info(pid)
    if current is None:
        return
    if any(current[k] != expected[k] for k in ("pid", "start_ticks", "command_sha256")):
        raise RuntimeError("refuse to stop a recycled process ID")
    parents = parent_map()
    selected = []
    for number in parents:
        if descendant(number, pid, parents):
            info = process_info(number)
            if info is not None and info["uid"] == os.getuid():
                selected.append(info)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for info in reversed(selected):
            current = process_info(info["pid"])
            if current == info:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(info["pid"], sig)
        if sig == signal.SIGTERM:
            time.sleep(2)
