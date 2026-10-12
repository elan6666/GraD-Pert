"""Explicit bounded admission for recorded low-memory foreign GPU jobs."""

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
DUAL_SCHEMA = "shared-gpu-admission-2"
DYNAMIC_SCHEMA = "shared-gpu-admission-3"
SCBUTTERFLY_AUTHORIZATION = "2026-10-12-user-scbutterfly-coexistence"
TOTALVI_AUTHORIZATION = "2026-10-12-user-scbutterfly-totalvi-coexistence"
TOTALVI_PROGRAM = "run_totalvi_source_validation_development.py"


def validate_dynamic_policy(policy: dict) -> None:
    """The explicitly approved account/program may restart within the same limits."""
    fields = {
        "schema",
        "devices",
        "foreign_uid",
        "command_basename",
        "max_foreign_processes_per_gpu",
        "max_foreign_memory_mib",
        "min_free_memory_mib",
        "max_own_memory_fraction",
        "authorization_reference",
    }
    devices = policy.get("devices")
    expanded = "additional_command_basenames" in policy
    if expanded:
        fields.add("additional_command_basenames")
    if (
        set(policy) != fields
        or policy.get("schema") != DYNAMIC_SCHEMA
        or policy.get("foreign_uid") != 1003
        or policy.get("command_basename") != "run_scbutterfly_archived_predictions.py"
        or type(policy.get("max_foreign_processes_per_gpu")) is not int
        or policy["max_foreign_processes_per_gpu"] != 1
        or policy.get("max_foreign_memory_mib") != 1024
        or policy.get("min_free_memory_mib") != 2048
        or policy.get("max_own_memory_fraction") != 0.85
        or policy.get("authorization_reference")
        != (TOTALVI_AUTHORIZATION if expanded else SCBUTTERFLY_AUTHORIZATION)
        or (expanded and policy["additional_command_basenames"] != [TOTALVI_PROGRAM])
        or not isinstance(devices, list)
        or len(devices) != 2
    ):
        raise ValueError("dynamic shared-GPU policy differs from its scoped authorization")
    for index, device in enumerate(devices):
        if (
            not isinstance(device, dict)
            or set(device) != {"gpu_index", "gpu_uuid"}
            or type(device.get("gpu_index")) is not int
            or device["gpu_index"] != index
            or not isinstance(device.get("gpu_uuid"), str)
            or not device["gpu_uuid"].startswith("GPU-")
        ):
            raise ValueError("dynamic shared-GPU policy requires the two fixed devices")
    if devices[0]["gpu_uuid"] == devices[1]["gpu_uuid"]:
        raise ValueError("dynamic shared-GPU policy requires distinct physical devices")


def validate_dual_policy(policy: dict) -> None:
    devices = policy.get("devices", [])
    if (
        not isinstance(devices, list)
        or len(devices) != 2
        or [d.get("gpu_index") for d in devices] != [0, 1]
        or policy.get("max_foreign_memory_mib") != 1024
        or policy.get("min_free_memory_mib") != 2048
        or policy.get("max_own_memory_fraction") != 0.85
    ):
        raise ValueError("dual shared-GPU policy differs from its bounded authorization")
    for device in devices:
        identity = device.get("foreign_identity", {})
        if (
            not isinstance(device.get("gpu_uuid"), str)
            or not device["gpu_uuid"].startswith("GPU-")
            or identity.get("uid") != 1003
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or not isinstance(identity.get("start_ticks"), str)
            or not identity["start_ticks"].isdigit()
            or not isinstance(identity.get("command_sha256"), str)
            or len(identity["command_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in identity["command_sha256"])
            or identity.get("script_basename") != "run_scbutterfly_archived_predictions.py"
        ):
            raise ValueError("dual shared-GPU policy requires each observed process identity")
    if (
        len({d["gpu_uuid"] for d in devices}) != 2
        or len({d["foreign_identity"]["pid"] for d in devices}) != 2
    ):
        raise ValueError("dual shared policy requires distinct devices and foreign processes")


def validate_policy(policy: dict) -> None:
    if policy.get("schema") == DYNAMIC_SCHEMA:
        validate_dynamic_policy(policy)
        return
    if policy.get("schema") == DUAL_SCHEMA:
        validate_dual_policy(policy)
        return
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
    if policy["schema"] == DYNAMIC_SCHEMA:
        admit_dynamic(policy, observed, gpus)
        return
    if policy["schema"] == DUAL_SCHEMA:
        admit_dual(policy, observed, gpus)
        return
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


def admit_dual(policy: dict, observed: dict, gpus: dict) -> None:
    devices = {d["gpu_uuid"]: d for d in policy["devices"]}
    if set(gpus) != {0, 1} or any(
        gpus[d["gpu_index"]]["uuid"] != d["gpu_uuid"] for d in devices.values()
    ):
        raise RuntimeError("shared-GPU physical device identity changed")
    seen = set()
    for entry in observed["processes"]:
        if entry["owned"]:
            continue
        device = devices.get(entry["gpu_uuid"])
        info = entry.get("identity")
        expected = device["foreign_identity"] if device is not None else {}
        if (
            device is None
            or info is None
            or any(
                info.get(k) != expected[k]
                for k in ("pid", "uid", "start_ticks", "command_sha256", "script_basename")
            )
            or entry["pid"] != expected["pid"]
            or not 0 <= entry["memory_mib"] <= policy["max_foreign_memory_mib"]
            or entry["gpu_uuid"] in seen
        ):
            raise RuntimeError(
                "unapproved or grown foreign GPU process; preserve it and stop our dispatch"
            )
        seen.add(entry["gpu_uuid"])
    # A recorded job may exit; a new PID cannot inherit its allowance.
    if any(g["free_mib"] < policy["min_free_memory_mib"] for g in gpus.values()):
        raise RuntimeError("shared-GPU physical free-memory reserve exhausted")


def admit_dynamic(policy: dict, observed: dict, gpus: dict) -> None:
    """Admit a new PID only for the approved UID/program and per-device budget."""
    devices = {d["gpu_uuid"]: d for d in policy["devices"]}
    if set(gpus) != {0, 1} or any(
        gpus[d["gpu_index"]]["uuid"] != d["gpu_uuid"] for d in devices.values()
    ):
        raise RuntimeError("shared-GPU physical device identity changed")
    seen, owned_memory = set(), {uuid: 0 for uuid in devices}
    for entry in observed["processes"]:
        uuid = entry.get("gpu_uuid")
        memory = entry.get("memory_mib")
        if uuid not in devices or type(memory) is not int or memory < 0:
            raise RuntimeError("shared-GPU process memory telemetry missing or invalid")
        if entry["owned"]:
            owned_memory[uuid] += memory
            continue
        info = entry.get("identity")
        if (
            info is None
            or info.get("uid") != policy["foreign_uid"]
            or info.get("script_basename")
            not in (policy["command_basename"], *policy.get("additional_command_basenames", []))
            or type(entry.get("pid")) is not int
            or entry["pid"] <= 0
            or info.get("pid") != entry["pid"]
            or memory > policy["max_foreign_memory_mib"]
            or uuid in seen
        ):
            raise RuntimeError(
                "unapproved or grown foreign GPU process; preserve it and stop our dispatch"
            )
        seen.add(uuid)
    for gpu in gpus.values():
        if (
            type(gpu.get("total_mib")) is not int
            or gpu["total_mib"] <= 0
            or type(gpu.get("free_mib")) is not int
            or not 0 <= gpu["free_mib"] <= gpu["total_mib"]
        ):
            raise RuntimeError("shared-GPU physical memory telemetry missing or invalid")
        if owned_memory[gpu["uuid"]] > gpu["total_mib"] * policy["max_own_memory_fraction"]:
            raise RuntimeError("own GPU workload exceeds the approved memory allowance")
        if gpu["free_mib"] < policy["min_free_memory_mib"]:
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
