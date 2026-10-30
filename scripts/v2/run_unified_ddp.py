"""One dual-rank fit at a time, with bounded last-test overlap when admitted."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from generate_unified_ddp import PRIORITY, verify_manifest
from run_group import lock, validate_preflight
from run_unified_group import (
    ALLOCATOR,
    LEASE_ROOT,
    child_environment,
    gpu_idle,
    probe_command,
    validate_completed,
)
from run_unified_overlap import evaluate_attempt, process_command, process_identity
from shared_gpu import DUAL_SCHEMA, DYNAMIC_SCHEMA, admit, snapshot, terminate_tree, validate_policy

from gradpert.config.step_schedule import load_training_schedule
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.execution.train_entry import resolve_plan
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file, sha256_json

SCHEMA = "unified-ddp-queue-1"
SERVER_ROOT = Path("/data/yilangliu")
CHILD_IDENTITY_TIMEOUT_SECONDS = 2.0
CHILD_IDENTITY_POLL_INTERVAL_SECONDS = 0.01
CHILD_STOP_TIMEOUT_SECONDS = 5.0
DEPENDENCY_POLL_SECONDS = 20
DEPENDENCY_ARMS = ("BR", "F1", "N1")
RECOVERY_SOURCE = "b37b171fd68cb29e5c8142c66778aac6da9a6f06"
RECOVERY_MODE = "resume_first_BR"
COMPLETED_BR_MODE = "continue_after_completed_BR"
COMPLETED_BR_SCHEMA = "unified-ddp-completed-br-continuation-1"
COMPLETED_BR_PARENT_NAME = "v2-ffn-recovery-10ad0aa-20261011T163726Z"
COMPLETED_BR_PARENT_SHA = "17601f562408c7661e1d893e0ba130a72cc7bd7cbda23ff2f81dc09e3f83dc0b"
COMPLETED_BR_COMPLETE_SHA = "d18f48a4133cb11f3e5993cff8abedfa5a6df154c9410f82f2187de9a2b30eba"
COMPLETED_BR_CHECKPOINT_SHA = "f679adf48082e4e334eb18601e487fb4a316ebc6e80365934e2c414c621fb9d2"
RECOVERY_SCHEMA_V1 = "unified-ddp-br-recovery-1"
RECOVERY_SCHEMA_V2 = "unified-ddp-br-recovery-2"
RESOURCE_POLICY_DECISION = ".byte-os/decisions/2026-10-12-scbutterfly-resource-policy.md"
RESOURCE_POLICY_AUTHORIZATION = "2026-10-12-user-scbutterfly-coexistence"
SHARED_RESOURCE_POLL_SECONDS = 20
WAITABLE_SHARED_OCCUPANCY_ERRORS = frozenset(
    {
        "unapproved foreign GPU process; preserve it and stop our dispatch",
        "unapproved or grown foreign GPU process; preserve it and stop our dispatch",
        "foreign GPU workload exceeds the one-job memory allowance",
        "shared-GPU physical free-memory reserve exhausted",
    }
)


def _stop_unverified_child(child: subprocess.Popen[bytes]) -> None:
    """Reap a still-owned direct child without signaling an unverified process tree."""
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=CHILD_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=CHILD_STOP_TIMEOUT_SECONDS)


def _capture_process_identity(child: subprocess.Popen[bytes]) -> dict:
    """Wait briefly for /proc identity after spawn, failing closed on races."""
    deadline = time.monotonic() + CHILD_IDENTITY_TIMEOUT_SECONDS
    while True:
        return_code = child.poll()
        if return_code is not None:
            raise RuntimeError(
                f"spawned child {child.pid} exited before its process identity was readable "
                f"(exit code {return_code})"
            )

        identity = process_identity(child.pid)
        if identity is not None:
            if identity.get("pid") != child.pid or not {"command_sha256", "start_ticks"}.issubset(
                identity
            ):
                _stop_unverified_child(child)
                raise RuntimeError(
                    "spawned child identity is incomplete or mismatched; stopped the direct "
                    "Popen child only, without signaling unverified descendants"
                )
            return identity

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_unverified_child(child)
            raise RuntimeError(
                f"timed out reading process identity for spawned child {child.pid}; stopped the "
                "direct Popen child only, without signaling unverified descendants"
            )
        time.sleep(min(CHILD_IDENTITY_POLL_INTERVAL_SECONDS, remaining))


def _terminate_active_child(active) -> None:
    """Stop only children whose identity was captured and can be revalidated."""
    child, _row, item = active
    identity = item.get("identity")
    if identity is not None:
        terminate_tree(child.pid, identity)


def matrix_contract(source: Path, manifest: Path) -> tuple[dict, tuple[str, ...], list[str]]:
    if read_json(manifest).get("schema") == "representation-precision-family-1":
        from generate_precision_family import CHANGES
        from generate_precision_family import verify_manifest as verify_precision

        return verify_precision(source, manifest), tuple(CHANGES), list(CHANGES)
    if read_json(manifest).get("schema") == "graph-mhc-four-1":
        from generate_graph_mhc import PRIORITY as graph_priority
        from generate_graph_mhc import verify_manifest as verify_graph

        return verify_graph(source, manifest), graph_priority, ["GM1"]
    if read_json(manifest).get("schema") == "ffn-four-1":
        from generate_ffn_group import PRIORITY as ffn_priority
        from generate_ffn_group import verify_manifest as verify_ffn

        return verify_ffn(source, manifest), ffn_priority, ["F1", "BR", "N1"]
    return verify_manifest(source, manifest), PRIORITY, ["CG1", "N0"]


def prepare(
    source: Path,
    manifest: Path,
    runtime: Path,
    output: Path,
    shared_policy: dict | None = None,
    dependency_queue: Path | None = None,
) -> dict:
    if not output.resolve().is_relative_to(SERVER_ROOT) or output.exists():
        raise ValueError("queue must be a fresh server directory")
    matrix, priority, initial = matrix_contract(source, manifest)
    if dependency_queue is not None and priority != ("GM1",):
        raise ValueError("only GM1 may wait on the BR/F1/N1 dependency")
    indexed = {r["name"]: r for r in matrix["rows"]}
    rows = []
    for name in priority:
        row = indexed[name]
        plan = resolve_plan(
            argparse.Namespace(
                config=source / row["config"],
                runtime=runtime,
                data_root=None,
                gpu="0,1",
                seed=1,
                defer_test=True,
            )
        )
        if plan["config_sha256"] != row["sha256"]:
            raise ValueError("launch config differs from matrix")
        if Path(plan["run_root"]).exists() or Path(plan["test_root"]).exists():
            raise FileExistsError("fresh run must not replace old artifacts")
        rows.append({"name": name, "plan": plan})
    queue = {
        "schema": SCHEMA,
        "source": str(source),
        "source_commit": rows[0]["plan"]["source_commit"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "runtime": str(runtime),
        "runtime_sha256": sha256_file(runtime),
        "rows": rows,
        "initial_preflight": initial,
        "steps": 10,
        "deferred_arms": [r["name"] for r in matrix["rows"] if r["name"] not in priority],
    }
    if shared_policy is not None:
        validate_policy(shared_policy)
        queue["shared_gpu_policy"] = copy.deepcopy(shared_policy)
    if dependency_queue is not None:
        queue["dependency"] = seal_dependency(dependency_queue)
    output.mkdir()
    atomic_json(output / "queue.json", queue)
    verify(queue)
    return queue


def verify(queue: dict) -> None:
    if (
        queue.get("mode") not in (None, RECOVERY_MODE, COMPLETED_BR_MODE)
        or ((queue.get("mode") == RECOVERY_MODE) != (queue.get("recovery") is not None))
        or (
            (queue.get("mode") == COMPLETED_BR_MODE)
            != (queue.get("completed_br_continuation") is not None)
        )
    ):
        raise ValueError("unknown or inconsistent queue recovery mode")
    if any(
        any(key.startswith("__resume") for scope in (row, row["plan"]) for key in scope)
        for row in queue["rows"]
    ):
        raise ValueError("resume is controller metadata, never a scientific plan flag")
    if queue.get("shared_gpu_policy") is not None:
        validate_policy(queue["shared_gpu_policy"])
    manifest, priority, initial = matrix_contract(Path(queue["source"]), Path(queue["manifest"]))
    if queue["schema"] != SCHEMA or [r["name"] for r in queue["rows"]] != list(priority):
        raise ValueError("queue scope/order changed")
    if queue["initial_preflight"] != initial or queue["steps"] != 10:
        raise ValueError("bounded preflight protocol changed")
    for field in ("manifest", "runtime"):
        if sha256_file(Path(queue[field])) != queue[field + "_sha256"]:
            raise ValueError("queue file changed")
    indexed = {r["name"]: r for r in manifest["rows"]}
    for row in queue["rows"]:
        plan = row["plan"]
        config = plan["resolved_config"]
        params, training = config["model"]["parameters"], config["training"]
        if (
            plan["source_commit"] != queue["source_commit"]
            or plan["gpu"] != "0,1"
            or plan["seed"] != 1
            or plan["postfit_policy"] != "deferred"
            or plan["config_sha256"] != indexed[row["name"]]["sha256"]
            or params["world_size"]["value"] != 2
            or params["accumulation"]["value"] != 2
            or training["train_batch_size"]["value"] != 4 * params["microbatch"]["value"]
            or params["validation_mode"]["value"] != "disabled"
            or training["max_epochs"]["value"] != manifest["epochs"]
            or config.get("continuation") is not None
        ):
            raise ValueError("DDP launch differs from the sealed fresh protocol")
        for field in ("config", "publication"):
            if sha256_file(Path(plan[field])) != plan[field + "_sha256"]:
                raise ValueError("sealed launch dependency changed")
    plan = queue["rows"][0]["plan"]
    identity = inspect_source_identity(
        queue["source"],
        formal=True,
        expected_repository=plan["resolved_config"]["source_code"]["repository"],
        publication_receipt=plan["publication"],
        expected_publication_receipt_sha256=plan["publication_sha256"],
    )
    if identity.commit != queue["source_commit"] or identity.dirty:
        raise ValueError("queue source is not clean and published")
    if queue.get("dependency") is not None:
        if manifest["schema"] != "graph-mhc-four-1" or priority != ("GM1",):
            raise ValueError("only GM1 may wait on the BR/F1/N1 dependency")
        verify_dependency(queue["dependency"])
    if queue.get("mode") == RECOVERY_MODE:
        verify_recovery(queue)
    elif queue.get("mode") == COMPLETED_BR_MODE:
        verify_completed_br_continuation(queue)


def _file_seal(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _sealed_json(entry: dict) -> dict:
    path = Path(entry["path"])
    _dependency_server_path(path)
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("recovery sealed file changed")
    return read_json(path)


def _assert_failed_parent(
    parent: dict, root: Path, *, stopped: bool = True, allow_recovery: bool = False
) -> dict:
    state = read_json(root / "state.json")
    if (
        state.get("status") != "failed"
        or state.get("phase") != "failed"
        or state.get("queue_sha256") != sha256_file(root / "queue.json")
        or read_json(root / "FAILURE.json") != state
        or (root / "COMPLETE.json").exists()
        or state.get("completed")
        or (
            parent.get("mode") is not None
            and not (allow_recovery and parent.get("mode") == RECOVERY_MODE)
        )
    ):
        raise ValueError("recovery requires the original unsuccessful BR-only queue")
    fit_events = [
        event
        for event in state.get("events", [])
        if event.get("event") == "start" and event.get("stage") == "fit"
    ]
    if [event["name"] for event in fit_events] != ["BR"]:
        raise ValueError("recovery only admits one interrupted BR fit, before F1/N1")
    if any(
        Path(row["plan"][key]).exists()
        for row in parent["rows"][1:]
        for key in ("run_root", "test_root")
    ):
        raise ValueError("original F1/N1 artifacts exist; recovery scope is no longer untouched")
    if stopped:
        pids = {
            state["pid"],
            *(event["pid"] for event in fit_events),
            *(item["pid"] for item in state.get("active", {}).values()),
        }
        if any(process_identity(pid) is not None for pid in pids):
            raise ValueError("original controller or fit still live; recovery cannot adopt it")
        paths = {str(root / "queue.json"), str(root / "BR.launch-plan.json")}
        br = parent["rows"][0]["plan"]
        paths.add(str(Path(br["run_root"]).parent / (br["run_id"] + ".launch-plan.json")))
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            if proc.parent.name == str(os.getpid()):
                continue
            try:
                command = proc.read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if any(path.encode() in command for path in paths):
                raise ValueError("original queue descendant is still live")
    return state


def _resume_evidence(plan: dict, *, allowed_epochs: tuple[int, ...] = (1,)) -> dict:
    root = Path(plan["run_root"])
    journal_path = root / "fit/epoch_state.json"
    journal = read_json(journal_path)
    epoch = journal["epoch"]
    identity = read_json(root / "run_manifest.json")
    config = plan["resolved_config"]
    params = config["model"]["parameters"]
    contract = {
        "schedule": asdict(load_training_schedule(config["training"]["scheduler"]["value"])),
        "teacher_start": params["teacher_start"]["value"],
        "teacher_end": params["teacher_end"]["value"],
        "microbatch": params["microbatch"]["value"],
        "bf16": True,
        "selection_metric": "final_epoch",
    }
    if (
        read_json(root / "launch.json") != plan
        or read_json(root / "resolved_config.json") != config
        or journal["identity"] != identity
        or identity["source"]["commit"] != RECOVERY_SOURCE
        or identity["source"]["published_commit"] != RECOVERY_SOURCE
        or identity["source"]["dirty"]
        or identity["config_sha256"] != plan["config_sha256"]
        or identity["resolved_config_sha256"] != sha256_json(config)
        or identity["run_id"] != plan["run_id"]
        or journal["budget"] != [4, 250]
        or journal["contract"] != contract
        or type(epoch) is not int
        or epoch not in allowed_epochs
        or journal["last"]["epoch"] != epoch
        or journal["last"]["file"] != f"epoch-{epoch:04d}.pt"
        or journal["best"] is not None
        or journal.get("selection_metric") != "final_epoch"
        or [row["epoch"] for row in read_json(root / "fit/history.json")]
        != list(range(1, epoch + 1))
        or (root / "COMPLETE.json").exists()
        or (root / "TRAIN_COMPLETE.json").exists()
        or (root / "FAILURE.json").exists()
        or Path(plan["test_root"]).exists()
        or any(root.rglob("*.pkl"))
    ):
        raise ValueError(
            "BR recovery requires an authorized committed epoch identity/budget/contract"
        )
    selected = (root / "fit" / journal["last"]["file"]).resolve(strict=True)
    role = root / "fit/last.pt"
    if (
        not selected.is_relative_to((root / "fit").resolve())
        or sha256_file(selected) != journal["last"]["sha256"]
        or sha256_file(role) != journal["last"]["sha256"]
    ):
        raise ValueError("BR committed checkpoint or last role hash differs")
    return {
        "epoch": epoch,
        "budget": [4, 250],
        "identity": identity,
        "contract": contract,
        "journal": _file_seal(journal_path),
        "checkpoint": _file_seal(selected),
        "last_role": _file_seal(role),
        "history": _file_seal(root / "fit/history.json"),
        "manifest": _file_seal(root / "run_manifest.json"),
    }


def _resolve_recovery_fresh(source: Path, row: dict, runtime: Path) -> dict:
    """Resolve in the immutable scientific package, with CUDA hidden from CPU preparation."""
    command = [
        sys.executable,
        "-c",
        "import argparse,json,sys;from pathlib import Path;"
        "from gradpert.execution.train_entry import resolve_plan;"
        "print(json.dumps(resolve_plan(argparse.Namespace(config=Path(sys.argv[1]),"
        "runtime=Path(sys.argv[2]),data_root=None,gpu='0,1',seed=1,defer_test=True))))",
        str(source / row["config"]),
        str(runtime),
    ]
    return json.loads(
        subprocess.check_output(
            command, cwd=source, env=child_environment(str(source), ""), text=True
        )
    )


def _reused_final_plan(original: dict, base: dict) -> dict:
    return {
        **copy.deepcopy(base),
        **{
            key: copy.deepcopy(original[key])
            for key in ("cuda_memory_fraction", "resource_schedule")
        },
    }


def recovery_fit_command(queue: dict, row: dict, path: Path) -> list[str]:
    if queue.get("mode") == RECOVERY_MODE and row["name"] == "BR":
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            "2",
            str(Path(queue["source"]) / "scripts/v2/distributed_train.py"),
            "--plan",
            str(path),
            "--resume",
        ]
    return [
        sys.executable,
        "-c",
        "import json,sys;from gradpert.execution.train_entry import execute_plan;"
        "execute_plan(json.load(open(sys.argv[1])))",
        str(path),
    ]


def _validate_execution_policy_override(previous: dict | None, active: dict) -> None:
    validate_policy(active)
    if previous is None or previous.get("schema") not in (DUAL_SCHEMA, DYNAMIC_SCHEMA):
        raise ValueError("recovery policy override requires the original fixed dual-GPU policy")
    validate_policy(previous)
    if active["schema"] != DYNAMIC_SCHEMA or (
        active.get("authorization_reference") != RESOURCE_POLICY_AUTHORIZATION
        or [
            {key: device[key] for key in ("gpu_index", "gpu_uuid")}
            for device in previous["devices"]
        ]
        != active["devices"]
        or any(
            active[key] != previous[key]
            for key in ("max_foreign_memory_mib", "min_free_memory_mib", "max_own_memory_fraction")
        )
    ):
        raise ValueError("recovery policy override changed its authorization, devices or budget")


def prepare_recovery(
    original_failed_queue_json: Path, output: Path, shared_policy: dict | None = None
) -> dict:
    original_failed_queue_json = original_failed_queue_json.resolve(strict=True)
    _dependency_server_path(original_failed_queue_json)
    _dependency_server_path(output)
    if output.exists():
        raise FileExistsError("recovery requires a fresh queue root")
    parent = read_json(original_failed_queue_json)
    _dependency_contract(parent, original_failed_queue_json)
    if parent["source_commit"] != RECOVERY_SOURCE or parent.get("dependency") is not None:
        raise ValueError("recovery is restricted to the original b37 BR/F1/N1 scientific source")
    ancestry = _recovery_ancestry(parent, original_failed_queue_json, output)
    verify(parent)
    if shared_policy is not None:
        _validate_execution_policy_override(parent.get("shared_gpu_policy"), shared_policy)
    old_root = original_failed_queue_json.parent
    state = _assert_failed_parent(parent, old_root, allow_recovery=True)
    if set(state.get("preflights", {})) != set(DEPENDENCY_ARMS) or (
        set(state.get("plans", {})) != set(DEPENDENCY_ARMS)
    ):
        raise ValueError("recovery needs all three original finalized ten-step preflights")
    source = Path(parent["source"])
    matrix, _, _ = matrix_contract(source, Path(parent["manifest"]))
    matrix_rows = {row["name"]: row for row in matrix["rows"]}
    rows, seeded, old_plans, final_plans = [], {}, {}, {}
    for row in parent["rows"]:
        name = row["name"]
        validate_preflight(row["plan"], state["preflights"][name])
        preflight = read_json(state["preflights"][name]["receipt"])
        if preflight.get("steps_completed") != 10 or preflight.get("kind") != "preflight_only":
            raise ValueError("recovery needs exact ten-update preflight receipts")
        old = _sealed_json(state["plans"][name])
        if state["plans"][name]["path"] != str(old_root / f"{name}.launch-plan.json"):
            raise ValueError("original finalized plan escaped its queue")
        if {
            k: v for k, v in old.items() if k not in ("cuda_memory_fraction", "resource_schedule")
        } != row["plan"]:
            raise ValueError("original finalized plan differs from frozen row")
        expected_final = finalized_plan(
            row,
            state["preflights"][name],
            old["resource_schedule"]["physical_total_bytes"],
            old["resource_schedule"].get("shared_gpu_policy"),
        )
        if old != expected_final:
            raise ValueError("original finalized budget differs from its measured preflight")
        old_plans[name] = _file_seal(Path(state["plans"][name]["path"]))
        base = (
            copy.deepcopy(row["plan"])
            if name == "BR"
            else _resolve_recovery_fresh(source, matrix_rows[name], Path(parent["runtime"]))
        )
        if name != "BR":
            if base["run_id"] in _ancestry_run_ids(ancestry):
                raise ValueError("recovery F1/N1 must use newly resolved run IDs")
            for key in ("run_root", "test_root"):
                _dependency_server_path(Path(base[key]))
                if Path(base[key]).exists():
                    raise FileExistsError("recovery F1/N1 cannot replace existing artifacts")
        rows.append({"name": name, "plan": base})
        final = _reused_final_plan(old, base)
        final_plans[name] = final
        seeded[name] = {
            "preflight": copy.deepcopy(state["preflights"][name]),
            "plan": {
                "path": str(output / f"{name}.launch-plan.json"),
                "sha256": sha256_json(final),
            },
        }
    resume = _resume_evidence(final_plans["BR"], allowed_epochs=(1, 2, 3))
    repeated = parent.get("mode") == RECOVERY_MODE
    if repeated and resume["epoch"] < parent["recovery"]["resume"]["epoch"]:
        raise ValueError("repeated recovery cannot regress the committed BR epoch")
    schema = (
        RECOVERY_SCHEMA_V2
        if repeated or resume["epoch"] != 1 or (shared_policy is not None)
        else RECOVERY_SCHEMA_V1
    )
    queue = {
        **copy.deepcopy(parent),
        "rows": rows,
        "mode": RECOVERY_MODE,
        "recovery": {
            "schema": schema,
            "output": str(output),
            "original_queue": _file_seal(original_failed_queue_json),
            "original_state": _file_seal(old_root / "state.json"),
            "original_failure": _file_seal(old_root / "FAILURE.json"),
            "original_plans": old_plans,
            "seeded": seeded,
            "resume": resume,
            "controller_source": {
                "repository_root": str(Path(__file__).resolve().parents[2]),
                "commit": subprocess.check_output(
                    ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "script": _file_seal(Path(__file__).resolve()),
            },
        },
    }
    if schema == RECOVERY_SCHEMA_V2:
        queue["recovery"]["ancestry"] = ancestry
    if shared_policy is not None:
        queue["shared_gpu_policy"] = copy.deepcopy(shared_policy)
        queue["recovery"]["execution_policy_override"] = {
            "previous_policy": copy.deepcopy(parent["shared_gpu_policy"]),
            "active_policy": copy.deepcopy(shared_policy),
            "authorization_reference": RESOURCE_POLICY_AUTHORIZATION,
            "decision": _file_seal(Path(__file__).resolve().parents[2] / RESOURCE_POLICY_DECISION),
        }
    output.mkdir()
    checkpoint_backup, journal_backup = _recovery_backup_paths(output, schema, resume["epoch"])
    shutil.copyfile(resume["checkpoint"]["path"], checkpoint_backup)
    if sha256_file(checkpoint_backup) != resume["checkpoint"]["sha256"]:
        raise ValueError("recovery checkpoint backup differs from the committed input")
    queue["recovery"]["checkpoint_backup"] = _file_seal(checkpoint_backup)
    shutil.copyfile(resume["journal"]["path"], journal_backup)
    if sha256_file(journal_backup) != resume["journal"]["sha256"]:
        raise ValueError("recovery journal backup differs from the committed input")
    queue["recovery"]["journal_backup"] = _file_seal(journal_backup)
    for name, plan in final_plans.items():
        atomic_json(output / f"{name}.launch-plan.json", plan)
        seeded[name]["plan"]["sha256"] = sha256_file(output / f"{name}.launch-plan.json")
    atomic_json(output / "queue.json", queue)
    verify(queue)
    return queue


def _completed_br_parent_state(parent: dict, path: Path) -> dict:
    _dependency_contract(parent, path)
    if (
        path.parent.name != COMPLETED_BR_PARENT_NAME
        or sha256_file(path) != COMPLETED_BR_PARENT_SHA
        or parent.get("mode") != RECOVERY_MODE
        or parent["source_commit"] != RECOVERY_SOURCE
    ):
        raise ValueError("completed BR continuation requires the exact retired 10ad parent")
    state = read_json(path.parent / "state.json")
    fits = [
        event
        for event in state.get("events", [])
        if event.get("event") == "start" and event.get("stage") == "fit"
    ]
    if (
        state.get("status") != "running"
        or state.get("phase") != "dual_gpu_fit"
        or state.get("queue_sha256") != COMPLETED_BR_PARENT_SHA
        or state.get("completed")
        or [event["name"] for event in fits] != ["BR"]
        or set(state.get("active", {})) != {"fit"}
        or state["active"]["fit"].get("name") != "BR"
        or set(state.get("plans", {})) != set(DEPENDENCY_ARMS)
        or set(state.get("preflights", {})) != set(DEPENDENCY_ARMS)
    ):
        raise ValueError("retired parent must preserve its BR-only state and untouched peers")
    if any(
        Path(row["plan"][key]).exists()
        for row in parent["rows"][1:]
        for key in ("run_root", "test_root")
    ):
        raise ValueError("original reserved F1/N1 artifacts must remain absent")
    pids = {
        state["pid"],
        *(event["pid"] for event in state.get("events", []) if "pid" in event),
        *(item["pid"] for item in state["active"].values()),
    }
    if any(process_identity(pid) is not None for pid in pids):
        raise ValueError("original controller or scientific child is still live")
    paths = {
        str(path),
        *(str(path.parent / f"{name}.launch-plan.json") for name in DEPENDENCY_ARMS),
    }
    paths.update(
        str(Path(row["plan"]["run_root"]).parent / (row["plan"]["run_id"] + ".launch-plan.json"))
        for row in parent["rows"]
    )
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        if proc.parent.name == str(os.getpid()):
            continue
        try:
            args = proc.read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(path.encode() in args for path in paths):
            raise ValueError("original queue science process is still live")
    return state


def _completed_br_preflight(plan: dict, entry: dict) -> None:
    validate_preflight(plan, entry)
    receipt = read_json(entry["receipt"])
    if (
        receipt.get("kind") != "preflight_only"
        or receipt.get("steps_requested") != 10
        or receipt.get("steps_completed") != 10
        or receipt.get("gpu") != "0,1"
        or receipt.get("world_size") != 2
        or receipt.get("inference_shape") != [300, 5000]
        or receipt["source"].get("publication_receipt_sha256") != plan["publication_sha256"]
    ):
        raise ValueError(
            "continuation requires the exact native ten-step restore/inference receipt"
        )


def _completed_br_evidence(plan: dict) -> dict:
    training = validate_training_stage(plan)
    root = Path(plan["run_root"])
    if (
        training["epoch"] != 4
        or training["last"]["epoch"] != 4
        or training["last"]["sha256"] != COMPLETED_BR_CHECKPOINT_SHA
        or sha256_file(root / "fit/last.pt") != COMPLETED_BR_CHECKPOINT_SHA
        or sha256_file(root / "COMPLETE.json") != COMPLETED_BR_COMPLETE_SHA
    ):
        raise ValueError("completed BR terminal or epoch4 last checkpoint changed")
    completed = validate_completed(plan)
    last_test = read_json(root / "fit/last-test.json")
    independent = Path(last_test["independent_evaluation_receipt"])
    if not independent.resolve().is_relative_to(root.resolve()):
        raise ValueError("completed BR evaluation receipt escaped its original run")
    files = [
        root / name
        for name in (
            "COMPLETE.json",
            "TRAIN_COMPLETE.json",
            "launch.json",
            "resolved_config.json",
            "run_manifest.json",
            "fit/epoch_state.json",
            "fit/history.json",
            "fit/last.pt",
            "fit/last-test.json",
        )
    ]
    files.extend((root / "fit" / training["last"]["file"], independent))
    return {
        "completed": completed,
        "training": training,
        "files": [_file_seal(path) for path in files],
    }


def _continuation_original_plans(parent: dict, path: Path, state: dict) -> dict:
    runtime = read_json(Path(parent["runtime"]))
    plans = {}
    for row in parent["rows"]:
        name, base = row["name"], row["plan"]
        if (
            Path(base["runtime"]).resolve() != Path(parent["runtime"]).resolve()
            or base["runtime_sha256"] != parent["runtime_sha256"]
            or Path(base["data_root"]).resolve() != Path(runtime["data_root"]).resolve()
            or Path(base["publication"]).resolve() != Path(runtime["publication_receipt"]).resolve()
            or base["publication_sha256"] != runtime["publication_sha256"]
        ):
            raise ValueError("continuation scientific runtime/data/publication binding changed")
        _completed_br_preflight(base, state["preflights"][name])
        old = _sealed_json(state["plans"][name])
        if (
            state["plans"][name]["path"] != str(path.parent / f"{name}.launch-plan.json")
            or {
                k: v
                for k, v in old.items()
                if k not in ("cuda_memory_fraction", "resource_schedule")
            }
            != base
            or old
            != finalized_plan(
                row,
                state["preflights"][name],
                old["resource_schedule"]["physical_total_bytes"],
                old["resource_schedule"].get("shared_gpu_policy"),
            )
            or old["cuda_memory_fraction"] != 0.65
            or old["resource_schedule"]["capped_eval_fraction"] != 0.25
        ):
            raise ValueError("continuation must preserve original measured fit/eval budgets")
        plans[name] = old
    return plans


def prepare_completed_br(
    original_queue: Path, output: Path, shared_policy: dict, controller_publication: Path
) -> dict:
    original_queue = original_queue.resolve(strict=True)
    output = output.resolve()
    _dependency_server_path(original_queue)
    _dependency_server_path(output)
    if output.exists() or output.is_relative_to(original_queue.parent):
        raise ValueError("completed BR continuation requires a fresh independent root")
    parent = read_json(original_queue)
    verify(parent)
    state = _completed_br_parent_state(parent, original_queue)
    _validate_execution_policy_override(parent["shared_gpu_policy"], shared_policy)
    old_plans = _continuation_original_plans(parent, original_queue, state)
    br = _completed_br_evidence(old_plans["BR"])
    source = Path(parent["source"])
    matrix, _, _ = matrix_contract(source, Path(parent["manifest"]))
    specs = {row["name"]: row for row in matrix["rows"]}
    reserved = {row["plan"]["run_id"] for row in parent["rows"]} | _ancestry_run_ids(
        parent["recovery"]["ancestry"]
    )
    rows, finals, seeded = [], {}, {}
    for row in parent["rows"]:
        name = row["name"]
        base = (
            copy.deepcopy(row["plan"])
            if name == "BR"
            else _resolve_recovery_fresh(source, specs[name], Path(parent["runtime"]))
        )
        if name != "BR" and (
            base["run_id"] in reserved
            or any(Path(base[key]).exists() for key in ("run_root", "test_root"))
        ):
            raise ValueError("F1/N1 continuation requires fresh native IDs and artifacts")
        rows.append({"name": name, "plan": base})
        finals[name] = _reused_final_plan(old_plans[name], base)
        seeded[name] = {
            "preflight": copy.deepcopy(state["preflights"][name]),
            "plan": {
                "path": str(output / f"{name}.launch-plan.json"),
                "sha256": sha256_json(finals[name]),
            },
        }
    root = Path(__file__).resolve().parents[2]
    for protected in (
        source,
        root,
        original_queue.parent,
        *(Path(row["plan"][key]) for row in parent["rows"] for key in ("run_root", "test_root")),
    ):
        if output.is_relative_to(protected.resolve()):
            raise ValueError("continuation must preserve old science/source/queue roots")
    publication = _file_seal(controller_publication.resolve(strict=True))
    identity = inspect_source_identity(
        root,
        formal=True,
        expected_repository=rows[0]["plan"]["resolved_config"]["source_code"]["repository"],
        publication_receipt=publication["path"],
        expected_publication_receipt_sha256=publication["sha256"],
    )
    if identity.dirty:
        raise ValueError("continuation controller must be clean and published")
    queue = {k: copy.deepcopy(v) for k, v in parent.items() if k not in ("recovery", "mode")}
    queue.update(
        rows=rows,
        mode=COMPLETED_BR_MODE,
        shared_gpu_policy=copy.deepcopy(shared_policy),
        completed_br_continuation={
            "schema": COMPLETED_BR_SCHEMA,
            "output": str(output),
            "original_queue": _file_seal(original_queue),
            "original_state": _file_seal(original_queue.parent / "state.json"),
            "original_plans": {
                name: _file_seal(Path(state["plans"][name]["path"])) for name in DEPENDENCY_ARMS
            },
            "seeded": seeded,
            "br": br,
            "controller_source": {
                "repository_root": str(root),
                "commit": identity.commit,
                "script": _file_seal(Path(__file__).resolve()),
                "publication": publication,
            },
            "execution_policy": {
                "previous": parent["shared_gpu_policy"],
                "active": copy.deepcopy(shared_policy),
                "decision": _file_seal(root / RESOURCE_POLICY_DECISION),
            },
        },
    )
    output.mkdir()
    for name, plan in finals.items():
        atomic_json(output / f"{name}.launch-plan.json", plan)
        seeded[name]["plan"]["sha256"] = sha256_file(output / f"{name}.launch-plan.json")
    atomic_json(output / "queue.json", queue)
    verify(queue)
    return queue


def verify_completed_br_continuation(queue: dict) -> None:
    meta = queue["completed_br_continuation"]
    if (
        set(meta)
        != {
            "schema",
            "output",
            "original_queue",
            "original_state",
            "original_plans",
            "seeded",
            "br",
            "controller_source",
            "execution_policy",
        }
        or meta["schema"] != COMPLETED_BR_SCHEMA
        or queue["source_commit"] != RECOVERY_SOURCE
        or queue.get("dependency") is not None
        or set(meta["seeded"]) != set(DEPENDENCY_ARMS)
        or set(meta["original_plans"]) != set(DEPENDENCY_ARMS)
    ):
        raise ValueError("unknown completed BR continuation scope")
    root = Path(meta["output"])
    _dependency_server_path(root)
    parent = _sealed_json(meta["original_queue"])
    verify(parent)
    path = Path(meta["original_queue"]["path"])
    state = _completed_br_parent_state(parent, path)
    if state != _sealed_json(meta["original_state"]):
        raise ValueError("immutable retired parent state changed")
    old_plans = _continuation_original_plans(parent, path, state)
    if meta["br"] != _completed_br_evidence(old_plans["BR"]):
        raise ValueError("imported BR completed evidence changed")
    controller = meta["controller_source"]
    controller_root = Path(controller["repository_root"])
    if (
        set(controller) != {"repository_root", "commit", "script", "publication"}
        or Path(controller["script"]["path"]) != controller_root / "scripts/v2/run_unified_ddp.py"
        or controller["script"] != _file_seal(Path(controller["script"]["path"]))
    ):
        raise ValueError("continuation controller source/script changed")
    publication = _sealed_json(controller["publication"])
    identity = inspect_source_identity(
        controller_root,
        formal=True,
        expected_repository=queue["rows"][0]["plan"]["resolved_config"]["source_code"][
            "repository"
        ],
        publication_receipt=controller["publication"]["path"],
        expected_publication_receipt_sha256=controller["publication"]["sha256"],
    )
    if (
        identity.dirty
        or identity.commit != controller["commit"]
        or publication["published_commit"] != identity.commit
    ):
        raise ValueError("continuation controller is not its clean published version")
    policy = meta["execution_policy"]
    if (
        set(policy) != {"previous", "active", "decision"}
        or policy["previous"] != parent["shared_gpu_policy"]
        or policy["active"] != queue["shared_gpu_policy"]
        or policy["decision"] != _file_seal(controller_root / RESOURCE_POLICY_DECISION)
    ):
        raise ValueError("continuation shared policy authorization changed")
    _validate_execution_policy_override(policy["previous"], policy["active"])
    reserved = {row["plan"]["run_id"] for row in parent["rows"]} | _ancestry_run_ids(
        parent["recovery"]["ancestry"]
    )
    originals = {row["name"]: row["plan"] for row in parent["rows"]}
    omitted = {"run_id", "run_root", "test_root"}
    for row in queue["rows"]:
        name, base = row["name"], row["plan"]
        seed = meta["seeded"][name]
        if (
            set(seed) != {"preflight", "plan"}
            or seed["preflight"] != state["preflights"][name]
            or meta["original_plans"][name] != _file_seal(Path(state["plans"][name]["path"]))
            or seed["plan"]["path"] != str(root / f"{name}.launch-plan.json")
            or _sealed_json(seed["plan"]) != _reused_final_plan(old_plans[name], base)
        ):
            raise ValueError("continuation changed its preflight or finalized plan seals")
        _completed_br_preflight(base, seed["preflight"])
        old = originals[name]
        if name == "BR":
            if base != old:
                raise ValueError("completed BR must keep its original scientific identity")
        elif (
            {k: v for k, v in base.items() if k not in omitted}
            != {k: v for k, v in old.items() if k not in omitted}
            or base["run_id"] in reserved
            or base["run_root"] != str(Path(old["run_root"]).parent / base["run_id"])
            or base["test_root"] != str(Path(old["test_root"]).parent / (base["run_id"] + "-test"))
        ):
            raise ValueError("continuation peers must be fresh with unchanged science")
        for key in ("run_root", "test_root"):
            _dependency_server_path(Path(base[key]))
    if any(len({row["plan"][key] for row in queue["rows"]}) != 3 for key in omitted):
        raise ValueError("continuation identities must be distinct")
    if any(
        root.resolve().is_relative_to(protected.resolve())
        for protected in (
            controller_root,
            Path(parent["source"]),
            path.parent,
            *(
                Path(row["plan"][key])
                for row in parent["rows"]
                for key in ("run_root", "test_root")
            ),
        )
    ):
        raise ValueError("continuation root overlaps immutable parent science")


def _recovery_backup_paths(root: Path, schema: str, epoch: int) -> tuple[Path, Path]:
    if schema == RECOVERY_SCHEMA_V1:
        return root / "BR-resume-epoch-0001.pt", root / "BR-resume-epoch_state.json"
    return (
        root / f"BR-resume-epoch-{epoch:04d}.pt",
        root / f"BR-resume-epoch-{epoch:04d}-journal.json",
    )


def _recovery_marker_receipt(queue: dict, root: Path, command: list[str], identity: dict) -> dict:
    receipt = {
        "queue_sha256": sha256_file(root / "queue.json"),
        "source_commit": queue["source_commit"],
        "command": command,
        "identity": identity,
    }
    recovery = queue["recovery"]
    if recovery["schema"] == RECOVERY_SCHEMA_V2:
        receipt.update(
            resume_epoch=recovery["resume"]["epoch"],
            resume_checkpoint_sha256=recovery["resume"]["checkpoint"]["sha256"],
            parent_queue_sha256=recovery["original_queue"]["sha256"],
        )
    return receipt


def _validate_recovery_marker(queue: dict, root: Path) -> dict:
    started = read_json(root / "RECOVERY_BR_STARTED.json")
    command = recovery_fit_command(queue, queue["rows"][0], root / "BR.launch-plan.json")
    expected = _recovery_marker_receipt(queue, root, command, started["identity"])
    if started != expected or not {"pid", "command_sha256", "start_ticks"}.issubset(
        started["identity"]
    ):
        raise ValueError("recovery progress lacks the exact authorized BR launch marker")
    return started


def _recovery_ancestry(parent: dict, path: Path, output: Path) -> list[dict]:
    """Freeze each terminal ancestor and its retained inputs; never rewrite an ancestor."""
    ancestry, seen = [], {output.resolve()}
    while True:
        path = path.resolve(strict=True)
        root = path.parent
        if path.name != "queue.json" or root in seen or output.resolve().is_relative_to(root):
            raise ValueError("recovery ancestry cycles or overlaps an existing queue root")
        seen.add(root)
        if any(
            output.resolve().is_relative_to(Path(row["plan"][key]).resolve())
            for row in parent["rows"]
            for key in ("run_root", "test_root")
        ):
            raise ValueError("recovery output must not modify an ancestor scientific root")
        state = _assert_failed_parent(parent, root, stopped=False, allow_recovery=True)
        entry = {
            "queue": _file_seal(path),
            "state": _file_seal(root / "state.json"),
            "failure": _file_seal(root / "FAILURE.json"),
            "br_plan": _file_seal(root / "BR.launch-plan.json"),
            "schema": None,
        }
        if parent.get("mode") is None:
            ancestry.append(entry)
            return ancestry
        recovery = parent["recovery"]
        if recovery["schema"] not in (RECOVERY_SCHEMA_V1, RECOVERY_SCHEMA_V2):
            raise ValueError("unknown recovery ancestor schema")
        started = _validate_recovery_marker(parent, root)
        fit = next(
            event
            for event in state["events"]
            if event.get("event") == "start" and event.get("stage") == "fit"
        )
        if fit["pid"] != started["identity"]["pid"] or fit.get("identity") != started["identity"]:
            raise ValueError("failed recovery ancestor does not bind its exact BR launch marker")
        entry.update(
            schema=recovery["schema"],
            marker=_file_seal(root / "RECOVERY_BR_STARTED.json"),
            checkpoint_backup=copy.deepcopy(recovery["checkpoint_backup"]),
            journal_backup=copy.deepcopy(recovery["journal_backup"]),
        )
        for key in ("checkpoint_backup", "journal_backup"):
            backup = entry[key]
            _dependency_server_path(Path(backup["path"]))
            if _file_seal(Path(backup["path"])) != backup:
                raise ValueError("recovery ancestor immutable input backup changed")
        ancestry.append(entry)
        path = Path(recovery["original_queue"]["path"])
        parent = _sealed_json(recovery["original_queue"])


def _ancestry_run_ids(ancestry: list[dict]) -> set[str]:
    return {
        row["plan"]["run_id"]
        for ancestor in ancestry
        for row in _sealed_json(ancestor["queue"])["rows"]
    }


def verify_recovery(queue: dict) -> None:
    schema = queue["recovery"]["schema"]
    if schema not in (RECOVERY_SCHEMA_V1, RECOVERY_SCHEMA_V2):
        raise ValueError("unknown recovery schema")
    _verify_recovery_contract(queue, schema=schema)


def _verify_recovery_contract(queue: dict, *, schema: str) -> None:
    recovery = queue["recovery"]
    fields = {
        "schema",
        "output",
        "original_queue",
        "original_state",
        "original_failure",
        "original_plans",
        "seeded",
        "resume",
        "controller_source",
        "checkpoint_backup",
        "journal_backup",
    }
    if schema == RECOVERY_SCHEMA_V2:
        fields.add("ancestry")
        if recovery.get("execution_policy_override") is not None:
            fields.add("execution_policy_override")
    if set(recovery) != fields or (
        recovery["schema"] != schema
        or queue["source_commit"] != RECOVERY_SOURCE
        or [row["name"] for row in queue["rows"]] != list(DEPENDENCY_ARMS)
        or queue.get("dependency") is not None
    ):
        raise ValueError("unknown recovery scope or metadata")
    root = Path(recovery["output"])
    _dependency_server_path(root)
    epoch = recovery["resume"]["epoch"]
    allowed_epochs = (1,) if schema == RECOVERY_SCHEMA_V1 else (1, 2, 3)
    checkpoint_path, journal_path = _recovery_backup_paths(root, schema, epoch)
    backup = recovery["checkpoint_backup"]
    if backup["path"] != str(checkpoint_path) or (
        backup["sha256"] != recovery["resume"]["checkpoint"]["sha256"]
        or sha256_file(Path(backup["path"])) != backup["sha256"]
    ):
        raise ValueError("recovery immutable checkpoint backup changed")
    journal_backup = recovery["journal_backup"]
    if journal_backup["path"] != str(journal_path) or (
        journal_backup["sha256"] != recovery["resume"]["journal"]["sha256"]
    ):
        raise ValueError("recovery immutable journal backup changed")
    journal_snapshot = _sealed_json(journal_backup)
    if schema == RECOVERY_SCHEMA_V2 and (
        journal_snapshot["identity"] != recovery["resume"]["identity"]
        or journal_snapshot["contract"] != recovery["resume"]["contract"]
        or journal_snapshot["budget"] != [4, 250]
        or journal_snapshot["epoch"] != epoch
        or journal_snapshot["last"]["epoch"] != epoch
        or journal_snapshot["last"]["file"] != f"epoch-{epoch:04d}.pt"
        or journal_snapshot["last"]["sha256"] != recovery["resume"]["checkpoint"]["sha256"]
        or recovery["resume"]["last_role"]["sha256"] != backup["sha256"]
    ):
        raise ValueError("recovery frozen journal does not bind its committed checkpoint")
    controller = recovery["controller_source"]
    script = Path(controller["script"]["path"])
    if set(controller) != {"repository_root", "commit", "script"} or (
        script != Path(controller["repository_root"]) / "scripts/v2/run_unified_ddp.py"
        or sha256_file(script) != controller["script"]["sha256"]
        or len(controller["commit"]) != 40
        or any(char not in "0123456789abcdef" for char in controller["commit"])
    ):
        raise ValueError("recovery controller source metadata differs from its sealed script")
    if (
        set(recovery["resume"])
        != {
            "epoch",
            "budget",
            "identity",
            "contract",
            "journal",
            "checkpoint",
            "last_role",
            "history",
            "manifest",
        }
        or type(epoch) is not int
        or epoch not in allowed_epochs
        or recovery["resume"]["budget"] != [4, 250]
    ):
        raise ValueError("recovery resume metadata changed its committed boundary")
    parent = _sealed_json(recovery["original_queue"])
    override = recovery.get("execution_policy_override")
    if override is not None:
        if (
            schema != RECOVERY_SCHEMA_V2
            or set(override)
            != {"previous_policy", "active_policy", "authorization_reference", "decision"}
            or (
                override["previous_policy"] != parent.get("shared_gpu_policy")
                or override["active_policy"] != queue.get("shared_gpu_policy")
                or override["authorization_reference"] != RESOURCE_POLICY_AUTHORIZATION
            )
        ):
            raise ValueError("recovery execution policy override seal changed")
        _validate_execution_policy_override(override["previous_policy"], override["active_policy"])
        decision = override["decision"]
        expected_decision = (
            Path(recovery["controller_source"]["repository_root"]) / RESOURCE_POLICY_DECISION
        )
        if decision["path"] != str(expected_decision) or (
            sha256_file(expected_decision) != decision["sha256"]
        ):
            raise ValueError("recovery execution policy authorization decision changed")
    elif queue.get("shared_gpu_policy") != parent.get("shared_gpu_policy"):
        raise ValueError("recovery execution policy changed without a sealed scoped override")
    if schema == RECOVERY_SCHEMA_V2:
        ancestry = _recovery_ancestry(parent, Path(recovery["original_queue"]["path"]), root)
        if ancestry != recovery["ancestry"]:
            raise ValueError("recovery sealed ancestry changed")
        if parent.get("mode") == RECOVERY_MODE and epoch < parent["recovery"]["resume"]["epoch"]:
            raise ValueError("repeated recovery cannot regress the committed BR epoch")
    verify(parent)
    old_state = _sealed_json(recovery["original_state"])
    _sealed_json(recovery["original_failure"])
    marker = root / "RECOVERY_BR_STARTED.json"
    _assert_failed_parent(
        parent,
        Path(recovery["original_queue"]["path"]).parent,
        stopped=not marker.exists(),
        allow_recovery=schema == RECOVERY_SCHEMA_V2,
    )
    if set(recovery["seeded"]) != set(DEPENDENCY_ARMS) or (
        set(recovery["original_plans"]) != set(DEPENDENCY_ARMS)
    ):
        raise ValueError("recovery preflight or plan scope changed")
    original_rows = {row["name"]: row for row in parent["rows"]}
    for row in queue["rows"]:
        name = row["name"]
        seed = recovery["seeded"][name]
        if set(seed) != {"preflight", "plan"} or (
            seed["preflight"] != old_state["preflights"][name]
            or recovery["original_plans"][name]
            != {key: old_state["plans"][name][key] for key in ("path", "sha256")}
        ):
            raise ValueError("recovery changed the original passed preflight evidence")
        validate_preflight(row["plan"], seed["preflight"])
        original = _sealed_json(recovery["original_plans"][name])
        actual = _sealed_json(seed["plan"])
        if seed["plan"]["path"] != str(root / f"{name}.launch-plan.json") or (
            actual != _reused_final_plan(original, row["plan"])
        ):
            raise ValueError("recovery finalized budget or launch identity changed")
        base = row["plan"]
        old = original_rows[name]["plan"]
        omitted = {"run_id", "run_root", "test_root"}
        if name == "BR":
            if base != old or actual != original:
                raise ValueError("only exact BR plan may resume")
        elif (
            {k: v for k, v in base.items() if k not in omitted}
            != {k: v for k, v in old.items() if k not in omitted}
            or base["run_id"]
            in (
                _ancestry_run_ids(recovery["ancestry"])
                if schema == RECOVERY_SCHEMA_V2
                else {r["plan"]["run_id"] for r in parent["rows"]}
            )
            or base["run_root"] != str(Path(old["run_root"]).parent / base["run_id"])
            or base["test_root"] != str(Path(old["test_root"]).parent / (base["run_id"] + "-test"))
        ):
            raise ValueError("F1/N1 recovery must be fresh with unchanged scientific settings")
    if any(
        len({row["plan"][key] for row in queue["rows"]}) != 3
        for key in ("run_id", "run_root", "test_root")
    ):
        raise ValueError("recovery run IDs and artifact roots must be distinct")
    if marker.exists():
        _validate_recovery_marker(queue, root)
    elif (
        _resume_evidence(read_json(root / "BR.launch-plan.json"), allowed_epochs=allowed_epochs)
        != recovery["resume"]
    ):
        raise ValueError("BR checkpoint/journal drifted before resume dispatch")


def _dependency_contract(parent: dict, path: Path) -> dict:
    """Bind existing parent IDs and paths, without ever resolving another run ID."""
    if (
        parent.get("schema") != SCHEMA
        or [row["name"] for row in parent.get("rows", [])] != list(DEPENDENCY_ARMS)
        or parent.get("dependency") is not None
        or read_json(Path(parent["manifest"])).get("schema") != "ffn-four-1"
    ):
        raise ValueError("dependency scope must be exactly BR/F1/N1")
    rows = []
    for row in parent["rows"]:
        plan = row["plan"]
        if (
            plan["source_commit"] != parent["source_commit"]
            or plan["repository_root"] != parent["source"]
            or plan["resolved_config"]["training"]["max_epochs"]["value"] != 4
        ):
            raise ValueError("dependency source or four-epoch protocol differs")
        entry = {
            key: plan[key]
            for key in (
                "run_id",
                "config",
                "config_sha256",
                "source_commit",
                "repository_root",
                "run_root",
                "test_root",
            )
        }
        entry.update(name=row["name"], checkpoint=str(Path(plan["run_root"]) / "fit/last.pt"))
        for key in ("config", "repository_root", "run_root", "test_root", "checkpoint"):
            _dependency_server_path(Path(entry[key]))
        rows.append(entry)
    if len({row["run_id"] for row in rows}) != 3 or len({row["run_root"] for row in rows}) != 3:
        raise ValueError("dependency run identities must be distinct")
    return {
        "schema": "unified-ddp-dependency-1",
        "queue": str(path),
        "queue_sha256": sha256_file(path),
        "source": parent["source"],
        "source_commit": parent["source_commit"],
        "rows": rows,
    }


def _dependency_server_path(path: Path) -> None:
    if not path.is_absolute() or not path.resolve().is_relative_to(SERVER_ROOT):
        raise ValueError("dependency paths must be absolute server paths")


def seal_dependency(path: Path) -> dict:
    path = path.resolve(strict=True)
    _dependency_server_path(path)
    parent = read_json(path)
    dependency = _dependency_contract(parent, path)
    verify(parent)
    state = read_json(path.parent / "state.json")
    if state.get("queue_sha256") != dependency["queue_sha256"]:
        raise ValueError("dependency controller state belongs to another queue")
    identity = process_identity(state["pid"])
    if identity is not None:
        command = process_command(state["pid"]).split(b"\0")
        if str(path).encode() not in command or not any(
            item.endswith(b"/scripts/v2/run_unified_ddp.py") for item in command
        ):
            raise ValueError("dependency controller command belongs to another queue")
    dependency["controller_identity"] = identity
    if identity is None and not (path.parent / "COMPLETE.json").exists():
        raise ValueError("dependency controller missing without scientific COMPLETE")
    return dependency


def verify_dependency(dependency: dict) -> dict:
    path = Path(dependency["queue"])
    _dependency_server_path(path)
    if sha256_file(path) != dependency["queue_sha256"]:
        raise ValueError("dependency queue hash changed")
    parent = read_json(path)
    expected = _dependency_contract(parent, path)
    sealed = {key: value for key, value in dependency.items() if key != "controller_identity"}
    if sealed != expected:
        raise ValueError("dependency sealed source or run identities changed")
    verify(parent)
    return parent


def wait_dependency(dependency: dict, state: dict, record) -> None:
    """Read-only parent supervision; no CUDA queries, leases or child processes."""
    root = Path(dependency["queue"]).parent
    state["phase"] = "waiting_dependency"
    record()

    def observe():
        raw = (root / "state.json").read_bytes()
        observed = json.loads(raw)
        state["dependency_observed"] = {
            "state": str(root / "state.json"),
            "state_sha256": hashlib.sha256(raw).hexdigest(),
            "status": observed.get("status"),
            "phase": observed.get("phase"),
            "pid": observed.get("pid"),
        }
        record()
        return observed

    while True:
        parent = verify_dependency(dependency)
        observed = observe()
        if observed.get("queue_sha256") != dependency["queue_sha256"]:
            raise ValueError("dependency state queue identity changed")
        if (root / "FAILURE.json").exists() or observed.get("status") == "failed":
            raise RuntimeError("dependency failed; GM1 cannot start")
        if (root / "COMPLETE.json").exists():
            terminal = read_json(root / "COMPLETE.json")
            # The controller writes final state before COMPLETE; refresh a sample
            # that may have been taken immediately before this atomic transition.
            observed = observe()
            if (
                terminal != observed
                or terminal.get("schema") != SCHEMA
                or terminal.get("status") != "complete"
                or terminal.get("phase") != "complete"
                or set(terminal.get("completed", {})) != set(DEPENDENCY_ARMS)
                or set(terminal.get("plans", {})) != set(DEPENDENCY_ARMS)
            ):
                raise ValueError("dependency lacks exact scientific COMPLETE state")
            completed = {}
            for row in parent["rows"]:
                entry = terminal["plans"][row["name"]]
                plan_path = root / f"{row['name']}.launch-plan.json"
                if entry["path"] != str(plan_path) or sha256_file(plan_path) != entry["sha256"]:
                    raise ValueError("dependency finalized plan path or hash changed")
                plan = read_json(plan_path)
                original = {
                    key: value
                    for key, value in plan.items()
                    if key
                    not in (
                        "cuda_memory_fraction",
                        "resource_schedule",
                    )
                }
                if original != row["plan"]:
                    raise ValueError("dependency finalized plan changed frozen run identity")
                training = validate_training_stage(plan)
                if (
                    training["epoch"] != 4
                    or training["last"]["epoch"] != 4
                    or sha256_file(Path(plan["run_root"]) / "fit/last.pt")
                    != training["last"]["sha256"]
                ):
                    raise ValueError(
                        "dependency checkpoint differs from the sealed epoch-4 last.pt"
                    )
                receipt = validate_completed(plan)
                if receipt != terminal["completed"][row["name"]]:
                    raise ValueError("dependency scientific receipt differs from COMPLETE")
                completed[row["name"]] = receipt
            if any(root.rglob("*.pkl")):
                raise ValueError("dependency successful queue contains PKL artifacts")
            state["dependency_completed"] = {
                "complete": str(root / "COMPLETE.json"),
                "complete_sha256": sha256_file(root / "COMPLETE.json"),
                "rows": completed,
            }
            record()
            return
        expected = dependency.get("controller_identity")
        if (
            observed.get("status") not in ("running", "complete")
            or expected is None
            or observed.get("pid") != expected["pid"]
            or process_identity(expected["pid"]) != expected
        ):
            if (root / "COMPLETE.json").exists():
                continue
            raise RuntimeError(
                "dependency controller missing or changed without scientific COMPLETE"
            )
        time.sleep(DEPENDENCY_POLL_SECONDS)


def probe(row: dict, directory: Path, shared: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        *probe_command({**row, "lane": "0,1"}, directory)[1:],
    ]
    if shared:
        command.extend(
            [
                "--shared-queue",
                str(directory / "queue.json"),
                "--resource-owner-pid",
                str(os.getpid()),
            ]
        )
    return command


def fit_budget(peak_reserved: int, total_bytes: int) -> float:
    """A 15% probe margin inside 60% VRAM admits a capped 65% fit +25% eval."""
    if min(peak_reserved, total_bytes) <= 0:
        raise ValueError("positive measured reserved and total memory are required")
    return 0.65 if peak_reserved * 1.15 <= total_bytes * 0.60 else 1.0


def validate_coexistence_acceptance(queue: dict, ready: dict, acceptance: dict) -> None:
    if acceptance != {
        "decision": "allow_formal",
        "queue_sha256": ready["queue_sha256"],
        "source_commit": queue["source_commit"],
        "preflight_sha256": {name: entry["sha256"] for name, entry in ready["preflights"].items()},
    }:
        raise ValueError(
            "coexistence acceptance differs from the exact tested source/queue/receipts"
        )


def needs_preflight_acceptance(queue: dict) -> bool:
    policy = queue.get("shared_gpu_policy")
    return (
        is_precision_queue(queue)
        or [row["name"] for row in queue["rows"]] == ["GM1"]
        or (policy is not None and policy["schema"] in (DUAL_SCHEMA, DYNAMIC_SCHEMA))
    )


def is_precision_queue(queue: dict) -> bool:
    return [row["name"] for row in queue["rows"]] == ["PREC-SSL32", "PREC-PRED32", "PREC-FFN8"]


def preflight_acceptance_receipt(queue: dict, state: dict) -> dict:
    return {
        "queue_sha256": state["queue_sha256"],
        "source_commit": queue["source_commit"],
        "preflights": copy.deepcopy(state["preflights"]),
        "scientific_complete": False,
        "formal_started": False,
    }


def gpu_processes_owned() -> bool:
    """Only descendants of this controller may occupy the leased GPUs."""
    parents = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    )
    for value in output.splitlines():
        pid = int(value.strip())
        visited = set()
        while pid != os.getpid() and pid in parents and pid not in visited:
            visited.add(pid)
            pid = parents[pid]
        if pid != os.getpid():
            return False
    return True


def finalized_plan(row: dict, entry: dict, total_bytes: int, shared_policy=None) -> dict:
    validate_preflight(row["plan"], entry)
    receipt = read_json(entry["receipt"])
    if receipt["steps_completed"] != 10 or receipt["kind"] != "preflight_only":
        raise ValueError("exact ten complete updates required")
    fraction = fit_budget(receipt["peak_reserved_bytes"], total_bytes)
    if shared_policy is not None:
        validate_policy(shared_policy)
        if receipt.get("shared_gpu_policy") != shared_policy:
            raise ValueError("probe did not exercise the sealed shared-GPU policy")
        fraction = min(fraction, shared_policy["max_own_memory_fraction"])
    plan = {
        **copy.deepcopy(row["plan"]),
        "cuda_memory_fraction": fraction,
        "resource_schedule": {
            "preflight_sha256": entry["sha256"],
            "fit_fraction": fraction,
            "capped_eval_fraction": 0.25,
            "probe_margin": 1.15,
            "measured_peak_reserved_bytes": receipt["peak_reserved_bytes"],
            "physical_total_bytes": total_bytes,
        },
    }
    if shared_policy is not None:
        plan["resource_schedule"]["shared_gpu_policy"] = copy.deepcopy(shared_policy)
    return plan


def execute(queue: dict, directory: Path) -> None:
    if queue.get("mode") == COMPLETED_BR_MODE:
        metadata = queue["completed_br_continuation"]
        controller = metadata["controller_source"]
        if (
            os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or directory.resolve() != Path(metadata["output"]).resolve()
            or read_json(directory / "queue.json") != queue
            or Path(controller["repository_root"]).resolve() != Path(__file__).resolve().parents[2]
            or controller["script"] != _file_seal(Path(__file__).resolve())
        ):
            raise ValueError(
                "completed BR continuation execution source/root/queue or CPU visibility changed"
            )
    if os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR:
        raise ValueError("allocator contract missing")
    if not directory.resolve().is_relative_to(SERVER_ROOT) or (directory / "state.json").exists():
        raise ValueError("execution must be a fresh server queue; no implicit restart")
    state = {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "status": "running",
        "phase": "waiting_dependency" if queue.get("dependency") is not None else "preflight",
        "queue_sha256": sha256_file(directory / "queue.json"),
        "active": {},
        "preflights": {},
        "plans": {},
        "completed": {},
        "events": [],
    }
    pending_evals: list[dict] = []
    active_eval = None
    active_fit = None
    shared_policy = queue.get("shared_gpu_policy")

    def shared_check(tag):
        resume_phase = state["phase"]
        waited = False
        while True:
            observed = snapshot(os.getpid())
            # Append evidence even when admission fails; never touch foreign processes.
            with (directory / "shared-gpu-telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps({"stage": tag, **observed}) + "\n")
            try:
                admit(shared_policy, observed)
            except RuntimeError as error:
                if state["active"] or str(error) not in WAITABLE_SHARED_OCCUPANCY_ERRORS:
                    raise
                waited = True
                state["phase"] = "waiting_shared_resources"
                state["shared_resource_wait"] = {
                    "stage": tag,
                    "waiting": True,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "active": {},
                    "observed_unix": observed["observed_unix"],
                }
                record()
                time.sleep(SHARED_RESOURCE_POLL_SECONDS)
                verify(queue)
            else:
                if waited:
                    state["phase"] = resume_phase
                    state["shared_resource_wait"].update(
                        waiting=False, admitted_unix=observed["observed_unix"]
                    )
                    record()
                return

    def record():
        state["updated_unix"] = time.time()
        atomic_json(directory / "state.json", state)

    def start(row, stage, command, gpu, stack, attempt=1):
        nonlocal active_eval, active_fit
        verify(queue)
        if shared_policy is not None:
            shared_check(f"{row['name']}-{stage}-start")
        elif not gpu_processes_owned():
            raise RuntimeError("unrelated GPU process appeared; preserve it and stop new dispatch")
        tag = f"{row['name']}-{stage}-attempt{attempt}"
        log = stack.enter_context((directory / (tag + ".log")).open("x"))
        child = subprocess.Popen(
            command,
            cwd=queue["source"],
            env=child_environment(queue["source"], gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        identity = _capture_process_identity(child)
        if queue.get("mode") == RECOVERY_MODE and row["name"] == "BR" and stage == "fit":
            marker = directory / "RECOVERY_BR_STARTED.json"
            if marker.exists() or command != recovery_fit_command(
                queue, row, directory / "BR.launch-plan.json"
            ):
                _stop_unverified_child(child)
                raise ValueError(
                    "recovery BR may be dispatched exactly once with its sealed command"
                )
            atomic_json(
                marker,
                _recovery_marker_receipt(queue, directory, command, identity),
            )
        item = {
            "name": row["name"],
            "stage": stage,
            "pid": child.pid,
            "identity": identity,
            "log": str(directory / (tag + ".log")),
            "started_unix": time.time(),
            "attempt": attempt,
            "run_id": row["plan"]["run_id"],
        }
        state["active"][stage] = item
        state["events"].append({"event": "start", **item})
        record()
        return child, row, item

    def finish(active):
        child, row, item = active
        if shared_policy is None:
            code = child.wait()
        else:
            try:
                while True:
                    try:
                        code = child.wait(timeout=5)
                        shared_check(f"{row['name']}-{item['stage']}-finish")
                        break
                    except subprocess.TimeoutExpired:
                        shared_check(f"{row['name']}-{item['stage']}-running")
            except BaseException:
                _terminate_active_child(active)
                child.wait(timeout=10)
                raise
        receipt = {**item, "exit_code": code, "finished_unix": time.time()}
        atomic_json(
            directory / f"{row['name']}-{item['stage']}-attempt{item['attempt']}.exit.json", receipt
        )
        state["active"].pop(item["stage"], None)
        state["events"].append({"event": "exit", **receipt})
        record()
        return code

    def drain_eval(stack):
        nonlocal active_eval
        if active_eval is None:
            return
        _child, row, item = active_eval
        code = finish(active_eval)
        active_eval = None
        if code:
            failure = Path(row["plan"]["run_root"]) / "POSTFIT_ATTEMPT1_FAILURE.json"
            if (
                queue.get("mode") != COMPLETED_BR_MODE
                and item["attempt"] == 1
                and failure.exists()
                and read_json(failure)["oom"]
            ):
                # The next fit is not launched until this exact eval retry drains.
                start_eval(row, stack, attempt=2)
                drain_eval(stack)
                return
            raise RuntimeError(f"{row['name']} eval failed with exit {code}")
        state["completed"][row["name"]] = validate_completed(row["plan"])
        record()

    def start_eval(row, stack, attempt=1, uncapped=False):
        nonlocal active_eval
        if queue.get("mode") == COMPLETED_BR_MODE and (attempt != 1 or uncapped):
            raise ValueError("continuation only permits capped attempt1 evaluation")
        command = [
            sys.executable,
            str(Path(queue["source"]) / "scripts/v2/run_unified_ddp.py")
            if queue.get("mode") == COMPLETED_BR_MODE
            else str(Path(__file__).resolve()),
            "--evaluate",
            str(directory / f"{row['name']}.launch-plan.json"),
            "--attempt",
            str(attempt),
        ]
        if uncapped:
            command.append("--uncapped")
        active_eval = start(row, "eval", command, "0", stack, attempt)

    def idle():
        if shared_policy is not None:
            shared_check("admission")
            return
        while True:
            usage = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            if all(gpu_idle(usage, gpu) for gpu in ("0", "1")):
                return
            state["phase"] = "waiting_idle_dual_gpu"
            record()
            time.sleep(20)

    def preflight(row, stack):
        name = row["name"]
        if name in state["preflights"]:
            return
        drain_eval(stack)
        idle()
        state["phase"] = "preflight"
        if (directory / f"{name}-preflight").exists():
            raise FileExistsError("preflight artifacts cannot be overwritten")
        active = start(
            row, "preflight", probe(row, directory, shared_policy is not None), "0,1", stack
        )
        if finish(active):
            raise RuntimeError(f"{name} DDP preflight failed; preserve evidence, do not train")
        receipt = directory / f"{name}-preflight/receipt.json"
        entry = {"receipt": str(receipt), "sha256": sha256_file(receipt)}
        validate_preflight(row["plan"], entry)
        total = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
        )
        plan = finalized_plan(
            row, entry, min(int(n) for n in total.splitlines()) * 1024**2, shared_policy
        )
        path = directory / f"{name}.launch-plan.json"
        if path.exists():
            raise FileExistsError("final launch plan already exists")
        atomic_json(path, plan)
        state["preflights"][name] = entry
        state["plans"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "fit_fraction": plan["cuda_memory_fraction"],
        }
        record()

    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(directory / "controller.lock"))
        record()
        try:
            verify(queue)
            if queue.get("mode") in (RECOVERY_MODE, COMPLETED_BR_MODE):
                metadata = (
                    queue["recovery"]
                    if queue["mode"] == RECOVERY_MODE
                    else queue["completed_br_continuation"]
                )
                if (
                    queue["mode"] == RECOVERY_MODE
                    and (directory / "RECOVERY_BR_STARTED.json").exists()
                ):
                    raise ValueError("recovery queue cannot restart an already dispatched BR")
                for name, seed in metadata["seeded"].items():
                    plan = read_json(seed["plan"]["path"])
                    validate_preflight(plan, seed["preflight"])
                    state["preflights"][name] = copy.deepcopy(seed["preflight"])
                    state["plans"][name] = {
                        **copy.deepcopy(seed["plan"]),
                        "fit_fraction": plan["cuda_memory_fraction"],
                    }
                state["phase"] = "recovery_preflight_reused"
                if queue["mode"] == COMPLETED_BR_MODE:
                    state["completed"]["BR"] = copy.deepcopy(metadata["br"]["completed"])
                    state["events"].append(
                        {
                            "event": "import_completed",
                            "name": "BR",
                            "run_id": queue["rows"][0]["plan"]["run_id"],
                        }
                    )
                    state["scientific_source_commit"] = queue["source_commit"]
                    state["controller_source"] = copy.deepcopy(metadata["controller_source"])
                    state["phase"] = "completed_br_preflights_reused"
                state["execution_shared_gpu_policy"] = copy.deepcopy(shared_policy)
                state["historical_measured_shared_gpu_policies"] = {
                    name: read_json(seed["plan"]["path"])["resource_schedule"].get(
                        "shared_gpu_policy"
                    )
                    for name, seed in metadata["seeded"].items()
                }
                record()
                if needs_preflight_acceptance(queue):
                    atomic_json(
                        directory / "PREFLIGHT_COMPLETE.json",
                        preflight_acceptance_receipt(queue, state),
                    )
            if queue.get("dependency") is not None:
                wait_dependency(queue["dependency"], state, record)
                verify(queue)
            if shared_policy is not None:
                shared_check("initial_admission")
            stack.enter_context(lock(LEASE_ROOT / "v2-gpu-0.lock"))
            stack.enter_context(lock(LEASE_ROOT / "v2-gpu-1.lock"))
            indexed = {r["name"]: r for r in queue["rows"]}
            # Expensive CG1 is checked before any formal fit at the common batch.
            for name in queue["initial_preflight"]:
                preflight(indexed[name], stack)
            if needs_preflight_acceptance(queue):
                # GM1 and new dual-GPU coexistence require receipt inspection before fitting.
                state["phase"] = "waiting_coexistence_acceptance"
                record()
                ready = preflight_acceptance_receipt(queue, state)
                atomic_json(directory / "PREFLIGHT_COMPLETE.json", ready)
                approval = directory / "COEXISTENCE_ACCEPTANCE.json"
                while not approval.exists():
                    if shared_policy is not None:
                        shared_check("waiting_coexistence_acceptance")
                    time.sleep(5)
                validate_coexistence_acceptance(queue, ready, read_json(approval))
            for row in queue["rows"]:
                if queue.get("mode") == COMPLETED_BR_MODE and row["name"] == "BR":
                    continue
                preflight(row, stack)
                path = directory / f"{row['name']}.launch-plan.json"
                actual = {**row, "plan": read_json(path)}
                budget = actual["plan"]["cuda_memory_fraction"]
                if budget > 0.65:
                    drain_eval(stack)
                    while pending_evals:
                        start_eval(pending_evals.pop(0), stack, uncapped=True)
                        drain_eval(stack)
                if active_eval is not None and active_eval[0].poll() is not None:
                    drain_eval(stack)
                resume_br = queue.get("mode") == RECOVERY_MODE and row["name"] == "BR"
                if Path(actual["plan"]["test_root"]).exists() or (
                    Path(actual["plan"]["run_root"]).exists() and not resume_br
                ):
                    raise FileExistsError("fresh fit cannot replace prior run")
                if (
                    resume_br
                    and _resume_evidence(
                        actual["plan"], allowed_epochs=(queue["recovery"]["resume"]["epoch"],)
                    )
                    != queue["recovery"]["resume"]
                ):
                    raise ValueError("BR resume checkpoint changed immediately before dispatch")
                state["phase"] = "dual_gpu_fit"
                command = recovery_fit_command(queue, actual, path)
                active_fit = start(actual, "fit", command, "0,1", stack)
                if budget == 0.65 and active_eval is None and pending_evals:
                    start_eval(pending_evals.pop(0), stack)
                    state["phase"] = "dual_gpu_fit_with_capped_eval"
                    record()
                if finish(active_fit):
                    active_fit = None
                    raise RuntimeError(f"{row['name']} fit failed; no later fit dispatched")
                active_fit = None
                validate_training_stage(actual["plan"])
                if queue.get("mode") == COMPLETED_BR_MODE:
                    start_eval(actual, stack)
                    drain_eval(stack)
                elif not is_precision_queue(queue):
                    pending_evals.append(actual)
            drain_eval(stack)
            while pending_evals:
                start_eval(pending_evals.pop(0), stack, uncapped=True)
                drain_eval(stack)
            state.update(status="complete", phase="complete")
            record()
            atomic_json(directory / "COMPLETE.json", state)
        except BaseException as error:
            if shared_policy is not None:
                for active in (active_fit, active_eval):
                    if active is not None:
                        _terminate_active_child(active)
            # Drain only the already-authorized evaluator, never launch peers.
            state.update(
                status="failed", phase="failed", error_type=type(error).__name__, error=str(error)
            )
            record()
            atomic_json(directory / "FAILURE.json", state)
            if active_eval is not None:
                with contextlib.suppress(BaseException):
                    drain_eval(stack)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--uncapped", action="store_true")
    parser.add_argument("--shared-policy", type=Path)
    parser.add_argument("--dependency-queue", type=Path)
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--continue-completed-br", type=Path)
    parser.add_argument("--controller-publication", type=Path)
    args = parser.parse_args()
    if args.continue_completed_br:
        if (
            not args.output
            or not args.shared_policy
            or not args.controller_publication
            or any(
                (
                    args.recover_from,
                    args.manifest,
                    args.runtime,
                    args.queue,
                    args.evaluate,
                    args.dependency_queue,
                    args.uncapped,
                )
            )
            or args.attempt != 1
        ):
            parser.error(
                "completed BR continuation requires parent/output/shared-policy/"
                "controller-publication only"
            )
        prepare_completed_br(
            args.continue_completed_br,
            args.output,
            read_json(args.shared_policy),
            args.controller_publication,
        )
    elif args.controller_publication:
        parser.error("controller-publication is only for completed BR continuation")
    elif args.recover_from:
        if (
            not args.output
            or any(
                (
                    args.manifest,
                    args.runtime,
                    args.queue,
                    args.evaluate,
                    args.dependency_queue,
                    args.uncapped,
                )
            )
            or args.attempt != 1
        ):
            parser.error(
                "recovery requires recover-from/output; only shared-policy may override execution"
            )
        if args.shared_policy:
            prepare_recovery(
                args.recover_from, args.output, shared_policy=read_json(args.shared_policy)
            )
        else:
            prepare_recovery(args.recover_from, args.output)
    elif args.evaluate:
        evaluate_attempt(
            read_json(args.evaluate),
            args.attempt,
            0.25 if args.attempt == 1 and not args.uncapped else 1.0,
            gpu="0",
            full_first=args.uncapped,
        )
    elif args.queue:
        execute(read_json(args.queue), args.queue.parent)
    else:
        if not all((args.manifest, args.runtime, args.output)):
            parser.error("preparation requires manifest/runtime/output")
        prepare(
            Path(__file__).resolve().parents[2],
            args.manifest,
            args.runtime,
            args.output,
            read_json(args.shared_policy) if args.shared_policy else None,
            args.dependency_queue,
        )


if __name__ == "__main__":
    main()
