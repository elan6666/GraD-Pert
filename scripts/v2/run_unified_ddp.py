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
from shared_gpu import DUAL_SCHEMA, admit, snapshot, terminate_tree, validate_policy

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
    if queue.get("mode") not in (None, RECOVERY_MODE) or (
        (queue.get("mode") == RECOVERY_MODE) != (queue.get("recovery") is not None)
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


def _file_seal(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _sealed_json(entry: dict) -> dict:
    path = Path(entry["path"])
    _dependency_server_path(path)
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("recovery sealed file changed")
    return read_json(path)


def _assert_failed_parent(parent: dict, root: Path, *, stopped: bool = True) -> dict:
    state = read_json(root / "state.json")
    if (
        state.get("status") != "failed"
        or state.get("phase") != "failed"
        or state.get("queue_sha256") != sha256_file(root / "queue.json")
        or read_json(root / "FAILURE.json") != state
        or (root / "COMPLETE.json").exists()
        or state.get("completed")
        or parent.get("mode") is not None
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


def _resume_evidence(plan: dict) -> dict:
    root = Path(plan["run_root"])
    journal_path = root / "fit/epoch_state.json"
    journal = read_json(journal_path)
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
        or journal["epoch"] != 1
        or journal["last"]["epoch"] != 1
        or journal["last"]["file"] != "epoch-0001.pt"
        or journal["best"] is not None
        or journal.get("selection_metric") != "final_epoch"
        or [row["epoch"] for row in read_json(root / "fit/history.json")] != [1]
        or (root / "COMPLETE.json").exists()
        or (root / "TRAIN_COMPLETE.json").exists()
        or (root / "FAILURE.json").exists()
        or Path(plan["test_root"]).exists()
        or any(root.rglob("*.pkl"))
    ):
        raise ValueError(
            "BR recovery requires the exact committed epoch-1 identity/budget/contract"
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
        "epoch": 1,
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


def prepare_recovery(original_failed_queue_json: Path, output: Path) -> dict:
    original_failed_queue_json = original_failed_queue_json.resolve(strict=True)
    _dependency_server_path(original_failed_queue_json)
    _dependency_server_path(output)
    if output.exists():
        raise FileExistsError("recovery requires a fresh queue root")
    parent = read_json(original_failed_queue_json)
    _dependency_contract(parent, original_failed_queue_json)
    if parent["source_commit"] != RECOVERY_SOURCE or parent.get("dependency") is not None:
        raise ValueError("recovery is restricted to the original b37 BR/F1/N1 scientific source")
    verify(parent)
    old_root = original_failed_queue_json.parent
    state = _assert_failed_parent(parent, old_root)
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
            parent.get("shared_gpu_policy"),
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
            if base["run_id"] in {r["plan"]["run_id"] for r in parent["rows"]}:
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
    resume = _resume_evidence(final_plans["BR"])
    queue = {
        **copy.deepcopy(parent),
        "rows": rows,
        "mode": RECOVERY_MODE,
        "recovery": {
            "schema": "unified-ddp-br-recovery-1",
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
    output.mkdir()
    checkpoint_backup = output / "BR-resume-epoch-0001.pt"
    shutil.copyfile(resume["checkpoint"]["path"], checkpoint_backup)
    if sha256_file(checkpoint_backup) != resume["checkpoint"]["sha256"]:
        raise ValueError("recovery checkpoint backup differs from the committed input")
    queue["recovery"]["checkpoint_backup"] = _file_seal(checkpoint_backup)
    journal_backup = output / "BR-resume-epoch_state.json"
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


def verify_recovery(queue: dict) -> None:
    recovery = queue["recovery"]
    if set(recovery) != {
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
    } or (
        recovery["schema"] != "unified-ddp-br-recovery-1"
        or queue["source_commit"] != RECOVERY_SOURCE
        or [row["name"] for row in queue["rows"]] != list(DEPENDENCY_ARMS)
        or queue.get("dependency") is not None
    ):
        raise ValueError("unknown recovery scope or metadata")
    root = Path(recovery["output"])
    _dependency_server_path(root)
    backup = recovery["checkpoint_backup"]
    if backup["path"] != str(root / "BR-resume-epoch-0001.pt") or (
        backup["sha256"] != recovery["resume"]["checkpoint"]["sha256"]
        or sha256_file(Path(backup["path"])) != backup["sha256"]
    ):
        raise ValueError("recovery immutable checkpoint backup changed")
    journal_backup = recovery["journal_backup"]
    if journal_backup["path"] != str(root / "BR-resume-epoch_state.json") or (
        journal_backup["sha256"] != recovery["resume"]["journal"]["sha256"]
    ):
        raise ValueError("recovery immutable journal backup changed")
    _sealed_json(journal_backup)
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
        or recovery["resume"]["epoch"] != 1
        or recovery["resume"]["budget"] != [4, 250]
    ):
        raise ValueError("recovery resume metadata changed its committed boundary")
    parent = _sealed_json(recovery["original_queue"])
    verify(parent)
    old_state = _sealed_json(recovery["original_state"])
    _sealed_json(recovery["original_failure"])
    marker = root / "RECOVERY_BR_STARTED.json"
    _assert_failed_parent(
        parent, Path(recovery["original_queue"]["path"]).parent, stopped=not marker.exists()
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
            or base["run_id"] in {r["plan"]["run_id"] for r in parent["rows"]}
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
        started = read_json(marker)
        expected_command = recovery_fit_command(
            queue, queue["rows"][0], root / "BR.launch-plan.json"
        )
        if set(started) != {"queue_sha256", "source_commit", "command", "identity"} or (
            started["queue_sha256"] != sha256_file(root / "queue.json")
            or started["source_commit"] != RECOVERY_SOURCE
            or started["command"] != expected_command
            or not {"pid", "command_sha256", "start_ticks"}.issubset(started["identity"])
        ):
            raise ValueError("recovery progress lacks the exact authorized BR launch marker")
    elif _resume_evidence(read_json(root / "BR.launch-plan.json")) != recovery["resume"]:
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
    return [row["name"] for row in queue["rows"]] == ["GM1"] or (
        policy is not None and policy["schema"] == DUAL_SCHEMA
    )


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
        observed = snapshot(os.getpid())
        # Append evidence even when admission fails; never touch foreign processes.
        with (directory / "shared-gpu-telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": tag, **observed}) + "\n")
        admit(shared_policy, observed)

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
                {
                    "queue_sha256": state["queue_sha256"],
                    "source_commit": queue["source_commit"],
                    "command": command,
                    "identity": identity,
                },
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
            if item["attempt"] == 1 and failure.exists() and read_json(failure)["oom"]:
                # The next fit is not launched until this exact eval retry drains.
                start_eval(row, stack, attempt=2)
                drain_eval(stack)
                return
            raise RuntimeError(f"{row['name']} eval failed with exit {code}")
        state["completed"][row["name"]] = validate_completed(row["plan"])
        record()

    def start_eval(row, stack, attempt=1, uncapped=False):
        nonlocal active_eval
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
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
            if queue.get("mode") == RECOVERY_MODE:
                if (directory / "RECOVERY_BR_STARTED.json").exists():
                    raise ValueError("recovery queue cannot restart an already dispatched BR")
                for name, seed in queue["recovery"]["seeded"].items():
                    plan = read_json(seed["plan"]["path"])
                    validate_preflight(plan, seed["preflight"])
                    state["preflights"][name] = copy.deepcopy(seed["preflight"])
                    state["plans"][name] = {
                        **copy.deepcopy(seed["plan"]),
                        "fit_fraction": plan["cuda_memory_fraction"],
                    }
                state["phase"] = "recovery_preflight_reused"
                record()
            if queue.get("dependency") is not None:
                wait_dependency(queue["dependency"], state, record)
                verify(queue)
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
                ready = {
                    "queue_sha256": state["queue_sha256"],
                    "source_commit": queue["source_commit"],
                    "preflights": copy.deepcopy(state["preflights"]),
                    "scientific_complete": False,
                    "formal_started": False,
                }
                atomic_json(directory / "PREFLIGHT_COMPLETE.json", ready)
                approval = directory / "COEXISTENCE_ACCEPTANCE.json"
                while not approval.exists():
                    if shared_policy is not None:
                        shared_check("waiting_coexistence_acceptance")
                    time.sleep(5)
                validate_coexistence_acceptance(queue, ready, read_json(approval))
            for row in queue["rows"]:
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
                if resume_br and _resume_evidence(actual["plan"]) != queue["recovery"]["resume"]:
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
    args = parser.parse_args()
    if args.recover_from:
        if (
            not args.output
            or any(
                (
                    args.manifest,
                    args.runtime,
                    args.queue,
                    args.evaluate,
                    args.shared_policy,
                    args.dependency_queue,
                    args.uncapped,
                )
            )
            or args.attempt != 1
        ):
            parser.error(
                "recovery requires only recover-from/output; no ordinary preparation flags"
            )
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
