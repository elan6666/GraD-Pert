"""Exact worker adoption, fail-closed admission and dispatcher-only migration."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/v2"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "official_parallel_test", SCRIPTS / "run_official_parallel.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identity(pid=20, ppid=10):
    return {
        "pid": pid,
        "uid": 1002,
        "ppid": ppid,
        "start_ticks": "123",
        "command_sha256": "a" * 64,
        "script_basename": None,
    }


def test_adoption_reparenting_does_not_change_identity():
    worker = identity()
    assert module.same_process(worker, {**worker, "ppid": 1})
    for field, changed in [
        ("pid", 21),
        ("uid", 1003),
        ("start_ticks", "124"),
        ("command_sha256", "b" * 64),
    ]:
        assert not module.same_process(worker, {**worker, field: changed})
    assert not module.same_process(worker, None)


def test_only_exact_adopted_tree_becomes_owned(monkeypatch):
    worker = identity()
    observed = {
        "processes": [
            {"pid": 20, "owned": False, "identity": identity()},
            {"pid": 21, "owned": False, "identity": identity(21, 20)},
            {"pid": 22, "owned": False, "identity": identity(22, 1)},
        ]
    }
    monkeypatch.setattr(module, "snapshot", lambda _: copy.deepcopy(observed))
    monkeypatch.setattr(module, "process_info", lambda _: {**worker, "ppid": 1})
    monkeypatch.setattr(module, "parent_map", lambda: {20: 1, 21: 20, 22: 1})
    assert [p["owned"] for p in module.owned_snapshot(worker)["processes"]] == [True, True, False]
    monkeypatch.setattr(module, "process_info", lambda _: {**worker, "start_ticks": "reused"})
    with pytest.raises(ValueError, match="reused"):
        module.owned_snapshot(worker)


def test_idle_gpu1_requires_no_process_even_an_approved_small_one(monkeypatch):
    policy = {
        "devices": [{"gpu_index": 0, "gpu_uuid": "gpu0"}, {"gpu_index": 1, "gpu_uuid": "gpu1"}]
    }
    monkeypatch.setattr(module, "admit", lambda *_: None)
    for uid in (1002, 1003):
        with pytest.raises(RuntimeError, match="occupied"):
            module.validate_idle_gpu1(
                policy,
                {"processes": [{"gpu_uuid": "gpu1", "pid": 25, "identity": {"uid": uid}}]},
                identity(),
            )
    observed = {
        "processes": [{"gpu_uuid": "gpu0", "pid": 20, "identity": identity(), "owned": False}]
    }
    module.validate_idle_gpu1(policy, observed, identity())
    assert observed["processes"][0]["owned"]


def migration_fixture(tmp_path, monkeypatch):
    previous, output, source = tmp_path / "previous", tmp_path / "next", tmp_path / "scientific"
    previous.mkdir()
    (source / "scripts/v2").mkdir(parents=True)
    (source / "scripts/v2/run_official_expression_group.py").write_text(
        "# reviewed default SIGTERM\n"
    )
    worker, controller = identity(), identity(10, 1)
    state = {
        "status": "running",
        "phase": "scdfm_full",
        "pid": 10,
        "active": {"name": "scdfm", "mode": "full", "pid": 20},
    }
    (previous / "state.json").write_text(json.dumps(state))
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    row = {
        "name": "gears",
        "runtime": str(runtime),
        "config": str(tmp_path / "config"),
        "run_root": str(tmp_path / "unused"),
        "preflight_root": str(tmp_path / "unused-preflight"),
    }
    queue = {
        "source": str(source),
        "source_commit": "c" * 40,
        "rows": [{}, row],
        "shared_gpu_policy": {},
    }
    monkeypatch.setattr(module, "SERVER", tmp_path)
    monkeypatch.setattr(module, "verify_parent", lambda *_: queue)
    monkeypatch.setattr(module, "scientific_tree", lambda _: {})
    monkeypatch.setattr(
        module, "inspect_source_identity", lambda *_, **__: SimpleNamespace(commit="d" * 40)
    )
    monkeypatch.setattr(module.os, "getuid", lambda: 1002)
    monkeypatch.setattr(module.os, "getpgid", lambda _: 20)
    monkeypatch.setattr(module, "validate_cpu_gate", lambda *_: None)
    monkeypatch.setattr(module, "validate_idle_gpu1", lambda *_: None)
    monkeypatch.setattr(module, "snapshot", lambda _: {})
    original = Path.read_bytes

    def command(path):
        if str(path) == "/proc/10/cmdline":
            return (
                b"python\0run_official_expression_group.py\0--queue\0"
                + str(previous / "queue.json").encode()
                + b"\0"
            )
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", command)
    signals = []
    retired = [False]

    def kill(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGTERM:
            retired[0] = True

    monkeypatch.setattr(module.os, "kill", kill)
    monkeypatch.setattr(
        module,
        "process_info",
        lambda pid: (None if retired[0] else controller) if pid == 10 else worker,
    )
    return previous, output, signals, state


def test_migration_signals_dispatcher_only_preserves_worker(tmp_path, monkeypatch):
    previous, output, signals, _ = migration_fixture(tmp_path, monkeypatch)
    module.migrate(previous, "seal", output, tmp_path / "publication", "pubseal")
    assert signals == [(10, signal.SIGSTOP), (10, signal.SIGTERM), (10, signal.SIGCONT)]
    assert module.read(output / "MIGRATION.json")["worker_preserved"]
    assert module.read(output / "schedule.json")["adopted_worker"]["pid"] == 20
    assert not (previous / "COMPLETE.json").exists()


def test_migration_abort_resumes_dispatcher_without_signalling_worker(tmp_path, monkeypatch):
    previous, output, signals, _ = migration_fixture(tmp_path, monkeypatch)
    original = module.read
    count = [0]

    def racing_read(path):
        result = original(path)
        if path == previous / "state.json":
            count[0] += 1
            if count[0] > 1:
                result["phase"] = "changed"
        return result

    monkeypatch.setattr(module, "read", racing_read)
    with pytest.raises(ValueError, match="advanced"):
        module.migrate(previous, "seal", output, tmp_path / "publication", "pubseal")
    assert signals == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]
    assert not output.exists()


def test_started_gears_cannot_be_adopted_or_launched_again(tmp_path, monkeypatch):
    previous, output, signals, _ = migration_fixture(tmp_path, monkeypatch)
    (tmp_path / "unused").mkdir()
    with pytest.raises(FileExistsError, match="already started"):
        module.migrate(previous, "seal", output, tmp_path / "publication", "pubseal")
    assert not signals


def test_tree_fingerprint_detects_scientific_changes(tmp_path):
    for name in ("src", "benchmarks"):
        (tmp_path / name).mkdir()
    source = tmp_path / "src/model.py"
    source.write_text("x=1")
    first = module.scientific_tree(tmp_path)
    source.write_text("x=2")
    assert first != module.scientific_tree(tmp_path)


def test_completion_forwarder_waits_for_both_and_never_overwrites(tmp_path, monkeypatch):
    previous, directory = tmp_path / "old", tmp_path / "new"
    previous.mkdir()
    directory.mkdir()
    (previous / "state.json").write_text(json.dumps({"source_commit": "science"}))
    (directory / "schedule.json").write_text("{}")
    queue = {"source_commit": "science", "rows": [{"name": "scdfm"}, {"name": "gears"}]}
    monkeypatch.setattr(module, "verify_parent", lambda *_: queue)
    monkeypatch.setattr(module, "terminal_entry", lambda row, _: {"receipt_sha256": row["name"]})
    schedule = {
        "previous": str(previous),
        "previous_queue_sha256": "seal",
        "controller_commit": "controller",
    }
    with pytest.raises(ValueError, match="both"):
        module.forward_completion(schedule, directory, {"scdfm": {}})
    assert not (previous / "COMPLETE.json").exists()
    completed = {name: {"receipt_sha256": name} for name in ("scdfm", "gears")}
    module.forward_completion(schedule, directory, completed)
    result = module.read(previous / "COMPLETE.json")
    assert result["source_commit"] == "science"
    assert result["completion_controller_commit"] == "controller"
    assert result["completed"] == completed
    with pytest.raises(FileExistsError, match="overwrite"):
        module.forward_completion(schedule, directory, completed)


def test_gears_failure_keeps_scdfm_running_and_no_terminal_forward(tmp_path, monkeypatch):
    previous, directory = tmp_path / "old", tmp_path / "new"
    previous.mkdir()
    directory.mkdir()
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"python": "official-python"}))
    gears = {
        "name": "gears",
        "config": "frozen-config",
        "runtime": str(runtime),
        "run_root": str(tmp_path / "gears"),
        "preflight_root": str(tmp_path / "gears-preflight"),
    }
    queue = {
        "rows": [{"name": "scdfm"}, gears],
        "source_commit": "science",
        "source": str(tmp_path),
        "shared_gpu_policy": {},
    }
    schedule = {
        "schema": module.SCHEMA,
        "previous": str(previous),
        "previous_queue_sha256": "seal",
        "controller_commit": "controller",
        "source_commit": "science",
        "rows": queue["rows"],
        "shared_gpu_policy": {},
        "gears_physical_gpu": "1",
        "publication": "pub",
        "publication_sha256": "pubseal",
        "retired_controller": identity(10),
        "adopted_worker": identity(),
    }
    (directory / "schedule.json").write_text(json.dumps(schedule))
    (directory / "MIGRATION.json").write_text(
        json.dumps(
            {
                "schedule_sha256": module.sha256_file(directory / "schedule.json"),
                "worker_preserved": True,
            }
        )
    )
    monkeypatch.setattr(module, "SERVER", tmp_path)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(
        module, "inspect_source_identity", lambda *_, **__: SimpleNamespace(commit="controller")
    )
    monkeypatch.setattr(module, "verify_parent", lambda *_: queue)
    monkeypatch.setattr(module, "process_info", lambda pid: None if pid == 10 else identity())
    monkeypatch.setattr(module, "owned_snapshot", lambda _: {})
    monkeypatch.setattr(module, "admit", lambda *_: None)
    monkeypatch.setattr(module, "lock", lambda _: contextlib.nullcontext())
    launches = []

    def launch(argv, **kwargs):
        launches.append((argv, kwargs))
        return SimpleNamespace(pid=30, returncode=1, poll=lambda: 1)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    signals = []
    monkeypatch.setattr(module.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match="GEARS preflight exited"):
        module.execute(directory)
    assert launches[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert launches[0][1]["cwd"] == queue["source"]
    assert not signals  # In particular, no adopted worker/process group is signalled.
    assert module.read(directory / "FAILURE.json")["adopted_worker_preserved"]
    assert not (previous / "COMPLETE.json").exists()


def test_forward_completion_checks_each_scientific_source_separately(tmp_path, monkeypatch):
    previous, directory = tmp_path / "old", tmp_path / "new"
    previous.mkdir()
    directory.mkdir()
    (previous / "state.json").write_text("{}")
    (directory / "schedule.json").write_text("{}")
    rows = [
        {"name": "scdfm", "scientific_source_commit": "preserved"},
        {"name": "gears", "scientific_source_commit": "new-adapter"},
    ]
    queue = {"source_commit": "controller", "rows": rows}
    monkeypatch.setattr(module, "verify_parent", lambda *_: queue)
    observed = []

    def terminal(row, source):
        observed.append((row["name"], source))
        return {"scientific_source": source}

    monkeypatch.setattr(module, "terminal_entry", terminal)
    completed = {
        row["name"]: {"scientific_source": row["scientific_source_commit"]} for row in rows
    }
    schedule = {
        "previous": str(previous),
        "previous_queue_sha256": "seal",
        "controller_commit": "controller",
    }
    module.forward_completion(schedule, directory, completed)
    assert observed == [("scdfm", "preserved"), ("gears", "new-adapter")]


def test_repair_retires_only_preparation_and_preserves_original_scdfm(tmp_path, monkeypatch):
    old, output, current_source = tmp_path / "old", tmp_path / "new", tmp_path / "source"
    old.mkdir()
    current_source.mkdir()
    config = current_source / "configs/v2/official_expression_20261013/gears_jurkat.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("frozen")
    controller, gears_child, worker = identity(10, 1), identity(30, 10), identity(20, 1)
    state = {
        "status": "running",
        "phase": "parallel",
        "pid": 10,
        "active": {
            "name": "gears",
            "mode": "preflight",
            "pid": 30,
            "run_root": str(tmp_path / "probe"),
        },
    }
    (old / "state.json").write_text(json.dumps(state))
    schedule = {
        "previous": str(tmp_path / "parent"),
        "previous_queue_sha256": "parent-seal",
        "adopted_worker": worker,
    }
    (old / "schedule.json").write_text(json.dumps(schedule))
    parent = {
        "source": str(current_source),
        "source_commit": "preserved-science",
        "shared_gpu_policy": {"devices": [{"gpu_uuid": "gpu0"}, {"gpu_uuid": "gpu1"}]},
        "rows": [{"name": "scdfm"}, {"name": "gears", "config_sha256": module.sha256_file(config)}],
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "source_commit": "new-science",
                "publication_sha256": "publication-sha",
                "runs_root": str(tmp_path / "runs"),
            }
        )
    )
    monkeypatch.setattr(module, "ROOT", current_source)
    monkeypatch.setattr(module, "SERVER", tmp_path)
    monkeypatch.setattr(module, "verify_parent", lambda *_: parent)
    monkeypatch.setattr(module, "scientific_tree", lambda _: {})
    monkeypatch.setattr(
        module, "inspect_source_identity", lambda *_, **__: SimpleNamespace(commit="new-science")
    )
    monkeypatch.setattr(module, "validate_cpu_gate", lambda *_: None)
    monkeypatch.setattr(module, "owned_snapshot", lambda _: {"processes": []})
    monkeypatch.setattr(module, "admit", lambda *_: None)
    monkeypatch.setattr(module.os, "getuid", lambda: 1002)
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    stopped = set()
    signals = []

    def kill(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGTERM:
            stopped.add(pid)

    monkeypatch.setattr(module.os, "kill", kill)
    monkeypatch.setattr(module.os, "killpg", kill)
    monkeypatch.setattr(
        module,
        "process_info",
        lambda pid: None if pid in stopped else {10: controller, 30: gears_child, 20: worker}[pid],
    )
    read_bytes = Path.read_bytes
    read_text = Path.read_text

    def proc_bytes(path):
        if str(path) == "/proc/10/cmdline":
            return b"python\0run_official_parallel.py\0--execute\0" + str(old).encode() + b"\0"
        return read_bytes(path)

    def proc_text(path, *args, **kwargs):
        if str(path) == "/proc/10/status":
            return "SigCgt: 0\nSigIgn: 0\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", proc_bytes)
    monkeypatch.setattr(Path, "read_text", proc_text)
    module.repair_gears(
        old,
        module.sha256_file(old / "schedule.json"),
        output,
        runtime_path,
        tmp_path / "pub",
        "publication-sha",
    )
    assert signals == [
        (10, signal.SIGSTOP),
        (30, signal.SIGTERM),
        (10, signal.SIGTERM),
        (10, signal.SIGCONT),
    ]
    assert 20 not in stopped
    new_queue = module.read(output / "official-parent/queue.json")
    assert new_queue["rows"][0]["scientific_source_commit"] == "preserved-science"
    assert new_queue["rows"][1]["scientific_source_commit"] == "new-science"
    assert module.read(output / "MIGRATION.json")["worker_preserved"]
