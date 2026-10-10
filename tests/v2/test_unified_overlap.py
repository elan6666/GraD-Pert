"""CPU scheduling contracts; no scientific tensors or GPU fits materialized."""

import contextlib
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
spec = importlib.util.spec_from_file_location(
    "unified_overlap", ROOT / "scripts/v2/run_unified_overlap.py"
)
assert spec and spec.loader
overlap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlap)


def test_full_occupancy_does_not_add_a_third_process():
    row = {"name": "C1"}
    assert overlap.eligible_stages({"fit": row, "eval": row}, [row], [row]) == []


def test_postfit_and_next_fit_are_eligible_together():
    fit, evaluation = {"name": "C1"}, {"name": "MR1"}
    assert overlap.eligible_stages({}, [fit], [evaluation]) == [("eval", evaluation), ("fit", fit)]


def test_adopted_inline_child_is_never_restarted_or_interrupted():
    row = {"name": "N0"}
    assert overlap.eligible_stages({"adopted": row}, [row], [row]) == []


def test_capped_eval_oom_retries_only_when_fit_has_drained():
    retry = {"name": "MR1", "attempt": 2}
    assert overlap.eligible_stages({"fit": {"name": "C1"}}, [{}], [retry]) == []
    assert overlap.eligible_stages({}, [{}], [retry]) == [("eval", retry)]
    assert overlap.eligible_stages({"eval": retry, "serial_retry": True}, [{}], []) == []


def test_allocation_budget_keeps_headroom_and_no_batch_override(tmp_path):
    assert overlap.FIT_FRACTION + overlap.EVAL_FRACTION < 0.9
    row = {"name": "N0", "plan": {}}
    fit = overlap.stage_command(tmp_path, row, "fit")
    evaluation = overlap.stage_command(tmp_path, row, "eval")
    assert "set_per_process_memory_fraction" in fit[2]
    assert "--memory-fraction" in evaluation
    assert not any("batch" in part or "diagnostic" in part for part in evaluation)


def test_recycled_pid_is_not_adopted(monkeypatch):
    monkeypatch.setattr(overlap, "process_identity", lambda pid: {"pid": pid, "start_ticks": "2"})
    with pytest.raises(ValueError, match="PID identity"):
        overlap.process_alive({"pid": 42, "start_ticks": "1"})


def test_method_tree_is_native_source_config_and_probe_identity():
    digest = overlap.method_tree(ROOT)
    assert len(digest) == 64


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(overlap, "SERVER_ROOT", tmp_path)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", overlap.ALLOCATOR)
    monkeypatch.setattr(overlap, "verify_schedule", lambda *a: None)
    monkeypatch.setattr(overlap, "gpu_processes_owned", lambda *a: True)
    monkeypatch.setattr(overlap, "validate_training_stage", lambda p: {"epoch": 6})
    monkeypatch.setattr(overlap, "validate_completed", lambda p: {"run_id": p["run_id"]})
    monkeypatch.setattr(overlap.subprocess, "check_output", lambda *a, **kw: "0, 2\n1, 2\n")
    monkeypatch.setattr(overlap, "lock", lambda p: contextlib.nullcontext(0))
    monkeypatch.setattr(overlap, "LEASE_ROOT", tmp_path)
    directory = tmp_path / "queue"
    directory.mkdir()
    rows = []
    for i, name in enumerate(overlap.ARMS):
        plan = {
            "run_id": name,
            "run_root": str(tmp_path / name),
            "test_root": str(tmp_path / (name + "-test")),
            "repository_root": str(ROOT),
            "config_sha256": name,
            "source_commit": "a" * 40,
            "seed": 1,
        }
        rows.append({"name": name, "lane": str(i % 2), "mode": "deferred", "plan": plan})
    schedule = {"rows": rows}
    (directory / "schedule.json").write_text(json.dumps(schedule))
    events = []
    active = {}
    failures = set()

    class Process:
        def __init__(self, command, **kwargs):
            self.stage = "eval" if "--evaluate" in command else "fit"
            self.attempt = (
                int(command[command.index("--attempt") + 1]) if self.stage == "eval" else 1
            )
            path = command[command.index("--evaluate") + 1] if self.stage == "eval" else command[3]
            self.name = Path(path).name.removesuffix(".launch-plan.json")
            self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            self.pid = 100 + len(events)
            self.returncode = None
            self.polls = 0
            assert (self.gpu, self.stage) not in active
            active[self.gpu, self.stage] = self
            events.append(("start", self.stage, self.name, self.gpu, len(active)))

        def poll(self):
            self.polls += 1
            limit = 4 if self.stage == "eval" else 1
            if self.returncode is None and self.polls >= limit:
                failed = (self.stage, self.name) in failures or (
                    self.stage,
                    self.name,
                    self.attempt,
                ) in failures
                self.returncode = 1 if failed else 0
                if failed and self.stage == "eval" and self.attempt == 1:
                    root = tmp_path / self.name
                    root.mkdir(exist_ok=True)
                    (root / "POSTFIT_ATTEMPT1_FAILURE.json").write_text(json.dumps({"oom": True}))
                del active[self.gpu, self.stage]
                events.append(("finish", self.stage, self.name, self.gpu, len(active)))
            return self.returncode

    monkeypatch.setattr(overlap.subprocess, "Popen", Process)
    return schedule, directory, events, failures


def test_end_to_end_pipeline_overlaps_and_requires_all_ten_tests(simulation):
    schedule, directory, events, _ = simulation
    overlap.execute_schedule(schedule, directory, tick_seconds=0)
    for gpu in ("0", "1"):
        first_eval = next(
            i for i, e in enumerate(events) if e[:2] == ("start", "eval") and e[3] == gpu
        )
        eval_end = next(
            i for i, e in enumerate(events) if e[:2] == ("finish", "eval") and e[3] == gpu
        )
        assert any(
            e[:2] == ("start", "fit") and e[3] == gpu for e in events[first_eval + 1 : eval_end]
        )
    complete = json.loads((directory / "COMPLETE.json").read_text())
    assert len(complete["completed"]) == 10
    assert sum(e[:2] == ("finish", "eval") for e in events) == 10


def test_fit_failure_blocks_new_dispatch_and_drains_active_peer(simulation):
    schedule, directory, events, failures = simulation
    failures.add(("fit", "N0"))
    with pytest.raises(RuntimeError, match="active peers preserved"):
        overlap.execute_schedule(schedule, directory, tick_seconds=0)
    assert [e[2] for e in events if e[0] == "start"] == ["N0", "U24"]
    assert (directory / "FAILURE.json").exists()
    assert not (directory / "COMPLETE.json").exists()


def test_capped_eval_oom_has_unique_idle_retry_and_no_training_restart(simulation):
    schedule, directory, events, failures = simulation
    failures.add(("eval", "N0", 1))
    overlap.execute_schedule(schedule, directory, tick_seconds=0)
    assert (directory / "N0-eval-attempt1.exit.json").exists()
    assert (directory / "N0-eval-attempt2.exit.json").exists()
    assert sum(e[:3] == ("start", "fit", "N0") for e in events) == 1
    receipt = json.loads((directory / "COMPLETE.json").read_text())
    assert any(e["event"] == "oom_idle_retry_queued" for e in receipt["events"])
    assert len(receipt["completed"]) == 10


def test_resource_gate_rejects_unrelated_compute_process(monkeypatch):
    def output(command, **kwargs):
        if "--query-gpu=index,uuid" in command:
            return "0, GPU-a\n1, GPU-b\n"
        return "GPU-a, 42\nGPU-b, 99\n"

    monkeypatch.setattr(overlap.subprocess, "check_output", output)
    lane = {"fit": {"pid": 42}, "eval": None}
    assert overlap.gpu_processes_owned("0", lane)
    assert not overlap.gpu_processes_owned("1", lane)


@pytest.fixture
def migration(tmp_path, monkeypatch):
    monkeypatch.setattr(overlap, "SERVER_ROOT", tmp_path)
    previous = tmp_path / "old-queue"
    previous.mkdir()
    source = tmp_path / "new-source"
    source.mkdir()
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    rows = []
    for i, name in enumerate(overlap.ARMS):
        plan = {
            "run_id": name,
            "run_root": str(tmp_path / name),
            "test_root": str(tmp_path / (name + "-test")),
            "config": str(tmp_path / "old-source/configs" / name),
            "resolved_config": {"mechanism": name},
            "data_root": str(tmp_path / "data"),
            "genept_sha256": "g" * 64,
        }
        rows.append({"name": name, "lane": str(i % 2), "plan": plan})
    old = {"source": str(tmp_path / "old-source"), "rows": rows}
    (previous / "queue.json").write_text(json.dumps(old))
    (previous / "PREFLIGHT_COMPLETE.json").write_text("{}")
    lanes = {str(i): {"row": rows[i]["name"], "child_pid": i + 200} for i in (0, 1)}
    (previous / "state.json").write_text(
        json.dumps({"phase": "formal", "status": "running", "pid": 100, "lanes": lanes})
    )
    for row in rows[:2]:
        r = Path(row["plan"]["run_root"])
        r.mkdir()
        (r / "launch.json").write_text(json.dumps(row["plan"]))
    monkeypatch.setattr(overlap, "verify_seal", lambda *a: None)
    monkeypatch.setattr(
        overlap, "_preflight_entries", lambda *a: {r["name"]: {"receipt": r["name"]} for r in rows}
    )
    monkeypatch.setattr(overlap, "method_tree", lambda source: "same-tree")
    monkeypatch.setattr(
        overlap,
        "process_command",
        lambda pid: (
            b"python\0/src/scripts/v2/run_unified_group.py\0" + str(previous).encode() + b"\0"
        ),
    )
    signals = []
    retired = [False]

    def kill(pid, sig):
        signals.append((pid, sig))
        if sig == overlap.signal.SIGTERM:
            retired[0] = True

    monkeypatch.setattr(overlap.os, "kill", kill)
    monkeypatch.setattr(
        overlap,
        "process_identity",
        lambda pid: (
            None
            if pid == 100 and retired[0]
            else {"pid": pid, "command_sha256": "x", "start_ticks": "1"}
        ),
    )
    monkeypatch.setattr(overlap.subprocess, "check_output", lambda *a, **kw: "a" * 40 + "\n")

    def resolver(args):
        name = args.config.name
        original = copy.deepcopy(next(r["plan"] for r in rows if r["name"] == name))
        original.update(
            run_id=name + "-new",
            run_root=str(tmp_path / (name + "-new")),
            postfit_policy="deferred",
        )
        return original

    monkeypatch.setattr(overlap, "resolve_plan", resolver)
    return previous, tmp_path / "new-queue", runtime, source, signals


def test_migration_retires_only_controller_and_preserves_two_live_launches(migration):
    previous, output, runtime, source, signals = migration
    old_bytes = (previous / "queue.json").read_bytes()
    schedule = overlap.prepare_schedule(previous, output, runtime, source)
    assert all(pid == 100 for pid, _ in signals)
    assert (previous / "queue.json").read_bytes() == old_bytes
    assert [r["mode"] for r in schedule["rows"]][:2] == ["adopt_inline", "adopt_inline"]
    assert all(r["plan"] == r["parent_plan"] for r in schedule["rows"][:2])
    assert all(r["plan"]["run_id"] != r["parent_plan"]["run_id"] for r in schedule["rows"][2:])
    assert (output / "MIGRATION.json").exists()


def test_migration_error_unfreezes_controller_without_killing_children(migration, monkeypatch):
    previous, output, runtime, source, signals = migration
    monkeypatch.setattr(
        overlap, "resolve_plan", lambda *a: (_ for _ in ()).throw(ValueError("bad runtime"))
    )
    with pytest.raises(ValueError, match="bad runtime"):
        overlap.prepare_schedule(previous, output, runtime, source)
    assert signals == [(100, overlap.signal.SIGSTOP), (100, overlap.signal.SIGCONT)]
    assert not output.exists()
