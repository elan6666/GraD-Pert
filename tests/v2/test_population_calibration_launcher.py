"""Synthetic lifecycle tests; no subprocess, GPU, server I/O or target arrays."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))

from gradpert.hashing import sha256_file  # noqa: E402
from scripts.v2 import run_population_calibration as launcher  # noqa: E402


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=sha256_file(path))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "source"
    script = root / "scripts/v2/run_population_calibration.py"
    script.parent.mkdir(parents=True)
    monkeypatch.setattr(launcher, "__file__", str(script))
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    original_relative = Path.is_relative_to
    monkeypatch.setattr(
        Path,
        "is_relative_to",
        lambda self, other: (
            self.is_relative_to(tmp_path)
            if str(other) == "/data/yilangliu"
            else original_relative(self, other)
        ),
    )
    output = tmp_path / "output"
    spec = dict(
        schema="population-initialization-calibration-v1",
        run_id="synthetic-first8",
        source_commit="a" * 40,
        config=write(root / "config.json", {}),
        publication=write(tmp_path / "publication.json", {}),
        policy=write(tmp_path / "policy.json", {}),
        data_root=str(tmp_path / "data"),
        output=str(output),
    )
    binding = write(tmp_path / "SPEC.json", spec)
    events, terminations, clock, processes = [], [], [0.0], {}
    owner = dict(
        pid=os.getpid(),
        uid=os.getuid(),
        start_ticks="owner",
        command_sha256="owner-hash",
        script_basename="run_population_calibration.py",
    )

    def identity(pid, ppid):
        return dict(
            pid=pid,
            ppid=ppid,
            uid=os.getuid(),
            start_ticks=str(pid),
            command_sha256=f"hash-{pid}",
            script_basename="calibrate_response_anchor.py",
        )

    child_identity = identity(20001, owner["pid"])
    processes[owner["pid"]] = owner
    # A same-UID but unrelated process must never enter the cleanup set.
    processes[29999] = identity(29999, 1)
    options = SimpleNamespace(
        response_anchor_loss="population_prediction", response_anchor_weight=1.0
    )
    state = SimpleNamespace(
        output=output,
        spec=spec,
        binding=binding,
        events=events,
        terminations=terminations,
        options=options,
        processes=processes,
        clock=clock,
        owner=owner,
        root=root,
        verified=True,
        observed_identity=child_identity,
        orphan=False,
        fail_resource=False,
        wait_error=False,
        terminate_error_pid=None,
        on_spawn=None,
        source_changed=False,
        on_sleep=None,
        native_wrong=False,
        exitcode=0,
        sleeps=0,
        admissions=0,
    )

    def checked(seal):
        path = Path(seal["path"])
        if sha256_file(path) != seal["sha256"]:
            raise ValueError("immutable input changed")
        return path

    monkeypatch.setattr(launcher, "checked", checked)
    monkeypatch.setattr(
        launcher,
        "load_experiment_config",
        lambda _: SimpleNamespace(
            model=SimpleNamespace(parameters={}),
            source_code=SimpleNamespace(repository="synthetic"),
        ),
    )
    monkeypatch.setattr(launcher.V2Options, "parse_parameters", lambda _: ({}, options))
    source_calls = []

    def source(*args, **kwargs):
        source_calls.append((args, kwargs))
        return SimpleNamespace(
            commit="b" * 40 if state.source_changed and len(source_calls) > 1 else "a" * 40
        )

    monkeypatch.setattr(launcher, "inspect_source_identity", source)
    monkeypatch.setattr(launcher, "process_info", lambda pid: processes.get(pid))
    monkeypatch.setattr(
        launcher, "parent_map", lambda: {pid: p.get("ppid", 1) for pid, p in processes.items()}
    )
    snapshots = []

    def snapshot(owner_pid):
        snapshots.append(owner_pid)
        return {"owner_pid": owner_pid}

    def admit(policy, observed):
        assert observed["owner_pid"] == owner["pid"]
        state.admissions += 1
        if state.fail_resource and state.admissions > 1:
            raise RuntimeError("synthetic resource rejection")

    monkeypatch.setattr(launcher, "snapshot", snapshot)
    monkeypatch.setattr(launcher, "admit", admit)
    monkeypatch.setattr(launcher, "DRAIN_SECONDS", 2.0)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock[0])

    class Child:
        pid = child_identity["pid"]
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, **kwargs):
            events.append(("wait", kwargs))
            if state.wait_error:
                raise RuntimeError("synthetic wait failure")
            if self.returncode is None:
                raise subprocess.TimeoutExpired("synthetic-child", kwargs.get("timeout", 0))
            processes.pop(self.pid, None)
            return self.returncode

    child = Child()
    state.child = child

    def spawn(out, label, command, environment, inspect, expected_owner, *, cwd):
        events.append(("spawn", command, environment, expected_owner, cwd))
        assert out == output and expected_owner == owner and cwd == root
        # The native calibrator itself is simulated, not invoked.
        processes[child.pid] = child_identity.copy()
        processes[20002] = identity(20002, child.pid)
        processes[20003] = identity(20003, child.pid)
        write(
            output / "calibration/COMPLETE.json",
            dict(
                run_id=spec["run_id"],
                response_anchor_loss="mse" if state.native_wrong else "population_prediction",
                source=dict(commit=spec["source_commit"]),
                calibration_complete=True,
                state_restored=True,
                world_size=2,
                optimizer_updates=0,
                teacher_updates=0,
                center_updates=0,
            ),
        )
        started = dict(
            label=label,
            process_identity_verified=state.verified,
            actual_process_identity=state.observed_identity,
            owner_process=owner,
        )
        write(output / f"{label}.STARTED.json", started)
        if state.on_spawn:
            state.on_spawn()
        return child, started

    def sleep(seconds):
        clock[0] += seconds
        state.sleeps += 1
        child.returncode = state.exitcode
        if not state.orphan:
            processes.pop(20002, None)
            processes.pop(20003, None)
        if state.on_sleep:
            state.on_sleep()

    def terminate(pid, expected):
        assert pid != 29999 and processes[pid] == expected
        terminations.append(pid)
        if pid == state.terminate_error_pid:
            raise RuntimeError("synthetic termination failure")
        processes.pop(pid, None)
        if pid == child.pid:
            child.returncode = -15
            processes.pop(20002, None)
            processes.pop(20003, None)

    monkeypatch.setattr(launcher, "spawn_phase", spawn)
    monkeypatch.setattr(launcher.time, "sleep", sleep)
    monkeypatch.setattr(launcher, "terminate_tree", terminate)
    state.run = lambda: launcher.execute(Path(binding["path"]), binding["sha256"])
    return state


def test_success_requires_actual_wait_and_owner_bound_launch(sandbox):
    s = sandbox
    s.run()
    done = json.loads((s.output / "COMPLETE.json").read_text())
    exit_receipt = json.loads((s.output / "native-calibration.EXIT.json").read_text())
    assert done["observed_owned_processes_drained"] and done["scientific_complete"] is False
    assert exit_receipt["poll_returncode"] == exit_receipt["wait_returncode"] == 0
    assert exit_receipt["wait_called"] and s.terminations == []
    spawns = [e for e in s.events if e[0] == "spawn"]
    assert len(spawns) == 1 and spawns[0][1][spawns[0][1].index("--resource-owner-pid") + 1] == str(
        s.owner["pid"]
    )
    assert spawns[0][2]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert spawns[0][1][spawns[0][1].index("2") + 1].endswith("calibrate_response_anchor.py")
    assert {p["pid"] for p in done["observed_process_identities"]} == {20001, 20002, 20003}


@pytest.mark.parametrize("verified_identity", [None, {"pid": 20001, "uid": 999}])
def test_unverified_child_is_rejected_immediately_and_never_signalled(sandbox, verified_identity):
    s = sandbox
    s.verified = False
    s.observed_identity = verified_identity
    with pytest.raises(RuntimeError, match="identity"):
        s.run()
    failed = json.loads((s.output / "FAILURE.json").read_text())
    assert s.admissions == 1 and s.sleeps == 0 and s.terminations == []
    assert (
        failed["unverified_child_never_signalled"] and failed["observed_process_identities"] == []
    )
    assert "wait_returncode" not in failed["child_exit_observation"]
    assert not (s.output / "COMPLETE.json").exists()


def test_resource_rejection_drains_only_verified_owned_tree_and_preserves_exit(sandbox):
    s = sandbox
    s.fail_resource = True
    with pytest.raises(RuntimeError, match="resource rejection"):
        s.run()
    failed = json.loads((s.output / "FAILURE.json").read_text())
    actual = json.loads((s.output / "native-calibration.EXIT.json").read_text())
    assert set(s.terminations) == {20001, 20002, 20003} and 29999 in s.processes
    assert failed["remaining_identity_matched_owned"] == []
    assert actual["wait_returncode"] == -15 and actual["status"] == "failed"


def test_zero_exit_with_live_observed_rank_cannot_write_success(sandbox):
    s = sandbox
    s.orphan = True
    with pytest.raises(RuntimeError, match="descendants remain"):
        s.run()
    failed = json.loads((s.output / "FAILURE.json").read_text())
    actual = json.loads((s.output / "native-calibration.EXIT.json").read_text())
    assert actual["wait_returncode"] == 0  # Real root success does not prove rank closure.
    assert set(s.terminations) == {20002, 20003} and 20001 not in s.terminations
    assert failed["remaining_identity_matched_owned"] == []
    assert not (s.output / "COMPLETE.json").exists()


def test_nonzero_native_exit_is_not_overridden_by_existing_complete_file(sandbox):
    s = sandbox
    s.exitcode = 7
    with pytest.raises(RuntimeError, match="native calibration failed"):
        s.run()
    actual = json.loads((s.output / "native-calibration.EXIT.json").read_text())
    assert actual["wait_returncode"] == 7 and actual["status"] == "failed"
    assert (s.output / "FAILURE.json").exists() and not (s.output / "COMPLETE.json").exists()


def test_termination_and_wait_errors_still_seal_failure_without_fake_exit(sandbox):
    s = sandbox
    s.fail_resource = True
    s.terminate_error_pid = 20003
    s.wait_error = True
    with pytest.raises(RuntimeError, match="resource rejection"):
        s.run()
    failed = json.loads((s.output / "FAILURE.json").read_text())
    assert any("termination failure" in e for e in failed["cleanup_errors"])
    assert any("wait failure" in e for e in failed["cleanup_errors"])
    assert failed["child_exit_observation"]["poll_returncode"] == -15
    assert "wait_returncode" not in failed["child_exit_observation"]
    assert not (s.output / "native-calibration.EXIT.json").exists()
    assert not (s.output / "COMPLETE.json").exists()


def test_mutated_sealed_input_stops_verified_tree(sandbox):
    s = sandbox
    s.on_spawn = lambda: Path(s.spec["policy"]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="immutable input"):
        s.run()
    assert set(s.terminations) == {20001, 20002, 20003}
    assert (s.output / "FAILURE.json").exists() and not (s.output / "COMPLETE.json").exists()


@pytest.mark.parametrize("problem", ["mode", "unit_weight", "allocator", "data_subtree"])
def test_prelaunch_contract_rejections_create_no_output_or_child(sandbox, monkeypatch, problem):
    s = sandbox
    if problem == "mode":
        s.options.response_anchor_loss = "cosine_direction"
    elif problem == "unit_weight":
        s.options.response_anchor_weight = 0.1
    elif problem == "allocator":
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "other")
    else:
        s.spec["output"] = str(Path(s.spec["data_root"]) / "do-not-create")
        s.binding.update(write(Path(s.binding["path"]), s.spec))
    with pytest.raises(ValueError):
        s.run()
    assert not any(e[0] == "spawn" for e in s.events)
    assert not Path(s.spec["output"]).exists()


def test_source_or_native_identity_change_cannot_complete(sandbox):
    s = sandbox
    s.source_changed = True
    with pytest.raises(ValueError, match="source changed"):
        s.run()
    assert (s.output / "FAILURE.json").exists() and not (s.output / "COMPLETE.json").exists()


def test_delayed_rank_exit_is_observed_before_success(sandbox):
    s = sandbox
    s.orphan = True

    def ranks_exit():
        if s.sleeps == 2:
            s.processes.pop(20002, None)
            s.processes.pop(20003, None)

    s.on_sleep = ranks_exit
    s.run()
    assert s.sleeps == 2 and s.terminations == []
    assert json.loads((s.output / "COMPLETE.json").read_text())["observed_owned_processes_drained"]


def test_recycled_rank_pid_is_never_signalled_as_the_old_owned_process(sandbox):
    s = sandbox
    s.orphan = True

    def recycle():
        if s.sleeps == 1:
            s.processes[20002] = dict(s.processes[20002], start_ticks="new-unrelated-process")

    s.on_sleep = recycle
    with pytest.raises(RuntimeError, match="descendants remain"):
        s.run()
    assert s.terminations == [20003]
    assert s.processes[20002]["start_ticks"] == "new-unrelated-process"
    assert (
        json.loads((s.output / "FAILURE.json").read_text())["remaining_identity_matched_owned"]
        == []
    )


def test_native_mode_mismatch_is_rejected_even_with_clean_zero_exit(sandbox):
    s = sandbox
    s.native_wrong = True
    with pytest.raises(ValueError, match="closure differs"):
        s.run()
    actual = json.loads((s.output / "native-calibration.EXIT.json").read_text())
    assert actual["wait_returncode"] == 0
    assert (s.output / "FAILURE.json").exists() and not (s.output / "COMPLETE.json").exists()
