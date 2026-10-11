"""The independent holder never signals BR normally or dispatches later arms."""

import os
import signal
import sys
from pathlib import Path

import pytest

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
import hold_after_br as holder


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def identity(pid):
        return {"pid": pid, "start_ticks": str(pid), "command_sha256": "hash"}

    parent, fit, gm = identity(101), identity(102), identity(103)
    qroot, groot = tmp_path / "queue", tmp_path / "gm"
    qroot.mkdir()
    groot.mkdir()
    state, gstate = {"status": "running", "active": {"fit": "BR"}}, {"phase": "waiting_dependency"}
    atomic_json(qroot / "state.json", state)
    atomic_json(groot / "state.json", gstate)
    atomic_json(qroot / "queue.json", {})
    atomic_json(groot / "queue.json", {})
    plan = {
        "schema": holder.SCHEMA,
        "queue": {"path": str(qroot / "queue.json"), "sha256": sha256_file(qroot / "queue.json")},
        "controller": parent,
        "fit": fit,
        "gm1": {
            "controller": gm,
            "queue": {
                "path": str(groot / "queue.json"),
                "sha256": sha256_file(groot / "queue.json"),
            },
        },
        "output": str(tmp_path / "holder"),
    }
    checkpoint = tmp_path / "run/fit/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"committed epoch4")
    launch = {"run_root": str(checkpoint.parents[1])}
    queue = {"shared_gpu_policy": {"sealed": True}}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(holder, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(holder, "validate_setup", lambda _: (queue, launch, state, gstate))
    alive, calls = {101: True, 102: True, 103: True}, []
    monkeypatch.setattr(
        holder,
        "process_info",
        lambda pid: {**identity(pid), "uid": os.getuid()} if alive[pid] else None,
    )
    monkeypatch.setattr(holder, "process_state", lambda _: "T")
    monkeypatch.setattr(holder.os, "getsid", lambda _: 102)
    monkeypatch.setattr(holder, "session_members", lambda _: [])
    monkeypatch.setattr(holder, "parent_map", lambda: {})
    monkeypatch.setattr(holder, "fit_exit_code", lambda _: 0)
    monkeypatch.setattr(holder, "snapshot", lambda _: {"guard": True})
    monkeypatch.setattr(holder, "admit", lambda *_: None)
    monkeypatch.setattr(
        holder,
        "validate_training_stage",
        lambda _: {"epoch": 4, "last": {"epoch": 4, "sha256": sha256_file(checkpoint)}},
    )

    def kill(pid, sig):
        calls.append((pid, sig))
        if sig == signal.SIGKILL:
            alive[pid] = False

    monkeypatch.setattr(holder.os, "kill", kill)
    monkeypatch.setattr(holder.time, "sleep", lambda _: alive.update({102: False}))
    return plan, alive, calls, qroot, groot


def test_normal_finish_never_signals_br_and_keeps_evaluation_pending(prepared):
    plan, _alive, calls, qroot, groot = prepared
    before = (qroot / "state.json").read_bytes(), (groot / "state.json").read_bytes()
    holder.execute(plan)
    assert not any(pid == 102 for pid, _sig in calls)
    assert not any(sig == signal.SIGCONT for _pid, sig in calls)
    receipt = read_json(Path(plan["output"]) / "READY.json")
    assert receipt["evaluation_pending"] and not receipt["scientific_complete"]
    assert ((qroot / "state.json").read_bytes(), (groot / "state.json").read_bytes()) == before


def test_missing_training_receipt_after_fit_exit_retires_without_cont(prepared, monkeypatch):
    plan, _alive, calls, _q, _g = prepared
    monkeypatch.setattr(
        holder,
        "validate_training_stage",
        lambda _: (_ for _ in ()).throw(FileNotFoundError("TRAIN_COMPLETE missing")),
    )
    with pytest.raises(FileNotFoundError):
        holder.execute(plan)
    assert not any(sig == signal.SIGCONT for _pid, sig in calls)
    assert (101, signal.SIGKILL) in calls and (103, signal.SIGKILL) in calls


def test_guard_failure_terminates_only_exact_br_and_retires_parents(prepared, monkeypatch):
    plan, alive, calls, _q, _g = prepared
    count, terminated = [], []

    def admit(*_):
        count.append(1)
        if len(count) == 2:
            raise RuntimeError("foreign resource failure")

    monkeypatch.setattr(holder, "admit", admit)
    monkeypatch.setattr(
        holder,
        "terminate_tree",
        lambda pid, identity: (terminated.append((pid, identity)), alive.update({pid: False})),
    )
    with pytest.raises(RuntimeError, match="foreign resource"):
        holder.execute(plan)
    assert terminated == [(102, plan["fit"])]
    assert not any(sig == signal.SIGCONT for _pid, sig in calls)


def test_pause_race_resumes_guarded_live_br_without_br_signal(prepared, monkeypatch):
    plan, _alive, calls, qroot, _g = prepared
    original = holder.pause

    def pause(identity):
        original(identity)
        if identity["pid"] == 103:
            atomic_json(qroot / "state.json", {"other_arm": "F1"})

    monkeypatch.setattr(holder, "pause", pause)
    with pytest.raises(ValueError, match="raced"):
        holder.execute(plan)
    assert (101, signal.SIGCONT) in calls and (103, signal.SIGCONT) in calls
    assert not any(pid == 102 for pid, _sig in calls)


def test_interrupt_returns_live_br_guard_to_exact_cpu_parents(prepared, monkeypatch):
    plan, _alive, calls, _q, _g = prepared
    monkeypatch.setattr(holder.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        holder.execute(plan)
    assert (101, signal.SIGCONT) in calls and (103, signal.SIGCONT) in calls


def test_recycled_identity_never_receives_signal(prepared, monkeypatch):
    plan, _alive, calls, _q, _g = prepared
    monkeypatch.setattr(
        holder,
        "process_info",
        lambda pid: {
            "pid": pid,
            "start_ticks": "recycled",
            "command_sha256": "hash",
            "uid": os.getuid(),
        },
    )
    with pytest.raises(ValueError, match="identity changed"):
        holder.direct(plan["controller"], signal.SIGSTOP)
    assert not calls


def test_holder_refuses_visible_cuda(prepared, monkeypatch):
    plan, _alive, calls, _q, _g = prepared
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        holder.execute(plan)
    assert not calls


def test_raced_f1_after_br_exit_returns_original_guard(prepared, monkeypatch):
    plan, alive, calls, qroot, _g = prepared
    original = holder.pause

    def pause(identity):
        original(identity)
        if identity["pid"] == 103:
            alive[102] = False
            atomic_json(qroot / "state.json", {"active": {"fit": {"name": "F1"}}})

    monkeypatch.setattr(holder, "pause", pause)
    with pytest.raises(ValueError, match="raced"):
        holder.execute(plan)
    assert (101, signal.SIGCONT) in calls and (103, signal.SIGCONT) in calls
    assert not any(sig == signal.SIGKILL for _pid, sig in calls)


def test_sealed_rank_in_independent_session_is_cleaned_on_launcher_loss(prepared, monkeypatch):
    plan, alive, calls, _q, _g = prepared
    alive[104] = True
    monkeypatch.setattr(holder, "parent_map", lambda: {104: 102, 102: 101})
    terminated = []
    monkeypatch.setattr(
        holder,
        "terminate_tree",
        lambda pid, _identity: (terminated.append(pid), alive.update({pid: False})),
    )
    with pytest.raises(RuntimeError, match="live sealed descendants"):
        holder.execute(plan)
    assert terminated == [104]
    assert not any(sig == signal.SIGCONT for _pid, sig in calls)


def test_raced_f1_handoff_first_cleans_exact_orphan_br_rank(prepared, monkeypatch):
    plan, alive, calls, qroot, _g = prepared
    alive[104] = True
    monkeypatch.setattr(holder, "parent_map", lambda: {104: 102, 102: 101})
    original = holder.pause
    terminated = []

    def pause(identity):
        original(identity)
        if identity["pid"] == 103:
            alive[102] = False
            atomic_json(qroot / "state.json", {"active": {"fit": {"name": "F1"}}})

    monkeypatch.setattr(holder, "pause", pause)
    monkeypatch.setattr(
        holder,
        "terminate_tree",
        lambda pid, _identity: (terminated.append(pid), alive.update({pid: False})),
    )
    with pytest.raises(ValueError, match="raced"):
        holder.execute(plan)
    assert terminated == [104]
    assert (101, signal.SIGCONT) in calls


def test_failed_setup_with_missing_parent_state_never_signals(prepared, monkeypatch):
    plan, _alive, calls, qroot, _g = prepared
    (qroot / "state.json").unlink()
    terminated = []
    monkeypatch.setattr(
        holder,
        "validate_setup",
        lambda _: (_ for _ in ()).throw(ValueError("setup failed before ownership")),
    )
    monkeypatch.setattr(holder, "terminate_tree", lambda *_: terminated.append(True))
    with pytest.raises(ValueError, match="before ownership"):
        holder.execute(plan)
    assert not calls and not terminated
