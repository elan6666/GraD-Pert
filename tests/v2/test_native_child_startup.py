"""Exercise the real launcher function with a transient empty /proc identity."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def launcher(tmp_path, identities, *, guard_failure=False, early_exit=False):
    script = Path(__file__).resolve().parents[2] / "scripts/v2/run_dataset_transfer_300.py"
    node = next(
        item
        for item in ast.parse(script.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "native_run"
    )
    events = []
    child = SimpleNamespace(pid=123, returncode=7 if early_exit else None)
    child.poll = lambda: child.returncode

    def wait(**kwargs):
        assert kwargs == {"timeout": 30}, "startup must never wait unbounded"
        events.append("wait")
        child.returncode = -15

    child.wait = wait
    child.terminate = lambda: events.append("terminate_direct")
    known = dict(pid=123, start_ticks="7", command_sha256="a" * 64)
    remaining = iter(identities)

    def identity(pid):
        assert pid == child.pid
        return next(remaining, None)

    def guard():
        events.append("guard")
        if guard_failure:
            raise RuntimeError("selected card rejected")

    def sleep(seconds):
        events.append(("sleep", seconds))
        if seconds == 15:
            child.returncode = 0

    def no_global(*args):
        pytest.fail("selected-device callback must replace global admission")

    namespace = {
        "SOURCE": tmp_path,
        "subprocess": SimpleNamespace(Popen=lambda *a, **kw: child, STDOUT=-2),
        "os": SimpleNamespace(environ={}, getpid=lambda: 17),
        "time": SimpleNamespace(sleep=sleep),
        "process_info": identity,
        "admit": no_global,
        "snapshot": no_global,
        "atomic_json": lambda path, value: events.append(("identity", value)),
        "terminate_tree": lambda pid, value: events.append(("terminate_tree", pid, value)),
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=[*future, node], type_ignores=[]), str(script), "exec"), namespace)
    return namespace["native_run"], guard, events, known


def test_transient_empty_identity_is_retried_under_guard(tmp_path):
    known = dict(pid=123, start_ticks="7", command_sha256="a" * 64)
    run, guard, events, _ = launcher(tmp_path, [None, None, known])
    run(["synthetic"], tmp_path / "worker.log", {}, admission_check=guard)
    assert events == [
        "guard",
        ("sleep", 0.05),
        "guard",
        ("sleep", 0.05),
        ("identity", known),
        "guard",
        ("sleep", 15),
    ]


def test_missing_identity_has_bounded_retries_and_stops_only_direct_child(tmp_path):
    run, guard, events, _ = launcher(tmp_path, [None])
    with pytest.raises(RuntimeError, match="bounded startup"):
        run(["synthetic"], tmp_path / "worker.log", {}, admission_check=guard)
    assert events.count("guard") == 20
    assert events.count(("sleep", 0.05)) == 20
    assert events[-2:] == ["terminate_direct", "wait"]
    assert not any(isinstance(e, tuple) and e[0] == "terminate_tree" for e in events)


def test_resource_rejection_during_identity_retry_stops_dispatch(tmp_path):
    run, guard, events, _ = launcher(tmp_path, [None], guard_failure=True)
    with pytest.raises(RuntimeError, match="selected card rejected"):
        run(["synthetic"], tmp_path / "worker.log", {}, admission_check=guard)
    assert events == ["guard", "terminate_direct", "wait"]


def test_early_exit_does_not_block_or_claim_identity(tmp_path):
    run, guard, events, _ = launcher(tmp_path, [None], early_exit=True)
    with pytest.raises(RuntimeError, match="exited before identity capture"):
        run(["synthetic"], tmp_path / "worker.log", {}, admission_check=guard)
    assert events == []


def test_known_identity_retains_descendant_cleanup(tmp_path):
    known = dict(pid=123, start_ticks="7", command_sha256="a" * 64)
    run, guard, events, _ = launcher(tmp_path, [known], guard_failure=True)
    with pytest.raises(RuntimeError, match="selected card rejected"):
        run(["synthetic"], tmp_path / "worker.log", {}, admission_check=guard)
    assert events == [("identity", known), "guard", ("terminate_tree", 123, known), "wait"]
