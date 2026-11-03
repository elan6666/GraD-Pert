"""Promotion cannot substitute a weaker probe or a negative main decision."""

import json
import runpy
from pathlib import Path

import pytest
from test_run_group import preflight

from gradpert.hashing import sha256_file


@pytest.fixture
def api(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/v2"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_dataset_transfer_300.py"))


def family(tmp_path):
    receipt, entry, plan = preflight(tmp_path)
    receipt.update(
        kind="preflight_only",
        steps_requested=10,
        steps_completed=10,
        inference_exercised=True,
        inference_shape=[300, 4883],
    )
    return receipt, entry, plan


def save(receipt, entry):
    Path(entry["receipt"]).write_text(json.dumps(receipt))
    entry["sha256"] = sha256_file(entry["receipt"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "integration_only"),
        ("steps_requested", 1),
        ("steps_completed", 11),
        ("inference_exercised", False),
        ("inference_shape", [1, 4883]),
    ],
)
def test_family_rejects_weaker_or_different_native_evidence(tmp_path, api, field, value):
    receipt, entry, plan = family(tmp_path)
    save(receipt, entry)
    api["validate_family_preflight"](plan, entry)
    receipt[field] = value
    save(receipt, entry)
    with pytest.raises(ValueError):
        api["validate_family_preflight"](plan, entry)


@pytest.mark.parametrize("approval", [False, None, 1, "true"])
def test_affirmative_main_acceptance_required(tmp_path, api, approval):
    _, _, plan = family(tmp_path)
    plan["run_id"] = "native-run"
    ready = tmp_path / "PREFLIGHT_READY.json"
    ready.write_text("{}")
    accepted = dict(
        approved=True,
        preflight_ready_sha256=sha256_file(ready),
        source_commit=plan["source_commit"],
        config_sha256=plan["config_sha256"],
        run_id=plan["run_id"],
        optimizer_updates=300,
    )
    api["validate_acceptance"](plan, ready, accepted)
    accepted["approved"] = approval
    with pytest.raises(ValueError):
        api["validate_acceptance"](plan, ready, accepted)


@pytest.mark.parametrize("guard_fails", [False, True])
def test_native_child_loop_rechecks_allocation_and_cleans_only_owned_child(
    tmp_path, api, monkeypatch, guard_fails
):
    from types import SimpleNamespace

    namespace = api["native_run"].__globals__
    events = []
    child = SimpleNamespace(pid=123, returncode=None)
    polls = iter([None, 0, 0] if not guard_fails else [None, None])

    def poll():
        value = next(polls)
        child.returncode = value
        return value

    child.poll = poll
    child.wait = lambda **kw: events.append("wait")
    monkeypatch.setattr(namespace["subprocess"], "Popen", lambda *a, **kw: child)
    monkeypatch.setitem(
        namespace,
        "process_info",
        lambda pid: dict(pid=pid, start_ticks="7", command_sha256="a" * 64),
    )
    monkeypatch.setitem(
        namespace,
        "terminate_tree",
        lambda pid, identity: events.append(("terminate", pid, identity["start_ticks"])),
    )
    monkeypatch.setitem(
        namespace, "admit", lambda *a: pytest.fail("explicit check must replace global admission")
    )
    monkeypatch.setattr(
        namespace["time"], "sleep", lambda duration: events.append(("sleep", duration))
    )

    def guard():
        events.append("guard")
        if guard_fails:
            raise RuntimeError("selected card guard")

    if guard_fails:
        with pytest.raises(RuntimeError, match="selected card guard"):
            api["native_run"](["synthetic"], tmp_path / "worker.log", {}, admission_check=guard)
        assert events == ["guard", ("terminate", 123, "7"), "wait"]
    else:
        api["native_run"](["synthetic"], tmp_path / "worker.log", {}, admission_check=guard)
        assert events == ["guard", ("sleep", 15)]
