"""Resource caps affect admission only; truncated probes never count as tests."""

import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from gradpert.data._io import atomic_json
from gradpert.execution.v2_checkpoint_eval import (
    diagnostic_conditions,
    evaluation_memory_fraction,
    execute_evaluation_plan,
)
from gradpert.hashing import sha256_file


@pytest.mark.parametrize("value", [0, -0.1, 1.1, True, "0.2", float("nan")])
def test_reject_invalid_allocation_budget(value):
    with pytest.raises(ValueError):
        evaluation_memory_fraction({"cuda_memory_fraction": value})


def test_default_has_no_cap_and_no_subset():
    assert evaluation_memory_fraction({}) is None
    assert evaluation_memory_fraction({"cuda_memory_fraction": 0.19}) == 0.19
    assert diagnostic_conditions({}, ("a", "b")) == ("a", "b")


def test_probe_is_explicit_and_cannot_write_scientific_receipt():
    plan = {"diagnostic_only": True, "diagnostic_condition_count": 1}
    assert diagnostic_conditions(plan, ("a", "b")) == ("a",)
    with pytest.raises(ValueError, match="COMPLETE"):
        execute_evaluation_plan(plan)
    with pytest.raises(ValueError):
        diagnostic_conditions({"diagnostic_condition_count": 1}, ("a", "b"))


def adopted_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
    import adopt_train_evaluation

    return adopt_train_evaluation


def test_reused_process_is_never_adopted(monkeypatch):
    module = adopted_module()
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"another-user\0")
    with pytest.raises(ValueError, match="reused"):
        module.verified_process(123, hashlib.sha256(b"our-fit\0").hexdigest())


def test_adoption_evaluates_u4_before_waiting_for_existing_u3(tmp_path, monkeypatch):
    module = adopted_module()
    monkeypatch.setattr(Path, "is_relative_to", lambda *args: True)
    old = tmp_path / "old"
    atomic_json(old / "queue.json", {})
    rows = []
    for name in ("U4", "U3"):
        launch = tmp_path / f"{name}.json"
        atomic_json(launch, {"run_root": str(tmp_path / name), "name": name})
        rows.append({"launch": str(launch), "launch_sha256": sha256_file(launch)})
    fit_process_checks = iter([True, False])
    events = []

    def process(pid, sha):
        if pid == 1:
            return False
        events.append("check_existing_U3")
        return next(fit_process_checks)

    @contextmanager
    def lease(path):
        yield

    monkeypatch.setattr(module, "verified_process", process)
    monkeypatch.setattr(module, "validate_training_stage", lambda plan: {})
    monkeypatch.setattr(module, "lock", lease)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kw: "0, 2\n1, 2")
    monkeypatch.setattr(
        module, "run_deferred_postfit", lambda plan, **kw: events.append(plan["name"] + "eval")
    )
    schedule = {
        "output": str(tmp_path / "output"),
        "old_queue": str(old),
        "old_queue_sha256": sha256_file(old / "queue.json"),
        "retired_controller_pid": 1,
        "controller_cmdline_sha256": "old",
        "fit_pid": 2,
        "fit_cmdline_sha256": "fit",
        "rows": rows,
        "shared_evaluation_gpu": "0",
        "evaluation_memory_fraction": 0.19,
        "evaluation_runtime": "new-release",
    }
    module.execute(schedule)
    assert events == ["U4eval", "check_existing_U3", "check_existing_U3", "U3eval"]
    assert (tmp_path / "output/COMPLETE.json").exists()
