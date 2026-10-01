"""Resource and stale-preflight boundaries for the approved cap40 queue."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
spec = importlib.util.spec_from_file_location(
    "cap40_queue", ROOT / "scripts/v2/run_cap40_ablations.py"
)
assert spec and spec.loader
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.mark.parametrize("usage", ["0, 2\n1, 728\n", "0, 2\n", "0, 513\n1, 2\n", ""])
def test_waits_when_either_gpu_is_occupied_or_unknown(usage):
    assert not queue.idle_devices(usage)


def test_idle_boundary_and_extra_gpu_do_not_override_selected_devices():
    assert queue.idle_devices("0, 512\n1, 2\n2, 9999\n\n")


def test_capacity_failure_is_not_launch_permission(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"kind": "capacity_only", "status": "failed", "steps_completed": 127})
    )
    entry = {"receipt": str(receipt), "sha256": queue.sha256_file(receipt)}
    with pytest.raises(ValueError, match="passed"):
        queue.validate_preflight({}, entry)


def test_one_step_capacity_cannot_pass_as_sustained_capacity(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "kind": "capacity_only",
                "status": "passed",
                "steps_completed": 1,
                "resume_checkpoint_sha256": "x",
            }
        )
    )
    entry = {"receipt": str(receipt), "sha256": queue.sha256_file(receipt)}
    with pytest.raises(ValueError, match="update/checkpoint"):
        queue.validate_preflight({}, entry)


def test_incomplete_baseline_is_rejected_before_planning(tmp_path):
    (tmp_path / "fit").mkdir()
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"epoch": 5, "zero_pkl": True}))
    (tmp_path / "fit/epoch_state.json").write_text(json.dumps({"epoch": 5}))
    with pytest.raises(ValueError, match="prerequisite"):
        queue.prepare(Path("unused"), tmp_path)


def test_unapproved_config_hash_is_rejected_before_resolving(tmp_path, monkeypatch):
    (tmp_path / "fit").mkdir()
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"epoch": 6, "zero_pkl": True}))
    (tmp_path / "fit/epoch_state.json").write_text(json.dumps({"epoch": 6}))
    monkeypatch.setattr(queue, "sha256_file", lambda path: "changed")
    with pytest.raises(ValueError, match="configuration changed"):
        queue.prepare(Path("unused"), tmp_path)


def test_existing_run_never_enters_fresh_formal_launch(tmp_path):
    plan = {"run_root": str(tmp_path)}
    with pytest.raises(ValueError, match="saved launch plan"):
        queue.require_fresh_formal(plan)


def test_valid_committed_run_is_not_implicitly_resumed(tmp_path):
    plan = {"run_root": str(tmp_path)}
    (tmp_path / "launch.json").write_text(json.dumps(plan))
    (tmp_path / "fit").mkdir()
    (tmp_path / "fit/epoch_state.json").write_text("{}")
    with pytest.raises(ValueError, match="cannot resume"):
        queue.require_fresh_formal(plan)
