"""Preserve a live predecessor and refuse missing/failed terminal evidence."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))
import run_cap40_ablations as controller

from gradpert.hashing import sha256_file


def dependency(tmp_path):
    root = tmp_path / "old-run"
    root.mkdir()
    plan = {"run_id": "original-U2", "run_root": str(root)}
    (root / "launch.json").write_text(json.dumps(plan))
    command = Path(f"/proc/{os.getpid()}/cmdline")
    cmd = command.read_bytes().decode().replace("\x00", " ") if command.exists() else ""
    return root, {
        "plan": plan,
        "launch_sha256": sha256_file(root / "launch.json"),
        "pid": os.getpid(),
        "pid_cmdline_sha256": hashlib.sha256(cmd.encode()).hexdigest(),
    }


def test_dependency_rejects_nonserver_path(tmp_path):
    _, dep = dependency(tmp_path)
    with pytest.raises(ValueError, match="server"):
        controller.wait_dependency(dep, lambda **kw: None)


def test_completion_gate_waits_then_checks_exact_terminal(tmp_path, monkeypatch):
    root, dep = dependency(tmp_path)
    monkeypatch.setattr(Path, "is_relative_to", lambda *args: True)
    records, sleeps, checked = [], [], []

    def terminal(plan):
        checked.append(plan)
        return "skip_complete"

    monkeypatch.setattr(controller, "next_action", terminal)

    def finish(seconds):
        sleeps.append(seconds)
        (root / "COMPLETE.json").write_text("{}")

    controller.wait_dependency(dep, lambda **kw: records.append(kw), sleep=finish)
    assert len(records) == 1 and sleeps == [60] and checked == [dep["plan"]]
    assert records[0]["phase"] == "waiting_for_dependency"


def test_failed_or_invalid_completion_blocks_followup(tmp_path, monkeypatch):
    root, dep = dependency(tmp_path)
    monkeypatch.setattr(Path, "is_relative_to", lambda *args: True)
    (root / "FAILURE.json").write_text("{}")
    with pytest.raises(RuntimeError, match="failed"):
        controller.wait_dependency(dep, lambda **kw: None)
    (root / "FAILURE.json").unlink()
    (root / "COMPLETE.json").write_text("{}")
    monkeypatch.setattr(controller, "next_action", lambda plan: "resume")
    with pytest.raises(ValueError, match="terminal evidence"):
        controller.wait_dependency(dep, lambda **kw: None)


def test_missing_pid_is_not_completion(tmp_path, monkeypatch):
    _, dep = dependency(tmp_path)
    monkeypatch.setattr(Path, "is_relative_to", lambda *args: True)
    dep["pid"] = 987654321
    with pytest.raises(RuntimeError, match="without terminal"):
        controller.wait_dependency(dep, lambda **kw: None, sleep=lambda seconds: None)


def test_changed_launch_or_reused_pid_is_rejected(tmp_path, monkeypatch):
    root, dep = dependency(tmp_path)
    monkeypatch.setattr(Path, "is_relative_to", lambda *args: True)
    dep["pid_cmdline_sha256"] = "wrong"
    if Path(f"/proc/{os.getpid()}/cmdline").exists():
        with pytest.raises(ValueError, match="reused"):
            controller.wait_dependency(dep, lambda **kw: None)
    (root / "launch.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        controller.wait_dependency(dep, lambda **kw: None)
