import pytest

from gradpert.data._io import atomic_json, read_json
from gradpert.execution import v2


def test_failure_receipt_keeps_original_identity_and_error(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "SERVER_ROOT", tmp_path)
    plan = {"run_root": str(tmp_path), "source_commit": "a" * 40}
    atomic_json(tmp_path / "launch.json", plan)
    atomic_json(tmp_path / "run_manifest.json", {"source": {"commit": "a" * 40, "dirty": False}})

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(v2, "_run_v2", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        v2.run_v2(plan)
    receipt = read_json(tmp_path / "FAILURE.json")
    assert receipt["identity"]["source"]["dirty"] is False
    assert receipt["error_type"] == "RuntimeError"
    assert receipt["status"] == "failed"


def test_foreign_launch_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(v2, "SERVER_ROOT", tmp_path)
    atomic_json(tmp_path / "launch.json", {"run_id": "another-run"})

    def fail(*args, **kwargs):
        raise ValueError("foreign source")

    monkeypatch.setattr(v2, "_run_v2", fail)
    with pytest.raises(ValueError):
        v2.run_v2({"run_root": str(tmp_path)})
    assert not (tmp_path / "FAILURE.json").exists()
