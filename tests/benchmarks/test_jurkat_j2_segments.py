import pytest

from benchmarks.jurkat_j2 import study
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file


def test_only_same_checkpoint_can_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    identity = {"scenario": "J2", "model": "perturbdiff", "source_commit": "fit", "gpu": 0}
    receipt = {
        **identity,
        "checkpoint_sha256": sha256_file(checkpoint),
        "test_updates": 0,
    }
    atomic_json(tmp_path / "training_result.json", receipt)
    assert (
        study.frozen_receipt(
            tmp_path, {**identity, "gpu": 1, "source_commit": "evaluation"}, checkpoint.name
        )
        == receipt
    )
    with pytest.raises(ValueError, match="receipt or checkpoint changed"):
        study.frozen_receipt(tmp_path, {**identity, "scenario": "C2"}, checkpoint.name)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="receipt or checkpoint changed"):
        study.frozen_receipt(tmp_path, identity, checkpoint.name)


def test_budget_boundary_preserves_checkpoint_identity(tmp_path, monkeypatch):
    receipt = {"checkpoint_sha256": "frozen"}

    def budget_boundary(predict, output, *, deadline, journal_identity, full_axis, native_columns):
        assert journal_identity == "frozen"
        assert not full_axis and native_columns is None
        atomic_json(output / "evaluation_progress.json", {"done": 517, "total": 592})
        raise study.EvaluationBudgetExpired

    monkeypatch.setattr(study, "evaluate", budget_boundary)
    assert study.evaluate_segment(None, tmp_path, receipt, deadline=0) is None
    tag = read_json(tmp_path / "segment.json")
    assert tag["status"] == "evaluation_segment_complete"
    assert tag["conditions_completed"] == 517
    assert tag["checkpoint_sha256"] == "frozen"
    assert not (tmp_path / "result.json").exists()
