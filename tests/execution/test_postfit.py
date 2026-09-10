import json
from copy import deepcopy

import pytest

pytest.importorskip("torch")

from gradpert.execution.postfit import checkpoint_progress, verify_existing
from gradpert.hashing import sha256_file


def fixture():
    identity = {
        k: "a" * 64
        for k in (
            "source_tree_sha256",
            "config_sha256",
            "environment_sha256",
            "canonical_data_sha256",
            "split_content_sha256",
        )
    }
    identity["source_commit"] = "b" * 40
    checkpoint = {
        "schema_version": "gradpert-training-checkpoint-v2",
        "identity": identity,
        "progress": {"completed_epochs": 50, "global_step": 29100, "test_evaluations": 0},
    }
    meta = {"source": {"tree_sha256": "a" * 64}, "max_epochs": 50, "steps_per_epoch": 582}
    return checkpoint, dict(identity), meta


def test_checkpoint_epoch_and_identity():
    checkpoint, manifest, meta = fixture()
    assert checkpoint_progress(checkpoint, manifest, meta) == 50
    checkpoint["progress"].update(completed_epochs=40, global_step=23280)
    assert checkpoint_progress(checkpoint, manifest, meta) == 40


@pytest.mark.parametrize(
    "key",
    [
        "source_commit",
        "config_sha256",
        "environment_sha256",
        "canonical_data_sha256",
        "split_content_sha256",
    ],
)
def test_foreign_checkpoint_rejected(key):
    checkpoint, manifest, meta = fixture()
    checkpoint = deepcopy(checkpoint)
    checkpoint["identity"][key] = "foreign"
    with pytest.raises(ValueError, match="identity"):
        checkpoint_progress(checkpoint, manifest, meta)


@pytest.mark.parametrize(
    "change",
    [
        dict(test_evaluations=1),
        dict(completed_epochs=0),
        dict(global_step=29099),
        dict(completed_epochs=51),
    ],
)
def test_invalid_progress_rejected(change):
    checkpoint, manifest, meta = fixture()
    checkpoint["progress"].update(change)
    with pytest.raises(ValueError):
        checkpoint_progress(checkpoint, manifest, meta)


def test_existing_complete_is_idempotent_but_detects_tamper(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"value": 0.3}')
    receipt = {"output_sha256": {"metrics.json": sha256_file(metrics)}}
    (tmp_path / "COMPLETE.json").write_text(json.dumps(receipt))
    assert verify_existing(tmp_path) == receipt
    metrics.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        verify_existing(tmp_path)


def test_started_without_complete_cannot_repeat_test(tmp_path):
    (tmp_path / "STARTED.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        verify_existing(tmp_path)


def test_complete_with_pkl_rejected(tmp_path):
    (tmp_path / "COMPLETE.json").write_text('{"output_sha256": {}}')
    (tmp_path / "test.pkl").write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="zero persistent"):
        verify_existing(tmp_path)
