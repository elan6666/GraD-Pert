import json
from types import SimpleNamespace as NS

import pytest

from benchmarks.common.r50_gate import require_r50_smoke, seal_external_step, seal_r50_smoke


@pytest.fixture
def gate(tmp_path):
    cfg = NS(
        training=NS(formal_run_policy="external_fixed_50"),
        source_code=NS(commit="upstream"),
        model_id="gears",
        dataset_id="nadig_jurkat",
    )
    data = NS(manifest=NS(canonical_adata_sha256="data"), split=NS(split_content_sha256="split"))
    for role in ("best", "last"):
        (tmp_path / f"{role}.pt").write_bytes(role.encode())
    common = dict(config=cfg, config_sha256="config", training_data=data, environment_sha256="env")
    seal_r50_smoke(
        tmp_path,
        **common,
        source=NS(commit="train", dirty=False, formal_eligible=True),
        best_checkpoint=tmp_path / "best.pt",
        last_checkpoint=tmp_path / "last.pt",
        validation_value=0.4,
    )
    return tmp_path, {**common, "source_commit": "train"}


def test_validation_only_two_checkpoint_gate(gate):
    root, kwargs = gate
    assert require_r50_smoke(root, **kwargs)["receipt_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("test_evaluations", 1),
        ("source_dirty", True),
        ("source_commit", "other"),
        ("environment_sha256", "other"),
        ("epochs_completed", 2),
        ("canonical_test_truth_opened", True),
        ("upstream_commit", "other"),
        ("validation", [{"epoch": 1, "value": float("nan")}]),
    ],
)
def test_gate_rejects_identity_and_scope_drift(gate, field, value):
    root, kwargs = gate
    path = root / "small_results/r50_smoke.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        require_r50_smoke(root, **kwargs)


@pytest.mark.parametrize(
    "name", ["extra.pt", "nested/leak.pkl", "nested/leak.PKL", ".result-work-1/a"]
)
def test_gate_rejects_extra_artifacts(gate, name):
    root, kwargs = gate
    path = root / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(b"unexpected")
    with pytest.raises(ValueError):
        require_r50_smoke(root, **kwargs)


def test_gate_rejects_checkpoint_tamper(gate):
    root, kwargs = gate
    (root / "last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        require_r50_smoke(root, **kwargs)


def test_one_step_receipt_is_not_an_epoch_smoke(tmp_path):
    cfg = NS(
        training=NS(formal_run_policy="external_fixed_50"),
        source_code=NS(commit="upstream"),
        model_id="gears",
        dataset_id="nadig_jurkat",
    )
    data = NS(manifest=NS(canonical_adata_sha256="data"), split=NS(split_content_sha256="split"))
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    kwargs = dict(config=cfg, config_sha256="config", training_data=data, environment_sha256="env")
    receipt = seal_external_step(
        tmp_path,
        **kwargs,
        source=NS(commit="train", dirty=False, formal_eligible=True),
        checkpoint=checkpoint,
        update=dict(completed_steps=1, completed_epochs=0, checkpoint_serialization_exact=True),
    )
    assert receipt["completed_epochs"] == 0
    assert receipt["test_evaluations"] == 0
    assert not (tmp_path / "small_results/r50_smoke.json").exists()
    assert require_r50_smoke(tmp_path, **kwargs, source_commit="train")["receipt_sha256"]
    path = tmp_path / "small_results/one_step_smoke.json"
    receipt["completed_epochs"] = 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        require_r50_smoke(tmp_path, **kwargs, source_commit="train")
