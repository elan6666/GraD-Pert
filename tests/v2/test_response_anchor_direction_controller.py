"""CPU contracts for separate direction calibration/fit provenance."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))

from gradpert.config import load_experiment_config  # noqa: E402
from gradpert.hashing import sha256_file, sha256_json  # noqa: E402
from scripts.v2 import generate_response_anchor as generator  # noqa: E402
from scripts.v2 import run_response_anchor as launcher  # noqa: E402


def diagnostics():
    return dict(
        response_anchor_loss="cosine_direction",
        eps=1e-6,
        target_count=1322,
        target_clamped_count=0,
        target_norm_min=9.0,
        target_norm_max=49.0,
        head_observation_count=64,
        head_clamped_count=0,
        head_norm_min=1.2,
        head_norm_max=6.0,
    )


@pytest.fixture
def direction_fixture(tmp_path, monkeypatch):
    parent = tmp_path / generator.PARENT
    parent.parent.mkdir(parents=True)
    parent.write_bytes((ROOT / generator.PARENT).read_bytes())
    probe, decision_path, mode = launcher.recipe(tmp_path, "R15-DIR")
    fit = decision_path.parent / "R15-DIR/gradpert_v2/nadig_jurkat.yaml"
    kwargs = dict(
        bank_path="/data/yilangliu/frozen-bank/manifest.json",
        manifest_sha256="1" * 64,
        arrays_sha256="2" * 64,
        loss_mode=mode,
    )
    for path, weight in ((probe, 1.0), (fit, 0.2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(generator.configuration(tmp_path, weight=weight, **kwargs)))
    decision = dict(arm="R15-DIR", response_anchor_loss=mode, calibration_source="c" * 40)
    decision_path.write_text(json.dumps(decision))
    calibration = dict(
        calibration_complete=True,
        state_restored=True,
        forward_precision="bfloat16_autocast",
        world_size=2,
        optimizer_updates=0,
        teacher_updates=0,
        center_updates=0,
        batches=[{}] * 8,
        source=dict(commit=decision["calibration_source"], dirty=False),
        config_sha256=sha256_file(probe),
        resolved_config_sha256=sha256_json(load_experiment_config(probe).model_dump(mode="json")),
        identity=dict(run_seed=1),
        bank_manifest_sha256=kwargs["manifest_sha256"],
        bank_arrays_sha256=kwargs["arrays_sha256"],
        selected_weight=0.2,
        response_anchor_loss=mode,
        direction_diagnostics=diagnostics(),
    )
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "scripts/v2/run_response_anchor.py"))
    return fit, calibration


def test_new_recipe_requires_own_calibration_and_explicit_arm(direction_fixture):
    fit, calibration = direction_fixture
    launcher.validate_calibration(fit, calibration, "R15-DIR")
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.validate_calibration(fit, calibration)
    old = copy.deepcopy(calibration)
    old.pop("response_anchor_loss")
    old.pop("direction_diagnostics")
    with pytest.raises(ValueError):
        launcher.validate_calibration(fit, old, "R15-DIR")


@pytest.mark.parametrize(
    "key,value",
    [
        ("response_anchor_loss", "mse"),
        ("config_sha256", "old"),
        ("resolved_config_sha256", "old"),
        ("selected_weight", 0.03148168297045377),
        ("world_size", 1),
        ("optimizer_updates", 1),
    ],
)
def test_direction_rejects_mislabeled_old_calibration(direction_fixture, key, value):
    fit, calibration = direction_fixture
    calibration[key] = value
    with pytest.raises(ValueError):
        launcher.validate_calibration(fit, calibration, "R15-DIR")


@pytest.mark.parametrize(
    "key,value",
    [
        ("response_anchor_loss", "mse"),
        ("eps", 1e-4),
        ("target_count", 1321),
        ("target_clamped_count", 1),
        ("target_norm_min", 0),
        ("head_observation_count", 0),
        ("head_norm_min", float("nan")),
        ("head_norm_max", float("inf")),
        ("head_clamped_count", 65),
    ],
)
def test_norm_receipt_rejects_undefined_or_unobserved_directions(key, value):
    data = diagnostics()
    data[key] = value
    with pytest.raises(ValueError):
        launcher.validate_direction_diagnostics(data)


def test_clamped_readout_is_recorded_without_changing_loss_or_dropping_conditions():
    data = diagnostics()
    data.update(head_norm_min=0.0, head_clamped_count=1)
    launcher.validate_direction_diagnostics(data)


def test_default_generator_is_exact_historical_recipe():
    path = ROOT / "configs/v2/response_anchor_20261028/R14-FRGA/gradpert_v2/nadig_jurkat.yaml"
    old = yaml.safe_load(path.read_text())
    p = old["model"]["parameters"]
    actual = generator.configuration(
        ROOT,
        bank_path=p["response_anchor_path"]["value"],
        manifest_sha256=p["response_anchor_sha256"]["value"],
        arrays_sha256=p["response_anchor_arrays_sha256"]["value"],
        weight=p["response_anchor_weight"]["value"],
    )
    assert actual == old
    assert "response_anchor_loss" not in actual["model"]["parameters"]
    with pytest.raises(ValueError):
        generator.configuration(
            ROOT,
            bank_path="x",
            manifest_sha256="a",
            arrays_sha256="b",
            weight=1,
            loss_mode="other",
        )
    with pytest.raises(ValueError):
        launcher.recipe(ROOT, "R15-FORGED")
