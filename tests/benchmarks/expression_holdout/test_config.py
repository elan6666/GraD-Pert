from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from benchmarks.expression_holdout.config import load_config, validate_preflight
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/v2/official_expression_20261013/scouter_jurkat.yaml"


@pytest.mark.parametrize("name", ["scouter", "scdfm", "gears"])
def test_frozen_official_recipes_and_shared_data(name):
    cfg = load_config(ROOT / f"configs/v2/official_expression_20261013/{name}_jurkat.yaml")
    assert cfg["data"] == load_config(CONFIG)["data"]
    assert cfg["data"]["exclude_test_target_expression"] is True
    assert len(cfg["source_code"]["commit"]) == 40


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["data"].update(exclude_test_target_expression=False),
        lambda c: c["evaluation"].update(control_cells=128),
        lambda c: c["source_code"].update(commit="a" * 40),
    ],
)
def test_protocol_and_source_drift_rejected(tmp_path, mutation):
    cfg = deepcopy(load_config(CONFIG))
    mutation(cfg)
    path = tmp_path / "modified.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_probe_requires_actual10_updates_and_restore_with_same_identity():
    cfg = load_config(CONFIG)
    receipt = {
        "schema": "official-expression-holdout-preflight-v1",
        "status": "passed",
        "config_sha256": sha256_file(CONFIG),
        "source_commit": "s",
        "source_clean": True,
        "official_commit": cfg["source_code"]["commit"],
        "runtime_sha256": "r",
        "optimizer_updates": 10,
        "restored_checkpoint_predictions_finite": True,
        "controls_per_condition": 300,
        "zero_pkl": True,
    }
    kwargs = dict(
        config=CONFIG,
        source_commit="s",
        official_commit=cfg["source_code"]["commit"],
        runtime_sha256="r",
    )
    validate_preflight(receipt, **kwargs)
    for change in [
        {"optimizer_updates": 2},
        {"source_clean": False},
        {"zero_pkl": False},
        {"restored_checkpoint_predictions_finite": False},
        {"config_sha256": "x"},
    ]:
        with pytest.raises(ValueError):
            validate_preflight(receipt | change, **kwargs)
