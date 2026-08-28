import copy
import hashlib
import json
from pathlib import Path

import yaml

from gradpert.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/experiments/genept_seed_priors/matrix.json"


def _normalized(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = payload["model"]["parameters"]
    for key in (
        "performance_pilot_variant",
        "genept_expected_sha256",
        "genept_artifact_path",
    ):
        parameters.pop(key)
    payload["artifacts"]["root"] = "<condition-root>"
    return copy.deepcopy(payload)


def test_genept_seed_prior_configs_are_pinned_and_otherwise_identical() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    rows = matrix["conditions"]
    assert [row["condition_id"] for row in rows] == [
        "latest_genept_model3",
        "genept_seed",
        "genept_seed_goexp",
    ]
    normalized: list[dict[str, object]] = []
    for row in rows:
        path = ROOT / row["config_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["config_sha256"]
        config = load_experiment_config(path)
        assert config.model.parameters["gene_feature_mode"].value == "genept_id_residual"
        assert config.model.parameters["genept_expected_sha256"].value == row["artifact_sha256"]
        assert config.model.parameters["genept_artifact_path"].value == row["artifact_path"]
        assert config.training.max_epochs.value == 10
        assert config.training.run_seeds == [1]
        normalized.append(_normalized(path))
    assert normalized[1:] == normalized[:-1]
