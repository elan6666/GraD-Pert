import copy
from pathlib import Path

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.schema import ExperimentConfig
from gradpert.data._io import read_json
from gradpert.training.v2.artifacts import compact_validation


def test_population_is_stored_once_with_exact_ids_and_compact_history(tmp_path):
    result = {
        "split": "val",
        "prediction_loss": 0.1,
        "metrics": [],
        "control_manifest_sha256": "a",
        "reference_sha256": "b",
        "query_recipe": {"query_count": 1000},
        "conditions": [
            {
                "condition_id": "p",
                "loss": 0.1,
                "control_row_ids": ["c2", "c1"],
                "control_row_ids_sha256": "c",
                "truth_row_ids": ["t1"],
                "truth_row_ids_sha256": "d",
            }
        ],
    }
    first = compact_validation(result, root=tmp_path)
    assert "control_row_ids" not in first["conditions"][0]
    assert result["conditions"][0]["control_row_ids"] == ["c2", "c1"]
    population = read_json(tmp_path / "validation_population.json")
    assert population["conditions"][0]["control_row_ids"] == ["c2", "c1"]
    changed_loss = copy.deepcopy(result)
    changed_loss["prediction_loss"] = 0.2
    second = compact_validation(changed_loss, root=tmp_path)
    assert first["population_receipt"] == second["population_receipt"]
    assert len(list(tmp_path.iterdir())) == 1
    result["conditions"][0]["control_row_ids"].reverse()
    with pytest.raises(ValueError, match="population"):
        compact_validation(result, root=tmp_path)


def test_v2_configuration_rejects_full_prediction_export():
    path = Path(__file__).resolve().parents[2] / "configs/v2/capacity/gradpert_v2/nadig_jurkat.yaml"
    payload = load_experiment_config(path).model_dump(mode="json")
    payload["artifacts"]["result_mode"] = "single_pkl"
    with pytest.raises(ValueError, match="metrics_only"):
        ExperimentConfig.model_validate(payload)
