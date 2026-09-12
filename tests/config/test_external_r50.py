from pathlib import Path

import pytest
from pydantic import ValidationError

from gradpert.config.loader import load_experiment_config
from gradpert.config.schema import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("scouter_genept_seed", "gears", "txpert_public")


@pytest.mark.parametrize("model_id", MODELS)
def test_external_r50_preserves_official_scientific_config(model_id):
    old = load_experiment_config(ROOT / "configs/external-full" / model_id / "nadig_jurkat.yaml")
    new = load_experiment_config(ROOT / "configs/r50-rerun" / model_id / "nadig_jurkat.yaml")
    assert new.source_code == old.source_code
    assert new.data == old.data
    assert new.model == old.model
    assert new.evaluation == old.evaluation
    assert new.training.formal_run_policy == "external_fixed_50"
    assert new.training.max_epochs.value == 50
    assert new.training.smoke_epochs.value == 1
    assert not new.training.early_stopping
    # The schema's model/dataset identity stays fixed; run IDs/roots are fresh.
    assert new.experiment_id == old.experiment_id
    assert new.artifacts.root != old.artifacts.root
    permitted = {"formal_run_policy", "max_epochs", "early_stopping", "early_stopping_patience"}
    before = old.training.model_dump()
    after = new.training.model_dump()
    assert {k: v for k, v in before.items() if k not in permitted} == {
        k: v for k, v in after.items() if k not in permitted
    }
    assert new.training.early_stopping_patience.value == old.training.early_stopping_patience.value


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize(
    "field,value", [("early_stopping", True), ("run_seeds", [2]), ("min_delta", -1)]
)
def test_external_r50_rejects_invalid_training_policy(model_id, field, value):
    cfg = load_experiment_config(ROOT / "configs/r50-rerun" / model_id / "nadig_jurkat.yaml")
    data = cfg.model_dump()
    data["training"][field] = value
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("epochs", [1, 49, 51, 100])
def test_external_r50_rejects_budget_override(epochs):
    cfg = load_experiment_config(ROOT / "configs/r50-rerun/gears/nadig_jurkat.yaml")
    data = cfg.model_dump()
    data["training"]["max_epochs"]["value"] = epochs
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(data)
