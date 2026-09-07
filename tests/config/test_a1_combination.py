from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gradpert.config import ExperimentConfig, NativeArchitectureOptions, load_experiment_config

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/combinations/a1_e3_l1_m1/gradpert_b2/nadig_jurkat.yaml"
)


def test_a1_contract():
    config = load_experiment_config(CONFIG)
    arch = NativeArchitectureOptions.from_parameters(config.model.parameters)
    assert arch.gene_feature_mode == "genept_initialized"
    assert arch.local_view_builder == "fanout"
    assert arch.graph_encoder_family == "single_source_gat"
    assert arch.graph_sources == ("string",)
    assert arch.graph_dropout == 0.2
    assert arch.local_view_count == 4
    assert arch.local_view_node_budget_ratio_denominator == 2
    assert arch.local_view_node_budget_ratio_numerator == 1
    assert config.training.max_epochs.value == 100
    assert config.training.early_stopping
    assert config.training.early_stopping_patience.value == 10
    assert config.training.run_seeds == [1]
    assert "performance_pilot_variant" not in config.model.parameters


@pytest.mark.parametrize("field,value", [("max_epochs", 200), ("early_stopping_patience", 5)])
def test_a1_budget_rejects_drift(field, value):
    data = deepcopy(yaml.safe_load(CONFIG.read_text()))
    data["training"][field]["value"] = value
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)


def test_a1_requires_early_stopping():
    data = yaml.safe_load(CONFIG.read_text())
    data["training"]["early_stopping"] = False
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(data)
