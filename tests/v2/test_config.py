from pathlib import Path

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.schema import ExperimentConfig
from gradpert.config.v2 import V2Options

PROBE = Path(__file__).resolve().parents[2] / "configs/v2/capacity/gradpert_v2/nadig_jurkat.yaml"


def test_capacity_probe_keeps_complete_method_and_fixed_protocol():
    config = load_experiment_config(PROBE)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert config.model.version == "v2"
    assert arch.width == 256 and arch.prototypes == 16384
    assert (options.lambda1, options.lambda2) == (1, 0.1)
    assert config.training.max_epochs.value == 50
    assert options.local_views == 4
    assert config.training.train_batch_size.value == options.microbatch * options.accumulation


@pytest.mark.parametrize("change", ["version", "batch", "policy", "unknown"])
def test_v2_config_rejects_ambiguous_or_inconsistent_execution(change):
    payload = load_experiment_config(PROBE).model_dump(mode="json")
    if change == "version":
        payload["model"]["version"] = "v1"
    elif change == "batch":
        payload["training"]["train_batch_size"]["value"] = 100
    elif change == "policy":
        payload["training"]["formal_run_policy"] = "r50_selection"
    else:
        payload["model"]["parameters"]["unregistered_switch"] = {
            "value": 1,
            "source": "project_preregistered",
            "reference": "test",
        }
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(payload)
