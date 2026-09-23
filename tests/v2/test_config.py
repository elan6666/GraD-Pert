from pathlib import Path

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.schema import ExperimentConfig
from gradpert.config.v2 import V2Options

PROBE = Path(__file__).resolve().parents[2] / "configs/v2/capacity/gradpert_v2/nadig_jurkat.yaml"
GLM53 = Path(__file__).resolve().parents[2] / "configs/v2/glm53_flash_jurkat"


@pytest.mark.parametrize("variant,topk", [("default", 500), ("top100", 100)])
def test_glm53_configs_keep_b1_seed_and_explicit_sparse_settings(variant, topk):
    path = GLM53 / variant / "gradpert_v2/nadig_jurkat.yaml"
    config = load_experiment_config(path)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert (arch.attention, arch.ffn_type, arch.sparse_topk) == ("hybrid_sparse", "swiglu", topk)
    assert arch.sparse_index_dim == 64 and arch.sparse_query_chunk == 8
    assert options.genept_artifact_path.endswith("v2-genept-pca256-b4e3a08.npz")
    assert config.training.max_epochs.value == 5


def test_capacity_probe_keeps_complete_method_and_fixed_protocol():
    config = load_experiment_config(PROBE)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert config.model.version == "v2"
    assert arch.width == 256 and arch.prototypes == 16384
    assert (options.lambda1, options.lambda2) == (1, 0.1)
    assert config.training.max_epochs.value == 50
    assert options.local_views == 4
    assert config.training.train_batch_size.value == options.microbatch * options.accumulation


def test_v2_five_epoch_protocol_is_explicit_and_does_not_reinterpret_old_runs():
    payload = load_experiment_config(PROBE).model_dump(mode="json")
    payload["training"]["formal_run_policy"] = "v2_fixed_5"
    payload["training"]["max_epochs"]["value"] = 5
    five = ExperimentConfig.model_validate(payload)
    assert five.training.max_epochs.value == 5
    payload["training"]["max_epochs"]["value"] = 50
    with pytest.raises(ValueError, match="exactly 5 epochs"):
        ExperimentConfig.model_validate(payload)


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


def test_expression_visibility_defaults_and_legacy_serialization():
    from gradpert.config.schema import ModelConfig

    v2 = load_experiment_config(PROBE).model
    assert v2.excludes_test_target_expression
    assert v2.model_dump()["exclude_test_target_expression"] is True
    for model_id, version in (("gradpert_b2", None), ("gradpert_v2", "v2")):
        payload = v2.model_dump()
        payload.update(model_id=model_id, version=version)
        payload.pop("exclude_test_target_expression")
        model = ModelConfig.model_validate(payload)
        assert model.excludes_test_target_expression == (model_id == "gradpert_v2")
        if model_id == "gradpert_b2":
            assert "exclude_test_target_expression" not in model.model_dump()
        for value in (False, True):
            changed = ModelConfig.model_validate(
                {**payload, "exclude_test_target_expression": value}
            )
            assert changed.excludes_test_target_expression is value
        with pytest.raises(ValueError):
            ModelConfig.model_validate({**payload, "exclude_test_target_expression": "false"})
