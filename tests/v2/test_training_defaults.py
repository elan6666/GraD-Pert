"""Throughput defaults are explicit new files and never mutate historical recipes."""

import copy
import importlib.util
from pathlib import Path

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.training_defaults import with_training_throughput_defaults
from gradpert.config.v2 import V2Options

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "configs/v2/training_speed_20261106/baseline/gradpert_v2/nadig_jurkat.yaml"


def test_defaults_change_only_two_explicit_execution_fields_without_mutating_parent():
    config = load_experiment_config(PARENT)
    original = config.model_dump()
    before = copy.deepcopy(original)
    prepared = with_training_throughput_defaults(original)
    assert original == before
    arch, _ = V2Options.parse_parameters(config.model.parameters)
    assert not arch.relay_validate_once and not arch.cache_kda_constants
    parameters = prepared["model"]["parameters"]
    for name in ("relay_validate_once", "cache_kda_constants"):
        assert parameters[name]["value"] is True
        parameters.pop(name)
    assert prepared == original


def test_prepare_requires_new_path_and_materializes_defaults(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_training_config", ROOT / "scripts/v2/prepare_training_config.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "new/gradpert_v2/nadig_jurkat.yaml"
    digest = module.prepare(PARENT, path)
    assert len(digest) == 64
    arch, _ = V2Options.parse_parameters(load_experiment_config(path).model.parameters)
    assert arch.relay_validate_once and arch.cache_kda_constants
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.prepare(PARENT, path)
    assert path.read_bytes() == before
    invalid = tmp_path / "invalid.yaml"
    with pytest.raises(ValueError, match="path must match"):
        module.prepare(PARENT, invalid)
    assert not invalid.exists()


@pytest.mark.parametrize("model,attention", [("gears", "relay_full"), ("gradpert_v2", "hybrid")])
def test_external_or_unsupported_recipe_is_not_rewritten(model, attention):
    value = load_experiment_config(PARENT).model_dump()
    value["model_id"] = model
    value["model"]["parameters"]["attention"]["value"] = attention
    with pytest.raises(ValueError, match="throughput defaults"):
        with_training_throughput_defaults(value)
