import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from test_relay_method import small_architecture

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from generate_graph_mhc import PARENT, verify_manifest  # noqa: E402


def test_gm1_changes_only_graph_residual_and_keeps_exact_br_protocol():
    manifest = verify_manifest(ROOT, ROOT / "configs/v2/graph_mhc_four_20261011/manifest.json")
    path = ROOT / manifest["rows"][0]["config"]
    config = load_experiment_config(path)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    actual = yaml.safe_load(path.read_text())
    switch = actual["model"]["parameters"].pop("graph_mhc")
    assert switch["value"] is True
    assert actual == yaml.safe_load((ROOT / PARENT).read_text())
    assert arch.graph_mhc and arch.streams == 4 and arch.graph_layers == 4
    assert config.training.max_epochs.value == 4
    assert options.validation_mode == "disabled"
    assert (options.microbatch, options.world_size, options.accumulation) == (48, 2, 2)


def test_graph_mhc_false_preserves_historical_architecture_identity():
    arch = small_architecture()
    assert "graph_mhc" not in arch.payload()
    assert replace(arch, graph_mhc=False).payload() == arch.payload()
    assert replace(arch, graph_mhc=True).payload()["graph_mhc"] is True


@pytest.mark.parametrize("value", [1, "true", None])
def test_graph_mhc_is_a_strict_boolean(value):
    with pytest.raises(ValueError, match="graph_mhc must be boolean"):
        replace(small_architecture(), graph_mhc=value)


def test_unsupported_graph_conditioning_combination_fails_explicitly():
    with pytest.raises(ValueError, match="graph mHC requires relay graph"):
        replace(
            small_architecture(),
            graph_mhc=True,
            control_conditioned_graph=True,
            mlp_profile="unified",
            graph_source_key_gate=True,
        )


def test_manifest_seals_one_arm_and_the_fixed_parent():
    value = json.loads((ROOT / "configs/v2/graph_mhc_four_20261011/manifest.json").read_text())
    assert value["priority"] == ["GM1"]
    assert len(value["rows"]) == 1
    assert value["parent_training_commit"] == "b37b171fd68cb29e5c8142c66778aac6da9a6f06"
