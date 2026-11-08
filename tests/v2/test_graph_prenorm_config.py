"""Single-axis graph normalization configs and historical checkpoint identity."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.training_defaults import with_training_throughput_defaults
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "configs/v2/graph_prenorm_20261108"


def test_only_graph_normalization_differs_from_exact_parent():
    manifest = json.loads((FAMILY / "manifest.json").read_text())
    parent_path = ROOT / manifest["parent_config"]
    assert sha256_file(parent_path) == manifest["parent_sha256"]
    parent = with_training_throughput_defaults(load_experiment_config(parent_path).model_dump())
    for row, norm in zip(manifest["rows"], ["historical", "rmsnorm"], strict=True):
        path = ROOT / row["config"]
        assert sha256_file(path) == row["sha256"]
        config = load_experiment_config(path)
        arch, options = V2Options.parse_parameters(config.model.parameters)
        assert arch.graph_input_norm == norm
        assert arch.ffn_input_norm == "historical" and arch.ffn_variant == "gelu"
        assert arch.relay_validate_once and arch.cache_kda_constants
        assert config.training.max_epochs.value == 2
        assert (options.world_size, options.microbatch, options.accumulation) == (1, 80, 4)
        current = config.model_dump()
        current["model"]["parameters"].pop("graph_input_norm")
        assert current == parent


def test_historical_payload_preserves_old_architecture_identity():
    arch = V2Architecture()
    assert "graph_input_norm" not in arch.payload()
    assert "graph_input_norm" not in replace(arch, graph_input_norm="historical").payload()


@pytest.mark.parametrize("value", [None, True, "layernorm", "none"])
def test_invalid_graph_norm_fails_closed(value):
    with pytest.raises(ValueError, match="graph input normalization"):
        V2Architecture(graph_input_norm=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"graph_mhc": True},
        {"ffn_input_norm": "rmsnorm"},
        {"attention": "hybrid"},
        {"graph_read_mode": "static"},
        {"mlp_profile": "unified"},
        {"attention_replacement": "softmax"},
    ],
)
def test_unsupported_combination_is_rejected(changes):
    config = load_experiment_config(FAMILY / "graph_rmsnorm/gradpert_v2/nadig_jurkat.yaml")
    arch, _ = V2Options.parse_parameters(config.model.parameters)
    with pytest.raises(ValueError, match="graph input normalization"):
        replace(arch, **copy.deepcopy(changes))
