"""Explicit Q rank, strict mechanism scope and unchanged default identity."""

from dataclasses import replace
from pathlib import Path

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options

ROOT = Path(__file__).resolve().parents[2]


def test_q_arm_changes_only_one_parameter_from_parent():
    base = load_experiment_config(
        ROOT / "configs/v2/graph_prenorm_20261108/baseline/gradpert_v2/nadig_jurkat.yaml"
    )
    candidate = load_experiment_config(
        ROOT / "configs/v2/mla_query_compression_20261108/q_rank64/gradpert_v2/nadig_jurkat.yaml"
    )
    current = candidate.model_dump()
    current["model"]["parameters"].pop("mla_query_rank")
    assert current == base.model_dump()
    arch, options = V2Options.parse_parameters(candidate.model.parameters)
    assert arch.mla_query_rank == arch.latent_rank == 64
    assert arch.graph_input_norm == arch.ffn_input_norm == "historical"
    assert arch.relay_validate_once and arch.cache_kda_constants
    assert (options.world_size, options.microbatch, options.accumulation) == (1, 80, 4)
    assert candidate.training.max_epochs.value == 2


def test_default_rank_does_not_change_historical_payload():
    assert "mla_query_rank" not in V2Architecture().payload()
    assert V2Architecture().payload() == replace(V2Architecture(), mla_query_rank=0).payload()


@pytest.mark.parametrize("value", [True, -1, None, "64", 256, 512])
def test_invalid_rank_is_rejected(value):
    with pytest.raises(ValueError, match="query rank"):
        V2Architecture(mla_query_rank=value)


@pytest.mark.parametrize(
    "changes",
    [
        {"attention": "hybrid"},
        {"mlp_profile": "historical"},
        {"graph_input_norm": "rmsnorm"},
        {"ffn_input_norm": "rmsnorm"},
        {"attention_replacement": "softmax"},
    ],
)
def test_combined_mechanisms_fail_closed(changes):
    config = load_experiment_config(
        ROOT / "configs/v2/mla_query_compression_20261108/q_rank64/gradpert_v2/nadig_jurkat.yaml"
    )
    arch, _ = V2Options.parse_parameters(config.model.parameters)
    with pytest.raises(ValueError):
        replace(arch, **changes)
