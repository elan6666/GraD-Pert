from __future__ import annotations

from types import SimpleNamespace

import pytest

from gradpert.training.data import CanonicalTrainingData


def _unopened_data() -> CanonicalTrainingData:
    data = object.__new__(CanonicalTrainingData)
    data.split = SimpleNamespace(policy_id="source__gears_default_graph_intersection_v1")
    return data


def test_experiment_data_contract_accepts_exact_policy() -> None:
    _unopened_data().require_experiment_data_contract(
        registry_version="datasets-v2",
        split_policy="source__gears_default_graph_intersection_v1",
    )


@pytest.mark.parametrize(
    ("registry_version", "split_policy"),
    [
        ("datasets-v1", "source__gears_default_graph_intersection_v1"),
        ("datasets-v2", "stale_source_policy"),
    ],
)
def test_experiment_data_contract_rejects_stale_config(
    registry_version: str,
    split_policy: str,
) -> None:
    with pytest.raises(ValueError):
        _unopened_data().require_experiment_data_contract(
            registry_version=registry_version,
            split_policy=split_policy,
        )
