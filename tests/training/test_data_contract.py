from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.hashing import sha256_json
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


def test_training_receipt_separates_runtime_and_canonical_graph_axes(
    tmp_path: Path,
) -> None:
    from gradpert.training.data import write_training_data_receipt

    data = _unopened_data()
    data.graph_gene_ids = ("G2", "PERT")
    data.manifest = SimpleNamespace(
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        canonical_adata_sha256="a" * 64,
        expression_gene_order_sha256="b" * 64,
        graph_gene_order_sha256="c" * 64,
    )
    data.split = SimpleNamespace(split_content_sha256="d" * 64)
    data.train_row_indices = (1, 2)
    destination = tmp_path / "receipt.json"
    write_training_data_receipt(data, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["canonical_graph_gene_order_sha256"] == "c" * 64
    assert payload["runtime_graph_gene_count"] == 2
    assert payload["runtime_graph_gene_order_sha256"] == sha256_json(["G2", "PERT"])
