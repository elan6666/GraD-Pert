from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from gradpert.hashing import sha256_json
from gradpert.training.data import (
    CanonicalTrainingData,
    TrainingPipelineStats,
    _TrainingBatchSpec,
)


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


def test_training_batch_identity_specs_are_expression_free_and_use_gene_ids() -> None:
    data = _unopened_data()
    data.graph_gene_ids = ("G0", "G1", "PERT")

    def batch_specs(
        _self: CanonicalTrainingData,
        *,
        epoch: int,
        batch_size: int,
        max_unique_conditions: int,
    ) -> tuple[_TrainingBatchSpec, ...]:
        assert (epoch, batch_size, max_unique_conditions) == (0, 256, 8)
        return (
            _TrainingBatchSpec(
                perturbed_indices=(4, 7),
                perturbed_row_ids=("row-4", "row-7"),
                control_indices=(1, 2),
                control_row_ids=("control-1", "control-2"),
                condition_ids=("PERT", "G0+PERT"),
                anchors_by_condition={"PERT": (2,), "G0+PERT": (0, 2)},
            ),
        )

    def reject_expression_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("identity-only schedule must not read expression arrays")

    data._batch_specs = MethodType(batch_specs, data)  # type: ignore[method-assign]
    data._read_expression_indices = reject_expression_read  # type: ignore[method-assign]
    identities = data.training_batch_identity_specs(
        epoch=0,
        batch_size=256,
        max_unique_conditions=8,
    )
    assert identities[0].perturbed_row_ids == ("row-4", "row-7")
    assert identities[0].control_row_ids == ("control-1", "control-2")
    assert identities[0].anchor_gene_ids_by_condition == {
        "PERT": ("PERT",),
        "G0+PERT": ("G0", "PERT"),
    }


def test_real_batch_spec_path_remains_expression_free() -> None:
    data = _unopened_data()
    data.run_seed = 1
    data.graph_gene_ids = ("G0", "G1")
    data.row_ids = ("p0", "p1", "p2", "p3", "c0", "c1")
    data.condition_ids = ("G0", "G1", "G0", "G1", "ctrl", "ctrl")
    data.context_ids = ("ctx",) * 6
    data.train_row_indices = (0, 1, 2, 3)
    data.control_pools = {"ctx": ("c0", "c1")}
    data._row_index = {row_id: index for index, row_id in enumerate(data.row_ids)}
    data.anchors_by_condition = {"G0": (0,), "G1": (1,)}
    data.pipeline_stats = TrainingPipelineStats()

    def reject_expression_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("real identity schedule must not read expression arrays")

    data._read_expression_indices = reject_expression_read  # type: ignore[method-assign]
    identities = data.training_batch_identity_specs(
        epoch=0,
        batch_size=2,
        max_unique_conditions=2,
    )
    assert len(identities) == 2
    assert sum(len(identity.perturbed_row_ids) for identity in identities) == 4
    assert all(set(identity.control_row_ids) <= {"c0", "c1"} for identity in identities)
    assert all(
        identity.anchor_gene_ids_by_condition
        == {condition: (condition,) for condition in dict.fromkeys(identity.condition_ids)}
        for identity in identities
    )
