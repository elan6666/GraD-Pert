from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData, TrainingPipelineStats
from gradpert.training.v2.train_selection import apply_train_selection


def _fixture(root: Path):
    data = object.__new__(CanonicalTrainingData)
    data.row_ids = ("a0", "a1", "b0", "b1", "control", "val0", "val1", "test")
    data.condition_ids = ("A", "A", "B", "B", "ctrl", "V", "W", "T")
    data.context_ids = ("ctx",) * 8
    data.train_row_indices = (0, 1, 2, 3)
    data.control_pools = {"ctx": ("control",)}
    data._row_index = {row: i for i, row in enumerate(data.row_ids)}
    data.anchors_by_condition = {"A": (0,), "B": (1,), "V": (2,), "W": (3,)}
    data.graph_gene_ids = ("A", "B", "V", "W", "T")
    data.run_seed = 1
    data.pipeline_stats = TrainingPipelineStats()
    data.split = SimpleNamespace(
        train_conditions=("A", "B"),
        val_conditions=("V", "W"),
        split_content_sha256="b" * 64,
    )
    data.manifest = SimpleNamespace(
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        canonical_adata_sha256="a" * 64,
        observation_order_sha256=sha256_json(list(data.row_ids)),
        expression_gene_order_sha256="c" * 64,
        graph_gene_order_sha256="d" * 64,
    )
    source = {
        "parent_h5ad_sha256": "f" * 64,
        "selected_train_row_ids": ["a1", "b0"],
        "selected_train_row_ids_sha256": sha256_json(["a1", "b0"]),
        "train_condition_ids": ["A", "B"],
        "unmodified_row_ids_sha256": sha256_json(list(data.row_ids[4:])),
        "cap": 1,
        "seed": 42,
        "sampling": "condition_batch_largest_remainder_uniform_without_replacement",
        "by_condition": {"A": ["a1"], "B": ["b0"]},
    }
    source_path, proof = root / "analysis.json", root / "equivalence.json"
    source_path.write_text(json.dumps(source))
    proof.write_text(
        json.dumps(
            {
                "schema_version": "gradpert-canonical-equivalence-1",
                "files": [{"sha256": "a" * 64}, {"sha256": "f" * 64}],
                "comparison": {"X_values_equal": True, "obs_equal": True, "var_equal": True},
            }
        )
    )
    payload = {
        **source,
        "schema_version": "gradpert-v2-training-selection-1",
        "state": "frozen_training_row_selection",
        "dataset_id": data.manifest.dataset_id,
        "protocol_id": data.manifest.protocol_id,
        "parent_h5ad_sha256": data.manifest.canonical_adata_sha256,
        "split_content_sha256": data.split.split_content_sha256,
        "observation_order_sha256": data.manifest.observation_order_sha256,
        "expression_gene_order_sha256": data.manifest.expression_gene_order_sha256,
        "graph_gene_order_sha256": data.manifest.graph_gene_order_sha256,
        "analysis_selection_path": str(source_path),
        "analysis_selection_sha256": sha256_file(source_path),
        "analysis_source_commit": "e" * 40,
        "equivalence_path": str(proof),
        "equivalence_sha256": sha256_file(proof),
    }
    path = root / "training_selection.json"
    path.write_text(json.dumps(payload))
    return data, path, payload


def test_subset_changes_training_schedule_only_and_is_frozen_across_epochs(tmp_path):
    data, path, _ = _fixture(tmp_path)
    data._iter_cpu_batches = lambda specs: iter(specs)
    data._to_training_batch = lambda spec, *, device: spec
    val_before = list(
        data.iter_validation_epoch(device="cpu", batch_size=2, max_unique_conditions=0)
    )
    receipt = apply_train_selection(data, path, sha256_file(path))
    assert (receipt["original_train_rows"], receipt["selected_train_rows"]) == (4, 2)
    for epoch in (0, 1, 5):
        specs = data.training_batch_identity_specs(
            epoch=epoch, batch_size=2, max_unique_conditions=0
        )
        assert {row for spec in specs for row in spec.perturbed_row_ids} == {"a1", "b0"}
        assert all(set(spec.control_row_ids) == {"control"} for spec in specs)
    assert (
        list(data.iter_validation_epoch(device="cpu", batch_size=2, max_unique_conditions=0))
        == val_before
    )
    assert data.row_ids[-1] == "test" and data.control_pools == {"ctx": ("control",)}


@pytest.mark.parametrize(
    "change,match",
    [
        ({"parent_h5ad_sha256": "f" * 64}, "canonical identity"),
        ({"split_content_sha256": "f" * 64}, "canonical identity"),
        ({"train_condition_ids": ["B", "A"]}, "condition axis"),
        ({"selected_train_row_ids": ["a1", "val0"]}, "non-training"),
        ({"selected_train_row_ids": ["a1", "a1"]}, "row IDs"),
        ({"selected_train_row_ids": ["b0", "a1"]}, "row order"),
        ({"selected_train_row_ids": ["a1"]}, "loses conditions"),
        ({"cap": 2}, "cap quotas"),
        ({"unmodified_row_ids_sha256": "f" * 64}, "non-training axis"),
        ({"analysis_selection_sha256": "f" * 64}, "evidence mismatch"),
    ],
)
def test_invalid_selection_fails_before_mutating_data(tmp_path, change, match):
    data, path, payload = _fixture(tmp_path)
    payload.update(change)
    if "selected_train_row_ids" in change:
        payload["selected_train_row_ids_sha256"] = sha256_json(payload["selected_train_row_ids"])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        apply_train_selection(data, path, sha256_file(path))
    assert data.train_row_indices == (0, 1, 2, 3)


def test_manifest_tampering_is_rejected(tmp_path):
    data, path, _ = _fixture(tmp_path)
    digest = sha256_file(path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="file/hash"):
        apply_train_selection(data, path, digest)


def test_false_canonical_equivalence_is_rejected_even_with_valid_file_hashes(tmp_path):
    data, path, payload = _fixture(tmp_path)
    proof = Path(payload["equivalence_path"])
    evidence = json.loads(proof.read_text())
    evidence["comparison"]["X_values_equal"] = False
    proof.write_text(json.dumps(evidence))
    payload["equivalence_sha256"] = sha256_file(proof)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="equivalence not verified"):
        apply_train_selection(data, path, sha256_file(path))
    assert data.train_row_indices == (0, 1, 2, 3)
