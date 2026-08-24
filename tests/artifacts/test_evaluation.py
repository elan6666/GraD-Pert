from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gradpert.artifacts import (
    EvaluationConditionInput,
    EvaluationProvenance,
    PredictionConditionArrays,
    PredictionProvenance,
    load_evaluation_bundle,
    seal_evaluation_bundle,
    seal_prediction_artifact,
)
from gradpert.evaluation import write_small_metric_exports
from gradpert.hashing import sha256_json

SHA = "a" * 64
COMMIT = "b" * 40


def _evaluation_provenance() -> EvaluationProvenance:
    return EvaluationProvenance(
        state_manifest_file_sha256=SHA,
        state_arrays_sha256=SHA,
        state_condition_ids_sha256=SHA,
        de_gene_indices_sha256=SHA,
        top_de_gene_indices_sha256=SHA,
        de_unavailable_reasons_sha256=SHA,
        de_method="scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets",
        de_reference="ctrl",
        de_source_commit=COMMIT,
        systema_reference_condition_ids=("PERT_A", "PERT_B"),
        systema_reference_condition_ids_sha256=sha256_json(["PERT_A", "PERT_B"]),
        metric_control_means_content_sha256=SHA,
    )


def _prediction(tmp_path: Path):
    genes = ("A", "B", "C")
    controls = np.repeat(np.asarray([[1.0, 2.0, 1.0]], dtype=np.float32), 300, axis=0)
    provenance = PredictionProvenance(
        model_id="gradpert_b2",
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        run_id="fixture-run",
        run_seed=1,
        source_commit=COMMIT,
        source_dirty=False,
        formal_eligible=True,
        config_sha256=SHA,
        environment_sha256=SHA,
        canonical_data_sha256=SHA,
        gene_order_sha256=sha256_json(list(genes)),
        split_content_sha256=SHA,
        control_manifest_sha256=SHA,
        checkpoint_sha256=SHA,
    )
    conditions = [
        PredictionConditionArrays(
            condition_id="PERT_A",
            prediction=controls + np.asarray([1.0, 2.0, 4.0]),
            input_control=controls,
            input_control_row_ids=tuple(f"ctrl-{index % 11}" for index in range(300)),
        ),
        PredictionConditionArrays(
            condition_id="PERT_B",
            prediction=controls + np.asarray([3.0, 1.0, 2.0]),
            input_control=controls,
            input_control_row_ids=tuple(f"ctrl-{index % 13}" for index in range(300)),
        ),
    ]
    return seal_prediction_artifact(
        tmp_path / "prediction.pkl",
        provenance=provenance,
        gene_ids=genes,
        conditions=conditions,
    )


def _evaluation_inputs() -> list[EvaluationConditionInput]:
    return [
        EvaluationConditionInput(
            condition_id="PERT_A",
            truth=np.asarray([[3.0, 5.0, 7.0], [4.0, 4.0, 8.0]], dtype=np.float32),
            truth_row_ids=("truth-a-0", "truth-a-1"),
            metric_control_pool_mean=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            de_gene_indices=(0, 1, 2),
            top_de_gene_indices=(0, 1, 2),
        ),
        EvaluationConditionInput(
            condition_id="PERT_B",
            truth=np.asarray([[6.0, 3.0, 4.0]], dtype=np.float32),
            truth_row_ids=("truth-b-0",),
            metric_control_pool_mean=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            de_gene_indices=(0, 1, 2),
            top_de_gene_indices=(0, 1, 2),
        ),
    ]


def test_evaluation_bundle_joins_truth_and_recomputes_all_metrics(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path)
    path = tmp_path / "evaluation.pkl"
    bundle = seal_evaluation_bundle(
        path,
        prediction=prediction,
        conditions=_evaluation_inputs(),
        systema_reference=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        provenance=_evaluation_provenance(),
    )

    assert tuple(bundle.conditions) == ("PERT_A", "PERT_B")
    assert bundle.conditions["PERT_A"].truth.shape == (2, 3)
    assert [item.metric_id for item in bundle.manifest.metrics] == [
        "txpert_macro_pearson_delta",
        "trishift_pearson_delta",
        "systema_pearson",
    ]
    assert all(item.total_condition_count == 2 for item in bundle.manifest.metrics)
    assert b"Truth" in path.read_bytes()
    assert b"Truth" not in (tmp_path / "prediction.pkl").read_bytes()


def test_small_metric_exports_are_deterministic_and_array_free(tmp_path: Path) -> None:
    bundle = seal_evaluation_bundle(
        tmp_path / "evaluation.pkl",
        prediction=_prediction(tmp_path),
        conditions=_evaluation_inputs(),
        systema_reference=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        provenance=_evaluation_provenance(),
    )
    first = write_small_metric_exports(bundle, tmp_path / "small-a")
    second = write_small_metric_exports(bundle, tmp_path / "small-b")

    assert first.files == second.files
    assert set(first.files) == {
        "metric_availability.json",
        "metrics_export_manifest.json",
        "metrics_per_condition.csv",
        "metrics_summary.csv",
        "metrics_summary.json",
    }
    combined = b"".join((first.root / filename).read_bytes() for filename in first.files)
    assert b"Truth" not in combined
    assert b"Pred" not in combined
    assert b"txpert_macro_pearson_delta" in combined
    summary = json.loads((first.root / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary["evaluator_state"]["manifest_file_sha256"] == SHA
    assert summary["evaluator_state"]["systema_reference_condition_ids"] == [
        "PERT_A",
        "PERT_B",
    ]


def test_evaluation_loader_rejects_metric_tampering_even_with_matching_file_hash(
    tmp_path: Path,
) -> None:
    bundle = seal_evaluation_bundle(
        tmp_path / "evaluation.pkl",
        prediction=_prediction(tmp_path),
        conditions=_evaluation_inputs(),
        systema_reference=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        provenance=_evaluation_provenance(),
    )
    path = tmp_path / "evaluation.pkl"
    with path.open("rb") as handle:
        package = pickle.load(handle)
    package["conditions"]["PERT_A"]["metrics"][0]["value"] = -0.75
    with path.open("wb") as handle:
        pickle.dump(package, handle)
    matching_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert matching_hash != bundle.file_sha256

    with pytest.raises(ValueError, match="metric mismatch"):
        load_evaluation_bundle(
            path,
            expected_file_sha256=matching_hash,
            trusted_root=tmp_path,
        )


def test_evaluation_bundle_rejects_condition_or_truth_row_mismatch(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path)
    with pytest.raises(ValueError, match="condition sets differ"):
        seal_evaluation_bundle(
            tmp_path / "missing.pkl",
            prediction=prediction,
            conditions=_evaluation_inputs()[:1],
            systema_reference=np.zeros(3),
            provenance=_evaluation_provenance(),
        )

    bad = _evaluation_inputs()
    bad[0] = EvaluationConditionInput(
        condition_id=bad[0].condition_id,
        truth=bad[0].truth,
        truth_row_ids=("only-one",),
        metric_control_pool_mean=bad[0].metric_control_pool_mean,
        de_gene_indices=bad[0].de_gene_indices,
        top_de_gene_indices=bad[0].top_de_gene_indices,
    )
    with pytest.raises(ValueError, match="truth row IDs"):
        seal_evaluation_bundle(
            tmp_path / "bad-rows.pkl",
            prediction=prediction,
            conditions=bad,
            systema_reference=np.zeros(3),
            provenance=_evaluation_provenance(),
        )
