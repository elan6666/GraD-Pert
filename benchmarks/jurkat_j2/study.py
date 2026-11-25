"""Shared frozen Jurkat J2 populations, supported expression axis, and scorer."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.common.condition_response import FrozenConditionResponse, summarize_response
from benchmarks.common.full_gene_evaluation import coverage, fill_missing_predictions
from benchmarks.common.study_fit_partition import open_jurkat_double_holdout_fit
from benchmarks.crosscell.remaining_c2_readiness import hgnc_links, reconcile
from benchmarks.crosscell.study_paths import COHORT_SHA, DATA, DATASET, PROTOCOL, SERVER, STUDY
from benchmarks.txpert.runner import _register_anndata_null_reader
from gradpert.data._io import atomic_json, atomic_text, read_json
from gradpert.evaluation.data import CanonicalEvaluationData, LoadedTruthRows
from gradpert.evaluation.metrics import compute_condition_metrics_v2, macro_summarize
from gradpert.evaluation.state import load_evaluation_state
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData

COHORT = STUDY / "frozen/cohorts.json"
VOCABULARY = SERVER / "upstreams/scfoundation-c2-study/model/OS_scRNA_gene_index.19264.tsv"
EXPECTED_CONDITIONS = 592


class EvaluationBudgetExpired(TimeoutError):
    """A segment used its budget between two complete conditions."""


def fit_data(seed):
    _register_anndata_null_reader()
    with CanonicalTrainingData(
        data_root=DATA, dataset_id=DATASET, protocol_id=PROTOCOL, run_seed=seed
    ) as data:
        return open_jurkat_double_holdout_fit(data, cohort_path=COHORT, cohort_sha256=COHORT_SHA)


def cohort():
    if sha256_file(COHORT) != COHORT_SHA:
        raise ValueError("the sealed Jurkat cohort changed")
    return read_json(COHORT)["jurkat"]


def score_axis(genes):
    from benchmarks.crosscell.remaining_c2_readiness import vocab_names

    contract = cohort()
    if sha256_json(list(genes)) != contract["expression_gene_order_sha256"]:
        raise ValueError("the canonical Jurkat axis changed")
    allowed = set(genes) - set(contract["masked_genes_in_expression_order"])
    audit = reconcile(allowed, vocab_names(VOCABULARY), hgnc_links(), verified_aliases={})
    if audit["collisions"] or audit["mapped_names"] != 4486:
        raise ValueError("the common J2 projection no longer matches its coverage audit")
    return np.flatnonzero(np.isin(genes, sorted(allowed - set(audit["unresolved"]))))


def context():
    _register_anndata_null_reader()
    return CanonicalEvaluationData(
        data_root=DATA, dataset_id=DATASET, protocol_id=PROTOCOL, split_name="test"
    )


def references():
    args = dict(data_root=DATA, dataset_id=DATASET, protocol_id=PROTOCOL, evaluation_protocol="v2")
    return load_evaluation_state(**args), load_evaluation_state(validation_only=True, **args)


def frozen_receipt(output, identity, checkpoint_name, *, replay=False):
    if not replay and (output / "result.json").exists():
        raise FileExistsError("the frozen J2 run already has a result")
    receipt = read_json(output / "training_result.json")
    if (
        any(
            receipt.get(key) != value
            for key, value in identity.items()
            if key not in {"gpu", "source_commit"}
        )
        or receipt.get("test_updates") != 0
        or sha256_file(output / checkpoint_name) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("the frozen J2 training receipt or checkpoint changed")
    return receipt


def evaluate_segment(predict, output, receipt, *, deadline, full_axis=False, native_columns=None):
    try:
        return evaluate(
            predict,
            output,
            deadline=deadline,
            journal_identity=receipt["checkpoint_sha256"],
            full_axis=full_axis,
            native_columns=native_columns,
        )
    except EvaluationBudgetExpired:
        progress = read_json(output / "evaluation_progress.json")
        atomic_json(
            output / "segment.json",
            {
                "status": "evaluation_segment_complete",
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "conditions_completed": progress["done"],
                "conditions_total": progress["total"],
                "test_updates": 0,
            },
        )
        return None


def evaluate(
    predict,
    output,
    *,
    deadline=None,
    journal_identity=None,
    assist=None,
    full_axis=False,
    native_columns=None,
):
    """Score J2 with fixed original input controls and train-only references."""
    reference, systema = references()
    with context() as data:
        axis = (
            np.arange(len(data.expression_gene_ids))
            if full_axis
            else score_axis(data.expression_gene_ids)
        )
        genes = [data.expression_gene_ids[i] for i in axis]
        if (
            systema.manifest.systema_reference_condition_ids != list(data.split.train_conditions)
            or data.control_manifest_file_sha256 != cohort()["evaluation_controls_sha256"]
            or len(data.control_manifest.draws) != EXPECTED_CONDITIONS
        ):
            raise ValueError("the train-only reference or held-out J2 cohort changed")
        axis_coverage = coverage(output, genes, native_columns, 5000) if full_axis else {}
        responses = FrozenConditionResponse(data, columns=axis)
        baseline = systema.systema_reference[axis]
        control_centers = dict(
            zip(
                reference.manifest.condition_ids,
                reference.metric_control_means[:, axis],
                strict=True,
            )
        )
        journal = Path(output) / "condition-receipts"
        journal.mkdir(exist_ok=True)
        key = sha256_json(
            [
                journal_identity,
                sha256_json(genes),
                systema.manifest.systema_reference_content_sha256,
                data.control_manifest_file_sha256,
                *([axis_coverage["evaluation_genes_sha256"]] if full_axis else []),
            ]
        )
        values, records = [], []
        draws = list(enumerate(data.control_manifest.draws, start=1))
        if assist is not None:
            shared, start = assist
            shared = Path(shared) / "condition-receipts"
            if not journal_identity or not shared.is_dir() or not 1 <= start <= len(draws):
                raise ValueError("invalid checkpoint-bound evaluation partition")
            draws = draws[start - 1 :][::-1]

        def saved_metric(path, expected_name):
            from gradpert.evaluation.metrics import ConditionMetricResult, ConditionMetrics

            saved_row = read_json(path)
            if saved_row.get("key") != key or saved_row["condition_id"] != expected_name:
                raise ValueError("per-condition receipt identity changed")
            return saved_row, ConditionMetrics(
                expected_name, tuple(ConditionMetricResult(**item) for item in saved_row["results"])
            )

        def report_progress(done):
            atomic_json(
                Path(output) / "evaluation_progress.json",
                {"done": done, "total": len(draws)},
            )

        for i, draw in draws:
            name = draw.condition_id
            saved = journal / f"{i:03d}.json"
            existing = shared / saved.name if assist is not None else saved
            if journal_identity is not None and existing.is_file():
                row, metric = saved_metric(existing, name)
            else:
                if deadline is not None and time.monotonic() >= deadline:
                    report_progress(len(records))
                    raise EvaluationBudgetExpired(
                        f"J2 scoring stopped after {len(records)} complete conditions"
                    )
                ids = tuple(draw.ordered_row_ids)
                controls = data.load_control_rows(ids)
                if controls.ordered_row_ids != ids or len(ids) != 300:
                    raise ValueError("the frozen J2 controls changed")
                full = controls.expression
                predicted = np.asarray(predict(name, full, ids, axis), dtype=np.float32)
                if full_axis and native_columns is not None:
                    predicted = fill_missing_predictions(predicted, full, native_columns)
                truth = data.load_truth_rows(name)
                projected = truth.expression[:, axis]
                metric = compute_condition_metrics_v2(
                    condition_id=name,
                    prediction=predicted,
                    input_control=full[:, axis],
                    truth=projected,
                    metric_control_pool_mean=control_centers[name],
                    systema_reference=baseline,
                    include_deg=False,
                    de_gene_indices=(),
                )
                response = responses(
                    name,
                    predicted,
                    full[:, axis],
                    LoadedTruthRows(name, truth.ordered_row_ids, projected),
                )
                row = {**asdict(metric), "response_metrics": response, "key": key}
                atomic_json(saved, row)
                if assist is not None:
                    try:
                        os.link(saved, existing)
                    except FileExistsError:
                        saved_metric(existing, name)
                elif i == 1 or i % 5 == 0:
                    report_progress(i)
            records.append(row)
            values.append(metric)
            if assist is not None and (len(records) == 1 or len(records) % 5 == 0):
                report_progress(len(records))
        report_progress(len(records))
        if assist is not None:
            return {
                "condition_partition_start": start,
                "condition_partition_stop": EXPECTED_CONDITIONS,
                "conditions": len(records),
                "systema_reference_scope": "train_only",
                "systema_reference_sha256": systema.manifest.systema_reference_content_sha256,
                "test_updates": 0,
            }
        import json

        csv_path = Path(output) / "conditions.jsonl"
        atomic_text(csv_path, "".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        response = summarize_response(records)
        return {
            **axis_coverage,
            "conditions": len(values),
            "test_controls_per_condition": 300,
            "test_panel_genes": len(genes),
            "test_panel_gene_order_sha256": sha256_json(genes),
            "condition_ids_sha256": sha256_json(list(data.split.test_conditions)),
            "paired_rows_sha256": data.control_manifest_file_sha256,
            "systema_reference_scope": "train_only",
            "systema_reference_sha256": systema.manifest.systema_reference_content_sha256,
            "systema_reference_condition_ids_sha256": (
                systema.manifest.systema_reference_condition_ids_sha256
            ),
            "conditions_file_sha256": sha256_file(csv_path),
            "shared_fit_support_file_sha256": data.split.split_content_sha256,
            "test_updates": 0,
            "pearson": [asdict(item) for item in macro_summarize(values)],
            "mean_mse": response["condition_mean_mse"]["macro_mean"],
            "retrieval_mean": response["txpert_normalized_retrieval"]["macro_mean"],
        }
