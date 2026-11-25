"""Frozen C2 role verification and evaluator shared by the baseline runners."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import anndata
import numpy as np

from benchmarks.common.full_gene_evaluation import coverage, fill_missing_predictions
from benchmarks.common.study_fit_partition import crosscell_fit_rows
from benchmarks.crosscell.study_paths import COHORT_SHA, CROSS, STUDY
from gradpert.data._io import atomic_json, atomic_text, read_json
from gradpert.evaluation.crosscell import read_rows, source_reference
from gradpert.evaluation.metrics import (
    ConditionMetricResult,
    ConditionMetrics,
    compute_condition_metrics_v2,
    macro_summarize,
)
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.crosscell import FOLDS_SHA, SOURCE_SHA, frozen_rows


def cohort_inputs():
    path = STUDY / "frozen/cohorts.json"
    source = CROSS / "source/K562_cross_cell_lines/de_adata_test.h5ad"
    fold_path = CROSS / "folds-baadb85/manifest.json"
    for candidate, digest in ((path, COHORT_SHA), (source, SOURCE_SHA), (fold_path, FOLDS_SHA)):
        if sha256_file(candidate) != digest:
            raise ValueError(f"fixed study input changed: {candidate}")
    cross = read_json(path)["cross_cell"]
    cohort = cross["engineered_k562"]
    fold = read_json(fold_path)["folds"][cross["target_cell_line"]]
    archive = fold_path.parent / fold["row_index_file"]
    if not sha256_file(archive) == fold["row_index_sha256"] == cross["frozen_fold_sha256"]:
        raise ValueError("frozen fold archive changed")
    adata = anndata.read_h5ad(source, backed="r")
    try:
        if sha256_json(list(adata.var_names)) != cross["expression_gene_order_sha256"]:
            raise ValueError("frozen cross-cell expression axis changed")
        rows = frozen_rows(archive, fold, adata.obs)
    finally:
        adata.file.close()
    return source, cross, cohort, rows


def score(
    predict_controls,
    metrics,
    source,
    cross,
    cohort,
    rows,
    compatibility,
    output,
    *,
    journal_key=None,
    include_control_row_ids=False,
    stop_starting_conditions_at=None,
    score_genes=None,
    prediction_on_score_panel=False,
    record_response=None,
    full_axis=False,
    native_columns=None,
):
    ids = cohort["condition_ids"]
    draw_path = Path(cohort["shared_frozen_draws_path"])
    if sha256_file(draw_path) != cross["shared_frozen_draws_sha256"]:
        raise ValueError("paired control and truth rows changed")
    frozen = read_json(draw_path)["draws"]
    draws = {condition: frozen[condition] for condition in ids}
    if sha256_json(draws) != cohort["per_condition_pairing_sha256"]:
        raise ValueError("C2 paired test cohort changed")
    if prediction_on_score_panel and score_genes is None:
        raise ValueError("a projected predictor requires a declared score panel")
    if full_axis and score_genes is not None:
        raise ValueError("full-gene evaluation cannot score a subset")
    if journal_key is not None:
        journal_key = sha256_json(
            {
                "run": journal_key,
                "systema_reference_scope": "train_only",
                "score_genes": list(score_genes) if score_genes is not None else None,
            }
        )
    journal = output / "condition-receipts" if journal_key is not None else None
    if journal is not None:
        journal.mkdir(exist_ok=True)
    records = []
    target_line = cross["target_cell_line"]
    adata = anndata.read_h5ad(source, backed="r")
    try:
        previous_means = metrics.PERT_MEANS.pop(target_line, None)
        axis_coverage = coverage(output, adata.var_names, native_columns, 3352) if full_axis else {}
        if full_axis and journal_key is not None:
            journal_key = sha256_json([journal_key, axis_coverage["evaluation_genes_sha256"]])
        if score_genes is None:
            columns = slice(None)
        else:
            columns = adata.var_names.get_indexer(score_genes)
            if len(columns) < 2 or columns[0] < 0 or np.any(np.diff(columns) <= 0):
                raise ValueError("the scoring panel must be a unique ordered subsequence of C2")
        conditions = adata.obs["condition"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        target_controls = rows["target_control"]
        fit = crosscell_fit_rows(rows, conditions, cohort, compatibility=compatibility)
        train_rows = np.sort(fit["train"])
        if np.any(conditions[train_rows] == "ctrl"):
            raise ValueError("Systema reference must exclude controls")
        reference = source_reference(adata, conditions, train_rows)[columns]
        evaluation_rows = np.sort(
            np.concatenate([target_controls] + [d["truth_rows"] for d in draws.values()])
        )
        reference_adata = anndata.AnnData(
            X=adata.X[evaluation_rows][:, columns],
            obs=adata.obs.iloc[evaluation_rows].copy(),
            var=adata.var.iloc[columns].copy() if score_genes is not None else None,
        )
        metrics.cache_perturbation_means(reference_adata, target_line)
        if len(metrics.PERT_MEANS[target_line]) != len(ids):
            raise ValueError("official retrieval candidate count differs from the frozen cohort")
        retrieval = metrics.RetrievalMetric(subset=False)
        for index, (condition, draw) in enumerate(draws.items(), start=1):
            receipt = journal / f"{index:03d}.json" if journal is not None else None
            if receipt is not None and receipt.is_file():
                saved = read_json(receipt)
                if saved["key"] != journal_key or saved["metrics"]["condition_id"] != condition:
                    raise ValueError("scored condition receipt differs from the frozen evaluation")
                records.append(saved["metrics"])
                continue
            if (
                stop_starting_conditions_at is not None
                and time.monotonic() >= stop_starting_conditions_at
            ):
                return {"conditions_completed": len(records), "conditions_total": len(ids)}
            control_ids = np.asarray(draw["input_control_rows"])
            if len(control_ids) != 300 or not np.isin(control_ids, target_controls).all():
                raise ValueError("prediction requires 300 ordered K562 controls")
            controls = read_rows(adata, control_ids)
            prediction = (
                predict_controls(condition, controls, control_ids)
                if include_control_row_ids
                else predict_controls(condition, controls)
            )
            if full_axis and native_columns is not None:
                prediction = fill_missing_predictions(prediction, controls, native_columns)
            projected_controls = controls[:, columns]
            expected_shape = (
                projected_controls.shape if prediction_on_score_panel else controls.shape
            )
            if prediction.shape != expected_shape or not np.isfinite(prediction).all():
                raise ValueError("frozen predictions must match their declared gene axis")
            if not prediction_on_score_panel:
                prediction = prediction[:, columns]
            truth_rows = np.asarray(draw["truth_rows"])
            if not np.array_equal(truth_rows, rows["test"][conditions[rows["test"]] == condition]):
                raise ValueError(f"frozen truth IDs changed for {condition}")
            truth = read_rows(adata, truth_rows)[:, columns]
            matching = target_controls[np.isin(batches[target_controls], batches[truth_rows])]
            control_mean = np.asarray(adata.X[matching].mean(axis=0)).ravel()[columns]
            values = compute_condition_metrics_v2(
                condition_id=condition,
                prediction=prediction,
                input_control=projected_controls,
                truth=truth,
                metric_control_pool_mean=control_mean,
                de_gene_indices=(),
                systema_reference=reference,
                include_deg=False,
            )
            predicted_mean = prediction.mean(0, dtype=np.float64)
            truth_mean = truth.mean(0, dtype=np.float64)
            error = predicted_mean - truth_mean
            if record_response is not None:
                record_response(condition, predicted_mean - control_mean, truth_mean - control_mean)
            row = {
                **asdict(values),
                "mean_mse": float(np.mean(error**2)),
                "retrieval": float(
                    retrieval(
                        prediction,
                        truth,
                        target_line,
                        projected_controls,
                        reference_adata,
                        condition,
                    )
                ),
            }
            if receipt is not None:
                atomic_json(receipt, {"key": journal_key, "metrics": row})
            records.append(row)
            if receipt is not None or index == 1 or index % 10 == 0 or index == len(ids):
                atomic_json(output / "evaluation_progress.json", {"done": index, "total": len(ids)})
    finally:
        metrics.PERT_MEANS.pop(target_line, None)
        if previous_means is not None:
            metrics.PERT_MEANS[target_line] = previous_means
        adata.file.close()
    scores_path = output / "conditions.jsonl"
    rendered = "".join(
        json.dumps(row, allow_nan=False, sort_keys=journal is not None) + "\n" for row in records
    )
    if scores_path.exists():
        if scores_path.read_text() != rendered:
            raise ValueError("previously finalized condition results have changed")
    else:
        atomic_text(scores_path, rendered)
    pearson = [
        ConditionMetrics(
            row["condition_id"], tuple(ConditionMetricResult(**m) for m in row["results"])
        )
        for row in records
    ]
    return {
        **axis_coverage,
        "conditions": len(records),
        "condition_ids_sha256": cohort["condition_ids_sha256"],
        "paired_rows_sha256": cohort["per_condition_pairing_sha256"],
        "test_panel_genes": cross["expression_panel_genes"]
        if score_genes is None
        else len(score_genes),
        **(
            {
                "test_panel_gene_order_sha256": sha256_json(list(score_genes)),
                "canonical_test_panel_genes": cross["expression_panel_genes"],
            }
            if score_genes is not None
            else {}
        ),
        "test_controls_per_condition": 300,
        "test_updates": 0,
        "systema_reference_scope": "train_only",
        "systema_reference_train_rows_sha256": sha256_json(train_rows.tolist()),
        "systema_reference_condition_ids_sha256": sha256_json(
            np.unique(conditions[train_rows]).tolist()
        ),
        "mean_mse": float(np.mean([row["mean_mse"] for row in records])),
        "retrieval_mean": float(np.mean([row["retrieval"] for row in records])),
        "pearson": [asdict(value) for value in macro_summarize(pearson)],
        "conditions_file_sha256": sha256_file(scores_path),
    }
