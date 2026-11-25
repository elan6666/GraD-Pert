"""Shared frozen Jurkat J2 populations, supported expression axis, and scorer."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.common.condition_response import FrozenConditionResponse, summarize_response
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


def evaluate(predict, output, *, deadline=None, journal_identity=None):
    """Score the common J2 axis. The model receives fixed original controls."""
    reference, systema = references()
    with context() as data:
        axis = score_axis(data.expression_gene_ids)
        genes = [data.expression_gene_ids[i] for i in axis]
        if (
            systema.manifest.systema_reference_condition_ids != list(data.split.train_conditions)
            or data.control_manifest_file_sha256 != cohort()["evaluation_controls_sha256"]
            or len(data.control_manifest.draws) != EXPECTED_CONDITIONS
        ):
            raise ValueError("the train-only reference or held-out J2 cohort changed")
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
            ]
        )
        values, records = [], []
        for i, draw in enumerate(data.control_manifest.draws, start=1):
            name = draw.condition_id
            saved = journal / f"{i:03d}.json"
            if journal_identity is not None and saved.is_file():
                from gradpert.evaluation.metrics import ConditionMetricResult, ConditionMetrics

                row = read_json(saved)
                if row.get("key") != key or row["condition_id"] != name:
                    raise ValueError("per-condition receipt identity changed")
                metric = ConditionMetrics(
                    name, tuple(ConditionMetricResult(**item) for item in row["results"])
                )
            else:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"J2 scoring stopped after {i - 1} complete conditions")
                ids = tuple(draw.ordered_row_ids)
                controls = data.load_control_rows(ids)
                if controls.ordered_row_ids != ids or len(ids) != 300:
                    raise ValueError("the frozen J2 controls changed")
                full = controls.expression
                predicted = np.asarray(predict(name, full, ids, axis), dtype=np.float32)
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
            records.append(row)
            values.append(metric)
            if i == 1 or i % 5 == 0 or i == EXPECTED_CONDITIONS:
                atomic_json(
                    Path(output) / "evaluation_progress.json",
                    {"done": i, "total": EXPECTED_CONDITIONS},
                )
        import json

        csv_path = Path(output) / "conditions.jsonl"
        atomic_text(csv_path, "".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        response = summarize_response(records)
        return {
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
