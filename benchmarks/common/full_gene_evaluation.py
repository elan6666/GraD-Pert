"""Shared full-gene scoring and frozen checkpoint provenance for the holdout study."""

from pathlib import Path

import numpy as np

from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file, sha256_json

IMPUTATION_RULE = "mean_of_fixed_condition_input_controls"


def fill_missing_predictions(prediction, controls, native_columns):
    """Use the same condition's input controls to fill unsupported output genes."""
    controls = np.asarray(controls)
    prediction = np.asarray(prediction)
    columns = np.asarray(native_columns)
    if (
        controls.ndim != 2
        or controls.shape[0] != 300
        or columns.ndim != 1
        or not np.issubdtype(columns.dtype, np.integer)
        or prediction.shape != (300, len(columns))
        or np.any(columns < 0)
        or np.any(columns >= controls.shape[1])
        or len(np.unique(columns)) != len(columns)
        or not np.isfinite(prediction).all()
        or not np.isfinite(controls).all()
    ):
        raise ValueError("invalid native predictions or frozen input controls")
    means = controls.mean(axis=0, dtype=np.float64)
    filled = np.broadcast_to(means, controls.shape).astype(np.float32, copy=True)
    filled[:, columns] = prediction
    return filled


def coverage(output, genes, native_columns, expected_count):
    """Publish the ordered axis and the mask alongside the metric receipts."""
    names = list(genes)
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError("the full canonical expression axis changed")
    native = (
        list(range(expected_count)) if native_columns is None else list(map(int, native_columns))
    )
    if native != sorted(set(native)) or not all(0 <= i < expected_count for i in native):
        raise ValueError("native coverage must be an ordered canonical panel")
    covered = set(native)
    missing = [gene for i, gene in enumerate(names) if i not in covered]
    path = Path(output) / "evaluation_genes.json"
    record = {
        "genes_in_expression_order": names,
        "native_genes_in_expression_order": [names[i] for i in native],
        "imputed_genes_in_expression_order": missing,
        "imputation_rule": IMPUTATION_RULE,
    }
    if path.exists() and read_json(path) != record:
        raise ValueError("the resumed full-gene evaluation axis or mask changed")
    atomic_json(path, record)
    return {
        "score_scope": "full_gene_axis",
        "test_panel_gene_order_sha256": sha256_json(names),
        "canonical_test_panel_genes": expected_count,
        "native_prediction_genes": len(native),
        "imputed_prediction_genes": len(missing),
        "imputed_genes_in_expression_order": missing,
        "imputation_rule": IMPUTATION_RULE,
        "evaluation_genes_sha256": sha256_file(path),
    }


def verify_replay(origin, identity, checkpoint, *, conditions):
    """Bind an evaluation to its accepted original run and untouched weights."""
    original = read_json(origin / "result.json")
    expected = {
        "status": "evaluated",
        "formal_eligible": True,
        "conditions": conditions,
        "test_controls_per_condition": 300,
        "test_updates": 0,
        "checkpoint_sha256": sha256_file(origin / checkpoint),
        **{key: identity[key] for key in ("scenario", "model", "official_commit", "cohort_sha256")},
    }
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError(f"the accepted frozen run no longer matches {origin.name}")
    return original


def replay_fields(origin, identity):
    return {
        "run_id": identity["run_id"],
        "original_run_id": origin.name,
        "original_receipt_sha256": sha256_file(origin / "result.json"),
        "evaluation_source_commit": identity["source_commit"],
        "replay_optimizer_updates": 0,
    }
