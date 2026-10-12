from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.expression_holdout.evaluation import evaluate
from gradpert.evaluation.metrics import compute_condition_metrics_v2, macro_summarize


def fixtures():
    genes = ("A", "B", "C", "T", "U")
    ids = tuple(f"c{i}" for i in range(300))
    control = np.arange(1500, dtype=np.float32).reshape(300, 5) / 100
    truth = np.array([[2, 3, 7, 1, 5], [3, 6, 2, 4, 9]], dtype=np.float32)
    draw = SimpleNamespace(condition_id="T+ctrl", ordered_row_ids=ids)
    data = SimpleNamespace(
        split_name="test",
        expression_gene_ids=genes,
        split=SimpleNamespace(
            test_conditions=["T+ctrl"], control_condition_id="ctrl", split_content_sha256="s"
        ),
        manifest=SimpleNamespace(canonical_adata_sha256="a"),
        control_manifest=SimpleNamespace(draws=[draw]),
        control_manifest_file_sha256="c",
        load_control_rows=lambda supplied: SimpleNamespace(
            ordered_row_ids=supplied, expression=control
        ),
        load_truth_rows=lambda cond: SimpleNamespace(
            ordered_row_ids=("t0", "t1"), expression=truth
        ),
    )
    manifest = SimpleNamespace(
        schema_version="evaluation-state-v2",
        condition_ids=["T+ctrl"],
        de_gene_indices={"T+ctrl": [0, 1, 3, 4]},
        de_unavailable_reasons={},
        systema_reference_content_sha256="sr",
    )
    ref = SimpleNamespace(
        manifest=manifest,
        metric_control_means=np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]),
        systema_reference=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        manifest_file_sha256="e",
    )
    return data, ref, control, truth


def test_all18_metrics_are_exact_common_evaluator_values():
    data, ref, control, truth = fixtures()
    prediction = (control + np.array([2.0, -1.0, 4.0, 1.0, -2.0])).astype(np.float32)
    calls = []

    def predict(x, targets, genes, **kwargs):
        assert targets == ("T",) and genes == data.expression_gene_ids
        np.testing.assert_array_equal(x, control)
        calls.append(kwargs)
        return prediction

    result = evaluate(
        data, ref, predict, seen_indices=(0, 1, 2), cell_batch=128, query_count=1000, seed=1
    )
    expected = compute_condition_metrics_v2(
        condition_id="T+ctrl",
        prediction=prediction,
        input_control=control,
        truth=truth,
        metric_control_pool_mean=ref.metric_control_means[0],
        de_gene_indices=[0, 1, 3, 4],
        systema_reference=ref.systema_reference,
    )
    assert result["summary"] == [asdict(x) for x in macro_summarize([expected])]
    assert len(result["summary"]) == 6
    assert len(result["gene_group_summaries"]["seen_expression"]) == 6
    assert len(result["gene_group_summaries"]["unseen_expression"]) == 6
    assert result["condition_metrics"][0]["control_count"] == 300
    assert calls and result["evaluation_state_sha256"] == "e"


def test_bad_axis_or_missing_prediction_rejected():
    data, ref, control, _ = fixtures()
    with pytest.raises(ValueError, match="actual training exposure"):
        evaluate(
            data,
            ref,
            lambda *a, **k: control,
            seen_indices=(2, 1),
            cell_batch=1,
            query_count=1000,
            seed=1,
        )
    with pytest.raises(ValueError, match="complete finite"):
        evaluate(
            data,
            ref,
            lambda *a, **k: control[:, :4],
            seen_indices=(0, 1),
            cell_batch=1,
            query_count=1000,
            seed=1,
        )
