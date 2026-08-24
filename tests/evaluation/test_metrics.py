from __future__ import annotations

import numpy as np
import pytest

from gradpert.evaluation import (
    build_systema_reference,
    compute_condition_metrics,
    macro_summarize,
    pearson_correlation,
)


def _repeat(vector: list[float], rows: int = 300) -> np.ndarray:
    return np.repeat(np.asarray([vector], dtype=np.float64), rows, axis=0)


def test_pearson_reports_undefined_inputs_instead_of_coercing_to_zero() -> None:
    assert pearson_correlation(np.array([1.0]), np.array([2.0])) == (
        None,
        "fewer_than_two_genes",
    )
    assert pearson_correlation(np.ones(3), np.arange(3.0)) == (
        None,
        "constant_vector",
    )
    assert pearson_correlation(np.array([1.0, np.nan]), np.ones(2)) == (
        None,
        "non_finite_input",
    )


def test_three_headline_metrics_keep_their_distinct_reference_contracts() -> None:
    prediction = _repeat([3.0, 5.0, 4.0, 8.0])
    input_control = _repeat([1.0, 1.0, 2.0, 2.0])
    truth = _repeat([4.0, 3.0, 6.0, 7.0], rows=7)
    metric_control = np.asarray([0.0, 2.0, 3.0, 1.0])
    systema_reference = np.asarray([2.0, 1.0, 1.0, 5.0])

    result = compute_condition_metrics(
        condition_id="PERT_A",
        prediction=prediction,
        input_control=input_control,
        truth=truth,
        metric_control_pool_mean=metric_control,
        de_gene_indices=[0, 1, 3],
        top_de_gene_indices=[1, 2, 3],
        systema_reference=systema_reference,
    )

    expected_txpert = np.corrcoef(
        prediction.mean(axis=0) - input_control.mean(axis=0),
        truth.mean(axis=0) - input_control.mean(axis=0),
    )[0, 1]
    expected_trishift = np.corrcoef(
        (prediction.mean(axis=0) - metric_control)[[0, 1, 3]],
        (truth.mean(axis=0) - metric_control)[[0, 1, 3]],
    )[0, 1]
    expected_systema = np.corrcoef(
        (prediction.mean(axis=0) - systema_reference)[[1, 2, 3]],
        (truth.mean(axis=0) - systema_reference)[[1, 2, 3]],
    )[0, 1]

    assert [item.metric_id for item in result.results] == [
        "txpert_macro_pearson_delta",
        "trishift_pearson_delta",
        "systema_pearson",
    ]
    assert [item.gene_count for item in result.results] == [4, 3, 3]
    assert [item.value for item in result.results] == pytest.approx(
        [expected_txpert, expected_trishift, expected_systema]
    )
    assert len({round(float(item.value), 8) for item in result.results}) == 3


def test_systema_reference_weights_conditions_not_cells() -> None:
    reference = build_systema_reference(
        {
            "condition_b": _repeat([10.0, 2.0], rows=9),
            "condition_a": _repeat([2.0, 6.0], rows=1),
        }
    )

    np.testing.assert_allclose(reference, [6.0, 4.0])


def test_macro_summary_preserves_denominator_and_reasons() -> None:
    valid = compute_condition_metrics(
        condition_id="valid",
        prediction=_repeat([1.0, 2.0, 4.0]),
        input_control=_repeat([0.0, 0.0, 0.0]),
        truth=_repeat([1.0, 2.0, 4.0], rows=2),
        metric_control_pool_mean=np.zeros(3),
        de_gene_indices=[0, 1, 2],
        top_de_gene_indices=[0, 1, 2],
        systema_reference=np.zeros(3),
    )
    invalid = compute_condition_metrics(
        condition_id="invalid",
        prediction=_repeat([2.0, 2.0, 2.0]),
        input_control=_repeat([0.0, 0.0, 0.0]),
        truth=_repeat([1.0, 2.0, 4.0], rows=2),
        metric_control_pool_mean=np.zeros(3),
        de_gene_indices=[0, 1, 2],
        top_de_gene_indices=[0, 1, 2],
        systema_reference=np.zeros(3),
    )

    summaries = macro_summarize([valid, invalid])

    for summary in summaries:
        assert summary.macro_mean == pytest.approx(1.0)
        assert summary.finite_condition_count == 1
        assert summary.total_condition_count == 2
        assert summary.unavailable_reasons == ("constant_vector",)


def test_de_metrics_are_explicitly_unavailable_when_t_test_cannot_be_ranked() -> None:
    result = compute_condition_metrics(
        condition_id="single-cell-condition",
        prediction=_repeat([1.0, 2.0, 4.0]),
        input_control=_repeat([0.0, 0.0, 0.0]),
        truth=_repeat([1.0, 2.0, 4.0], rows=1),
        metric_control_pool_mean=np.zeros(3),
        de_gene_indices=[],
        top_de_gene_indices=[],
        systema_reference=np.zeros(3),
        de_unavailable_reason="insufficient_truth_cells_for_t_test:n=1",
    )

    assert result.results[0].value == pytest.approx(1.0)
    assert [item.value for item in result.results[1:]] == [None, None]
    assert [item.gene_count for item in result.results[1:]] == [0, 0]
    assert all(
        item.reason == "de_unavailable:insufficient_truth_cells_for_t_test:n=1"
        for item in result.results[1:]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prediction", np.ones((299, 3)), "preserve 300 rows"),
        ("input_control", np.ones((301, 3)), "preserve 300 rows"),
        ("de_gene_indices", [0, 0], "duplicate"),
        ("top_de_gene_indices", [0, 3], "out-of-range"),
    ],
)
def test_condition_metric_contract_rejects_invalid_shapes_and_indices(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "condition_id": "PERT_A",
        "prediction": np.ones((300, 3)),
        "input_control": np.ones((300, 3)),
        "truth": np.ones((5, 3)),
        "metric_control_pool_mean": np.zeros(3),
        "de_gene_indices": [0, 1],
        "top_de_gene_indices": [1, 2],
        "systema_reference": np.zeros(3),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        compute_condition_metrics(**arguments)  # type: ignore[arg-type]
