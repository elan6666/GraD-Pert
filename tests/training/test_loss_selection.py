"""Loss selection remains independent of the legacy Pearson maximizer."""

import pytest

from gradpert.training.selection import EarlyStoppingState


def test_loss_minimizes_and_stops_after_ten_ties():
    state = EarlyStoppingState(mode="min")
    assert state.update(epoch=0, validation_metric=2.0) == (True, False)
    assert state.update(epoch=1, validation_metric=1.0) == (True, False)
    for epoch in range(2, 11):
        assert state.update(epoch=epoch, validation_metric=1.0) == (False, False)
    assert state.update(epoch=11, validation_metric=1.0) == (False, True)
    assert state.best_epoch == 1


def test_legacy_max_and_mode_validation():
    state = EarlyStoppingState()
    state.update(epoch=0, validation_metric=0.5)
    assert state.update(epoch=1, validation_metric=0.4) == (False, False)
    with pytest.raises(ValueError):
        EarlyStoppingState(mode="invalid")


def test_loss_state_roundtrip():
    from dataclasses import asdict

    state = EarlyStoppingState(mode="min")
    state.update(epoch=0, validation_metric=1.0)
    restored = EarlyStoppingState(**asdict(state))
    assert restored.update(epoch=1, validation_metric=0.5) == (True, False)


def test_unpaired_mean_expression_mse():
    import numpy as np

    from gradpert.training.validation import mean_expression_mse

    # Predicted mean [2,4], truth mean [1,2]: (1+4)/2 = 2.5.
    prediction = np.array([[1.0, 3.0], [3.0, 5.0]])
    truth = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert mean_expression_mse(prediction, truth) == 2.5
    assert mean_expression_mse(prediction, np.repeat(truth, 2, axis=0)) == 2.5
    with pytest.raises(ValueError):
        mean_expression_mse(prediction, truth[:, :1])
    with pytest.raises(RuntimeError):
        mean_expression_mse(prediction * np.nan, truth)


def test_validation_reports_both_and_macro_weights_conditions(monkeypatch):
    from types import SimpleNamespace as NS

    import numpy as np

    from gradpert.training.validation import evaluate_validation_macro_delta

    predictions = [
        NS(condition_id="a", prediction=np.array([[1.0, 3.0]]), input_control=np.zeros((1, 2))),
        NS(condition_id="b", prediction=np.array([[2.0, 4.0]]), input_control=np.zeros((1, 2))),
    ]
    truths = {"a": np.array([[1.0, 2.0]]), "b": np.array([[0.0, 2.0]] * 3)}
    monkeypatch.setattr(
        "gradpert.training.inference.iter_frozen_control_predictions", lambda **kw: predictions
    )
    data = NS(
        split_name="val",
        control_manifest=NS(split_name="val", draws=[NS(condition_id=x) for x in truths]),
        load_control_rows=lambda ids: None,
        load_truth_rows=lambda c: NS(expression=truths[c]),
    )
    result = evaluate_validation_macro_delta(
        model=None,
        topology=None,
        data=data,
        anchors_by_condition={"a": (0,), "b": (1,)},
        device=None,
        decode_batch_size=256,
        prediction_view=object(),
    )
    assert result.prediction_loss == 2.25  # (0.5 + 4) / 2, not cell-weighted.
    assert result.txpert_macro_pearson_delta == pytest.approx(1.0)
    assert result.finite_condition_count == 2
    data.split_name = "test"
    with pytest.raises(ValueError, match="validation data only"):
        evaluate_validation_macro_delta(
            model=None,
            topology=None,
            data=data,
            anchors_by_condition={},
            device=None,
            decode_batch_size=256,
        )


def test_validation_three_metrics_match_canonical_function(monkeypatch):
    from types import SimpleNamespace as NS

    import numpy as np

    from gradpert.evaluation.metrics import compute_condition_metrics
    from gradpert.training.validation import evaluate_validation_macro_delta

    pred = np.tile([1.0, 3.0, 2.0], (300, 1))
    ctrl = np.zeros_like(pred)
    truth = np.tile([1.0, 2.0, 4.0], (4, 1))
    item = NS(condition_id="a", prediction=pred, input_control=ctrl)
    monkeypatch.setattr(
        "gradpert.training.inference.iter_frozen_control_predictions", lambda **kw: [item]
    )
    data = NS(
        split_name="val",
        split=NS(train_conditions=["train"]),
        control_manifest=NS(split_name="val", draws=[NS(condition_id="a")]),
        load_control_rows=lambda ids: None,
        load_truth_rows=lambda c: NS(expression=truth),
    )
    state = NS(
        manifest=NS(
            condition_ids=["a"],
            systema_reference_condition_ids=["train"],
            de_gene_indices={"a": [0, 1, 2]},
            top_de_gene_indices={"a": [0, 1, 2]},
            de_unavailable_reasons={},
        ),
        metric_control_means=np.zeros((1, 3)),
        systema_reference=np.ones(3),
        manifest_file_sha256="fixture",
    )
    result = evaluate_validation_macro_delta(
        model=None,
        topology=None,
        data=data,
        anchors_by_condition={"a": (0,)},
        device=None,
        decode_batch_size=256,
        prediction_view=object(),
        evaluation_state=state,
    )
    expected = compute_condition_metrics(
        condition_id="a",
        prediction=pred,
        input_control=ctrl,
        truth=truth,
        metric_control_pool_mean=np.zeros(3),
        de_gene_indices=[0, 1, 2],
        top_de_gene_indices=[0, 1, 2],
        systema_reference=np.ones(3),
    )
    for metric in expected.results:
        assert getattr(result, metric.metric_id) == pytest.approx(metric.value)
    assert len(result.metric_details) == 3
    state.manifest.condition_ids.append("test")
    with pytest.raises(ValueError, match="only ordered validation"):
        evaluate_validation_macro_delta(
            model=None,
            topology=None,
            data=data,
            anchors_by_condition={"a": (0,)},
            device=None,
            decode_batch_size=256,
            prediction_view=object(),
            evaluation_state=state,
        )
