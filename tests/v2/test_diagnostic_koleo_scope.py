"""Small diagnostic probes explicitly omit global KoLeo without changing training."""

import copy
import json
from dataclasses import replace

import pytest
import torch
from test_components import fixture

from gradpert.training.v2.diagnostic_objective import diagnostic_objective
from gradpert.training.v2.loss_diagnostics import (
    evaluate_loss_diagnostics,
    evaluate_ssl_diagnostics,
)
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.process_diagnostics import (
    _read_only,
    evaluate_graph_gradient_diagnostics,
    evaluate_process_diagnostics,
)


def single_condition(*, weight=0.1, reduction="row_mean"):
    model, batch = fixture()
    obj = JointObjective(
        model,
        ssl2_weights=(1, 1, weight),
        loss_reduction=reduction,
        koleo_exclude_same_condition=reduction is not None,
    ).train()
    return obj, replace(batch, condition_index=torch.zeros_like(batch.condition_index))


def assert_scope(result):
    scope = result["diagnostic_objective_scope"]
    assert scope["unavailable_terms"] == {"ssl2_koleo": "requires_complete_global_population"}
    assert scope["joint_scope"] == "prediction_and_ssl_excluding_unavailable_koleo"
    assert scope["ssl2_gradient_scope"] == "dino_and_ibot_excluding_unavailable_koleo"
    assert "ssl2_koleo" not in result.get("components", {})
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "diagnose",
    [evaluate_loss_diagnostics, evaluate_process_diagnostics, evaluate_graph_gradient_diagnostics],
)
def test_single_condition_probes_work_with_explicit_unavailable_scope_and_no_mutation(diagnose):
    obj, batch = single_condition()
    before = copy.deepcopy(obj.state_dict())
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in obj.modules()]
    result = diagnose(obj, batch, bf16=False)
    assert_scope(result)
    assert modes == [m.training for m in obj.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert not obj.pending and all(p.grad is None for p in obj.parameters())
    for key, value in before.items():
        torch.testing.assert_close(value, obj.state_dict()[key], rtol=0, atol=0)
    # The diagnostic must not disable the real training neighborhood rule.
    with pytest.raises(ValueError, match="unlike-condition neighbor"):
        obj(batch)
    obj.pending.clear()


def test_ssl_only_probe_also_marks_missing_global_koleo():
    obj, batch = single_condition()
    obj.prediction_enabled = False
    obj.decode_ssl_views = False
    before = copy.deepcopy(obj.state_dict())
    rng = torch.get_rng_state().clone()
    result = evaluate_ssl_diagnostics(obj, batch, bf16=False)
    assert_scope(result)
    assert result["training_stage"] == "ssl_pretrain"
    assert torch.equal(rng, torch.get_rng_state())
    assert not obj.pending
    for key, value in before.items():
        torch.testing.assert_close(value, obj.state_dict()[key], rtol=0, atol=0)


def test_disabled_koleo_exact_original_forward_and_metadata():
    obj, batch = single_condition(weight=0)
    with _read_only(obj, batch, 431):
        original, original_terms = obj(batch)
    with _read_only(obj, batch, 431):
        actual, terms, metadata = diagnostic_objective(obj, batch)
    assert metadata == {}
    assert terms.keys() == original_terms.keys()
    torch.testing.assert_close(original, actual, rtol=0, atol=0)
    for key, value in terms.items():
        torch.testing.assert_close(value, original_terms[key], rtol=0, atol=0)


def test_legacy_nonunified_koleo_is_rejected_before_any_forward(monkeypatch):
    obj, batch = single_condition(reduction=None)
    monkeypatch.setattr(obj, "forward", lambda *a, **k: pytest.fail("forward must not execute"))
    with pytest.raises(ValueError, match="requires unified loss_reduction"):
        diagnostic_objective(obj, batch)
    assert not obj.pending


def test_deferred_diagnostic_loss_equals_available_terms_and_retains_dino_gradients():
    obj, batch = single_condition()
    with _read_only(obj, batch, 431):
        loss, terms, metadata = diagnostic_objective(obj, batch)
        expected = terms["prediction"]
        for index, names in enumerate((("condition", "node", "spread"), ("dino", "ibot"))):
            coefficient = (obj.lambda1, obj.lambda2)[index]
            for weight, name in zip(obj.weights[index][: len(names)], names, strict=True):
                expected = expected + coefficient * weight * terms[f"ssl{index + 1}_{name}"]
        torch.testing.assert_close(loss, expected)
        gradients = torch.autograd.grad(
            terms["ssl2_dino"], tuple(obj.student.ssl2_cls.parameters()), allow_unused=True
        )
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
        assert_scope(metadata)
        assert "ssl2_koleo" not in terms
