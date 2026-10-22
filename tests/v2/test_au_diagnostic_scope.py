"""Small probes cannot masquerade as effective-batch AU objectives."""

import copy

import torch
from test_components import fixture

from gradpert.training.v2.diagnostic_objective import diagnostic_objective
from gradpert.training.v2.joint_validation import evaluate_joint_loss
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.process_diagnostics import _read_only


def objects():
    model, batch = fixture()
    objective = JointObjective(
        model,
        loss_reduction="row_mean",
        ssl2_weights=(1, 1, 0),
        au_alignment_weight=0.1,
        au_uniformity_weight=0.01,
    ).train()
    return objective, batch


def test_partial_au_probe_omits_terms_without_requiring_context_or_changing_state():
    objective, batch = objects()
    before = copy.deepcopy(objective.state_dict())
    rng = torch.get_rng_state().clone()
    with _read_only(objective, batch, 431):
        loss, terms, metadata = diagnostic_objective(objective, batch)
        assert loss.requires_grad and torch.isfinite(loss)
        assert "au_alignment" not in terms and "au_uniformity" not in terms
        assert metadata["diagnostic_objective_scope"]["unavailable_terms"] == {
            "au_alignment": "requires_complete_global_population",
            "au_uniformity": "requires_complete_global_population",
        }
        assert "prediction" in terms and "ssl2_dino" in terms
    assert torch.equal(rng, torch.get_rng_state())
    assert not objective.pending and all(p.grad is None for p in objective.parameters())
    for key, value in before.items():
        torch.testing.assert_close(value, objective.state_dict()[key], rtol=0, atol=0)


def test_validation_probe_explicitly_excludes_au_and_preserves_teacher_and_rng():
    objective, batch = objects()
    before = copy.deepcopy(objective.state_dict())
    rng = torch.get_rng_state().clone()
    result = evaluate_joint_loss(objective, [(batch, "fixed-population")], bf16=False)
    assert result["batch_count"] == 1 and result["teacher_update"] == "none"
    assert set(result["diagnostic_objective_scope"]["unavailable_terms"]) == {
        "au_alignment",
        "au_uniformity",
    }
    assert "au_uniformity" not in result["components"]
    assert torch.equal(rng, torch.get_rng_state()) and objective.training
    assert not objective.pending
    for key, value in before.items():
        torch.testing.assert_close(value, objective.state_dict()[key], rtol=0, atol=0)
