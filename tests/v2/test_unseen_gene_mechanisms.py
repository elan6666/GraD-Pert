"""Unseen-expression mechanisms: identity, initialization and update contracts."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_relay_method import (
    assert_tree_close,
    relay_training_fixture,
    small_architecture,
    small_index,
)

from gradpert.config.v2 import V2Architecture
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.evaluation import predict_query_set
from gradpert.training.v2.loss_diagnostics import evaluate_loss_diagnostics
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer, routes

FLAGS = ("prior_shared_adapter", "gene_conditioned_readout", "direct_target_flag")


def model_for(flag=None):
    torch.manual_seed(291)
    seeds = torch.randn(9, 8)
    arch = replace(small_architecture(), relay_passes=1)
    return GraDPertV2(seeds, replace(arch, **({flag: True} if flag else {})))


@pytest.mark.parametrize("flag", FLAGS)
def test_flags_are_strict_optional_architecture_identity(flag):
    assert flag not in V2Architecture().payload()
    with pytest.raises(ValueError, match="boolean"):
        replace(small_architecture(), **{flag: 1})
    with pytest.raises(ValueError, match="relay"):
        V2Architecture(**{flag: True})
    assert model_for(flag).options.payload()[flag] is True


@pytest.mark.parametrize("flag", FLAGS)
def test_zero_initialization_preserves_backbone_rng_and_starting_prediction(flag):
    baseline = model_for().eval()
    rng = torch.get_rng_state().clone()
    model = model_for(flag).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in baseline.state_dict().items():
        actual = model.state_dict()[key]
        if flag == "direct_target_flag" and key == "prediction.0.weight":
            actual = actual[:, :-1]
        torch.testing.assert_close(actual, value, atol=0, rtol=0)
    gene, control, condition = torch.randn(3, 8), torch.randn(2, 3), torch.randn(2, 8)
    metadata = model.prediction_metadata(torch.tensor([7, 2, 5]), torch.tensor([[2, 8], [5, -1]]))
    expected = baseline.encode_response(gene, control, condition)
    actual = model.encode_response(gene, control, condition, **metadata)
    assert_tree_close(actual, expected)


def test_gene_conditioned_readout_uses_initial_prior_on_explicit_reordered_ids():
    model = model_for("gene_conditioned_readout")
    prior = model.readout_prior.clone()
    with torch.no_grad():
        model.graph.embedding.weight.add_(100)
        model.prior_readout[2].weight.fill_(0.03)
    assert torch.equal(model.readout_prior, prior)
    joint = torch.randn(2, 3, 16, requires_grad=True)
    ids = torch.tensor([7, 2, 5])
    features = model.prediction[1](model.prediction[0](joint))
    correction = model.prior_readout(prior[ids])
    expected = model.prediction[2](features).squeeze(-1) + torch.einsum(
        "bgd,gd->bg", features, correction
    )
    actual = model.decode_delta(joint, query_gene_ids=ids)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model.prior_readout[2].weight.grad.abs().sum() > 0
    assert model.readout_prior.grad is None
    assert model.graph.embedding.weight.requires_grad
    with pytest.raises(ValueError, match="explicit"):
        model.decode_delta(joint)


def test_target_flags_use_global_ids_valid_padding_and_all_targets():
    model = model_for("direct_target_flag")
    kwargs = model.prediction_metadata(torch.tensor([7, 2, 5]), torch.tensor([[2, 8], [5, -1]]))
    assert torch.equal(
        kwargs["target_mask"], torch.tensor([[False, True, False], [False, False, True]])
    )
    with pytest.raises(ValueError, match="target mask"):
        model.decode_delta(torch.randn(2, 3, 16))
    joint = torch.randn(2, 3, 16)
    with torch.no_grad():
        model.prediction[0].weight[:, -1].fill_(0.5)
    actual = model.decode_delta(joint, **kwargs)
    expected = model.prediction(
        torch.cat((joint, kwargs["target_mask"].float().unsqueeze(-1)), -1)
    ).squeeze(-1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("flag", (*FLAGS, "learned_genept_projection"))
def test_three_complete_nonzero_lr_updates_teacher_centers_and_resume(flag, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    original, batch = relay_training_fixture(False)
    seeds = original.student.graph.embedding.weight.detach().clone()
    if flag == "learned_genept_projection":
        seeds = torch.cat((seeds, seeds.flip(-1)), -1)
    model = GraDPertV2(
        seeds,
        replace(original.student.options, **{flag: True}),
    )
    objective = JointObjective(model, loss_reduction="row_mean", koleo_exclude_same_condition=True)
    optimizer = V2Optimizer(model, 1e-3, 0)
    frozen = (
        model.graph.embedding.weight.detach().clone()
        if flag in (FLAGS[0], "learned_genept_projection")
        else None
    )
    prior = model.readout_prior.clone() if flag == FLAGS[1] else None
    rng = np.random.default_rng(7)

    def step():
        old_teacher = copy.deepcopy(objective.teacher.state_dict())
        result = optimizer_step(
            objective, optimizer, batch, microbatch=2, lr=1e-3, momentum=0.99, bf16=False
        )
        assert all(p.grad is None for p in objective.teacher.parameters())
        assert not objective.pending
        for name, p in objective.teacher.named_parameters():
            torch.testing.assert_close(
                p, torch.lerp(old_teacher[name], dict(model.named_parameters())[name], 0.01)
            )
        assert torch.isfinite(objective.ssl1_cls_center).all()
        assert objective.ssl1_cls_center.abs().sum() > 0
        return result

    step()
    step()
    assert optimizer.steps == 2
    if frozen is not None:
        assert torch.equal(model.graph.embedding.weight, frozen)
        assert torch.equal(objective.teacher.graph.embedding.weight, frozen)
        assert "graph.embedding.weight" not in {r["name"] for r in routes(model)}
        if flag == "learned_genept_projection":
            assert model.graph.adapter.weight.grad is not None
        else:
            assert model.graph.prior_adapter[2].weight.abs().sum() > 0
    if prior is not None:
        assert torch.equal(model.readout_prior, prior)
        assert torch.equal(objective.teacher.readout_prior, prior)
    save_checkpoint(
        tmp_path / "last.pt",
        objective,
        optimizer,
        identity={"case": flag},
        progress={"step": 2},
        generator=rng,
    )
    expected_terms = step()
    expected = copy.deepcopy(
        (objective.state_dict(), optimizer.state_dict(), torch.get_rng_state())
    )
    load_checkpoint(
        tmp_path / "last.pt", objective, optimizer, identity={"case": flag}, generator=rng
    )
    assert_tree_close(step(), expected_terms)
    assert_tree_close(
        (objective.state_dict(), optimizer.state_dict(), torch.get_rng_state()), expected
    )
    # Test the real inference entry and gradient diagnostics with identity wiring.
    predicted = predict_query_set(
        model.eval(),
        small_index(16),
        np.zeros((3, 16), dtype=np.float32),
        (0,),
        np.array([2, 7, 9]),
        device=torch.device("cpu"),
        cell_batch=2,
    )
    assert predicted.shape == (3, 3) and np.isfinite(predicted).all()
    diagnostic = evaluate_loss_diagnostics(objective, batch, bf16=False)
    assert diagnostic["selection_use"] == "none"
    assert diagnostic["prediction_mse"] >= 0


def test_all_three_flags_can_combine_without_changing_gene_identity():
    model = model_for()
    combined = GraDPertV2(
        model.graph.embedding.weight.detach().clone(),
        replace(model.options, **dict.fromkeys(FLAGS, True)),
    ).eval()
    output = combined.encode_response(
        torch.randn(3, 8),
        torch.randn(2, 3),
        torch.randn(2, 8),
        **combined.prediction_metadata(torch.tensor([7, 2, 5]), torch.tensor([[2], [5]])),
    )
    assert output["prediction"].shape == (2, 3)


def test_disabled_flags_load_historical_synthetic_state_without_new_keys():
    a = model_for()
    b = GraDPertV2(
        a.graph.embedding.weight.detach().clone(), replace(a.options, **dict.fromkeys(FLAGS, False))
    )
    b.load_state_dict(a.state_dict(), strict=True)
    assert a.options.payload() == b.options.payload()
    assert a.state_dict().keys() == b.state_dict().keys()


def test_u4_random_linear_formula_frozen_prior_and_shared_gradients():
    torch.manual_seed(17)
    raw = torch.randn(9, 16)
    reduced = torch.randn(9, 8)
    arch = replace(small_architecture(), relay_passes=1)
    torch.manual_seed(291)
    base = GraDPertV2(reduced, arch)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(291)
    model = GraDPertV2(raw, replace(arch, learned_genept_projection=True))
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.graph.embedding.weight.requires_grad
    assert torch.equal(model.graph.embedding.weight, raw)
    assert model.graph.adapter.weight.shape == (8, 16)
    assert model.graph.adapter.weight.abs().sum() > 0
    assert torch.count_nonzero(model.graph.adapter.bias) == 0
    assert model.graph.prior_adapter is None
    for name, value in base.state_dict().items():
        if not name.startswith("graph.embedding."):
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    expected = raw @ model.graph.adapter.weight.T + model.graph.adapter.bias
    torch.testing.assert_close(model.graph.adapter(raw), expected)
    old_unseen = model.graph.adapter(raw[8]).detach().clone()
    optimizer = torch.optim.SGD(model.graph.adapter.parameters(), lr=0.01)
    model.graph.adapter(raw[:3]).square().mean().backward()
    assert model.graph.adapter.weight.grad.abs().sum() > 0
    assert model.graph.embedding.weight.grad is None
    optimizer.step()
    assert not torch.equal(old_unseen, model.graph.adapter(raw[8]))
    assert torch.equal(model.graph.embedding.weight, raw)
    assert "graph.embedding.weight" not in {r["name"] for r in routes(model)}
    assert "graph.adapter.weight" in {r["name"] for r in routes(model)}


def test_u4_rejects_reduced_input_and_incompatible_flags():
    arch = replace(small_architecture(), learned_genept_projection=True)
    with pytest.raises(ValueError, match="higher-width"):
        GraDPertV2(torch.randn(9, 8), arch)
    for flag in ("prior_shared_adapter", "gene_conditioned_readout"):
        with pytest.raises(ValueError, match="reduced-prior"):
            replace(arch, **{flag: True})
    assert "learned_genept_projection" not in V2Architecture().payload()
    with pytest.raises(ValueError, match="boolean"):
        replace(small_architecture(), learned_genept_projection=1)
