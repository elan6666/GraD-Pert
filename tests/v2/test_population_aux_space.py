"""Move only complete-population A/CR to the existing pre-L2 projector output."""

import copy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_au_integration import GradientReferenceOptimizer, _round2_options, step
from test_cr_integration import fixture as cr_fixture
from test_koleo_accumulation import assert_nested_close

from gradpert.config.v2 import V2Options
from gradpert.training.v2.alignment_uniformity import alignment_uniformity, negative_coding_rate
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.diagnostic_objective import diagnostic_objective
from gradpert.training.v2.engine import optimizer_step, slice_cells
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer


def fixture(*, checkpointed=False, warmup=0):
    objective, batch = cr_fixture(checkpointed=checkpointed, warmup=warmup)
    objective.population_aux_space = "ssl2_cls_bottleneck"
    return objective, batch


def test_optional_space_is_not_architecture_or_checkpoint_state_and_allows_observers():
    arch, options = _round2_options()
    assert options.population_aux_space == "response_cls"
    assert "population_aux_space" not in arch.payload()
    new = replace(options, population_aux_space="ssl2_cls_bottleneck")
    assert new.au_alignment_weight == new.cr_weight == 0
    # Diagnostic assembly may disable the losses while retaining the source space.
    assert replace(new, au_alignment_weight=0, au_uniformity_weight=0, cr_weight=0) == new
    from pathlib import Path

    from gradpert.config import load_experiment_config

    path = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/representation_round2_20261022/R2-RI/gradpert_v2/nadig_jurkat.yaml"
    )
    parameters = dict(load_experiment_config(path).model.parameters)
    parameters["population_aux_space"] = SimpleNamespace(value="ssl2_cls_bottleneck")
    new_arch, parsed = V2Options.parse_parameters(parameters)
    assert new_arch == arch and parsed == new
    old, _ = cr_fixture()
    changed, _ = fixture()
    assert old.state_dict().keys() == changed.state_dict().keys()
    changed.load_state_dict(old.state_dict(), strict=True)
    assert JointObjective(copy.deepcopy(old.student)).population_aux_space == "response_cls"
    assert (
        JointObjective(
            copy.deepcopy(old.student), population_aux_space="ssl2_cls_bottleneck"
        ).population_aux_space
        == "ssl2_cls_bottleneck"
    )
    for invalid in ("prototype", "projector", "", None, True):
        with pytest.raises(ValueError, match="auxiliary space"):
            replace(options, population_aux_space=invalid)
        with pytest.raises(ValueError, match="auxiliary space"):
            JointObjective(copy.deepcopy(old.student), population_aux_space=invalid)


def test_projector_shared_output_matches_legacy_forward_gradient_and_rng_exactly():
    obj, _ = fixture()
    head = copy.deepcopy(obj.student.ssl2_cls)
    reference = copy.deepcopy(head)
    x = torch.randn(5, obj.student.options.width, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    before_keys = list(head.state_dict())
    calls, hook_outputs = [], []
    handle = head.features.register_forward_hook(lambda m, args, output: calls.append(output))
    head_handle = head.register_forward_hook(lambda m, args, output: hook_outputs.append(output))
    try:
        before_rng = torch.get_rng_state().clone()
        features, logits = head(x, return_features=True)
        assert torch.equal(before_rng, torch.get_rng_state())
        assert len(calls) == 1 and features is calls[0]
        assert isinstance(hook_outputs[0], tuple)
        expected = reference(other)
        assert torch.equal(logits, expected)
        logits.square().sum().backward()
        expected.square().sum().backward()
        assert torch.equal(x.grad, other.grad)
        for (name, actual), (_, expected_parameter) in zip(
            head.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert torch.equal(actual.grad, expected_parameter.grad), name
        assert list(head.state_dict()) == before_keys
        assert isinstance(head(x.detach()), torch.Tensor)
        assert isinstance(hook_outputs[-1], torch.Tensor)
    finally:
        handle.remove()
        head_handle.remove()


@pytest.mark.parametrize("dino", [False, True])
def test_actual_deferred_space_reuses_global_features_deduplicates_and_routes_only_aux_gradient(
    dino,
):
    obj, batch = fixture()
    if not dino:
        obj.weights = (obj.weights[0], (0, 1, 0))
    batch = replace(batch, unique_cell_indices=torch.tensor([0, 2, 4]))
    features = []
    handle = obj.student.ssl2_cls.features.register_forward_hook(
        lambda m, args, output: features.append(output)
    )
    collected = []
    try:
        obj(batch, deferred_au=collected)
    finally:
        handle.remove()
    assert len(features) == (len(batch.cell_views) if dino else 2)
    first, second, ids, contexts = collected[0]
    assert torch.equal(first, features[0][batch.unique_cell_indices])
    assert torch.equal(second, features[1][batch.unique_cell_indices])
    assert first.shape[1] == obj.student.options.projector_bottleneck
    assert ids.tolist() == batch.condition_index[batch.unique_cell_indices].tolist()
    assert contexts.tolist() == batch.context_index[batch.unique_cell_indices].tolist()
    aux = 0.05 * alignment_uniformity(
        first, second, ids, contexts, compute_uniformity=False
    ).alignment + 0.00004 * negative_coding_rate(first, second)
    aux.backward()
    for prefix in ("ssl2_cls.features", "response", "cell", "graph"):
        gradients = [
            p.grad
            for name, p in obj.student.named_parameters()
            if name.startswith(prefix) and p.grad is not None
        ]
        assert gradients and sum(float(g.abs().sum()) for g in gradients) > 0, prefix
    assert all(p.grad is None for p in obj.student.ssl2_cls.prototypes.parameters())
    assert all(p.grad is None for p in obj.student.prediction.parameters())
    assert all(p.grad is None and not p.requires_grad for p in obj.teacher.parameters())


def test_active_space_leaves_base_forward_and_base_gradients_identical():
    old, batch = cr_fixture()
    new = copy.deepcopy(old)
    new.population_aux_space = "ssl2_cls_bottleneck"
    old_values, new_values = [], []
    rng = torch.get_rng_state().clone()
    expected, expected_terms = old(batch, deferred_au=old_values)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual, actual_terms = new(batch, deferred_au=new_values)
    assert torch.equal(actual, expected) and actual_terms.keys() == expected_terms.keys()
    assert all(torch.equal(actual_terms[k], expected_terms[k]) for k in actual_terms)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    expected.backward()
    actual.backward()
    for (name, left), (_, right) in zip(
        old.named_parameters(), new.named_parameters(), strict=True
    ):
        assert (left.grad is None and right.grad is None) or torch.equal(left.grad, right.grad), (
            name
        )
    assert all(
        torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items()
    )


def test_disabled_aux_space_preserves_update_rng_and_partial_diagnostic_scope_exactly():
    original, batch = cr_fixture()
    original.au_enabled = original.cr_enabled = False
    original.au_alignment_weight = original.cr_weight = 0
    new = copy.deepcopy(original)
    new.population_aux_space = "ssl2_cls_bottleneck"
    first, second = V2Optimizer(original.student, 1e-3, 0), V2Optimizer(new.student, 1e-3, 0)
    rng = torch.get_rng_state().clone()
    expected = step(original, first, batch, 2)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = step(new, second, batch, 2)
    assert expected == actual and torch.equal(expected_rng, torch.get_rng_state())
    assert all(
        torch.equal(value, new.state_dict()[name]) for name, value in original.state_dict().items()
    )
    active, batch = fixture()
    state = copy.deepcopy(active.state_dict())
    loss, terms, scope = diagnostic_objective(active, slice_cells(batch, 0, 1))
    assert torch.isfinite(loss) and "cr_negative" not in terms
    assert scope["diagnostic_objective_scope"]["unavailable_terms"] == {
        "au_alignment": "requires_complete_global_population",
        "cr_negative": "requires_complete_global_population",
    }
    assert all(torch.equal(value, active.state_dict()[name]) for name, value in state.items())


@pytest.mark.parametrize("checkpointed", [False, True])
def test_projected_full_microbatch_gradient_update_and_center_parity(checkpointed):
    full, batch = fixture(checkpointed=checkpointed)
    chunks = copy.deepcopy(full)
    first, second = (
        GradientReferenceOptimizer(full.student, 1e-3, 0),
        GradientReferenceOptimizer(chunks.student, 1e-3, 0),
    )
    rng = torch.get_rng_state().clone()
    expected = step(full, first, batch, 5)
    torch.set_rng_state(rng)
    actual = step(chunks, second, batch, 2)
    assert_nested_close(actual, expected)
    assert_nested_close(chunks.state_dict(), full.state_dict())
    for (_, left), (_, right) in zip(
        full.student.named_parameters(), chunks.student.named_parameters(), strict=True
    ):
        assert_nested_close(left.grad, right.grad)


def test_projected_native_optimizer_exact_checkpoint_resume(tmp_path):
    obj, batch = fixture(checkpointed=True, warmup=30)
    opt = V2Optimizer(obj.student, 1e-3, 0)
    for _ in range(2):
        step(obj, opt, batch, 2)
    path = tmp_path / "projected.pt"
    generator = np.random.default_rng(17)
    identity = {"population_aux_space": "ssl2_cls_bottleneck"}
    save_checkpoint(path, obj, opt, identity=identity, progress={"step": 2}, generator=generator)
    expected = step(obj, opt, batch, 2)
    state = copy.deepcopy(obj.state_dict())
    optimizer_state = copy.deepcopy(opt.state_dict())
    restored, _ = fixture(checkpointed=True, warmup=30)
    restored_opt = V2Optimizer(restored.student, 1e-3, 0)
    load_checkpoint(path, restored, restored_opt, identity=identity, generator=generator)
    actual = step(restored, restored_opt, batch, 2)
    assert expected == actual
    assert actual["cr_warmup"] == pytest.approx(2 / 30)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in state.items())
    assert_nested_close(restored_opt.state_dict(), optimizer_state, atol=0)


def _projected_distributed_worker(rank, rendezvous, path, microbatch):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        obj, batch = fixture(checkpointed=True)
        opt = GradientReferenceOptimizer(obj.student, 1e-3, 0)
        local = slice_cells(batch, 0, 2) if rank == 0 else slice_cells(batch, 2, 5)
        actual = optimizer_step(
            obj,
            opt,
            local,
            microbatch=microbatch,
            lr=1e-3,
            momentum=0.99,
            bf16=False,
            global_condition_index=batch.condition_index,
        )
        expected = torch.load(path, weights_only=True)
        assert_nested_close(actual, expected["metrics"])
        assert_nested_close(obj.state_dict(), expected["state"])
        for name, parameter in obj.student.named_parameters():
            assert_nested_close(parameter.grad, expected["gradients"][name])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("microbatch", [1, 5])
def test_projected_two_rank_uneven_population_parity(tmp_path, microbatch):
    obj, batch = fixture(checkpointed=True)
    opt = GradientReferenceOptimizer(obj.student, 1e-3, 0)
    metrics = step(obj, opt, batch, 5)
    path = str(tmp_path / "reference.pt")
    torch.save(
        {
            "metrics": metrics,
            "state": obj.state_dict(),
            "gradients": {name: p.grad for name, p in obj.student.named_parameters()},
        },
        path,
    )
    mp.spawn(
        _projected_distributed_worker,
        args=(str(tmp_path / "gloo"), path, microbatch),
        nprocs=2,
        join=True,
    )
