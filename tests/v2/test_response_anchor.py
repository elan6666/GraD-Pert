"""Complete-group gradient and fixed-bank boundary tests, all synthetic CPU."""

import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from gradpert.training.v2.response_anchor import (
    FixedResponseBank,
    attach_anchor_readout,
    group_loss_coefficient,
    group_mean_loss,
    group_schedule,
    isolated_rng,
)


def _states():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_states_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def _bank(tmp_path, change=None, metadata_change=None):
    values = {
        "allowed_gene_indices": np.array([1, 3, 9], dtype=np.int64),
        "condition_targets": np.arange(512, dtype=np.float32).reshape(2, 256),
        "condition_anchor_gene_indices": np.array([8, 4], dtype=np.int64),
        "control_expression": np.arange(21, dtype=np.float32).reshape(7, 3),
        "control_slot_offsets": np.array([0, 4, 9], dtype=np.int64),
        "control_slot_indices": np.array([0, 1, 0, 3, 4, 5, 6, 0, 1], dtype=np.int64),
    }
    if change:
        change(values)
    np.savez(tmp_path / "bank.npz", **values)
    metadata = {
        "schema": "fixed-response-group-anchor-bank-v1",
        "condition_ids": ["p0", "p1"],
        "arrays": {
            "path": "bank.npz",
            "sha256": hashlib.sha256((tmp_path / "bank.npz").read_bytes()).hexdigest(),
        },
        "identity": {"split": "selected_train", "source_commit": "synthetic"},
    }
    if metadata_change:
        metadata_change(metadata)
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha


def test_bank_loads_odd_group_and_repeated_slots_without_model_buffers(tmp_path):
    path, sha = _bank(tmp_path)
    bank = FixedResponseBank.from_files(path, sha, [1, 3, 9])
    assert bank.condition_ids == ("p0", "p1")
    np.testing.assert_array_equal(
        bank.control_for_condition(1), bank.control_expression[[4, 5, 6, 0, 1]]
    )
    assert bank.control_for_condition(1).shape == (5, 3)
    assert not bank.condition_targets.flags.writeable
    assert not isinstance(bank, nn.Module)
    model = nn.Linear(3, 3)
    model.fixed_bank = bank
    assert "condition_targets" not in model.state_dict()
    assert not list(model.buffers())
    with pytest.raises(ValueError):
        bank.control_for_condition(2)


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(condition_targets=np.full((2, 256), np.nan, np.float32)),
        lambda a: a.update(control_expression=np.full((7, 3), np.inf, np.float32)),
        lambda a: a.update(condition_targets=np.zeros((2, 256), np.float32)),
        lambda a: a.update(condition_targets=np.ones((2, 256), np.float32)),
        lambda a: a.update(condition_targets=np.ones((2, 255), np.float32)),
        lambda a: a.update(control_slot_offsets=np.array([0, 3, 9], dtype=np.int64)),
        lambda a: a.update(control_slot_offsets=np.array([1, 4, 9], dtype=np.int64)),
        lambda a: a.update(control_slot_indices=np.array([7] * 9, dtype=np.int64)),
        lambda a: a.update(allowed_gene_indices=np.array([1, 1, 9], dtype=np.int64)),
        lambda a: a.update(condition_anchor_gene_indices=np.array([-1, 4], dtype=np.int64)),
        lambda a: a.update(control_slot_indices=np.zeros(9, dtype=np.int32)),
        lambda a: a.update(unapproved_array=np.ones((2, 2))),
    ],
)
def test_bank_rejects_invalid_scientific_arrays(tmp_path, change):
    path, sha = _bank(tmp_path, change)
    with pytest.raises(ValueError):
        FixedResponseBank.from_files(path, sha, [1, 3, 9])


def test_bank_rejects_41_rows_axis_mismatch_and_hash_tampering(tmp_path):
    def oversized(a):
        a["control_slot_offsets"] = np.array([0, 41, 45], dtype=np.int64)
        a["control_slot_indices"] = np.zeros(45, dtype=np.int64)

    path, sha = _bank(tmp_path, oversized)
    with pytest.raises(ValueError, match=r"4\.\.40"):
        FixedResponseBank.from_files(path, sha, [1, 3, 9])
    path, sha = _bank(tmp_path)
    with pytest.raises(ValueError, match="axis"):
        FixedResponseBank.from_files(path, sha, [1, 9, 3])
    with pytest.raises(ValueError, match="SHA256"):
        FixedResponseBank.from_files(path, "0" * 64, [1, 3, 9])
    with (tmp_path / "bank.npz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        FixedResponseBank.from_files(path, sha, [1, 3, 9])


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(condition_ids=["p0", "p0"]),
        lambda m: m.update(condition_ids=["p0"]),
        lambda m: m.update(identity={}),
        lambda m: m.update(schema="other"),
    ],
)
def test_bank_metadata_mismatch_rejected(tmp_path, change):
    path, sha = _bank(tmp_path, metadata_change=change)
    with pytest.raises(ValueError):
        FixedResponseBank.from_files(path, sha, [1, 3, 9])


@pytest.mark.parametrize("n", [4, 5, 40])
def test_complete_group_gradient_matches_explicit_chain_rule(n):
    with isolated_rng(54):
        x = torch.randn(n, 3, dtype=torch.float64)
        encoder = nn.Linear(3, 2, dtype=torch.float64)
        head = nn.Linear(2, 2, dtype=torch.float64)
    z = encoder(x)
    z.retain_grad()
    target = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    loss = group_mean_loss(z, target, head)
    error = head(z.mean(0)).detach() - target.detach()
    expected_dz = (2 / 2) * (head.weight.detach().T @ error) / n
    loss.backward()
    torch.testing.assert_close(z.grad, expected_dz.expand_as(z))
    torch.testing.assert_close(head.weight.grad, 2 / 2 * torch.outer(error, z.detach().mean(0)))
    torch.testing.assert_close(head.bias.grad, 2 / 2 * error)
    torch.testing.assert_close(encoder.weight.grad, expected_dz[:, None] @ x.sum(0)[None, :])
    assert target.grad is None


def test_group_mean_loss_is_not_cell_mse_or_chunk_average():
    z = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]], requires_grad=True)
    head = nn.Linear(1, 1)
    with torch.no_grad():
        head.weight.fill_(1)
        head.bias.zero_()
    assert group_mean_loss(z, torch.zeros(1), head).item() == 0
    assert (head(z).square().mean()).item() == 1
    assert (head(z[:2].mean(0)).square() + head(z[2:].mean(0)).square()).item() / 2 == 1
    with pytest.raises(ValueError):
        group_mean_loss(z.repeat(11, 1), torch.zeros(1), head)


def test_streamed_variable_groups_keep_main_grad_and_match_joint_reference():
    with isolated_rng(8):
        model = nn.Linear(3, 2, dtype=torch.float64)
        head = nn.Linear(2, 2, dtype=torch.float64)
        xs = [torch.randn(n, 3, dtype=torch.float64) for n in [4, 5, 40]]
        targets = [torch.randn(2, dtype=torch.float64) for _ in xs]
    reference, reference_head = copy.deepcopy(model), copy.deepcopy(head)
    main = model.weight.square().sum()
    main.backward()
    for x, target in zip(xs, targets, strict=True):
        group_mean_loss(model(x), target, head).backward()
    ref_loss = reference.weight.square().sum() + sum(
        (reference_head(reference(x).mean(0)) - target).square().mean()
        for x, target in zip(xs, targets, strict=True)
    )
    ref_loss.backward()
    for a, b in zip(
        (*model.parameters(), *head.parameters()),
        (*reference.parameters(), *reference_head.parameters()),
        strict=True,
    ):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("p,steps,world", [(1322, 150, 1), (11, 3, 4), (2, 5, 3)])
def test_schedule_exact_once_and_rank_mean_gradient_weight(p, steps, world):
    before = _states()
    schedules = [group_schedule(p, steps, 2, 17, world, rank) for rank in range(world)]
    _assert_states_equal(before, _states())
    flat = [condition for schedule in schedules for step in schedule for condition in step]
    assert sorted(flat) == list(range(p))
    assert schedules[0] == group_schedule(p, steps, 2, 17, world, 0)
    global_sizes = [sum(len(s[j]) for s in schedules) for j in range(steps)]
    assert max(global_sizes) - min(global_sizes) <= 1
    if p == 11:
        assert len({sum(map(len, schedule)) for schedule in schedules}) > 1
    losses = torch.arange(1.0, p + 1, dtype=torch.float64).square()
    coefficient = group_loss_coefficient(p, steps, world, weight=0.3)
    reduced_step_average = (
        sum(coefficient * losses[list(step)].sum() for schedule in schedules for step in schedule)
        / world
        / steps
    )
    torch.testing.assert_close(reduced_step_average, 0.3 * losses.mean())


def test_ddp_rank_mean_gradients_equal_condition_mean_reference():
    p, steps, world = 11, 3, 4
    with isolated_rng(71):
        model = nn.Linear(3, 2, dtype=torch.float64)
        head = nn.Linear(2, 2, dtype=torch.float64)
        xs = [torch.randn(4 + i % 3, 3, dtype=torch.float64) for i in range(p)]
        targets = [torch.randn(2, dtype=torch.float64) for _ in range(p)]
    reference, ref_head = copy.deepcopy(model), copy.deepcopy(head)
    parameters = (*model.parameters(), *head.parameters())
    accumulated = [torch.zeros_like(value) for value in parameters]
    for rank in range(world):
        model.zero_grad()
        head.zero_grad()
        for step in group_schedule(p, steps, 0, 9, world, rank):
            for i in step:
                (
                    group_loss_coefficient(p, steps, world)
                    * group_mean_loss(model(xs[i]), targets[i], head)
                ).backward()
        for total, value in zip(accumulated, parameters, strict=True):
            if value.grad is not None:
                total.add_(value.grad / world / steps)
    sum(
        group_mean_loss(reference(x), t, ref_head) / p for x, t in zip(xs, targets, strict=True)
    ).backward()
    for value, expected in zip(
        accumulated, (*reference.parameters(), *ref_head.parameters()), strict=True
    ):
        torch.testing.assert_close(value, expected.grad)


def test_default_off_and_attach_rng_key_isolation():
    model = nn.Linear(3, 3, dtype=torch.float64)
    original_keys = tuple(model.state_dict())
    original_weights = {key: value.clone() for key, value in model.state_dict().items()}
    after_model = _states()
    # Historical/default-off path never invokes attach, and has no new key.
    assert tuple(model.state_dict()) == original_keys
    _assert_states_equal(after_model, _states())
    head = attach_anchor_readout(model, 3, 73)
    _assert_states_equal(after_model, _states())
    for key, value in original_weights.items():
        assert torch.equal(model.state_dict()[key], value)
    assert set(model.state_dict()) - set(original_keys) == {
        "response_anchor_readout.weight",
        "response_anchor_readout.bias",
    }
    with isolated_rng(73):
        expected = nn.Linear(3, 3, dtype=torch.float64)
    torch.testing.assert_close(head.weight, expected.weight)
    torch.testing.assert_close(head.bias, expected.bias)
    assert all(parameter.requires_grad for parameter in head.parameters())
    assert not list(model.buffers())
    with pytest.raises(ValueError, match="already exists"):
        attach_anchor_readout(model, 3, 73)


@pytest.mark.parametrize("failure", [False, True])
def test_rng_restored_for_python_numpy_torch_and_nested_error(failure):
    before = _states()
    outputs = []
    for _ in range(2):
        try:
            with isolated_rng(32):
                outputs.append((random.random(), np.random.rand(), torch.rand(3)))
                outer = _states()
                with isolated_rng(19):
                    random.random()
                    np.random.rand()
                    torch.rand(5)
                _assert_states_equal(outer, _states())
                if failure:
                    raise RuntimeError("synthetic")
        except RuntimeError:
            assert failure
        _assert_states_equal(before, _states())
    assert outputs[0][:2] == outputs[1][:2]
    assert torch.equal(outputs[0][2], outputs[1][2])


def test_schedule_rank_shards_and_resume_cursor_are_the_same_global_plan():
    global_plan = group_schedule(31, 4, 0, 23)
    ranks = [group_schedule(31, 4, 0, 23, 3, rank) for rank in range(3)]
    for j, group in enumerate(global_plan):
        for rank in range(3):
            assert ranks[rank][j] == group[rank::3]
    assert group_schedule(31, 4, 1, 23) != global_plan
    # No mutable cursor: saved main-update offset reconstructs the untouched tail.
    resumed = group_schedule(31, 4, 0, 23)[2:]
    assert resumed == global_plan[2:]
    assert set(sum(global_plan[:2], ())).isdisjoint(sum(resumed, ()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"P": 0},
        {"steps_per_epoch": 0},
        {"epoch": -1},
        {"seed": True},
        {"world": 0},
        {"rank": 2, "world": 2},
    ],
)
def test_invalid_schedule_fails_without_rng_change(kwargs):
    args = dict(P=11, steps_per_epoch=3, epoch=0, seed=19)
    args.update(kwargs)
    before = _states()
    with pytest.raises(ValueError):
        group_schedule(**args)
    _assert_states_equal(before, _states())


def test_nonfinite_loss_and_negative_weight_fail_closed():
    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.fill_(float("inf"))
    with pytest.raises(ValueError, match="nonfinite"):
        group_mean_loss(torch.ones(4, 2), torch.ones(2), head)
    for value in [-0.1, float("nan"), float("inf"), True]:
        with pytest.raises(ValueError):
            group_loss_coefficient(11, 3, weight=value)
    assert group_loss_coefficient(11, 3, weight=0.0) == 0


def test_partial_invalid_rng_request_does_not_change_any_state():
    before = _states()
    for seed, devices in [(-1, ()), (2**63, ()), (2, (0, 0))]:
        with pytest.raises(ValueError), isolated_rng(seed, devices):
            pytest.fail("invalid scope entered")
        _assert_states_equal(before, _states())
