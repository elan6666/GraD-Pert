"""Population loss mathematics and fixed-budget sampler invariants."""

import copy

import numpy as np
import pytest
import torch

from gradpert.training.v2.population import population_epoch_batches, population_response_loss


def direct_mmd(prediction, truth, bandwidth):
    dxx = (prediction[:, None] - prediction[None, :]).square().sum(-1)
    dyy = (truth[:, None] - truth[None, :]).square().sum(-1)
    dxy = (prediction[:, None] - truth[None, :]).square().sum(-1)
    terms = []
    m, n = len(prediction), len(truth)
    for factor in (0.5, 1, 2, 4):
        denominator = 2 * factor * bandwidth + 1e-12
        xx, yy, xy = [(-distance / denominator).exp() for distance in (dxx, dyy, dxy)]
        terms.append(
            (xx.sum() - xx.diagonal().sum()) / (m * (m - 1))
            + (yy.sum() - yy.diagonal().sum()) / (n * (n - 1))
            - 2 * xy.mean()
        )
    return torch.stack(terms).mean()


def populations():
    rng = torch.Generator().manual_seed(83)
    return (
        torch.randn(4, 7, generator=rng, dtype=torch.float64),
        torch.randn(6, 7, generator=rng, dtype=torch.float64),
    )


def test_components_match_direct_formula_and_prediction_gradients():
    prediction, truth = populations()
    prediction.requires_grad_()
    result = population_response_loss(prediction, truth)
    truth_distances = (truth[:, None] - truth[None, :]).square().sum(-1)
    off_diagonal = ~torch.eye(len(truth), dtype=torch.bool)
    bandwidth = truth_distances[off_diagonal].median()
    expected_mmd = direct_mmd(prediction, truth, bandwidth)
    expected_mean = (prediction.mean(0) - truth.mean(0)).square().mean()
    assert result.mmd_valid and result.skip_reason is None
    assert result.prediction_rows == 4 and result.truth_rows == 6
    torch.testing.assert_close(result.bandwidth_squared, bandwidth)
    torch.testing.assert_close(result.mean_mse, expected_mean)
    torch.testing.assert_close(result.mmd_unbiased, expected_mmd)
    actual_gradient = torch.autograd.grad(result.mean_mse + result.mmd_unbiased, prediction)[0]
    reference_gradient = torch.autograd.grad(expected_mean + expected_mmd, prediction)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)
    assert torch.isfinite(actual_gradient).all() and actual_gradient.abs().sum() > 0


def test_negative_unbiased_estimate_is_retained():
    truth = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    prediction = truth.clone().requires_grad_()
    result = population_response_loss(prediction, truth)
    assert result.mmd_valid and result.mmd_unbiased < 0
    torch.testing.assert_close(result.mean_mse, torch.zeros((), dtype=torch.float64))
    result.mmd_unbiased.backward()
    assert torch.isfinite(prediction.grad).all()


def test_separate_row_permutations_preserve_components():
    prediction, truth = populations()
    first = population_response_loss(prediction, truth)
    second = population_response_loss(prediction[[2, 0, 3, 1]], truth[[5, 2, 0, 4, 1, 3]])
    torch.testing.assert_close(first.mean_mse, second.mean_mse)
    torch.testing.assert_close(first.mmd_unbiased, second.mmd_unbiased)
    torch.testing.assert_close(first.bandwidth_squared, second.bandwidth_squared)


def test_bandwidth_uses_only_detached_truth_and_matches_fixed_bandwidth_gradient():
    prediction, truth = populations()
    prediction.requires_grad_()
    truth.requires_grad_()
    result = population_response_loss(prediction, truth)
    shifted = population_response_loss(prediction + 1000, truth)
    assert not result.bandwidth_squared.requires_grad
    torch.testing.assert_close(result.bandwidth_squared, shifted.bandwidth_squared)
    actual = torch.autograd.grad(result.mmd_unbiased, truth, retain_graph=True)[0]
    reference = torch.autograd.grad(direct_mmd(prediction, truth, result.bandwidth_squared), truth)[
        0
    ]
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize(
    "prediction,truth,reason",
    [
        ([[0.0]], [[0.0], [2.0]], "prediction_rows_lt_two"),
        ([[0.0], [2.0]], [[1.0]], "truth_rows_lt_two"),
        ([[0.0], [2.0]], [[1.0], [1.0]], "truth_has_no_distinct_vectors"),
        ([[0.0], [2.0]], [[1.0], [1.0], [1.0], [2.0]], "truth_bandwidth_nonpositive"),
    ],
)
def test_degenerate_mmd_is_explicit_zero_with_mean_supervision(prediction, truth, reason):
    prediction = torch.tensor(prediction, requires_grad=True)
    truth = torch.tensor(truth)
    result = population_response_loss(prediction, truth)
    assert not result.mmd_valid and result.skip_reason == reason
    assert result.mmd_unbiased.item() == 0 and result.mmd_unbiased.requires_grad
    torch.testing.assert_close(
        result.mean_mse, (prediction.mean(0) - truth.mean(0)).square().mean()
    )
    (result.mean_mse + result.mmd_unbiased).backward()
    assert torch.isfinite(prediction.grad).all()


def test_duplicate_prediction_vectors_are_valid_samples():
    prediction = torch.ones(4, 3, requires_grad=True)
    truth = torch.arange(12.0).reshape(4, 3)
    result = population_response_loss(prediction, truth)
    assert result.mmd_valid
    (result.mean_mse + result.mmd_unbiased).backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


def test_common_large_offset_preserves_mmd_without_gram_cancellation():
    prediction, truth = populations()
    baseline = population_response_loss(prediction, truth)
    shifted = population_response_loss(prediction + 1e8, truth + 1e8)
    torch.testing.assert_close(baseline.mmd_unbiased, shifted.mmd_unbiased)


def test_autocast_does_not_lower_pairwise_distance_precision():
    prediction, truth = populations()
    prediction, truth = prediction.float(), truth.float()
    expected = population_response_loss(prediction, truth)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = population_response_loss(prediction, truth)
    assert actual.mean_mse.dtype == actual.mmd_unbiased.dtype == torch.float32
    torch.testing.assert_close(expected.mmd_unbiased, actual.mmd_unbiased, atol=0, rtol=0)


def test_nonzero_learning_rate_multi_update_preserves_finite_training():
    rng = torch.Generator().manual_seed(91)
    control = torch.randn(8, 4, generator=rng)
    truth = 1.3 * control + 0.5
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = copy.deepcopy(model.state_dict())
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        result = population_response_loss(model(control), truth)
        assert result.mmd_valid
        loss = result.mean_mse + result.mmd_unbiased
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert not torch.equal(initial["weight"], model.weight)
    assert all(state["step"].item() == 6 for state in optimizer.state.values())


@pytest.mark.parametrize(
    "prediction,truth,factors",
    [
        (torch.ones(3), torch.ones(3), (1,)),
        (torch.ones(2, 3), torch.ones(2, 4), (1,)),
        (torch.ones(0, 3), torch.ones(2, 3), (1,)),
        (torch.ones(2, 0), torch.ones(2, 0), (1,)),
        (torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3), (1,)),
        (torch.ones(2, 3), torch.ones(2, 3), ()),
        (torch.ones(2, 3), torch.ones(2, 3), (0,)),
        (torch.ones(2, 3), torch.ones(2, 3), (float("nan"),)),
    ],
)
def test_invalid_population_contract_rejected(prediction, truth, factors):
    with pytest.raises(ValueError):
        population_response_loss(prediction, truth, bandwidth_factors=factors)


def sampler_inputs():
    return {
        "truth_row_ids": [2, 7, 9, 14, 18, 22, 27],
        "condition_keys": ["a", "a", "b", "a", "b", "b", "a"],
        "context_keys": [
            "cell::b1",
            "cell::b2",
            "cell::b1",
            "cell::b1",
            "cell::b2",
            "cell::b2",
            "cell::b2",
        ],
        "control_pools": {"cell::b1": [100, 101], "cell::b2": [201]},
        "batch_size": 3,
    }


def test_sampler_preserves_actual_train_membership_contexts_and_exact_epoch_budget():
    inputs = sampler_inputs()
    batches = list(population_epoch_batches(**inputs, generator=np.random.default_rng(17)))
    observed = [row for batch in batches for row in batch.truth_rows]
    assert sorted(observed) == inputs["truth_row_ids"] and len(set(observed)) == len(observed)
    conditions = dict(zip(inputs["truth_row_ids"], inputs["condition_keys"], strict=True))
    contexts = dict(zip(inputs["truth_row_ids"], inputs["context_keys"], strict=True))
    for batch in batches:
        assert 1 <= len(batch.truth_rows) <= inputs["batch_size"]
        assert len(batch.truth_rows) == len(batch.control_rows) == len(batch.contexts)
        for row, control, context in zip(
            batch.truth_rows, batch.control_rows, batch.contexts, strict=True
        ):
            assert conditions[row] == batch.condition and contexts[row] == context
            assert control in inputs["control_pools"][context]
    assert len(batches) == 3 and any(len(batch.truth_rows) == 1 for batch in batches)


def test_sampler_replays_private_generator_state_without_global_rng_changes():
    inputs = sampler_inputs()
    generator = np.random.default_rng(171)
    state = copy.deepcopy(generator.bit_generator.state)
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first = list(population_epoch_batches(**inputs, generator=generator))
    generator.bit_generator.state = state
    second = list(population_epoch_batches(**inputs, generator=generator))
    assert first == second
    assert torch.equal(torch_state, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and np.array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]


@pytest.mark.parametrize(
    "change", ["missing_pool", "empty_pool", "duplicate_truth", "unaligned", "zero_batch"]
)
def test_sampler_rejects_invalid_membership_contract(change):
    inputs = sampler_inputs()
    if change == "missing_pool":
        del inputs["control_pools"]["cell::b2"]
    elif change == "empty_pool":
        inputs["control_pools"]["cell::b2"] = []
    elif change == "duplicate_truth":
        inputs["truth_row_ids"][1] = inputs["truth_row_ids"][0]
    elif change == "unaligned":
        inputs["context_keys"].pop()
    else:
        inputs["batch_size"] = 0
    with pytest.raises(ValueError):
        list(population_epoch_batches(**inputs, generator=np.random.default_rng(17)))
