"""Analytic and gradient contracts for the native full-population AU objective."""

import math

import pytest
import torch
import torch.nn.functional as F

from gradpert.training.v2.alignment_uniformity import alignment_uniformity, pair_census


def compute(z1, z2=None, conditions=None, contexts=None, **kwargs):
    n = z1.shape[0]
    return alignment_uniformity(
        z1,
        z1 if z2 is None else z2,
        torch.arange(n) if conditions is None else conditions,
        torch.zeros(n, dtype=torch.long) if contexts is None else contexts,
        **kwargs,
    )


def reference_uniformity(values, conditions, contexts, t=2.0):
    vectors = F.normalize(values.double(), dim=1)
    scores = [
        2 * t * (vectors[i].dot(vectors[j]) - 1)
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if contexts[i] == contexts[j] and conditions[i] != conditions[j]
    ]
    return torch.logsumexp(torch.stack(scores), 0) - math.log(len(scores))


def test_equal_average_cosine_does_not_mean_equal_uniformity():
    two_clusters = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    four_directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.allclose((two_clusters @ two_clusters.T)[upper].mean(), torch.tensor(-1 / 3))
    assert torch.allclose((four_directions @ four_directions.T)[upper].mean(), torch.tensor(-1 / 3))
    cluster = compute(two_clusters)
    spread = compute(four_directions)
    assert cluster.uniformity.item() == pytest.approx(math.log((2 + 4 * math.exp(-8)) / 6))
    assert spread.uniformity.item() == pytest.approx(
        math.log((4 * math.exp(-4) + 2 * math.exp(-8)) / 6)
    )
    assert spread.uniformity < cluster.uniformity
    assert cluster.alignment.item() == 0
    assert spread.alignment.item() == 0
    assert cluster.pair_count == 6
    assert cluster.input_count == 4


def test_mask_and_context_pair_weights_match_explicit_enumeration():
    z1 = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, -1.0], [-1.0, 0.0]])
    z2 = torch.tensor([[1.0, 1.0], [1.0, 0.0], [-1.0, 1.0], [1.0, 1.0], [0.0, -1.0]])
    conditions = torch.tensor([2, 2, 3, 2, 4])
    contexts = torch.tensor([10, 10, 10, 20, 20])
    result = compute(z1, z2, conditions, contexts)
    expected = (
        reference_uniformity(z1, conditions, contexts)
        + reference_uniformity(z2, conditions, contexts)
    ) / 2
    assert result.uniformity.item() == pytest.approx(expected.item(), abs=2e-6)
    assert result.alignment.item() == pytest.approx(
        (1 - F.cosine_similarity(z1, z2)).mean().item(), abs=1e-6
    )
    assert (result.pair_count, result.input_count, result.context_count) == (3, 5, 2)
    census = pair_census(conditions, contexts)
    assert census["pair_count"] == 3
    assert census["contexts"] == [
        {
            "context_id": 10,
            "input_count": 3,
            "condition_count": 2,
            "pair_count": 2,
            "pair_weight": 2 / 3,
        },
        {
            "context_id": 20,
            "input_count": 2,
            "condition_count": 2,
            "pair_count": 1,
            "pair_weight": 1 / 3,
        },
    ]


def test_view_uniformities_are_averaged_after_each_logmeanexp():
    z1 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    z2 = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    result = compute(z1, z2)
    assert result.uniformity.item() == pytest.approx(-4.0)
    assert result.alignment.item() == pytest.approx(1.0)
    pooled_logmean = math.log((1 + math.exp(-8)) / 2)
    assert abs(result.uniformity.item() - pooled_logmean) > 3
    assert result.input_count == 2  # never stack views into four independent inputs


def test_microbatch_logmeanexp_average_is_not_full_population_objective():
    values = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    full = compute(values).uniformity
    micro_mean = (compute(values[:2]).uniformity + compute(values[2:]).uniformity) / 2
    assert micro_mean.item() == pytest.approx(-4.0)
    assert abs(full.item() - micro_mean.item()) > 1
    assert full.item() == pytest.approx(math.log((1 + 4 * math.exp(-4) + math.exp(-8)) / 6))


def test_contexts_are_pair_weighted_not_equal_weighted():
    values = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    result = compute(values, contexts=torch.tensor([0, 0, 0, 1, 1]))
    assert result.pair_count == 4
    assert result.uniformity.item() == pytest.approx(math.log((3 + math.exp(-8)) / 4))
    assert result.uniformity.item() != pytest.approx(-4.0)


def test_both_global_branches_receive_finite_nonzero_gradients_without_mutation():
    generator = torch.Generator().manual_seed(71)
    z1 = torch.randn(7, 5, generator=generator, requires_grad=True)
    z2 = torch.randn(7, 5, generator=generator, requires_grad=True)
    before1, before2 = z1.detach().clone(), z2.detach().clone()
    result = compute(z1, z2)
    for objective in (result.alignment, result.uniformity):
        gradients = torch.autograd.grad(objective, (z1, z2), retain_graph=True)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
    assert torch.equal(before1, z1)
    assert torch.equal(before2, z2)


def test_gradient_matches_independent_float64_pairwise_reference():
    generator = torch.Generator().manual_seed(19)
    z1 = torch.randn(6, 4, generator=generator, requires_grad=True)
    z2 = torch.randn(6, 4, generator=generator, requires_grad=True)
    conditions = torch.tensor([0, 0, 1, 2, 2, 3])
    contexts = torch.tensor([0, 0, 0, 1, 1, 1])
    result = compute(z1, z2, conditions, contexts)
    actual = torch.autograd.grad(result.alignment + 0.3 * result.uniformity, (z1, z2))
    positive = (
        1 - (F.normalize(z1.double(), dim=1) * F.normalize(z2.double(), dim=1)).sum(1)
    ).mean()
    uniformity = (
        reference_uniformity(z1, conditions, contexts)
        + reference_uniformity(z2, conditions, contexts)
    ) / 2
    expected = torch.autograd.grad(positive + 0.3 * uniformity, (z1, z2))
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-7)


def test_row_permutation_and_positive_row_scaling_preserve_objective():
    generator = torch.Generator().manual_seed(9)
    z1, z2 = torch.randn(6, 4, generator=generator), torch.randn(6, 4, generator=generator)
    conditions = torch.tensor([0, 0, 1, 2, 2, 3])
    contexts = torch.tensor([0, 0, 0, 1, 1, 1])
    expected = compute(z1, z2, conditions, contexts)
    order = torch.tensor([5, 3, 1, 2, 0, 4])
    scales = torch.arange(1.0, 7.0).unsqueeze(1)
    actual = compute((z1 * scales)[order], (z2 / scales)[order], conditions[order], contexts[order])
    torch.testing.assert_close(actual.alignment, expected.alignment)
    torch.testing.assert_close(actual.uniformity, expected.uniformity)
    assert actual.pair_count == expected.pair_count


def test_nearly_identical_directions_have_finite_gradients():
    z1 = torch.ones(8, 4, requires_grad=True)
    z2 = (torch.ones(8, 4) + torch.arange(32).reshape(8, 4) * 1e-6).requires_grad_()
    result = compute(z1, z2)
    gradients = torch.autograd.grad(result.alignment + result.uniformity, (z1, z2))
    assert torch.isfinite(result.alignment) and torch.isfinite(result.uniformity)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_exact_collapsed_state_is_stationary_not_a_claimed_escape_guarantee():
    z1 = torch.tensor([[1.0, 0.0]] * 4, requires_grad=True)
    z2 = z1.detach().clone().requires_grad_()
    result = compute(z1, z2)
    assert result.uniformity.item() == pytest.approx(0.0)
    assert result.alignment.item() == pytest.approx(0.0)
    gradients = torch.autograd.grad(result.alignment + result.uniformity, (z1, z2))
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


def test_fp32_auxiliary_computation_under_autocast_retains_input_gradient():
    z1 = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.bfloat16, requires_grad=True
    )
    z2 = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [-1.0, 0.0]], dtype=torch.bfloat16, requires_grad=True
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = compute(z1, z2)
    reference = compute(z1.float(), z2.float())
    assert result.uniformity.dtype == torch.float32
    assert result.alignment.dtype == torch.float32
    assert torch.equal(result.uniformity, reference.uniformity)
    assert torch.equal(result.alignment, reference.alignment)
    result.alignment.backward()
    assert z1.grad is not None and z2.grad is not None
    assert torch.isfinite(z1.grad).all() and torch.isfinite(z2.grad).all()


@pytest.mark.parametrize("field", ["z1", "z2"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_inputs_rejected(field, bad):
    first, second = torch.eye(2), torch.eye(2)
    (first if field == "z1" else second)[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        compute(first, second)


@pytest.mark.parametrize("field", ["z1", "z2"])
def test_zero_norm_rows_rejected_instead_of_fabricating_unit_vectors(field):
    first, second = torch.eye(2), torch.eye(2)
    (first if field == "z1" else second)[0] = 0
    with pytest.raises(ValueError, match="zero or non-finite FP32 row norm"):
        compute(first, second)


@pytest.mark.parametrize("conditions,contexts", [([0, 0], [0, 0]), ([0, 1], [0, 1])])
def test_no_eligible_pairs_fail_explicitly_but_census_remains_available(conditions, contexts):
    condition_ids, context_ids = torch.tensor(conditions), torch.tensor(contexts)
    census = pair_census(condition_ids, context_ids)
    assert census["pair_count"] == 0
    assert all(row["pair_weight"] == 0 for row in census["contexts"])
    with pytest.raises(ValueError, match="no eligible"):
        compute(torch.eye(2), conditions=condition_ids, contexts=context_ids)


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([[0, 1]]),
        torch.tensor([0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([False, True]),
    ],
)
@pytest.mark.parametrize("field", ["conditions", "contexts"])
def test_invalid_id_shapes_and_dtypes_rejected(ids, field):
    with pytest.raises(ValueError, match=r"integer|N representation"):
        compute(torch.eye(2), **{field: ids})


@pytest.mark.parametrize(
    "first,second",
    [
        (torch.ones(1, 2), torch.ones(1, 2)),
        (torch.ones(0, 2), torch.ones(0, 2)),
        (torch.ones(2, 0), torch.ones(2, 0)),
        (torch.ones(2, 2), torch.ones(3, 2)),
        (torch.ones(2, 2, 2), torch.ones(2, 2, 2)),
        (torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.long)),
    ],
)
def test_invalid_population_and_representation_shape_rejected(first, second):
    with pytest.raises(ValueError, match=r"N >= 2|shape|floating-point"):
        compute(first, second)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), 1e40, True, "2"])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        compute(torch.eye(2), temperature=temperature)


def test_census_empty_population_has_explicit_zero_counts():
    ids = torch.empty(0, dtype=torch.long)
    assert pair_census(ids, ids) == {
        "input_count": 0,
        "pair_count": 0,
        "context_count": 0,
        "contexts": [],
    }
