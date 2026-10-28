"""Explicit replication oracle for condition bootstrap; no scientific arrays."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src/gradpert/evaluation/response_geometry.py"
spec = importlib.util.spec_from_file_location("staged_response_geometry", SOURCE)
rg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rg
spec.loader.exec_module(rg)


def average_ranks(x):
    return np.array([1 + np.sum(x < v) + (np.sum(x == v) - 1) / 2 for v in x], dtype=float)


def explicit(x, y, weights):
    xx, yy = np.repeat(x, weights), np.repeat(y, weights)
    if len(xx) < 2 or np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return None
    return float(np.corrcoef(average_ranks(xx), average_ranks(yy))[0, 1])


def arrays(p=6):
    rng = np.random.default_rng(54)
    truth = rng.normal(size=(p, 8))
    return dict(
        condition_ids=np.array([f"p{i}" for i in range(p)]),
        allowed_gene_indices=np.array([0, 2, 5, 7, 10, 14, 18, 24]),
        own_target_indices=np.array([[0, -1] if i % 2 else [5, 7] for i in range(p)]),
        prediction_delta=truth + rng.normal(size=truth.shape) * 0.3,
        truth_delta=truth,
        cls_means=rng.normal(size=(p, 4)),
    )


def test_weighted_reranking_matches_expansion_ties_and_zero_counts():
    x = np.array([1, 1, 3, 4, 7, 7.0])
    y = np.array([3, 1, 3, 9, 0, 0.0])
    for weights in ([0, 3, 1, 2, 0, 4], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0]):
        expected = explicit(x, y, np.array(weights))
        actual = rg.weighted_spearman(x, y, weights)
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, abs=1e-14)
    weights = np.array([0, 3, 1, 2, 0, 4])
    # Fixed ranks only reweighted are a different statistic under multiplicities.
    rx, ry = average_ranks(x), average_ranks(y)
    active = weights > 0
    w = weights[active]
    a = rx[active]
    b = ry[active]
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    wrong = np.dot(w, a * b) / np.sqrt(np.dot(w, a * a) * np.dot(w, b * b))
    assert abs(wrong - rg.weighted_spearman(x, y, weights)) > 1e-3


def test_ownmask_noncontiguous_global_axis_and_condition_balance():
    a = arrays()
    result = rg.summarize_endpoint(**a)
    scores = []
    errors = []
    for i, row in enumerate(result["per_condition"]):
        keep = ~np.isin(a["allowed_gene_indices"], a["own_target_indices"][i])
        x, y = a["prediction_delta"][i, keep], a["truth_delta"][i, keep]
        score = np.corrcoef(x, y)[0, 1]
        error = np.mean((x - y) ** 2)
        assert row["ownmask_input_delta_Pearson"] == pytest.approx(score)
        assert row["ownmask_prediction_delta_MSE"] == pytest.approx(error)
        scores.append(score)
        errors.append(error)
    assert result["ownmask_input_delta_Pearson"]["mean"] == pytest.approx(np.mean(scores))
    assert result["ownmask_prediction_delta_MSE"]["mean"] == pytest.approx(np.mean(errors))


def test_pair_union_excludes_both_targets_and_direct_target_only_is_NA():
    p = 5
    a = arrays(p)
    a["allowed_gene_indices"] = np.arange(p)
    a["own_target_indices"] = np.arange(p)[:, None]
    a["truth_delta"] = np.eye(p)
    a["prediction_delta"] = np.eye(p)
    g = rg._geometry(**a)
    assert np.all(g.response_distance == 0)
    assert np.all(g.pair_genes == p - 2)
    summary = rg.summarize_endpoint(**a)
    assert summary["response_CLS_RSA"]["Spearman"] is None
    result = rg.compare_endpoints(a, a, draws=20)
    assert result["RSA_delta"]["valid_draws"] == 0
    assert result["RSA_delta"]["invalid_draws"] == 20


def test_condition_bootstrap_matches_resampled_pair_oracle():
    a = arrays(5)
    b = {**a, "prediction_delta": a["truth_delta"] + 0.1, "cls_means": a["cls_means"] ** 2}
    draws = 51
    seed = 20261028
    actual = rg.compare_endpoints(a, b, draws=draws, seed=seed)
    r, c = rg._geometry(**a), rg._geometry(**b)
    rng = np.random.default_rng(seed)
    rsa = []
    pd = []
    md = []
    invalid = 0
    for _ in range(draws):
        sampled = rng.integers(5, size=5)
        counts = np.bincount(sampled, minlength=5)
        weights = counts[r.pair_i] * counts[r.pair_j]
        rs = explicit(r.response_distance, r.cls_distance, weights)
        cs = explicit(c.response_distance, c.cls_distance, weights)
        if rs is None or cs is None:
            invalid += 1
        else:
            rsa.append(cs - rs)
        pd.append(np.mean((c.pearson - r.pearson)[sampled]))
        md.append(np.mean((c.mse - r.mse)[sampled]))
    for key, values in (
        ("RSA_delta", rsa),
        ("ownmask_Pearson_delta", pd),
        ("ownmask_MSE_delta", md),
    ):
        np.testing.assert_allclose(
            actual[key]["CI95"], np.quantile(values, [0.025, 0.975]), atol=2e-14
        )
    assert actual["RSA_delta"]["invalid_draws"] == invalid
    assert actual["RSA_delta"]["valid_draws"] == len(rsa)


def test_constant_point_one_exact_NA_and_no_axis_shrinking():
    a = arrays()
    b = {**a, "prediction_delta": a["prediction_delta"].copy()}
    b["prediction_delta"][2] = 0.1
    summary = rg.summarize_endpoint(**b)
    assert summary["per_condition"][2]["ownmask_input_delta_Pearson"] is None
    assert summary["per_condition"][2]["Pearson_NA_reason"] == "constant_vector"
    result = rg.compare_endpoints(a, b, draws=10)
    assert result["candidate_new_Pearson_NA"] == 1
    assert result["ownmask_Pearson_delta"]["CI95"] is None
    assert result["ownmask_Pearson_delta"]["invalid_draws"] == 10


def test_nonfinite_emptymask_and_identity_rejected():
    a = arrays()
    a["prediction_delta"][1, 1] = np.nan
    summary = rg.summarize_endpoint(**a)
    assert summary["per_condition"][1]["Pearson_NA_reason"] == "nonfinite_values"
    b = {**a, "condition_ids": a["condition_ids"][::-1]}
    with pytest.raises(ValueError, match="identity"):
        rg.compare_endpoints(a, b, draws=2)
    a = arrays()
    a["own_target_indices"] = np.tile(a["allowed_gene_indices"], (6, 1))
    result = rg.summarize_endpoint(**a)
    assert result["ownmask_prediction_delta_MSE"]["finite_N"] == 0
    assert result["response_CLS_RSA"]["Spearman"] is None


def test_squared_distance_is_sum_not_mean():
    a = arrays()
    g = rg._geometry(**a)
    keep = ~np.isin(
        a["allowed_gene_indices"], np.r_[a["own_target_indices"][0], a["own_target_indices"][1]]
    )
    expected = np.sum((a["truth_delta"][0, keep] - a["truth_delta"][1, keep]) ** 2)
    assert g.response_distance[0] == pytest.approx(expected)
    assert g.cls_distance[0] == pytest.approx(np.sum((a["cls_means"][0] - a["cls_means"][1]) ** 2))


def test_mask_precedes_finiteness_and_pair_rdm():
    a = arrays()
    # Condition1's direct target global0 is masked for its own score and all its pairs.
    a["prediction_delta"][1, 0] = np.nan
    a["truth_delta"][1, 0] = np.nan
    g = rg._geometry(**a)
    assert np.isfinite(g.pearson[1])
    assert np.isfinite(g.response_distance[(g.pair_i == 1) | (g.pair_j == 1)]).all()
    other = {**a, "truth_delta": a["truth_delta"].copy()}
    other["truth_delta"][0, 1] += 1
    with pytest.raises(ValueError, match="identity"):
        rg.compare_endpoints(a, other, draws=2)


def test_two_conditions_invalid_RSA_draws_are_not_rerolled():
    a = arrays(2)
    result = rg.compare_endpoints(a, a, draws=17)
    assert result["RSA_delta"]["invalid_draws"] == 17
    assert result["RSA_delta"]["valid_draws"] == 0
    assert result["ownmask_Pearson_delta"]["CI95"] == [0.0, 0.0]


def test_default_frozen_10000_seed_and_identical_endpoint_delta():
    a = arrays(4)
    result = rg.compare_endpoints(a, a)
    assert result["draws"] == 10000 and result["seed"] == 20261028
    assert result["RSA_delta"]["CI95"] == [0.0, 0.0]
    assert result["RSA_delta"]["valid_draws"] + result["RSA_delta"]["invalid_draws"] == 10000
    assert result["ownmask_MSE_delta"]["CI95"] == [0.0, 0.0]
