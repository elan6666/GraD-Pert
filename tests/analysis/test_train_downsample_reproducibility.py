"""Sampling, grouping and numeric invariants of the train-only cap analysis."""

import numpy as np
import pytest

from gradpert.evaluation.metrics import pearson_correlation
from scripts.analysis.train_downsample_reproducibility import (
    analyze,
    export_subset,
    fraction_count,
    half_assignments,
    proportional_sample,
    repeatability,
    row_pearson,
)


def test_fraction_rounding_minimum_and_stratified_quota():
    assert [fraction_count(n, 0.5) for n in range(1, 8)] == [1, 2, 2, 2, 3, 3, 4]
    assert [fraction_count(n, 0.25) for n in range(1, 10)] == [1, 2, 2, 2, 2, 2, 2, 2, 2]
    batches = np.asarray(["a"] * 61 + ["b"] * 23 + ["c"] * 17)
    for fraction in (0.5, 0.25):
        count = fraction_count(len(batches), fraction)
        rows = proportional_sample(batches, count, seed=42)
        assert len(rows) == count and len(np.unique(rows)) == count
        for name in np.unique(batches):
            quota = (batches == name).sum() * count / len(batches)
            assert (batches[rows] == name).sum() in (np.floor(quota), np.ceil(quota))
    for n, fraction in ((0, 0.5), (3, 0), (3, np.nan), (3, 1.1)):
        with pytest.raises(ValueError):
            fraction_count(n, fraction)


def test_cross_run_comparison_rejects_incompatible_control_reference(tmp_path):
    import json
    from pathlib import Path

    from gradpert.hashing import sha256_file
    from scripts.analysis.compare_train_sampling import collect

    fixture = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/data/jurkat-cap40-5a8a7bb-20260930T094437Z"
    )
    inputs = []
    for method in ("cap40", "proportion50", "proportion25"):
        root = tmp_path / method
        root.mkdir()
        for name in ("summary.json", "conditions.json", "receipt.json", "COMPLETE.json"):
            (root / name).write_bytes((fixture / name).read_bytes())
        summary = json.loads((root / "summary.json").read_text())
        summary["method"] = method
        (root / "summary.json").write_text(json.dumps(summary))
        for name in ("receipt.json", "COMPLETE.json"):
            payload = json.loads((root / name).read_text())
            payload["summary_sha256"] = sha256_file(root / "summary.json")
            (root / name).write_text(json.dumps(payload))
        inputs.append(root)
    sources, records = collect(inputs)
    assert set(sources) == {"cap40", "proportion50", "proportion25"}
    assert len(records) == 1335
    summary["control_reference_content_sha256"] = "invalid-control"
    (inputs[-1] / "summary.json").write_text(json.dumps(summary))
    for name in ("receipt.json", "COMPLETE.json"):
        payload = json.loads((inputs[-1] / name).read_text())
        payload["summary_sha256"] = sha256_file(inputs[-1] / "summary.json")
        (inputs[-1] / name).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incompatible reference"):
        collect(inputs)


def test_cap_keeps_small_conditions_and_preserves_batch_quotas():
    batches = np.asarray(["a"] * 61 + ["b"] * 23 + ["c"] * 16)
    rows = proportional_sample(batches, 40, seed=7)
    assert len(rows) == 40 and len(np.unique(rows)) == 40
    np.testing.assert_array_equal(rows, proportional_sample(batches, 40, seed=7))
    for name in np.unique(batches):
        quota = (batches == name).sum() * 0.4
        observed = (batches[rows] == name).sum()
        assert observed in (np.floor(quota), np.ceil(quota))
    np.testing.assert_array_equal(proportional_sample(batches[:20], 40, seed=7), np.arange(20))
    with pytest.raises(ValueError):
        proportional_sample(batches, 0, seed=7)


def test_halves_balance_every_batch_and_randomize_singletons():
    batches = np.asarray(["a"] * 7 + ["b"] * 4 + ["c", "d", "e", "f"])
    assignments = half_assignments(batches, repeats=100, seed=19)
    np.testing.assert_array_equal(assignments, half_assignments(batches, repeats=100, seed=19))
    assert set(np.unique(assignments)) == {0, 1}
    assert np.all(np.abs((assignments == 0).sum(axis=1) - (assignments == 1).sum(axis=1)) <= 1)
    for name in np.unique(batches):
        local = assignments[:, batches == name]
        assert np.all(np.abs((local == 0).sum(axis=1) - (local == 1).sum(axis=1)) <= 1)
    assert set(assignments[:, -1]) == {0, 1}


def test_vectorized_pearson_and_half_means_match_direct_calculation():
    values = np.random.default_rng(4).normal(size=(9, 12))
    batches = np.asarray(["a"] * 5 + ["b"] * 4)
    control = np.linspace(0, 1, 12)
    delta, raw, _ = repeatability(values, batches, control, repeats=10, seed=23)
    assignments = half_assignments(batches, repeats=10, seed=23)
    for i, labels in enumerate(assignments):
        left = values[labels == 0].mean(axis=0)
        right = values[labels == 1].mean(axis=0)
        assert delta[i] == pytest.approx(
            pearson_correlation(left - control, right - control)[0], abs=1e-14
        )
        assert raw[i] == pytest.approx(pearson_correlation(left, right)[0], abs=1e-14)
    result = row_pearson(np.ones((1, 4)), np.arange(4)[None])
    assert np.isnan(result[0])
    unavailable, _, _ = repeatability(values[:1], batches[:1], control, repeats=10, seed=23)
    assert np.isnan(unavailable).all()
    rows = proportional_sample(batches, 40, seed=1)
    unchanged, _, _ = repeatability(values[rows], batches[rows], control, repeats=10, seed=23)
    np.testing.assert_array_equal(delta, unchanged)


def test_h5ad_export_preserves_nontrain_rows_gene_identity_and_all_values(tmp_path):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    expression = np.arange(60, dtype=np.float32).reshape(10, 6)
    data = ad.AnnData(
        expression,
        obs=pd.DataFrame(
            {
                "condition": pd.Categorical(["A"] * 5 + ["B"] * 2 + ["ctrl"] * 3),
                "annotation": pd.Categorical(
                    ["kept", "removed", "kept", "removed", "removed"] + ["kept"] * 5,
                    categories=["removed", "kept", "unused"],
                    ordered=True,
                ),
            },
            index=[f"row{i}" for i in range(10)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )
    # Only A is the training condition; B and control are untouched.
    kept = np.asarray([0, 2, 5, 6, 7, 8, 9])
    receipt = export_subset(data, kept, tmp_path / "cap.h5ad", {"purpose": "synthetic"})
    assert receipt["retained_values_exact"] and receipt["obs_var_metadata_exact"]
    output = ad.read_h5ad(tmp_path / "cap.h5ad")
    np.testing.assert_array_equal(output.X, expression[kept])
    assert output.obs_names.tolist() == [f"row{i}" for i in kept]
    pd.testing.assert_frame_equal(output.obs, data.obs.iloc[kept])
    assert output.uns["gradpert_downsample"]["purpose"] == "synthetic"


@pytest.mark.parametrize("fraction,expected", [(None, 40), (0.5, 23), (0.25, 11)])
def test_complete_synthetic_analysis_keeps_split_controls_and_skips_one_cell_condition(
    tmp_path, fraction, expected
):
    import json

    from gradpert.hashing import sha256_file, sha256_json

    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    root = tmp_path / "parent"
    (root / "canonical").mkdir(parents=True)
    (root / "manifests").mkdir()
    labels = ["A"] * 45 + ["B"] + ["ctrl"] * 10 + ["V"] * 4 + ["T"] * 4 + ["E"] * 2
    matrix = np.random.default_rng(3).uniform(size=(len(labels), 6)).astype(np.float32)
    data = ad.AnnData(
        matrix,
        obs=pd.DataFrame(
            {
                "condition": labels,
                "batch": ["x", "y"] * (len(labels) // 2),
                "control": np.asarray(labels) == "ctrl",
            },
            index=[f"row{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(
            {"gene_name": [f"g{i}" for i in range(6)]},
            index=[f"ENSG{i}" for i in range(6)],
        ),
    )
    data.write_h5ad(root / "canonical/adata.h5ad")
    (root / "canonical/expression_gene_ids.txt").write_text("g0\ng1\ng2\ng3\n")
    manifests = {
        "canonical.json": {
            "canonical_adata_sha256": sha256_file(root / "canonical/adata.h5ad"),
            "n_cells": len(labels),
            "n_graph_genes": 6,
            "n_expression_genes": 4,
            "expression_gene_order_sha256": sha256_json([f"g{i}" for i in range(4)]),
            "observation_order_sha256": sha256_json(data.obs_names.tolist()),
        },
        "split.json": {
            "train_conditions": ["A", "B"],
            "val_conditions": ["V"],
            "test_conditions": ["T"],
            "control_condition_id": "ctrl",
        },
        "evaluation_controls.val.json": {"immutable": "val"},
        "evaluation_controls.test.json": {"immutable": "test"},
    }
    for name, payload in manifests.items():
        (root / "manifests" / name).write_text(json.dumps(payload))
    output = tmp_path / "analysis"
    output.mkdir()
    result = analyze(root, output, repeats=5, seed=42, source="a" * 40, fraction=fraction)
    method = "cap40" if fraction is None else f"proportion{int(fraction * 100)}"
    assert result["original_train_cells"] == 46
    assert result["sampled_train_cells"] == expected + 1
    assert result["original_data_unchanged"]
    assert result["original"]["finite_count"] == result[method]["finite_count"] == 1
    selected = json.loads((output / "selection.json").read_text())
    assert len(selected["by_condition"]["A"]) == expected
    assert selected["fraction"] == fraction
    assert selected["by_condition"]["B"] == ["row45"]
    derivative = ad.read_h5ad(output / f"{method}.h5ad")
    original = ad.read_h5ad(root / "canonical/adata.h5ad")
    keep = original.obs_names.isin(derivative.obs_names)
    np.testing.assert_array_equal(derivative.X, original.X[keep])
    assert (
        derivative.obs_names[derivative.obs.condition.isin(["ctrl", "V", "T", "E"])].tolist()
        == original.obs_names[original.obs.condition.isin(["ctrl", "V", "T", "E"])].tolist()
    )
    assert result["derived_h5ad"]["genes"] == 6
    assert (output / "comparison.pdf").stat().st_size > 1000
