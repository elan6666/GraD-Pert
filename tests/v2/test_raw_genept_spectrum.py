"""Raw prior spectra and frozen gene grouping; no scientific artifacts used."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gradpert.hashing import sha256_file, sha256_json

SPEC = importlib.util.spec_from_file_location(
    "raw_genept", Path(__file__).resolve().parents[2] / "scripts/v2/analyze_raw_genept.py"
)
raw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raw)


def test_rank_one_and_isotropic():
    rank1 = np.array([[-2, -4], [-1, -2], [1, 2], [2, 4]])
    stats, _ = raw.spectrum(rank1)
    assert stats["effective_rank"] == pytest.approx(1)
    assert stats["top1_fraction"] == pytest.approx(1)
    isotropic = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    stats, eig = raw.spectrum(isotropic)
    np.testing.assert_allclose(eig, [0.5, 0.5])
    assert stats["effective_rank"] == pytest.approx(2)
    assert stats["relative_variance"] == pytest.approx(1)
    assert stats["k90"] == stats["k95"] == stats["k99"] == 2
    assert stats["mean_pair_cosine"] == pytest.approx(-1 / 3)


def test_constant_zero_and_single_row_definedness():
    stats, eig = raw.spectrum(np.ones((5, 3)))
    assert stats["effective_rank"] == 0 and stats["relative_variance"] == 0
    assert stats["top1_fraction"] is None and stats["k90"] is None
    assert stats["mean_pair_cosine"] == pytest.approx(1)
    assert not eig.any()
    stats, _ = raw.spectrum(np.zeros((2, 3)))
    assert stats["relative_variance"] is None and stats["mean_pair_cosine"] is None
    assert stats["cosine_zero_rows_excluded"] == 2
    stats, _ = raw.spectrum([[1, 2]])
    assert stats["effective_rank"] is None and stats["normalized_effective_rank"] is None


def test_scale_translation_and_cosine_reference():
    x = np.random.default_rng(5).normal(size=(9, 4))
    base, eig = raw.spectrum(x)
    scaled, bigger = raw.spectrum(7 * x)
    shifted, translated = raw.spectrum(x + np.array([10, 20, 30, 40]))
    for key in ("effective_rank", "relative_variance", "mean_pair_cosine"):
        assert scaled[key] == pytest.approx(base[key])
    np.testing.assert_allclose(bigger, eig * 49)
    np.testing.assert_allclose(translated, eig)
    assert shifted["effective_rank"] == pytest.approx(base["effective_rank"])
    assert shifted["relative_variance"] < base["relative_variance"]
    assert scaled["absolute_variance"] == pytest.approx(base["absolute_variance"] * 49)
    normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    brute = normalized @ normalized.T
    assert base["mean_pair_cosine"] == pytest.approx(brute[~np.eye(len(x), dtype=bool)].mean())


@pytest.mark.parametrize("shape", [(4, 11), (11, 4), (4, 4)])
def test_gram_covariance_parity(shape):
    x = np.random.default_rng(42).normal(size=shape)
    cov, a = raw.spectrum(x, method="covariance")
    gram, b = raw.spectrum(x, method="gram")
    np.testing.assert_allclose(a, b, atol=1e-13)
    assert len(a) == shape[1]
    for key in (
        "effective_rank",
        "normalized_effective_rank",
        "top1_fraction",
        "absolute_variance",
        "relative_variance",
        "mean_pair_cosine",
        "k90",
        "k95",
        "k99",
    ):
        assert cov[key] == pytest.approx(gram[key])


@pytest.mark.parametrize("x", [[], [[float("nan")]], [[float("inf")]], [1, 2]])
def test_nonfinite_or_invalid_input_rejected(x):
    with pytest.raises(ValueError, match="finite two-dimensional"):
        raw.spectrum(x)


def inputs(tmp_path):
    genes = ("a", "b", "c", "d", "e", "f")
    prior = tmp_path / "prior.npz"
    np.savez(
        prior,
        genes=np.array(genes),
        vectors=np.arange(18).reshape(6, 3) + 1,
        model=np.array("synthetic"),
    )
    graph, expression = ("b", "a", "d", "c"), ("a", "b", "c")
    gp, ep = tmp_path / "graph.txt", tmp_path / "expression.txt"
    gp.write_text("\n".join(graph) + "\n")
    ep.write_text("\n".join(expression) + "\n")
    data = {
        "graph_gene_order_sha256": sha256_json(list(graph)),
        "expression_gene_order_sha256": sha256_json(list(expression)),
        "genept_sha256": sha256_file(prior),
        "training_expression_policy": {
            "exclude_test_target_expression": True,
            "excluded_expression_gene_ids": ["b"],
            "excluded_expression_gene_ids_sha256": sha256_json(["b"]),
            "allowed_expression_gene_ids_sha256": sha256_json(["a", "c"]),
            "expression_gene_order_sha256": sha256_json(list(expression)),
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"data": data}))
    return prior, gp, ep, manifest, graph, expression, {"data": data}


def test_verified_raw_group_axes_do_not_mix_graph_only_genes(tmp_path):
    prior, gp, ep, manifest, _, _, _ = inputs(tmp_path)
    before = {p: sha256_file(p) for p in (prior, gp, ep, manifest)}
    matrices, groups, _, model = raw.load_groups(
        prior, sha256_file(prior), gp, ep, manifest, sha256_file(manifest), width=3, source_count=6
    )
    assert model == "synthetic"
    assert groups["seen_expression"] == ("a", "c") and groups["unseen_expression"] == ("b",)
    assert "d" in groups["graph_all"] and "d" not in groups["seen_expression"]
    np.testing.assert_array_equal(matrices["graph_all"][0], [4, 5, 6])
    assert before == {p: sha256_file(p) for p in before}


@pytest.mark.parametrize(
    "key",
    [
        "graph_gene_order_sha256",
        "expression_gene_order_sha256",
        "genept_sha256",
        "allowed_expression_gene_ids_sha256",
        "excluded_expression_gene_ids_sha256",
    ],
)
def test_group_hash_mismatch_rejected(tmp_path, key):
    prior, _, _, _, graph, expression, manifest = inputs(tmp_path)
    bad = copy.deepcopy(manifest)
    target = bad["data"] if key in bad["data"] else bad["data"]["training_expression_policy"]
    target[key] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        raw.validate_groups(bad, graph, expression, sha256_file(prior))


def test_manifest_prior_hashes_and_server_path_rejected(tmp_path):
    prior, gp, ep, manifest, _, _, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="run manifest hash"):
        raw.load_groups(
            prior, sha256_file(prior), gp, ep, manifest, "0" * 64, width=3, source_count=6
        )
    with pytest.raises(ValueError, match="genept_sha256 mismatch"):
        raw.load_groups(
            prior, "0" * 64, gp, ep, manifest, sha256_file(manifest), width=3, source_count=6
        )
    with pytest.raises(ValueError, match="/data/yilangliu"):
        raw.server_path(tmp_path)
