"""Raw project/official prior comparison and fixed-budget Scouter contracts."""

import pickle

import numpy as np
import pytest

from benchmarks.expression_holdout.scouter import (
    declared_prior_runtime,
    load_official_prior,
    resolve_training_recipe,
)
from gradpert.hashing import sha256_file


def _runtime(path, kind="project_genept_seed", dimension=2048, **changes):
    return {
        "prior_path": str(path),
        "prior_sha256": sha256_file(path),
        "prior_kind": kind,
        "prior_dimension": dimension,
        "prior_representation": "raw",
        "prior_gene_key": "genes",
        "prior_embedding_key": "vectors",
        "required_target_genes": ["A", "T"],
        **changes,
    }


def test_raw_project_npz_vectors_have_separate_model_and_dimension_provenance(tmp_path):
    path = tmp_path / "raw_project.npz"
    vectors = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048)
    np.savez(
        path,
        genes=np.asarray(["A", "T", "unused"]),
        vectors=vectors,
        model=np.asarray("doubao-embedding-vision"),
    )
    frame, receipt = load_official_prior(_runtime(path), {"A", "ctrl"})
    assert frame.shape == (3, 2048)
    assert list(frame.index) == ["A", "T", "ctrl"]
    np.testing.assert_array_equal(frame.loc["T"].values, vectors[1])
    assert receipt["prior_kind"] == "project_genept_seed"
    assert receipt["prior_dimension"] == 2048
    assert receipt["prior_source_model"] == "doubao-embedding-vision"
    assert receipt["prior_representation"] == "raw"
    with pytest.raises(ValueError, match="dimension differs"):
        load_official_prior(_runtime(path, dimension=1536), {"A"})
    with pytest.raises(ValueError, match="raw source"):
        load_official_prior(_runtime(path, prior_representation="PCA"), {"A"})


@pytest.mark.parametrize("bad", ["duplicates", "nonfinite", "axes", "object"])
def test_npz_rejects_invalid_axes_and_vectors(tmp_path, bad):
    path = tmp_path / "invalid.npz"
    genes = np.asarray(["A", "T"])
    vectors = np.ones((2, 2048), dtype=np.float32)
    if bad == "duplicates":
        genes[1] = "A"
    elif bad == "nonfinite":
        vectors[0, 0] = np.nan
    elif bad == "axes":
        vectors = vectors[:1]
    else:
        genes = genes.astype(object)
    np.savez(path, genes=genes, vectors=vectors)
    with pytest.raises(ValueError):
        load_official_prior(_runtime(path), {"A"})


def test_official_raw_pickle_width_does_not_merge_with_project_vectors(tmp_path):
    path = tmp_path / "raw_official.pickle"
    with path.open("wb") as handle:
        pickle.dump({"A": np.ones(1536), "T": np.full(1536, 2)}, handle)
    frame, receipt = load_official_prior(
        _runtime(path, kind="official_genept", dimension=1536, prior_source_model="GPT-3.5 GenePT"),
        {"A"},
    )
    assert frame.shape == (3, 1536)
    assert receipt["prior_kind"] == "official_genept"
    assert receipt["prior_dimension"] == 1536
    with path.open("wb") as handle:
        pickle.dump({"A": np.ones(1536), "T": np.ones(2048)}, handle)
    with pytest.raises(ValueError, match="mixes embedding dimensions"):
        load_official_prior(_runtime(path, kind="official_genept", dimension=1536), {"A"})


def test_only_explicit_missing_condition_target_can_be_omitted(tmp_path):
    path = tmp_path / "official.npz"
    np.savez(path, genes=np.asarray(["A", "T"]), vectors=np.ones((2, 2048)))
    runtime = _runtime(path, kind="official_genept", required_target_genes=["A", "T", "CCDC130"])
    with pytest.raises(ValueError, match="fit/test targets"):
        load_official_prior(runtime, {"A"})
    runtime["excluded_conditions"] = ["CCDC130+ctrl"]
    frame, receipt = load_official_prior(runtime, {"A"})
    assert "CCDC130" not in frame.index
    assert receipt["authorized_exclusions"] == ["CCDC130+ctrl"]
    assert receipt["excluded_missing_target_genes"] == ["CCDC130"]
    with pytest.raises(ValueError, match="remains in prepared fit"):
        load_official_prior(runtime, {"A", "CCDC130"})
    runtime["required_target_genes"] = ["A", "T", "CCDC130", "other_missing"]
    with pytest.raises(ValueError, match="fit/test targets"):
        load_official_prior(runtime, {"A"})


def test_four_epochs_keep_official_patience_five_and_never_stop_before_budget():
    recipe = {"n_epochs": 4, "patience": 5, "lr": 0.001, "sched_gamma": 0.9}
    assert resolve_training_recipe(recipe) == recipe
    assert resolve_training_recipe({**recipe, "patience": 2})["patience"] == 4
    assert recipe["patience"] == 5
    with pytest.raises(ValueError, match="positive"):
        resolve_training_recipe({**recipe, "n_epochs": 0})


def test_runtime_cannot_expand_sealed_config_condition_exclusions():
    config = {"data": {"omit_condition_ids": ["CCDC130+ctrl"]}}
    runtime = {"excluded_conditions": ["CCDC130+ctrl"], "prior_kind": "official_genept"}
    assert declared_prior_runtime(config, runtime)["excluded_conditions"] == ("CCDC130+ctrl",)
    assert declared_prior_runtime(config, {})["excluded_conditions"] == ("CCDC130+ctrl",)
    with pytest.raises(ValueError, match="exclusions disagree"):
        declared_prior_runtime({}, runtime)
    with pytest.raises(ValueError, match="exclusions disagree"):
        declared_prior_runtime(config, {"excluded_conditions": []})
    with pytest.raises(ValueError, match="exclusions disagree"):
        declared_prior_runtime(config, {"excluded_conditions": ["CCDC130+ctrl", "OTHER+ctrl"]})
    assert runtime["excluded_conditions"] == ["CCDC130+ctrl"]
