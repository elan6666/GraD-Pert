"""Synthetic adapter invariants; real frozen-model integration is a separate gate."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.expression_holdout.contract import FittedAdapter, PreparedData
from benchmarks.expression_holdout.scdfm import (
    ScDFMState,
    _safe_graph_matrix,
    _training_groups,
    _verify_checkout,
    official_namespace,
    perturbation_tokens,
    predict,
    sample_batch,
)


@pytest.fixture
def prepared():
    import anndata

    adata = anndata.AnnData(
        X=np.array([[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0], [90, 91, 0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"condition": ["control", "control", "a+control", "a+control", "b+control"]},
            index=["c0", "c1", "t0", "t1", "v0"],
        ),
        var=pd.DataFrame(index=["a", "b", "u"]),
    )
    return PreparedData(
        adata=adata,
        gene_ids=("a", "b", "u"),
        graph_gene_ids=("a", "b", "u", "g"),
        train_conditions=("a+control",),
        val_conditions=("b+control",),
        allowed_indices=(0, 1),
        excluded_indices=(2,),
        train_row_ids=("t0", "t1"),
        val_row_ids=("v0",),
        control_row_ids=("c0", "c1"),
        row_context_ids=("ctx0", "ctx1", "ctx0", "ctx1", "ctx0"),
        train_control_indices={"t0": (0,), "t1": (1,)},
        val_control_indices={"v0": (0,)},
        receipt={},
    )


def test_grouped_sampling_preserves_context_and_allowed_expression(prepared):
    groups = _training_groups(prepared)
    observed = set()
    rng = np.random.default_rng(7)
    for _ in range(50):
        source, target, condition, controls, targets = sample_batch(prepared, groups, 8, rng)
        assert condition == "a+control"
        assert source.shape == target.shape == (8, 2)
        assert set(targets) <= {2, 3}
        assert all(
            prepared.row_context_ids[c] == prepared.row_context_ids[t]
            for c, t in zip(controls, targets, strict=True)
        )
        assert len({prepared.row_context_ids[t] for t in targets}) == 1
        observed.update(targets)
    assert observed == {2, 3}


def test_context_mismatch_fails_before_training(prepared):
    prepared.train_control_indices["t0"] = (1,)
    with pytest.raises(ValueError, match="context mismatch"):
        _training_groups(prepared)


def test_graph_statistics_exclude_validation_values_and_unmeasured_identities(prepared):
    graph = _safe_graph_matrix(prepared, prepared.graph_gene_ids)
    assert graph.shape == (4, 4)
    np.testing.assert_array_equal(graph[:, 2:], 0)
    prepared.adata.X[4, :2] = 123456
    np.testing.assert_array_equal(_safe_graph_matrix(prepared, prepared.graph_gene_ids), graph)


def test_unsanitized_excluded_values_fail_closed(prepared):
    prepared.adata.X[0, 2] = 999
    with pytest.raises(ValueError, match="unsanitized"):
        _safe_graph_matrix(prepared, prepared.graph_gene_ids)


def test_official_namespace_restores_existing_modules_after_failure(tmp_path):
    sentinel = ModuleType("src")
    previous = sys.modules.get("src")
    sys.modules["src"] = sentinel
    cached = {}
    old_path = sys.path[:]
    try:
        with pytest.raises(RuntimeError), official_namespace(tmp_path, cached):
            assert "src" not in sys.modules
            sys.modules["src"] = ModuleType("src")
            raise RuntimeError("diagnostic")
        assert sys.modules["src"] is sentinel
        assert cached["src"] is not sentinel
        assert sys.path == old_path
    finally:
        if previous is None:
            sys.modules.pop("src", None)
        else:
            sys.modules["src"] = previous


def test_frozen_commit_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "bad-version\n")
    with pytest.raises(ValueError, match="frozen version mismatch"):
        _verify_checkout(Path("/diagnostic-only"))


class TinyVocab(dict):
    def encode(self, genes):
        return [self[gene] for gene in genes]


def test_single_target_retains_official_control_token():
    vocab = TinyVocab(a=1, b=2, control=0)
    assert perturbation_tokens(vocab, ("a",)) == [1, 0]
    assert perturbation_tokens(vocab, ("a", "b")) == [1, 2]
    with pytest.raises(ValueError, match="missing"):
        perturbation_tokens(vocab, ("u",))


def test_inference_covers_all_genes_and_noise_does_not_depend_on_cell_batch(tmp_path):
    import torch

    # An official-compatible test double checks only adapter IO/noise semantics.
    # It is not used as a replacement model or scientific integration evidence.
    run = SimpleNamespace(torch=torch, wrapped_vf=object())
    observed = []

    def generate_sample(wrapped, source, condition_vec, vf, gene_ids, gene_all, steps, method):
        assert wrapped is run.wrapped_vf
        assert steps == 20 and method == "rk4"
        assert run.config.infer_top_gene == source.shape[1]
        assert torch.equal(gene_ids, gene_all)
        assert torch.equal(condition_vec[:, 1], torch.zeros(len(source), dtype=torch.long))
        noise = run.torch.randn(source.shape[0], run.config.infer_top_gene, device=source.device)
        observed.append((source.clone(), gene_ids.clone(), noise.clone()))
        return noise + gene_ids[None, :] + source * 0.001

    run.generate_sample = generate_sample
    official = SimpleNamespace(run=run, modules={})
    state = ScDFMState(
        torch.nn.Linear(1, 1),
        official,
        tmp_path,
        TinyVocab(a=1, b=2, u=3, control=0),
        ("a", "b", "u"),
        torch.device("cpu"),
        {"noise_type": "Gaussian"},
    )
    fitted = FittedAdapter(state, {})
    controls = np.arange(15, dtype=np.float32).reshape(5, 3)
    first = predict(fitted, controls, ("a",), state.gene_ids, cell_batch=2, query_count=2, seed=8)
    hashes = [item["sha256"] for item in fitted.receipt["inference_noise"]]
    fitted.receipt["inference_noise"] = []
    second = predict(fitted, controls, ("a",), state.gene_ids, cell_batch=5, query_count=2, seed=8)
    np.testing.assert_array_equal(first, second)
    assert hashes == [item["sha256"] for item in fitted.receipt["inference_noise"]]
    assert first.shape == (5, 3) and np.isfinite(first).all()
    assert any(gene.tolist() == [3] for _, gene, _ in observed)
    assert run.torch is torch
    with pytest.raises(ValueError, match="axis"):
        predict(fitted, controls, ("a",), ("b", "a", "u"), cell_batch=2, query_count=2, seed=8)
