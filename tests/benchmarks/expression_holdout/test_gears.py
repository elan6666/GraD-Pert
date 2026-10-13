"""GEARS official-loss/control and context/statistics adapter invariants."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks.expression_holdout.gears import (
    allowed_training_statistics,
    context_graph_builder,
    copy_official_assets,
    masked_official_loss,
    official_control_metadata,
    predict,
)
from tests.benchmarks.expression_holdout.test_scouter import _prepared


def test_control_loss_cannot_supervise_excluded_zero_placeholders():
    torch = pytest.importorskip("torch")
    calls = []

    def official(pred, truth, perts, ctrl, direction_lambda, dict_filter):
        calls.append((pred.shape[1], ctrl.tolist(), dict_filter, direction_lambda))
        # Frozen upstream special-cases ctrl and ignores dict_filter for it.
        return (pred - truth).pow(4).mean()

    pred = torch.tensor([[1.0, 99.0, 2.0]], requires_grad=True)
    loss = masked_official_loss(official, np.asarray([0, 2]))(
        pred,
        torch.zeros_like(pred),
        ["ctrl"],
        torch.tensor([3.0, 88.0, 4.0]),
        direction_lambda=0.1,
        dict_filter={"A+ctrl": [0, 1, 2]},
    )
    loss.backward()
    assert loss.item() == 8.5
    assert pred.grad.tolist() == [[2, 0, 16]]
    assert calls == [(2, [3, 4], {"A+ctrl": [0, 1]}, 0.1)]


def test_context_builder_calls_official_sampler_and_records_exact_selected_ids():
    prepared = _prepared()
    instance = SimpleNamespace(ctrl_adata=prepared.adata[["c0", "c1"]])
    original_controls = instance.ctrl_adata
    seen = []

    def original(adata, condition, num_samples):
        sampled = np.random.randint(0, len(instance.ctrl_adata), num_samples)
        seen.append(
            (tuple(adata.obs_names), tuple(instance.ctrl_adata.obs_names), sampled.tolist())
        )
        return [SimpleNamespace() for _ in sampled]

    audit = []
    randint = np.random.randint
    graphs = context_graph_builder(original, prepared, audit)(instance, prepared.adata, "A+ctrl", 2)
    assert len(graphs) == 2
    assert seen == [(("t",), ("c0",), [0, 0])]
    assert audit == [("t", "c0"), ("t", "c0")]
    assert instance.ctrl_adata is original_controls
    assert np.random.randint is randint


def test_context_builder_restores_scopes_after_failure():
    prepared = _prepared()
    controls = prepared.adata[["c0", "c1"]]
    instance = SimpleNamespace(ctrl_adata=controls)

    def fail(*args):
        raise RuntimeError("upstream failure")

    randint = np.random.randint
    with pytest.raises(RuntimeError, match="upstream failure"):
        context_graph_builder(fail, prepared, [])(instance, prepared.adata, "A+ctrl", 1)
    assert np.random.randint is randint
    assert instance.ctrl_adata is controls


def test_cached_context_pools_keep_graphs_audit_rng_and_inputs_exact():
    from copy import deepcopy

    prepared = _prepared()
    # Two same-context rows and a second condition use the identical basal pool.
    extra = prepared.adata[["t", "t"]].copy()
    extra.obs_names = ["t2", "t3"]
    import anndata
    import pandas as pd

    adata = anndata.concat([prepared.adata, extra], merge="same")
    adata.obs["condition"] = adata.obs.condition.astype(str)
    adata.obs.loc["t3", "condition"] = "B+ctrl"
    adata.uns["immutable_metadata"] = {"values": list(range(100))}
    safe = SimpleNamespace(
        adata=adata,
        row_context_ids=["ctx0", "ctx1", "ctx0", "ctx1", "ctx0", "ctx0"],
        train_control_indices={"t": (2,), "t2": (2,), "t3": (2,)},
        val_control_indices={"v": (3,)},
    )
    before_X = adata.X.copy()
    before_obs, before_var, before_uns = adata.obs.copy(), adata.var.copy(), deepcopy(adata.uns)
    instance = SimpleNamespace(ctrl_adata=adata[["c0", "c1"]])

    def official(target, condition, num_samples):
        selected = np.random.randint(0, len(instance.ctrl_adata), num_samples)
        return [
            (instance.ctrl_adata.X[i].toarray(), target.X.toarray(), condition) for i in selected
        ]

    expected_graphs, expected_audit = [], []
    maps = {**safe.train_control_indices, **safe.val_control_indices}
    np.random.seed(71)
    for condition in ("A+ctrl", "B+ctrl"):
        for row in adata.obs_names[adata.obs.condition == condition]:
            pool = maps[row]
            instance.ctrl_adata = adata[[adata.obs_names[i] for i in pool]].copy()
            expected_graphs.extend(official(adata[[row]].copy(), condition, 3))
            expected_audit.extend([(str(row), str(adata.obs_names[pool[0]]))] * 3)
    expected_rng = np.random.get_state()
    audit = []
    builder = context_graph_builder(official, safe, audit)
    np.random.seed(71)
    graphs = []
    for condition in ("A+ctrl", "B+ctrl"):
        graphs.extend(builder(instance, adata, condition, 3))
    assert len(graphs) == len(expected_graphs)
    for actual, expected in zip(graphs, expected_graphs, strict=True):
        for a, e in zip(actual[:2], expected[:2], strict=True):
            np.testing.assert_array_equal(a, e)
        assert actual[2] == expected[2]
    assert audit == expected_audit
    actual_rng = np.random.get_state()
    assert actual_rng[0] == expected_rng[0] and actual_rng[2:] == expected_rng[2:]
    np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
    assert builder.cache_statistics == {
        "pool_materializations": 2,
        "pool_hits": 2,
        "official_row_calls": 4,
    }
    np.testing.assert_array_equal(adata.X.toarray(), before_X.toarray())
    pd.testing.assert_frame_equal(adata.obs, before_obs)
    pd.testing.assert_frame_equal(adata.var, before_var)
    assert adata.uns == before_uns


def test_bad_new_pool_is_checked_even_after_context_cache_hit():
    prepared = _prepared()
    mutable_maps = dict(prepared.train_control_indices)
    safe = SimpleNamespace(
        adata=prepared.adata,
        row_context_ids=prepared.row_context_ids,
        train_control_indices=mutable_maps,
        val_control_indices=prepared.val_control_indices,
    )
    instance = SimpleNamespace(ctrl_adata=prepared.adata[["c0"]])

    def official(*args):
        np.random.randint(0, len(instance.ctrl_adata), 1)
        return [SimpleNamespace()]

    builder = context_graph_builder(official, safe, [])
    builder(instance, prepared.adata, "A+ctrl", 1)
    mutable_maps["t"] = (3,)
    # The closure snapshots the mappings. A second independently constructed
    # builder must reject cross-context candidates instead of trusting context.
    builder = context_graph_builder(official, safe, [])
    with pytest.raises(ValueError, match="invalid canonical"):
        builder(instance, prepared.adata, "A+ctrl", 1)


def test_ranking_cache_requires_exact_input_and_sealed_parent(tmp_path, monkeypatch):
    import hashlib
    import inspect
    import json

    from benchmarks.expression_holdout import gears
    from gradpert.hashing import sha256_file

    prepared = _prepared()
    adata = official_control_metadata(prepared.adata.copy())
    adata.uns["rank_genes_groups_cov_all"] = {"cond": ["A", "B"]}
    adata.uns["non_zeros_gene_idx"] = {"cond": [0, 2]}
    adata.uns["non_dropout_gene_idx"] = {"cond": [0, 2]}
    parent = tmp_path / "old"
    path = parent / "fit/official_data/expression_holdout/perturb_processed.h5ad"
    path.parent.mkdir(parents=True)
    adata.write_h5ad(path)
    (parent / "RUN.json").write_text(
        json.dumps(
            {
                "source_commit": "original",
                "config_sha256": "config",
                "environment_lock_sha256": "env",
            }
        )
    )
    (parent / "data-receipt.json").write_text("{}")
    cache = {
        "path": str(path),
        "sha256": sha256_file(path),
        "parent_run_root": str(parent),
        "parent_run_sha256": sha256_file(parent / "RUN.json"),
        "data_receipt_sha256": sha256_file(parent / "data-receipt.json"),
        "ranking_method_sha256": hashlib.sha256(
            inspect.getsource(gears._ranking_metadata).encode()
        ).hexdigest(),
        "source_commit": "original",
        "config_sha256": "config",
    }
    runtime = {"environment_lock_sha256": "env", "gears_ranking_cache": cache}
    monkeypatch.setattr(gears, "CACHE_SERVER_ROOT", tmp_path)
    target = adata.copy()
    target.uns.clear()
    result = gears.reuse_ranking_metadata(target, np.asarray([0, 2]), runtime)
    assert result["cache_reused"]
    assert list(target.uns["rank_genes_groups_cov_all"]["cond"]) == ["A", "B"]
    wrong = target.copy()
    wrong.X[0, 0] += 1
    with pytest.raises(ValueError, match="expression differs"):
        gears.reuse_ranking_metadata(wrong, np.asarray([0, 2]), runtime)
    with pytest.raises(ValueError, match="identity changed"):
        gears.reuse_ranking_metadata(
            target,
            np.asarray([0, 2]),
            {**runtime, "gears_ranking_cache": {**cache, "sha256": "bad"}},
        )
    with pytest.raises(ValueError, match="excluded"):
        gears.reuse_ranking_metadata(target, np.asarray([0]), runtime)


def test_coexpression_and_internal_metric_statistical_axes_are_excluded():
    prepared = _prepared()
    seen = []

    def network(**kwargs):
        seen.append(tuple(kwargs["adata"].var_names))
        return "official_edges"

    def metrics(results):
        assert results["pred"].shape == (1, 2)
        assert results["truth"].tolist() == [[1, 3]]
        assert results["pred_de"].shape == (1, 2)
        return "official_metrics"

    module = SimpleNamespace(get_similarity_network=network, compute_metrics=metrics)
    with allowed_training_statistics(module, np.asarray([0, 2])):
        assert (
            module.get_similarity_network(network_type="co-express", adata=prepared.adata)
            == "official_edges"
        )
        assert (
            module.get_similarity_network(network_type="go", adata=prepared.adata)
            == "official_edges"
        )
        assert (
            module.compute_metrics(
                {
                    "pred": np.asarray([[1, 99, 3]]),
                    "truth": np.asarray([[1, 77, 3]]),
                    "pred_de": np.zeros((1, 2)),
                }
            )
            == "official_metrics"
        )
    assert seen == [("A", "B"), ("A", "heldout", "B")]
    assert module.get_similarity_network is network
    assert module.compute_metrics is metrics


def test_inference_uses_final_model_and_exact_controls(monkeypatch):
    torch = pytest.importorskip("torch")
    from benchmarks.expression_holdout.contract import FittedAdapter

    module = ModuleType("gears.utils")
    module.create_cell_graph_for_prediction = lambda row, indices, targets: torch.tensor(row)
    loader_module = ModuleType("torch_geometric.loader")

    def loader(graphs, batch_size, shuffle):
        assert not shuffle
        for offset in range(0, len(graphs), batch_size):
            yield torch.stack(graphs[offset : offset + batch_size])

    loader_module.DataLoader = loader
    monkeypatch.setitem(sys.modules, "gears.utils", module)
    monkeypatch.setitem(sys.modules, "torch_geometric.loader", loader_module)

    class Network(torch.nn.Module):
        def forward(self, controls):
            return controls + 2

    model = SimpleNamespace(
        model=Network(),
        best_model=None,
        device="cpu",
        gene_list=["A", "heldout", "B"],
        pert_list=["A", "B"],
    )
    controls = np.arange(21, dtype=np.float32).reshape(7, 3)
    result = predict(
        FittedAdapter(model, {}),
        controls,
        ("A",),
        tuple(model.gene_list),
        cell_batch=3,
        query_count=3,
        seed=1,
    )
    np.testing.assert_array_equal(result, controls + 2)


def test_asset_source_hashes_all_pass_before_any_copy(tmp_path):
    from gradpert.hashing import sha256_file

    assets = tmp_path / "assets"
    copies = tmp_path / "copies"
    assets.mkdir()
    copies.mkdir()
    names = ("gene2go_all.pkl", "essential_all_data_pert_genes.pkl")
    for i, name in enumerate(names):
        (assets / name).write_bytes(f"synthetic graph asset {i}".encode())
    hashes = {name: sha256_file(assets / name) for name in names}
    go_csv = assets / "go_essential_all/go_essential_all.csv"
    go_csv.parent.mkdir()
    go_csv.write_text("source,target,importance\nA,A,1\n")
    runtime = {
        "gears_assets": str(assets),
        "gears_asset_sha256": hashes,
        "gears_go_csv_sha256": sha256_file(go_csv),
    }
    (assets / names[1]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        copy_official_assets(runtime, copies)
    assert list(copies.iterdir()) == []
    hashes[names[1]] = sha256_file(assets / names[1])
    observed = copy_official_assets(runtime, copies)
    assert observed == {**hashes, "go_essential_all/go_essential_all.csv": sha256_file(go_csv)}
    assert {name: sha256_file(copies / name) for name in names} == hashes
    assert sha256_file(copies / "go_essential_all/go_essential_all.csv") == sha256_file(go_csv)
    with pytest.raises(FileExistsError, match="overwrite"):
        copy_official_assets(runtime, copies)


def test_official_control_metadata_uses_integer_flags_without_touching_expression():
    prepared = _prepared()
    adata = prepared.adata.copy()
    before = adata.X.copy()
    before_ids = tuple(adata.obs_names)
    assert adata.obs.control.dtype == bool
    assert official_control_metadata(adata) is adata
    assert adata.obs.control.dtype == np.int64
    assert adata.obs.control.tolist() == [0, 0, 1, 1]
    assert tuple(adata.obs_names) == before_ids
    np.testing.assert_array_equal(adata.X.toarray(), before.toarray())
