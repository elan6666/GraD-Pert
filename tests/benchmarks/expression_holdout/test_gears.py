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
