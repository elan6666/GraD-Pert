"""Synthetic-only contracts for the isolated official Scouter adapter."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.scouter.official_api import independent_best_snapshot, predict_exact_controls
from gradpert.config import load_experiment_config


def test_selected_snapshot_does_not_alias_future_parameters():
    torch = pytest.importorskip("torch")
    layer = torch.nn.Linear(2, 1)
    original = layer.weight.detach().clone()
    with independent_best_snapshot(layer):
        best = layer.state_dict()
        with torch.no_grad():
            layer.weight.add_(7)
        assert torch.equal(best["weight"], original)
        assert not torch.equal(layer.weight, best["weight"])
        layer.load_state_dict(best)
        assert torch.equal(layer.weight, original)
    assert "state_dict" not in layer.__dict__


def test_snapshot_scope_restores_on_error():
    torch = pytest.importorskip("torch")
    layer = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError), independent_best_snapshot(layer):
        raise RuntimeError("failure")
    assert "state_dict" not in layer.__dict__


def test_prediction_preserves_all_300_rows_in_order():
    torch = pytest.importorskip("torch")

    class Network(torch.nn.Module):
        def forward(self, idx, x):
            assert idx.tolist() == [[1, 0]] * len(x)
            return x + 2

    model = SimpleNamespace(network=Network(), device="cpu", embd_idx_dict={"ctrl": 0, "A": 1})
    controls = np.arange(600, dtype=np.float32).reshape(300, 2)
    actual = predict_exact_controls(model, torch, "A+ctrl", controls, 256)
    np.testing.assert_array_equal(actual, controls + 2)
    with pytest.raises(ValueError, match="300"):
        predict_exact_controls(model, torch, "A+ctrl", controls[:299], 256)


def test_external_full_scouter_profile_is_explicit():
    config = load_experiment_config("configs/external-full/scouter_genept_seed/nadig_jurkat.yaml")
    assert config.training.max_epochs.value == 100
    assert config.training.early_stopping_patience.value == 10
    assert config.training.train_batch_size.value == 256
    assert config.training.learning_rate.value == 0.001
    assert config.training.min_delta == 0.001
    assert config.source_code.commit == "0cfddd000e19b72ff033ba67c8315f7bc3304932"
    assert config.source_code.execution == "isolated"


def test_frozen_official_scouter_synthetic_fit(tmp_path):
    checkout = os.environ.get("SCOUTER_OFFICIAL_CHECKOUT")
    if not checkout:
        pytest.skip("frozen official Scouter checkout not configured")
    pytest.importorskip("torch")
    import anndata
    import pandas as pd

    from benchmarks.common import AdaptedCanonicalData, official_module_session
    from benchmarks.scouter.official_api import fit_official, prepare_data

    adata = anndata.AnnData(
        np.random.default_rng(1).random((24, 5)).astype(np.float32),
        obs=pd.DataFrame(
            {"condition": ["ctrl"] * 8 + ["A+ctrl"] * 8 + ["B+ctrl"] * 8},
            index=[str(x) for x in range(24)],
        ),
        var=pd.DataFrame({"gene_name": list("ABCDE")}, index=list("ABCDE")),
    )
    adapted = AdaptedCanonicalData(adata, ("A+ctrl",), ("B+ctrl",), 5, {})
    prior = SimpleNamespace(
        embedding_width=3, values=np.ones((3, 3), dtype=np.float32), gene_ids=("A", "B", "C")
    )
    config = load_experiment_config("configs/external-full/scouter_genept_seed/nadig_jurkat.yaml")
    p = {k: SimpleNamespace(value=v.value) for k, v in config.model.parameters.items()}
    p.update(
        encoder=SimpleNamespace(value="8,4"),
        generator=SimpleNamespace(value="8"),
        bottleneck=SimpleNamespace(value=2),
    )
    tiny = SimpleNamespace(
        model=SimpleNamespace(parameters=p),
        training=SimpleNamespace(
            train_batch_size=SimpleNamespace(value=4),
            learning_rate=SimpleNamespace(value=0.001),
            early_stopping_patience=SimpleNamespace(value=10),
        ),
    )
    with official_module_session(
        checkout_root=Path(checkout),
        expected_commit=config.source_code.commit,
        module_names=("scouter",),
    ) as (modules, _):
        data = prepare_data(modules["scouter"], adapted, prior)
        assert "C" in data.embd.index  # held-out target embedding must survive setup
        assert data.test_adata.n_obs == 0
        model = modules["scouter"].Scouter(data, device="cpu")
        receipt = fit_official(model, tiny, epochs=1, progress_path=tmp_path / "progress.json")
        assert receipt["epochs_completed"] == 1
        assert receipt["training_forwards"] == 2
