"""Loss-axis, fixed contextual pairing and official-loop stop invariants."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.expression_holdout.data import build_prepared
from benchmarks.expression_holdout.scouter import (
    _UpdateLimitReached,
    bounded_official_updates,
    contextual_dataset_factory,
    load_official_prior,
    masked_official_loss,
    official_nonzero_masks,
    predict,
)


def _prepared():
    return build_prepared(
        matrix=np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
        allowed_indices=(0, 2),
        gene_ids=("A", "heldout", "B"),
        graph_gene_ids=("A", "heldout", "B"),
        row_ids=("t", "v", "c0", "c1"),
        conditions=("A+ctrl", "B+ctrl", "ctrl", "ctrl"),
        contexts=("cell::x", "cell::y", "cell::x", "cell::y"),
        train_row_ids=("t",),
        val_row_ids=("v",),
        control_row_ids=("c0", "c1"),
        train_conditions=("A+ctrl",),
        val_conditions=("B+ctrl",),
    )


def test_loss_uses_original_rule_only_on_allowed_nonzero_coordinates():
    torch = pytest.importorskip("torch")
    called = []

    def official(pred, truth, ctrl, groups, nonzero_idx_dict, gamma, lambda_):
        called.append((pred.shape, nonzero_idx_dict, gamma, lambda_))
        assert torch.equal(truth, torch.tensor([[1.0, 3.0]]))
        return (
            (pred[:, nonzero_idx_dict[groups[0]]] - truth[:, nonzero_idx_dict[groups[0]]])
            .square()
            .mean()
        )

    pred = torch.tensor([[0.0, 999.0, 0.0]], requires_grad=True)
    truth = torch.tensor([[1.0, 444.0, 3.0]])
    loss = masked_official_loss(official, np.asarray([0, 2]))(
        pred,
        truth,
        truth,
        ["A+ctrl"],
        {"A+ctrl": [0, 1, 2]},
        0.5,
        0.3,
    )
    loss.backward()
    assert loss.item() == 5
    assert pred.grad.tolist() == [[-1, 0, -3]]
    assert called == [(torch.Size([1, 2]), {"A+ctrl": [0, 1]}, 0.5, 0.3)]
    with pytest.raises(ValueError, match="no allowed"):
        masked_official_loss(official, np.asarray([0, 2]))(
            pred,
            truth,
            truth,
            ["A+ctrl"],
            {"A+ctrl": [1]},
            0,
            0,
        )


def test_context_adapter_retains_official_perturbation_order_and_restores_identity():
    prepared = _prepared()

    class Original:
        def __init__(self, adata, key_label, key_embd_index, shuffle=True, seed=24):
            mask = adata.obs[key_label] != "ctrl"
            self.bcode = list(adata.obs_names[mask])
            self.PertCellIndx = np.arange(len(self.bcode))[::-1]

    audit = []
    dataset = contextual_dataset_factory(Original, prepared, audit)(
        prepared.adata,
        "condition",
        "embd_index",
    )
    assert dataset.PertCellIndx.tolist() == [1, 0]
    assert dataset.CtrlCellIndx.tolist() == [1, 0]
    assert audit == [("v", "c1"), ("t", "c0")]
    invalid = SimpleNamespace(**{**vars(prepared), "train_control_indices": {"t": (3,)}})
    with pytest.raises(ValueError, match="canonical context"):
        contextual_dataset_factory(Original, invalid, [])(prepared.adata, "condition", "embd_index")


def test_diagnostic_stop_occurs_after_real_nonzero_lr_optimizer_step_and_restores_class():
    torch = pytest.importorskip("torch")
    module = SimpleNamespace(optim=SimpleNamespace(Adam=torch.optim.Adam))
    original = module.optim.Adam
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    counter = {"updates": 0}
    with bounded_official_updates(module, 3, counter):
        optimizer = module.optim.Adam([parameter], lr=0.01)
        with pytest.raises(_UpdateLimitReached):
            for _ in range(4):
                optimizer.zero_grad()
                parameter.square().sum().backward()
                optimizer.step()
    assert counter["updates"] == 3
    assert parameter.item() < 0.98
    assert int(optimizer.state[parameter]["step"]) == 3
    assert module.optim.Adam is original


def test_inference_keeps_exact_rows_and_reintroduced_excluded_expression():
    torch = pytest.importorskip("torch")
    from benchmarks.expression_holdout.contract import FittedAdapter

    class Network(torch.nn.Module):
        def forward(self, index, controls):
            assert index.tolist() == [[0, 2]] * len(controls)
            return controls + 1

    prepared = _prepared()
    model = SimpleNamespace(
        network=Network(),
        device="cpu",
        all_adata=prepared.adata,
        embd_idx_dict={"A": 0, "B": 1, "ctrl": 2},
    )
    controls = np.arange(15, dtype=np.float32).reshape(5, 3)
    result = predict(
        FittedAdapter(model, {}),
        controls,
        ("A",),
        prepared.gene_ids,
        cell_batch=2,
        query_count=3,
        seed=1,
    )
    np.testing.assert_array_equal(result, controls + 1)
    with pytest.raises(ValueError, match="identity/order"):
        predict(
            FittedAdapter(model, {}),
            controls,
            ("A",),
            prepared.gene_ids[::-1],
            cell_batch=2,
            query_count=3,
            seed=1,
        )


def test_prior_preflights_test_only_targets_and_requires_audited_aliases(tmp_path):
    import pandas as pd

    from gradpert.hashing import sha256_file, sha256_json

    path = tmp_path / "synthetic_prior.csv"
    pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=["A", "TEST_ALIAS", "unused"]).to_csv(path)
    runtime = {
        "prior_path": str(path),
        "prior_sha256": sha256_file(path),
        "required_target_genes": ["A", "TEST"],
    }
    with pytest.raises(ValueError, match="fit/test targets"):
        load_official_prior(runtime, {"A", "ctrl"})
    aliases = {"TEST": "TEST_ALIAS"}
    with pytest.raises(ValueError, match="alias digest"):
        load_official_prior({**runtime, "prior_aliases": aliases}, {"A"})
    vectors, receipt = load_official_prior(
        {**runtime, "prior_aliases": aliases, "prior_aliases_sha256": sha256_json(aliases)},
        {"A"},
    )
    assert list(vectors.index) == ["A", "TEST", "ctrl"]
    assert vectors.loc["TEST"].tolist() == [3, 4]
    assert vectors.loc["ctrl"].tolist() == [0, 0]
    assert receipt["required_target_gene_count"] == 2
    assert "unused" not in vectors.index


def test_official_dropout_statistics_only_receive_observed_columns():
    prepared = _prepared()

    class Data:
        adata = prepared.adata.copy()

        def get_dropout_non_zero_genes(self):
            assert tuple(self.adata.var_names) == ("A", "B")
            self.adata.uns["gene_idx_non_zeros"] = {"A+ctrl": np.asarray([0, 1])}
            self.adata.uns["gene_idx_non_dropout"] = {"A+ctrl": np.asarray([1])}

    data = Data()
    original = data.adata
    masks = official_nonzero_masks(data, np.asarray([0, 2]))
    assert masks["A+ctrl"].tolist() == [0, 2]
    assert data.adata.uns["gene_idx_non_dropout"]["A+ctrl"].tolist() == [2]
    assert data.adata is original
