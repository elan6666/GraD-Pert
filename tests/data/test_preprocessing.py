from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from gradpert.data import preprocessing as preprocessing_module
from gradpert.data.preprocessing import (
    _append_graph_only_targets,
    _require_raw_integer_counts,
    canonicalize_metadata,
    filter_cells_by_perturbation_effect,
    preprocess_norman,
    preprocess_raw_within_cell,
    preprocess_upstream_within_cell,
)
from gradpert.data.registry import load_dataset_registry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = ROOT / "registry" / "datasets"


class MiniAnnData:
    def __init__(self, x: np.ndarray[Any, Any], obs: pd.DataFrame, var: pd.DataFrame):
        self.X = x
        self.obs = obs
        self.var = var
        self.uns: dict[str, Any] = {}

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.X.shape[1])

    def copy(self) -> MiniAnnData:
        result = MiniAnnData(self.X.copy(), self.obs.copy(), self.var.copy())
        result.uns = deepcopy(self.uns)
        return result

    def __getitem__(self, key: tuple[Any, Any]) -> MiniAnnData:
        rows, columns = key
        row_positions = np.arange(self.n_obs)[rows]
        column_positions = np.arange(self.n_vars)[columns]
        row_positions = np.atleast_1d(row_positions)
        column_positions = np.atleast_1d(column_positions)
        return MiniAnnData(
            self.X[np.ix_(row_positions, column_positions)],
            self.obs.iloc[row_positions].copy(),
            self.var.iloc[column_positions].copy(),
        )


def _nadig_fixture() -> MiniAnnData:
    obs = pd.DataFrame(
        {
            "gene": ["non-targeting", "TP53", "TP53", "MISSING"],
            "gem_group": ["1", "1", "2", "2"],
        },
        index=["ctrl-0", "tp53-low", "tp53-high", "missing-0"],
    )
    var = pd.DataFrame({"gene_name": ["TP53", "OTHER"]}, index=["g0", "g1"])
    x = np.asarray([[10, 1], [0, 2], [15, 3], [4, 4]], dtype=float)
    return MiniAnnData(x, obs, var)


def test_nadig_metadata_is_canonicalized_from_observed_raw_columns() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "nadig_hepg2.yaml")
    result, report = canonicalize_metadata(_nadig_fixture(), entry)
    assert result.obs["condition"].tolist() == [
        "ctrl",
        "TP53+ctrl",
        "TP53+ctrl",
        "MISSING+ctrl",
    ]
    assert result.obs["control"].tolist() == [1, 0, 0, 0]
    assert set(result.obs["cell_type"]) == {"hepg2"}
    assert result.obs.loc["tp53-low", "condition_name"] == "hepg2_TP53+ctrl_1+1"
    assert result.var["gene_name"].tolist() == ["TP53", "OTHER"]
    assert report.n_conditions == 3
    assert report.n_controls == 1


def test_signal_filter_uses_raw_target_ids_before_condition_encoding() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "nadig_hepg2.yaml")
    result, report = filter_cells_by_perturbation_effect(_nadig_fixture(), entry)
    assert result.obs.index.tolist() == ["ctrl-0", "tp53-low", "missing-0"]
    tp53 = next(item for item in report.conditions if item.source_condition_id == "TP53")
    assert tp53.target_gene_present
    assert tp53.cells_before == 2
    assert tp53.cells_after == 1
    missing = next(item for item in report.conditions if item.source_condition_id == "MISSING")
    assert not missing.target_gene_present
    assert missing.cells_after == missing.cells_before == 1


def test_nadig_duplicate_symbol_policy_preserves_first_target_and_suffixes_later_row() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "nadig_hepg2.yaml")
    obs = pd.DataFrame(
        {"gene": ["non-targeting", "HSPA14"], "gem_group": ["1", "1"]},
        index=["ctrl-0", "hspa14-0"],
    )
    var = pd.DataFrame(
        {"gene_name": ["HSPA14", "HSPA14", "OTHER"]},
        index=["ENSG-FIRST", "ENSG-SECOND", "ENSG-OTHER"],
    )
    result, report = canonicalize_metadata(
        MiniAnnData(np.ones((2, 3), dtype=float), obs, var),
        entry,
    )
    assert result.var["gene_name"].tolist() == [
        "HSPA14",
        "HSPA14__ENSG-SECOND",
        "OTHER",
    ]
    assert report.source_gene_symbol_column.endswith("[suffix_later_with_var_index]")


def test_norman_aliases_and_incorrect_upstream_cell_label_are_explicitly_canonicalized() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "norman.yaml")
    obs = pd.DataFrame(
        {
            "condition": ["ctrl", "ctrl+CEBPE", "CEBPE+ctrl", "KLF1+MAP2K6"],
            "cell_type": ["A549"] * 4,
        },
        index=["ctrl-0", "cebpe-a", "cebpe-b", "combo-0"],
    )
    var = pd.DataFrame({"gene_name": ["CEBPE", "KLF1", "MAP2K6"]})
    result, report = canonicalize_metadata(
        MiniAnnData(np.ones((4, 3), dtype=float), obs, var),
        entry,
    )
    assert result.obs["condition"].tolist() == [
        "ctrl",
        "CEBPE+ctrl",
        "CEBPE+ctrl",
        "KLF1+MAP2K6",
    ]
    assert set(result.obs["batch"]) == {"norman_upstream_single_batch"}
    assert set(result.obs["cell_type"]) == {"K562"}
    assert report.n_conditions == 3
    assert report.source_batch_column == "<constant:norman_upstream_single_batch>"


def test_norman_rejects_unexpected_upstream_cell_type_value() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "norman.yaml")
    obs = pd.DataFrame(
        {"condition": ["ctrl", "CEBPE+ctrl"], "cell_type": ["K562", "K562"]},
        index=["ctrl-0", "cebpe-0"],
    )
    var = pd.DataFrame({"gene_name": ["CEBPE"]})
    with pytest.raises(ValueError, match="frozen observed values"):
        canonicalize_metadata(MiniAnnData(np.ones((2, 1)), obs, var), entry)


def test_norman_preserves_verified_gears_expression_without_reprocessing() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "norman.yaml")
    genes = ["CEBPE", "KLF1", "MAP2K6", *[f"G{index}" for index in range(5042)]]
    obs = pd.DataFrame(
        {
            "condition": ["ctrl", "ctrl+CEBPE", "KLF1+MAP2K6"],
            "cell_type": ["A549"] * 3,
        },
        index=["ctrl-0", "cebpe-0", "combo-0"],
    )
    var = pd.DataFrame({"gene_name": genes}, index=[f"v{index}" for index in range(5045)])
    x = np.zeros((3, 5045), dtype=np.float32)
    x[:, :4] = np.asarray(
        [
            [0.0, 0.125, 1.75, 0.5],
            [0.25, 0.0, 2.125, 0.75],
            [1.5, 0.625, 0.0, 0.875],
        ],
        dtype=np.float32,
    )
    source = MiniAnnData(x, obs, var)
    source.uns["top_non_zero_de_20"] = {"stale": ["G0"]}

    result, report = preprocess_norman(source, entry)

    assert np.array_equal(result.X, x)
    assert result.var["gene_name"].tolist() == genes
    assert result.var["expression_output_gene"].tolist() == [True] * 5045
    assert "top_non_zero_de_20" not in result.uns
    assert report.input_expression_state == "verified_upstream_log1p"
    assert report.expression_scale_action == ("preserved_verified_gears_processed_log_expression")
    assert result.uns["gradpert_preprocessing"]["expression_scale_action"] == (
        "preserved_verified_gears_processed_log_expression"
    )


def test_raw_count_audit_rejects_previously_transformed_expression() -> None:
    _require_raw_integer_counts(
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        label="fixture",
    )
    with pytest.raises(ValueError, match="raw integer counts"):
        _require_raw_integer_counts(
            np.asarray([[0.0, np.log1p(2.0)]], dtype=np.float32),
            label="fixture",
        )


@pytest.mark.parametrize(
    ("dataset_id", "invalid_expression", "message"),
    [
        ("replogle_rpe1_essential", np.asarray([[0.0, -1.0]]), "nonnegative"),
        ("nadig_jurkat", np.asarray([[0.0, np.nan]]), "finite"),
        ("nadig_hepg2", sparse.csr_matrix([[0.0, np.log1p(2.0)]]), "raw integer counts"),
    ],
)
def test_raw_dataset_entrypoints_fail_before_filter_or_scanpy(
    monkeypatch,
    dataset_id: str,
    invalid_expression: Any,
    message: str,
) -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / f"{dataset_id}.yaml")
    source = MiniAnnData(
        invalid_expression,
        pd.DataFrame({"unused": ["row"]}),
        pd.DataFrame({"gene_name": ["G0", "G1"]}),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("raw audit must fail before preprocessing transforms")

    monkeypatch.setattr(preprocessing_module, "filter_cells_by_perturbation_effect", forbidden)
    monkeypatch.setattr(preprocessing_module, "_require_scanpy", forbidden)

    with pytest.raises(ValueError, match=message):
        preprocess_raw_within_cell(source, entry, known_candidate_targets=set())


def test_upstream_k562_appends_missing_targets_only_to_graph_axis() -> None:
    ad = pytest.importorskip("anndata")
    entry = load_dataset_registry(REGISTRY_ROOT / "replogle_k562_essential.yaml")
    gene_ids = [f"G{index}" for index in range(5000)]
    expression = np.arange(10_000, dtype=np.float32).reshape(2, 5000) / 10_000
    source = ad.AnnData(
        X=expression.copy(),
        obs=pd.DataFrame(
            {
                "condition": ["ctrl", "TARGET+ctrl"],
                "batch": ["1", "1"],
                "cell_line": ["K562", "K562"],
            },
            index=["ctrl-0", "target-0"],
        ),
        var=pd.DataFrame(index=gene_ids),
    )
    result, report = preprocess_upstream_within_cell(source, entry)
    assert result.shape == (2, 5001)
    assert result.var["gene_name"].tolist()[:5000] == gene_ids
    assert result.var["gene_name"].tolist()[-1] == "TARGET"
    assert result.var["expression_output_gene"].tolist() == [True] * 5000 + [False]
    assert result.var["forced_candidate_target"].tolist() == [False] * 5000 + [True]
    assert result.X[:, :5000].dtype == expression.dtype
    assert np.array_equal(np.asarray(result.X[:, :5000]), expression)
    appended = result.X[:, -1]
    if hasattr(appended, "toarray"):
        appended = appended.toarray()
    assert np.asarray(appended).reshape(-1).tolist() == [0.0, 0.0]
    assert report.n_expression_genes == 5000
    assert report.n_graph_genes == 5001
    assert report.forced_candidate_targets == ("TARGET",)
    assert report.missing_candidate_targets == ()


def test_upstream_k562_preserves_sparse_expression_values_dtype_and_order() -> None:
    ad = pytest.importorskip("anndata")
    entry = load_dataset_registry(REGISTRY_ROOT / "replogle_k562_essential.yaml")
    gene_ids = [f"G{index}" for index in range(5000)]
    dense = np.zeros((2, 5000), dtype=np.float32)
    dense[0, [0, 123, 4999]] = [0.125, 1.75, 3.5]
    dense[1, [2, 321, 4000]] = [0.25, 2.125, 4.5]
    source = ad.AnnData(
        X=sparse.csr_matrix(dense),
        obs=pd.DataFrame(
            {
                "condition": ["ctrl", "TARGET+ctrl"],
                "batch": ["1", "1"],
                "cell_line": ["K562", "K562"],
            },
            index=["ctrl-0", "target-0"],
        ),
        var=pd.DataFrame(index=gene_ids),
    )

    result, _ = preprocess_upstream_within_cell(source, entry)

    preserved = result.X[:, :5000]
    assert sparse.issparse(preserved)
    assert preserved.dtype == np.float32
    assert np.array_equal(preserved.toarray(), dense)
    assert result.var["gene_name"].tolist()[:5000] == gene_ids


def test_graph_only_append_preserves_existing_expression_and_forced_flags() -> None:
    ad = pytest.importorskip("anndata")
    entry = load_dataset_registry(REGISTRY_ROOT / "nadig_hepg2.yaml")
    source = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(
            {
                "gene_name": ["HVG", "PRESENT_TARGET"],
                "highly_variable": [True, False],
                "forced_candidate_target": [False, True],
                "expression_output_gene": [True, False],
            },
            index=["hvg", "present"],
        ),
    )
    result, appended = _append_graph_only_targets(
        source,
        entry,
        {"PRESENT_TARGET", "MISSING_TARGET"},
    )
    assert appended == ("MISSING_TARGET",)
    assert result.var["gene_name"].tolist() == [
        "HVG",
        "PRESENT_TARGET",
        "MISSING_TARGET",
    ]
    assert result.var["expression_output_gene"].tolist() == [True, False, False]
    assert result.var["forced_candidate_target"].tolist() == [False, True, True]
