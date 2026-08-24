from __future__ import annotations

import numpy as np
from scipy import sparse

from benchmarks.gears.runner import _ensure_official_sparse_expression


class _Adata:
    def __init__(self, expression: object) -> None:
        self.X = expression


def test_gears_adapter_converts_dense_expression_to_csr_without_value_drift() -> None:
    values = np.asarray([[0.0, 1.5], [2.0, 0.0]], dtype=np.float32)
    adata = _Adata(values.copy())

    receipt = _ensure_official_sparse_expression(adata)

    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_array_equal(adata.X.toarray(), values)
    assert receipt == {
        "input_expression_storage": "dense",
        "official_expression_storage": "scipy_csr_matrix",
    }


def test_gears_adapter_normalizes_sparse_expression_to_csr() -> None:
    values = np.asarray([[0.0, 3.0], [4.0, 0.0]], dtype=np.float32)
    adata = _Adata(sparse.csc_matrix(values))

    receipt = _ensure_official_sparse_expression(adata)

    assert sparse.isspmatrix_csr(adata.X)
    np.testing.assert_array_equal(adata.X.toarray(), values)
    assert receipt["input_expression_storage"] == "sparse"
