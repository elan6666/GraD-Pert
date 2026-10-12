import numpy as np
import pytest

from benchmarks.expression_holdout.data import build_prepared


def toy(**overrides):
    args = dict(
        matrix=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        allowed_indices=(0, 2),
        gene_ids=("A", "T", "B"),
        graph_gene_ids=("A", "T", "B", "Z"),
        row_ids=("c0", "a", "c1", "v"),
        conditions=("ctrl", "A+ctrl", "ctrl", "B+ctrl"),
        contexts=("J::0", "J::0", "J::1", "J::1"),
        train_row_ids=("a",),
        val_row_ids=("v",),
        control_row_ids=("c0", "c1"),
        train_conditions=("A+ctrl",),
        val_conditions=("B+ctrl",),
    )
    return build_prepared(**(args | overrides))


def test_removes_excluded_values_without_removing_gene_identity():
    p = toy()
    np.testing.assert_array_equal(p.adata.X.toarray(), [[1, 0, 2], [3, 0, 4], [5, 0, 6], [7, 0, 8]])
    assert tuple(p.adata.var_names) == ("A", "T", "B")
    assert p.excluded_indices == (1,)
    assert p.adata.raw is None and not p.adata.layers and not p.adata.uns
    assert p.train_control_indices == {"a": (0,)}
    assert p.val_control_indices == {"v": (2,)}


def test_unmatched_context_and_rogue_row_rejected():
    with pytest.raises(ValueError, match="context-compatible"):
        toy(contexts=("J::0", "J::9", "J::1", "J::1"))
    with pytest.raises(ValueError, match="partition"):
        toy(train_row_ids=("a", "v"))
    with pytest.raises(ValueError, match="condition split"):
        toy(conditions=("ctrl", "T+ctrl", "ctrl", "B+ctrl"))


def test_heldout_identity_or_nan_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="axis"):
        toy(allowed_indices=(2, 0))
    with pytest.raises(ValueError, match="nonfinite"):
        toy(matrix=np.array([[1, np.nan], [3, 4], [5, 6], [7, 8]]))
