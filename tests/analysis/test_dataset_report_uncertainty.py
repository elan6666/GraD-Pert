import numpy as np
import pytest

from scripts.analysis.dataset_report_uncertainty import fdr_bh, retrieval_null, split_summary


def test_joint_retrieval_null_does_not_inflate_identical_repeats():
    winners = np.arange(8)[None, :]
    one = retrieval_null(winners, permutations=199, seed=42)
    twenty = retrieval_null(np.repeat(winners, 20, axis=0), permutations=199, seed=42)
    assert one == twenty


def test_fdr_monotone_and_original_order():
    assert np.allclose(fdr_bh([0.04, 0.001, 0.03]), [0.04, 0.003, 0.04])


def test_reject_incomplete_splits_and_preserve_values():
    rows = [
        dict(condition="A", repeat=i, truth_cells=10, pearson_delta=v, pearson_expression=0.99)
        for i, v in enumerate([0.2, 0.8])
    ]
    result = split_summary(rows, repeats=2)
    assert result["conditions"][0]["values"] == [0.2, 0.8]
    assert result["conditions"][0]["median"] == 0.5
    with pytest.raises(ValueError, match="missing/duplicate"):
        split_summary(rows[:1], repeats=2)
