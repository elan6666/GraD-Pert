import numpy as np
from scipy import sparse

from scripts.analysis.six_dataset_landscape import (
    _aggregate_expression,
    _batch_information,
    _group_codes,
    _retrieval,
)


class _TinyData:
    def __init__(self) -> None:
        self.X = sparse.csr_matrix([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [7.0, 6.0]])
        self.n_obs = 4


def test_sparse_streaming_aggregation_keeps_conditions_separate() -> None:
    lines = np.array(["K562"] * 4)
    conditions = np.array(["ctrl", "ctrl", "A+ctrl", "A+ctrl"])
    batches = np.array(["b1"] * 4)
    controls = np.array([True, True, False, False])
    keys, codes = _group_codes(lines, conditions, batches, controls, np.ones(4, bool), 42)
    sums, counts = _aggregate_expression(_TinyData(), codes, 2 * len(keys), 2, 2)
    merged = sums[0::2] + sums[1::2]
    assert counts.sum() == 4
    assert np.array_equal(merged[keys.index("K562\x1fA+ctrl")], [12.0, 10.0])
    assert np.array_equal(merged[keys.index("K562\x1fctrl\x1fb1")], [4.0, 2.0])


def test_batch_information_detects_perfect_condition_batch_confounding() -> None:
    associated = _batch_information(np.array(["A", "A", "B", "B"]), np.array(["1", "1", "2", "2"]))
    balanced = _batch_information(np.array(["A", "A", "B", "B"]), np.array(["1", "2", "1", "2"]))
    assert associated["batch_information_fraction"] == 1.0
    assert balanced["batch_information_fraction"] == 0.0


def test_retrieval_uses_matching_half_as_truth() -> None:
    vectors = {
        "A": (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 30),
        "B": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), 30),
    }
    result = _retrieval(vectors, 42)
    assert result["top1"] == 1.0
    assert result["random_top1"] == 0.5
