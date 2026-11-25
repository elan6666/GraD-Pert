import numpy as np
import pytest

from benchmarks.common.full_gene_evaluation import coverage, fill_missing_predictions
from gradpert.data._io import read_json
from gradpert.hashing import sha256_file


def test_fill_uses_each_conditions_fixed_controls_and_preserves_native_predictions():
    controls = np.column_stack((np.arange(300), np.arange(300) - 25, np.arange(300) + 5))
    prediction = np.full((300, 2), [42.0, 61.0], dtype=np.float32)

    result = fill_missing_predictions(prediction, controls, [0, 2])

    np.testing.assert_array_equal(result[:, [0, 2]], prediction)
    np.testing.assert_array_equal(result[:, 1], 124.5)
    shifted = fill_missing_predictions(prediction, controls + np.array([0, 5, 0]), [0, 2])
    np.testing.assert_array_equal(shifted[:, 1], 129.5)


def test_coverage_receipt_binds_missing_columns_to_full_gene_order(tmp_path):
    record = coverage(tmp_path, ["A", "B", "C"], [0, 2], 3)
    path = tmp_path / "evaluation_genes.json"
    assert record["evaluation_genes_sha256"] == sha256_file(path)
    assert record["native_prediction_genes"] == 2
    assert record["imputed_prediction_genes"] == 1
    assert read_json(path)["genes_in_expression_order"] == ["A", "B", "C"]
    assert record["imputed_genes_in_expression_order"] == ["B"]
    with pytest.raises(ValueError, match="mask changed"):
        coverage(tmp_path, ["A", "B", "C"], [0, 1], 3)
