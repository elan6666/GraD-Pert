import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from benchmarks.crosscell.study_data import score
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json


def test_c2_scoring_projects_all_metric_populations(tmp_path):
    genes = ["A", "omitted", "B", "C"]
    panel = ["A", "B", "C"]
    source = tmp_path / "miniature.h5ad"
    observations = ["fit+ctrl", "fit+ctrl", "ctrl"] + ["ctrl"] * 300 + ["held+ctrl"]
    data = np.array([[2, 40, 3, 4]] * 3 + [[1, 50, 2, 3]] * 300 + [[3, 80, 6, 2]])
    anndata.AnnData(
        X=csr_matrix(data),
        obs=pd.DataFrame(
            {"condition": observations, "batch": ["b1"] * len(data)},
            index=list(map(str, range(len(data)))),
        ),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(source)
    rows = {
        "train": np.array([0]),
        "val": np.array([1]),
        "source_control": np.array([2]),
        "target_control": np.arange(3, 303),
        "test": np.array([303]),
    }
    draws = {"held+ctrl": {"input_control_rows": list(range(302, 2, -1)), "truth_rows": [303]}}
    draws_path = tmp_path / "draws.json"
    atomic_json(draws_path, {"draws": draws})
    cohort = {
        "condition_ids": list(draws),
        "condition_ids_sha256": sha256_json(list(draws)),
        "per_condition_pairing_sha256": sha256_json(draws),
        "shared_frozen_draws_path": str(draws_path),
        "fit_row_sha256": {
            key: sha256_json(rows[key].tolist()) for key in ("train", "val", "source_control")
        },
    }
    cross = {
        "shared_frozen_draws_sha256": sha256_file(draws_path),
        "target_cell_line": "K562",
        "expression_panel_genes": len(genes),
    }

    class Metrics:
        def __init__(self):
            self.PERT_MEANS = {"K562": "previous gene panel"}

        def cache_perturbation_means(self, reference, line):
            assert list(reference.var_names) == panel
            np.testing.assert_array_equal(reference.X[-1].toarray(), [[3, 6, 2]])
            assert line not in self.PERT_MEANS
            self.PERT_MEANS[line] = draws

        @staticmethod
        def RetrievalMetric(*, subset):
            assert not subset

            def retrieval(prediction, truth, line, controls, reference, condition):
                assert condition == "held+ctrl"
                assert line == "K562"
                assert list(reference.var_names) == panel
                np.testing.assert_array_equal(prediction[0], [3, 6, 2])
                np.testing.assert_array_equal(truth[0], [3, 6, 2])
                np.testing.assert_array_equal(controls[0], [1, 2, 3])
                return 1.0

            return retrieval

    def full_predictor(condition, controls, control_rows):
        assert condition == "held+ctrl"
        assert controls.shape == (300, 4)
        np.testing.assert_array_equal(control_rows, draws[condition]["input_control_rows"])
        return np.tile([3, 100_000, 6, 2], (300, 1))

    results = []
    metrics = Metrics()
    for projected in (False, True):
        output = tmp_path / str(projected)
        output.mkdir()

        def predict(*args, projected=projected):
            prediction = full_predictor(*args)
            return prediction[:, [0, 2, 3]] if projected else prediction

        results.append(
            score(
                predict,
                metrics,
                source,
                cross,
                cohort,
                rows,
                None,
                output,
                journal_key="checkpoint-identity",
                include_control_row_ids=True,
                score_genes=panel,
                prediction_on_score_panel=projected,
            )
        )
    assert metrics.PERT_MEANS == {"K562": "previous gene panel"}
    assert results[0] == results[1]
    assert results[0]["mean_mse"] == 0.0
    assert results[0]["retrieval_mean"] == 1.0
    assert results[0]["test_panel_genes"] == 3
    assert results[0]["test_panel_gene_order_sha256"] == sha256_json(panel)
    assert len(results[0]["pearson"]) == 3
    with pytest.raises(ValueError, match="ordered subsequence"):
        score(
            full_predictor,
            Metrics(),
            source,
            cross,
            cohort,
            rows,
            None,
            tmp_path,
            score_genes=list(reversed(panel)),
        )
