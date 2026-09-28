from types import SimpleNamespace

import numpy as np

from gradpert.evaluation.state import (
    EvaluationStateLayout,
    _ranked_de_indices,
    _systema_reference_conditions,
)


def test_v2_deg_keeps_ranked_target_and_v1_state_is_unchanged(tmp_path) -> None:
    genes = {"TARGET": 0, "OTHER": 1, "DROPOUT": 2}
    ranked = ["TARGET", "DROPOUT", "OTHER"]
    valid = np.array([True, True, False])
    common = dict(condition="TARGET+ctrl", control_id="ctrl")
    assert _ranked_de_indices(ranked, genes, valid, evaluation_protocol="v1", **common) == [1]
    assert _ranked_de_indices(ranked, genes, valid, evaluation_protocol="v2", **common) == [0, 1]
    assert EvaluationStateLayout(tmp_path, "data", "protocol").root.name == "evaluation"
    assert (
        EvaluationStateLayout(tmp_path, "data", "protocol", evaluation_protocol="v2").root.name
        == "evaluation_v2"
    )


def test_systema_reference_population_is_unchanged() -> None:
    split = SimpleNamespace(train_conditions=["train-a"], val_conditions=["val-b"])
    assert _systema_reference_conditions(split, validation_only=True) == ["train-a"]
    assert _systema_reference_conditions(split, validation_only=False) == ["train-a", "val-b"]
