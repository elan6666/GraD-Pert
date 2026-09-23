import numpy as np
import pytest

from benchmarks.crosscell.prepare_fixed_axis import CELL_LINES, build_folds


def test_all_target_perturbations_are_test_only_and_target_controls_are_available() -> None:
    lines = []
    conditions = []
    controls = []
    for line in (*CELL_LINES, "K562_adamson"):
        lines.extend([line] * 22)
        conditions.extend(["ctrl"] * 2 + ["A+ctrl"] * 10 + ["B+ctrl"] * 10)
        controls.extend([1] * 2 + [0] * 20)
    line_array = np.array(lines)
    condition_array = np.array(conditions)
    control_array = np.array(controls)

    folds = build_folds(line_array, condition_array, control_array)

    assert set(folds) == set(CELL_LINES)
    for target, fold in folds.items():
        assert len(fold["test"]) == 20
        assert len(fold["target_control"]) == 2
        assert set(line_array[fold["test"]]) == {target}
        assert set(line_array[fold["train"]]) == set(CELL_LINES) - {target}
        assert set(line_array[fold["val"]]) == set(CELL_LINES) - {target}
        assert len(fold["train"]) == 54
        assert len(fold["val"]) == 6
        assert not any("K562_adamson" in line_array[rows] for rows in fold.values())
        assert set(condition_array[fold["test"]]) == {"A+ctrl", "B+ctrl"}


def test_control_flag_must_agree_with_condition() -> None:
    lines = np.array(["K562", "RPE1", "jurkat", "hepg2", "K562_adamson"])
    with pytest.raises(ValueError, match="control flag"):
        build_folds(lines, np.array(["ctrl"] * 5), np.array([1, 1, 1, 1, 0]))
