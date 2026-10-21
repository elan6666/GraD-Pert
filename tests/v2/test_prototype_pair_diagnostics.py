"""Synthetic identity/aggregation contracts for frozen prototype diagnostics."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).parents[2] / "scripts/v2/prototype_pair_diagnostics.py"
spec = importlib.util.spec_from_file_location("prototype_pair_diagnostics", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records():
    return [
        dict(condition_id=p, control_row_id=c, context_id="batch1", cell_line="Jurkat")
        for p, c in (("A", "x"), ("A", "y"), ("B", "x"), ("B", "z"))
    ]


def bank(values, rows=None, unit="control_condition"):
    obj = module.PrototypePairBank()
    rows = rows or records()
    for role, views in (
        ("teacher", ("global0", "global1")),
        ("student", ("global0", "global1", "local0", "local1")),
    ):
        for view in views:
            obj.add("ssl2.cls", role, view, rows, values, unit=unit)
    return obj


def select(result, stratum):
    return next(r for r in result["pairs"] if r["stratum"] == stratum)


def test_uniform_ce_entropy_kl_and_six_allowed_cross_view_pairs():
    result = bank(np.zeros((4, 8))).result()
    assert len(result["pairs"]) == 6 * 4
    assert all(r["teacher_view"] != r["student_view"] for r in result["pairs"])
    for row in result["pairs"]:
        assert row["ce_pair_mean"] == pytest.approx(math.log(8))
        assert row["teacher_entropy_pair_mean"] == pytest.approx(math.log(8))
        assert row["kl_pair_mean"] == pytest.approx(0, abs=1e-12)
        if row["stratum"] != "same_sample":
            assert row["wrong_minus_correct_ce_pair_mean"] == pytest.approx(0, abs=1e-12)
    json.dumps(result, allow_nan=False)


def test_sample_specific_outputs_have_positive_wrong_pair_gap_and_decomposition():
    result = bank(np.eye(4) * 20).result()
    for row in result["pairs"]:
        for reduction in ("pair_mean", "condition_balanced_mean"):
            assert row["ce_" + reduction] == pytest.approx(
                row["teacher_entropy_" + reduction] + row["kl_" + reduction], abs=1e-12
            )
        if row["stratum"] != "same_sample":
            assert row["wrong_minus_correct_ce_pair_mean"] > 19
    assert select(result, "same_sample")["pair_count"] == 4
    assert select(result, "same_condition_different_control")["pair_count"] == 4
    assert select(result, "different_condition_same_control")["pair_count"] == 2
    assert select(result, "different_condition_different_control")["pair_count"] == 6


def test_repeated_controls_deduplicated_first_realization():
    rows = records()
    values = np.eye(4) * 10
    result = bank(np.concatenate([values, [[99, 0, 0, 0]]]), [*rows, rows[0]]).result()
    reference = bank(values).result()
    for actual, expected in zip(result["pairs"], reference["pairs"], strict=True):
        assert actual["duplicate_rows_ignored"] == 6
        for field in ("pair_count", "independent_samples", "ce_pair_mean", "kl_pair_mean"):
            assert actual[field] == pytest.approx(expected[field])


def test_context_isolation_and_missing_pairs_are_null():
    rows = records()[:2]
    rows[1] = {**rows[1], "context_id": "batch2"}
    result = bank(np.eye(2) * 20, rows).result()
    assert {r["context_id"] for r in result["pairs"]} == {"batch1", "batch2"}
    for row in result["pairs"]:
        if row["stratum"] != "same_sample":
            assert row["pair_count"] == 0
            assert row["ce_pair_mean"] is None
            assert row["wrong_minus_correct_ce_pair_mean"] is None
    json.dumps(result, allow_nan=False)


def test_static_ssl1_deduplicates_control_and_has_only_condition_strata():
    result = bank(np.eye(4) * 10, unit="perturbation_condition").result()
    assert {r["stratum"] for r in result["pairs"]} == {"same_sample", "different_condition"}
    assert all(r["independent_samples"] == 2 for r in result["pairs"])
    assert all(r["independent_controls"] is None for r in result["pairs"])
    assert select(result, "different_condition")["pair_count"] == 2


def test_condition_balancing_matches_brute_force_directed_pairs():
    rng = np.random.default_rng(7)
    rows = [
        *records(),
        dict(condition_id="C", control_row_id="k", context_id="batch1", cell_line="Jurkat"),
    ]
    teacher, student = rng.normal(size=(5, 7)), rng.normal(size=(5, 7))
    obj = module.PrototypePairBank()
    obj.add("ssl2", "teacher", "global0", rows, teacher)
    obj.add("ssl2", "student", "global1", rows, student)
    actual = select(obj.result(), "different_condition_different_control")
    qt = np.exp(module._log_softmax(teacher))
    ls = module._log_softmax(student)
    groups, vals, correct = {}, [], []
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if (
                a["condition_id"] != b["condition_id"]
                and a["control_row_id"] != b["control_row_id"]
            ):
                ce = -qt[i] @ ls[j]
                vals.append(ce)
                correct.append(-qt[i] @ ls[i])
                groups.setdefault((a["condition_id"], b["condition_id"]), []).append(ce)
    assert actual["ce_pair_mean"] == pytest.approx(np.mean(vals))
    assert actual["ce_condition_balanced_mean"] == pytest.approx(
        np.mean([np.mean(v) for v in groups.values()])
    )
    assert actual["wrong_minus_correct_ce_pair_mean"] == pytest.approx(
        np.mean(vals) - np.mean(correct)
    )
    assert actual["valid_condition_pairs"] == 6


def test_extreme_logits_stable_and_reject_invalid_identity_or_teacher_local():
    obj = bank(np.array([[10000, -10000]] * 4))
    assert select(obj.result(), "same_sample")["ce_pair_mean"] == pytest.approx(0)
    with pytest.raises(ValueError, match="Global"):
        obj.add("ssl2", "teacher", "local0", records(), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="identities"):
        obj.add("ssl2", "student", "global0", [{"condition_id": "A"}], np.zeros((1, 2)))
    with pytest.raises(ValueError, match="width"):
        obj.add("ssl2.cls", "student", "global0", records(), np.zeros((4, 3)))


def test_add_is_streaming_and_input_arrays_are_not_modified():
    values = np.arange(20, dtype=np.float64).reshape(4, 5)
    saved = values.copy()
    full = bank(values).result()
    streamed = module.PrototypePairBank()
    for role, views in (
        ("teacher", ("global0", "global1")),
        ("student", ("global0", "global1", "local0", "local1")),
    ):
        for view in views:
            for start in (0, 2):
                streamed.add(
                    "ssl2.cls", role, view, records()[start : start + 2], values[start : start + 2]
                )
    assert streamed.result() == full
    np.testing.assert_array_equal(saved, values)
