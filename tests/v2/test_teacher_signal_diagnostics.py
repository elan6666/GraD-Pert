"""Synthetic behavioral checks for zero-update Teacher signal localization."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

scripts = Path(__file__).parents[2] / "scripts/v2"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location(
        "teacher_signal_diagnostics", scripts / "teacher_signal_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


class Head(nn.Module):
    def __init__(self, teacher=False):
        super().__init__()
        self.features = nn.Linear(2, 2, bias=False)
        self.prototypes = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.2 if teacher else 1.0]]))
            self.prototypes.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            )

    def forward_with_features(self, values):
        b = self.features(values)
        return b, self.prototypes(F.normalize(b.float(), dim=-1))

    def forward(self, values, *, return_features=False):
        b, logits = self.forward_with_features(values)
        return (b, logits) if return_features else logits


class HistoricalHead(Head):
    forward_with_features = None

    def forward(self, values, *, return_features=False):
        b = self.features(values)
        return self.prototypes(F.normalize(b.float(), dim=-1))


class Objective(nn.Module):
    def __init__(self, head=Head):
        super().__init__()
        self.student, self.teacher = nn.Module(), nn.Module()
        self.student.ssl2_cls, self.teacher.ssl2_cls = head(), head(True)
        self.lambda2, self.weights = 1.0, ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.teacher_temperature = 0.4
        self.register_buffer("ssl2_cls_center", torch.tensor([0.1, -0.1, 0.05, -0.05]))
        self.eval()


def records(context="batch1"):
    return [
        dict(condition_id=p, control_row_id=c, context_id=context, cell_line="Jurkat")
        for p, c in (("A", "x"), ("A", "y"), ("B", "x"), ("B", "y"))
    ]


def inputs():
    return torch.tensor([[1.0, 0.2], [1.0, 0.3], [0.2, 1.0], [0.3, 1.0]])


def native(obj, student=None, teacher=None, locals_=True):
    student = inputs() if student is None else student
    teacher = inputs() * torch.tensor([1.0, 0.3]) if teacher is None else teacher
    for _i in range(2):
        obj.student.ssl2_cls(student, return_features=True)
    for _i in range(2):
        obj.teacher.ssl2_cls(teacher)
    if locals_:
        obj.student.ssl2_cls(student)
        obj.student.ssl2_cls(student)


def captured(obj=None, rows=None):
    obj = Objective() if obj is None else obj
    bank = module.TeacherSignalBank()
    with bank.capture(obj, records() if rows is None else rows):
        native(obj)
    return bank


def test_all_four_paths_replay_and_state_are_exact_and_outputs_aggregate_only():
    obj = Objective()
    before = {k: v.clone() for k, v in obj.state_dict().items()}
    result = captured(obj).result()
    assert result["status"] == "completed"
    assert result["updates"] == result["encoder_recompute"] == 0
    assert {x["path"] for x in result["paths"]} == {"SS", "ST", "TS", "TT"}
    for receipt in result["captures"]:
        assert receipt["native_head_calls"] == {"student": 4, "teacher": 2}
        assert receipt["head_only_replays"] == 8
        assert all(p["bottleneck_maxabs"] == p["logits_maxabs"] == 0 for p in receipt["parity"])
    for name, value in obj.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not obj.student.ssl2_cls._forward_hooks
    assert not obj.teacher.ssl2_cls.features._forward_hooks
    assert {r["teacher_view"] for r in result["pairs"]} == {"global0", "global1"}
    assert {r["student_view"] for r in result["pairs"]} == {"global0", "global1"}
    json.dumps(result, allow_nan=False)


def test_paths_have_correct_encoder_head_semantics_and_pre_l2_is_not_unit():
    bank = captured()
    obj = Objective()
    values = inputs()
    for path, (encoder, head_role) in module._PATHS.items():
        actual_input = values if encoder == "student" else values * torch.tensor([1.0, 0.3])
        b, logits = getattr(obj, head_role).ssl2_cls.forward_with_features(actual_input)
        entry = next(x for x in bank._entries if x["path"] == path and x["view"] == "global0")
        np.testing.assert_array_equal(entry["values"]["input"], actual_input.numpy())
        np.testing.assert_array_equal(entry["values"]["bottleneck_pre_l2"], b.detach().numpy())
        np.testing.assert_allclose(np.linalg.norm(entry["values"]["bottleneck_unit"], axis=1), 1)
        np.testing.assert_array_equal(entry["values"]["raw_logits"], logits.detach().numpy())
    assert not np.allclose(
        bank._entries[0]["values"]["bottleneck_pre_l2"],
        bank._entries[0]["values"]["bottleneck_unit"],
    )


def test_rank_variance_and_entropy_js_match_direct_float64_formulas():
    bank = captured()
    result = bank.result()
    x = inputs().numpy().astype(np.float64)
    cov = (x - x.mean(0)).T @ (x - x.mean(0)) / len(x)
    eigen = np.linalg.eigvalsh(cov)
    p = eigen / eigen.sum()
    expected_rank = np.exp(-(p * np.log(p)).sum())
    row = next(
        r
        for r in result["paths"]
        if r["path"] == "SS" and r["view"] == "global0" and r["stage"] == "input"
    )
    assert row["effective_rank"] == pytest.approx(expected_rank)
    assert row["relative_variance"] == pytest.approx(np.trace(cov) / np.square(x).sum(1).mean())
    assert row["normalized_rank"] == pytest.approx(expected_rank / 2)
    logits = bank._entries[0]["values"]["raw_logits"].astype(np.float64)
    scaled = (logits - bank._center) / bank._temperature
    q = np.exp(scaled - scaled.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    h = -(q * np.log(q)).sum(1).mean()
    marginal_h = -(q.mean(0) * np.log(q.mean(0))).sum()
    row = next(
        r
        for r in result["sensitivity"]
        if r["path"] == "SS"
        and r["view"] == "global0"
        and r["center_mode"] == "frozen_teacher"
        and r["temperature_factor"] == 1.0
    )
    assert row["H_sample"] == pytest.approx(h)
    assert row["JS"] == pytest.approx(marginal_h - h)
    assert {r["temperature_factor"] for r in result["sensitivity"]} == {0.5, 1.0, 2.0}
    assert {r["center_mode"] for r in result["sensitivity"]} == {"zero", "frozen_teacher"}


def test_uniform_entropy_kl_and_zero_rank():
    obj = Objective()
    with torch.no_grad():
        for role in (obj.student, obj.teacher):
            role.ssl2_cls.prototypes.weight.zero_()
    result = captured(obj).result()
    for row in result["sensitivity"]:
        if row["center_mode"] == "zero":
            assert row["H_sample"] == pytest.approx(math.log(4))
            assert row["JS"] == pytest.approx(0, abs=1e-12)
    assert all(r["effective_rank"] == 0 for r in result["paths"] if r["stage"] == "raw_logits")
    for row in result["pairs"]:
        if row["center_mode"] == "zero":
            assert row["kl_pair_mean"] == pytest.approx(0, abs=1e-12)


def test_pairing_keeps_context_and_condition_control_strata():
    obj = Objective()
    bank = captured(obj)
    with bank.capture(obj, records("batch2")):
        native(obj)
    result = bank.result()
    assert {r["context_id"] for r in result["pairs"]} == {"batch1", "batch2"}
    for row in result["native_pairs"]:
        assert row["independent_samples"] == 4
        assert row["conditions"] == 2
        expected = {
            "same_sample": 4,
            "same_condition_different_control": 4,
            "different_condition_same_control": 4,
            "different_condition_different_control": 4,
        }
        assert row["pair_count"] == expected[row["stratum"]]
        assert row["ce_pair_mean"] == pytest.approx(
            row["teacher_entropy_pair_mean"] + row["kl_pair_mean"]
        )
    native_rows = result["native_pairs"]
    matching = [
        r
        for r in result["pairs"]
        if r["teacher_path"] == "TT"
        and r["center_mode"] == "frozen_teacher"
        and r["temperature_factor"] == 1
    ]
    for a, b in zip(native_rows, matching, strict=True):
        assert a["ce_pair_mean"] == pytest.approx(b["ce_pair_mean"], abs=1e-6)


def test_hooks_removed_on_forward_failure_and_failure_cannot_report_success():
    obj = Objective()
    bank = module.TeacherSignalBank()
    with pytest.raises(RuntimeError, match="forward failed"), bank.capture(obj, records()):
        obj.student.ssl2_cls(inputs())
        raise RuntimeError("forward failed")
    for role in (obj.student, obj.teacher):
        assert not role.ssl2_cls._forward_hooks
        assert not role.ssl2_cls.features._forward_hooks
    with pytest.raises(ValueError, match="failed"):
        bank.result()


def test_missing_globals_leftover_collector_hooks_and_bad_replay_fail_closed():
    obj = Objective()
    bank = module.TeacherSignalBank()
    with pytest.raises(ValueError, match="two Teacher Globals"), bank.capture(obj, records()):
        obj.student.ssl2_cls(inputs())
    leftover = obj.student.ssl2_cls.register_forward_hook(lambda *_: None)
    try:
        with pytest.raises(ValueError, match="Collector/head hooks"):
            captured(obj)
    finally:
        leftover.remove()
    obj = Objective()

    def corrupt(_module, _inputs, output):
        return output + 1

    handle = obj.student.ssl2_cls.features.register_forward_hook(corrupt)
    try:
        with pytest.raises(ValueError, match="Collector/head hooks"):
            captured(obj)
    finally:
        handle.remove()


def test_native_replay_detects_changed_head_and_leaves_no_hooks():
    obj = Objective()
    bank = module.TeacherSignalBank()
    with pytest.raises(ValueError, match="replay exceeds"), bank.capture(obj, records()):
        native(obj)
        with torch.no_grad():
            obj.student.ssl2_cls.prototypes.weight.add_(1.0)
    assert not obj.student.ssl2_cls._forward_hooks
    assert not obj.student.ssl2_cls.features._forward_hooks


def test_disabled_head_explicitly_unavailable_and_never_replayed():
    obj = Objective()
    obj.lambda2 = 0
    bank = module.TeacherSignalBank()
    with bank.capture(obj, records()):
        pass
    assert bank.result()["status"] == "unavailable:ssl2_cls_disabled"
    assert bank.result()["captures"] == []


def test_duplicate_rows_use_first_realization_and_result_is_repeatable():
    obj = Objective()
    bank = captured(obj)
    first = bank.result()
    with bank.capture(obj, records()):
        native(obj, student=inputs() * 3)
    repeated = bank.result()
    assert repeated["duplicate_path_view_rows_ignored"] == 32
    for key in ("paths", "sensitivity", "pairs", "native_pairs", "feature_pairs"):
        assert repeated[key] == first[key]
    assert bank.result() == repeated


def test_historical_head_without_shared_forward_api_uses_actual_forward_once():
    result = captured(Objective(HistoricalHead)).result()
    assert result["status"] == "completed"
    assert all(
        method == "native_forward_with_scoped_features_hook"
        for method in result["captures"][0]["replay_methods"].values()
    )
    assert all(
        p["bottleneck_maxabs"] == p["logits_maxabs"] == 0 for p in result["captures"][0]["parity"]
    )


def test_tuple_returning_head_and_bfloat16_autocast_replay():
    class TupleHead(Head):
        def forward(self, values, *, return_features=False):
            return self.forward_with_features(values)

    obj = Objective(TupleHead)
    bank = module.TeacherSignalBank()
    with torch.autocast("cpu", dtype=torch.bfloat16), bank.capture(obj, records()):
        native(obj)
    result = bank.result()
    assert all(p["output_dtype"] == "torch.bfloat16" for p in result["captures"][0]["parity"])
    assert all(
        p["bottleneck_maxabs"] == p["logits_maxabs"] == 0 for p in result["captures"][0]["parity"]
    )


def test_collector_hook_can_close_inside_capture_before_replay():
    obj = Objective()
    bank = module.TeacherSignalBank()
    with bank.capture(obj, records()):
        handle = obj.student.ssl2_cls.register_forward_hook(lambda *_: None)
        try:
            native(obj)
        finally:
            handle.remove()
    assert bank.result()["status"] == "completed"


def test_center_changes_during_capture_and_never_observed_bank_fail_closed():
    with pytest.raises(ValueError, match="not observed"):
        module.TeacherSignalBank().result()
    obj = Objective()
    bank = module.TeacherSignalBank()
    with pytest.raises(ValueError, match="changed during"), bank.capture(obj, records()):
        native(obj)
        obj.ssl2_cls_center.add_(0.1)
    assert not obj.student.ssl2_cls._forward_hooks
    assert not obj.teacher.ssl2_cls.features._forward_hooks
