"""CPU synthetic delegation/observer boundaries; no native CUDA acceptance."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.v2.capacity_probe import state_fingerprint, validation_state

SCRIPT = Path(__file__).parents[2] / "scripts/v2/response_anchor_parity.py"
loader = importlib.util.spec_from_file_location("response_anchor_parity_tested", SCRIPT)
pilot = importlib.util.module_from_spec(loader)
loader.loader.exec_module(pilot)

METRICS = (
    "txpert_macro_pearson_delta_all",
    "txpert_macro_pearson_delta_deg",
    "trishift_pearson_delta_all",
    "trishift_pearson_delta_deg",
    "systema_pearson_all",
    "systema_pearson_deg",
)


def fixture(*, controls=5, width=7, query=3, batch=2, failure=None):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.response = nn.Identity()
            self.weight = nn.Parameter(torch.ones(1))

    class Objective(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = Model()
            self.teacher = Model()
            self.pending = {}
            self._temperature_updates = 0
            self.register_buffer("temperature_update_index", torch.zeros((), dtype=torch.long))
            self.register_buffer(
                "private_clock", torch.zeros((), dtype=torch.long), persistent=False
            )

        def commit_statistics(self):
            raise AssertionError("must never commit")

    objective = Objective().eval()
    runtime = SimpleNamespace(
        objective=objective,
        optimizer=torch.optim.SGD(objective.parameters(), lr=0.1),
        index=None,
        generator=np.random.default_rng(7),
        device=torch.device("cpu"),
    )
    x = np.arange(controls * width, dtype=np.float32).reshape(controls, width) / 10
    ids = tuple(f"c{i}" for i in range(controls))
    truth = SimpleNamespace(expression=x[:2] + 0.2, ordered_row_ids=("t0", "t1"))

    class Data:
        def load_control_rows(self, ordered_ids):
            return SimpleNamespace(ordered_row_ids=tuple(ordered_ids), expression=x)

        def load_truth_rows(self, condition):
            assert condition == "p0"
            return truth

    data, native = Data(), SimpleNamespace()
    calls = []
    changed_state = []

    def predict_query_set(model, index, controls, targets, queries, *, device, cell_batch):
        results = []
        for start in range(0, len(controls), cell_batch):
            block = torch.tensor(controls[start : start + cell_batch, queries])
            tokens = torch.cat(
                [
                    block[:, :, None].expand(-1, -1, 4),
                    block.mean(1)[:, None, None].expand(-1, 1, 4),
                ],
                dim=1,
            )
            output = model.response(tokens)
            results.append(output[:, :-1, 0].detach().numpy())
        return np.concatenate(results)

    def predict_controls(model, index, controls, targets, *, device, cell_batch, query_count):
        results = [
            native.predict_query_set(
                model,
                index,
                controls,
                targets,
                np.arange(i, min(i + query_count, width)),
                device=device,
                cell_batch=cell_batch,
            )
            for i in range(0, width, query_count)
        ]
        result = np.concatenate(results, axis=1)
        if failure == "prediction" and len(calls) == 2:
            result[0, 0] += 1
        return result

    def evaluate(model, index, data, reference, **kwargs):
        assert kwargs["condition_ids"] == ("p0",)
        assert kwargs["expected_split"] == "val"
        calls.append(dict(kwargs))
        loaded = data.load_control_rows(ids)
        native.predict_controls(
            model,
            index,
            loaded.expression,
            (0,),
            device=kwargs["device"],
            cell_batch=kwargs["cell_batch"],
            query_count=kwargs["query_count"],
        )
        data.load_truth_rows("p0")
        if failure == "exception":
            raise RuntimeError("native failure")
        if failure == "state":
            changed_state.append(1)
        if failure == "teacher":
            objective.teacher.response(torch.ones(1))
        if failure == "rng":
            torch.rand(1)
        if failure != "callback_missing":
            kwargs["on_condition_complete"](1, 1, "p1" if failure == "callback" else "p0")
        metrics = [
            dict(metric_id=name, total_condition_count=1, finite_condition_count=0, value=None)
            for name in METRICS
        ]
        result = dict(
            split="val",
            prediction_loss=0.1,
            metrics=metrics,
            metric_gene_groups={
                name: dict(metrics=metrics) for name in ("seen_expression", "unseen_expression")
            },
            conditions=[
                dict(
                    condition_id="p0",
                    control_row_ids_sha256=pilot.postfit.digest(list(ids)),
                    truth_row_ids_sha256=pilot.postfit.digest(list(truth.ordered_row_ids)),
                )
            ],
        )
        if failure == "metrics" and len(calls) == 2:
            result["prediction_loss"] = 0.2
        if failure == "rows" and len(calls) == 2:
            result["conditions"][0]["truth_row_ids_sha256"] = "changed"
        return result

    native.predict_query_set, native.predict_controls, native.evaluate = (
        predict_query_set,
        predict_controls,
        evaluate,
    )
    collector = pilot.postfit.NativeValCollector(
        ("p0",),
        {"p0": (0,)},
        {"p0": ids},
        np.arange(width),
        controls=controls,
        expression_genes=width,
        query_count=query,
    )
    arguments = dict(
        expected_split="val",
        device="cpu",
        cell_batch=batch,
        query_count=query,
        condition_ids=("p0",),
    )

    def state():
        return dict(rng=torch.get_rng_state().numpy().tobytes().hex(), changed=list(changed_state))

    admissions = []
    return (
        (native, runtime, data, None, collector, arguments, state, lambda: admissions.append(1)),
        calls,
        admissions,
    )


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_native_delegation_twice_one_condition_exact_prediction_and_na_payload(batch):
    args, calls, admissions = fixture(batch=batch)
    receipt, result, arrays = pilot.paired_evaluate(*args)
    assert len(calls) == 2 and len(admissions) == 5
    assert receipt["passed"] and receipt["atol"] == receipt["max_abs"] == 0
    assert receipt["prediction_sha256"][0] == receipt["prediction_sha256"][1]
    assert result["metrics"][0]["value"] is None
    assert arrays["cls_means"].shape == (1, 4)
    assert receipt["observer_calls"]["query_blocks"] == 3


def test_exact300_by5000_and_five_blocks_no_extra_prediction_forward():
    args, calls, _ = fixture(controls=300, width=5000, query=1000, batch=128)
    receipt, _, _ = pilot.paired_evaluate(*args)
    assert len(calls) == 2
    assert receipt["prediction_shape"] == [300, 5000]
    assert receipt["observer_calls"]["query_blocks"] == 5
    assert receipt["observer_calls"]["response_forward"] == 15


@pytest.mark.parametrize(
    "failure,match",
    [
        ("prediction", "changed native predictions"),
        ("metrics", "changed native metrics"),
        ("rows", "changed native metrics"),
        ("state", "mutated protected state"),
        ("rng", "mutated protected state"),
        ("callback", "callback differs"),
        ("callback_missing", "coverage differs"),
        ("teacher", "forbidden postfit path"),
        ("exception", "native failure"),
    ],
)
def test_fail_closed_and_restore_original_methods_hooks(failure, match):
    args, _, _ = fixture(failure=failure)
    native, runtime, data = args[:3]
    methods = native.predict_controls, native.predict_query_set
    with pytest.raises((ValueError, RuntimeError), match=match):
        pilot.paired_evaluate(*args)
    assert (native.predict_controls, native.predict_query_set) == methods
    assert not runtime.objective.student.response._forward_hooks
    assert not runtime.objective.teacher.response._forward_pre_hooks
    assert "load_control_rows" not in vars(data) and "load_truth_rows" not in vars(data)


def test_full445_spec_order_is_unchanged_first_only_and_mismatch_rejected():
    ordered = tuple(f"p{i}" for i in range(445))
    validation = dict(conditions=list(ordered))
    assert pilot.first_condition(validation, ordered) == ("p0",)
    assert validation["conditions"] == list(ordered)
    with pytest.raises(ValueError, match="full frozen445"):
        pilot.first_condition(validation, tuple(reversed(ordered)))
    with pytest.raises(ValueError, match="full frozen445"):
        pilot.first_condition(dict(conditions=["p0"]), ("p0",))


def test_more_than_one_actual_condition_is_rejected_before_native_forward():
    args, calls, _ = fixture()
    args[5]["condition_ids"] = ("p0", "p1")
    with pytest.raises(ValueError, match="exactly the first"):
        pilot.paired_evaluate(*args)
    assert not calls


def test_capture_returns_original_object_and_restores_on_error():
    value = np.ones((2, 3))

    def original():
        return value

    native = SimpleNamespace(predict_controls=original)
    with pytest.raises(RuntimeError), pilot.capture_prediction(native, controls=2, width=3) as raw:
        assert native.predict_controls() is value
        assert raw[0] is not value and np.array_equal(raw[0], value)
        raise RuntimeError("stop")
    assert native.predict_controls is original


def test_private_output_server_boundary_and_fullval_overlap_rejected():
    prefix = "/data/yilangliu/synthetic-parity-output-test"
    spec = dict(
        data_root=prefix + "/data",
        analysis_source=dict(root=prefix + "/analysis"),
        output=prefix + "/fullval",
        endpoints=[
            dict(name="R3-RI", run_root=prefix + "/run", source=dict(root=prefix + "/native"))
        ],
    )
    with pytest.raises(ValueError, match="canonical server path"):
        pilot.output_path(spec, "R3-RI", "/tmp/forbidden-parity-output")
    with pytest.raises(ValueError, match="overlaps"):
        pilot.output_path(spec, "R3-RI", prefix + "/fullval/pilot")
    assert pilot.output_path(spec, "R3-RI", prefix + "/pilot") == Path(prefix + "/pilot")


def complete_state(runtime):
    return pilot.protected_state(
        runtime, validation_state=validation_state, fingerprint=state_fingerprint
    )


@pytest.mark.parametrize("mutation", ["temperature_both", "private_clock", "private_counter"])
def test_real_historical_validation_state_gap_is_closed(mutation):
    args, calls, _ = fixture()
    native, runtime = args[:2]
    objective = runtime.objective
    objective._private_diagnostic_counter = 0
    original_evaluate = native.evaluate

    def mutate():
        if mutation == "temperature_both":
            objective._temperature_updates += 1
            objective.temperature_update_index.add_(1)
        elif mutation == "private_clock":
            objective.private_clock.add_(1)
        else:
            objective._private_diagnostic_counter += 1

    # Reproduce the reviewer's historical boundary before exercising the repair.
    historical, full = validation_state(runtime), complete_state(runtime)
    mutate()
    assert validation_state(runtime) == historical
    assert complete_state(runtime) != full

    def injected(*positional, **kwargs):
        result = original_evaluate(*positional, **kwargs)
        mutate()
        return result

    native.evaluate = injected
    new_args = (*args[:6], lambda: complete_state(runtime), args[7])
    with pytest.raises(ValueError, match="mutated protected state or RNG"):
        pilot.paired_evaluate(*new_args)
    assert len(calls) == 1  # Fail after the first delegated pass; never accept off/on parity.
    assert not objective.student.response._forward_hooks
    assert not objective.teacher.response._forward_pre_hooks


def test_anchor_forward_with_real_state_helper_fails_before_readout_and_restores_hooks():
    args, calls, _ = fixture()
    native, runtime = args[:2]
    head = nn.Linear(4, 4)
    runtime.objective.student.response_anchor_readout = head
    completed_head_forwards = []
    existing = head.register_forward_hook(lambda *a: completed_head_forwards.append(1))
    original_hooks = dict(head._forward_hooks)
    original_pre_hooks = dict(head._forward_pre_hooks)
    original_evaluate = native.evaluate

    def injected(*positional, **kwargs):
        result = original_evaluate(*positional, **kwargs)
        head(torch.ones(1, 4))
        return result

    native.evaluate = injected
    new_args = (*args[:6], lambda: complete_state(runtime), args[7])
    before = complete_state(runtime)
    try:
        with pytest.raises(ValueError, match="forbidden postfit path: anchor_forward"):
            pilot.paired_evaluate(*new_args)
        assert len(calls) == 1 and not completed_head_forwards
        assert complete_state(runtime) == before
        assert runtime.objective.student.response_anchor_readout is head
        assert dict(head._forward_hooks) == original_hooks
        assert dict(head._forward_pre_hooks) == original_pre_hooks
        assert not runtime.objective.student.response._forward_hooks
        assert not runtime.objective.teacher.response._forward_pre_hooks
    finally:
        existing.remove()


def test_anchor_guard_counts_rejected_call_and_restores_on_independent_error():
    args, _, _ = fixture()
    runtime = args[1]
    head = nn.Linear(4, 4)
    runtime.objective.student.response_anchor_readout = head
    with pilot.forbid_anchor_forward(runtime) as counts:
        with pytest.raises(ValueError, match="anchor_forward"):
            head(torch.ones(1, 4))
        assert counts == dict(anchor_forward=1)
    assert not head._forward_pre_hooks
    with pytest.raises(RuntimeError), pilot.forbid_anchor_forward(runtime):
        raise RuntimeError("independent failure")
    assert not head._forward_pre_hooks


def test_complete_objective_buffers_teacher_and_center_mutations_are_bound():
    args, _, _ = fixture()
    runtime = args[1]
    objective = runtime.objective
    objective.register_buffer("extra_objective_buffer", torch.zeros(1))
    objective.register_buffer("response_center", torch.zeros(1))
    objective.teacher.register_buffer("teacher_extra_buffer", torch.zeros(1))
    for buffer in (
        objective.extra_objective_buffer,
        objective.response_center,
        objective.teacher.teacher_extra_buffer,
    ):
        before = complete_state(runtime)
        buffer.add_(1)
        assert complete_state(runtime) != before


def test_actual_validation_state_success_records_zero_anchor_forward():
    args, _, _ = fixture()
    runtime = args[1]
    runtime.objective.student.response_anchor_readout = nn.Linear(4, 4)
    new_args = (*args[:6], lambda: complete_state(runtime), args[7])
    receipt, _, _ = pilot.paired_evaluate(*new_args)
    assert receipt["state_before"] == receipt["state_after"]
    assert all(row["anchor_forward"] == 0 for row in receipt["forbidden_calls"])
    assert not runtime.objective.student.response_anchor_readout._forward_pre_hooks
