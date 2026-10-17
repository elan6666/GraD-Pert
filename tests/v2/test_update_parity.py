"""Evidence helpers must notice gradient and ordered-input changes."""

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gradpert.config.v2 import V2Architecture

SPEC = importlib.util.spec_from_file_location(
    "update_parity", Path(__file__).resolve().parents[2] / "scripts/v2/update_parity.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_input_digest_preserves_gene_order_dtype_and_rng_bits():
    @dataclass
    class Batch:
        genes: torch.Tensor
        rng: np.ndarray

    batch = Batch(torch.tensor([1, 2]), np.array([3, 4], dtype=np.uint32))
    original = MODULE.tree_digest(batch)
    assert original == MODULE.tree_digest(Batch(batch.genes.clone(), batch.rng.copy()))
    assert original != MODULE.tree_digest(Batch(batch.genes.flip(0), batch.rng))
    assert original != MODULE.tree_digest(Batch(batch.genes.float(), batch.rng))
    batch.rng[0] += 1
    assert original != MODULE.tree_digest(batch)


def test_output_capture_keeps_autograd_rng_and_restores_methods_on_exception():
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def encode_response(self, x):
            return {"prediction": self.weight * x, "response_cls": x}

    objective = SimpleNamespace(student=Model(), teacher=Model())
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError), MODULE.capture_response_outputs(objective) as captured:
        result = objective.student.encode_response(torch.tensor(3.0))
        result["prediction"].backward()
        assert captured["student"][0]["prediction"].item() == 6
        assert not captured["student"][0]["prediction"].requires_grad
        assert objective.student.weight.grad.item() == 3
        raise RuntimeError("injected")
    assert torch.equal(before, torch.get_rng_state())
    assert all("encode_response" not in m.__dict__ for m in [objective.student, objective.teacher])


def test_parity_checks_small_gradients_and_optimizer_structure():
    reference = {"gradient": torch.zeros(4), "optimizer": {"step": 2, "state": torch.ones(2)}}
    candidate = MODULE.cpu_copy(reference)
    assert MODULE.compare_trees(reference, candidate)["passed"]
    candidate["gradient"][0] = 1e-3
    failed = MODULE.compare_trees(reference, candidate)
    assert not failed["passed"]
    assert failed["failures"][0]["path"] == "root/gradient"
    assert torch.equal(reference["gradient"], torch.zeros(4))
    del candidate["optimizer"]["step"]
    assert not MODULE.compare_trees(reference, candidate)["passed"]


def test_nonfinite_or_dtype_changes_do_not_pass_parity():
    assert not MODULE.compare_trees(torch.ones(2), torch.tensor([1.0, float("nan")]))["passed"]
    assert not MODULE.compare_trees(torch.ones(2), torch.ones(2, dtype=torch.float64))["passed"]


def test_recording_finite_drift_preserves_the_original_failed_gate():
    reference = {"weight": torch.zeros(2), "updates": 3}
    drifted = {"weight": torch.tensor([1e-3, 0.0]), "updates": 3}
    result = MODULE.compare_trees(reference, drifted)
    assert not result["passed"] and result["finite"] and result["structural_exact"]
    assert result["numeric_failure_count"] == 1
    assert result["max_relative_l2"] > 0
    assert MODULE.may_continue_comparison(
        {"weights": result}, exact_inputs_rng=True, record_finite_drift=True
    )
    assert not MODULE.may_continue_comparison(
        {"weights": result}, exact_inputs_rng=True, record_finite_drift=False
    )
    assert not MODULE.may_continue_comparison(
        {"weights": result}, exact_inputs_rng=False, record_finite_drift=True
    )


@pytest.mark.parametrize(
    "reference,actual",
    [
        (torch.zeros(1), torch.tensor([float("nan")])),
        (torch.zeros(1), torch.tensor([float("inf")])),
        (torch.zeros(1), torch.zeros(2)),
        (torch.zeros(1), torch.zeros(1, dtype=torch.float64)),
        (torch.tensor([1000000]), torch.tensor([1000001])),
        (1000000, 1000001),
        ({"grad": None}, {"grad": torch.zeros(1)}),
    ],
)
def test_finite_drift_does_not_bypass_nonfinite_identity_or_gradient_structure(reference, actual):
    result = MODULE.compare_trees(reference, actual)
    assert not result["passed"]
    assert not MODULE.may_continue_comparison(
        {"result": result}, exact_inputs_rng=True, record_finite_drift=True
    )


def test_comparison_stream_can_request_three_updates_and_closes_once():
    from types import SimpleNamespace

    closed = []

    def batches(epoch, *, cpu_prefetch):
        try:
            yield from range(20)
        finally:
            closed.append(epoch)

    runtime = SimpleNamespace(batches=batches)
    assert list(
        MODULE.comparison_batches(runtime, staged=False, cpu_prefetch=False, updates=3)
    ) == [0, 1, 2]
    assert closed == [0]


@pytest.mark.parametrize("rate", ["0", "-0.1", "nan", "inf"])
def test_explicit_nonzero_learning_rate_rejects_invalid_values_before_cuda(rate):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/v2/update_parity.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "unused",
            "--candidate-config",
            "unused",
            "--data-root",
            "/data/yilangliu",
            "--output",
            "/data/yilangliu/unused",
            "--gpu",
            "0,1",
            "--publication",
            "unused",
            "--publication-sha256",
            "unused",
            "--diagnostic-learning-rate",
            rate,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "positive and finite" in result.stderr


def test_digest_accepts_zero_stride_singletons_broadcast_and_bfloat16():
    for dtype in (torch.int64, torch.bfloat16):
        seed = torch.tensor([3], dtype=dtype)
        singleton = torch.as_strided(seed, (1,), (0,))
        assert MODULE.tree_digest(singleton) == MODULE.tree_digest(seed)
        broadcast = singleton.expand(5)
        assert MODULE.tree_digest(broadcast) == MODULE.tree_digest(torch.full((5,), 3, dtype=dtype))


def test_reference_repeat_cannot_hide_an_architecture_change():
    original = V2Architecture(
        attention="relay_full", graph_read_mode="relay", graph_layers=4, kda_layers=2
    )
    assert MODULE.execution_changes(original, original, True) == []
    changed = replace(original, relay_validate_once=True)
    assert MODULE.execution_changes(original, changed, False) == ["relay_validate_once"]
    with pytest.raises(AssertionError, match="identical architecture"):
        MODULE.execution_changes(original, changed, True)
    with pytest.raises(AssertionError, match="one execution factor"):
        MODULE.execution_changes(original, original, False)
    with pytest.raises(AssertionError, match="one execution factor"):
        MODULE.execution_changes(original, replace(changed, relay_kernel="inductor"), False)


def test_staged_parity_reopens_and_closes_data_at_the_phase_boundary():
    from types import SimpleNamespace

    steps, opened, closed = [], [], []
    runtime = SimpleNamespace(step=0)

    def batches(epoch, *, cpu_prefetch):
        assert not cpu_prefetch
        phase = "ssl" if runtime.step < 2 else "mse"
        opened.append((epoch, phase))
        try:
            for _ in range(20):
                yield phase
        finally:
            closed.append(epoch)

    runtime.batches = batches
    for phase in MODULE.comparison_batches(runtime, staged=True, cpu_prefetch=False):
        steps.append(phase)
        runtime.step += 1
    assert steps == ["ssl", "ssl", "mse", "mse"]
    assert opened == [(0, "ssl"), (1, "mse")]
    assert closed == [0, 1]


def test_sequence_checkpoint_diagnostic_changes_both_sides_only():
    from types import SimpleNamespace

    def model():
        return SimpleNamespace(
            cell=SimpleNamespace(checkpoint_layers=True),
            response=SimpleNamespace(checkpoint_layers=True),
            graph=SimpleNamespace(checkpoint_chunks=True),
        )

    objective = SimpleNamespace(student=model(), teacher=model())
    MODULE.disable_sequence_checkpoint(objective)
    for network in (objective.student, objective.teacher):
        assert not network.cell.checkpoint_layers
        assert not network.response.checkpoint_layers
        assert network.graph.checkpoint_chunks


@pytest.mark.parametrize("conflict", ["--reference-repeat", "--no-sequence-checkpoint"])
def test_candidate_checkpoint_diagnostic_rejects_ambiguous_overrides(conflict):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/v2/update_parity.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "unused",
            "--candidate-config",
            "unused",
            "--data-root",
            "/data/yilangliu",
            "--output",
            "/data/yilangliu/unused",
            "--gpu",
            "0,1",
            "--publication",
            "unused",
            "--publication-sha256",
            "unused",
            "--candidate-no-sequence-checkpoint",
            conflict,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "candidate-only checkpoint diagnostic" in result.stderr


def test_no_grad_elision_diagnostic_rejects_mixed_execution_factors():
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/v2/update_parity.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "unused",
            "--candidate-config",
            "unused",
            "--data-root",
            "/data/yilangliu",
            "--output",
            "/data/yilangliu/unused",
            "--gpu",
            "0,1",
            "--publication",
            "unused",
            "--publication-sha256",
            "unused",
            "--candidate-no-grad-elision",
            "--candidate-fused-gram",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "no-grad storage diagnostic" in result.stderr


def test_gram_forward_audit_reports_first_actual_mismatch(monkeypatch):
    from gradpert.modeling.v2 import weighted_gram

    monkeypatch.setattr(
        weighted_gram,
        "fused_weighted_gram",
        lambda k, g: weighted_gram.weighted_gram_reference(k, g) + 0.01,
    )
    MODULE.enable_gram_forward_audit()
    with pytest.raises(RuntimeError, match="Gram forward audit mismatch") as error:
        weighted_gram.fused_weighted_gram(torch.ones(1, 1, 2, 3), torch.zeros(1, 1, 2, 3))
    assert '"call": 1' in str(error.value)
    assert '"shape": [1, 1, 2, 3]' in str(error.value)


def test_gram_backward_audit_checks_real_upstream(monkeypatch):
    from gradpert.modeling.v2 import weighted_gram

    monkeypatch.setattr(weighted_gram, "fused_weighted_gram", weighted_gram.weighted_gram_reference)
    original = weighted_gram.weighted_gram_backward
    monkeypatch.setattr(
        weighted_gram,
        "weighted_gram_backward",
        lambda k, g, u: tuple(v + 0.01 for v in original(k, g, u)),
    )
    MODULE.enable_gram_forward_audit(backward=True)
    key = torch.ones(1, 1, 2, 3, requires_grad=True)
    gate = torch.zeros_like(key, requires_grad=True)
    with pytest.raises(RuntimeError, match="Gram backward audit mismatch"):
        weighted_gram.fused_weighted_gram(key, gate).sum().backward()
