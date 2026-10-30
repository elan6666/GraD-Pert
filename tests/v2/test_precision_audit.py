"""Audit gates and hook cleanup; CPU tests do not establish native CUDA support."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_restored_ffn import restored

from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.v2 import precision_audit as audit
from gradpert.training.v2.objective import JointObjective


def model_pair(**changes):
    obj, _ = restored("BR")
    old = obj.student
    options = replace(old.options, ffn_variant="gelu", **changes)
    new = GraDPertV2(old.graph.embedding.weight.detach().clone(), options)
    return old, new


def test_historical_policy_has_no_precision_observer():
    options = SimpleNamespace()
    assert not audit.enabled(options)
    options.ssl_head_precision = "float32"
    assert audit.enabled(options)


def test_native_backend_does_not_treat_cpu_as_a_cuda_pass():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        audit.backend_facts(torch.device("cpu"), fp8=True)


def test_fp32_head_observation_and_cleanup(monkeypatch):
    # Only the backend prerequisite is stubbed; real CPU head computation and
    # dtype hooks run. This is a unit contract, never a native pass receipt.
    monkeypatch.setattr(audit, "backend_facts", lambda *a, **k: {"unit_test_only": True})
    _, model = model_pair(ssl_head_precision="float32")
    obj = JointObjective(model)
    with audit.PrecisionAudit(obj, model.options, torch.device("cpu")) as observer:
        for phase in ("before_restore", "after_restore"):
            if phase == "after_restore":
                observer.mark_restored(True)
            for side in (obj.student, obj.teacher):
                for name in ("ssl1_cls", "ssl2_cls", "ssl2_node"):
                    getattr(side, name)(torch.randn(3, model.options.width))
        assert observer.finish()["status"] == "passed"
        assert observer.head_counts
    assert all(not m._forward_hooks for m in model.modules())


def test_audit_cannot_pass_without_selected_heads(monkeypatch):
    monkeypatch.setattr(audit, "backend_facts", lambda *a, **k: {"unit_test_only": True})
    _, model = model_pair(ssl_head_precision="float32")
    with (
        audit.PrecisionAudit(JointObjective(model), model.options, torch.device("cpu")) as obs,
        pytest.raises(RuntimeError, match="not every selected"),
    ):
        obs.mark_restored(True)
        obs.finish()


def test_fp8_finish_needs_each_gemm_phase(monkeypatch):
    monkeypatch.setattr(audit, "backend_facts", lambda *a, **k: {"unit_test_only": True})
    _, model = model_pair()
    obj = JointObjective(model)
    options = replace(model.options, ffn_compute_precision="fp8_hybrid")
    obs = audit.PrecisionAudit(obj, options, torch.device("cpu"))
    obs.parameter_names[123] = "student.ffn.up.weight"
    obs.mark_restored(True)
    obs.gemm_counts[
        (
            "before_restore",
            "student.ffn.up.weight",
            "forward",
            "torch.float8_e4m3fn",
            "torch.float8_e4m3fn",
            "torch.bfloat16",
        )
    ] = 1
    with pytest.raises(RuntimeError, match="module/owner/restore-phase"):
        obs.finish()


def test_teacher_storage_rejection_leaves_no_student_hooks(monkeypatch):
    monkeypatch.setattr(audit, "backend_facts", lambda *a, **k: {"unit_test_only": True})
    _, model = model_pair(ssl_head_precision="float32")
    obj = JointObjective(model)
    obj.teacher.half()
    with (
        pytest.raises(RuntimeError, match="parameter storage"),
        audit.PrecisionAudit(obj, model.options, torch.device("cpu")),
    ):
        pass
    assert all(not m._forward_hooks for m in model.modules())


def test_no_restore_cannot_pass_even_if_kernels_were_seen(monkeypatch):
    monkeypatch.setattr(audit, "backend_facts", lambda *a, **k: {"unit_test_only": True})
    _, model = model_pair(ssl_head_precision="float32")
    with (
        audit.PrecisionAudit(JointObjective(model), model.options, torch.device("cpu")) as obs,
        pytest.raises(RuntimeError, match="full-state restore"),
    ):
        obs.finish()


def test_precision_restore_fingerprint_includes_native_rng(monkeypatch):
    import numpy as np

    from scripts.v2 import capacity_probe

    options = SimpleNamespace(ssl_head_precision="float32")
    runtime = SimpleNamespace(
        objective=SimpleNamespace(
            student=SimpleNamespace(options=options), state_dict=lambda: {"weight": 1}
        ),
        optimizer=SimpleNamespace(state_dict=lambda: {"step": 5}),
        generator=np.random.default_rng(1),
    )
    monkeypatch.setattr(
        capacity_probe, "validation_state", lambda runtime: {"native_rng": torch.get_rng_state()}
    )
    before = capacity_probe.staged_restore_fingerprint(runtime, continuing_joint=False)
    with torch.random.fork_rng():
        torch.rand(1)
        after = capacity_probe.staged_restore_fingerprint(runtime, continuing_joint=False)
    assert before != after
    assert before == capacity_probe.staged_restore_fingerprint(runtime, continuing_joint=False)
