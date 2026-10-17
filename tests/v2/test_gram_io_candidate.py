"""Independent derivative oracle, scope cleanup, and opt-in kernel execution."""

import copy
import importlib.util
import json
import math
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gradpert.modeling.v2 import weighted_gram
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]


def helper(path):
    spec = importlib.util.spec_from_file_location("gram_io_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANDIDATE = helper("scripts/v2/gram_io_candidate.py")


@pytest.mark.parametrize("length", [1, 3, 17, 32])
@pytest.mark.parametrize("decay_scale", [0.1, 1000.0])
def test_incident_edge_derivatives_against_float64_autograd(length, decay_scale):
    torch.manual_seed(917 + length)
    keys = torch.randn(2, length, 4, 64, dtype=torch.float64).transpose(1, 2).requires_grad_()
    gates = (-torch.rand_like(keys).cumsum(-2) * decay_scale).requires_grad_()
    output = weighted_gram.weighted_gram_reference(keys, gates)
    upstream = torch.randn_like(output).transpose(-1, -2)
    expected = torch.autograd.grad(output, (keys, gates), upstream)
    actual = CANDIDATE.backward_tiled_reference(keys.detach(), gates.detach(), upstream)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_checkpointed_update_wiring_records_cpu_oracle_drift(monkeypatch, record_property):
    """Gate wiring/RNG/finite updates; record, do not accept, numerical drift.

    Mock CPU kernels cannot establish real CUDA multi-update parity. Even using
    the original eager derivative can change gradient accumulation scheduling.
    Preserve statistics at the original gate for the subsequent GPU decision.
    """
    torch.manual_seed(441)  # fixture initialization must not depend on prior tests
    fixture = helper("tests/v2/test_relay_method.py")
    enable = helper("scripts/v2/update_parity.py").enable_fused_gram
    objective, batch = fixture.relay_training_fixture(checkpointed=True)
    monkeypatch.setitem(
        sys.modules,
        "gradpert.modeling.v2._weighted_gram_cuda",
        SimpleNamespace(forward=weighted_gram.weighted_gram_reference),
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.performance.gram_io_cuda",
        # This test checks hook/optimizer/EMA/RNG wiring with the original CPU
        # derivative ordering. The actual new reduction is checked separately.
        SimpleNamespace(backward=weighted_gram.weighted_gram_backward),
    )
    monkeypatch.setattr(CANDIDATE, "fused_gram_io", CANDIDATE._GramIO.apply)
    results = []
    original = weighted_gram.fused_weighted_gram
    for candidate in (False, True):
        current = copy.deepcopy(objective)
        optimizer = V2Optimizer(current.student, 0.001, 0)
        torch.manual_seed(441)
        if candidate:
            assert enable(current) > 0

        def updates(current, optimizer):
            values = []
            for _ in range(3):
                loss = optimizer_step(
                    current, optimizer, batch, microbatch=2, lr=0.001, momentum=0.99, bf16=False
                )
                values.append(
                    (
                        loss,
                        {
                            n: p.grad.clone()
                            for n, p in current.student.named_parameters()
                            if p.grad is not None
                        },
                    )
                )
            return values

        if candidate:
            with CANDIDATE.candidate_scope():
                values = updates(current, optimizer)
        else:
            values = updates(current, optimizer)
        assert weighted_gram.fused_weighted_gram is original
        results.append(
            (values, current.state_dict(), optimizer.state_dict(), torch.get_rng_state())
        )

    drift = []

    def compare(a, b, path="root"):
        if isinstance(a, torch.Tensor):
            floating = a.is_floating_point()
            if floating:
                assert a.dtype == b.dtype and a.shape == b.shape
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                delta = (a.double() - b.double()).reshape(-1)
                drift.append(
                    {
                        "path": path,
                        "max_absolute": delta.abs().max().item() if delta.numel() else 0,
                        "relative_l2": delta.norm().item() / max(b.double().norm().item(), 1e-30),
                        "passes_original_element_gate": torch.allclose(a, b, atol=3e-5, rtol=3e-4),
                    }
                )
            else:
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                compare(left, right, f"{path}/{index}")
        elif isinstance(a, float):
            assert math.isfinite(a) and math.isfinite(b)
            drift.append(
                {
                    "path": path,
                    "max_absolute": abs(a - b),
                    "passes_original_element_gate": math.isclose(a, b, abs_tol=3e-5, rel_tol=3e-4),
                }
            )
        else:
            assert a == b

    compare(*results)
    record_property("cpu_oracle_drift_not_cuda_acceptance", json.dumps(drift))
    with pytest.raises(RuntimeError, match="injected"), CANDIDATE.candidate_scope():
        raise RuntimeError("injected")
    assert weighted_gram.fused_weighted_gram is original


@pytest.mark.parametrize("length", [1, 17, 32])
@pytest.mark.parametrize("scale", [0.1, 1000.0])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.skipif(
    os.environ.get("GRADPERT_TEST_GRAM_IO") not in ("interpret", "cuda"),
    reason="explicit CPU interpreter or idle-CUDA diagnostic only",
)
def test_actual_kernel_derivatives(length, scale, contiguous):
    mode = os.environ["GRADPERT_TEST_GRAM_IO"]
    if mode == "interpret":
        assert os.environ.get("TRITON_INTERPRET") == "1"
    else:
        assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
    from scripts.performance.gram_io_cuda import backward

    torch.manual_seed(120 + length)
    device = "cpu" if mode == "interpret" else "cuda"
    keys = torch.nn.functional.normalize(
        torch.randn(2, length, 4, 64, device=device), dim=-1
    ).transpose(1, 2)
    gates = -torch.rand_like(keys).cumsum(-2) * scale
    if contiguous:
        keys, gates = keys.contiguous(), gates.contiguous()
    upstream = torch.randn(2, 4, length, length, device=device).transpose(-1, -2)
    expected = weighted_gram.weighted_gram_backward(keys, gates, upstream)
    rng = torch.get_rng_state().clone()
    actual = backward(keys, gates, upstream)
    for a, b in zip(actual, expected, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    "script, flags, message",
    [
        ("capacity_probe.py", ["--gram-io"], "benchmark-only"),
        (
            "capacity_probe.py",
            ["--gram-io", "--benchmark-only", "--profile-last-update"],
            "unprofiled",
        ),
        ("capacity_probe.py", ["--gram-io", "--benchmark-only", "--aten-kda"], "isolated"),
        ("update_parity.py", ["--candidate-gram-io", "--candidate-aten-kda"], "isolate"),
        ("update_parity.py", ["--candidate-gram-io", "--candidate-fused-gram"], "isolate"),
    ],
)
def test_diagnostic_scope_rejects_before_cuda(monkeypatch, capsys, script, flags, message):
    path = ROOT / "scripts/v2" / script
    args = [
        str(path),
        "--config",
        "unused.yaml",
        "--data-root",
        "/data/yilangliu",
        "--output",
        "/data/yilangliu/unused",
        "--gpu",
        "0,1",
        "--publication",
        "unused.json",
        "--publication-sha256",
        "0" * 64,
    ]
    if script == "update_parity.py":
        args += ["--candidate-config", "unused.yaml"]
    monkeypatch.setattr(sys, "argv", [*args, *flags])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(path), run_name="__main__")
    assert error.value.code == 2
    assert message in capsys.readouterr().err
