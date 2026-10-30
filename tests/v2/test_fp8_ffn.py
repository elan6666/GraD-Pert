"""CPU fake-GEMM tests prove algebra/routing only, never real FP8 execution."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from gradpert.modeling.v2.fp8_ffn import _HybridLinear, _quantize, fp8_linear
from gradpert.modeling.v2.restored_ffn import MatchedGELUFeedForward
from gradpert.training.v2.optimizer import routes


@pytest.fixture
def fake_scaled_mm(monkeypatch):
    calls = []

    def fake(a, b, *, scale_a, scale_b, out_dtype, use_fast_accum):
        assert a.stride(1) == 1 and b.stride(0) == 1
        assert all(dim % 16 == 0 for dim in (*a.shape, *b.shape))
        assert scale_a.dtype == scale_b.dtype == torch.float32
        assert torch.isfinite(scale_a).all() and bool((scale_a > 0).all())
        assert torch.isfinite(scale_b).all() and bool((scale_b > 0).all())
        assert use_fast_accum is False
        calls.append((a.dtype, b.dtype, out_dtype))
        return ((a.float() * scale_a) @ (b.float() * scale_b)).to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake)
    return calls


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7)])
def test_cpu_fake_forward_backward_padding_and_hybrid_algebra(fake_scaled_mm, shape):
    torch.manual_seed(11)
    x = torch.randn(shape, requires_grad=True)
    w = torch.randn(19, 7, requires_grad=True)
    actual = _HybridLinear.apply(x, w)
    x8, sx = _quantize(x.detach().reshape(-1, 7), torch.float8_e4m3fn)
    w8, sw = _quantize(w.detach(), torch.float8_e4m3fn)
    expected = ((x8.float() * sx) @ (w8.float() * sw).t()).to(torch.bfloat16)
    assert torch.equal(actual.reshape(-1, 19), expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    g8, sg = _quantize(grad.reshape(-1, 19), torch.float8_e5m2)
    dx = ((g8.float() * sg) @ (w8.float() * sw)).reshape(shape)
    dw = (g8.float() * sg).t() @ (x8.float() * sx)
    torch.testing.assert_close(x.grad, dx, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(w.grad, dw, rtol=1e-6, atol=1e-6)
    assert w.grad.dtype == torch.float32
    assert fake_scaled_mm == [
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e5m2, torch.float8_e4m3fn, torch.float32),
        (torch.float8_e5m2, torch.float8_e4m3fn, torch.float32),
    ]


def test_default_forward_and_initialization_are_unchanged():
    torch.manual_seed(13)
    ordinary = MatchedGELUFeedForward(7, 0.2)
    after = torch.get_rng_state()
    torch.manual_seed(13)
    opted = MatchedGELUFeedForward(7, 0.2, "fp8_hybrid")
    assert torch.equal(after, torch.get_rng_state())
    assert routes(ordinary) == routes(opted)
    assert list(ordinary.state_dict()) == list(opted.state_dict())
    for name, value in ordinary.state_dict().items():
        assert torch.equal(value, opted.state_dict()[name])
    x = torch.randn(2, 7)
    rng = torch.get_rng_state()
    actual = ordinary(x)
    torch.set_rng_state(rng)
    expected = ordinary.dropout(ordinary.down(ordinary.dropout(F.gelu(ordinary.up(x)))))
    assert torch.equal(actual, expected)


def test_cpu_fake_checkpoint_dropout_and_parameter_ownership(fake_scaled_mm, monkeypatch):
    # Explicit CPU fake hook bypasses only the public CUDA gate for this test.
    monkeypatch.setattr("gradpert.modeling.v2.fp8_ffn.fp8_linear", _HybridLinear.apply)
    eager = MatchedGELUFeedForward(7, 0.3, "fp8_hybrid")
    restored = copy.deepcopy(eager)
    identities = {name: id(p) for name, p in eager.named_parameters()}
    x = torch.randn(2, 3, 7, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    rng = torch.get_rng_state()
    result = eager(x)
    result.float().square().sum().backward()
    end_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    replay = checkpoint(restored, y, use_reentrant=False)
    replay.float().square().sum().backward()
    assert torch.equal(end_rng, torch.get_rng_state())
    assert torch.equal(result, replay)
    assert torch.equal(x.grad, y.grad)
    for (_, a), (_, b) in zip(eager.named_parameters(), restored.named_parameters(), strict=True):
        assert a is not b
        assert torch.equal(a.grad, b.grad)
    assert identities == {name: id(p) for name, p in eager.named_parameters()}
    assert set(eager.state_dict()) == {"up.weight", "down.weight"}


def test_public_path_rejects_cpu_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="same CUDA device"):
        fp8_linear(torch.randn(2, 7), torch.randn(19, 7))


def test_backend_failure_propagates(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("unsupported real backend")

    monkeypatch.setattr(torch, "_scaled_mm", unavailable)
    with pytest.raises(RuntimeError, match="unsupported real backend"):
        _HybridLinear.apply(torch.randn(2, 7), torch.randn(19, 7))


def test_zero_scaling_is_positive_and_finite():
    quantized, scale = _quantize(torch.zeros(3, 7), torch.float8_e4m3fn)
    assert bool(torch.isfinite(scale)) and bool(scale > 0)
    assert torch.equal(quantized.float(), torch.zeros(3, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA FP8 backend unavailable")
def test_real_cuda_fp8_forward_backward_and_exact_reload():
    # On a CUDA host, backend incompatibility is a failure, never a skip/fallback.
    module = MatchedGELUFeedForward(17, 0.2, "fp8_hybrid").cuda()
    x = torch.randn(3, 17, device="cuda", requires_grad=True)
    saved = copy.deepcopy(module.state_dict())
    rng = torch.cuda.get_rng_state()
    out = module(x)
    out.float().square().mean().backward()
    assert out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all())
    assert all(p.grad.dtype == torch.float32 for p in module.parameters())
    gradients = [p.grad.clone() for p in module.parameters()]
    module.load_state_dict(saved)
    module.zero_grad(set_to_none=True)
    torch.cuda.set_rng_state(rng)
    repeat = checkpoint(module, x.detach().requires_grad_(), use_reentrant=False)
    repeat.float().square().mean().backward()
    assert torch.equal(out, repeat)
    for p, prior in zip(module.parameters(), gradients, strict=True):
        assert torch.equal(p.grad, prior)
