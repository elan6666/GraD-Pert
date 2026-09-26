import pytest
import torch

from gradpert.modeling.v2.operators import sinkhorn
from gradpert.modeling.v2.sinkhorn_fused import (
    fused_sinkhorn,
    reference_backward,
    reference_with_saved,
)


@pytest.mark.parametrize("iterations", [1, 3, 20])
@pytest.mark.parametrize("scale", [0.2, 4.0, 20.0])
def test_analytic_sinkhorn_gradient(iterations, scale):
    torch.manual_seed(71)
    x = (torch.randn(3, 4, 4) * scale).requires_grad_()
    output = sinkhorn(x, iterations)
    upstream = torch.randn_like(output)
    gradient = torch.autograd.grad(output, x, upstream)[0]
    actual, saved = reference_with_saved(x.detach(), iterations)
    torch.testing.assert_close(actual, output, atol=0, rtol=0)
    torch.testing.assert_close(
        reference_backward(upstream, actual, saved), gradient, atol=3e-6, rtol=3e-5
    )


def test_cpu_fused_path_fails_explicitly():
    with pytest.raises(ValueError, match="CUDA"):
        fused_sinkhorn(torch.zeros(2, 4, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("count", [1, 17, 4096])
def test_no_grad_fused_path_matches_training_forward(count):
    from gradpert.modeling.v2._sinkhorn_cuda import forward

    torch.manual_seed(73)
    logits = (torch.randn(count, 4, 4, device="cuda") * 4).requires_grad_()
    training_output, _ = forward(logits.detach(), 20)
    with torch.no_grad():
        teacher_output = fused_sinkhorn(logits)
        saved_output = fused_sinkhorn(logits, save_no_grad_intermediates=True)
    torch.testing.assert_close(teacher_output, training_output, atol=0, rtol=0)
    torch.testing.assert_close(teacher_output, saved_output, atol=0, rtol=0)
    assert teacher_output.grad_fn is None

    # Frozen parameters also need the memory-saving path when grad mode is on.
    frozen_output = fused_sinkhorn(logits.detach())
    torch.testing.assert_close(frozen_output, training_output, atol=0, rtol=0)
