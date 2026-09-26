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
