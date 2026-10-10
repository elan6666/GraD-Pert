"""Shared nonlinear building block for the separately configured v2 profile."""

from typing import cast

from torch import Tensor, nn
from torch.nn import functional as F


class UnifiedMLP(nn.Module):
    """Pre-normalized, biased clipped SwiGLU without a residual connection.

    ``normalize=False`` lets an encoder supply the one required pre-norm.
    Scalar expression inputs must be lifted to a vector before normalization.
    Dropout is applied to the gated product and the final projection, matching
    the existing v2 feed-forward placement. Output activations (such as a
    contribution sigmoid) and linear task heads remain the caller's concern.
    """

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        dropout: float = 0.0,
        normalize: bool = True,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        if min(input_width, hidden_width, output_width) < 1:
            raise ValueError("MLP widths must be positive")
        if normalize and input_width == 1:
            raise ValueError("Lift scalar expression to a vector before RMSNorm")
        self.norm: nn.Module = nn.RMSNorm(input_width, eps=1e-6) if normalize else nn.Identity()
        self.gate = nn.Linear(input_width, hidden_width, bias=True)
        self.up = nn.Linear(input_width, hidden_width, bias=True)
        self.down = nn.Linear(hidden_width, output_width, bias=True)
        self.dropout = nn.Dropout(dropout)
        if zero_output:
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm(x)
        gate = self.gate(normalized).clamp(max=10)
        up = self.up(normalized).clamp(min=-10, max=10)
        product = self.dropout(F.silu(gate) * up)
        return cast(Tensor, self.dropout(self.down(product)))
