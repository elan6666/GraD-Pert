"""Hoisting structural validation must preserve graph math and RNG exactly."""

import copy

import pytest
import torch

from gradpert.modeling.v2.model import RelayGraphLayer


def test_validate_once_preserves_checkpointed_outputs_gradients_rng_and_reduces_scalar_reads():
    torch.manual_seed(2)
    reference = RelayGraphLayer(8, 2, 0.1, chunk_rows=2, checkpoint_chunks=True)
    memory = torch.randn(9, 8)
    neighbors = torch.stack([torch.arange(9), torch.arange(9).roll(1), torch.arange(9).roll(2)], -1)
    valid = torch.ones_like(neighbors, dtype=torch.bool)
    sources = torch.randint(0, 2, (9, 3, 4))
    outputs, counts = [], []
    for once in (False, True):
        layer = copy.deepcopy(reference)
        layer.validate_once = once
        x = memory.clone().requires_grad_()
        torch.manual_seed(29)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            y = layer(x, neighbors, valid, sources)
            y.square().sum().backward()
        outputs.append(
            (y.detach(), x.grad, [p.grad for p in layer.parameters()], torch.get_rng_state())
        )
        counts.append(
            sum(e.count for e in profile.key_averages() if e.key == "aten::_local_scalar_dense")
        )
    for a, b in zip(outputs[0][:2], outputs[1][:2], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(outputs[0][2], outputs[1][2], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(outputs[0][3], outputs[1][3])
    assert counts[1] == 1 and counts[0] > counts[1]


@pytest.mark.parametrize("once", [False, True])
def test_empty_neighbor_row_still_fails_before_returning_output(once):
    layer = RelayGraphLayer(8, 2, 0.1, chunk_rows=2)
    layer.validate_once = once
    valid = torch.ones(3, 2, dtype=torch.bool)
    valid[2] = False
    with pytest.raises(ValueError, match="every graph target"):
        layer(torch.randn(3, 8), torch.zeros(3, 2, dtype=torch.long), valid, torch.zeros(3, 2, 4))
