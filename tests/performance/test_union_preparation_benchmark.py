"""The diagnostic alternative must preserve every prepared channel element."""

import numpy as np
import pytest
import torch

from scripts.performance.benchmark_union_preparation import prepare


@pytest.mark.parametrize("pairs", [(), ((0, 1),), ((0, 1), (1, 0)), ((2, 2), (0, 1), (0, 1))])
def test_preparation_exact(pairs):
    sources = (("string", pairs), ("go", ((2, 0),)))
    rng = torch.get_rng_state().clone()
    reference = prepare(3, sources, array_native=False)
    candidate = prepare(3, sources, array_native=True)
    assert reference[0] == candidate[0]
    np.testing.assert_array_equal(reference[1], candidate[1])
    np.testing.assert_array_equal(reference[2], candidate[2])
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("edge", [(-1, 0), (0, 3)])
def test_invalid_axis_rejected(mode, edge):
    with pytest.raises(ValueError, match="outside node axis"):
        prepare(3, (("string", (edge,)),), array_native=mode)
