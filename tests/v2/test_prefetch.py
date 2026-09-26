import copy
from threading import Event

import numpy as np
import pytest

from gradpert.training.v2.prefetch import prefetch_cpu


def test_lookahead_preserves_each_consumed_rng_state_and_resume():
    def factory(rng):
        for _ in range(4):
            yield rng.integers(0, 100, size=11)

    expected_rng, actual_rng = np.random.default_rng(17), np.random.default_rng(17)
    stream = prefetch_cpu(factory, actual_rng)
    expected = factory(expected_rng)
    for _ in range(2):
        np.testing.assert_array_equal(next(stream), next(expected))
        assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    saved = copy.deepcopy(actual_rng.bit_generator.state)
    stream.close()
    assert actual_rng.bit_generator.state == saved
    np.testing.assert_array_equal(next(factory(actual_rng)), next(expected))


def test_next_batch_prepares_while_consumer_is_busy():
    ready = Event()

    def factory(rng):
        yield 1
        ready.set()
        yield 2

    stream = prefetch_cpu(factory, np.random.default_rng(1))
    assert next(stream) == 1
    assert ready.wait(2), "next CPU batch did not overlap consumer work"
    assert list(stream) == [2]


def test_failure_propagates_without_committing_failed_batch_rng():
    def factory(rng):
        yield rng.random()
        rng.random()
        raise ValueError("CPU data failed")

    rng = np.random.default_rng(1)
    stream = prefetch_cpu(factory, rng)
    next(stream)
    saved = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="CPU data failed"):
        next(stream)
    assert rng.bit_generator.state == saved


def test_runtime_prefetch_transfers_nested_batches_without_changing_rng_or_values():
    from dataclasses import dataclass
    from types import MethodType

    import torch

    from gradpert.training.v2.runtime import Runtime

    @dataclass
    class Batch:
        values: torch.Tensor
        nested: tuple
        optional: object = None

    def build(self, epoch, device, generator):
        assert device.type == "cpu"
        for _ in range(3):
            yield Batch(torch.tensor(generator.random(4)), (torch.tensor([epoch]),))

    def runtime():
        instance = object.__new__(Runtime)
        instance.device = torch.device("cpu")
        instance.generator = np.random.default_rng(7)
        instance._batches = MethodType(build, instance)
        return instance

    reference, candidate = runtime(), runtime()
    expected = reference.batches(4)
    observed = candidate.batches(4, cpu_prefetch=True)
    try:
        for left, right in zip(expected, observed, strict=True):
            assert torch.equal(left.values, right.values)
            assert torch.equal(left.nested[0], right.nested[0])
            assert right.optional is None
            assert (
                reference.generator.bit_generator.state == candidate.generator.bit_generator.state
            )
    finally:
        observed.close()
