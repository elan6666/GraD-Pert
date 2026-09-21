import pytest

from gradpert.training.epoch import execute_epoch


def test_sealed_epoch_rejects_excess_before_update():
    calls = []
    with pytest.raises(ValueError, match="exceeds"):
        execute_epoch([10, 20, 30], lambda batch, i: calls.append((batch, i)), maximum_steps=2)
    assert calls == [(10, 0), (20, 1)]


def test_legacy_adapter_observes_all_batches_for_its_existing_count_check():
    calls = []
    count = execute_epoch([10, 20, 30], lambda batch, i: calls.append((batch, i)))
    assert count == 3
    assert calls == [(10, 0), (20, 1), (30, 2)]


def test_update_failure_propagates_without_consuming_next_batch():
    consumed = []

    def batches():
        for value in range(3):
            consumed.append(value)
            yield value

    def update(batch, index):
        if index == 1:
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        execute_epoch(batches(), update)
    assert consumed == [0, 1]
