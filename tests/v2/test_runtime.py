from types import SimpleNamespace

from gradpert.training.v2.runtime import Runtime


def test_condition_cap_is_bounded_by_actual_probe_batch():
    calls = []

    def steps_per_epoch(**kwargs):
        calls.append(kwargs)
        return 7

    runtime = Runtime(
        data=SimpleNamespace(steps_per_epoch=steps_per_epoch),
        options=SimpleNamespace(max_conditions=8),
        index=None,
        objective=None,
        optimizer=None,
        generator=None,
        device=None,
        batch_size=2,
        identity={},
    )
    assert runtime.steps_per_epoch == 7
    assert calls == [{"batch_size": 2, "max_unique_conditions": 2}]
