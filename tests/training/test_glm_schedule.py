import math

import pytest

from gradpert.config.step_schedule import (
    EndpointLRWarmupCosine,
    LRWarmupCosine,
    load_training_schedule,
)


def test_registered_glm_endpoint_schedule_and_resume_index():
    schedule = load_training_schedule(
        {
            "name": "lr_warmup_cosine_endpoint",
            "interval": "step",
            "max_lr": 0.001,
            "min_lr": 0.0002,
            "warmup_fraction": 8 / 50,
        }
    )
    assert isinstance(schedule, EndpointLRWarmupCosine)
    total, warmup = 16750, 2680
    for step in (0, 1, warmup - 1, warmup, warmup + 1, 10000, total - 1):
        expected = (
            0.001 * step / warmup
            if step < warmup
            else (
                0.0002 + 0.0004 * (1 + math.cos(math.pi * (step - warmup) / (total - 1 - warmup)))
            )
        )
        assert schedule.at_step(step, total) == {"learning_rate": pytest.approx(expected)}
    assert schedule.at_step(0, total)["learning_rate"] == 0
    assert schedule.at_step(warmup, total)["learning_rate"] == 0.001
    assert schedule.at_step(total - 1, total)["learning_rate"] == 0.0002
    # Resume is a pure function of original global step, never a new horizon.
    full = [schedule.at_step(s, total) for s in range(total)]
    assert full[335:] == [schedule.at_step(s, total) for s in range(335, total)]


def test_legacy_schedule_arithmetic_is_unchanged():
    legacy = LRWarmupCosine(0.001, 0.000001, 0.16)
    assert legacy.at_step(2679, 16750)["learning_rate"] == 0.001
    assert legacy.at_step(16749, 16750)["learning_rate"] > 0.000001


@pytest.mark.parametrize("step,total", [(-1, 16750), (False, 16750), (0, 1), (0, 2)])
def test_endpoint_rejects_invalid_horizon(step, total):
    with pytest.raises(ValueError):
        EndpointLRWarmupCosine(0.001, 0.0002, 0.16).at_step(step, total)
