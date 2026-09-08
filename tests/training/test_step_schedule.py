from pathlib import Path

import numpy as np
import pytest

from gradpert.config.loader import load_experiment_config
from gradpert.config.step_schedule import StepWarmupCosine, load_training_schedule


def schedule(batch=1024):
    return StepWarmupCosine(2e-4, 1e-6, batch, 0.16, 0.994, 1.0)


def test_reference_array_arithmetic_and_endpoints():
    # Independently evaluated frozen reference np.linspace/np.arange definition.
    s = schedule()
    n, w = 1000, 160
    lr = np.concatenate(
        (
            np.linspace(0, 2e-4, w),
            1e-6 + 0.5 * (2e-4 - 1e-6) * (1 + np.cos(np.pi * np.arange(n - w) / (n - w))),
        )
    )
    ema = 1 + 0.5 * (0.994 - 1) * (1 + np.cos(np.pi * np.arange(n) / n))
    for i in range(n):
        assert s.at_step(i, n)["learning_rate"] == pytest.approx(lr[i], abs=1e-18)
        assert s.at_step(i, n)["teacher_momentum"] == pytest.approx(ema[i], abs=1e-15)
    assert s.at_step(0, n) == {"learning_rate": 0, "teacher_momentum": 0.994}
    assert s.at_step(n, n) == {"learning_rate": 1e-6, "teacher_momentum": 1.0}
    assert s.at_step(n - 1, n)["teacher_momentum"] < 1
    assert schedule(3072).max_lr == pytest.approx(2e-4 * np.sqrt(3))


def test_new_configs_only_requested_delta_and_pair():
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    configs = []
    for row, prior in (("b0", "_e3"), ("b1", "")):
        old = load_experiment_config(
            root / f"{row}_historical_b2{prior}_schedule_batch128/gradpert_b2/nadig_jurkat.yaml"
        )
        new = load_experiment_config(
            root / f"{row}_historical_b2{prior}_step_batch1024/gradpert_b2/nadig_jurkat.yaml"
        )
        assert new.training.train_batch_size.value == 1024
        assert new.training.eval_batch_size.value == 256
        assert new.training.learning_rate.value == 2e-4
        assert isinstance(load_training_schedule(new.training.scheduler.value), StepWarmupCosine)
        for key, value in old.model.parameters.items():
            assert new.model.parameters[key].value == (
                0.994 if key == "teacher_ema_start" else value.value
            )
        configs.append(new)
    assert configs[0].training == configs[1].training
    bad = configs[0].training.scheduler.value.copy()
    bad["warmup_fraction"] = 1
    with pytest.raises(ValueError):
        load_training_schedule(bad)
    with pytest.raises(ValueError):
        configs[0].training.__class__.model_validate(
            {
                **configs[0].training.model_dump(),
                "train_batch_size": {"value": 128, "source": "user_locked", "reference": "test"},
            }
        )


def test_tiny_schedule_and_invalid_indices():
    assert schedule().at_step(0, 1)["learning_rate"] == 2e-4
    with pytest.raises(ValueError):
        schedule().at_step(-1, 10)
