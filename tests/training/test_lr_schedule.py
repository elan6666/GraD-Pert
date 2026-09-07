import ast
import math
import os
from pathlib import Path

import pytest

from gradpert.config.loader import load_experiment_config
from gradpert.config.lr_schedule import EpochWarmupCosineRestarts


def schedule():
    return EpochWarmupCosineRestarts(15, 2, 1e-4, 1e-6, 5, 0.9)


def test_epoch_boundaries_and_decaying_restart_peaks():
    s = schedule()
    for epoch, cycle, length, rate in (
        (0, 0, 15, 1e-6),
        (5, 0, 15, 1e-4),
        (15, 1, 25, 1e-6),
        (20, 1, 25, 9e-5),
        (40, 2, 45, 1e-6),
        (45, 2, 45, 8.1e-5),
        (85, 3, 85, 1e-6),
        (90, 3, 85, 7.29e-5),
    ):
        row = s.at_epoch(epoch)
        assert (row["cycle"], row["cycle_length"]) == (cycle, length)
        assert row["learning_rate"] == pytest.approx(rate, abs=1e-18)
    assert s.at_epoch(14)["learning_rate"] > 1e-6
    assert [s.at_epoch(i)["learning_rate"] for i in range(6)] == sorted(
        s.at_epoch(i)["learning_rate"] for i in range(6)
    )


def test_explicit_config_and_unchanged_architecture():
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    old = load_experiment_config(root / "a3_a0_e3_lr5e5_batch128/gradpert_b2/nadig_jurkat.yaml")
    new = load_experiment_config(
        root / "a3_a0_e3_sclong_schedule_batch128/gradpert_b2/nadig_jurkat.yaml"
    )
    assert old.model == new.model and old.data == new.data and old.evaluation == new.evaluation
    assert new.training.learning_rate.value == 1e-4
    assert EpochWarmupCosineRestarts.from_config(new.training.scheduler.value) == schedule()
    assert EpochWarmupCosineRestarts.from_config("none") is None


@pytest.mark.parametrize("epoch", [-1, 1.5, True])
def test_invalid_epoch_rejected(epoch):
    with pytest.raises(ValueError):
        schedule().at_epoch(epoch)


def test_exact_official_sequential_step_parity_when_reference_available():
    torch = pytest.importorskip("torch")
    reference = os.environ.get("SCLONG_SCHEDULER_REFERENCE")
    if reference is None:
        pytest.skip("optional frozen scLong scheduler reference is not configured")
    source = Path(reference).read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "CosineAnnealingWarmupRestarts"
    )
    namespace = {
        "torch": torch,
        "math": math,
        "_LRScheduler": torch.optim.lr_scheduler._LRScheduler,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), reference, "exec"), namespace)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    reference_scheduler = namespace["CosineAnnealingWarmupRestarts"](
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=1e-4,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9,
    )
    for epoch in range(200):
        assert optimizer.param_groups[0]["lr"] == schedule().at_epoch(epoch)["learning_rate"]
        reference_scheduler.step()
