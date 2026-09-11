import ast
from pathlib import Path

import pytest
import yaml

from gradpert.config.loader import load_experiment_config
from gradpert.config.step_schedule import LRWarmupCosine, load_training_schedule

ROOT = Path(__file__).resolve().parents[2]


def test_only_scheduler_changes():
    base = ROOT / "configs/r50/ref/gradpert_b2/nadig_jurkat.yaml"
    new = ROOT / "configs/r50/sched/gradpert_b2/nadig_jurkat.yaml"
    cfg = load_experiment_config(new)
    a, b = yaml.safe_load(base.read_text()), yaml.safe_load(new.read_text())
    b["training"]["scheduler"] = a["training"]["scheduler"]
    b["artifacts"]["root"] = a["artifacts"]["root"]
    assert a == b
    schedule = load_training_schedule(cfg.training.scheduler.value)
    assert isinstance(schedule, LRWarmupCosine)
    assert schedule.at_step(0, 29100) == {"learning_rate": 0.0}
    assert schedule.at_step(4655, 29100)["learning_rate"] == 0.001
    assert schedule.at_step(4656, 29100)["learning_rate"] == 0.001
    assert schedule.at_step(29100, 29100)["learning_rate"] == 1e-6
    assert schedule.at_step(29099, 29100)["learning_rate"] == pytest.approx(1e-6, abs=1e-10)
    assert all(
        schedule.at_step(i, 29100)["learning_rate"]
        >= schedule.at_step(i + 1, 29100)["learning_rate"]
        for i in range(4656, 29099)
    )


def test_lr_only_does_not_override_teacher_ema():
    tree = ast.parse((ROOT / "src/gradpert/training/step.py").read_text())
    guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "momentum" for t in x.targets)
            for x in n.body
        )
    ]
    assert any(
        ast.unparse(n.test) == "isinstance(self.step_schedule, StepWarmupCosine)" for n in guards
    )
