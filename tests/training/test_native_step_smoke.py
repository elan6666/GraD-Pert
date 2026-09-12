from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

import gradpert.execution.native as native
from gradpert.training.step import GraDPertStepEngine
from gradpert.training.trainer import GraDPertTrainer
from scripts.server import native_step_smoke as smoke

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("failure", [False, True])
def test_one_step_stops_without_epoch_or_evaluation_and_restores_hooks(
    tmp_path, monkeypatch, failure
):
    @dataclass
    class Metrics:
        loss: float = 1.0

    steps = []
    engine = NS(model=torch.nn.Linear(1, 1), total_schedule_steps=5000)

    def train_step(self, batch, *, global_step):
        steps.append(global_step)
        if failure:
            raise ValueError("actual failure")
        return Metrics()

    def fit(self, **kwargs):
        GraDPertStepEngine.train_step(self.engine, None, global_step=0)
        pytest.fail("second step or epoch boundary must not execute")

    monkeypatch.setattr(GraDPertStepEngine, "train_step", train_step)
    monkeypatch.setattr(GraDPertTrainer, "fit", fit)
    monkeypatch.setattr(smoke, "_exact_engine_state", lambda e: {"model": "digest"})
    monkeypatch.setattr(smoke, "_checkpoint_roundtrip", lambda *a: "checkpoint-digest")
    root = tmp_path / "fresh"

    def run(**kwargs):
        small = root / "small_results"
        small.mkdir(parents=True)
        (small / "source_identity.json").write_text("{}")
        with native.CanonicalEvaluationData(split_name="val") as data:
            assert data.configure_expression_cache(enabled=True) == 0
        trainer = NS(identity="identity", engine=engine, max_epochs=50, steps_per_epoch=100)
        GraDPertTrainer.fit(trainer, validate=lambda *a: pytest.fail("validation"))

    monkeypatch.setattr(native, "run_native_experiment", run)
    original_eval = native.CanonicalEvaluationData
    kwargs = dict(
        config_path=ROOT / "configs/r50/g1_muon/gradpert_b2/nadig_jurkat.yaml",
        formal=True,
        mode="full",
        run_root=root,
    )
    if failure:
        with pytest.raises(ValueError, match="actual failure"):
            smoke.run_native_step_smoke(**kwargs)
        assert not (root / "small_results/one_step_smoke.json").exists()
    else:
        receipt = smoke.run_native_step_smoke(**kwargs)
        assert receipt["steps"][0]["completed_epochs"] == 0
        assert receipt["steps"][0]["completed_steps"] == 1
        assert receipt["steps"][0]["schedule_horizon_steps"] == 5000
        assert not receipt["scientific_completion"]
        assert receipt["test_evaluations"] == 0
    assert steps == [0]
    assert GraDPertTrainer.fit is fit
    assert GraDPertStepEngine.train_step is train_step
    assert native.CanonicalEvaluationData is original_eval
