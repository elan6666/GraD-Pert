from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from benchmarks.gears.official_api import OfficialGearsAPI
from benchmarks.scouter.official_api import fit_official
from benchmarks.txpert.official_api import OfficialPublicAPI
from gradpert.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]


def update_once(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(torch.ones(2, 1)).square().mean().backward()
    optimizer.step()
    pytest.fail("must stop before validation or second batch")


@pytest.mark.parametrize("name", ["scouter", "gears", "txpert"])
def test_official_api_single_step_does_not_run_epoch_validation(tmp_path, name):
    checkpoint = tmp_path / "step.pt"
    progress = tmp_path / "progress.json"
    if name == "scouter":

        class Model:
            def model_init(self, **kwargs):
                self.network = torch.nn.Linear(1, 1)
                self.loss_history = {"train_loss": [], "val_loss": []}

            def train(self, **kwargs):
                assert kwargs["n_epochs"] == 50
                update_once(self.network)

        model = Model()
        cfg = load_experiment_config(
            ROOT / "configs/r50-rerun/scouter_genept_seed/nadig_jurkat.yaml"
        )
        result = fit_official(
            model, cfg, epochs=50, r50=True, step_checkpoint_path=checkpoint, progress_path=progress
        )
    elif name == "gears":

        class Model:
            def __init__(self):
                self.dataloader = {}

            def model_initialize(self, **kwargs):
                self.model = torch.nn.Linear(1, 1)

            def train(self, **kwargs):
                assert kwargs["epochs"] == 50
                update_once(self.model)

        api = object.__new__(OfficialGearsAPI)
        api.modules = NS(package=NS(GEARS=lambda *a, **k: Model()), torch=torch)
        model = api.fit_one_epoch(
            pert_data=None,
            parameters=NS(official_kwargs=lambda: {}),
            learning_rate=0.001,
            weight_decay=0,
            checkpoint_dir=tmp_path / "best",
            device="cpu",
            experiment_name="step",
            epochs=50,
            r50=True,
            step_checkpoint_path=checkpoint,
            progress_path=progress,
        )
        result = model.gradpert_step_smoke
    else:

        class Trainer:
            def __init__(self, **kwargs):
                assert kwargs["max_epochs"] == 50
                assert kwargs["limit_val_batches"] == 0

            def fit(self, model, **kwargs):
                update_once(model)

        api = object.__new__(OfficialPublicAPI)
        api.modules = NS(
            predictor=NS(), torch=torch, lightning=NS(Trainer=Trainer, Callback=object)
        )
        trainer = api.fit_with_validation(
            model=torch.nn.Linear(1, 1),
            training_only_data_module=NS(train_dataloader=lambda: []),
            checkpoint_path=tmp_path / "best.ckpt",
            accelerator="cpu",
            epochs=50,
            r50=True,
            step_checkpoint_path=checkpoint,
            progress_path=progress,
        )
        result = trainer.gradpert_step_smoke
    assert result["completed_steps"] == 1
    assert result["completed_epochs"] == 0
    assert checkpoint.is_file()
    assert progress.is_file()
    assert not (tmp_path / "best").exists()
    assert not (tmp_path / "best.ckpt").exists()
