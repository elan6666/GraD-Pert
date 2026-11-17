"""Check external validation isolation without importing the official model."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmarks.txpert.official_api import OfficialPublicAPI
from benchmarks.txpert.released_recipe import WITHIN_CELL_EXPHORMER_MG


@pytest.mark.parametrize(
    "epochs,r50,recipe,completed,best_epoch",
    [
        (100, False, None, 11, 1),
        (50, True, None, 50, 1),
        (1, True, None, 1, 1),
        (50, True, WITHIN_CELL_EXPHORMER_MG, 41, 10),
    ],
)
def test_full_validation_never_calls_test_or_resets_optimizer(
    tmp_path, epochs, r50, recipe, completed, best_epoch
):
    events = []
    snapshots = {}

    class Model:
        device = "cpu"
        training = True
        state = 0

        def train(self, mode=True):
            self.training = mode

        def load_state_dict(self, state, strict):
            assert strict
            self.state = state["value"]

        def on_validation_epoch_end(self):
            raise AssertionError("unsafe upstream val/test hook must not execute")

    class Trainer:
        def __init__(self, **kwargs):
            assert kwargs["limit_val_batches"] == 0
            self.callback = kwargs["callbacks"][0]
            self.current_epoch = 0
            self.global_step = 0
            self.should_stop = False
            self.max_epochs = kwargs["max_epochs"]

        def fit(self, model, *, train_dataloaders):
            events.append("one_continuous_fit")
            self.model = model
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch
                self.global_step += 5
                model.state = epoch
                self.callback.on_train_epoch_end(self, model)
                if self.should_stop:
                    break

        def save_checkpoint(self, path):
            snapshots[path] = {"state_dict": {"value": self.model.state}}

    def evaluate(loader, model, device, adata, id2pert):
        assert loader == "val_only"
        events.append("validation")
        model.training = False
        return model.state

    # __init__ validates unrelated official construction symbols, so this unit
    # test injects only the already-audited fit dependencies.
    api = object.__new__(OfficialPublicAPI)
    api.modules = SimpleNamespace(
        predictor=SimpleNamespace(
            evaluate=evaluate,
            compute_metrics=lambda *a, **k: ({"pearson_delta": 0.5}, {}),
            cs=SimpleNamespace(FAST="fast"),
        ),
        torch=SimpleNamespace(no_grad=nullcontext, load=lambda path, **k: snapshots[path]),
        lightning=SimpleNamespace(Trainer=Trainer, Callback=object),
    )

    def forbidden():
        raise AssertionError("canonical test loader accessed")

    data = SimpleNamespace(
        train_dataloader=lambda: "train",
        val_dataloader=lambda: "val_only",
        test_dataloader=forbidden,
        adata=object(),
        id2pert={},
    )
    model = Model()
    trainer = api.fit_with_validation(
        model=model,
        training_only_data_module=data,
        checkpoint_path=tmp_path / "best.ckpt",
        accelerator="cpu",
        epochs=epochs,
        progress_path=tmp_path / "progress.json",
        r50=r50,
        last_checkpoint_path=tmp_path / "last.pt" if r50 else None,
        released_recipe=recipe,
    )
    assert events.count("one_continuous_fit") == 1
    assert events.count("validation") == completed
    assert trainer.gradpert_best_epoch == best_epoch
    assert model.state == best_epoch - 1
    assert model.training
    assert len(snapshots) == (2 if r50 else 1)
    if r50:
        assert snapshots[str(tmp_path / "last.pt")]["state_dict"]["value"] == completed - 1
        assert trainer.gradpert_last_epoch == completed
