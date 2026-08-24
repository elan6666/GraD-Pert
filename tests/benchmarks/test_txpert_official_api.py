from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np

from benchmarks.txpert.official_api import OfficialPublicAPI, OfficialPublicModules


class FakeTensor:
    def __init__(self, value: np.ndarray[Any, Any]) -> None:
        self.value = value

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.value


class FakeGraph:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls.append(kwargs)


class FakePredictor:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.eval_called = False
        self.device = "cuda"
        self.calls.append(kwargs)

    def to(self, device: str) -> FakePredictor:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def forward(
        self,
        control: FakeTensor,
        perturbation_batch: tuple[tuple[int, ...], ...],
        embedding: FakeTensor,
    ) -> tuple[FakeTensor, None, None]:
        assert all(item == (4,) for item in perturbation_batch)
        assert embedding.value.shape == (control.value.shape[0], 2)
        return FakeTensor(control.value + 2.0), None, None


class FakeTrainer:
    instances: ClassVar[list[FakeTrainer]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fit_args: tuple[object, object] | None = None
        self.checkpoint_path: str | None = None
        self.instances.append(self)

    def fit(self, model: object, datamodule: object) -> None:
        self.fit_args = (model, datamodule)

    def save_checkpoint(self, path: str) -> None:
        self.checkpoint_path = path


class FakeTorch:
    float32 = np.float32
    no_grad = staticmethod(nullcontext)

    @staticmethod
    def as_tensor(
        value: np.ndarray[Any, Any],
        dtype: object,
        device: object,
    ) -> FakeTensor:
        return FakeTensor(np.asarray(value, dtype=dtype))

    @staticmethod
    def zeros(shape: tuple[int, int], dtype: object, device: object) -> FakeTensor:
        return FakeTensor(np.zeros(shape, dtype=dtype))


def _api() -> OfficialPublicAPI:
    return OfficialPublicAPI(
        OfficialPublicModules(
            predictor=SimpleNamespace(PertPredictor=FakePredictor),  # type: ignore[arg-type]
            datamodule=SimpleNamespace(PertDataModule=object),  # type: ignore[arg-type]
            graphmodule=SimpleNamespace(GSPGraph=FakeGraph),  # type: ignore[arg-type]
            lightning=SimpleNamespace(Trainer=FakeTrainer),  # type: ignore[arg-type]
            torch=FakeTorch,  # type: ignore[arg-type]
        )
    )


def test_build_and_fit_delegate_to_official_graph_model_and_lightning(tmp_path: Path) -> None:
    FakeGraph.calls.clear()
    FakePredictor.calls.clear()
    FakeTrainer.instances.clear()
    api = _api()
    data = SimpleNamespace(
        pert2id={"A": 4},
        gene2id={"A": 0, "B": 1},
        input_dim=2,
        output_dim=2,
        adata_output_dim=2,
    )
    model_args = {"model_type": "txpert", "hidden_dim": 512}
    graph_args = {"graph_cfg": {"graph0": {"graph_type": "string"}}}

    model = api.build_model(
        data_module=data,
        model_args=model_args,
        graph_args=graph_args,
        learning_rate=0.001,
        weight_decay=0.0,
        device="cuda",
        match_control_for_eval=True,
    )
    assert model_args == {"model_type": "txpert", "hidden_dim": 512}
    assert graph_args == {"graph_cfg": {"graph0": {"graph_type": "string"}}}
    assert FakePredictor.calls[0]["model_args"] == model_args

    training_data = object()
    checkpoint = tmp_path / "smoke.ckpt"
    trainer = api.fit_one_epoch(
        model=model,
        training_only_data_module=training_data,
        checkpoint_path=checkpoint,
        accelerator="gpu",
    )
    assert trainer.kwargs["max_epochs"] == 1
    assert trainer.kwargs["limit_val_batches"] == 0
    assert trainer.kwargs["num_sanity_val_steps"] == 0
    assert trainer.fit_args == (model, training_data)
    assert trainer.checkpoint_path == str(checkpoint)


def test_exact_control_forward_retains_300_rows() -> None:
    api = _api()
    model = FakePredictor()
    controls = np.arange(900, dtype=np.float32).reshape(300, 3)

    prediction = api.predict_exact_controls(
        trained_model=model,
        perturbation_genes=["A"],
        perturbation_to_id={"A": 4},
        input_controls=controls,
        batch_size=64,
    )

    np.testing.assert_allclose(prediction, controls + 2.0)
    assert prediction.shape == (300, 3)
    assert model.eval_called
