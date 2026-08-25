from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

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

    def fit(self, model: object, train_dataloaders: object) -> None:
        self.fit_args = (model, train_dataloaders)

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

    official_training_loader = object()
    loader_calls: list[str] = []

    def train_dataloader() -> object:
        loader_calls.append("official_train_dataloader")
        return official_training_loader

    training_data = SimpleNamespace(train_dataloader=train_dataloader)
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
    assert trainer.fit_args == (model, official_training_loader)
    assert loader_calls == ["official_train_dataloader"]
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


def test_training_control_rows_use_official_numeric_control_id() -> None:
    api = _api()
    original = pd.Series(
        [[4, -1], ["ctrl"], [7, -1]],
        index=["treated-a", "control", "treated-b"],
        dtype=object,
    )
    data_module = SimpleNamespace(
        pert2id={"A": 4, "B": 7, "ctrl": -1},
        train_data=SimpleNamespace(pert_conditions=original.copy(deep=True)),
    )

    receipt = api.normalize_training_perturbation_indices(data_module)

    assert data_module.train_data.pert_conditions.tolist() == [[4, -1], [-1], [7, -1]]
    assert data_module.train_data.pert_conditions.index.tolist() == original.index.tolist()
    assert receipt["policy"] == "map_official_control_label_to_official_numeric_id"
    assert receipt["official_control_id"] == -1
    assert receipt["converted_condition_count"] == 1
    assert receipt["converted_component_count"] == 1
    assert receipt["before_sha256"] != receipt["after_sha256"]
    assert receipt["all_components_numeric_after"] is True
    perturbation_tensor = np.arange(8, dtype=np.float32).reshape(8, 1)
    for condition in data_module.train_data.pert_conditions:
        for component in condition:
            assert perturbation_tensor[component].shape == (1,)


@pytest.mark.parametrize("invalid_component", ["A", 99, True, None])
def test_training_index_adapter_rejects_noncontrol_or_unknown_components(
    invalid_component: object,
) -> None:
    api = _api()
    conditions = pd.Series([["ctrl"], [invalid_component]], dtype=object)
    data_module = SimpleNamespace(
        pert2id={"A": 4, "ctrl": -1},
        train_data=SimpleNamespace(pert_conditions=conditions.copy(deep=True)),
    )

    with pytest.raises(ValueError, match=r"unsupported component|unknown numeric ID"):
        api.normalize_training_perturbation_indices(data_module)

    assert data_module.train_data.pert_conditions.tolist() == conditions.tolist()


def test_training_index_adapter_requires_observed_control_rows() -> None:
    api = _api()
    conditions = pd.Series([[4, -1], [4, -1]], dtype=object)
    data_module = SimpleNamespace(
        pert2id={"A": 4, "ctrl": -1},
        train_data=SimpleNamespace(pert_conditions=conditions.copy(deep=True)),
    )

    with pytest.raises(ValueError, match="no control-label rows"):
        api.normalize_training_perturbation_indices(data_module)

    assert data_module.train_data.pert_conditions.tolist() == conditions.tolist()


@pytest.mark.parametrize("control_id", ["ctrl", True, None])
def test_training_index_adapter_requires_numeric_official_control_id(
    control_id: object,
) -> None:
    api = _api()
    conditions = pd.Series([["ctrl"]], dtype=object)
    data_module = SimpleNamespace(
        pert2id={"A": 4, "ctrl": control_id},
        train_data=SimpleNamespace(pert_conditions=conditions.copy(deep=True)),
    )

    with pytest.raises(ValueError, match="lacks a numeric control ID"):
        api.normalize_training_perturbation_indices(data_module)

    assert data_module.train_data.pert_conditions.tolist() == conditions.tolist()


@pytest.mark.parametrize("condition", [[], "ctrl", None])
def test_training_index_adapter_rejects_malformed_conditions(condition: object) -> None:
    api = _api()
    conditions = pd.Series([["ctrl"], condition], dtype=object)
    data_module = SimpleNamespace(
        pert2id={"A": 4, "ctrl": -1},
        train_data=SimpleNamespace(pert_conditions=conditions.copy(deep=True)),
    )

    with pytest.raises(ValueError, match="must be a non-empty list or tuple"):
        api.normalize_training_perturbation_indices(data_module)

    assert data_module.train_data.pert_conditions.tolist() == conditions.tolist()
