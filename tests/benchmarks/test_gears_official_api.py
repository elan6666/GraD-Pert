from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from benchmarks.gears.official_api import (
    GearsModelParameters,
    GearsOfficialModules,
    OfficialGearsAPI,
)


class MiniAdata:
    def __init__(self, expression: np.ndarray[Any, Any], obs: pd.DataFrame) -> None:
        self.X = expression
        self.obs = obs
        self.uns: dict[str, object] = {}

    def __getitem__(self, mask: object) -> MiniAdata:
        indices = np.asarray(mask, dtype=bool)
        return MiniAdata(self.X[indices], self.obs.loc[indices].copy())

    def write_h5ad(self, path: str | Path) -> None:
        Path(path).write_bytes(b"fixture")


class FakePertData:
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        self.calls: list[tuple[str, object]] = []
        self.dataloader: dict[str, object] = {}
        self.dataset_path = str(Path(data_root) / "fixture")
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)

    def new_data_process(
        self,
        dataset_id: str,
        adata: object,
        skip_calc_de: bool,
    ) -> None:
        self.calls.append(("new_data_process", (dataset_id, adata, skip_calc_de)))
        self.adata = adata

    def prepare_split(self, **kwargs: object) -> None:
        self.calls.append(("prepare_split", kwargs))

    def get_dataloader(self, **kwargs: object) -> None:
        self.calls.append(("get_dataloader", kwargs))
        self.dataloader = {
            "train_loader": object(),
            "val_loader": object(),
            "test_loader": object(),
        }


class FakeTensor:
    def __init__(self, value: np.ndarray[Any, Any]) -> None:
        self.value = value

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.value


class FakeBatch:
    def __init__(self, controls: np.ndarray[Any, Any]) -> None:
        self.controls = controls
        self.device: str | None = None

    def to(self, device: str) -> FakeBatch:
        self.device = device
        return self


class FakeOfficialModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.eval_called = False

    def to(self, device: str) -> FakeOfficialModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, batch: FakeBatch) -> FakeTensor:
        return FakeTensor(batch.controls + 1.0)


class FakeGears:
    instances: ClassVar[list[FakeGears]] = []

    def __init__(self, pert_data: FakePertData, **kwargs: object) -> None:
        self.pert_data = pert_data
        self.init_kwargs = kwargs
        self.model_kwargs: dict[str, object] | None = None
        self.train_kwargs: dict[str, object] | None = None
        self.saved_path: str | None = None
        self.pert_list = np.asarray(["A", "B"])
        self.device = "cuda"
        self.best_model = FakeOfficialModel()
        self.instances.append(self)

    def model_initialize(self, **kwargs: object) -> None:
        self.model_kwargs = kwargs

    def train(self, **kwargs: object) -> None:
        self.train_kwargs = kwargs

    def save_model(self, path: str) -> None:
        self.saved_path = path


class FakeLoader:
    def __init__(
        self,
        graphs: list[np.ndarray[Any, Any]],
        batch_size: int,
        shuffle: bool,
    ) -> None:
        assert batch_size == 128
        assert not shuffle
        self.graphs = graphs

    def __iter__(self):
        yield FakeBatch(np.vstack(self.graphs[:128]))
        yield FakeBatch(np.vstack(self.graphs[128:256]))
        yield FakeBatch(np.vstack(self.graphs[256:]))


def _modules() -> GearsOfficialModules:
    package = SimpleNamespace(PertData=FakePertData, GEARS=FakeGears)
    utils = SimpleNamespace(
        create_cell_graph_for_prediction=lambda row, pert_idx, genes: np.asarray(row)
    )
    torch = SimpleNamespace(no_grad=nullcontext)
    pyg_loader = SimpleNamespace(DataLoader=FakeLoader)
    return GearsOfficialModules(
        package=package,  # type: ignore[arg-type]
        utils=utils,  # type: ignore[arg-type]
        torch=torch,  # type: ignore[arg-type]
        pyg_loader=pyg_loader,  # type: ignore[arg-type]
    )


def _parameters() -> GearsModelParameters:
    return GearsModelParameters(
        hidden_size=64,
        num_go_gnn_layers=1,
        num_gene_gnn_layers=1,
        decoder_hidden_size=16,
        num_similar_genes_go_graph=20,
        num_similar_genes_co_express_graph=20,
        coexpress_threshold=0.4,
        uncertainty=False,
        uncertainty_reg=1.0,
        direction_lambda=0.1,
        no_perturb=False,
    )


def test_adapter_calls_official_data_train_checkpoint_apis(tmp_path: Path) -> None:
    FakeGears.instances.clear()
    api = OfficialGearsAPI(_modules())
    adata = MiniAdata(
        np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        pd.DataFrame(
            {
                "condition": ["ctrl", "A+ctrl", "A+ctrl"],
                "condition_name": ["K562_ctrl_1", "K562_A+ctrl_1", "K562_A+ctrl_1"],
            },
            index=["ctrl-0", "pert-0", "pert-1"],
        ),
    )
    pert_data = api.prepare_training_data(
        data_root=tmp_path / "data",
        dataset_id="fixture",
        training_validation_adata=adata,
        split_pickle_path=tmp_path / "split.pkl",
        train_batch_size=32,
        eval_batch_size=128,
    )

    assert [name for name, _ in pert_data.calls] == [
        "new_data_process",
        "prepare_split",
        "get_dataloader",
    ]
    assert pert_data.calls[0][1][2] is False
    assert "test_loader" not in pert_data.dataloader

    model = api.fit_one_epoch(
        pert_data=pert_data,
        parameters=_parameters(),
        learning_rate=0.001,
        weight_decay=0.0005,
        checkpoint_dir=tmp_path / "checkpoint",
        device="cuda",
        experiment_name="fixture",
    )

    assert model.model_kwargs == _parameters().official_kwargs()
    assert model.train_kwargs == {"epochs": 1, "lr": 0.001, "weight_decay": 0.0005}
    assert model.saved_path == str(tmp_path / "checkpoint")


def test_exact_control_forward_preserves_all_300_rows() -> None:
    FakeGears.instances.clear()
    api = OfficialGearsAPI(_modules())
    model = FakeGears(FakePertData("fixture"))
    controls = np.arange(900, dtype=np.float32).reshape(300, 3)

    prediction = api.predict_exact_controls(
        trained_model=model,
        perturbation_genes=["A"],
        input_controls=controls,
        batch_size=128,
    )

    np.testing.assert_allclose(prediction, controls + 1.0)
    assert prediction.shape == (300, 3)
    assert model.best_model.eval_called
