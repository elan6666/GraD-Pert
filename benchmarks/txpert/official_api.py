"""Narrow calls to the frozen public package; no local model implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OfficialPublicModules:
    predictor: ModuleType
    datamodule: ModuleType
    graphmodule: ModuleType
    lightning: ModuleType
    torch: ModuleType

    @classmethod
    def from_mapping(cls, modules: Mapping[str, ModuleType]) -> OfficialPublicModules:
        return cls(
            predictor=modules["gspp.predictor"],
            datamodule=modules["gspp.data.datamodule"],
            graphmodule=modules["gspp.data.graphmodule"],
            lightning=modules["lightning"],
            torch=modules["torch"],
        )


class OfficialPublicAPI:
    """Delegate graph/model/training/forward operations to the frozen package."""

    def __init__(self, modules: OfficialPublicModules) -> None:
        self.modules = modules
        required = (
            (modules.predictor, "PertPredictor"),
            (modules.datamodule, "PertDataModule"),
            (modules.graphmodule, "GSPGraph"),
            (modules.lightning, "Trainer"),
        )
        for module, symbol in required:
            if not hasattr(module, symbol):
                raise ValueError(f"official public module lacks {symbol}")

    def prepare_training_data_module(
        self,
        *,
        cache_path: str | Path,
        batch_size: int,
        cell_type: str,
    ) -> Any:
        """Instantiate the official data module on an external train/val-only cache."""

        official_raw = self.modules.datamodule.cs.ObsmKey.RAW
        data_module = self.modules.datamodule.PertDataModule(
            batch_size=int(batch_size),
            match_cntr=True,
            avg_cntr=True,
            embed_cntr=True,
            obsm_key=official_raw,
            task_type="gradpert_canonical_adapter",
            mode="baseline",
            train_cell_types=[cell_type],
            val_cell_type=None,
            test_cell_type=cell_type,
            suppress_cell_type_validation=True,
        )
        data_module.cache_path = Path(cache_path)
        data_module.prepare_data()
        data_module.setup("fit")
        return data_module

    @staticmethod
    def require_perturbation_coverage(
        data_module: Any,
        condition_ids: Sequence[str],
    ) -> tuple[str, ...]:
        targets = tuple(
            sorted(
                {
                    component
                    for condition in condition_ids
                    for component in condition.split("+")
                    if component != "ctrl"
                }
            )
        )
        missing = sorted(set(targets) - set(data_module.pert2id))
        if missing:
            raise ValueError(f"official public perturbation graph lacks targets: {missing}")
        observed = set(data_module.treatment_data.obs["condition"].astype(str).unique())
        expected_fit = set(data_module.condition_group["train"]) | set(
            data_module.condition_group["val"]
        )
        if observed != expected_fit:
            raise ValueError("official public data module filtered canonical train/val conditions")
        return targets

    def build_model(
        self,
        *,
        data_module: Any,
        model_args: Mapping[str, object],
        graph_args: Mapping[str, object],
        learning_rate: float,
        weight_decay: float,
        device: str,
        match_control_for_eval: bool,
    ) -> Any:
        """Instantiate the official graph and predictor from official config values."""

        graph = self.modules.graphmodule.GSPGraph(
            pert2id=data_module.pert2id,
            gene2id=data_module.gene2id,
            **deepcopy(dict(graph_args)),
        )
        return self.modules.predictor.PertPredictor(
            input_dim=int(data_module.input_dim),
            output_dim=int(data_module.output_dim),
            adata_output_dim=int(data_module.adata_output_dim),
            model_args=deepcopy(dict(model_args)),
            graph=graph,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            device=device,
            match_cntr_for_eval=bool(match_control_for_eval),
        ).to(device)

    def fit_one_epoch(
        self,
        *,
        model: Any,
        training_only_data_module: Any,
        checkpoint_path: str | Path,
        accelerator: str,
    ) -> Any:
        """Let Lightning run the official training_step/optimizer for exactly one epoch."""

        trainer = self.modules.lightning.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        trainer.fit(model, datamodule=training_only_data_module)
        trainer.save_checkpoint(str(Path(checkpoint_path)))
        return trainer

    def predict_exact_controls(
        self,
        *,
        trained_model: Any,
        perturbation_genes: Sequence[str],
        perturbation_to_id: Mapping[str, int],
        input_controls: np.ndarray[Any, Any],
        batch_size: int,
    ) -> np.ndarray[Any, Any]:
        """Call the official forward while retaining the exact 300 control rows."""

        controls = np.asarray(input_controls, dtype=np.float32)
        if controls.ndim != 2 or controls.shape[0] != 300:
            raise ValueError("official public prediction requires exact [300,G] controls")
        if not perturbation_genes:
            raise ValueError("prediction requires at least one perturbation gene")
        try:
            perturbation_ids = tuple(int(perturbation_to_id[gene]) for gene in perturbation_genes)
        except KeyError as exc:
            raise ValueError(f"perturbation is absent from official graph: {exc.args[0]}") from exc

        trained_model.eval()
        outputs: list[np.ndarray[Any, Any]] = []
        with self.modules.torch.no_grad():
            for start in range(0, controls.shape[0], int(batch_size)):
                batch = controls[start : start + int(batch_size)]
                control_tensor = self.modules.torch.as_tensor(
                    batch,
                    dtype=self.modules.torch.float32,
                    device=trained_model.device,
                )
                perturbation_batch = tuple(perturbation_ids for _ in range(batch.shape[0]))
                embedding = self.modules.torch.zeros(
                    (batch.shape[0], 2),
                    dtype=self.modules.torch.float32,
                    device=trained_model.device,
                )
                result = trained_model.forward(control_tensor, perturbation_batch, embedding)
                prediction = result[0] if isinstance(result, tuple) else result
                outputs.append(prediction.detach().cpu().numpy())
        result_array = np.asarray(np.vstack(outputs), dtype=np.float32)
        if result_array.shape != controls.shape or not np.isfinite(result_array).all():
            raise ValueError(
                f"official public prediction shape/finite gate failed: {result_array.shape}"
            )
        return result_array
