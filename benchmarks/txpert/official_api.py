"""Narrow calls to the frozen public package; no local model implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from gradpert.hashing import sha256_json


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

    @staticmethod
    def normalize_training_perturbation_indices(data_module: Any) -> dict[str, object]:
        """Encode official control-only training rows with the official numeric ID.

        The frozen data module emits numeric perturbation IDs for treatment rows,
        including ``-1`` for the embedded control component, but extends the
        training dataset with control-only rows represented as ``["ctrl"]``.
        The frozen model indexes a tensor with every component and therefore
        requires the same numeric control ID for those rows.  This adapter only
        translates that one official label through ``data_module.pert2id``; it
        rejects every other non-numeric or unknown component.
        """

        perturbation_to_id = data_module.pert2id
        control_label = "ctrl"
        control_id = perturbation_to_id.get(control_label)
        if not isinstance(control_id, int) or isinstance(control_id, bool):
            raise ValueError("official public data module lacks a numeric control ID")
        valid_ids = {
            int(value)
            for value in perturbation_to_id.values()
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        }
        conditions = data_module.train_data.pert_conditions
        if not callable(getattr(conditions, "tolist", None)) or not hasattr(conditions, "index"):
            raise TypeError("official training perturbation conditions must be a pandas Series")

        original_rows = conditions.tolist()
        original_hash_rows: list[list[int | str]] = []
        normalized_rows: list[list[int]] = []
        converted_condition_count = 0
        converted_component_count = 0
        component_count = 0
        for condition_index, row in enumerate(original_rows):
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError(
                    "official training perturbation condition "
                    f"{condition_index} must be a non-empty list or tuple"
                )
            normalized_row: list[int] = []
            original_hash_row: list[int | str] = []
            condition_converted = False
            for component in row:
                component_count += 1
                if isinstance(component, str):
                    original_hash_row.append(component)
                    if component != control_label:
                        raise ValueError(
                            "official training perturbation condition "
                            f"{condition_index} contains unsupported component {component!r}"
                        )
                    normalized_row.append(control_id)
                    converted_component_count += 1
                    condition_converted = True
                    continue
                if isinstance(component, (bool, np.bool_)) or not isinstance(
                    component, (int, np.integer)
                ):
                    raise ValueError(
                        "official training perturbation condition "
                        f"{condition_index} contains unsupported component {component!r}"
                    )
                numeric_component = int(component)
                if numeric_component not in valid_ids:
                    raise ValueError(
                        "official training perturbation condition "
                        f"{condition_index} contains unknown numeric ID {numeric_component}"
                    )
                original_hash_row.append(numeric_component)
                normalized_row.append(numeric_component)
            original_hash_rows.append(original_hash_row)
            normalized_rows.append(normalized_row)
            converted_condition_count += int(condition_converted)

        if converted_component_count == 0:
            raise ValueError("official training data contains no control-label rows to normalize")
        normalized = conditions.__class__(
            normalized_rows,
            index=conditions.index,
            dtype=object,
        )
        data_module.train_data.pert_conditions = normalized
        return {
            "schema_version": "txpert-training-index-adapter-v1",
            "policy": "map_official_control_label_to_official_numeric_id",
            "official_control_label": control_label,
            "official_control_id": control_id,
            "condition_count": len(normalized_rows),
            "component_count": component_count,
            "converted_condition_count": converted_condition_count,
            "converted_component_count": converted_component_count,
            "valid_official_id_count": len(valid_ids),
            "before_sha256": sha256_json(original_hash_rows),
            "after_sha256": sha256_json(normalized_rows),
            "all_components_numeric_after": True,
        }

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
        """Fit one epoch from the already-adapted official training loader.

        Passing the data module back to Lightning would call its ``setup`` a
        second time and replace the validated training dataset.  Materialize
        the frozen module's own loader after adaptation so the official
        collate function, shuffle, batching, training step, and optimizer are
        preserved without that destructive rebuild.
        """

        official_training_loader = training_only_data_module.train_dataloader()
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
        trainer.fit(model, train_dataloaders=official_training_loader)
        trainer.save_checkpoint(str(Path(checkpoint_path)))
        return trainer

    @staticmethod
    def restore_post_fit_device(model: Any, device: str) -> dict[str, object]:
        """Restore parameters moved to CPU by Lightning's fit teardown."""

        model.to(device)
        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
        if parameter_devices != [device]:
            raise RuntimeError(
                "official model parameters did not return to the requested inference device"
            )
        if buffer_devices and buffer_devices != [device]:
            raise RuntimeError(
                "official model buffers did not return to the requested inference device"
            )
        return {
            "schema_version": "txpert-post-fit-device-restore-v1",
            "policy": "official_module_to_requested_device_after_lightning_fit_teardown",
            "requested_device": device,
            "parameter_devices": parameter_devices,
            "buffer_devices": buffer_devices,
            "official_reference": "main.py: load_from_checkpoint(...).to(device)",
        }

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
