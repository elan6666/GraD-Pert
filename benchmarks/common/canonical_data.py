"""Canonical train/validation adapters shared by isolated official runners."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData


@dataclass(frozen=True)
class AdaptedCanonicalData:
    adata: Any
    train_conditions: tuple[str, ...]
    val_conditions: tuple[str, ...]
    expression_gene_count: int
    receipt: dict[str, object]


def build_training_validation_adata(
    training_data: CanonicalTrainingData,
    *,
    axis: Literal["expression", "graph"],
) -> AdaptedCanonicalData:
    """Copy controls plus canonical train/val rows; canonical test rows are excluded."""

    train = set(training_data.split.train_conditions)
    val = set(training_data.split.val_conditions)
    selected = tuple(
        index
        for index, condition in enumerate(training_data.condition_ids)
        if condition == training_data.split.control_condition_id
        or condition in train
        or condition in val
    )
    selected_conditions = {training_data.condition_ids[index] for index in selected}
    expected = {training_data.split.control_condition_id, *train, *val}
    if selected_conditions != expected:
        raise ValueError(
            "adapted official data does not contain exact train/val/control conditions"
        )
    if any(
        training_data.condition_ids[index] in training_data.split.test_conditions
        for index in selected
    ):
        raise ValueError("adapted official data contains canonical test truth")
    gene_count = (
        training_data.manifest.n_expression_genes
        if axis == "expression"
        else training_data.manifest.n_graph_genes
    )
    source = training_data._adata[np.asarray(selected), :gene_count].to_memory()
    source.obs_names = [training_data.row_ids[index] for index in selected]
    source.obs["condition"] = [training_data.condition_ids[index] for index in selected]
    source.obs["cell_type"] = [
        training_data.context_ids[index].split("::", 1)[0] for index in selected
    ]
    source.obs["batch"] = [training_data.batch_ids[index] for index in selected]
    source.obs["control"] = np.asarray(
        [
            training_data.condition_ids[index] == training_data.split.control_condition_id
            for index in selected
        ],
        dtype=np.int8,
    )
    source.obs["dose_val"] = [
        "1+1" if len(condition.split("+")) == 2 and "ctrl" not in condition.split("+") else "1"
        for condition in source.obs["condition"].astype(str)
    ]
    source.obs["condition_name"] = [
        f"{cell_type}_{condition}_{dose}"
        for cell_type, condition, dose in zip(
            source.obs["cell_type"],
            source.obs["condition"],
            source.obs["dose_val"],
            strict=True,
        )
    ]
    gene_ids = (
        training_data.expression_gene_ids if axis == "expression" else training_data.graph_gene_ids
    )
    source.var_names = list(gene_ids)
    source.var["gene_name"] = list(gene_ids)
    receipt: dict[str, object] = {
        "schema_version": "official-canonical-adapter-v1",
        "dataset_id": training_data.manifest.dataset_id,
        "protocol_id": training_data.manifest.protocol_id,
        "axis": axis,
        "canonical_data_sha256": training_data.manifest.canonical_adata_sha256,
        "split_content_sha256": training_data.split.split_content_sha256,
        "train_conditions": list(training_data.split.train_conditions),
        "train_conditions_sha256": sha256_json(list(training_data.split.train_conditions)),
        "val_conditions": list(training_data.split.val_conditions),
        "val_conditions_sha256": sha256_json(list(training_data.split.val_conditions)),
        "canonical_test_conditions_excluded": list(training_data.split.test_conditions),
        "canonical_test_conditions_excluded_sha256": sha256_json(
            list(training_data.split.test_conditions)
        ),
        "row_count": len(selected),
        "gene_count": gene_count,
        "observation_ids_sha256": sha256_json(list(source.obs_names.astype(str))),
    }
    return AdaptedCanonicalData(
        adata=source,
        train_conditions=tuple(training_data.split.train_conditions),
        val_conditions=tuple(training_data.split.val_conditions),
        expression_gene_count=training_data.manifest.n_expression_genes,
        receipt=receipt,
    )


def write_pickle(path: str | Path, payload: object) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    return sha256_file(destination)


def write_adapter_receipt(path: str | Path, receipt: dict[str, object]) -> None:
    atomic_json(path, receipt)
