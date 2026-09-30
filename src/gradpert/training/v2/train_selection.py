"""Sealed train-only row subsets; canonical validation and test data stay intact."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData


def apply_train_selection(data: CanonicalTrainingData, path: Path, digest: str) -> dict[str, Any]:
    """Validate the complete frozen selection before changing the training indices.

    The manifest binds row IDs to the current canonical file, split and axes.
    The original analysis selection and equivalence receipt are separately sealed;
    no derivative H5AD is relabeled canonical and no cells are resampled here.
    """
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise ValueError("training row selection file/hash mismatch")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "gradpert-v2-training-selection-1":
        raise ValueError("unsupported training row selection schema")
    expected = {
        "dataset_id": data.manifest.dataset_id,
        "protocol_id": data.manifest.protocol_id,
        "parent_h5ad_sha256": data.manifest.canonical_adata_sha256,
        "split_content_sha256": data.split.split_content_sha256,
        "observation_order_sha256": data.manifest.observation_order_sha256,
        "expression_gene_order_sha256": data.manifest.expression_gene_order_sha256,
        "graph_gene_order_sha256": data.manifest.graph_gene_order_sha256,
        "state": "frozen_training_row_selection",
        "sampling": "condition_batch_largest_remainder_uniform_without_replacement",
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("training row selection canonical identity mismatch")
    cap, seed = selection.get("cap"), selection.get("seed")
    if type(cap) is not int or cap < 1 or type(seed) is not int or seed < 0:
        raise ValueError("training row selection cap/seed invalid")
    rows = selection.get("selected_train_row_ids")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, str) for row in rows)
        or len(set(rows)) != len(rows)
        or sha256_json(rows) != selection.get("selected_train_row_ids_sha256")
    ):
        raise ValueError("training row selection row IDs/hash invalid")
    original = tuple(data.train_row_indices)
    by_id = {data.row_ids[i]: i for i in original}
    if not set(rows) <= by_id.keys():
        raise ValueError("training row selection contains non-training rows")
    indices = tuple(by_id[row] for row in rows)
    if indices != tuple(sorted(indices)):
        raise ValueError("training row selection must preserve canonical row order")
    conditions = list(data.split.train_conditions)
    if selection.get("train_condition_ids") != conditions:
        raise ValueError("training row selection condition axis mismatch")
    source_counts = Counter(data.condition_ids[i] for i in original)
    selected_counts = Counter(data.condition_ids[i] for i in indices)
    if selected_counts != Counter({p: min(cap, n) for p, n in source_counts.items()}):
        raise ValueError("training row selection loses conditions or violates cap quotas")
    original_set = set(original)
    unchanged = [row for i, row in enumerate(data.row_ids) if i not in original_set]
    if sha256_json(unchanged) != selection.get("unmodified_row_ids_sha256"):
        raise ValueError("training row selection non-training axis mismatch")
    # Audit linkage remains on the server and is checked before any mutation.
    for name in ("analysis_selection", "equivalence"):
        audit_path = Path(selection[f"{name}_path"])
        if (
            audit_path.is_symlink()
            or not audit_path.is_file()
            or not audit_path.resolve().is_relative_to(path.resolve().parents[1])
            or sha256_file(audit_path) != selection[f"{name}_sha256"]
        ):
            raise ValueError(f"training row selection {name} evidence mismatch")
    source = json.loads(Path(selection["analysis_selection_path"]).read_text())
    proof = json.loads(Path(selection["equivalence_path"]).read_text())
    compared_hashes = {item["sha256"] for item in proof.get("files", [])}
    checks = proof.get("comparison", {})
    if (
        proof.get("schema_version") != "gradpert-canonical-equivalence-1"
        or not {source["parent_h5ad_sha256"], expected["parent_h5ad_sha256"]} <= compared_hashes
        or any(
            checks.get(name) is not True for name in ("X_values_equal", "obs_equal", "var_equal")
        )
    ):
        raise ValueError("training row selection canonical equivalence not verified")
    for name in (
        "selected_train_row_ids",
        "selected_train_row_ids_sha256",
        "train_condition_ids",
        "unmodified_row_ids_sha256",
        "cap",
        "seed",
        "sampling",
        "by_condition",
    ):
        if selection.get(name) != source.get(name):
            raise ValueError("training row selection differs from accepted analysis samples")
    receipt = {
        "schema_version": selection["schema_version"],
        "manifest_path": str(path),
        "manifest_sha256": digest,
        "cap": cap,
        "sampling_seed": seed,
        "sampling": selection["sampling"],
        "original_train_rows": len(original),
        "selected_train_rows": len(indices),
        "train_condition_count": len(source_counts),
        "selected_train_row_ids_sha256": selection["selected_train_row_ids_sha256"],
        "analysis_selection_sha256": selection["analysis_selection_sha256"],
        "analysis_source_commit": selection["analysis_source_commit"],
        "equivalence_sha256": selection["equivalence_sha256"],
        "non_training_rows_changed": 0,
    }
    data.train_row_indices = indices
    return receipt
