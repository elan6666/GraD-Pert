"""Freeze row-level four-fold splits for the verified TxPert cross-cell cache.

This is a new fixed-axis protocol. It does not reuse the K562-specific condition
lists or claim to reproduce the paper's fold-specific HVG selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

CELL_LINES = ("K562", "RPE1", "jurkat", "hepg2")
SOURCE_SHA256 = "1b557390148eba358304e43e0b239538d9ae0691b26ec843f41cf544960307a8"
POLICY = "txpert_cache_fixed_axis_leave_one_line_out_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def build_folds(
    cell_lines: np.ndarray,
    conditions: np.ndarray,
    controls: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Split source perturbation rows 90/10 within each line and condition.

    Every test-line perturbation is test-only. Controls are separate from
    supervised train/validation/test rows; target controls remain available.
    Row positions, rather than obs names, are authoritative because the source
    H5AD has nonunique observation names.
    """

    n = len(cell_lines)
    if len(conditions) != n or len(controls) != n:
        raise ValueError("observation columns must have equal length")
    if set(np.unique(cell_lines)) != set(CELL_LINES) | {"K562_adamson"}:
        raise ValueError("unexpected source cell-line set")
    if not np.isin(controls, [0, 1]).all():
        raise ValueError("control column must be binary")
    if np.any((controls == 1) != (conditions == "ctrl")):
        raise ValueError("control flag and condition disagree")

    folds: dict[str, dict[str, np.ndarray]] = {}
    positions = np.arange(n, dtype=np.int64)
    for target in CELL_LINES:
        source = np.isin(cell_lines, [line for line in CELL_LINES if line != target])
        target_mask = cell_lines == target
        source_pert = positions[source & (controls == 0)]
        test = positions[target_mask & (controls == 0)]
        target_control = positions[target_mask & (controls == 1)]
        source_control = positions[source & (controls == 1)]
        if not len(test) or not len(target_control) or not len(source_pert):
            raise ValueError(f"empty data partition for {target}")

        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for line in CELL_LINES:
            if line == target:
                continue
            line_rows = source_pert[cell_lines[source_pert] == line]
            line_conditions = conditions[line_rows]
            order = np.argsort(line_conditions, kind="stable")
            grouped_rows = line_rows[order]
            grouped_conditions = line_conditions[order]
            boundaries = np.concatenate(
                (
                    [0],
                    np.flatnonzero(grouped_conditions[1:] != grouped_conditions[:-1]) + 1,
                    [len(order)],
                )
            )
            for start, stop in pairwise(boundaries):
                rows = grouped_rows[start:stop]
                condition = str(grouped_conditions[start])
                if len(rows) < 2:
                    train_parts.append(rows)
                    continue
                count = min(len(rows) - 1, max(1, round(len(rows) * 0.1)))
                rng_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}:{line}:{condition}".encode()).digest()[:8],
                    "little",
                )
                selected = np.random.default_rng(rng_seed).choice(
                    len(rows), size=count, replace=False
                )
                val_mask = np.zeros(len(rows), dtype=bool)
                val_mask[selected] = True
                train_parts.append(rows[~val_mask])
                val_parts.append(rows[val_mask])

        fold = {
            "train": np.sort(np.concatenate(train_parts)),
            "val": np.sort(np.concatenate(val_parts)),
            "test": test,
            "target_control": target_control,
            "source_control": source_control,
        }
        if not len(fold["val"]):
            raise ValueError(f"empty validation partition for {target}")
        if not np.array_equal(np.sort(np.concatenate((fold["train"], fold["val"]))), source_pert):
            raise ValueError(f"source perturbation rows are missing or repeated in {target}")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if np.intersect1d(fold[left], fold[right]).size:
                raise ValueError(f"overlapping {left}/{right} rows in {target}")
        folds[target] = fold
    return folds


def prepare(source: Path, output: Path, *, seed: int = 42) -> dict[str, object]:
    import anndata as ad

    if _sha256_file(source) != SOURCE_SHA256:
        raise ValueError("H5AD SHA256 does not match the audited official cache")
    adata = ad.read_h5ad(source, backed="r")
    try:
        obs = adata.obs
        lines = obs["cell_line"].astype(str).to_numpy()
        conditions = obs["condition"].astype(str).to_numpy()
        controls = obs["control"].astype(int).to_numpy()
        genes = list(map(str, adata.var_names))
        if len(genes) != 3352 or len(genes) != len(set(genes)):
            raise ValueError("unexpected fixed gene axis")
        folds = build_folds(lines, conditions, controls, seed=seed)
        output.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, object] = {
            "schema_version": "crosscell-fixed-axis-v1",
            "policy": POLICY,
            "source_sha256": SOURCE_SHA256,
            "source_rows": adata.n_obs,
            "gene_count": len(genes),
            "gene_order_sha256": _sha256_json(genes),
            "seed": seed,
            "validation_rule": (
                "10% of source perturbation rows per (cell_line, condition); "
                "minimum 1 and retain at least 1 train row"
            ),
            "excluded_cell_line": "K562_adamson",
            "folds": {},
        }
        for target, fold in folds.items():
            file = output / f"fold-{target}.npz"
            if file.exists():
                raise FileExistsError(f"refusing to overwrite {file}")
            np.savez_compressed(file, **fold)
            counts = {name: len(rows) for name, rows in fold.items()}
            seen = set(conditions[fold["train"]])
            test_conditions = set(conditions[fold["test"]])
            metadata["folds"][target] = {
                "row_index_file": file.name,
                "row_index_sha256": _sha256_file(file),
                "counts": counts,
                "test_condition_count": len(test_conditions),
                "test_seen_condition_count": len(test_conditions & seen),
                "test_unseen_condition_count": len(test_conditions - seen),
                "train_cell_lines": [line for line in CELL_LINES if line != target],
                "test_cell_line": target,
                "row_index_hashes": {
                    name: _sha256_json(rows.tolist()) for name, rows in fold.items()
                },
            }
        metadata_path = output / "manifest.json"
        if metadata_path.exists():
            raise FileExistsError(f"refusing to overwrite {metadata_path}")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    finally:
        adata.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
