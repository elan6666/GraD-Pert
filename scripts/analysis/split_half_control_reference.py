"""Server-only comparison of fixed-300 and complete compatible control baselines.

Use the same frozen gene axis, test conditions, and 20 perturbation half draws.
Only the control reference changes. Reuse the model evaluation's float32 pool
mean routine; condition aggregation is equally weighted, never cell weighted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gradpert.evaluation.state import _mean_rows
from gradpert.hashing import sha256_file
from scripts.analysis.five_dataset_diagnostics import DATASETS, split_half_assignments
from scripts.analysis.render_dataset_uncertainty import stats
from scripts.analysis.split_half_significance import (
    correlation_rows,
    hash_indices,
    write_compressed_public,
    write_csv,
)


def compatible_control_indices(control_contexts: np.ndarray, truth_contexts: tuple) -> np.ndarray:
    """Same context union and ordering as prepare_evaluation_state, without sampling."""
    if not truth_contexts:
        raise ValueError("empty truth contexts")
    groups = [np.flatnonzero(control_contexts == c) for c in sorted(set(truth_contexts))]
    if any(len(g) == 0 for g in groups):
        raise ValueError("truth context lacks controls")
    return np.concatenate(groups)


def analyze(data_root: Path, name: str, old_root: Path, output: Path) -> dict:
    import anndata as ad
    from numba import njit
    from scipy import sparse

    root = data_root / name / DATASETS[name]
    canonical = root / "canonical/adata.h5ad"
    splitpath = root / "manifests/split.json"
    ctrlpath = root / "manifests/evaluation_controls.test.json"
    manifest = json.loads((root / "manifests/canonical.json").read_text())
    split = json.loads(splitpath.read_text())
    draws_manifest = json.loads(ctrlpath.read_text())
    data = ad.read_h5ad(canonical, backed="r")
    try:
        genes = manifest["n_expression_genes"]
        conditions = data.obs.condition.astype(str).to_numpy()
        batches = data.obs.batch.astype(str).to_numpy()
        types = data.obs.cell_type.astype(str).to_numpy()
        contexts = np.char.add(np.char.add(types, "::"), batches)
        controls = data.obs.control.astype(bool).to_numpy()
        test = [c for c in split["test_conditions"] if c != split["control_condition_id"]]
        codes, halves, ordered = split_half_assignments(
            conditions, batches, test, repeats=20, seed=42
        )
        selected = np.flatnonzero((codes >= 0) | controls)
        lookup = np.full(data.n_obs, -1, dtype=int)
        lookup[selected] = np.arange(len(selected))
        expressions = np.empty((len(selected), genes), dtype=np.float32)
        for start in range(0, data.n_obs, 512):
            stop = min(start + 512, data.n_obs)
            mask = lookup[start:stop] >= 0
            if not mask.any():
                continue
            block = data.X[start:stop, :genes]
            if sparse.issparse(block):
                block = block.toarray()
            expressions[lookup[start:stop][mask]] = np.asarray(block, dtype=np.float32)[mask]
        ctrlrows = np.flatnonzero(controls)
        ctrlx = expressions[lookup[ctrlrows]]
        # Sparse/dense matrix semantics and float32 result match model evaluation.
        control_matrix = SimpleNamespace(X=data.X[ctrlrows, :genes])
        row_index = {str(row): i for i, row in enumerate(data.obs_names)}
        ctrl_index = {int(row): i for i, row in enumerate(ctrlrows)}
        draws = {
            entry["condition_id"]: [row_index[r] for r in entry["ordered_row_ids"]]
            for entry in draws_manifest["draws"]
        }
        frozen, complete, details = [], [], []
        cache = {}
        for c in ordered:
            draw = [ctrl_index[r] for r in draws[c]]
            if len(draw) != 300:
                raise ValueError("expected frozen 300-control draw")
            frozen.append(ctrlx[draw].mean(axis=0, dtype=float))
            truth_contexts = tuple(sorted(set(contexts[conditions == c])))
            if truth_contexts not in cache:
                indices = compatible_control_indices(contexts[ctrlrows], truth_contexts)
                cache[truth_contexts] = (_mean_rows(control_matrix, indices), indices)
            mean, indices = cache[truth_contexts]
            complete.append(mean)
            details.append(
                {
                    "compatible_context_count": len(truth_contexts),
                    "compatible_control_cells": len(indices),
                    "control_pool_row_indices_sha256": hash_indices(ctrlrows[indices]),
                }
            )
        frozen, complete = np.asarray(frozen), np.asarray(complete)
        sums = np.zeros((20, 2, len(ordered), genes), dtype=float)
        counts = np.zeros((20, 2, len(ordered)), dtype=np.int64)

        @njit
        def aggregate(values, row_codes, row_halves, totals, sizes):
            for row in range(len(values)):
                code = row_codes[row]
                if code < 0:
                    continue
                for repeat in range(20):
                    side = row_halves[repeat, row]
                    sizes[repeat, side, code] += 1
                    for gene in range(values.shape[1]):
                        totals[repeat, side, code, gene] += values[row, gene]

        aggregate(expressions, codes[selected], halves[:, selected], sums, counts)
        old = {
            (r["condition"], r["repeat"]): r["pearson_delta"]
            for r in json.loads((old_root / name / "split_half_conditions.json").read_text())
        }
        rows, differences = [], []
        for repeat in range(20):
            for code, c in enumerate(ordered):
                a, b = counts[repeat, :, code]
                row = {
                    "dataset": name,
                    "condition": c,
                    "repeat": repeat,
                    "seed": 42 + repeat,
                    "genes": genes,
                    "left_cells": int(a),
                    "right_cells": int(b),
                    "frozen_300": None,
                    "complete_compatible_pool": None,
                    "pool_minus_frozen": None,
                    **details[code],
                }
                previous = old[(c, repeat)]
                if min(a, b) == 0:
                    if previous is not None:
                        raise ValueError("historical missing-value mismatch")
                    row["unavailable_reason"] = "fewer_than_two_truth_cells"
                else:
                    left, right = sums[repeat, 0, code] / a, sums[repeat, 1, code] / b
                    original = float(correlation_rows(left - frozen[code], right - frozen[code]))
                    pooled = float(correlation_rows(left - complete[code], right - complete[code]))
                    if previous is None:
                        raise ValueError("historical availability mismatch")
                    differences.append(abs(original - previous))
                    row.update(
                        frozen_300=original,
                        complete_compatible_pool=pooled,
                        pool_minus_frozen=pooled - original,
                        unavailable_reason="",
                    )
                rows.append(row)
        if max(differences, default=0) > 1e-12:
            raise ValueError("frozen baseline parity failed")
        write_csv(output / f"{name}-control-reference-long.csv", rows)
        write_compressed_public(output / f"{name}-control-reference-long.csv.gz", rows)
        per_split = []
        for repeat in range(20):
            valid = [r for r in rows if r["repeat"] == repeat and r["frozen_300"] is not None]
            values = {
                key: [r[key] for r in valid] for key in ["frozen_300", "complete_compatible_pool"]
            }
            per_split.append(
                {
                    "repeat": repeat,
                    "seed": 42 + repeat,
                    "conditions": len(valid),
                    **{f"{key}_condition_mean": float(np.mean(val)) for key, val in values.items()},
                    **{
                        f"{key}_condition_median": float(np.median(val))
                        for key, val in values.items()
                    },
                }
            )
        summaries = {
            key: stats([r[key] for r in per_split])
            for key in per_split[0]
            if key not in ["repeat", "seed", "conditions"]
        }
        return {
            "dataset": name,
            "genes": genes,
            "test_conditions": len(ordered),
            "valid_conditions": per_split[0]["conditions"],
            "rows": len(rows),
            "missing_rows": sum(r["frozen_300"] is None for r in rows),
            "frozen_parity_max_abs": max(differences, default=0),
            "summary": summaries,
            "per_split": per_split,
            "complete_control_count_range": [
                min(d["compatible_control_cells"] for d in details),
                max(d["compatible_control_cells"] for d in details),
            ],
            "canonical_sha256": sha256_file(canonical),
            "split_sha256": sha256_file(splitpath),
            "controls_sha256": sha256_file(ctrlpath),
        }
    finally:
        data.file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--legacy-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(
        not p.resolve().is_relative_to(Path("/data/yilangliu"))
        for p in [args.data_root, args.old_root, args.output]
    ):
        raise ValueError("formal computation stays on server")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("source must be clean")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "running",
        "source_commit": commit,
        "source_clean": True,
        "started_unix": time.time(),
        "script_sha256": sha256_file(Path(__file__)),
        "config": {
            "repeats": 20,
            "seed": 42,
            "gene_axis": "all retained expression genes",
            "reference": "union of all control cells in full-condition cell_type::batch contexts",
            "pool_weighting": "control-cell weighted, not truth-frequency or batch equal",
            "reference_dtype": "float32 via existing evaluation.state._mean_rows",
            "shared_reference_between_halves": True,
        },
        "legacy_receipt_sha256": sha256_file(args.legacy_receipt),
        "thread_limits": {
            k: os.environ.get(k)
            for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "datasets": {},
    }
    receipt["config_sha256"] = hashlib.sha256(
        json.dumps(receipt["config"], sort_keys=True).encode()
    ).hexdigest()
    receipt["environment_versions"] = {
        package: importlib.metadata.version(package)
        for package in ["numpy", "anndata", "scipy", "numba"]
    }
    legacy = json.loads(args.legacy_receipt.read_text())
    try:
        for name in DATASETS:
            print("START", name, flush=True)
            path = args.data_root / name / DATASETS[name] / "canonical/adata.h5ad"
            if sha256_file(path) != legacy["datasets"][name]["canonical_adata_sha256"]:
                raise ValueError("canonical hash mismatch: " + name)
            receipt["datasets"][name] = analyze(args.data_root, name, args.old_root, args.output)
            (args.output / "manifest.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False)
            )
            print("DONE", name, flush=True)
        receipt.update(
            status="complete",
            completed_unix=time.time(),
            pkl_files=len(list(args.output.rglob("*.pkl"))),
        )
        if receipt["pkl_files"]:
            raise ValueError("zero-PKL failed")
        (args.output / "summary.json").write_text(
            json.dumps(receipt["datasets"], indent=2, allow_nan=False)
        )
        receipt["transfer_allowlist"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in args.output.iterdir()
            if p.suffix == ".gz" or p.name == "summary.json"
        }
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
