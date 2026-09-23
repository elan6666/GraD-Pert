"""Observed-only matched-batch distribution and cell-count audit.

Each line gets a seeded, cell-count-stratified sample of up to 120 perturbation
conditions. E-distance is computed in a fixed 96-dimensional Gaussian random
projection of the full expression axis, with 99 within-batch label permutations.
It is a bounded distribution diagnostic, not the exact scPerturb E-test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

from gradpert.hashing import sha256_file
from scripts.analysis.six_dataset_landscape import DATASETS, _median, _pearson

PROJECTION_DIM = 96
MAX_CONDITIONS_PER_LINE = 120
MAX_CELLS_PER_ARM = 80
MIN_CELLS_PER_ARM = 20
PERMUTATIONS = 99


def _seed(base: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{key}".encode()).digest()[:8], "little")


def _energy_distance(distance: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Unbiased two-sample energy statistic; finite-sample values may be negative."""
    n, m = len(left), len(right)
    if n < 2 or m < 2:
        return float("nan")
    x = distance[np.ix_(left, left)]
    y = distance[np.ix_(right, right)]
    xy = distance[np.ix_(left, right)]
    return float(2 * xy.mean() - x.sum() / (n * (n - 1)) - y.sum() / (m * (m - 1)))


def _bh_fdr(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    q = np.empty(len(values))
    ranked = np.minimum(1.0, values[order] * len(values) / np.arange(1, len(values) + 1))
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return q.tolist()


def _select_stratified(
    eligible: list[tuple[str, str, int]], *, seed: int
) -> list[tuple[str, str, int]]:
    if len(eligible) <= MAX_CONDITIONS_PER_LINE:
        return eligible
    counts = np.array([item[2] for item in eligible])
    edges = np.quantile(counts, [0.25, 0.5, 0.75])
    bins = np.searchsorted(edges, counts, side="right")
    rng = np.random.default_rng(seed)
    chosen = []
    for bucket in range(4):
        members = np.flatnonzero(bins == bucket)
        take = min(30, len(members))
        chosen.extend(rng.choice(members, take, replace=False).tolist())
    if len(chosen) < MAX_CONDITIONS_PER_LINE:
        remaining = np.setdiff1d(np.arange(len(eligible)), chosen)
        take = min(MAX_CONDITIONS_PER_LINE - len(chosen), len(remaining))
        chosen.extend(rng.choice(remaining, take, replace=False).tolist())
    return [eligible[i] for i in sorted(chosen)]


def _project_selected(adata: object, rows: np.ndarray, *, genes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(genes, PROJECTION_DIM)).astype(np.float32)
    projection /= np.sqrt(PROJECTION_DIM)
    output = np.empty((len(rows), PROJECTION_DIM), dtype=np.float32)
    for start in range(0, len(rows), 512):
        end = min(start + 512, len(rows))
        expression = adata.X[rows[start:end], :genes]
        projected = expression @ projection
        if sparse.issparse(projected):
            projected = projected.toarray()
        output[start:end] = np.asarray(projected, dtype=np.float32)
    return output


def _analyze(path: Path, *, genes: int, crosscell: bool, seed: int) -> dict:
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        obs = data.obs
        lines = obs["cell_line" if crosscell else "cell_type"].astype(str).to_numpy()
        conditions = obs["condition"].astype(str).to_numpy()
        batches = obs["batch"].astype(str).to_numpy()
        controls = obs["control"].astype(int).to_numpy().astype(bool)
        valid = lines != "K562_adamson" if crosscell else np.ones(data.n_obs, dtype=bool)
        result = {}
        for line in sorted(set(lines[valid])):
            line_rows = np.flatnonzero(valid & (lines == line))
            by_condition_batch: dict[str, dict[str, list[int]]] = defaultdict(
                lambda: defaultdict(list)
            )
            control_batches: dict[str, list[int]] = defaultdict(list)
            for row in line_rows:
                if controls[row]:
                    control_batches[batches[row]].append(int(row))
                else:
                    by_condition_batch[conditions[row]][batches[row]].append(int(row))
            eligible = []
            for condition, groups in by_condition_batch.items():
                matches = [
                    (min(len(indices), len(control_batches[batch])), batch)
                    for batch, indices in groups.items()
                    if batch in control_batches
                ]
                if not matches:
                    continue
                pairable, best_batch = max(matches, key=lambda item: (item[0], item[1]))
                if pairable >= MIN_CELLS_PER_ARM:
                    eligible.append((condition, best_batch, pairable))
            eligible.sort()
            chosen = _select_stratified(eligible, seed=_seed(seed, line))
            samples = {}
            all_rows = set()
            for condition, batch, pairable in chosen:
                count = min(MAX_CELLS_PER_ARM, pairable)
                rng = np.random.default_rng(_seed(seed, f"{line}:{condition}:{batch}"))
                perturbed = np.sort(
                    rng.choice(by_condition_batch[condition][batch], count, replace=False)
                )
                control = np.sort(rng.choice(control_batches[batch], count, replace=False))
                samples[condition] = (perturbed, control, batch)
                all_rows.update(perturbed.tolist())
                all_rows.update(control.tolist())
            if not samples:
                result[line] = {"eligible_conditions": len(eligible), "analyzed_conditions": 0}
                continue
            ordered_rows = np.asarray(sorted(all_rows), dtype=np.int64)
            projected = _project_selected(
                data, ordered_rows, genes=genes, seed=_seed(seed, f"projection:{line}")
            )
            positions = {int(row): i for i, row in enumerate(ordered_rows)}
            details = []
            curves: dict[int, list[float]] = defaultdict(list)
            for condition, (perturbed, control, batch) in samples.items():
                p = projected[[positions[int(row)] for row in perturbed]]
                c = projected[[positions[int(row)] for row in control]]
                count = min(40, len(p))
                pooled = np.concatenate((p[:count], c[:count]))
                distance = cdist(pooled, pooled, metric="euclidean")
                left = np.arange(count)
                right = np.arange(count, 2 * count)
                observed = _energy_distance(distance, left, right)
                rng = np.random.default_rng(_seed(seed, f"null:{line}:{condition}:{batch}"))
                null = []
                for _ in range(PERMUTATIONS):
                    perm = rng.permutation(2 * count)
                    null.append(_energy_distance(distance, perm[:count], perm[count:]))
                p_value = (1 + sum(value >= observed for value in null)) / (PERMUTATIONS + 1)
                var_p = float(np.var(p, axis=0, ddof=1).sum())
                var_c = float(np.var(c, axis=0, ddof=1).sum())
                for n in (10, 20, 40):
                    if len(p) < 2 * n or len(c) < 2 * n:
                        continue
                    for _repeat in range(3):
                        pp = rng.permutation(len(p))[: 2 * n]
                        cc = rng.permutation(len(c))[: 2 * n]
                        first = p[pp[:n]].mean(axis=0) - c[cc[:n]].mean(axis=0)
                        second = p[pp[n:]].mean(axis=0) - c[cc[n:]].mean(axis=0)
                        curves[n].append(_pearson(first, second))
                details.append(
                    {
                        "condition": condition,
                        "matched_batch": batch,
                        "sampled_cells_per_arm": len(p),
                        "energy_rp96": observed,
                        "permutation_p": p_value,
                        "log2_perturbed_over_control_variance_trace": float(
                            np.log2(max(var_p, 1e-12) / max(var_c, 1e-12))
                        ),
                    }
                )
            q_values = _bh_fdr([entry["permutation_p"] for entry in details])
            for entry, q in zip(details, q_values, strict=True):
                entry["bh_q_within_sample"] = q
            result[line] = {
                "all_noncontrol_conditions": len(by_condition_batch),
                "eligible_conditions": len(eligible),
                "analyzed_conditions": len(details),
                "selection": "up to 30 from each pairable-cell-count quartile, then seeded fill",
                "minimum_pairable_cells": MIN_CELLS_PER_ARM,
                "matched_batch_rule": "single batch maximizing min(perturbed, control) rows",
                "median_energy_rp96": _median([x["energy_rp96"] for x in details]),
                "fraction_bh_q_below_0_1": float(np.mean(np.asarray(q_values) < 0.1)),
                "median_log2_variance_trace_ratio": _median(
                    [x["log2_perturbed_over_control_variance_trace"] for x in details]
                ),
                "equal_cell_split_half_curve": {
                    str(n): {
                        "comparisons": len(values),
                        "median_pearson_delta_rp96": _median(values),
                    }
                    for n, values in curves.items()
                },
                "condition_details": details,
            }
        return {"h5ad": str(path), "rows": data.n_obs, "genes": genes, "cell_lines": result}
    finally:
        data.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--crosscell-h5ad", type=Path, required=True)
    parser.add_argument("--crosscell-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("output must be a new directory under /data/yilangliu")
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        parser.error("source checkout must be clean")
    fold_path = args.crosscell_folds / "manifest.json"
    fold = json.loads(fold_path.read_text())
    if fold["source_sha256"] != sha256_file(args.crosscell_h5ad):
        parser.error("cross-cell H5AD differs from sealed folds")
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "source_commit": commit,
        "script_sha256": sha256_file(Path(__file__)),
        "crosscell_fold_manifest_sha256": sha256_file(fold_path),
        "projection_dim": PROJECTION_DIM,
        "permutations": PERMUTATIONS,
        "seed": args.seed,
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "datasets": {},
    }
    try:
        for name, protocol in DATASETS.items():
            root = args.data_root / name / protocol
            manifest = json.loads((root / "manifests/canonical.json").read_text())
            print(f"START {name}", flush=True)
            result = _analyze(
                root / "canonical/adata.h5ad",
                genes=manifest["n_expression_genes"],
                crosscell=False,
                seed=args.seed,
            )
            result["canonical_sha256"] = manifest["canonical_adata_sha256"]
            (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            receipt["datasets"][name] = sha256_file(args.output / f"{name}.json")
            print(f"DONE {name}", flush=True)
        print("START crosscell", flush=True)
        result = _analyze(
            args.crosscell_h5ad, genes=fold["gene_count"], crosscell=True, seed=args.seed
        )
        result["source_sha256"] = fold["source_sha256"]
        (args.output / "crosscell.json").write_text(json.dumps(result, indent=2) + "\n")
        receipt["datasets"]["crosscell"] = sha256_file(args.output / "crosscell.json")
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
