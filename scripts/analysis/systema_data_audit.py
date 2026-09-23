"""CPU-only Systema-inspired data audit on the five canonical datasets and cross-cell cache.

This describes observed data and a train-only perturbed-mean baseline. It never
selects a checkpoint or writes per-cell expression outside the source H5AD.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from gradpert.hashing import sha256_file
from scripts.analysis.six_dataset_landscape import (
    DATASETS,
    SEPARATOR,
    _aggregate_expression,
    _group_codes,
    _median,
    _pearson,
)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / norm) if norm > 1e-12 else float("nan")


def _centroid_accuracy(
    prediction: np.ndarray, truth: np.ndarray, *, max_conditions: int, seed: int
) -> dict:
    """Systema's per-condition fraction of other truth centroids farther away."""
    n = len(truth)
    if n > max_conditions:
        selected = np.sort(np.random.default_rng(seed).choice(n, max_conditions, replace=False))
        truth = truth[selected]
        n = len(truth)
    if n < 2:
        return {"conditions": n, "mean": None}
    squared = np.sum((truth - prediction) ** 2, axis=1)
    own = squared.copy()
    accuracy = np.empty(n)
    for i in range(n):
        accuracy[i] = np.mean(squared[np.arange(n) != i] > own[i])
    return {"conditions": n, "mean": float(np.mean(accuracy)), "median": float(np.median(accuracy))}


def _analyze(
    path: Path, *, genes: int, split: dict | None, crosscell: bool, seed: int, chunk: int
) -> dict:
    import anndata as ad

    data = ad.read_h5ad(path, backed="r")
    try:
        obs = data.obs
        line_key = "cell_line" if crosscell else "cell_type"
        lines = obs[line_key].astype(str).to_numpy()
        conditions = obs["condition"].astype(str).to_numpy()
        batches = obs["batch"].astype(str).to_numpy()
        controls = obs["control"].astype(int).to_numpy().astype(bool)
        valid = lines != "K562_adamson" if crosscell else np.ones(data.n_obs, dtype=bool)
        keys, codes = _group_codes(lines, conditions, batches, controls, valid, seed)
        sums, counts = _aggregate_expression(data, codes, 2 * len(keys), genes, chunk)
        grouped_sums = sums[0::2] + sums[1::2]
        grouped_counts = counts[0::2] + counts[1::2]

        control_sums: dict[str, np.ndarray] = {}
        control_counts: dict[str, int] = {}
        centroids: dict[str, dict[str, np.ndarray]] = {}
        sizes: dict[str, dict[str, int]] = {}
        halves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for i, key in enumerate(keys):
            parts = key.split(SEPARATOR)
            if len(parts) != 3:
                continue
            line = parts[0]
            control_sums[line] = control_sums.get(line, np.zeros(genes)) + grouped_sums[i]
            control_counts[line] = control_counts.get(line, 0) + int(grouped_counts[i])
        controls_by_line = {
            line: total / control_counts[line] for line, total in control_sums.items()
        }
        for i, key in enumerate(keys):
            parts = key.split(SEPARATOR)
            if len(parts) != 2 or grouped_counts[i] == 0:
                continue
            line, condition = parts
            centroids.setdefault(line, {})[condition] = grouped_sums[i] / grouped_counts[i]
            sizes.setdefault(line, {})[condition] = int(grouped_counts[i])
            if counts[2 * i] and counts[2 * i + 1]:
                halves.setdefault(line, {})[condition] = (
                    sums[2 * i] / counts[2 * i],
                    sums[2 * i + 1] / counts[2 * i + 1],
                )

        results = {}
        for line, profiles in centroids.items():
            control = controls_by_line[line]
            names = sorted(profiles)
            vectors = np.stack([profiles[name] - control for name in names])
            weights = np.array([sizes[line][name] for name in names])
            mean_shift = np.average(vectors, axis=0, weights=weights)
            cosines = np.array([_cosine(vector, mean_shift) for vector in vectors])
            results[line] = {
                "conditions": len(names),
                "systematic_variation_mean_cosine": float(np.nanmean(cosines)),
                "systematic_variation_median_cosine": _median(cosines.tolist()),
            }
            if split is None:
                # The fixed-axis folds are row-level cross-cell splits. A within-line
                # train mean would include the target test line and leak its truth.
                continue
            else:
                train_set = set(split["train_conditions"])
                test_set = set(split["test_conditions"])
                train_names = [name for name in names if name in train_set]
                test_names = [name for name in names if name in test_set]
            if not train_names or not test_names:
                raise ValueError(f"empty train or test conditions: {line}")
            train_mean = np.average(
                np.stack([profiles[name] for name in train_names]),
                axis=0,
                weights=[sizes[line][name] for name in train_names],
            )
            test_truth = np.stack([profiles[name] for name in test_names])
            baseline_delta = train_mean - control
            baseline_pearson = [_pearson(truth - control, baseline_delta) for truth in test_truth]
            # Systema's perturbed reference averages TRAIN condition centroids,
            # unlike its cell-weighted perturbed-mean predictor.
            train_reference = np.mean(np.stack([profiles[name] for name in train_names]), axis=0)
            baseline_perturbed_delta = train_mean - train_reference
            baseline_perturbed_pearson = [
                _pearson(truth - train_reference, baseline_perturbed_delta) for truth in test_truth
            ]
            oracle_pair = [
                (
                    _pearson(halves[line][name][0] - control, halves[line][name][1] - control),
                    _pearson(
                        halves[line][name][0] - train_reference,
                        halves[line][name][1] - train_reference,
                    ),
                )
                for name in test_names
                if name in halves.get(line, {})
            ]
            count_bins = {}
            for low, high in ((0, 20), (20, 50), (50, 100), (100, 10**9)):
                subset = [name for name in test_names if low <= sizes[line][name] < high]
                values = [
                    _pearson(halves[line][name][0] - control, halves[line][name][1] - control)
                    for name in subset
                    if name in halves.get(line, {})
                ]
                count_bins[f"{low}-{high if high < 10**9 else 'plus'}"] = {
                    "conditions": len(subset),
                    "split_half_pearson_delta_median": _median(values),
                }
            results[line].update(
                {
                    "test_conditions": len(test_names),
                    "train_conditions": len(train_names),
                    "train_perturbed_mean_baseline_control_pearson_delta_median": _median(
                        baseline_pearson
                    ),
                    "train_perturbed_mean_baseline_systema_pearson_delta_median": _median(
                        baseline_perturbed_pearson
                    ),
                    "oracle_split_half_control_pearson_delta_median": _median(
                        [x[0] for x in oracle_pair]
                    ),
                    "oracle_split_half_train_perturbed_reference_pearson_delta_median": _median(
                        [x[1] for x in oracle_pair]
                    ),
                    "perturbed_mean_baseline_centroid_accuracy": _centroid_accuracy(
                        train_mean, test_truth, max_conditions=500, seed=seed
                    ),
                    "test_count_bins": count_bins,
                }
            )
        return {"cell_lines": results, "rows": data.n_obs, "expression_genes": genes}
    finally:
        data.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--crosscell-h5ad", type=Path, required=True)
    parser.add_argument("--crosscell-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk", type=int, default=512)
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
        parser.error("analysis requires a clean source checkout")
    fold_manifest = json.loads((args.crosscell_folds / "manifest.json").read_text())
    if fold_manifest["source_sha256"] != sha256_file(args.crosscell_h5ad):
        parser.error("cross-cell source differs from sealed fold source")
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "source_commit": commit,
        "script_sha256": sha256_file(Path(__file__)),
        "crosscell_fold_manifest_sha256": sha256_file(args.crosscell_folds / "manifest.json"),
        "thread_limits": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "datasets": {},
    }
    try:
        for name, protocol in DATASETS.items():
            root = args.data_root / name / protocol
            canonical = json.loads((root / "manifests/canonical.json").read_text())
            split_path = root / "manifests/split.json"
            split = json.loads(split_path.read_text())
            print(f"START {name}", flush=True)
            result = _analyze(
                root / "canonical/adata.h5ad",
                genes=canonical["n_expression_genes"],
                split=split,
                crosscell=False,
                seed=args.seed,
                chunk=args.chunk,
            )
            result["canonical_sha256"] = canonical["canonical_adata_sha256"]
            result["split_sha256"] = sha256_file(split_path)
            receipt["datasets"][name] = result
            (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"DONE {name}", flush=True)
        print("START crosscell", flush=True)
        result = _analyze(
            args.crosscell_h5ad,
            genes=fold_manifest["gene_count"],
            split=None,
            crosscell=True,
            seed=args.seed,
            chunk=args.chunk,
        )
        result["source_sha256"] = fold_manifest["source_sha256"]
        result["protocol_note"] = (
            "descriptive within-line common-response audit; cross-cell train-only "
            "baselines require row-level fold adaptation"
        )
        receipt["datasets"]["crosscell"] = result
        (args.output / "crosscell.json").write_text(json.dumps(result, indent=2) + "\n")
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
