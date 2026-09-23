"""CPU-only data landscape for five canonical datasets and the fixed-axis cache.

The analysis is descriptive and must never select checkpoints or hyperparameters.
Only aggregated JSON and figures are written; cell matrices remain on the server.
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

from gradpert.hashing import sha256_file

DATASETS = {
    "replogle_k562_essential": "within_cell_unseen_single",
    "replogle_rpe1_essential": "within_cell_unseen_single",
    "nadig_jurkat": "within_cell_unseen_single",
    "nadig_hepg2": "within_cell_unseen_single",
    "norman": "norman_combo_seen2",
}
SEPARATOR = "\x1f"


def _median(values: list[float]) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _batch_information(
    conditions: np.ndarray, batches: np.ndarray
) -> dict[str, float | int | None]:
    import pandas as pd

    table = pd.crosstab(pd.Series(conditions), pd.Series(batches)).to_numpy(dtype=float)
    total = table.sum()
    row = table.sum(axis=1)
    column = table.sum(axis=0)
    probabilities = column / total
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum())
    conditional = 0.0
    for counts, size in zip(table, row, strict=True):
        p = counts[counts > 0] / size
        conditional += (size / total) * float(-(p * np.log(p)).sum())
    return {
        "n_conditions": len(row),
        "n_batches": len(column),
        "single_batch_condition_fraction": float(np.mean((table > 0).sum(axis=1) == 1)),
        "batch_information_fraction": (
            float(max(0.0, min(1.0, (entropy - conditional) / entropy))) if entropy > 0 else None
        ),
    }


def _group_codes(
    lines: np.ndarray,
    conditions: np.ndarray,
    batches: np.ndarray,
    controls: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    keys = [
        (
            f"{lines[i]}{SEPARATOR}ctrl{SEPARATOR}{batches[i]}"
            if controls[i]
            else f"{lines[i]}{SEPARATOR}{conditions[i]}"
        )
        for i in np.flatnonzero(valid)
    ]
    unique = sorted(set(keys))
    index = {key: i for i, key in enumerate(unique)}
    codes = np.full(len(lines), -1, dtype=np.int32)
    positions = np.flatnonzero(valid)
    codes[positions] = np.fromiter((index[key] for key in keys), dtype=np.int32)
    strata: dict[str, list[int]] = defaultdict(list)
    for i in positions:
        strata[f"{lines[i]}{SEPARATOR}{conditions[i]}{SEPARATOR}{batches[i]}"].append(int(i))
    halves = np.zeros(len(lines), dtype=np.int8)
    for key, rows in strata.items():
        values = np.asarray(rows)
        rng = np.random.default_rng(_stable_seed(seed, key))
        rng.shuffle(values)
        halves[values[len(values) // 2 :]] = 1
    codes[valid] = 2 * codes[valid] + halves[valid]
    return unique, codes


def _aggregate_expression(
    adata: object, codes: np.ndarray, groups: int, genes: int, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    from scipy import sparse

    sums = np.zeros((groups, genes), dtype=np.float64)
    counts = np.bincount(codes[codes >= 0], minlength=groups).astype(np.int64)
    for start in range(0, adata.n_obs, chunk):
        stop = min(start + chunk, adata.n_obs)
        part = codes[start:stop]
        valid = part >= 0
        if not valid.any():
            continue
        expression = adata.X[start:stop, :genes]
        expression = expression[valid]
        active, inverse = np.unique(part[valid], return_inverse=True)
        indicator = sparse.csr_matrix(
            (np.ones(len(inverse)), (inverse, np.arange(len(inverse)))),
            shape=(len(active), len(inverse)),
        )
        result = indicator @ expression
        sums[active] += result.toarray() if sparse.issparse(result) else result
    return sums, counts


def _retrieval(
    deltas: dict[str, tuple[np.ndarray, np.ndarray, int]], seed: int
) -> dict[str, float | int | None]:
    eligible = [name for name, (_, _, n) in deltas.items() if n >= 20]
    if len(eligible) > 500:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(eligible), 500, replace=False)
        eligible = [eligible[i] for i in sorted(selected)]
    left = np.stack([deltas[name][0] for name in eligible]) if eligible else np.empty((0, 0))
    right = np.stack([deltas[name][1] for name in eligible]) if eligible else np.empty((0, 0))
    if len(eligible) < 2:
        return {"conditions": len(eligible), "top1": None, "top10": None, "random_top1": None}
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    ln = np.linalg.norm(left, axis=1)
    rn = np.linalg.norm(right, axis=1)
    good = (ln > 1e-12) & (rn > 1e-12)
    left = left[good] / ln[good, None]
    right = right[good] / rn[good, None]
    n = len(left)
    if n < 2:
        return {"conditions": n, "top1": None, "top10": None, "random_top1": None}
    similarity = left @ right.T
    ranking = np.argsort(-similarity, axis=1)
    truth = np.arange(n)
    return {
        "conditions": n,
        "top1": float(np.mean(ranking[:, 0] == truth)),
        "top10": float(np.mean((ranking[:, : min(10, n)] == truth[:, None]).any(axis=1))),
        "random_top1": 1 / n,
        "split_half_pearson_delta_median": float(np.median(similarity[truth, truth])),
    }


def _analyze(adata_path: Path, *, crosscell: bool, seed: int, chunk: int, genes: int) -> dict:
    import anndata as ad

    adata = ad.read_h5ad(adata_path, backed="r")
    try:
        obs = adata.obs
        lines = obs["cell_line" if crosscell else "cell_type"].astype(str).to_numpy()
        conditions = obs["condition"].astype(str).to_numpy()
        batches = obs["batch"].astype(str).to_numpy()
        controls = obs["control"].astype(int).to_numpy().astype(bool)
        valid = lines != "K562_adamson" if crosscell else np.ones(adata.n_obs, dtype=bool)
        if np.any(controls[valid] != (conditions[valid] == "ctrl")):
            raise ValueError("control annotation disagrees with condition")
        keys, codes = _group_codes(lines, conditions, batches, controls, valid, seed)
        sums, counts = _aggregate_expression(adata, codes, len(keys) * 2, genes, chunk)
        group_sum = sums[0::2] + sums[1::2]
        group_count = counts[0::2] + counts[1::2]

        line_controls: dict[str, np.ndarray] = {}
        control_batches: dict[str, list[tuple[str, np.ndarray, np.ndarray, int]]] = defaultdict(
            list
        )
        for i, key in enumerate(keys):
            parts = key.split(SEPARATOR)
            if len(parts) == 3 and parts[1] == "ctrl":
                line, _, batch = parts
                control_batches[line].append(
                    (batch, sums[2 * i], sums[2 * i + 1], int(group_count[i]))
                )
        for line, entries in control_batches.items():
            total = sum((left + right for _, left, right, _ in entries), np.zeros(genes))
            n = sum(count for _, _, _, count in entries)
            line_controls[line] = total / n

        deltas: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        half_deltas: dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]] = defaultdict(dict)
        counts_by_line: dict[str, dict[str, int]] = defaultdict(dict)
        for i, key in enumerate(keys):
            parts = key.split(SEPARATOR)
            if len(parts) != 2:
                continue
            line, condition = parts
            control = line_controls[line]
            if group_count[i] < 1:
                continue
            deltas[line][condition] = group_sum[i] / group_count[i] - control
            counts_by_line[line][condition] = int(group_count[i])
            if counts[2 * i] and counts[2 * i + 1]:
                half_deltas[line][condition] = (
                    sums[2 * i] / counts[2 * i] - control,
                    sums[2 * i + 1] / counts[2 * i + 1] - control,
                    int(group_count[i]),
                )

        by_line = {}
        for line in sorted(deltas):
            mask = valid & (lines == line) & ~controls
            batch_info = _batch_information(conditions[mask], batches[mask])
            entries = [entry for entry in control_batches[line] if entry[3] >= 20]
            within = [
                _pearson(left / (n // 2), right / (n - n // 2))
                for _, left, right, n in entries
                if n // 2 and n - n // 2
            ]
            between = [
                _pearson(
                    (entries[i][1] + entries[i][2]) / entries[i][3],
                    (entries[j][1] + entries[j][2]) / entries[j][3],
                )
                for i in range(len(entries))
                for j in range(i + 1, len(entries))
            ]
            vectors = list(deltas[line].values())
            general = np.mean(vectors, axis=0)
            strengths = [float(np.linalg.norm(v) / np.sqrt(genes)) for v in vectors]
            by_line[line] = {
                "rows": int(np.sum(valid & (lines == line))),
                "control_rows": int(np.sum(valid & (lines == line) & controls)),
                "perturbation_conditions": len(vectors),
                "batch_information": batch_info,
                "control_correlation": {
                    "within_batch_median": _median(within),
                    "across_batch_median": _median(between),
                    "within_pairs": len(within),
                    "across_pairs": len(between),
                },
                "effect_rms_median": _median(strengths),
                "effect_rms_p90": float(np.quantile(strengths, 0.9)),
                "general_response_pearson_median": _median([_pearson(v, general) for v in vectors]),
                "replicate_retrieval": _retrieval(half_deltas[line], seed),
            }

        result = {
            "h5ad": str(adata_path),
            "rows": adata.n_obs,
            "expression_genes": genes,
            "seed": seed,
            "cell_lines": by_line,
        }
        if crosscell:
            transfer = {}
            for target in ("K562", "RPE1", "jurkat", "hepg2"):
                correlations = []
                per_source = {}
                for source in ("K562", "RPE1", "jurkat", "hepg2"):
                    if source == target:
                        continue
                    shared = set(deltas[target]) & set(deltas[source])
                    per_source[source] = {
                        "shared_conditions": len(shared),
                        "median_pearson_delta": _median(
                            [_pearson(deltas[target][c], deltas[source][c]) for c in shared]
                        ),
                    }
                for condition, target_delta in deltas[target].items():
                    sources = [
                        deltas[line][condition]
                        for line in ("K562", "RPE1", "jurkat", "hepg2")
                        if line != target and condition in deltas[line]
                    ]
                    if sources:
                        correlations.append(_pearson(target_delta, np.mean(sources, axis=0)))
                transfer[target] = {
                    "source_average_shared_conditions": len(correlations),
                    "source_average_pearson_delta_median": _median(correlations),
                    "pairwise": per_source,
                }
            result["crosscell_transfer"] = transfer
        else:
            for vectors in deltas.values():
                singles = {}
                doubles = {}
                for condition, vector in vectors.items():
                    components = tuple(x for x in condition.split("+") if x != "ctrl")
                    if len(components) == 1:
                        singles[components[0]] = vector
                    elif len(components) == 2:
                        doubles[condition] = (components, vector)
                if doubles:
                    additive = []
                    residual = []
                    for (left, right), observed in doubles.values():
                        if left not in singles or right not in singles:
                            continue
                        prediction = singles[left] + singles[right]
                        additive.append(_pearson(observed, prediction))
                        residual.append(
                            float(
                                np.linalg.norm(observed - prediction)
                                / max(np.linalg.norm(observed), 1e-12)
                            )
                        )
                    result["double_additivity"] = {
                        "double_conditions": len(doubles),
                        "with_both_singles": len(additive),
                        "observed_vs_additive_pearson_median": _median(additive),
                        "relative_residual_median": _median(residual),
                    }
        return result
    finally:
        adata.file.close()


def _plot(summaries: dict[str, dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    info = []
    for dataset, result in summaries.items():
        for line, item in result["cell_lines"].items():
            names.append(f"{dataset}\n{line}" if dataset == "crosscell" else dataset)
            info.append(item)
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].bar(x, [v["batch_information"]["batch_information_fraction"] or 0 for v in info])
    axes[0, 0].set_title("Batch information explained by perturbation ID")
    width = 0.35
    axes[0, 1].bar(
        x - width / 2,
        [v["control_correlation"]["within_batch_median"] or 0 for v in info],
        width,
        label="within",
    )
    axes[0, 1].bar(
        x + width / 2,
        [v["control_correlation"]["across_batch_median"] or 0 for v in info],
        width,
        label="across",
    )
    axes[0, 1].set_title("Control mean Pearson by batch")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].bar(x, [v["replicate_retrieval"]["top1"] or 0 for v in info], label="observed")
    axes[1, 0].plot(
        x,
        [v["replicate_retrieval"]["random_top1"] or 0 for v in info],
        "o",
        color="black",
        label="random",
    )
    axes[1, 0].set_title("Split-half perturbation retrieval top-1")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].bar(x, [v["general_response_pearson_median"] or 0 for v in info])
    axes[1, 1].set_title("Correlation with shared mean response")
    for ax in axes.flat:
        ax.set_xticks(x, names, rotation=50, ha="right", fontsize=7)
    fig.savefig(output, dpi=160)
    plt.close(fig)


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
    if args.chunk < 1 or args.seed < 0:
        parser.error("chunk must be positive and seed nonnegative")
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
        parser.error("cross-cell H5AD differs from the sealed four-fold source")
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "source_commit": commit,
        "script_sha256": sha256_file(Path(__file__)),
        "seed": args.seed,
        "chunk": args.chunk,
        "crosscell_fold_manifest_sha256": sha256_file(args.crosscell_folds / "manifest.json"),
        "datasets": {},
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    try:
        for name, protocol in DATASETS.items():
            root = args.data_root / name / protocol
            canonical = json.loads((root / "manifests/canonical.json").read_text())
            print(f"START {name}", flush=True)
            result = _analyze(
                root / "canonical/adata.h5ad",
                crosscell=False,
                seed=args.seed,
                chunk=args.chunk,
                genes=canonical["n_expression_genes"],
            )
            result["canonical_adata_sha256"] = canonical["canonical_adata_sha256"]
            receipt["datasets"][name] = result
            (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"DONE {name}", flush=True)
        print("START crosscell", flush=True)
        result = _analyze(
            args.crosscell_h5ad, crosscell=True, seed=args.seed, chunk=args.chunk, genes=3352
        )
        result["fold_manifest_sha256"] = receipt["crosscell_fold_manifest_sha256"]
        receipt["datasets"]["crosscell"] = result
        (args.output / "crosscell.json").write_text(json.dumps(result, indent=2) + "\n")
        _plot(receipt["datasets"], args.output / "landscape.png")
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
