"""Train-only condition-cap sampling and observed-versus-observed repeatability.

Scientific data and the derived H5AD stay on the server. No training config,
canonical manifest, gene selection, or evaluation population is replaced.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from gradpert.data._io import atomic_json
from gradpert.data.registry import load_dataset_registry
from gradpert.evaluation.metrics import pearson_correlation
from gradpert.hashing import sha256_file, sha256_json

DATASET = "nadig_jurkat"
PROTOCOL = "within_cell_unseen_single"


def stable_seed(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little")


def proportional_sample(batches: np.ndarray, cap: int, *, seed: int) -> np.ndarray:
    """Largest-remainder batch quotas, then uniform sampling without replacement."""
    if cap < 1 or batches.ndim != 1 or not len(batches):
        raise ValueError("sampling requires nonempty batch labels and positive cap")
    if len(batches) <= cap:
        return np.arange(len(batches), dtype=np.int64)
    names, counts = np.unique(batches, return_counts=True)
    quotas = counts * (cap / len(batches))
    take = np.floor(quotas).astype(np.int64)
    rng = np.random.default_rng(seed)
    priority = np.lexsort((rng.random(len(names)), -(quotas - take)))
    take[priority[: cap - int(take.sum())]] += 1
    selected = [
        rng.choice(np.flatnonzero(batches == name), int(count), replace=False)
        for name, count in zip(names, take, strict=True)
        if count
    ]
    rows = np.sort(np.concatenate(selected))
    if len(rows) != cap or len(np.unique(rows)) != cap or np.any(take > counts):
        raise AssertionError("invalid stratified sample")
    return rows


def half_assignments(batches: np.ndarray, *, repeats: int, seed: int) -> np.ndarray:
    """Disjoint balanced halves, also balanced within each batch.

    Randomize odd-batch ties and batch traversal so singleton batches do not
    receive a deterministic side on every repetition.
    """
    if repeats < 1 or batches.ndim != 1 or not len(batches):
        raise ValueError("halves require nonempty batch labels and positive repeats")
    groups = [np.flatnonzero(batches == name) for name in np.unique(batches)]
    result = np.empty((repeats, len(batches)), dtype=np.int8)
    for repeat in range(repeats):
        rng = np.random.default_rng(stable_seed(seed, str(repeat)))
        sizes = [0, 0]
        for index in rng.permutation(len(groups)):
            rows = rng.permutation(groups[index])
            half = len(rows) // 2
            result[repeat, rows[:half]] = 0
            result[repeat, rows[half : 2 * half]] = 1
            sizes[0] += half
            sizes[1] += half
            if len(rows) % 2:
                side = int(rng.integers(2)) if sizes[0] == sizes[1] else int(sizes[0] > sizes[1])
                result[repeat, rows[-1]] = side
                sizes[side] += 1
        if abs(sizes[0] - sizes[1]) > 1:
            raise AssertionError("unbalanced condition halves")
    return result


def row_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Float64 Pearson over the gene axis; undefined correlations remain NaN."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] < 2:
        raise ValueError("Pearson inputs must have matching row-by-gene shapes")
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sum(x * y, axis=1) / denominator
    return np.where(np.isfinite(result), result, np.nan)


def repeatability(
    expression: np.ndarray, batches: np.ndarray, control: np.ndarray, *, repeats: int, seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    halves = half_assignments(batches, repeats=repeats, seed=seed)
    identity = hashlib.sha256(halves.tobytes()).hexdigest()
    if len(expression) < 2:
        return np.full(repeats, np.nan), np.full(repeats, np.nan), identity
    values = np.asarray(expression, dtype=np.float64)
    weights = (halves == 0).astype(np.float64)
    left_count = weights.sum(axis=1)
    right_count = len(values) - left_count
    left_sum = weights @ values
    left = left_sum / left_count[:, None]
    right = (values.sum(axis=0) - left_sum) / right_count[:, None]
    return row_pearson(left - control, right - control), row_pearson(left, right), identity


def summarize(values: list | np.ndarray) -> dict:
    finite = np.asarray([x for x in values if x is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "finite_count": len(finite),
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "p10": float(np.quantile(finite, 0.1)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
    }


def csv_row(writer: csv.DictWriter, payload: dict) -> None:
    writer.writerow(
        {
            key: "" if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in payload.items()
        }
    )


def plot_comparison(records: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = [
        r
        for r in records
        if r["affected"] and r["original_delta"] is not None and r["cap_delta"] is not None
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    bins = np.linspace(-1, 1, 41)
    for version, label, color in (("original", "Original", "#235f9a"), ("cap", "Cap40", "#d5762c")):
        values = [r[f"{version}_delta"] for r in records if r[f"{version}_delta"] is not None]
        axes[0].hist(values, bins=bins, histtype="step", linewidth=2, label=label, color=color)
    axes[0].set(xlabel="Mean split-half Pearson delta (100 splits)", ylabel="Conditions")
    axes[0].legend()
    axes[1].scatter(
        [r["original_delta"] for r in paired],
        [r["cap_delta"] for r in paired],
        s=10,
        alpha=0.5,
        color="#235f9a",
    )
    axes[1].plot([-1, 1], [-1, 1], "--", color="gray")
    axes[1].set(
        xlabel="Original Pearson delta",
        ylabel="Cap40 Pearson delta",
        title="Conditions with >40 cells",
    )
    axes[2].scatter(
        [r["original_cells"] for r in paired],
        [r["cap_delta"] - r["original_delta"] for r in paired],
        s=10,
        alpha=0.5,
        color="#d5762c",
    )
    axes[2].axhline(0, linestyle="--", color="gray")
    axes[2].set(
        xscale="log",
        xlabel="Original cells per condition",
        ylabel="Cap40 minus original Pearson delta",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180)
    plt.close(fig)


def export_subset(adata, kept_rows: np.ndarray, output: Path, provenance: dict) -> dict:
    """Use the native H5AD writer, then compare every retained matrix value."""
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    original = adata.to_memory()
    subset = original[kept_rows].copy()
    subset.uns["gradpert_downsample"] = provenance
    subset.write_h5ad(output)
    del subset
    gc.collect()
    check = ad.read_h5ad(output, backed="r")
    try:
        pd.testing.assert_frame_equal(check.obs, original.obs.iloc[kept_rows])
        pd.testing.assert_frame_equal(check.var, original.var)
        for start in range(0, len(kept_rows), 512):
            stop = min(start + 512, len(kept_rows))
            left = check.X[start:stop]
            right = original.X[kept_rows[start:stop]]
            left = left.toarray() if sparse.issparse(left) else np.asarray(left)
            right = right.toarray() if sparse.issparse(right) else np.asarray(right)
            if left.dtype != right.dtype or not np.array_equal(left, right):
                raise ValueError("derived matrix changed retained expression values")
        return {
            "rows": check.n_obs,
            "genes": check.n_vars,
            "sha256": sha256_file(output),
            "retained_values_exact": True,
            "obs_var_metadata_exact": True,
        }
    finally:
        check.file.close()


def analyze(root: Path, output: Path, *, repeats: int, seed: int, source: str) -> dict:
    import anndata as ad
    from scipy import sparse

    started = time.monotonic()
    manifest_paths = [
        root / "manifests" / name
        for name in (
            "canonical.json",
            "split.json",
            "evaluation_controls.val.json",
            "evaluation_controls.test.json",
        )
    ]
    before = {str(path): sha256_file(path) for path in manifest_paths}
    canonical = json.loads(manifest_paths[0].read_text())
    split = json.loads(manifest_paths[1].read_text())
    path = root / "canonical/adata.h5ad"
    parent_sha = sha256_file(path)
    if parent_sha != canonical["canonical_adata_sha256"]:
        raise ValueError("current H5AD differs from the canonical manifest")
    data = ad.read_h5ad(path, backed="r")
    try:
        genes = int(canonical["n_expression_genes"])
        if data.shape != (canonical["n_cells"], canonical["n_graph_genes"]):
            raise ValueError("canonical shape differs from its manifest")
        gene_ids = (root / "canonical/expression_gene_ids.txt").read_text().splitlines()
        registry = load_dataset_registry(
            Path(__file__).resolve().parents[2] / "registry/datasets/nadig_jurkat.yaml"
        )
        symbol_column = registry.canonical_metadata.gene_symbol_column
        symbols = data.var[symbol_column].astype(str)
        if (
            symbols.iloc[:genes].tolist() != gene_ids
            or not symbols.is_unique
            or not data.obs_names.is_unique
            or sha256_json(gene_ids) != canonical["expression_gene_order_sha256"]
            or sha256_json(data.obs_names.tolist()) != canonical["observation_order_sha256"]
        ):
            raise ValueError("expression axis or observation identities differ")
        conditions = data.obs["condition"].astype(str).to_numpy()
        batches = data.obs["batch"].astype(str).to_numpy()
        control_mask = data.obs["control"].astype(bool).to_numpy()
        names = [c for c in split["train_conditions"] if c != split["control_condition_id"]]
        train_mask = np.isin(conditions, names) & ~control_mask
        train_rows = np.flatnonzero(train_mask)
        train_positions = np.full(data.n_obs, -1, dtype=np.int64)
        train_positions[train_rows] = np.arange(len(train_rows))
        values = np.empty((len(train_rows), genes), dtype=data.X.dtype)
        control_names = sorted(set(batches[control_mask]))
        control_code = {name: i for i, name in enumerate(control_names)}
        control_sums = np.zeros((len(control_names), genes), dtype=np.float64)
        control_counts = np.zeros(len(control_names), dtype=np.int64)
        for start in range(0, data.n_obs, 512):
            stop = min(start + 512, data.n_obs)
            block = data.X[start:stop, :genes]
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            if not np.isfinite(block).all():
                raise ValueError("nonfinite canonical expression")
            positions = train_positions[start:stop]
            keep = positions >= 0
            values[positions[keep]] = block[keep]
            ctrls = control_mask[start:stop]
            codes = np.asarray([control_code[b] for b in batches[start:stop][ctrls]], dtype=int)
            np.add.at(control_sums, codes, block[ctrls])
            np.add.at(control_counts, codes, 1)
            if start % (512 * 64) == 0:
                print(f"READ {stop}/{data.n_obs}", flush=True)
        records = []
        selected_rows = []
        per_condition_selection = {}
        hashes = hashlib.sha256()
        fields = ["condition", "version", "repeat", "cells", "pearson_delta", "pearson_expression"]
        with (output / "repeat_scores.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, condition in enumerate(names):
                rows = np.flatnonzero(train_mask & (conditions == condition))
                if not len(rows):
                    raise ValueError(f"training condition has no source cells: {condition}")
                batch = batches[rows]
                chosen = proportional_sample(
                    batch, 40, seed=stable_seed(seed, f"sample:{condition}")
                )
                selected_rows.extend(rows[chosen].tolist())
                per_condition_selection[condition] = data.obs_names[rows[chosen]].tolist()
                codes = np.asarray([control_code[b] for b in sorted(set(batch))], dtype=int)
                control = control_sums[codes].sum(axis=0) / control_counts[codes].sum()
                hashes.update(control.tobytes())
                expression = values[train_positions[rows]]
                sampled = expression[chosen]
                half_seed = stable_seed(seed, f"halves:{condition}")
                record = {
                    "condition": condition,
                    "original_cells": len(rows),
                    "cap_cells": len(chosen),
                    "affected": len(rows) > 40,
                    "original_batches": len(set(batch)),
                    "cap_batches": len(set(batch[chosen])),
                }
                for version, x, b in (
                    ("original", expression, batch),
                    ("cap", sampled, batch[chosen]),
                ):
                    delta, raw, identity = repeatability(
                        x, b, control, repeats=repeats, seed=half_seed
                    )
                    record[f"{version}_delta"] = summarize(delta)["mean"]
                    record[f"{version}_expression"] = summarize(raw)["mean"]
                    record[f"{version}_valid_repeats"] = int(np.isfinite(delta).sum())
                    record[f"{version}_half_assignment_sha256"] = identity
                    for repeat in range(repeats):
                        csv_row(
                            writer,
                            {
                                "condition": condition,
                                "version": version,
                                "repeat": repeat,
                                "cells": len(x),
                                "pearson_delta": float(delta[repeat]),
                                "pearson_expression": float(raw[repeat]),
                            },
                        )
                record["sample_vs_full_delta"] = pearson_correlation(
                    sampled.mean(axis=0, dtype=np.float64) - control,
                    expression.mean(axis=0, dtype=np.float64) - control,
                )[0]
                removed = np.setdiff1d(np.arange(len(expression)), chosen)
                record["retained_vs_removed_delta"] = (
                    pearson_correlation(
                        sampled.mean(axis=0, dtype=np.float64) - control,
                        expression[removed].mean(axis=0, dtype=np.float64) - control,
                    )[0]
                    if len(removed)
                    else None
                )
                records.append(record)
                if (i + 1) % 25 == 0 or i + 1 == len(names):
                    atomic_json(
                        output / "progress.json",
                        {"phase": "pearson", "done": i + 1, "total": len(names)},
                    )
                    print(f"PEARSON {i + 1}/{len(names)}", flush=True)
        selected_rows = np.asarray(sorted(selected_rows), dtype=np.int64)
        kept = ~train_mask
        kept[selected_rows] = True
        kept_rows = np.flatnonzero(kept)
        row_ids = data.obs_names[selected_rows].tolist()
        selection = {
            "schema_version": "gradpert-train-cap-selection-1",
            "dataset_id": DATASET,
            "state": "analysis_derivative_not_canonical_ready",
            "source_commit": source,
            "parent_h5ad_sha256": parent_sha,
            "cap": 40,
            "seed": seed,
            "sampling": "condition_batch_largest_remainder_uniform_without_replacement",
            "train_condition_ids": names,
            "selected_train_row_ids": row_ids,
            "selected_train_row_ids_sha256": sha256_json(row_ids),
            "by_condition": per_condition_selection,
            "unmodified_row_ids_sha256": sha256_json(data.obs_names[~train_mask].tolist()),
        }
        atomic_json(output / "selection.json", selection)
        atomic_json(output / "conditions.json", records)
        paired = [
            r for r in records if r["original_delta"] is not None and r["cap_delta"] is not None
        ]
        differences = np.asarray([r["cap_delta"] - r["original_delta"] for r in paired])
        rng = np.random.default_rng(stable_seed(seed, "condition-bootstrap"))
        boot = differences[rng.integers(len(differences), size=(1000, len(differences)))].mean(
            axis=1
        )

        def stratum(n: int) -> str:
            return "1-40" if n <= 40 else "41-80" if n <= 80 else "81-160" if n <= 160 else ">160"

        strata = {
            label: {
                "conditions": sum(stratum(r["original_cells"]) == label for r in records),
                **{
                    v: summarize(
                        [r[f"{v}_delta"] for r in records if stratum(r["original_cells"]) == label]
                    )
                    for v in ("original", "cap")
                },
            }
            for label in ("1-40", "41-80", "81-160", ">160")
        }
        summary = {
            "dataset_id": DATASET,
            "source_commit": source,
            "source_clean": True,
            "parent_h5ad_sha256": parent_sha,
            "parent_split_sha256": before[str(manifest_paths[1])],
            "seed": seed,
            "repeats": repeats,
            "cap": 40,
            "expression_genes": genes,
            "original_train_cells": len(train_rows),
            "sampled_train_cells": len(selected_rows),
            "train_cells_removed_fraction": 1 - len(selected_rows) / len(train_rows),
            "conditions": len(names),
            "affected_conditions": sum(r["affected"] for r in records),
            "unchanged_nontrain_cells": int((~train_mask).sum()),
            "control_cells": int(control_mask.sum()),
            "derivative_total_cells": len(kept_rows),
            "control_reference": (
                "same_original_control_pool_within_original_condition_batch_contexts"
            ),
            "control_reference_content_sha256": hashes.hexdigest(),
            "original": summarize([r["original_delta"] for r in records]),
            "cap40": summarize([r["cap_delta"] for r in records]),
            "raw_expression": {
                v: summarize([r[f"{v}_expression"] for r in records]) for v in ("original", "cap")
            },
            "paired_condition_delta": {
                "conditions": len(paired),
                "mean": float(differences.mean()),
                "condition_bootstrap_95_interval": np.quantile(boot, [0.025, 0.975]).tolist(),
                "interpretation": (
                    "condition-level descriptive interval, not biological-replicate uncertainty"
                ),
            },
            "affected_sample_vs_full": summarize(
                [r["sample_vs_full_delta"] for r in records if r["affected"]]
            ),
            "affected_retained_vs_removed": summarize(
                [r["retained_vs_removed_delta"] for r in records if r["affected"]]
            ),
            "cell_count_strata": strata,
            "interpretation": (
                "100 splits of one fixed sample; observed repeatability, "
                "not model scores or a theoretical upper bound"
            ),
        }
        plot_comparison(records, output)
        del values
        gc.collect()
        print("EXPORT H5AD", flush=True)
        atomic_json(output / "progress.json", {"phase": "export_h5ad"})
        provenance = {
            "purpose": "train_only_cap40_analysis",
            "state": "analysis_derivative_not_canonical_ready",
            "source_commit": source,
            "parent_h5ad_sha256": parent_sha,
            "cap": 40,
            "seed": seed,
            "selection_sha256": sha256_file(output / "selection.json"),
        }
        summary["derived_h5ad"] = export_subset(data, kept_rows, output / "cap40.h5ad", provenance)
        after = {str(p): sha256_file(p) for p in manifest_paths}
        if after != before or sha256_file(path) != parent_sha:
            raise ValueError("original data/manifests changed during analysis")
        summary["original_data_unchanged"] = True
        summary["parent_manifest_sha256"] = before
        summary["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        data.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("output must be a new directory under /data/yilangliu")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    repository = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip():
        parser.error("analysis requires a clean published checkout")
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "source_commit": source,
        "source_clean": True,
        "script_sha256": sha256_file(Path(__file__)),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "data_root": str(args.data_root),
        "output": str(args.output),
        "seed": args.seed,
        "repeats": args.repeats,
        "cap": 40,
        "thread_limits": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    atomic_json(args.output / "receipt.json", receipt)
    try:
        summary = analyze(
            args.data_root / DATASET / PROTOCOL,
            args.output,
            repeats=args.repeats,
            seed=args.seed,
            source=source,
        )
        receipt.update(
            status="complete",
            summary_sha256=sha256_file(args.output / "summary.json"),
            outputs={
                p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in args.output.iterdir()
                if p.is_file() and p.name not in ("receipt.json", "progress.json")
            },
            elapsed_seconds=summary["elapsed_seconds"],
        )
        atomic_json(
            args.output / "COMPLETE.json",
            {
                "source_commit": source,
                "status": "complete",
                "summary_sha256": receipt["summary_sha256"],
            },
        )
        print("COMPLETE", flush=True)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(
            args.output / "FAILURE.json", {"source_commit": source, "error": receipt["error"]}
        )
        raise
    finally:
        atomic_json(args.output / "receipt.json", receipt)


if __name__ == "__main__":
    main()
