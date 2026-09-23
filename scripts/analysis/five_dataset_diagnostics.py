"""CPU-only, hash-bound descriptive analysis of the five canonical datasets.

This is a split-half reproducibility diagnostic, not an attainable upper bound.
It never writes predictions or modifies the canonical data. All large arrays
stay under the server-side output directory and are discarded after plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gradpert.evaluation.metrics import pearson_correlation
from gradpert.hashing import sha256_file

DATASETS = {
    "replogle_k562_essential": "within_cell_unseen_single",
    "replogle_rpe1_essential": "within_cell_unseen_single",
    "nadig_jurkat": "within_cell_unseen_single",
    "nadig_hepg2": "within_cell_unseen_single",
    "norman": "norman_combo_seen2",
}
KIND_NAMES = {0: "control", 1: "single", 2: "double"}


def perturbation_kind(condition: str, control: bool) -> int:
    """Count non-control perturbation targets, never merge controls with singles."""
    targets = sum(token != "ctrl" for token in condition.split("+"))
    if (targets == 0) != control or targets not in KIND_NAMES:
        raise ValueError(f"control/target annotation inconsistent: {condition}")
    return targets


def split_half_assignments(
    conditions: np.ndarray,
    batches: np.ndarray,
    test_conditions: list[str],
    *,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Disjoint, batch-stratified half labels for each test perturbation."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    test = sorted(set(test_conditions))
    if len(test) != len(test_conditions):
        raise ValueError("test conditions contain duplicates")
    codes = np.full(len(conditions), -1, dtype=np.int32)
    groups: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    code_by_condition = {name: i for i, name in enumerate(test)}
    for row, (condition, batch) in enumerate(zip(conditions, batches, strict=True)):
        code = code_by_condition.get(str(condition))
        if code is not None:
            codes[row] = code
            groups[code][str(batch)].append(row)
    halves = np.full((repeats, len(conditions)), -1, dtype=np.int8)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        for code in range(len(test)):
            counts = [0, 0]
            for batch in sorted(groups[code]):
                rows = np.asarray(groups[code][batch], dtype=np.int64)
                rng.shuffle(rows)
                n_first = len(rows) // 2
                halves[repeat, rows[:n_first]] = 0
                halves[repeat, rows[n_first : 2 * n_first]] = 1
                counts[0] += n_first
                counts[1] += n_first
                if len(rows) % 2:
                    side = 0 if counts[0] <= counts[1] else 1
                    halves[repeat, rows[-1]] = side
                    counts[side] += 1
            if sum(counts) >= 2 and min(counts) == 0:
                occupied = 0 if counts[0] else 1
                selected = np.flatnonzero((codes == code) & (halves[repeat] == occupied))
                halves[repeat, selected[-1]] = 1 - occupied
    if np.any(halves[:, codes >= 0] < 0):
        raise AssertionError("test row escaped split assignment")
    return codes, halves, test


def select_visualization_rows(
    kinds: np.ndarray,
    conditions: np.ndarray,
    *,
    per_kind: int,
    seed: int,
    per_condition_cap: int = 24,
) -> np.ndarray:
    """Balance classes and prevent a large perturbation from dominating plots."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for kind in sorted(set(int(value) for value in kinds)):
        candidates = np.flatnonzero(kinds == kind)
        rng.shuffle(candidates)
        counts: Counter[str] = Counter()
        chosen = 0
        for row in candidates:
            condition = str(conditions[row])
            if kind and counts[condition] >= per_condition_cap:
                continue
            selected.append(int(row))
            counts[condition] += 1
            chosen += 1
            if chosen == per_kind:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _plot_embedding(
    pca_xy: np.ndarray,
    umap_xy: np.ndarray,
    kind: np.ndarray,
    split: np.ndarray,
    batch: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    batch_counts = Counter(batch.tolist())
    top_batches = {name for name, _ in batch_counts.most_common(8)}
    batch_display = np.asarray([name if name in top_batches else "other" for name in batch])
    panels = (
        (
            "perturbation type",
            kind,
            {"control": "#444444", "single": "#2675bd", "double": "#df8530"},
        ),
        (
            "split",
            split,
            {"control": "#444444", "train": "#2675bd", "val": "#df8530", "test": "#ca4e72"},
        ),
        ("batch (top 8)", batch_display, None),
    )
    for row, (title, coordinates) in enumerate((("PCA", pca_xy), ("UMAP", umap_xy))):
        for col, (label, values, fixed_colors) in enumerate(panels):
            ax = axes[row, col]
            levels = sorted(set(values))
            palette = plt.get_cmap("tab10")
            for i, level in enumerate(levels):
                mask = values == level
                color = fixed_colors.get(level) if fixed_colors else palette(i % 10)
                ax.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=4,
                    alpha=0.48,
                    rasterized=True,
                    label=level,
                    color=color,
                )
            ax.set_title(f"{title}: {label}")
            ax.set_xlabel(f"{title} 1")
            ax.set_ylabel(f"{title} 2")
            ax.legend(markerscale=3, fontsize=7, frameon=False, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_reproducibility(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for kind in ("single", "double"):
        subset = [row for row in rows if row["kind"] == kind and row["pearson_delta"] is not None]
        if not subset:
            continue
        color = "#2675bd" if kind == "single" else "#df8530"
        axes[0].hist(
            [row["pearson_delta"] for row in subset], bins=35, alpha=0.55, label=kind, color=color
        )
        axes[1].scatter(
            [row["truth_cells"] for row in subset],
            [row["pearson_delta"] for row in subset],
            s=7,
            alpha=0.35,
            label=kind,
            color=color,
        )
    axes[0].set(xlabel="split-half Pearson Δ", ylabel="condition x repeat count")
    axes[1].set(xlabel="truth cells per condition", ylabel="split-half Pearson Δ", xscale="log")
    for ax in axes:
        ax.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_dataset(
    dataset_id: str,
    protocol_id: str,
    *,
    data_root: Path,
    output: Path,
    repeats: int,
    seed: int,
    per_kind: int,
    chunk_size: int,
) -> dict:
    import anndata as ad
    import scipy.sparse as sp
    import umap
    from numba import njit
    from sklearn.decomposition import PCA

    root = data_root / dataset_id / protocol_id
    manifest_path = root / "manifests/canonical.json"
    split_path = root / "manifests/split.json"
    control_path = root / "manifests/evaluation_controls.test.json"
    manifest = json.loads(manifest_path.read_text())
    split_manifest = json.loads(split_path.read_text())
    controls_manifest = json.loads(control_path.read_text())
    adata = ad.read_h5ad(root / "canonical/adata.h5ad", backed="r")
    try:
        gene_count = int(manifest["n_expression_genes"])
        if adata.n_obs != manifest["n_cells"] or adata.n_vars != manifest["n_graph_genes"]:
            raise ValueError("canonical matrix shape differs from manifest")
        conditions = adata.obs["condition"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        control_mask = adata.obs["control"].astype(bool).to_numpy()
        row_ids = np.asarray(adata.obs_names.astype(str))
        kinds = np.asarray(
            [
                perturbation_kind(c, bool(control))
                for c, control in zip(conditions, control_mask, strict=True)
            ],
            dtype=np.int8,
        )
        split_by_condition = {
            condition: part
            for part in ("train", "val", "test")
            for condition in split_manifest[f"{part}_conditions"]
        }
        plot_split = np.asarray(
            [
                "control" if kind == 0 else split_by_condition[condition]
                for condition, kind in zip(conditions, kinds, strict=True)
            ]
        )
        sampled = select_visualization_rows(kinds, conditions, per_kind=per_kind, seed=seed)
        sampled_positions = np.full(adata.n_obs, -1, dtype=np.int32)
        sampled_positions[sampled] = np.arange(len(sampled), dtype=np.int32)
        sample_expression = np.empty((len(sampled), gene_count), dtype=np.float32)

        test_conditions = [
            condition
            for condition in split_manifest["test_conditions"]
            if condition != split_manifest["control_condition_id"]
        ]
        codes, halves, ordered_test = split_half_assignments(
            conditions, batches, test_conditions, repeats=repeats, seed=seed
        )
        sums = np.zeros((repeats, 2, len(ordered_test), gene_count), dtype=np.float64)
        counts = np.zeros((repeats, 2, len(ordered_test)), dtype=np.int64)
        control_rows = np.flatnonzero(control_mask)
        control_positions = np.full(adata.n_obs, -1, dtype=np.int32)
        control_positions[control_rows] = np.arange(len(control_rows), dtype=np.int32)
        control_expression = np.empty((len(control_rows), gene_count), dtype=np.float32)

        @njit(cache=False)
        def add_test_rows(values, row_codes, row_halves, totals, sizes):
            for row in range(values.shape[0]):
                code = row_codes[row]
                if code < 0:
                    continue
                for repeat in range(row_halves.shape[0]):
                    side = row_halves[repeat, row]
                    sizes[repeat, side, code] += 1
                    for gene in range(values.shape[1]):
                        totals[repeat, side, code, gene] += values[row, gene]

        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            block = adata.X[start:stop, :gene_count]
            if sp.issparse(block):
                block = block.toarray()
            values = np.asarray(block, dtype=np.float32)
            add_test_rows(values, codes[start:stop], halves[:, start:stop], sums, counts)
            visual = sampled_positions[start:stop]
            keep = visual >= 0
            sample_expression[visual[keep]] = values[keep]
            control = control_positions[start:stop]
            keep = control >= 0
            control_expression[control[keep]] = values[keep]

        index_by_row_id = {row_id: i for i, row_id in enumerate(row_ids)}
        if len(index_by_row_id) != len(row_ids):
            raise ValueError("canonical row IDs are not unique")
        draws = {
            entry["condition_id"]: entry["ordered_row_ids"] for entry in controls_manifest["draws"]
        }
        results: list[dict] = []
        for code, condition in enumerate(ordered_test):
            control_indices = [
                control_positions[index_by_row_id[row_id]] for row_id in draws[condition]
            ]
            if len(control_indices) != 300 or min(control_indices) < 0:
                raise ValueError("frozen evaluation control draw is malformed")
            control_mean = control_expression[control_indices].mean(axis=0, dtype=np.float64)
            truth_cells = int(counts[0, :, code].sum())
            kind = KIND_NAMES[perturbation_kind(condition, False)]
            for repeat in range(repeats):
                left_count, right_count = counts[repeat, :, code]
                if not left_count or not right_count:
                    results.append(
                        {
                            "condition": condition,
                            "kind": kind,
                            "repeat": repeat,
                            "truth_cells": truth_cells,
                            "pearson_delta": None,
                            "pearson_expression": None,
                            "unavailable_reason": "fewer_than_two_truth_cells",
                        }
                    )
                    continue
                left = sums[repeat, 0, code] / left_count
                right = sums[repeat, 1, code] / right_count
                delta, reason = pearson_correlation(left - control_mean, right - control_mean)
                raw, _ = pearson_correlation(left, right)
                results.append(
                    {
                        "condition": condition,
                        "kind": kind,
                        "repeat": repeat,
                        "truth_cells": truth_cells,
                        "left_cells": int(left_count),
                        "right_cells": int(right_count),
                        "pearson_delta": delta,
                        "pearson_expression": raw,
                        "unavailable_reason": reason,
                    }
                )

        pca = PCA(
            n_components=min(50, len(sampled) - 1, gene_count),
            svd_solver="randomized",
            random_state=seed,
        )
        pcs = pca.fit_transform(sample_expression)
        embedding = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            n_components=2,
            random_state=seed,
            n_jobs=1,
        ).fit_transform(pcs[:, : min(30, pcs.shape[1])])
        labels = np.asarray([KIND_NAMES[int(kind)] for kind in kinds[sampled]])
        _plot_embedding(
            pcs[:, :2],
            embedding,
            labels,
            plot_split[sampled],
            batches[sampled],
            output / "embedding.png",
        )
        _plot_reproducibility(results, output / "split_half.png")
        (output / "split_half_conditions.json").write_text(json.dumps(results, indent=2) + "\n")

        class_cells = {KIND_NAMES[k]: int((kinds == k).sum()) for k in KIND_NAMES}
        class_conditions = {KIND_NAMES[k]: len(set(conditions[kinds == k])) for k in KIND_NAMES}
        summaries = {}
        for kind in ("single", "double"):
            for metric in ("pearson_delta", "pearson_expression"):
                values = [
                    row[metric]
                    for row in results
                    if row["kind"] == kind and row[metric] is not None
                ]
                summaries[f"{kind}_{metric}"] = _summarize(values)
        summary = {
            "dataset_id": dataset_id,
            "protocol_id": protocol_id,
            "canonical_adata_sha256": manifest["canonical_adata_sha256"],
            "split_manifest_sha256": sha256_file(split_path),
            "test_control_manifest_sha256": sha256_file(control_path),
            "seed": seed,
            "repeats": repeats,
            "n_cells": int(adata.n_obs),
            "n_genes": gene_count,
            "n_batches": len(set(batches)),
            "class_cells": class_cells,
            "class_conditions": class_conditions,
            "visualization_sample": {
                KIND_NAMES[k]: int((kinds[sampled] == k).sum()) for k in KIND_NAMES
            },
            "pca_variance_explained_2": float(pca.explained_variance_ratio_[:2].sum()),
            "split_half": summaries,
            "unavailable_test_condition_repeats": sum(
                row["pearson_delta"] is None for row in results
            ),
            "interpretation": (
                "split-half experimental reproducibility; not a theoretical Pearson upper bound"
            ),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        adata.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-kind", type=int, default=2500)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("scientific artifacts must stay under /data/yilangliu")
    if args.output.exists():
        parser.error("output directory already exists; analysis never overwrites")
    if min(args.repeats, args.per_kind, args.chunk_size) < 1:
        parser.error("positive repeat, sample and chunk budgets required")
    source = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        parser.error("analysis requires a clean source checkout")
    args.output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "source_commit": commit,
        "script_sha256": sha256_file(Path(__file__)),
        "data_root": str(args.data_root.resolve()),
        "datasets": args.dataset or list(DATASETS),
        "repeats": args.repeats,
        "seed": args.seed,
        "per_kind": args.per_kind,
        "chunk_size": args.chunk_size,
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        summaries = []
        for dataset in manifest["datasets"]:
            print(f"START {dataset}", flush=True)
            directory = args.output / dataset
            directory.mkdir()
            result = analyze_dataset(
                dataset,
                DATASETS[dataset],
                data_root=args.data_root,
                output=directory,
                repeats=args.repeats,
                seed=args.seed,
                per_kind=args.per_kind,
                chunk_size=args.chunk_size,
            )
            summaries.append(result)
            print(f"DONE {dataset}: {result['split_half']}", flush=True)
        (args.output / "five_dataset_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
