"""Observed-only perturbation effect atlas for five canonical datasets and cross-cell.

Runs on the data server with a clean published source commit. No model outputs,
per-cell matrices, or raw H5AD data are written to the result directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
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

PROGRAM_GO_IDS = {
    "dna_replication": "GO:0006260",
    "dna_repair": "GO:0006281",
    "g1s_transition": "GO:0000082",
    "g2m_transition": "GO:0000086",
    "ribosome_biogenesis": "GO:0042254",
    "translation": "GO:0006412",
    "apoptotic_process": "GO:0006915",
    "oxidative_stress": "GO:0034599",
}
GUIDE_COLUMNS = {
    "replogle_rpe1_essential": "guide_id",
    "nadig_jurkat": "sgID_AB",
    "nadig_hepg2": "sgID_AB",
}


def _mean(values: list[float]) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else None


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _percentile(values: list[float], q: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q)) if len(finite) else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def _top_gene_agreement(a: np.ndarray, b: np.ndarray, k: int = 20) -> tuple[float, float]:
    k = min(k, len(a))
    left = np.argpartition(np.abs(a), -k)[-k:]
    right = np.argpartition(np.abs(b), -k)[-k:]
    overlap = np.intersect1d(left, right, assume_unique=True)
    union = np.union1d(left, right)
    jaccard = float(len(overlap) / len(union))
    signs = (
        float(np.mean(np.sign(a[overlap]) == np.sign(b[overlap]))) if len(overlap) else float("nan")
    )
    return jaccard, signs


def _read_programs(gmt_path: Path) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    with gmt_path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            for label, go_id in PROGRAM_GO_IDS.items():
                if f"({go_id})" in parts[0]:
                    selected[label] = set(parts[2:])
    missing = set(PROGRAM_GO_IDS) - set(selected)
    if missing:
        raise ValueError(f"missing required GO programs: {sorted(missing)}")
    return selected


def _effective_rank(vectors: np.ndarray, *, seed: int, cap: int = 500) -> float | None:
    if len(vectors) < 2:
        return None
    if len(vectors) > cap:
        chosen = np.random.default_rng(seed).choice(len(vectors), cap, replace=False)
        vectors = vectors[chosen]
    centered = vectors - vectors.mean(axis=0)
    gram = centered @ centered.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = eigenvalues.sum()
    if total <= 1e-12:
        return None
    probabilities = eigenvalues[eigenvalues > 0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _norman_interactions(
    deltas: dict[str, np.ndarray], halves: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict:
    singles = {}
    single_halves = {}
    for condition, vector in deltas.items():
        parts = tuple(piece for piece in condition.split("+") if piece != "ctrl")
        if len(parts) == 1:
            singles[parts[0]] = vector
            if condition in halves:
                single_halves[parts[0]] = halves[condition]
    coefficients = []
    fitted_residuals = []
    additive_residuals = []
    half_residual_agreement = []
    for condition, observed in deltas.items():
        parts = tuple(piece for piece in condition.split("+") if piece != "ctrl")
        if len(parts) != 2 or parts[0] not in singles or parts[1] not in singles:
            continue
        design = np.column_stack((singles[parts[0]], singles[parts[1]]))
        coefficient, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
        if rank < 2:
            continue
        coefficients.append(coefficient.tolist())
        fitted_residuals.append(
            float(
                np.linalg.norm(observed - design @ coefficient)
                / max(np.linalg.norm(observed), 1e-12)
            )
        )
        additive_residuals.append(
            float(
                np.linalg.norm(observed - design.sum(axis=1)) / max(np.linalg.norm(observed), 1e-12)
            )
        )
        if condition in halves and all(name in single_halves for name in parts):
            residual = []
            for side in (0, 1):
                x = np.column_stack((single_halves[parts[0]][side], single_halves[parts[1]][side]))
                y = halves[condition][side]
                c, _, r, _ = np.linalg.lstsq(x, y, rcond=None)
                if r == 2:
                    residual.append(y - x @ c)
            if len(residual) == 2:
                half_residual_agreement.append(_pearson(*residual))
    a = [entry[0] for entry in coefficients]
    b = [entry[1] for entry in coefficients]
    return {
        "fitted_double_conditions": len(coefficients),
        "coefficient_a_median": _median(a),
        "coefficient_b_median": _median(b),
        "fitted_relative_residual_median": _median(fitted_residuals),
        "unit_additive_relative_residual_median": _median(additive_residuals),
        "fitted_residual_split_half_pearson_median": _median(half_residual_agreement),
        "fitted_residual_split_half_conditions": len(half_residual_agreement),
    }


def _guide_agreement(
    adata: object,
    lines: np.ndarray,
    conditions: np.ndarray,
    controls: np.ndarray,
    control_means: dict[str, np.ndarray],
    *,
    column: str,
    genes: int,
    chunk: int,
) -> dict:
    guides = adata.obs[column].astype(str).to_numpy()
    valid = ~controls & ~np.isin(guides, ("", "nan", "None", "NA"))
    keys = [
        f"{lines[i]}{SEPARATOR}{conditions[i]}{SEPARATOR}{guides[i]}" for i in np.flatnonzero(valid)
    ]
    unique = sorted(set(keys))
    mapping = {key: i for i, key in enumerate(unique)}
    codes = np.full(adata.n_obs, -1, dtype=np.int32)
    codes[valid] = np.fromiter((mapping[key] for key in keys), dtype=np.int32)
    sums, counts = _aggregate_expression(adata, codes, len(unique), genes, chunk)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, key in enumerate(unique):
        if counts[i] < 10:
            continue
        line, condition, _ = key.split(SEPARATOR)
        grouped[f"{line}{SEPARATOR}{condition}"].append(sums[i] / counts[i] - control_means[line])
    correlations = []
    for values in grouped.values():
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                correlations.append(_pearson(values[i], values[j]))
    return {
        "guide_column": column,
        "eligible_conditions_with_two_guides": sum(len(v) >= 2 for v in grouped.values()),
        "guide_pairs": len(correlations),
        "guide_pair_pearson_delta_median": _median(correlations),
        "minimum_cells_per_guide": 10,
        "note": "Descriptive; guide and batch assignments may be confounded.",
    }


def _analyze(
    path: Path,
    *,
    genes: int,
    programs: dict[str, set[str]],
    guide_column: str | None,
    crosscell: bool,
    seed: int,
    chunk: int,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        lines = adata.obs["cell_line" if crosscell else "cell_type"].astype(str).to_numpy()
        conditions = adata.obs["condition"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        controls = adata.obs["control"].astype(int).to_numpy().astype(bool)
        valid = lines != "K562_adamson" if crosscell else np.ones(adata.n_obs, dtype=bool)
        keys, codes = _group_codes(lines, conditions, batches, controls, valid, seed)
        sums, counts = _aggregate_expression(adata, codes, 2 * len(keys), genes, chunk)
        total_sums = sums[0::2] + sums[1::2]
        total_counts = counts[0::2] + counts[1::2]
        controls_by_line: dict[str, np.ndarray] = {}
        control_counts: dict[str, int] = {}
        for i, key in enumerate(keys):
            parts = key.split(SEPARATOR)
            if len(parts) == 3 and parts[1] == "ctrl":
                line = parts[0]
                controls_by_line[line] = controls_by_line.get(line, np.zeros(genes)) + total_sums[i]
                control_counts[line] = control_counts.get(line, 0) + int(total_counts[i])
        controls_by_line = {
            line: value / control_counts[line] for line, value in controls_by_line.items()
        }
        deltas: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        half_deltas: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
        sizes: dict[str, dict[str, int]] = defaultdict(dict)
        for i, key in enumerate(keys):
            parts = key.split(SEPARATOR)
            if len(parts) != 2 or total_counts[i] == 0:
                continue
            line, condition = parts
            control = controls_by_line[line]
            deltas[line][condition] = total_sums[i] / total_counts[i] - control
            sizes[line][condition] = int(total_counts[i])
            if counts[2 * i] and counts[2 * i + 1]:
                half_deltas[line][condition] = (
                    sums[2 * i] / counts[2 * i] - control,
                    sums[2 * i + 1] / counts[2 * i + 1] - control,
                )

        gene_names = (
            adata.var["gene_name"].astype(str).to_numpy()[:genes]
            if "gene_name" in adata.var
            else np.asarray(adata.var_names.astype(str))[:genes]
        )
        program_indices = {
            name: np.flatnonzero(np.isin(gene_names, list(members)))
            for name, members in programs.items()
        }
        line_summaries = {}
        condition_details = {}
        for line in sorted(deltas):
            names = sorted(deltas[line])
            matrix = np.stack([deltas[line][name] for name in names])
            weights = np.asarray([sizes[line][name] for name in names], dtype=float)
            common = np.average(matrix, axis=0, weights=weights)
            residual = matrix - common
            effect_rms = np.linalg.norm(matrix, axis=1) / np.sqrt(genes)
            residual_rms = np.linalg.norm(residual, axis=1) / np.sqrt(genes)
            alignment = [_cosine(vector, common) for vector in matrix]
            repeatability = []
            overlaps = []
            sign_agreements = []
            rows = []
            for i, name in enumerate(names):
                halves = half_deltas[line].get(name)
                repeated = _pearson(*halves) if halves else float("nan")
                jaccard, sign = (
                    _top_gene_agreement(*halves) if halves else (float("nan"), float("nan"))
                )
                repeatability.append(repeated)
                overlaps.append(jaccard)
                sign_agreements.append(sign)
                rows.append(
                    {
                        "condition": name,
                        "cells": sizes[line][name],
                        "effect_rms": float(effect_rms[i]),
                        "specific_residual_rms": float(residual_rms[i]),
                        "common_alignment_cosine": alignment[i],
                        "split_half_pearson_delta": repeated,
                        "split_half_top20_jaccard": jaccard,
                        "split_half_top20_sign_agreement": sign,
                    }
                )
            program_changes = {
                name: {
                    "covered_genes": len(indices),
                    "median_abs_condition_mean_shift": _median(
                        np.abs(matrix[:, indices].mean(axis=1)).tolist()
                    )
                    if len(indices) >= 10
                    else None,
                    "common_mean_shift": float(common[indices].mean())
                    if len(indices) >= 10
                    else None,
                }
                for name, indices in program_indices.items()
            }
            batch_controls = {
                str(batch): int(np.sum(valid & (lines == line) & controls & (batches == batch)))
                for batch in np.unique(batches[valid & (lines == line)])
            }
            line_summaries[line] = {
                "rows": int(np.sum(valid & (lines == line))),
                "control_rows": control_counts[line],
                "conditions": len(names),
                "condition_cells_median": _median(weights.tolist()),
                "condition_cells_p10": _percentile(weights.tolist(), 0.1),
                "condition_cells_p90": _percentile(weights.tolist(), 0.9),
                "conditions_below_20_cells": int(np.sum(weights < 20)),
                "control_rows_by_batch": batch_controls,
                "effect_rms_median": _median(effect_rms.tolist()),
                "specific_residual_rms_median": _median(residual_rms.tolist()),
                "common_alignment_cosine_mean": _mean(alignment),
                "effective_rank_of_condition_effects_max500": _effective_rank(matrix, seed=seed),
                "split_half_pearson_delta_median": _median(repeatability),
                "split_half_top20_jaccard_median": _median(overlaps),
                "split_half_top20_sign_agreement_median": _median(sign_agreements),
                "programs": program_changes,
            }
            condition_details[line] = rows
            if not crosscell and any(
                len([piece for piece in name.split("+") if piece != "ctrl"]) == 2 for name in names
            ):
                line_summaries[line]["double_interactions"] = _norman_interactions(
                    deltas[line], half_deltas[line]
                )
        if guide_column:
            guide = _guide_agreement(
                adata,
                lines,
                conditions,
                controls,
                controls_by_line,
                column=guide_column,
                genes=genes,
                chunk=chunk,
            )
            for line in line_summaries:
                line_summaries[line]["guide_agreement"] = guide
        result = {
            "h5ad": str(path),
            "rows": adata.n_obs,
            "genes": genes,
            "obs_columns": list(adata.obs.columns),
            "layers": list(adata.layers.keys()),
            "gene_set_coverage": {name: len(indices) for name, indices in program_indices.items()},
            "cell_lines": line_summaries,
            "condition_details": condition_details,
        }
        return result, deltas
    finally:
        adata.file.close()


def _crosscell_summary(deltas: dict[str, dict[str, np.ndarray]]) -> dict:
    lines = ("K562", "RPE1", "jurkat", "hepg2")
    result = {}
    for target in lines:
        for source in lines:
            if target == source:
                continue
            shared = sorted(set(deltas[target]) & set(deltas[source]))
            correlations = []
            strength_ratios = []
            signs = []
            for name in shared:
                a, b = deltas[target][name], deltas[source][name]
                correlations.append(_pearson(a, b))
                strength_ratios.append(float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-12)))
                indices = np.argpartition(np.abs(a), -min(100, len(a)))[-min(100, len(a)) :]
                signs.append(float(np.mean(np.sign(a[indices]) == np.sign(b[indices]))))
            result[f"{target}<-{source}"] = {
                "shared_conditions": len(shared),
                "median_pearson_delta": _median(correlations),
                "median_target_over_source_effect_norm": _median(strength_ratios),
                "median_top100_target_gene_sign_agreement": _median(signs),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--crosscell-h5ad", type=Path, required=True)
    parser.add_argument("--crosscell-folds", type=Path, required=True)
    parser.add_argument("--go-gmt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to("/data/yilangliu"):
        parser.error("output must be a new directory under /data/yilangliu")
    if args.chunk < 1 or args.seed < 0:
        parser.error("chunk must be positive and seed nonnegative")
    repo = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        parser.error("source checkout must be clean")
    fold_path = args.crosscell_folds / "manifest.json"
    fold = json.loads(fold_path.read_text())
    if fold["source_sha256"] != sha256_file(args.crosscell_h5ad):
        parser.error("cross-cell source differs from sealed folds")
    programs = _read_programs(args.go_gmt)
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "source_commit": source_commit,
        "script_sha256": sha256_file(Path(__file__)),
        "go_gmt_sha256": sha256_file(args.go_gmt),
        "crosscell_fold_manifest_sha256": sha256_file(fold_path),
        "seed": args.seed,
        "chunk": args.chunk,
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
            result, _ = _analyze(
                root / "canonical/adata.h5ad",
                genes=manifest["n_expression_genes"],
                programs=programs,
                guide_column=GUIDE_COLUMNS.get(name),
                crosscell=False,
                seed=args.seed,
                chunk=args.chunk,
            )
            result["canonical_sha256"] = manifest["canonical_adata_sha256"]
            result["split_sha256"] = sha256_file(root / "manifests/split.json")
            (args.output / f"{name}.json").write_text(
                json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n"
            )
            receipt["datasets"][name] = {
                "result_sha256": sha256_file(args.output / f"{name}.json"),
                "rows": result["rows"],
                "genes": result["genes"],
            }
            print(f"DONE {name}", flush=True)
        print("START crosscell", flush=True)
        result, deltas = _analyze(
            args.crosscell_h5ad,
            genes=fold["gene_count"],
            programs=programs,
            guide_column=None,
            crosscell=True,
            seed=args.seed,
            chunk=args.chunk,
        )
        result["four_line_transfer"] = _crosscell_summary(deltas)
        result["source_sha256"] = fold["source_sha256"]
        (args.output / "crosscell.json").write_text(
            json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n"
        )
        receipt["datasets"]["crosscell"] = {
            "result_sha256": sha256_file(args.output / "crosscell.json"),
            "rows": result["rows"],
            "genes": result["genes"],
        }
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
