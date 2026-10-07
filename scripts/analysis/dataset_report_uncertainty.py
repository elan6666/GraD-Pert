"""Server-only observational distributions and conditional randomization tests.

No biological-replicate inference is made. Candidate identities are permuted
jointly across repeated splits, preserving dependence between split results.
Only scalar summaries, labels and 2-D diagnostic coordinates are exported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from gradpert.hashing import sha256_file
from scripts.analysis.five_dataset_diagnostics import (
    DATASETS,
    perturbation_kind,
    select_visualization_rows,
)
from scripts.analysis.six_dataset_landscape import SEPARATOR, _group_codes, _pearson


def fdr_bh(values: list[float]) -> list[float]:
    """Benjamini-Hochberg q values in the original input order."""
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def retrieval_null(winners: np.ndarray, *, permutations: int, seed: int) -> dict:
    """Random bijections of reference identities, shared across all splits."""
    if winners.ndim != 2 or winners.shape[1] < 2:
        raise ValueError("retrieval requires repeats by at least two candidates")
    n = winners.shape[1]
    observed = float(np.mean(winners == np.arange(n)))
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [np.mean(winners == rng.permutation(n)[None, :]) for _ in range(permutations)]
    )
    return {
        "observed_mean_top1": observed,
        "random_expectation": 1 / n,
        "null": null.tolist(),
        "p": float((1 + np.sum(null >= observed)) / (permutations + 1)),
        "null_hypothesis": (
            "uniform random reference-identity bijection, conditional on similarities"
        ),
        "independent_biological_replicates": False,
    }


def split_summary(rows: list[dict], *, repeats: int) -> dict:
    """Keep conditions and repeated random partitions distinct."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    conditions = []
    for condition, records in grouped.items():
        indices = sorted(r["repeat"] for r in records)
        if indices != list(range(repeats)):
            raise ValueError(f"missing/duplicate split for {condition}")
        records = sorted(records, key=lambda r: r["repeat"])
        values = [r["pearson_delta"] for r in records]
        finite = np.asarray([v for v in values if v is not None], dtype=float)
        raw = np.asarray(
            [r["pearson_expression"] for r in records if r["pearson_expression"] is not None]
        )
        record = {
            "condition": condition,
            "truth_cells": records[0]["truth_cells"],
            "values": values,
            "expression_values": [r["pearson_expression"] for r in records],
            "valid_repeats": len(finite),
        }
        if len(finite):
            record.update(
                mean=float(finite.mean()),
                sd=float(finite.std(ddof=1)) if len(finite) > 1 else None,
                median=float(np.median(finite)),
                q25=float(np.quantile(finite, 0.25)),
                q75=float(np.quantile(finite, 0.75)),
                minimum=float(finite.min()),
                maximum=float(finite.max()),
                expression_median=float(np.median(raw)),
            )
        conditions.append(record)
    per_split = []
    for repeat in range(repeats):
        active = [r for r in rows if r["repeat"] == repeat and r["pearson_delta"] is not None]
        per_split.append(
            {
                "repeat": repeat,
                "seed": 42 + repeat,
                "conditions": len(active),
                "median_delta": float(np.median([r["pearson_delta"] for r in active])),
                "mean_delta": float(np.mean([r["pearson_delta"] for r in active])),
                "median_expression": float(np.median([r["pearson_expression"] for r in active])),
            }
        )
    return {
        "conditions": conditions,
        "per_split": per_split,
        "scope": (
            "frozen test conditions; fixed per-condition 300-control draws; split sensitivity only"
        ),
    }


def export_split_tables(root: Path, output: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["repeats"] != 20:
        raise ValueError("requires terminal completed 20-split input")
    result = {}
    for dataset in DATASETS:
        rows = json.loads((root / dataset / "split_half_conditions.json").read_text())
        result[dataset] = split_summary(rows, repeats=20)
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (output / f"{dataset}-split-long.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "seed", *fields])
            writer.writeheader()
            writer.writerows({"dataset": dataset, "seed": 42 + r["repeat"], **r} for r in rows)
        fields = [
            "condition",
            "truth_cells",
            "valid_repeats",
            "mean",
            "sd",
            "median",
            "q25",
            "q75",
            "minimum",
            "maximum",
        ]
        with (output / f"{dataset}-split-wide.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields + [f"split_{i + 1:02d}" for i in range(20)]
            )
            writer.writeheader()
            for r in result[dataset]["conditions"]:
                writer.writerow(
                    {
                        **{k: r.get(k) for k in fields},
                        **{f"split_{i + 1:02d}": v for i, v in enumerate(r["values"])},
                    }
                )
    return result


def analyze_landscape(
    path: Path, *, genes: int, crosscell: bool, permutations: int, split_manifest: Path | None
) -> dict:
    import anndata as ad
    from scipy import sparse

    data = ad.read_h5ad(path, backed="r")
    try:
        lines = data.obs["cell_line" if crosscell else "cell_type"].astype(str).to_numpy()
        conditions = data.obs["condition"].astype(str).to_numpy()
        batches = data.obs["batch"].astype(str).to_numpy()
        controls = data.obs["control"].astype(int).to_numpy().astype(bool)
        valid = lines != "K562_adamson" if crosscell else np.ones(data.n_obs, dtype=bool)
        expression = data.X[:, :genes]
        output = {}
        for line in sorted(set(lines[valid])):
            active = valid & (lines == line)
            keys, codes = _group_codes(lines, conditions, batches, controls, active, 42)
            positions = np.flatnonzero(active)
            selected = expression[positions]
            totals = []
            sizes = []
            # Aggregate one partition at a time; no repeat-by-cell-by-gene array.
            for repeat in range(20):
                _, codes = _group_codes(lines, conditions, batches, controls, active, 42 + repeat)
                counts = np.bincount(codes[active], minlength=2 * len(keys))
                indicator = sparse.csr_matrix(
                    (np.ones(len(positions)), (codes[active], np.arange(len(positions)))),
                    shape=(2 * len(keys), len(positions)),
                )
                sums = indicator @ selected
                totals.append(sums.toarray() if sparse.issparse(sums) else sums)
                sizes.append(counts)
            sums = totals[0][::2] + totals[0][1::2]
            counts = sizes[0][::2] + sizes[0][1::2]
            ctrl = [i for i, k in enumerate(keys) if len(k.split(SEPARATOR)) == 3]
            baseline = sums[ctrl].sum(axis=0) / counts[ctrl].sum()
            perturb = [
                i for i, k in enumerate(keys) if len(k.split(SEPARATOR)) == 2 and counts[i] > 0
            ]
            names = [keys[i].split(SEPARATOR)[1] for i in perturb]
            effects = np.stack([sums[i] / counts[i] - baseline for i in perturb])
            shared = effects.mean(axis=0)
            common = [
                {
                    "condition": name,
                    "cells": int(counts[i]),
                    "pearson_shared": _pearson(v, shared),
                    "pearson_leave_one_out": _pearson(
                        v, (len(effects) * shared - v) / (len(effects) - 1)
                    ),
                }
                for name, i, v in zip(names, perturb, effects, strict=True)
            ]
            eligible = [
                i
                for i in perturb
                if counts[i] >= 20 and all(c[2 * i] > 0 and c[2 * i + 1] > 0 for c in sizes)
            ]
            if len(eligible) > 500:
                choice = np.sort(
                    np.random.default_rng(42).choice(len(eligible), 500, replace=False)
                )
                eligible = [eligible[i] for i in choice]
            winners = []
            retrieval = []
            within = []
            for repeat, (total, count) in enumerate(zip(totals, sizes, strict=True)):
                left = np.stack([total[2 * i] / count[2 * i] - baseline for i in eligible])
                right = np.stack([total[2 * i + 1] / count[2 * i + 1] - baseline for i in eligible])
                left -= left.mean(axis=1, keepdims=True)
                right -= right.mean(axis=1, keepdims=True)
                ln = np.linalg.norm(left, axis=1)
                rn = np.linalg.norm(right, axis=1)
                if np.any(ln < 1e-12) or np.any(rn < 1e-12):
                    raise ValueError("degenerate retrieval candidate")
                similarity = (left / ln[:, None]) @ (right / rn[:, None]).T
                ranking = np.argsort(-similarity, axis=1)
                winner = ranking[:, 0]
                winners.append(winner)
                n = len(eligible)
                retrieval.append(
                    {
                        "repeat": repeat,
                        "seed": 42 + repeat,
                        "top1": float(np.mean(winner == np.arange(n))),
                        "top10": float(
                            np.mean((ranking[:, : min(10, n)] == np.arange(n)[:, None]).any(axis=1))
                        ),
                        "queries": [
                            {
                                "condition": keys[i].split(SEPARATOR)[1],
                                "rank": int(np.flatnonzero(ranking[j] == j)[0]) + 1,
                                "matched_condition": keys[eligible[winner[j]]].split(SEPARATOR)[1],
                            }
                            for j, i in enumerate(eligible)
                        ],
                    }
                )
                for i in ctrl:
                    if counts[i] >= 20 and count[2 * i] and count[2 * i + 1]:
                        within.append(
                            {
                                "batch": keys[i].split(SEPARATOR)[2],
                                "repeat": repeat,
                                "dissimilarity": 1000
                                * (
                                    1
                                    - _pearson(
                                        total[2 * i] / count[2 * i],
                                        total[2 * i + 1] / count[2 * i + 1],
                                    )
                                ),
                            }
                        )
            validctrl = [i for i in ctrl if counts[i] >= 20]
            across = [
                {
                    "batch1": keys[i].split(SEPARATOR)[2],
                    "batch2": keys[j].split(SEPARATOR)[2],
                    "dissimilarity": 1000
                    * (1 - _pearson(sums[i] / counts[i], sums[j] / counts[j])),
                }
                for a, i in enumerate(validctrl)
                for j in validctrl[a + 1 :]
            ]
            mask = active & ~controls
            pc = np.unique(conditions[mask], return_inverse=True)[1]
            bc = np.unique(batches[mask], return_inverse=True)[1]
            nb = int(bc.max()) + 1
            nc = int(pc.max()) + 1
            N = len(pc)
            cp = np.bincount(pc, minlength=nc) / N
            bp = np.bincount(bc, minlength=nb) / N
            entropy = -np.sum(bp[bp > 0] * np.log(bp[bp > 0]))

            def info(labels, pc=pc, nb=nb, nc=nc, N=N, cp=cp, bp=bp, entropy=entropy):
                tab = np.bincount(pc * nb + labels, minlength=nc * nb).reshape(nc, nb) / N
                occupied = tab > 0
                return float(
                    np.sum(
                        tab[occupied]
                        * np.log(tab[occupied] / (cp[:, None] * bp[None, :])[occupied])
                    )
                    / entropy
                )

            batch = None
            if nb > 1:
                observed = info(bc)
                rng = np.random.default_rng(42)
                null = np.asarray([info(rng.permutation(bc)) for _ in range(permutations)])
                batch = {
                    "observed": observed,
                    "null": null.tolist(),
                    "excess": float(observed - null.mean()),
                    "p": float((1 + (null >= observed).sum()) / (permutations + 1)),
                    "null_hypothesis": (
                        "condition-independent batch labels at fixed label counts; "
                        "cell-level metadata randomization"
                    ),
                }
            output[line] = {
                "batch_information": batch,
                "within_control": within,
                "across_control": across,
                "common_response": common,
                "retrieval": retrieval,
                "retrieval_randomization": retrieval_null(
                    np.asarray(winners), permutations=permutations, seed=42
                ),
                "candidate_conditions": [keys[i].split(SEPARATOR)[1] for i in eligible],
                "control_reference": (
                    "all observed controls in the cell line (legacy landscape scope)"
                ),
                "n_genes": genes,
                "n_cells": int(active.sum()),
            }
            del totals, sizes
        if not crosscell and split_manifest:
            # Export only 2-D coordinates with anonymous sample indices.
            import umap
            from sklearn.decomposition import PCA

            sm = json.loads(split_manifest.read_text())
            kinds = np.asarray(
                [
                    perturbation_kind(c, bool(ctrl))
                    for c, ctrl in zip(conditions, controls, strict=True)
                ]
            )
            sampled = select_visualization_rows(kinds, conditions, per_kind=2500, seed=42)
            X = expression[sampled]
            X = X.toarray() if sparse.issparse(X) else np.asarray(X)
            X = np.asarray(X, dtype=np.float32)
            pca = PCA(
                n_components=min(50, len(sampled) - 1, genes),
                svd_solver="randomized",
                random_state=42,
            )
            pcs = pca.fit_transform(X)
            xy = umap.UMAP(
                n_neighbors=30, min_dist=0.3, n_components=2, random_state=42, n_jobs=1
            ).fit_transform(pcs[:, :30])
            part = {c: s for s in ["train", "val", "test"] for c in sm[f"{s}_conditions"]}
            output["embedding"] = {
                "pca_variance_2": float(pca.explained_variance_ratio_[:2].sum()),
                "points": [
                    {
                        "sample_index": j,
                        "pca1": float(pcs[j, 0]),
                        "pca2": float(pcs[j, 1]),
                        "umap1": float(xy[j, 0]),
                        "umap2": float(xy[j, 1]),
                        "kind": int(kinds[i]),
                        "split": "control" if controls[i] else part.get(conditions[i], "excluded"),
                        "batch": str(batches[i]),
                    }
                    for j, i in enumerate(sampled)
                ],
            }
        return output
    finally:
        data.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--legacy-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to("/data/yilangliu") or args.output.exists():
        parser.error("new server-only output required")
    source = Path(__file__).resolve().parents[2]
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        parser.error("clean source required")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    args.output.mkdir(parents=True)
    config = {
        "permutations": args.permutations,
        "repeats": 20,
        "seed": 42,
        "candidate_limit": 500,
        "minimum_candidate_cells": 20,
        "split_root": str(args.split_root),
        "legacy_receipt": str(args.legacy_receipt),
    }
    manifest = {
        "status": "running",
        "source_commit": commit,
        "source_clean": True,
        "config": config,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": __import__("hashlib")
        .sha256(json.dumps(config, sort_keys=True).encode())
        .hexdigest(),
        "split_manifest_sha256": sha256_file(args.split_root / "manifest.json"),
        "legacy_receipt_sha256": sha256_file(args.legacy_receipt),
        "thread_limits": {
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            ]
        },
    }
    try:
        splits = export_split_tables(args.split_root, args.output)
        (args.output / "split_summary.json").write_text(
            json.dumps(splits, indent=2, allow_nan=False)
        )
        receipt = json.loads(args.legacy_receipt.read_text())
        datasets = {}
        for name, item in receipt["datasets"].items():
            print("START", name, flush=True)
            expected_hash = item.get("canonical_adata_sha256")
            if expected_hash and sha256_file(Path(item["h5ad"])) != expected_hash:
                raise ValueError("canonical source hash mismatch: " + name)
            splitpath = None
            if name in DATASETS:
                splitpath = Path(item["h5ad"]).parents[1] / "manifests/split.json"
            result = analyze_landscape(
                Path(item["h5ad"]),
                genes=item["expression_genes"],
                crosscell=name == "crosscell",
                permutations=args.permutations,
                split_manifest=splitpath,
            )
            datasets[name] = result
            (args.output / f"{name}-landscape-details.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
            print("DONE", name, flush=True)
        family = []
        for dataset, lines in datasets.items():
            for line, item in lines.items():
                if line == "embedding":
                    continue
                for key in ["batch_information", "retrieval_randomization"]:
                    if item[key]:
                        family.append((dataset, line, key, item[key]))
        for record, q in zip(family, fdr_bh([v[3]["p"] for v in family]), strict=True):
            record[3]["q"] = q
        # Re-save after the declared 17-test family BH correction.
        for name, result in datasets.items():
            (args.output / f"{name}-landscape-details.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
        (args.output / "randomization_tests.json").write_text(
            json.dumps(
                [
                    {
                        "dataset": dataset_name,
                        "line": line_name,
                        "test": test_name,
                        **{x: v for x, v in item.items() if x != "null"},
                    }
                    for dataset_name, line_name, test_name, item in family
                ],
                indent=2,
            )
        )
        manifest["canonical_hashes"] = {
            name: item.get("canonical_adata_sha256") for name, item in receipt["datasets"].items()
        }
        manifest["status"] = "complete"
        manifest["pkl_files"] = len(list(args.output.rglob("*.pkl")))
        if manifest["pkl_files"]:
            raise ValueError("zero-PKL postcondition failed")
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
