"""Server-only split-half control sensitivity and blocked identity randomization.

The randomization unit is a condition identity within a recorded batch, not a
resplit or a biological replicate. Full gene profiles remain on the server.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gradpert.hashing import sha256_file
from scripts.analysis.five_dataset_diagnostics import DATASETS, split_half_assignments


def correlation_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=-1, keepdims=True)
    right = right - right.mean(axis=-1, keepdims=True)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    result = np.full(denominator.shape, np.nan)
    np.divide(np.sum(left * right, axis=-1), denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def holm_adjust(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p values must be within [0,1]")
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))).clip(0, 1)
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def blocked_null(blocks: list[dict], *, permutations: int, seed: int) -> dict:
    """Permute an entire repeated-profile identity jointly across all repeats.

    The score matrix is averaged over resplits before permutation; this is
    exactly the same statistic as reusing one bijection in every resplit.
    Weights give each covered condition equal mass, then each of its batches
    equal mass. The stated null is uniform identity pairing within each block.
    """
    if not blocks or permutations < 1:
        raise ValueError("nonempty blocks and positive permutations required")
    matrices = []
    weights = []
    for block in blocks:
        similarity = np.asarray(block["similarity"], dtype=float)
        weight = np.asarray(block["weights"], dtype=float)
        if similarity.ndim != 3 or similarity.shape[1] != similarity.shape[2]:
            raise ValueError("requires repeats by condition by condition")
        if similarity.shape[1] < 2 or len(weight) != similarity.shape[1]:
            raise ValueError("requires at least two conditions per block")
        if not np.isfinite(similarity).all() or np.any(weight < 0):
            raise ValueError("invalid similarity or weights")
        matrices.append(similarity.mean(axis=0))
        weights.append(weight)
    if not np.isclose(sum(w.sum() for w in weights), 1):
        raise ValueError("condition weights must sum to one")
    observed = float(sum(np.dot(np.diag(m), w) for m, w in zip(matrices, weights, strict=True)))
    rng = np.random.default_rng(seed)
    null = np.zeros(permutations)
    for m, w in zip(matrices, weights, strict=True):
        n = len(m)
        for i in range(permutations):
            null[i] += np.dot(m[np.arange(n), rng.permutation(n)], w)
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "excess": float(observed - null.mean()),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "p": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
        "null": null.tolist(),
        "null_hypothesis": (
            "uniform condition-identity pairing within batch and target-count blocks"
        ),
        "scope": (
            "conditional identity randomization; no independent biological-replicate inference"
        ),
    }


def control_pools(batches: np.ndarray, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    result = {}
    for batch in sorted(set(batches)):
        rows = np.flatnonzero(batches == batch)
        rng.shuffle(rows)
        middle = len(rows) // 2
        result[str(batch)] = (rows[:middle], rows[middle:])
    return result


def draw_matched_controls(
    pools: dict[str, tuple[np.ndarray, np.ndarray]], weights: dict[str, int], rng
) -> tuple[np.ndarray, np.ndarray] | None:
    a, b = [], []
    for batch, count in sorted(weights.items()):
        first, second = pools[batch]
        if not len(first) or not len(second):
            return None
        a.extend(rng.choice(first, count, replace=True).tolist())
        b.extend(rng.choice(second, count, replace=True).tolist())
    a, b = np.asarray(a), np.asarray(b)
    if np.intersect1d(a, b).size:
        raise AssertionError("control pools overlap")
    return a, b


def hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_compressed_public(path: Path, rows: list[dict]) -> None:
    fields = [
        x for x in dict.fromkeys(key for row in rows for key in row) if not x.endswith("sha256")
    ]
    with gzip.open(path, "wt") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def target_count(condition: str) -> int:
    return sum(t != "ctrl" for t in condition.split("+"))


def analyze_dataset(root: Path, name: str, old_root: Path, permutations: int, output: Path) -> dict:
    import anndata as ad
    from numba import njit
    from scipy import sparse

    path = root / name / DATASETS[name]
    canonical = path / "canonical/adata.h5ad"
    splitpath = path / "manifests/split.json"
    ctrlpath = path / "manifests/evaluation_controls.test.json"
    manifest = json.loads((path / "manifests/canonical.json").read_text())
    splits = json.loads(splitpath.read_text())
    draw_manifest = json.loads(ctrlpath.read_text())
    dataset = ad.read_h5ad(canonical, backed="r")
    try:
        conditions = dataset.obs.condition.astype(str).to_numpy()
        batches = dataset.obs.batch.astype(str).to_numpy()
        controls = dataset.obs.control.astype(bool).to_numpy()
        names = [x for x in splits["test_conditions"] if x != splits["control_condition_id"]]
        codes, halves, ordered = split_half_assignments(
            conditions, batches, names, repeats=20, seed=42
        )
        genes = manifest["n_expression_genes"]
        relevant = np.flatnonzero((codes >= 0) | controls)
        expressions = np.empty((len(relevant), genes), dtype=np.float32)
        lookup = np.full(dataset.n_obs, -1, dtype=int)
        lookup[relevant] = np.arange(len(relevant))
        for start in range(0, dataset.n_obs, 512):
            stop = min(start + 512, dataset.n_obs)
            selected = lookup[start:stop]
            keep = selected >= 0
            if not keep.any():
                continue
            block = dataset.X[start:stop, :genes]
            if sparse.issparse(block):
                block = block.toarray()
            expressions[selected[keep]] = np.asarray(block, dtype=np.float32)[keep]
        ctrlrows = np.flatnonzero(controls)
        ctrlx = expressions[lookup[ctrlrows]]
        ctrlbatch = batches[ctrlrows]
        row_index = {str(row): i for i, row in enumerate(dataset.obs_names)}
        draws = {
            x["condition_id"]: [row_index[r] for r in x["ordered_row_ids"]]
            for x in draw_manifest["draws"]
        }
        ctrl_index = {int(row): i for i, row in enumerate(ctrlrows)}
        frozen = np.stack(
            [
                ctrlx[[ctrl_index[r] for r in draws[c]]].mean(axis=0, dtype=np.float64)
                for c in ordered
            ]
        )
        contexts = [dict(Counter(batches[draws[c]])) for c in ordered]
        old = {
            (r["condition"], r["repeat"]): r
            for r in json.loads((old_root / name / "split_half_conditions.json").read_text())
        }
        sums = np.zeros((20, 2, len(ordered), genes), dtype=float)
        counts = np.zeros((20, 2, len(ordered)), dtype=np.int64)

        @njit
        def aggregate(values, rowcodes, rowhalves, totals, sizes):
            for row in range(len(values)):
                code = rowcodes[row]
                if code < 0:
                    continue
                for repeat in range(20):
                    side = rowhalves[repeat, row]
                    sizes[repeat, side, code] += 1
                    for gene in range(values.shape[1]):
                        totals[repeat, side, code, gene] += values[row, gene]

        aggregate(expressions, codes[relevant], halves[:, relevant], sums, counts)
        valid = np.all(counts > 0, axis=(0, 1))
        means = sums / np.maximum(counts[..., None], 1)
        del sums
        rows = []
        errors = []
        for repeat in range(20):
            pools = control_pools(ctrlbatch, 100042 + repeat)
            rng = np.random.default_rng(200042 + repeat)
            for code, c in enumerate(ordered):
                rec = {
                    "dataset": name,
                    "condition": c,
                    "repeat": repeat,
                    "seed": 42 + repeat,
                    "truth_cells": int(counts[repeat, :, code].sum()),
                    "left_cells": int(counts[repeat, 0, code]),
                    "right_cells": int(counts[repeat, 1, code]),
                    "frozen_300": None,
                    "shared_a_300": None,
                    "shared_b_300": None,
                    "shared_symmetric_300": None,
                    "disjoint_300": None,
                    "disjoint_ab_300": None,
                    "disjoint_ba_300": None,
                }
                if not valid[code]:
                    rec["unavailable_reason"] = "fewer_than_two_truth_cells"
                    rows.append(rec)
                    continue
                left, right = means[repeat, :, code]
                rec["frozen_300"] = float(
                    correlation_rows(left - frozen[code], right - frozen[code])
                )
                previous = old[(c, repeat)]["pearson_delta"]
                if previous is None:
                    raise ValueError("legacy availability mismatch")
                errors.append(abs(previous - rec["frozen_300"]))
                draw = draw_matched_controls(pools, contexts[code], rng)
                if draw is None:
                    rec["unavailable_reason"] = "batch_without_two_distinct_controls"
                    rows.append(rec)
                    continue
                a, b = draw
                ca, cb = ctrlx[a].mean(axis=0, dtype=float), ctrlx[b].mean(axis=0, dtype=float)
                shared_a = float(correlation_rows(left - ca, right - ca))
                shared_b = float(correlation_rows(left - cb, right - cb))
                rec.update(
                    shared_a_300=shared_a,
                    shared_b_300=shared_b,
                    shared_symmetric_300=(shared_a + shared_b) / 2,
                    disjoint_300=float(
                        (
                            correlation_rows(left - ca, right - cb)
                            + correlation_rows(left - cb, right - ca)
                        )
                        / 2
                    ),
                    disjoint_ab_300=float(correlation_rows(left - ca, right - cb)),
                    disjoint_ba_300=float(correlation_rows(left - cb, right - ca)),
                    control_a_sha256=hash_indices(ctrlrows[a]),
                    control_b_sha256=hash_indices(ctrlrows[b]),
                    control_overlap=0,
                    unique_controls_a=len(set(a)),
                    unique_controls_b=len(set(b)),
                    unavailable_reason="",
                )
                rows.append(rec)
        if max(errors, default=0) > 1e-12:
            raise ValueError("frozen-300 reproduction failed")
        write_csv(output / f"{name}-control-sensitivity-long.csv", rows)
        write_compressed_public(output / f"{name}-control-sensitivity-long.csv.gz", rows)
        sensitivity = {}
        for metric in ["frozen_300", "shared_symmetric_300", "disjoint_300"]:
            values = [
                float(
                    np.median(
                        [r[metric] for r in rows if r["repeat"] == repeat and r[metric] is not None]
                    )
                )
                for repeat in range(20)
            ]
            sensitivity[metric] = {
                "per_split_median": values,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
            }
        sensitivity["coverage"] = {
            "total_conditions": len(ordered),
            "valid_conditions": int(valid.sum()),
            "disjoint_valid_conditions": len(
                {r["condition"] for r in rows if r["disjoint_300"] is not None}
            ),
            "rows": len(rows),
            "missing_rows": sum(r["disjoint_300"] is None for r in rows),
        }
        sensitivity["frozen_parity_max_abs"] = max(errors, default=0)
        del means, counts, frozen
        groups = defaultdict(list)
        for row in np.flatnonzero(codes >= 0):
            groups[(str(batches[row]), str(conditions[row]))].append(int(row))
        tests = []
        block_rows = []
        for total_cells in [4, 10]:
            candidate = defaultdict(list)
            for (batch, condition), members in sorted(groups.items()):
                if len(members) >= total_cells and np.count_nonzero(ctrlbatch == batch) >= 20:
                    candidate[(batch, target_count(condition))].append(
                        (condition, np.asarray(members))
                    )
            candidate = {k: v for k, v in candidate.items() if len(v) >= 2}
            condition_batches = Counter(c for items in candidate.values() for c, _ in items)
            covered = len(condition_batches)
            null_blocks = []
            same_scores = defaultdict(list)
            for block_number, ((batch, kind), items) in enumerate(sorted(candidate.items())):
                size = len(items)
                matrices = np.empty((20, size, size))
                weights = np.array([1 / covered / condition_batches[c] for c, _ in items])
                ctrlmembers = np.flatnonzero(ctrlbatch == batch)
                rng = np.random.default_rng(300042 + total_cells * 1000 + block_number)
                for repeat in range(20):
                    control_order = rng.permutation(ctrlmembers)
                    ncontrol = len(ctrlmembers) // 2
                    ca = ctrlx[control_order[:ncontrol]].mean(axis=0, dtype=float)
                    cb = ctrlx[control_order[ncontrol : 2 * ncontrol]].mean(axis=0, dtype=float)
                    left, right = [], []
                    for c, members in items:
                        sampled = rng.choice(members, total_cells, replace=False)
                        n = total_cells // 2
                        a = expressions[lookup[sampled[:n]]].mean(axis=0, dtype=float) - ca
                        b = expressions[lookup[sampled[n:]]].mean(axis=0, dtype=float) - cb
                        left.append(a)
                        right.append(b)
                        r = float(correlation_rows(a, b))
                        same_scores[c].append((repeat, r))
                        block_rows.append(
                            {
                                "dataset": name,
                                "cells_per_condition_batch": total_cells,
                                "batch": batch,
                                "condition": c,
                                "target_count": kind,
                                "repeat": repeat,
                                "available_condition_cells": len(members),
                                "left_cells": n,
                                "right_cells": n,
                                "left_controls": ncontrol,
                                "right_controls": ncontrol,
                                "control_overlap": 0,
                                "same_condition_r": r,
                                "block_conditions": size,
                                "left_truth_sha256": hash_indices(sampled[:n]),
                                "right_truth_sha256": hash_indices(sampled[n:]),
                                "left_control_sha256": hash_indices(
                                    ctrlrows[control_order[:ncontrol]]
                                ),
                                "right_control_sha256": hash_indices(
                                    ctrlrows[control_order[ncontrol : 2 * ncontrol]]
                                ),
                            }
                        )
                    left, right = np.asarray(left), np.asarray(right)
                    left -= left.mean(axis=1, keepdims=True)
                    right -= right.mean(axis=1, keepdims=True)
                    ln, rn = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
                    if (ln == 0).any() or (rn == 0).any():
                        raise ValueError("zero variance profile in blocked test")
                    matrices[repeat] = np.clip(
                        (left / ln[:, None]) @ (right / rn[:, None]).T, -1, 1
                    )
                null_blocks.append({"similarity": matrices, "weights": weights})
            if not null_blocks:
                tests.append(
                    {
                        "dataset": name,
                        "cells_per_condition_batch": total_cells,
                        "status": "unavailable",
                        "reason": "no_batch_with_two_eligible_test_conditions",
                    }
                )
                continue
            test = blocked_null(null_blocks, permutations=permutations, seed=400042 + total_cells)
            per_split = [
                float(
                    np.mean(
                        [
                            np.mean([v for rep, v in same_scores[c] if rep == repeat])
                            for c in same_scores
                        ]
                    )
                )
                for repeat in range(20)
            ]
            test.update(
                dataset=name,
                cells_per_condition_batch=total_cells,
                status="complete",
                covered_conditions=covered,
                total_test_conditions=len(ordered),
                coverage=covered / len(ordered),
                condition_batch_pairs=sum(len(v) for v in candidate.values()),
                blocks=len(candidate),
                per_split=per_split,
                metric=(
                    "condition-equal mean Pearson, batch-equal within condition, "
                    "mean over 20 resplits"
                ),
            )
            tests.append(test)
        write_csv(output / f"{name}-batch-matched-long.csv", block_rows)
        write_compressed_public(output / f"{name}-batch-matched-long.csv.gz", block_rows)
        return {
            "dataset": name,
            "sensitivity": sensitivity,
            "tests": tests,
            "canonical_sha256": sha256_file(canonical),
            "split_sha256": sha256_file(splitpath),
            "controls_sha256": sha256_file(ctrlpath),
            "ordered_row_ids_sha256": hashlib.sha256(
                json.dumps(list(dataset.obs_names)).encode()
            ).hexdigest(),
            "genes": genes,
        }
    finally:
        dataset.file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--legacy-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    args = parser.parse_args()
    for path in [args.data_root, args.old_root, args.output]:
        if not path.resolve().is_relative_to(Path("/data/yilangliu")):
            raise ValueError("formal analysis stays on server")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("source must be clean")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "repeats": 20,
        "permutations": args.permutations,
        "seed": 42,
        "batch_samples": [4, 10],
        "minimum_batch_controls": 20,
        "control_draws": 300,
        "blocked_correction": "Holm within each prespecified five-dataset family",
    }
    receipt = {
        "status": "running",
        "source_commit": commit,
        "source_clean": True,
        "script_sha256": sha256_file(Path(__file__)),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "legacy_receipt_sha256": sha256_file(args.legacy_receipt),
        "old_manifest_sha256": sha256_file(args.old_root / "manifest.json"),
        "environment": {
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "threads": {
                k: os.environ.get(k)
                for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
            },
        },
        "results": {},
        "started_unix": time.time(),
    }
    legacy = json.loads(args.legacy_receipt.read_text())
    try:
        for name in DATASETS:
            print("START", name, flush=True)
            canonical = args.data_root / name / DATASETS[name] / "canonical/adata.h5ad"
            expected = legacy["datasets"][name]["canonical_adata_sha256"]
            if sha256_file(canonical) != expected:
                raise ValueError("canonical hash mismatch: " + name)
            result = analyze_dataset(
                args.data_root, name, args.old_root, args.permutations, args.output
            )
            receipt["results"][name] = result
            (args.output / f"{name}-summary.json").write_text(
                json.dumps(result, indent=2, allow_nan=False)
            )
            (args.output / "manifest.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False)
            )
            print("DONE", name, flush=True)
        for n in [4, 10]:
            family = [
                r["tests"][n == 10]
                for r in receipt["results"].values()
                if r["tests"][n == 10]["status"] == "complete"
            ]
            for item, p in zip(family, holm_adjust([x["p"] for x in family]), strict=True):
                item["p_holm"] = p
                item["family_size"] = len(family)
        receipt["status"] = "complete"
        receipt["completed_unix"] = time.time()
        receipt["pkl_files"] = len(list(args.output.rglob("*.pkl")))
        if receipt["pkl_files"]:
            raise ValueError("zero-PKL postcondition failed")
        summary = {
            k: {
                **v,
                "tests": [
                    {key: val for key, val in t.items() if key != "null"} for t in v["tests"]
                ],
            }
            for k, v in receipt["results"].items()
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        write_csv(
            args.output / "tests.csv",
            [
                {key: val for key, val in t.items() if key not in ["null", "per_split"]}
                for v in summary.values()
                for t in v["tests"]
            ],
        )
        outputs = {
            p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in args.output.iterdir()
            if p.is_file() and (p.suffix == ".gz" or p.name in ["summary.json", "tests.csv"])
        }
        receipt["transfer_allowlist"] = outputs
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
