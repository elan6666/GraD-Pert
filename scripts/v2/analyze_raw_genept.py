"""CPU-only spectra of sealed raw GenePT vectors, before learned projection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

# Set before NumPy/BLAS import; this entry point never initializes CUDA.
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402

from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.features.text_prior import verify_text_prior_npz  # noqa: E402
from gradpert.hashing import sha256_file, sha256_json  # noqa: E402


def spectrum(values, *, method="auto"):
    """Population covariance (1/N), no row normalization; cosine is separate."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("nonempty finite two-dimensional values required")
    if method not in {"auto", "covariance", "gram"}:
        raise ValueError("unknown spectrum method")
    n, d = x.shape
    mean = x.mean(axis=0)
    centered = x - mean
    trace = float(np.square(centered).sum() / n)
    mean_energy = float(mean @ mean)
    use_gram = method == "gram" or (method == "auto" and n < d)
    cov = (centered @ centered.T if use_gram else centered.T @ centered) / n
    eig = np.linalg.eigvalsh(cov)[::-1]
    tolerance = float(np.finfo(np.float64).eps * max(n, d) * max(float(eig[0]), 0.0))
    if np.any(eig < -max(tolerance, np.finfo(np.float64).tiny)):
        raise ValueError("covariance has materially negative eigenvalues")
    eig[eig <= tolerance] = 0.0
    eig = np.pad(eig[:d], (0, max(0, d - len(eig))))
    mass = float(eig.sum())
    p = eig[eig > 0] / mass if mass else np.empty(0)
    effective = float(np.exp(-(p * np.log(p)).sum())) if mass else 0.0
    limit = min(d, n - 1)
    norm = np.linalg.norm(x, axis=1)
    valid = norm > 0
    unit = x[valid] / norm[valid, None]
    valid_n = int(valid.sum())
    unit_sum = unit.sum(axis=0)
    pair_cos = (
        float(np.clip((unit_sum @ unit_sum - valid_n) / (valid_n * (valid_n - 1)), -1, 1))
        if valid_n > 1
        else None
    )
    total = trace + mean_energy
    result = {
        "n": n,
        "width": d,
        "rank_limit": limit,
        "effective_rank": effective if n > 1 else None,
        "normalized_effective_rank": effective / limit if limit else None,
        "relative_variance": trace / total if total else None,
        "absolute_variance": trace,
        "mean_energy": mean_energy,
        "total_energy": total,
        "top1_fraction": float(eig[0] / mass) if mass else None,
        "mean_pair_cosine": pair_cos,
        "cosine_valid_rows": valid_n,
        "cosine_zero_rows_excluded": n - valid_n,
        "spectrum_method": "gram" if use_gram else "covariance",
        "eigenvalue_roundoff_threshold": tolerance,
        "spectrum_trace": mass,
    }
    for percent in (90, 95, 99):
        result[f"k{percent}"] = (
            int(np.searchsorted(np.cumsum(eig) / mass, percent / 100) + 1) if mass else None
        )
    return result, eig


def read_axis(path):
    ids = tuple(Path(path).read_text().splitlines())
    if not ids or any(not g or g != g.strip() for g in ids) or len(set(ids)) != len(ids):
        raise ValueError("gene axis must have unique nonempty exact IDs")
    return ids


def validate_groups(manifest, graph, expression, prior_sha256):
    data = manifest["data"]
    for key, ids in (
        ("graph_gene_order_sha256", graph),
        ("expression_gene_order_sha256", expression),
    ):
        if data[key] != sha256_json(list(ids)):
            raise ValueError(f"{key} mismatch")
    if data["genept_sha256"] != prior_sha256:
        raise ValueError("genept_sha256 mismatch")
    if not set(expression).issubset(graph):
        raise ValueError("expression axis is not contained in graph")
    policy = data["training_expression_policy"]
    if policy.get("exclude_test_target_expression") is not True:
        raise ValueError("active expression exclusion policy required")
    if policy["expression_gene_order_sha256"] != sha256_json(list(expression)):
        raise ValueError("policy expression axis mismatch")
    excluded = policy["excluded_expression_gene_ids"]
    if not isinstance(excluded, list) or any(not isinstance(g, str) for g in excluded):
        raise ValueError("invalid excluded expression IDs")
    unseen_set = set(excluded)
    unseen = tuple(g for g in expression if g in unseen_set)
    seen = tuple(g for g in expression if g not in unseen_set)
    if list(unseen) != excluded or not seen or not unseen:
        raise ValueError("excluded IDs must be nonempty proper subset in expression order")
    for key, ids in (
        ("excluded_expression_gene_ids_sha256", unseen),
        ("allowed_expression_gene_ids_sha256", seen),
    ):
        if policy[key] != sha256_json(list(ids)):
            raise ValueError(f"{key} mismatch")
    return {"graph_all": graph, "seen_expression": seen, "unseen_expression": unseen}


def load_groups(
    prior,
    prior_sha256,
    graph_path,
    expression_path,
    manifest_path,
    manifest_sha256,
    *,
    width=2048,
    source_count=17730,
):
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("run manifest hash mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    graph, expression = read_axis(graph_path), read_axis(expression_path)
    groups = validate_groups(manifest, graph, expression, prior_sha256)
    if sha256_file(prior) != prior_sha256:
        raise ValueError("prior hash mismatch")
    with np.load(prior, allow_pickle=False) as payload:
        source_ids = tuple(payload["genes"].tolist())
    all_prior = verify_text_prior_npz(
        prior,
        expected_sha256=prior_sha256,
        expected_gene_ids=source_ids,
        expected_embedding_width=width,
        expected_source_gene_count=source_count,
    )
    graph_prior = verify_text_prior_npz(
        prior,
        expected_sha256=prior_sha256,
        expected_gene_ids=graph,
        expected_embedding_width=width,
        expected_source_gene_count=source_count,
    )
    if graph_prior.gene_ids != graph:
        raise ValueError("sealed prior is missing graph genes")
    groups = {"source_all": source_ids, **groups}
    source_index = {g: i for i, g in enumerate(source_ids)}
    matrices = {
        name: all_prior.values[[source_index[g] for g in ids]] for name, ids in groups.items()
    }
    return matrices, groups, manifest, all_prior.model


def server_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path("/data/yilangliu")):
        raise ValueError("formal raw GenePT analysis requires /data/yilangliu paths")
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "prior",
        "prior-sha256",
        "graph-gene-ids",
        "expression-gene-ids",
        "run-manifest",
        "run-manifest-sha256",
        "output",
        "publication-receipt",
        "publication-receipt-sha256",
    ):
        parser.add_argument(f"--{field}", required=True)
    args = parser.parse_args(argv)
    paths = {
        name: server_path(getattr(args, name))
        for name in (
            "prior",
            "graph_gene_ids",
            "expression_gene_ids",
            "run_manifest",
            "publication_receipt",
            "output",
        )
    }
    if paths["output"].exists():
        raise FileExistsError("output must be a fresh directory")
    source = inspect_source_identity(
        server_path(Path(__file__).resolve().parents[2]),
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert",
        publication_receipt=paths["publication_receipt"],
        expected_publication_receipt_sha256=args.publication_receipt_sha256,
    )
    input_hashes = {key: sha256_file(value) for key, value in paths.items() if key != "output"}
    matrices, groups, manifest, model = load_groups(
        paths["prior"],
        args.prior_sha256,
        paths["graph_gene_ids"],
        paths["expression_gene_ids"],
        paths["run_manifest"],
        args.run_manifest_sha256,
    )
    counts = {k: len(v) for k, v in groups.items()}
    if counts != {
        "source_all": 17730,
        "graph_all": 6506,
        "seen_expression": 4775,
        "unseen_expression": 225,
    }:
        raise ValueError(f"frozen Jurkat group counts mismatch: {counts}")
    summaries, spectra = [], []
    for name, values in matrices.items():
        summary, eig = spectrum(values)
        summaries.append(
            {
                "group": name,
                "gene_order_sha256": sha256_json(list(groups[name])),
                "raw_float32_matrix_sha256": hashlib.sha256(
                    np.ascontiguousarray(values, dtype=np.float32).tobytes(order="C")
                ).hexdigest(),
                **summary,
            }
        )
        spectra.extend(
            {"group": name, "component": i + 1, "eigenvalue": float(v)} for i, v in enumerate(eig)
        )
    if input_hashes != {key: sha256_file(value) for key, value in paths.items() if key != "output"}:
        raise ValueError("input file changed during analysis")
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("summary.csv", summaries), ("eigenvalues.csv", spectra)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    receipt = {
        "schema": "raw-genept-spectrum-v1",
        "analysis_source": source.payload(),
        "method_script_sha256": sha256_file(__file__),
        "numpy_version": np.__version__,
        "training_source": manifest.get("source"),
        "training_run_id": manifest.get("run_id"),
        "input_paths": {k: str(v) for k, v in paths.items() if k != "output"},
        "input_file_sha256": input_hashes,
        "prior_model": model,
        "method": (
            "raw before learned projection; float64 centered population covariance; "
            "eigenvalues <= eps*max(N,d)*lambda_max treated as roundoff; "
            "mean pair cosine excludes zero vectors; no expression values/checkpoints read"
        ),
        "thread_environment": {
            k: os.environ[k]
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "groups": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"output": str(output), "groups": counts}))


if __name__ == "__main__":
    main()
