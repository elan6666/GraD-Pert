"""Server-only, metadata-first preparation of a fixed TRAIN response anchor bank."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2 import response_repeatability as rr

REPEAT_SHA = "f7777481138e2dbd16c4d86e48919b57d5dc6a5e8eee7f786001e19ef7088a95"
FIELDS = {
    "allowed_gene_indices",
    "condition_targets",
    "condition_anchor_gene_indices",
    "control_expression",
    "control_slot_offsets",
    "control_slot_indices",
}


def train_boundary(plan, metadata):
    """Return only eligible selected TRAIN and their TA/TB controls, never plan.used_rows."""
    entries = plan["splits"]["selected_train"]["eligible"]
    treated = {i for e in entries for i in e["rows"]}
    if not treated <= set(metadata["train_row_indices"]):
        raise ValueError("non-TRAIN treated row")
    contexts = {metadata["context_ids"][i] for i in treated}
    controls = {i for k in contexts for f in ("TA", "TB") for i in plan["controls"][k][f]}
    if treated & controls:
        raise ValueError("treated/control overlap")
    return entries, sorted(treated | controls)


def projection(delta, *, dimensions=256):
    delta = np.asarray(delta, dtype=np.float64)
    if (
        delta.ndim != 2
        or min(delta.shape) < 2
        or delta.shape[1] < dimensions
        or not np.isfinite(delta).all()
    ):
        raise ValueError("finite response matrix with sufficient allowed axis required")
    q, r = np.linalg.qr(
        np.random.Generator(np.random.PCG64(20261028)).standard_normal(
            (delta.shape[1], dimensions)
        ),
        mode="reduced",
    )
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)
    q *= np.sqrt(delta.shape[1] / dimensions)
    center = delta.mean(axis=0)
    targets = (delta - center) @ q
    rms = float(np.sqrt(np.mean(targets**2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("zero/nonfinite TRAIN target RMS")
    targets /= rms
    pairs = np.array([(i, j) for i in range(len(delta)) for j in range(i)], dtype=np.int64)
    if len(pairs) < 4096:
        raise ValueError("4096 unique TRAIN pairs required")
    rng = np.random.Generator(np.random.PCG64(20261029))
    pairs = pairs[rng.choice(len(pairs), 4096, replace=False)]
    original = np.sum((delta[pairs[:, 0]] - delta[pairs[:, 1]]) ** 2, axis=1)
    projected = np.sum(((delta[pairs[:, 0]] - delta[pairs[:, 1]]) @ q) ** 2, axis=1)
    denominator = float(np.sqrt(np.mean(original**2)))
    error = (
        float(np.sqrt(np.mean((projected - original) ** 2)) / denominator)
        if denominator
        else float("inf")
    )
    if not np.isfinite(error) or error > 0.25:
        raise ValueError("fixed projection distance relative RMS error exceeds .25")
    targets = targets.astype(np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("float32 target overflow")
    return targets, q, center, rms, error, sha256_json(pairs.tolist())


def build_bank(cache, used_rows, entries, plan, metadata, allowed, graph_genes, expression_genes):
    """Pure deterministic construction; control means use full matching TA union TB pools."""
    allowed = np.asarray(allowed)
    if (
        allowed.dtype != np.int64
        or np.any(np.diff(allowed) <= 0)
        or len(allowed) != cache.shape[1]
        or np.any(allowed < 0)
        or np.any(allowed >= len(expression_genes))
    ):
        raise ValueError("sorted unique int64 allowed expression axis required")
    if not np.isfinite(cache).all() or cache.dtype != np.float32:
        raise ValueError("finite canonical float32 cache required")
    expected_entries, expected_rows = train_boundary(plan, metadata)
    if entries != expected_entries or list(used_rows) != expected_rows:
        raise ValueError("numeric row boundary differs")
    pos = {row: i for i, row in enumerate(used_rows)}
    ids, contexts = metadata["row_ids"], metadata["context_ids"]
    pools = {k: sorted(set(v["TA"]) | set(v["TB"])) for k, v in plan["controls"].items()}
    delta, halves, slot_rows, offsets, anchors = [], [[], []], [], [0], []
    for entry in entries:
        p = entry["condition_id"]
        genes = set(p.split("+")) - {metadata["control_condition_id"]}
        if len(genes) != 1:
            raise ValueError("single target conditions required")
        gene = next(iter(genes))
        anchors.append(graph_genes.index(gene))

        def response(rows, fold=None):
            counts = Counter(contexts[i] for i in rows)
            treated = cache[[pos[i] for i in rows]].mean(axis=0, dtype=np.float64)
            control = sum(
                n
                / len(rows)
                * cache[[pos[i] for i in (plan["controls"][k][fold] if fold else pools[k])]].mean(
                    axis=0, dtype=np.float64
                )
                for k, n in counts.items()
            )
            return treated - control

        delta.append(response(entry["rows"]))
        for h, fold in enumerate(("TA", "TB")):
            d = response(entry[("A", "B")[h]], fold)
            # Raw own-target mask BEFORE random projection; never reconstruct excluded columns.
            d[np.array([expression_genes[int(i)] == gene for i in allowed])] = 0
            halves[h].append(d)
        for k, n in sorted(Counter(contexts[i] for i in entry["rows"]).items()):
            pool = sorted(pools[k], key=lambda i: (sha256_json([p, k, ids[i]]), i))
            if not pool:
                raise ValueError("empty matched control pool")
            slot_rows.extend(pool[j % len(pool)] for j in range(n))
        offsets.append(len(slot_rows))
    delta = np.asarray(delta)
    targets, q, center, rms, error, pair_hash = projection(delta)
    projected_halves = [np.asarray(h) @ q for h in halves]
    cross_s = float(np.mean(projected_halves[0] * projected_halves[1]))
    if not np.isfinite(cross_s) or cross_s <= 0:
        raise ValueError("fixed raw-ownmask projected cross-half S must be positive")
    unique = sorted(set(slot_rows))
    lookup = {row: i for i, row in enumerate(unique)}
    bank = dict(
        allowed_gene_indices=allowed,
        condition_targets=targets,
        condition_anchor_gene_indices=np.asarray(anchors, dtype=np.int64),
        control_expression=cache[[pos[i] for i in unique]],
        control_slot_offsets=np.asarray(offsets, dtype=np.int64),
        control_slot_indices=np.asarray([lookup[i] for i in slot_rows], dtype=np.int64),
    )
    return (
        bank,
        dict(Q=q, delta=delta, center=center),
        dict(
            target_rms_before_normalization=rms,
            distance_relative_RMS_error=error,
            pair_sha256=pair_hash,
            raw_ownmask_projected_cross_half_S=cross_s,
            control_unique=len(unique),
        ),
    )


def run(spec_path: Path, spec_sha: str):
    c = rr.census
    if sha256_file(spec_path) != spec_sha:
        raise ValueError("spec hash differs")
    spec = json.loads(spec_path.read_text())
    if (
        set(spec)
        != {
            "schema",
            "analysis_source",
            "census_complete",
            "repeatability_complete",
            "preprocessing",
            "output",
            "expected_counts",
            "cpu_budget",
        }
        or spec["schema"] != "fixed-response-group-anchor-preparation-v1"
    ):
        raise ValueError("complete anchor preparation spec required")
    if spec["expected_counts"] != dict(
        conditions=1322, treated_rows=47807, allowed=4775, forbidden=225
    ) or spec["cpu_budget"] != dict(threads=1, max_allowed_cache_bytes=4 * 1024**3):
        raise ValueError("fixed population and CPU budget required")
    if (
        spec["census_complete"]["sha256"] != rr.CENSUS_SHA
        or spec["repeatability_complete"]["sha256"] != REPEAT_SHA
        or spec["preprocessing"]["sha256"] != rr.SCALE_SHA
    ):
        raise ValueError("accepted scientific provenance differs")
    original = json.loads(c._checked(spec["census_complete"]).read_text())
    repeat = json.loads(c._checked(spec["repeatability_complete"]).read_text())
    output_path = c._server_path(spec["output"]).resolve()
    if output_path.is_relative_to(Path(spec["repeatability_complete"]["path"]).parent.resolve()):
        raise ValueError("output must not modify accepted repeatability directory")
    output = rr._claim_output(spec, original)
    audit = c.new_audit()
    reader = None
    data = None
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
            os.environ.get(k) != "1"
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        ):
            raise ValueError("CPU-only one-thread environment required before startup")
        from gradpert.training.expression_policy import expression_policy

        source = rr._extra_source(spec["analysis_source"], expression_policy)
        root = Path(spec["analysis_source"]["root"])
        relative = "scripts/v2/prepare_response_anchor.py"

        def check_source():
            if (
                rr._extra_source(spec["analysis_source"], expression_policy) != source
                or Path(__file__).resolve() != root.resolve() / relative
            ):
                raise ValueError("source identity changed")
            raw = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{source['commit']}:{relative}"], timeout=30
            )
            if rr.hashlib.sha256(raw).hexdigest() != sha256_file(Path(__file__)):
                raise ValueError("anchor executable differs from published source")

        check_source()
        seals = [
            dict(path=str(spec_path), sha256=spec_sha),
            spec["analysis_source"]["publication"],
            spec["census_complete"],
            spec["repeatability_complete"],
            spec["preprocessing"],
            *original["input_seals"],
            *repeat["input_seals"],
        ]
        for seal in seals:
            c._checked(seal)
        if (
            not repeat["closure_verified"]
            or not repeat["zero_PKL_within_output"]
            or repeat["source"]["commit"] != "809346f55f9025e84dfd1a3b2ca1643eda9182f4"
        ):
            raise ValueError("accepted repeatability closure/source required")
        old_spec = json.loads(Path(original["input_seals"][0]["path"]).read_text())
        historical = json.loads(c._checked(old_spec["historical_run_manifest"]).read_text())
        summary_seal = dict(
            path=str(Path(spec["census_complete"]["path"]).parent / "summary.json"),
            sha256=original["outputs_sha256"]["summary.json"],
        )
        seals.append(summary_seal)
        old_summary = json.loads(c._checked(summary_seal).read_text())
        layout = DatasetLayout(c.DATA_ROOT, c.DATASET, c.PROTOCOL)
        if (
            json.loads(layout.canonical_manifest.read_text())["preprocessing_manifest_sha256"]
            != rr.SCALE_SHA
            or json.loads(c._checked(spec["preprocessing"]).read_text())["report"][
                "expression_scale_action"
            ]
            != "normalize_total_4000_then_log1p"
        ):
            raise ValueError("native existing canonical scale differs")
        stat = c._file_stat(layout.canonical_adata)
        private = output / "server-only"
        private.mkdir()
        with c.metadata_only_guard(c.CanonicalTrainingData, audit):
            data = c.CanonicalTrainingData(
                dataset_id=c.DATASET, protocol_id=c.PROTOCOL, data_root=c.DATA_ROOT, run_seed=1
            )
            if (
                c.apply_train_selection(data, c.SELECTION, c.SELECTION_SHA)
                != original["selection_receipt"]
            ):
                raise ValueError("native selection differs")
            metadata = dict(
                row_ids=data.row_ids,
                condition_ids=data.condition_ids,
                context_ids=data.context_ids,
                control_pools=data.control_pools,
                train_row_indices=data.train_row_indices,
                train_conditions=data.split.train_conditions,
                val_conditions=data.split.val_conditions,
                control_condition_id=data.split.control_condition_id,
            )
            metadata_hash = sha256_json(metadata)
            plan = rr.make_plan(metadata)
            if plan["census_summary"] != old_summary:
                raise ValueError("complete metadata census replay differs")
            entries, rows = train_boundary(plan, metadata)
            allowed, policy = expression_policy(
                tuple(data.expression_gene_ids),
                tuple(data.split.test_conditions),
                data.split.control_condition_id,
                enabled=True,
            )
            if (
                allowed is None
                or len(allowed) != 4775
                or policy != historical["data"]["training_expression_policy"]
                or policy["allowed_expression_gene_ids_sha256"] != rr.ALLOWED_SHA
                or policy["excluded_expression_gene_ids_sha256"] != rr.FORBIDDEN_SHA
            ):
                raise ValueError("native allowed/forbidden axis differs")
            if (
                len(entries) != 1322
                or sum(len(e["rows"]) for e in entries) != 47807
                or max(len(e["rows"]) for e in entries) > 40
            ):
                raise ValueError("TRAIN bank population differs")
            # Bind exact fixed-half replay to accepted original metadata receipt.
            _, coverage = rr.membership_receipts(plan, metadata)
            prior_pre_seal = dict(
                path=str(
                    Path(spec["repeatability_complete"]["path"]).parent
                    / "PRE_NUMERICAL_METADATA.json"
                ),
                sha256=repeat["pre_numerical_sha256"],
            )
            seals.append(prior_pre_seal)
            if json.loads(c._checked(prior_pre_seal).read_text())["coverage"] != coverage:
                raise ValueError("accepted half partition differs")
            estimated = len(rows) * 4775 * 4
            pre = dict(
                schema="response-anchor-pre-numerical-v1",
                source=source,
                spec_sha256=spec_sha,
                conditions=len(entries),
                treated_rows=47807,
                used_rows=len(rows),
                allowed=4775,
                forbidden=225,
                logical_cache_bytes=estimated,
                numerical_read_calls=0,
                audit_counts=audit.copy(),
                scientific_complete=False,
            )
            atomic_json(output / "PRE_NUMERICAL_METADATA.json", pre)
            if estimated > spec["cpu_budget"]["max_allowed_cache_bytes"]:
                raise ValueError("allowed cache exceeds 4GiB")
            if rr.importlib.metadata.version("anndata") != "0.13.2":
                raise ValueError("reviewed AnnData version differs")
            reader = rr.AllowedReader(
                data._adata,
                rows,
                allowed.tolist(),
                row_block=2048,
                expected_dtype=np.dtype("float32"),
            )
            cache = reader.read()
            bank, transform, diagnostics = build_bank(
                cache,
                rows,
                entries,
                plan,
                metadata,
                np.asarray(allowed, dtype=np.int64),
                list(data.canonical_graph_gene_ids),
                list(data.expression_gene_ids),
            )
            np.savez(private / "anchor_bank.npz", **bank)
            np.savez(private / "transform.npz", **transform)
            atomic_json(
                private / "manifest.json",
                dict(
                    schema="fixed-response-group-anchor-bank-v1",
                    condition_ids=[e["condition_id"] for e in entries],
                    arrays=dict(
                        path="anchor_bank.npz", sha256=sha256_file(private / "anchor_bank.npz")
                    ),
                    identity=dict(
                        source=source,
                        data_manifest=rr.sha256_json(data.manifest.payload())
                        if hasattr(data.manifest, "payload")
                        else sha256_file(layout.canonical_manifest),
                        split_content_sha256=data.split.split_content_sha256,
                        selection=original["selection_receipt"],
                        ordered_used_row_ids_sha256=sha256_json([data.row_ids[i] for i in rows]),
                        allowed_policy=policy,
                        counts=spec["expected_counts"],
                        metadata_sha256=sha256_json(metadata),
                    ),
                    projection=dict(
                        seed=20261028,
                        pair_seed=20261029,
                        dimensions=256,
                        scaling="sqrt(A/256)",
                        transform_sha256=sha256_file(private / "transform.npz"),
                    ),
                    normalizer=diagnostics,
                    control_panel=dict(
                        rule="hash(condition,context,controlID) sorted cyclic per-context slots",
                        offsets_sha256=sha256_json(bank["control_slot_offsets"].tolist()),
                        unique_controls=diagnostics["control_unique"],
                        total_slots=len(bank["control_slot_indices"]),
                        reused_slots=len(bank["control_slot_indices"])
                        - diagnostics["control_unique"],
                        controls_are_train_TA_union_TB=True,
                    ),
                ),
            )
            for seal in seals:
                c._checked(seal)
            check_source()
            if (
                c._file_stat(layout.canonical_adata) != stat
                or sha256_json(metadata) != metadata_hash
                or any(audit.values())
                or any(output.rglob("*.pkl"))
                or sha256_json(list(data.row_ids)) != c.OBS_ORDER_SHA
                or sha256_json([data.row_ids[i] for i in data.train_row_indices])
                != c.SELECTED_IDS_SHA
            ):
                raise ValueError("source/input/metadata/zero-PKL closure failed")
            atomic_json(
                output / "summary.json",
                dict(
                    **diagnostics,
                    conditions=1322,
                    treated_slots=47807,
                    target_dimensions=256,
                    scientific_complete=False,
                ),
            )
            atomic_json(
                output / "COMPLETE.json",
                dict(
                    schema="fixed-response-group-anchor-preparation-complete-v1",
                    source=source,
                    spec_sha256=spec_sha,
                    closure_verified=True,
                    zero_PKL_within_output=True,
                    numerical_read_calls=reader.calls,
                    input_seals=seals,
                    outputs_sha256={
                        str(p.relative_to(output)): sha256_file(p)
                        for p in output.rglob("*")
                        if p.is_file()
                    },
                    transfer_allowlist=[
                        "PRE_NUMERICAL_METADATA.json",
                        "summary.json",
                        "COMPLETE.json",
                    ],
                    scientific_complete=False,
                ),
            )
    except Exception as error:
        (output / "server-only").mkdir(exist_ok=True)
        atomic_json(
            output / "server-only" / "FAILURE_DETAILS.json",
            dict(type=type(error).__name__, message=str(error)),
        )
        atomic_json(
            output / "FAILURE.json",
            dict(status="failed", error_type=type(error).__name__, details="server-only"),
        )
        raise
    finally:
        if data is not None:
            data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    run(args.spec, args.spec_sha256)


if __name__ == "__main__":
    main()
