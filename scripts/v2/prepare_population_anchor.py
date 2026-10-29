"""Build an immutable server-only R16 TRAIN treated-mean target sidecar."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.population_anchor import (
    COMPLETE_SCHEMA,
    COUNTS,
    SCHEMA,
    TARGET_RULE,
    checked_seal,
    server_path,
    validate_arrays,
    validate_manifest,
)
from scripts.v2 import response_repeatability as rr

REPEAT_SHA = "f7777481138e2dbd16c4d86e48919b57d5dc6a5e8eee7f786001e19ef7088a95"
SPEC_FIELDS = {
    "schema",
    "analysis_source",
    "census_complete",
    "repeatability_complete",
    "preprocessing",
    "sampling_bank",
    "output",
    "expected_counts",
    "cpu_budget",
}
CPU_BUDGET = dict(threads=1, max_allowed_cache_bytes=4 * 1024**3)


def sampling_metadata(binding, metadata, allowed):
    """Hash the old bank but decode only its four integer metadata arrays.

    Neither control_expression nor condition_targets is accessed/materialized.
    Their original bytes remain bound by the full NPZ SHA256.
    """
    manifest = checked_seal(dict(path=binding["path"], sha256=binding["sha256"]))
    if metadata.get("schema") != "fixed-response-group-anchor-bank-v1":
        raise ValueError("original six-array sampling bank schema required")
    relative = metadata["arrays"]["path"]
    if not isinstance(relative, str) or relative != Path(relative).name:
        raise ValueError("sampling arrays must be adjacent")
    arrays = checked_seal(
        dict(path=str(manifest.parent / relative), sha256=binding["arrays_sha256"])
    )
    payload = arrays.read_bytes()
    if rr.hashlib.sha256(payload).hexdigest() != binding["arrays_sha256"]:
        raise ValueError("sampling array bytes changed")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != 6 or set(archive.files) != {
            "allowed_gene_indices",
            "condition_targets",
            "condition_anchor_gene_indices",
            "control_expression",
            "control_slot_offsets",
            "control_slot_indices",
        }:
            raise ValueError("original sampling NPZ must have exactly six fields")
        axis = archive["allowed_gene_indices"].copy()
        offsets = archive["control_slot_offsets"].copy()
        slots = archive["control_slot_indices"].copy()
        anchors = archive["condition_anchor_gene_indices"].copy()
    p = len(metadata["condition_ids"])
    if (
        axis.dtype != np.int64
        or axis.ndim != 1
        or not np.array_equal(axis, allowed)
        or offsets.dtype != np.int64
        or offsets.shape != (p + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(slots)
        or np.any(np.diff(offsets) < 4)
        or np.any(np.diff(offsets) > 40)
        or slots.dtype != np.int64
        or slots.ndim != 1
        or np.any(slots < 0)
        or anchors.dtype != np.int64
        or anchors.shape != (p,)
        or np.any(anchors < 0)
    ):
        raise ValueError("sampling integer metadata axis/slots mismatch")
    return SimpleNamespace(
        manifest_path=manifest,
        manifest_sha256=binding["sha256"],
        arrays_path=arrays,
        arrays_sha256=binding["arrays_sha256"],
        condition_ids=tuple(metadata["condition_ids"]),
        identity=metadata["identity"],
        allowed_gene_indices=axis,
        control_slot_offsets=offsets,
    )


def treated_boundary(plan, metadata):
    """Only exact eligible selected TRAIN treated rows, never controls/val/test."""
    entries = plan["splits"]["selected_train"]["eligible"]
    train = set(metadata["train_row_indices"])
    conditions = metadata["condition_ids"]
    seen = set()
    for e in entries:
        rows = e["rows"]
        if (
            not isinstance(rows, list)
            or not 4 <= len(rows) <= 40
            or any(type(i) is not int or i < 0 or i >= len(conditions) for i in rows)
            or rows != sorted(set(rows))
            or not set(rows) <= train
            or seen.intersection(rows)
            or e["condition_id"] not in metadata["train_conditions"]
            or e["condition_id"] == metadata["control_condition_id"]
            or any(conditions[i] != e["condition_id"] for i in rows)
        ):
            raise ValueError("eligible complete selected TRAIN treated membership required")
        seen.update(rows)
    if not entries or len({e["condition_id"] for e in entries}) != len(entries):
        raise ValueError("unique nonempty TRAIN condition axis required")
    return entries, sorted(seen)


def build_treated_means(cache, used_rows, entries, plan, metadata, allowed):
    """Accumulate selected cells in FP64, preserving their empirical context weights."""
    expected_entries, expected_rows = treated_boundary(plan, metadata)
    if entries != expected_entries or list(used_rows) != expected_rows:
        raise ValueError("numeric treated-row whitelist mismatch")
    if (
        not isinstance(cache, np.ndarray)
        or cache.dtype != np.float32
        or cache.ndim != 2
        or cache.shape != (len(expected_rows), len(allowed))
        or not np.isfinite(cache).all()
        or np.any(cache < 0)
        or not isinstance(allowed, np.ndarray)
        or allowed.dtype != np.int64
        or allowed.ndim != 1
        or not len(allowed)
        or np.any(allowed < 0)
        or np.any(np.diff(allowed) <= 0)
    ):
        raise ValueError("finite nonnegative FP32 allowed-only treated cache required")
    pos = {row: i for i, row in enumerate(used_rows)}
    result = np.asarray(
        [cache[[pos[i] for i in e["rows"]]].mean(axis=0, dtype=np.float64) for e in entries],
        dtype=np.float32,
    )
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("float32 treated-mean overflow/nonfinite/negative")
    return result


def _membership(private):
    entries = [e for e in private["splits"]["selected_train"] if e["eligible"] is True]
    return dict(
        condition_ids_sha256=sha256_json([e["condition_id"] for e in entries]),
        ordered_treated_row_ids_sha256=sha256_json([r for e in entries for r in e["rows"]["rows"]]),
        selected_membership_sha256=sha256_json(entries),
        group_sizes=[len(e["rows"]["rows"]) for e in entries],
    )


def _check_spec(spec):
    if (
        not isinstance(spec, dict)
        or set(spec) != SPEC_FIELDS
        or spec["schema"] != "population-prediction-preparation-spec-v1"
        or spec["expected_counts"] != COUNTS
        or spec["cpu_budget"] != CPU_BUDGET
        or spec["census_complete"]["sha256"] != rr.CENSUS_SHA
        or spec["repeatability_complete"]["sha256"] != REPEAT_SHA
        or spec["preprocessing"]["sha256"] != rr.SCALE_SHA
        or not isinstance(spec["sampling_bank"], dict)
        or set(spec["sampling_bank"]) != {"path", "sha256", "arrays_sha256"}
    ):
        raise ValueError("exact R16 population preparation spec and accepted bindings required")
    for key in ("census_complete", "repeatability_complete", "preprocessing"):
        checked_seal(spec[key])
    bound = spec["sampling_bank"]
    checked_seal(dict(path=bound["path"], sha256=bound["sha256"]))
    bankmeta = json.loads(Path(bound["path"]).read_text())
    if bankmeta.get("arrays", {}).get("sha256") != bound["arrays_sha256"]:
        raise ValueError("sampling bank arrays binding mismatch")
    relative = bankmeta["arrays"]["path"]
    if not isinstance(relative, str) or relative != Path(relative).name:
        raise ValueError("sampling bank arrays must be adjacent")
    checked_seal(
        dict(path=str(Path(bound["path"]).parent / relative), sha256=bound["arrays_sha256"])
    )
    return bankmeta


def run(spec_path: Path, spec_sha: str):
    c = rr.census
    spec_path = checked_seal(dict(path=str(spec_path), sha256=spec_sha))
    spec = json.loads(spec_path.read_text())
    bankmeta = _check_spec(spec)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(k) != "1"
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("CPU-only one-thread environment required before startup")
    from gradpert.training.expression_policy import expression_policy

    source = rr._extra_source(spec["analysis_source"], expression_policy)
    root = server_path(spec["analysis_source"]["root"])

    def check_source():
        if rr._extra_source(spec["analysis_source"], expression_policy) != source:
            raise ValueError("population source changed")
        for relative, actual in (
            ("scripts/v2/prepare_population_anchor.py", Path(__file__)),
            (
                "src/gradpert/training/v2/population_anchor.py",
                Path(__import__(validate_manifest.__module__, fromlist=["x"]).__file__),
            ),
        ):
            raw = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{source['commit']}:{relative}"], timeout=30
            )
            if actual.resolve() != root / relative or rr.hashlib.sha256(
                raw
            ).hexdigest() != sha256_file(actual):
                raise ValueError("population executable differs from published source")

    check_source()
    original = json.loads(checked_seal(spec["census_complete"]).read_text())
    repeat = json.loads(checked_seal(spec["repeatability_complete"]).read_text())
    if (
        repeat.get("closure_verified") is not True
        or repeat.get("zero_PKL_within_output") is not True
        or repeat["source"]["commit"] != "809346f55f9025e84dfd1a3b2ca1643eda9182f4"
    ):
        raise ValueError("accepted repeatability closure/source required")
    membership_seal = dict(
        path=str(
            Path(spec["repeatability_complete"]["path"]).parent / "server-only/memberships.json"
        ),
        sha256=repeat["output_sha256"]["server-only/memberships.json"],
    )
    accepted_membership = json.loads(checked_seal(membership_seal).read_text())
    old_spec = json.loads(checked_seal(original["input_seals"][0]).read_text())
    historical = json.loads(checked_seal(old_spec["historical_run_manifest"]).read_text())
    summary_seal = dict(
        path=str(Path(spec["census_complete"]["path"]).parent / "summary.json"),
        sha256=original["outputs_sha256"]["summary.json"],
    )
    old_summary = json.loads(checked_seal(summary_seal).read_text())
    seals = [
        dict(path=str(spec_path), sha256=spec_sha),
        spec["analysis_source"]["publication"],
        spec["census_complete"],
        spec["repeatability_complete"],
        spec["preprocessing"],
        membership_seal,
        summary_seal,
        *original["input_seals"],
        *repeat["input_seals"],
    ]
    for seal in seals:
        checked_seal(seal)
    output_path = server_path(spec["output"], exists=False)
    if any(
        output_path.is_relative_to(p)
        for p in (
            Path(spec["sampling_bank"]["path"]).parent.parent,
            Path(spec["repeatability_complete"]["path"]).parent,
        )
    ):
        raise ValueError("output must preserve accepted bank/repeatability trees")
    output = rr._claim_output(spec, original)
    data = None
    reader = None
    audit = c.new_audit()
    try:
        private = output / "server-only"
        private.mkdir()
        layout = DatasetLayout(c.DATA_ROOT, c.DATASET, c.PROTOCOL)
        if (
            json.loads(layout.canonical_manifest.read_text())["preprocessing_manifest_sha256"]
            != rr.SCALE_SHA
            or json.loads(checked_seal(spec["preprocessing"]).read_text())["report"][
                "expression_scale_action"
            ]
            != "normalize_total_4000_then_log1p"
        ):
            raise ValueError("canonical expression scale differs")
        stat = c._file_stat(layout.canonical_adata)
        with c.metadata_only_guard(c.CanonicalTrainingData, audit):
            data = c.CanonicalTrainingData(
                dataset_id=c.DATASET, protocol_id=c.PROTOCOL, data_root=c.DATA_ROOT, run_seed=1
            )
            selection = c.apply_train_selection(data, c.SELECTION, c.SELECTION_SHA)
            if selection != original["selection_receipt"]:
                raise ValueError("native selected TRAIN membership differs")
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
            plan = rr.make_plan(metadata)
            if plan["census_summary"] != old_summary:
                raise ValueError("accepted metadata census replay differs")
            membership, _ = rr.membership_receipts(plan, metadata)
            if rr._clean(membership) != accepted_membership:
                raise ValueError("accepted full membership replay differs")
            entries, rows = treated_boundary(plan, metadata)
            allowed, policy = expression_policy(
                tuple(data.expression_gene_ids),
                tuple(data.split.test_conditions),
                data.split.control_condition_id,
                enabled=True,
            )
            if (
                allowed is None
                or len(allowed) != COUNTS["allowed"]
                or policy != historical["data"]["training_expression_policy"]
                or policy["allowed_expression_gene_ids_sha256"] != rr.ALLOWED_SHA
                or policy["excluded_expression_gene_ids_sha256"] != rr.FORBIDDEN_SHA
                or len(entries) != COUNTS["conditions"]
                or len(rows) != COUNTS["treated_rows"]
            ):
                raise ValueError("frozen TRAIN population/allowed-excluded policy differs")
            bank = sampling_metadata(spec["sampling_bank"], bankmeta, allowed)
            # Reproduce the ORIGINAL bank identity, whose used rows included TA/TB controls.
            contexts = {data.context_ids[i] for i in rows}
            controls = {i for k in contexts for f in ("TA", "TB") for i in plan["controls"][k][f]}
            bank_used = sorted(set(rows) | controls)
            identity = dict(
                source=bank.identity["source"],
                data_manifest=sha256_json(data.manifest.payload())
                if hasattr(data.manifest, "payload")
                else sha256_file(layout.canonical_manifest),
                split_content_sha256=data.split.split_content_sha256,
                selection=selection,
                ordered_used_row_ids_sha256=sha256_json([data.row_ids[i] for i in bank_used]),
                allowed_policy=policy,
                counts=COUNTS,
                metadata_sha256=sha256_json(metadata),
            )
            if (
                identity != bank.identity
                or bankmeta["identity"] != identity
                or list(bank.condition_ids) != [e["condition_id"] for e in entries]
                or not np.array_equal(
                    np.diff(bank.control_slot_offsets), [len(e["rows"]) for e in entries]
                )
            ):
                raise ValueError(
                    "sampling bank/canonical/split/selection/metadata identity differs"
                )
            estimated = len(rows) * len(allowed) * 4
            if estimated > CPU_BUDGET["max_allowed_cache_bytes"]:
                raise ValueError("treated-only allowed cache exceeds 4GiB")
            atomic_json(
                output / "PRE_NUMERICAL_METADATA.json",
                dict(
                    schema="population-prediction-pre-numerical-v1",
                    source=source,
                    spec_sha256=spec_sha,
                    counts=COUNTS,
                    logical_cache_bytes=estimated,
                    numerical_read_calls=0,
                    audit_counts=audit.copy(),
                    metadata_sha256=identity["metadata_sha256"],
                    membership=_membership(accepted_membership),
                    scientific_complete=False,
                ),
            )
            if rr.importlib.metadata.version("anndata") != "0.13.2":
                raise ValueError("reviewed allowed-first AnnData version differs")
            reader = rr.AllowedReader(
                data._adata, rows, list(allowed), row_block=2048, expected_dtype=np.dtype("float32")
            )
            cache = reader.read()
            means = build_treated_means(
                cache, rows, entries, plan, metadata, np.asarray(allowed, dtype=np.int64)
            )
            validate_arrays(np.asarray(allowed, dtype=np.int64), means, bank)
            np.savez(
                private / "population_targets.npz",
                allowed_gene_indices=np.asarray(allowed, dtype=np.int64),
                treated_mean=means,
            )
            manifest = dict(
                schema=SCHEMA,
                mode="population_prediction",
                condition_ids=list(bank.condition_ids),
                sampling_bank=spec["sampling_bank"],
                identity=identity,
                arrays=dict(
                    path="population_targets.npz",
                    sha256=sha256_file(private / "population_targets.npz"),
                ),
                target_rule=TARGET_RULE,
                provenance=dict(
                    spec=dict(path=str(spec_path), sha256=spec_sha),
                    source=source,
                    publication=spec["analysis_source"]["publication"],
                    census_complete=spec["census_complete"],
                    repeatability_complete=spec["repeatability_complete"],
                    preprocessing=spec["preprocessing"],
                    memberships=membership_seal,
                ),
                membership=_membership(accepted_membership),
                numeric_read_audit=dict(
                    rows=len(rows),
                    columns=len(allowed),
                    logical_values=reader.logical_values,
                    calls=reader.calls,
                    read_row_ids=[data.row_ids[i] for i in rows],
                    row_ids_sha256=sha256_json([data.row_ids[i] for i in rows]),
                    metadata_sha256=identity["metadata_sha256"],
                    forbidden_calls=audit.copy(),
                ),
            )
            validate_manifest(manifest, bank, manifest["arrays"]["sha256"])
            atomic_json(private / "manifest.json", manifest)
            for seal in seals:
                checked_seal(seal)
            check_source()
            if (
                c._file_stat(layout.canonical_adata) != stat
                or sha256_json(metadata) != identity["metadata_sha256"]
                or any(audit.values())
                or any(output.rglob("*.pkl"))
                or sha256_json(list(data.row_ids)) != c.OBS_ORDER_SHA
                or sha256_json([data.row_ids[i] for i in data.train_row_indices])
                != c.SELECTED_IDS_SHA
                or sha256_file(bank.manifest_path) != bank.manifest_sha256
                or sha256_file(bank.arrays_path) != bank.arrays_sha256
            ):
                raise ValueError("source/input/metadata/zero-PKL closure failed")
            atomic_json(
                output / "summary.json",
                dict(
                    schema="population-prediction-preparation-summary-v1",
                    counts=COUNTS,
                    numerical_read_calls=reader.calls,
                    logical_values=reader.logical_values,
                    scientific_complete=False,
                ),
            )
            atomic_json(
                output / "COMPLETE.json",
                dict(
                    schema=COMPLETE_SCHEMA,
                    source=source,
                    spec_sha256=spec_sha,
                    closure_verified=True,
                    zero_PKL_within_output=True,
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
        atomic_json(
            output / "server-only/FAILURE_DETAILS.json",
            dict(type=type(error).__name__, message=str(error)),
        )
        atomic_json(
            output / "FAILURE.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                details="server-only",
                scientific_complete=False,
            ),
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
