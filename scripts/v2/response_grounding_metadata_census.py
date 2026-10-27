"""Sealed TRAIN/validation metadata census; never read expression or construct a model.

CLI materialization is server-only. Pure summarization and guarded native metadata
access can be exercised locally with synthetic metadata and a fake backed object.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.data import CanonicalTrainingData
from gradpert.training.v2.train_selection import apply_train_selection

SERVER = Path("/data/yilangliu")
DATA_ROOT = SERVER / "GraD-Pert/data-vnext-a942114"
DATASET = "nadig_jurkat"
PROTOCOL = "within_cell_unseen_single"
REPOSITORY = "https://github.com/elan6666/GraD-Pert.git"
SELECTION = (
    SERVER / "GraD-Pert/development/jurkat-cap40-v2-binding-20260930/training_selection.json"
)
SELECTION_SHA = "aa0916dfb91fff78d7a43f7d7325879a11c1a904e9bf31586420ca11cce84132"
SELECTED_IDS_SHA = "c8e79242336c194833ac329a807b7eeb76651707204b4ec1379f6a7483aa9a93"
MANIFEST_SHAS = {
    "canonical": "cb23e89cb7e7b0697e70a16230e6cd7c3dd53ec37d0152d75638008f10e1d95b",
    "split": "d8807af3f03f4ba6c57ddb4f77acdb915b3b180592bca4ba1619f3e49cbe6890",
}
DECLARED_H5AD_SHA = "f051343c191dcdb02cabfae66a6bfe10b2770752b6503802da33d4b74e845861"
SPLIT_CONTENT_SHA = "ecb2099cea9c231e9bf16c7db9bf94f2ac715df6bffece79caf0b661e863cdd0"
OBS_ORDER_SHA = "cc8029dd8bc6274e28dc90c149be181ab7cd8a18f1008d1fefc9db42ca7f637e"
BINS = ("0", "1", "2", "3", "ge4")


class MetadataIntegrityError(ValueError):
    """Count-only failure details; do not export row/condition/context IDs."""

    def __init__(self, reason: str, count: int = 1):
        self.details = {"reason": reason, "count": count}
        super().__init__(f"{reason} (count={count})")


def _bin(n: int) -> str:
    return str(n) if n < 4 else "ge4"


def summarize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Count complete split-condition x actual treated-context grids, including n0.

    groups contains server-only condition/context IDs and ordered membership hashes.
    summary/histogram/coverage contain no membership IDs. min2 halves are a counting
    eligibility flag, not a reliability or scientific-acceptance claim.
    """
    rows = tuple(metadata["row_ids"])
    conditions = tuple(metadata["condition_ids"])
    contexts = tuple(metadata["context_ids"])
    if not len(rows) == len(conditions) == len(contexts) or not rows:
        raise MetadataIntegrityError("metadata axes differ or are empty")
    if any(not isinstance(x, str) or not x for axis in (rows, conditions, contexts) for x in axis):
        raise MetadataIntegrityError("metadata IDs must be nonempty strings")
    if len(set(rows)) != len(rows):
        raise MetadataIntegrityError("duplicate canonical row IDs", len(rows) - len(set(rows)))
    control = metadata["control_condition_id"]
    expected_pools: dict[str, list[str]] = defaultdict(list)
    for row, condition, context in zip(rows, conditions, contexts, strict=True):
        if condition == control:
            expected_pools[context].append(row)
    pools = {k: tuple(v) for k, v in metadata["control_pools"].items()}
    if pools != {k: tuple(v) for k, v in expected_pools.items()}:
        raise MetadataIntegrityError("control pools differ from canonical control membership/order")
    train_axis = tuple(metadata["train_conditions"])
    val_axis = tuple(metadata["val_conditions"])
    for axis in (train_axis, val_axis):
        if not axis or len(set(axis)) != len(axis) or control in axis:
            raise MetadataIntegrityError("invalid split condition axis")
    if set(train_axis) & set(val_axis):
        raise MetadataIntegrityError("TRAIN and validation conditions overlap")
    train_set = set(train_axis)
    control_hashes = {k: sha256_json(list(v)) for k, v in pools.items()}
    train_indices = tuple(metadata["train_row_indices"])
    if (
        any(type(i) is not int or i < 0 or i >= len(rows) for i in train_indices)
        or tuple(sorted(set(train_indices))) != train_indices
        or any(conditions[i] not in train_set for i in train_indices)
    ):
        raise MetadataIntegrityError("selected TRAIN indices are duplicated/unordered/non-TRAIN")
    val_set = set(val_axis)
    indices_by_split = {
        "selected_train": train_indices,
        "validation": tuple(i for i, c in enumerate(conditions) if c in val_set),
    }
    result: dict[str, Any] = {
        "summary": {
            "schema_version": "response-grounding-metadata-census-1",
            "canonical_rows": len(rows),
            "canonical_metadata_row_order_sha256": sha256_json(list(rows)),
            "canonical_control_contexts": len(pools),
            "canonical_control_rows_unique": sum(map(len, pools.values())),
            "canonical_control_membership_sha256": sha256_json(
                [r for r, c in zip(rows, conditions, strict=True) if c == control]
            ),
            "splits": {},
            "interpretation": (
                "min2 halves require n>=4 both; not reliability/scientific acceptance"
            ),
            "coverage_note": (
                "conditions/contexts deduplicated within each bin; not additive across bins"
            ),
            "scientific_complete": False,
        },
        "histogram": [],
        "coverage": [],
        "groups": [],
    }
    for split, indices in indices_by_split.items():
        axis = train_axis if split == "selected_train" else val_axis
        observed = {conditions[i] for i in indices}
        if observed != set(axis):
            raise MetadataIntegrityError(
                f"{split} has zero-row/mismatched conditions", len(set(axis) ^ observed)
            )
        active_contexts = sorted({contexts[i] for i in indices})
        missing = set(active_contexts) - pools.keys()
        if missing:
            raise MetadataIntegrityError(f"{split} contexts lack controls", len(missing))
        members: dict[tuple[str, str], list[str]] = defaultdict(list)
        for i in indices:
            members[conditions[i], contexts[i]].append(rows[i])
        groups = []
        for condition in axis:
            for context in active_contexts:
                selected = members.get((condition, context), [])
                n, m = len(selected), len(pools[context])
                groups.append(
                    dict(
                        split=split,
                        condition_id=condition,
                        context_id=context,
                        treated_n=n,
                        control_n=m,
                        treated_bin=_bin(n),
                        control_bin=_bin(m),
                        can_form_two_disjoint_halves_min2=n >= 4 and m >= 4,
                        treated_ordered_row_ids_sha256=sha256_json(selected),
                        control_ordered_row_ids_sha256=control_hashes[context],
                    )
                )
        result["groups"].extend(groups)
        hist = Counter((g["treated_bin"], g["control_bin"]) for g in groups)
        for treated_bin in BINS:
            for control_bin in BINS:
                result["histogram"].append(
                    dict(
                        split=split,
                        treated_bin=treated_bin,
                        control_bin=control_bin,
                        groups=hist[treated_bin, control_bin],
                    )
                )
        for scope in ("cartesian", "observed"):
            for treated_bin in BINS:
                subset = [
                    g
                    for g in groups
                    if g["treated_bin"] == treated_bin
                    and (scope == "cartesian" or g["treated_n"] > 0)
                ]
                result["coverage"].append(
                    dict(
                        split=split,
                        scope=scope,
                        treated_bin=treated_bin,
                        groups=len(subset),
                        conditions=len({g["condition_id"] for g in subset}),
                        contexts=len({g["context_id"] for g in subset}),
                    )
                )
        eligible = [g for g in groups if g["can_form_two_disjoint_halves_min2"]]
        by_context = Counter(g["context_id"] for g in eligible)
        context_hist = Counter(str(min(by_context[k], 3)) for k in active_contexts)
        reasons = Counter(
            "both_lt4"
            if g["treated_n"] < 4 and g["control_n"] < 4
            else "treated_lt4"
            if g["treated_n"] < 4
            else "control_lt4"
            for g in groups
            if not g["can_form_two_disjoint_halves_min2"]
        )
        eligible_rows = sum(g["treated_n"] for g in eligible)
        stat = dict(
            treated_rows=len(indices),
            condition_axis_count=len(axis),
            contexts=len(active_contexts),
            observed_groups=len(members),
            cartesian_groups=len(groups),
            structural_n0_groups=len(groups) - len(members),
            control_rows_unique_in_split_contexts=sum(len(pools[k]) for k in active_contexts),
            ordered_treated_membership_sha256=sha256_json([rows[i] for i in indices]),
            condition_axis_sha256=sha256_json(list(axis)),
            eligible_groups=len(eligible),
            eligible_conditions=len({g["condition_id"] for g in eligible}),
            eligible_contexts=len({g["context_id"] for g in eligible}),
            eligible_treated_rows=eligible_rows,
            eligible_treated_row_fraction=eligible_rows / len(indices),
            ineligible_groups=len(groups) - len(eligible),
            ineligible_reasons=dict(reasons),
            eligible_condition_count_by_context_histogram={
                "0": context_hist["0"],
                "1": context_hist["1"],
                "2": context_hist["2"],
                "ge3": context_hist["3"],
            },
        )
        assert sum(g["treated_n"] for g in groups) == len(indices)
        assert sum(hist.values()) == len(axis) * len(active_contexts)
        assert sum(context_hist.values()) == len(active_contexts)
        result["summary"]["splits"][split] = stat
    return result


def new_audit() -> dict[str, int]:
    return dict(
        expression_loader_calls=0,
        batch_calls=0,
        model_constructions=0,
        objective_calls=0,
        checkpoint_reads=0,
        numerical_test_calls=0,
        updates=0,
    )


@contextmanager
def metadata_only_guard(data_type: type, audit: dict[str, int]) -> Iterator[None]:
    """Install before construction; restore class attributes even after a failure."""
    if not callable(getattr(data_type, "_read_expression_indices", None)):
        raise ValueError("native expression guard target absent")
    methods = {"_read_expression_indices": "expression_loader_calls"}
    methods.update(
        {
            name: "batch_calls"
            for name in (
                "iter_train_epoch",
                "iter_validation_epoch",
                "training_batch_identity_specs",
                "load_control_rows",
                "load_baseline_fit_data",
                "configure_system_optimizations",
                "_iter_cpu_batches",
                "_to_training_batch",
            )
            if hasattr(data_type, name)
        }
    )
    original = {name: (name in data_type.__dict__, getattr(data_type, name)) for name in methods}
    try:
        for name, counter in methods.items():

            def forbidden(
                *args: Any, _name: str = name, _counter: str = counter, **kwargs: Any
            ) -> Any:
                audit[_counter] += 1
                raise RuntimeError(f"metadata census forbids {_name}")

            setattr(data_type, name, forbidden)
        yield
    finally:
        for name, (owned, value) in original.items():
            if owned:
                setattr(data_type, name, value)
            else:
                delattr(data_type, name)


def collect_metadata(
    data_args: Mapping[str, Any],
    selection: Mapping[str, str],
    audit: dict[str, int],
    *,
    data_type: type = CanonicalTrainingData,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the native constructor/selection, retaining a handle even if init fails."""
    data = object.__new__(data_type)
    with metadata_only_guard(data_type, audit):
        try:
            data_type.__init__(data, **data_args)
            receipt = apply_train_selection(data, Path(selection["path"]), selection["sha256"])
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
            # Deep-copy metadata only; no H5AD view, X, layers, raw or obsm is read.
            metadata = json.loads(json.dumps(metadata))
            return metadata, receipt
        finally:
            data.close()


def _server_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(SERVER):
        raise ValueError("census paths must remain under /data/yilangliu")
    return path


def _checked(binding: Mapping[str, str]) -> Path:
    if set(binding) != {"path", "sha256"} or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]):
        raise ValueError("complete file/hash binding required")
    path = _server_path(binding["path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError("sealed input file/hash mismatch")
    return path


def _source(binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"root", "commit", "repository", "publication"}:
        raise ValueError("complete analysis source/publication binding required")
    if binding["repository"] != REPOSITORY or not re.fullmatch(r"[0-9a-f]{40}", binding["commit"]):
        raise ValueError("analysis source must name the exact published repository/commit")
    root = _server_path(binding["root"]).resolve(strict=True)
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("executing script does not belong to bound analysis source")
    publication = _checked(binding["publication"])
    method = json.loads(publication.read_text())["verification_method"]
    identity = inspect_source_identity(
        root,
        formal=True,
        expected_repository=REPOSITORY,
        publication_receipt=publication,
        expected_publication_receipt_sha256=binding["publication"]["sha256"],
        verification_method=method,
    ).payload()
    if identity["commit"] != binding["commit"] or not identity["formal_eligible"]:
        raise ValueError("analysis source identity differs")
    for relative, actual in (
        ("scripts/v2/response_grounding_metadata_census.py", Path(__file__)),
        (
            "src/gradpert/training/v2/train_selection.py",
            Path(__import__(apply_train_selection.__module__, fromlist=["x"]).__file__),
        ),
        (
            "src/gradpert/training/data.py",
            Path(__import__(CanonicalTrainingData.__module__, fromlist=["x"]).__file__),
        ),
    ):
        raw = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{identity['commit']}:{relative}"], timeout=30
        )
        if actual.resolve() != root / relative or hashlib.sha256(raw).hexdigest() != sha256_file(
            actual
        ):
            raise ValueError("executed script/native loader differs from published Git bytes")
    return identity


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    output: Path,
    result: Mapping[str, Any],
    receipt: dict[str, Any],
    closure_check: Callable[[], None],
) -> None:
    """No COMPLETE until all scalar files exist and the caller's closure gate passed."""
    if any(receipt["audit_counts"].values()):
        raise ValueError("zero forbidden calls required before COMPLETE")
    if any(output.iterdir()):
        raise FileExistsError(
            "census output must remain fresh; partial/completed output is not resumed"
        )
    atomic_json(output / "summary.json", result["summary"])
    _csv(output / "histogram.csv", result["histogram"])
    _csv(output / "coverage.csv", result["coverage"])
    private = output / "server-only"
    private.mkdir(exist_ok=False)
    _csv(private / "groups.csv", result["groups"])
    files = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*") if p.is_file()}
    closure_check()
    if any(receipt["audit_counts"].values()):
        raise ValueError("closure changed forbidden-call counts")
    atomic_json(
        output / "COMPLETE.json",
        {
            **receipt,
            "closure_verified": True,
            "outputs_sha256": files,
            "transfer_allowlist": [
                "COMPLETE.json",
                "summary.json",
                "histogram.csv",
                "coverage.csv",
            ],
            "server_only": ["server-only/groups.csv"],
            "scientific_complete": False,
        },
    )


def _file_stat(path: Path) -> tuple[int, int, int, int, int]:
    """Metadata-only mutation check; exclude atime, which a normal read can change."""
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def run(spec_path: Path, expected_sha: str) -> None:
    """Materialize only a fresh server run; a failure never overwrites a prior run."""
    if sha256_file(spec_path) != expected_sha:
        raise ValueError("spec file/hash mismatch")
    spec = json.loads(spec_path.read_text())
    if (
        set(spec)
        != {
            "schema_version",
            "analysis_source",
            "data",
            "historical_run_manifest",
            "historical_config",
            "historical_resolved_config",
            "selection",
            "output",
        }
        or spec["schema_version"] != "response-grounding-metadata-census-spec-1"
    ):
        raise ValueError("unsupported or incomplete census spec")
    output = _server_path(spec["output"])
    if output.resolve().is_relative_to(DATA_ROOT) or DATA_ROOT.is_relative_to(output.resolve()):
        raise ValueError("output must be independent of the canonical data root")
    for field in ("historical_run_manifest", "historical_config", "historical_resolved_config"):
        path = _server_path(spec[field]["path"]).resolve()
        if output.resolve().is_relative_to(path.parent) or path.is_relative_to(output.resolve()):
            raise ValueError("output must be independent of historical input directories")
    source_root = _server_path(spec["analysis_source"]["root"]).resolve()
    if output.resolve().is_relative_to(source_root) or output.resolve().is_relative_to(
        SELECTION.parent
    ):
        raise ValueError("output must be independent of source/accepted selection directories")
    output.mkdir(parents=True, exist_ok=False)
    audit = new_audit()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU census requires CUDA_VISIBLE_DEVICES='' before process start")
        source = _source(spec["analysis_source"])
        d = spec["data"]
        if set(d) != {
            "root",
            "dataset_id",
            "protocol_id",
            "canonical_manifest",
            "split_manifest",
            "manifest_evidence",
        } or (d["root"], d["dataset_id"], d["protocol_id"]) != (str(DATA_ROOT), DATASET, PROTOCOL):
            raise ValueError("fixed dataset/protocol/data root binding differs")
        layout = DatasetLayout(DATA_ROOT, DATASET, PROTOCOL)
        for name in ("canonical", "split"):
            if d[name + "_manifest"] != dict(
                path=str(layout.manifests / (name + ".json")), sha256=MANIFEST_SHAS[name]
            ):
                raise ValueError("manifest file-byte binding differs from frozen census input")
        seals = [
            dict(path=str(spec_path), sha256=expected_sha),
            spec["analysis_source"]["publication"],
            d["canonical_manifest"],
            d["split_manifest"],
            d["manifest_evidence"],
            spec["historical_run_manifest"],
            spec["historical_config"],
            spec["historical_resolved_config"],
            spec["selection"],
        ]
        for seal in seals:
            _checked(seal)
        evidence = json.loads(Path(d["manifest_evidence"]["path"]).read_text())
        if {x["path"]: x["file_sha256"] for x in evidence["files"]} != {
            d[k + "_manifest"]["path"]: MANIFEST_SHAS[k] for k in MANIFEST_SHAS
        }:
            raise ValueError("sealed manifest-byte evidence differs")
        historical = json.loads(Path(spec["historical_run_manifest"]["path"]).read_text())
        resolved = json.loads(Path(spec["historical_resolved_config"]["path"]).read_text())
        if (
            historical["config_sha256"] != spec["historical_config"]["sha256"]
            or historical["resolved_config_sha256"] != sha256_json(resolved)
            or resolved["dataset_id"] != DATASET
            or resolved["data"]["protocol_id"] != PROTOCOL
            or historical["data"]["run_seed"] != 1
        ):
            raise ValueError("historical run/config/resolved identity differs")
        if historical["source"]["dirty"] or not re.fullmatch(
            r"[0-9a-f]{40}", historical["source"]["commit"]
        ):
            raise ValueError("historical training source identity missing/dirty")
        if spec["selection"] != dict(path=str(SELECTION), sha256=SELECTION_SHA):
            raise ValueError("exact cap40 selection binding required")
        selected_manifest = json.loads(SELECTION.read_text())
        for name in ("analysis_selection", "equivalence"):
            seals.append(
                dict(
                    path=selected_manifest[name + "_path"],
                    sha256=selected_manifest[name + "_sha256"],
                )
            )
            _checked(seals[-1])
        canonical = json.loads(layout.canonical_manifest.read_text())
        split = json.loads((layout.manifests / "split.json").read_text())
        if (
            canonical["canonical_adata_sha256"] != DECLARED_H5AD_SHA
            or canonical["observation_order_sha256"] != OBS_ORDER_SHA
            or split["split_content_sha256"] != SPLIT_CONTENT_SHA
            or historical["data"]["canonical_sha256"] != DECLARED_H5AD_SHA
            or historical["data"]["split_sha256"] != SPLIT_CONTENT_SHA
        ):
            raise ValueError("declared H5AD/split/observation identity differs")
        _server_path(str(layout.canonical_adata))
        h5ad_stat = _file_stat(layout.canonical_adata)
        metadata, selection_receipt = collect_metadata(
            dict(dataset_id=DATASET, protocol_id=PROTOCOL, data_root=DATA_ROOT, run_seed=1),
            spec["selection"],
            audit,
        )
        if selection_receipt != historical["data"]["training_row_selection"]:
            raise ValueError("fresh native selection receipt differs from historical run")
        result = summarize_metadata(metadata)
        train = result["summary"]["splits"]["selected_train"]
        if (
            train["treated_rows"],
            train["condition_axis_count"],
            train["ordered_treated_membership_sha256"],
        ) != (47836, 1335, SELECTED_IDS_SHA) or result["summary"][
            "canonical_metadata_row_order_sha256"
        ] != OBS_ORDER_SHA:
            raise ValueError("fresh selected TRAIN/metadata order identity differs")

        def verify_closure() -> None:
            for seal in seals:
                _checked(seal)
            if (
                source != _source(spec["analysis_source"])
                or _file_stat(layout.canonical_adata) != h5ad_stat
            ):
                raise ValueError("source or H5AD metadata file stat changed during census")
            if sha256_json(metadata["row_ids"]) != OBS_ORDER_SHA:
                raise ValueError("metadata row order changed during census")
            if (
                sha256_json([metadata["row_ids"][i] for i in metadata["train_row_indices"]])
                != SELECTED_IDS_SHA
            ):
                raise ValueError("TRAIN metadata membership changed during census")
            val_axis = set(metadata["val_conditions"])
            val_ids = [
                r
                for r, c in zip(metadata["row_ids"], metadata["condition_ids"], strict=True)
                if c in val_axis
            ]
            if (
                sha256_json(val_ids)
                != result["summary"]["splits"]["validation"]["ordered_treated_membership_sha256"]
            ):
                raise ValueError("validation metadata membership changed during census")

        receipt = dict(
            schema_version="response-grounding-metadata-census-complete-1",
            source=source,
            historical_training_source=historical["source"],
            spec_sha256=expected_sha,
            input_seals=seals,
            selection_receipt=selection_receipt,
            declared_H5AD_sha256_not_recomputed=DECLARED_H5AD_SHA,
            fresh_metadata_order_sha256=result["summary"]["canonical_metadata_row_order_sha256"],
            audit_counts=audit,
            access_scope=(
                "backed obs/var/names/shape + sealed JSON/axis files; "
                "test split/anchor identity allowed; no numerical test/expression/model/batches"
            ),
            physical_expression_bytes_read_not_measured=True,
        )
        write_results(output, result, receipt, verify_closure)
    except Exception as error:
        private = output / "server-only"
        private.mkdir(exist_ok=True)
        atomic_json(private / "failure_details.json", {"error": str(error)})
        atomic_json(
            output / "FAILURE.json",
            dict(
                status="failed",
                error_type=type(error).__name__,
                error="Source/input/metadata/closure gate failed; details remain server-only",
                integrity_details=getattr(error, "details", None),
                audit_counts=audit,
                spec_sha256=expected_sha,
                scientific_complete=False,
            ),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    run(args.spec, args.spec_sha256)


if __name__ == "__main__":
    main()
