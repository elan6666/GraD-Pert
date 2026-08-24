"""Explicit, hash-pinned result catalog for notebooks and downstream analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from gradpert.config.matrix import DATASET_IDS, MODEL_IDS
from gradpert.config.schema import DatasetId, ModelId
from gradpert.contracts import (
    ResultCatalogEntry,
    ResultCatalogManifest,
    RunManifest,
    ServerArtifactPointer,
)
from gradpert.hashing import sha256_json

FINAL_METRIC_IDS = (
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
)
FINAL_METRIC_FIELDS = (
    "metric_id",
    "available",
    "macro_mean",
    "reason",
    "finite_condition_count",
    "total_condition_count",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolve_trusted_file(path: str | Path, trusted_root: str | Path) -> Path:
    candidate = Path(path)
    root = Path(trusted_root).resolve(strict=True)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"catalog file is outside trusted_root: {path}")
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"catalog target must be a regular non-symlink file: {path}")
    return resolved


def _read_hash_pinned_json(
    path: str | Path,
    *,
    expected_sha256: str,
    trusted_root: str | Path,
) -> tuple[Path, object]:
    resolved = _resolve_trusted_file(path, trusted_root)
    if _sha256_file(resolved) != expected_sha256:
        raise ValueError(f"catalog dependency SHA-256 mismatch: {path}")
    try:
        return resolved, json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"catalog dependency is not valid JSON: {path}") from exc


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


@dataclass(frozen=True)
class CatalogEntrySource:
    """Exact small-file inputs used to build one catalog entry."""

    run_manifest_path: str | Path
    server_pointer_path: str | Path
    metrics_path: str | Path


@dataclass(frozen=True)
class LoadedCatalogEntry:
    contract: ResultCatalogEntry
    run_manifest: RunManifest
    server_pointer: ServerArtifactPointer
    metrics_path: Path


@dataclass(frozen=True)
class ResultCatalog:
    manifest: ResultCatalogManifest
    entries: tuple[LoadedCatalogEntry, ...]
    file_sha256: str

    def select(
        self,
        *,
        model_id: ModelId | None = None,
        dataset_id: DatasetId | None = None,
    ) -> tuple[LoadedCatalogEntry, ...]:
        """Return deterministic matches without filesystem discovery."""

        return tuple(
            entry
            for entry in self.entries
            if (model_id is None or entry.contract.model_id == model_id)
            and (dataset_id is None or entry.contract.dataset_id == dataset_id)
        )

    def require_run(self, run_id: str) -> LoadedCatalogEntry:
        matches = [entry for entry in self.entries if entry.contract.run_id == run_id]
        if len(matches) != 1:
            raise KeyError(f"catalog does not contain exactly one run_id={run_id!r}")
        return matches[0]


@dataclass(frozen=True)
class FinalCatalogAudit:
    entry_count: int
    dataset_count: int
    model_count: int
    source_commit: str
    coordinates_sha256: str
    fairness_by_dataset: dict[str, dict[str, object]]

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "final-catalog-audit-v1",
            "entry_count": self.entry_count,
            "dataset_count": self.dataset_count,
            "model_count": self.model_count,
            "source_commit": self.source_commit,
            "coordinates_sha256": self.coordinates_sha256,
            "fairness_by_dataset": self.fairness_by_dataset,
        }


@dataclass(frozen=True)
class FinalCatalogPlan:
    manifest: ResultCatalogManifest
    audit: FinalCatalogAudit
    source_spec_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "final-catalog-plan-v1",
            "source_spec_sha256": self.source_spec_sha256,
            "catalog": self.manifest.model_dump(mode="json"),
            "audit": self.audit.payload(),
        }


def _read_final_metric_summary(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FINAL_METRIC_FIELDS:
            raise ValueError(f"final metrics summary schema mismatch: {path}")
        rows = list(reader)
    if len(rows) != len(FINAL_METRIC_IDS):
        raise ValueError(f"final metrics summary must contain three rows: {path}")
    observed_ids = tuple(row["metric_id"] for row in rows)
    if observed_ids != FINAL_METRIC_IDS:
        raise ValueError(f"final metrics summary registry/order mismatch: {path}")
    totals: set[int] = set()
    for row in rows:
        if row["available"] not in {"true", "false"}:
            raise ValueError(f"final metrics availability is invalid: {path}")
        try:
            finite = int(row["finite_condition_count"])
            total = int(row["total_condition_count"])
            value = None if row["macro_mean"] == "" else float(row["macro_mean"])
        except ValueError as error:
            raise ValueError(f"final metrics summary contains invalid numbers: {path}") from error
        if total <= 0 or not 0 <= finite <= total:
            raise ValueError(f"final metrics summary count is invalid: {path}")
        available = row["available"] == "true"
        if available != (value is not None) or (value is not None and not math.isfinite(value)):
            raise ValueError(f"final metrics availability/value mismatch: {path}")
        if available and (finite == 0 or value is None or abs(value) > 1.0 + 1e-12):
            raise ValueError(f"available final Pearson metric is invalid: {path}")
        if not available and finite != 0:
            raise ValueError(f"unavailable final metric has finite conditions: {path}")
        if not available and not row["reason"]:
            raise ValueError(f"unavailable final metric requires a reason: {path}")
        totals.add(total)
    if len(totals) != 1:
        raise ValueError(f"final metric denominators differ within run: {path}")
    return totals.pop()


def require_final_benchmark_catalog(catalog: ResultCatalog) -> FinalCatalogAudit:
    """Require the exact 45-run formal comparison surface used by notebooks."""

    expected_coordinates = {
        (model_id, dataset_id, 1) for model_id in MODEL_IDS for dataset_id in DATASET_IDS
    } | {("gradpert_b2", dataset_id, seed) for dataset_id in DATASET_IDS for seed in (2, 3, 4)}
    observed_coordinates = [
        (
            entry.run_manifest.model_id,
            entry.run_manifest.dataset_id,
            entry.run_manifest.run_seed,
        )
        for entry in catalog.entries
    ]
    if len(observed_coordinates) != len(set(observed_coordinates)):
        raise ValueError("final catalog contains duplicate model/dataset/seed coordinates")
    if set(observed_coordinates) != expected_coordinates:
        missing = sorted(expected_coordinates - set(observed_coordinates))
        extra = sorted(set(observed_coordinates) - expected_coordinates)
        raise ValueError(f"final catalog coordinate mismatch: missing={missing}, extra={extra}")

    commits = {entry.run_manifest.source_commit for entry in catalog.entries}
    if len(commits) != 1:
        raise ValueError("final catalog runs do not share one published source commit")
    fairness_by_dataset: dict[str, dict[str, object]] = {}
    for dataset_id in DATASET_IDS:
        entries = [
            entry for entry in catalog.entries if entry.run_manifest.dataset_id == dataset_id
        ]
        fairness = {
            (
                entry.run_manifest.protocol_id,
                entry.run_manifest.canonical_data_sha256,
                entry.run_manifest.split_content_sha256,
                entry.run_manifest.control_manifest_sha256,
            )
            for entry in entries
        }
        if len(fairness) != 1:
            raise ValueError(f"final catalog fairness hashes differ for dataset: {dataset_id}")
        native_configs = {
            entry.run_manifest.config_sha256
            for entry in entries
            if entry.run_manifest.model_id == "gradpert_b2"
        }
        if len(native_configs) != 1:
            raise ValueError(f"native seed configs differ for dataset: {dataset_id}")
        denominators = {_read_final_metric_summary(entry.metrics_path) for entry in entries}
        if len(denominators) != 1:
            raise ValueError(f"metric condition denominators differ for dataset: {dataset_id}")
        protocol_id, data_hash, split_hash, control_hash = next(iter(fairness))
        fairness_by_dataset[dataset_id] = {
            "run_count": len(entries),
            "protocol_id": protocol_id,
            "canonical_data_sha256": data_hash,
            "split_content_sha256": split_hash,
            "control_manifest_sha256": control_hash,
            "total_condition_count": denominators.pop(),
        }
    coordinates_payload = [
        {"model_id": model, "dataset_id": dataset, "run_seed": seed}
        for model, dataset, seed in sorted(expected_coordinates)
    ]
    return FinalCatalogAudit(
        entry_count=len(catalog.entries),
        dataset_count=len(DATASET_IDS),
        model_count=len(MODEL_IDS),
        source_commit=next(iter(commits)),
        coordinates_sha256=sha256_json(coordinates_payload),
        fairness_by_dataset=fairness_by_dataset,
    )


def build_catalog_entry(
    source: CatalogEntrySource,
    *,
    trusted_root: str | Path,
) -> ResultCatalogEntry:
    """Validate an evaluated run and hash its exact notebook-facing files."""

    root = Path(trusted_root).resolve(strict=True)
    run_path = _resolve_trusted_file(source.run_manifest_path, root)
    pointer_path = _resolve_trusted_file(source.server_pointer_path, root)
    metrics_path = _resolve_trusted_file(source.metrics_path, root)
    run = RunManifest.model_validate_json(run_path.read_text(encoding="utf-8"))
    pointer = ServerArtifactPointer.model_validate_json(pointer_path.read_text(encoding="utf-8"))
    if (
        run.status != "evaluated"
        or run.test_evaluations != 1
        or not run.formal_eligible
        or run.source_dirty
    ):
        raise ValueError(
            "catalog admits only evaluated runs with exactly one test evaluation, "
            "clean source, and formal eligibility"
        )
    if pointer.run_id != run.run_id or pointer.source_commit != run.source_commit:
        raise ValueError("server pointer identity does not match run manifest")
    return ResultCatalogEntry(
        run_id=run.run_id,
        model_id=run.model_id,
        dataset_id=run.dataset_id,
        run_manifest_path=str(run_path.relative_to(root)),
        run_manifest_sha256=_sha256_file(run_path),
        server_pointer_path=str(pointer_path.relative_to(root)),
        server_pointer_sha256=_sha256_file(pointer_path),
        metrics_path=str(metrics_path.relative_to(root)),
        metrics_sha256=_sha256_file(metrics_path),
    )


def seal_result_catalog(
    path: str | Path,
    *,
    catalog_id: str,
    entries: Sequence[ResultCatalogEntry],
) -> str:
    """Atomically write a deterministic small-file catalog and return its SHA-256."""

    manifest = ResultCatalogManifest(
        schema_version="result-catalog-v1",
        catalog_id=catalog_id,
        entries=sorted(entries, key=lambda item: item.run_id),
    )
    target = Path(path).resolve()
    _atomic_json(target, manifest.model_dump(mode="json"))
    return _sha256_file(target)


def load_result_catalog(
    path: str | Path,
    *,
    expected_file_sha256: str,
    trusted_root: str | Path,
) -> ResultCatalog:
    """Load one exact catalog and verify every referenced small file."""

    root = Path(trusted_root).resolve(strict=True)
    catalog_path, raw = _read_hash_pinned_json(
        path,
        expected_sha256=expected_file_sha256,
        trusted_root=root,
    )
    manifest = ResultCatalogManifest.model_validate(raw)
    loaded: list[LoadedCatalogEntry] = []
    for item in manifest.entries:
        _, raw_run = _read_hash_pinned_json(
            item.run_manifest_path,
            expected_sha256=item.run_manifest_sha256,
            trusted_root=root,
        )
        _, raw_pointer = _read_hash_pinned_json(
            item.server_pointer_path,
            expected_sha256=item.server_pointer_sha256,
            trusted_root=root,
        )
        metrics_path = _resolve_trusted_file(item.metrics_path, root)
        if _sha256_file(metrics_path) != item.metrics_sha256:
            raise ValueError(f"catalog dependency SHA-256 mismatch: {item.metrics_path}")
        run = RunManifest.model_validate(raw_run)
        pointer = ServerArtifactPointer.model_validate(raw_pointer)
        if (
            run.run_id != item.run_id
            or run.model_id != item.model_id
            or run.dataset_id != item.dataset_id
            or run.status != "evaluated"
            or run.test_evaluations != 1
            or not run.formal_eligible
            or run.source_dirty
        ):
            raise ValueError(f"catalog run identity/lifecycle mismatch: {item.run_id}")
        if pointer.run_id != run.run_id or pointer.source_commit != run.source_commit:
            raise ValueError(f"catalog server pointer identity mismatch: {item.run_id}")
        loaded.append(
            LoadedCatalogEntry(
                contract=item,
                run_manifest=run,
                server_pointer=pointer,
                metrics_path=metrics_path,
            )
        )
    return ResultCatalog(
        manifest=manifest,
        entries=tuple(loaded),
        file_sha256=_sha256_file(catalog_path),
    )


def load_final_result_catalog(
    path: str | Path,
    *,
    expected_file_sha256: str,
    trusted_root: str | Path,
) -> tuple[ResultCatalog, FinalCatalogAudit]:
    """Load and require the complete fair 45-run benchmark catalog."""

    catalog = load_result_catalog(
        path,
        expected_file_sha256=expected_file_sha256,
        trusted_root=trusted_root,
    )
    return catalog, require_final_benchmark_catalog(catalog)


def plan_final_result_catalog(
    source_spec_path: str | Path,
    *,
    trusted_root: str | Path,
) -> FinalCatalogPlan:
    """Validate one explicit source spec without discovering or writing results."""

    root = Path(trusted_root).resolve(strict=True)
    spec_path = _resolve_trusted_file(source_spec_path, root)
    try:
        raw = cast(dict[str, Any], json.loads(spec_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise ValueError("final catalog source spec is not valid JSON") from error
    if set(raw) != {"schema_version", "catalog_id", "entries"}:
        raise ValueError("final catalog source spec has missing or unknown fields")
    if raw["schema_version"] != "result-catalog-source-spec-v1":
        raise ValueError("unsupported final catalog source spec schema")
    if not isinstance(raw["catalog_id"], str) or not raw["catalog_id"]:
        raise ValueError("final catalog source spec requires a catalog_id")
    raw_entries = raw["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("final catalog source spec entries must be a list")

    contracts: list[ResultCatalogEntry] = []
    loaded: list[LoadedCatalogEntry] = []
    required = {"run_manifest_path", "server_pointer_path", "metrics_path"}
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"final catalog source entry {index} has invalid fields")
        if any(not isinstance(item[name], str) or not item[name] for name in required):
            raise ValueError(f"final catalog source entry {index} has invalid paths")
        source = CatalogEntrySource(
            run_manifest_path=item["run_manifest_path"],
            server_pointer_path=item["server_pointer_path"],
            metrics_path=item["metrics_path"],
        )
        contract = build_catalog_entry(source, trusted_root=root)
        _, raw_run = _read_hash_pinned_json(
            contract.run_manifest_path,
            expected_sha256=contract.run_manifest_sha256,
            trusted_root=root,
        )
        _, raw_pointer = _read_hash_pinned_json(
            contract.server_pointer_path,
            expected_sha256=contract.server_pointer_sha256,
            trusted_root=root,
        )
        metrics_path = _resolve_trusted_file(contract.metrics_path, root)
        if _sha256_file(metrics_path) != contract.metrics_sha256:
            raise ValueError(f"catalog dependency SHA-256 mismatch: {metrics_path}")
        run = RunManifest.model_validate(raw_run)
        pointer = ServerArtifactPointer.model_validate(raw_pointer)
        contracts.append(contract)
        loaded.append(
            LoadedCatalogEntry(
                contract=contract,
                run_manifest=run,
                server_pointer=pointer,
                metrics_path=metrics_path,
            )
        )
    manifest = ResultCatalogManifest(
        schema_version="result-catalog-v1",
        catalog_id=raw["catalog_id"],
        entries=sorted(contracts, key=lambda entry: entry.run_id),
    )
    catalog = ResultCatalog(
        manifest=manifest,
        entries=tuple(sorted(loaded, key=lambda entry: entry.contract.run_id)),
        file_sha256="0" * 64,
    )
    audit = require_final_benchmark_catalog(catalog)
    return FinalCatalogPlan(
        manifest=manifest,
        audit=audit,
        source_spec_sha256=_sha256_file(spec_path),
    )


def seal_final_result_catalog_from_spec(
    output_path: str | Path,
    *,
    source_spec_path: str | Path,
    trusted_root: str | Path,
) -> tuple[str, FinalCatalogAudit]:
    """Seal a new final catalog and trusted SHA sidecar from an explicit spec."""

    output = Path(output_path).resolve()
    root = Path(trusted_root).resolve(strict=True)
    if not output.is_relative_to(root):
        raise ValueError("final result catalog output must be inside trusted_root")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError("final result catalog and sidecar must be new")
    plan = plan_final_result_catalog(source_spec_path, trusted_root=trusted_root)
    file_sha256 = seal_result_catalog(
        output,
        catalog_id=plan.manifest.catalog_id,
        entries=plan.manifest.entries,
    )
    _atomic_text(sidecar, f"{file_sha256}  {output.name}\n")
    loaded, audit = load_final_result_catalog(
        output,
        expected_file_sha256=file_sha256,
        trusted_root=trusted_root,
    )
    if loaded.file_sha256 != file_sha256 or audit != plan.audit:
        raise RuntimeError("sealed final result catalog did not round-trip")
    return file_sha256, audit
