#!/usr/bin/env python3
"""Run one bounded, training-only stage for one frozen ablation-matrix row.

The worker deliberately reuses ``run_native_experiment`` and stops from the
native ``GraDPertStepEngine.train_step`` immediately after the frozen step
budget.  It cannot run P3, validation, test evaluation, or a scientific epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import ablation_performance_census as census  # noqa: E402

from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.execution.system_resources import (  # noqa: E402
    host_available_memory_bytes,
)
from gradpert.pilots.txpert_candidate_graph_axis import (  # noqa: E402
    TXPERT_CANDIDATE_GENE_COUNT,
    TXPERT_CANDIDATE_GENE_SET_SHA256,
    TXPERT_PUBLIC_COMMIT,
    TxPertCandidateGraphManifest,
)

GIB = 1024**3
SUPPORTED_STAGES = ("p1_capacity", "p2_timing", "diagnostic_profile")


class BoundedCensusComplete(RuntimeError):
    """Signal a successful stop immediately after the final bounded step."""


class TrainingOnlyAccessError(RuntimeError):
    """Raised if the bounded worker reaches validation or test truth."""


class WorkerGateError(RuntimeError):
    """Raised when a frozen worker safety or capacity gate fails."""


@dataclass(frozen=True)
class RuntimeModules:
    torch: Any
    native_execution: Any
    engine_class: type[Any]


@dataclass
class WorkerState:
    batches: list[Any] = field(default_factory=list)
    steps: list[dict[str, object]] = field(default_factory=list)
    timing_samples_ms: list[float] = field(default_factory=list)
    evaluation: dict[str, object] = field(
        default_factory=lambda: {
            "real_canonical_evaluation_constructor_count": 0,
            "guard_bindings": [],
            "validation_cache_requested": False,
            "validation_cache_materialized": False,
            "validation_callback_count": 0,
            "truth_access_attempts": [],
        }
    )
    stage_observer_failures: list[dict[str, object]] = field(default_factory=list)
    primary_failure: BaseException | None = None
    teardown_failures: list[dict[str, str]] = field(default_factory=list)
    profiler_trace: Path | None = None
    profiler_table: Path | None = None
    repository_identity: dict[str, object] | None = None
    final_repository_identity: dict[str, object] | None = None
    final_immutable_input_evidence: dict[str, object] | None = None
    native_identity_receipts: dict[str, object] = field(default_factory=dict)
    persistent_pkl_scan: dict[str, object] = field(default_factory=dict)
    terminal_stage_progress: dict[str, object] | None = None
    batch_gate_failure: dict[str, object] | None = None
    native_runtime_started: bool = False


def _sha256_argument(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256")
    return value


def _commit_argument(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase 40-character Git commit")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--expected-matrix-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--stage-id", choices=SUPPORTED_STAGES, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--development-commit", type=_commit_argument, required=True)
    parser.add_argument("--source-publication-receipt", type=Path, required=True)
    parser.add_argument(
        "--source-publication-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--p0-preflight-receipt", type=Path, required=True)
    parser.add_argument(
        "--p0-preflight-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--batch-manifest-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--p1-receipt", type=Path)
    parser.add_argument("--p1-receipt-sha256", type=_sha256_argument)
    parser.add_argument("--genept-preflight-receipt", type=Path)
    parser.add_argument(
        "--genept-preflight-receipt-sha256",
        type=_sha256_argument,
    )
    parser.add_argument("--minimum-gpu-headroom-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-gpu-free-bytes", type=int, default=4 * GIB)
    parser.add_argument("--maximum-idle-gpu-utilization-percent", type=float, default=5.0)
    parser.add_argument("--maximum-idle-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--minimum-disk-free-bytes", type=int, default=20 * GIB)
    parser.add_argument("--minimum-host-available-bytes", type=int, default=16 * GIB)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _failure_payload(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hash_pinned_json(
    path: Path, expected_sha256: str, *, label: str
) -> tuple[Path, dict[str, object]]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise WorkerGateError(f"{label} must be a regular file")
    if _sha256_file(resolved) != expected_sha256:
        raise WorkerGateError(f"{label} SHA-256 differs")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerGateError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise WorkerGateError(f"{label} payload must be an object")
    return resolved, payload


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_finite_json_payload(value: object, *, label: str) -> None:
    """Reject non-JSON/non-finite runtime evidence before it reaches a receipt."""

    def walk(candidate: object, path: str) -> None:
        if candidate is None or isinstance(candidate, (bool, int, str)):
            return
        if isinstance(candidate, float):
            if not math.isfinite(candidate):
                raise WorkerGateError(f"{label} contains a non-finite float at {path}")
            return
        if isinstance(candidate, Mapping):
            for key, nested in candidate.items():
                if not isinstance(key, str):
                    raise WorkerGateError(f"{label} contains a non-string key at {path}")
                walk(nested, f"{path}.{key}")
            return
        if isinstance(candidate, (list, tuple)):
            for index, nested in enumerate(candidate):
                walk(nested, f"{path}[{index}]")
            return
        raise WorkerGateError(
            f"{label} contains a non-JSON value at {path}: {type(candidate).__name__}"
        )

    walk(value, "$")


def _resolve_p0_preflight(
    args: argparse.Namespace,
    *,
    binding: Any,
    batch_manifest: Any,
) -> dict[str, object]:
    resolved, payload = _load_hash_pinned_json(
        args.p0_preflight_receipt,
        args.p0_preflight_receipt_sha256,
        label="P0 preflight receipt",
    )
    if (
        payload.get("schema_version") != "nadig-vnext-performance-p0-preflight-v1"
        or payload.get("status") != "passed"
        or payload.get("evidence_class") != "performance_preflight_only"
        or payload.get("scientific_completion") is not False
        or payload.get("matrix_sha256") != binding.matrix_sha256
        or payload.get("matrix_row_count") != census.MATRIX_ROW_COUNT
    ):
        raise WorkerGateError("P0 preflight top-level identity differs")
    row_counts = payload.get("row_status_counts")
    cross_h = payload.get("cross_h_audit")
    if (
        not isinstance(row_counts, dict)
        or row_counts != {"blocked": 0, "passed": census.MATRIX_ROW_COUNT}
        or not isinstance(cross_h, dict)
        or cross_h.get("status") != "passed"
    ):
        raise WorkerGateError("P0 preflight did not close every row and H axis")
    source = payload.get("source")
    if not isinstance(source, dict) or any(
        source.get(name) != args.development_commit
        for name in ("expected_commit", "observed_commit")
    ):
        raise WorkerGateError("P0 preflight source identity differs")
    if source.get("source_dirty") is not False or source.get("repository_root") != str(
        args.repository_root.resolve(strict=True)
    ):
        raise WorkerGateError("P0 preflight source is not clean")
    publication_receipt = _regular_file_evidence(
        args.source_publication_receipt,
        args.source_publication_receipt_sha256,
        label="source publication receipt",
        expected_size_bytes=source.get("publication_receipt_size_bytes"),
    )
    source_tree_sha256 = source.get("source_tree_sha256")
    if (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_tree_sha256)
        or not isinstance(source.get("remote_url"), str)
        or not source.get("remote_url")
        or source.get("remote_ref") != "refs/heads/codex/vnext-performance"
        or source.get("published_commit") != args.development_commit
        or source.get("formal_eligible") is not True
        or source.get("publication_receipt_path") != publication_receipt["path"]
        or source.get("publication_receipt_sha256") != publication_receipt["sha256"]
        or source.get("publication_receipt_size_bytes") != publication_receipt["size_bytes"]
    ):
        raise WorkerGateError("P0 preflight published source identity differs")
    data = payload.get("data")
    if (
        not isinstance(data, dict)
        or data.get("dataset_id") != batch_manifest.dataset_id
        or data.get("protocol_id") != batch_manifest.protocol_id
        or data.get("canonical_data_sha256") != batch_manifest.canonical_data_sha256
        or data.get("observation_order_sha256") != batch_manifest.observation_order_sha256
        or data.get("split_content_sha256") != batch_manifest.split_content_sha256
    ):
        raise WorkerGateError("P0 preflight data identity differs from batch manifest")
    forbidden = payload.get("forbidden_runtime")
    if forbidden != {
        "cuda_initialized": False,
        "model_constructed": False,
        "canonical_training_data_constructed": False,
        "canonical_validation_data_constructed": False,
        "canonical_test_data_constructed": False,
    }:
        raise WorkerGateError("P0 preflight used a forbidden runtime surface")
    rows = payload.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != census.MATRIX_ROW_COUNT
        or any(
            not isinstance(value, dict)
            or value.get("matrix_row_index") != index
            or value.get("status") != "passed"
            for index, value in enumerate(rows)
        )
    ):
        raise WorkerGateError("P0 preflight row payload is malformed")
    if binding.matrix_row_index < 0 or binding.matrix_row_index >= len(rows):
        raise WorkerGateError("matrix row index is outside the P0 preflight")
    row = rows[binding.matrix_row_index]
    if (
        not isinstance(row, dict)
        or row.get("matrix_row_index") != binding.matrix_row_index
        or row.get("variant_id") != binding.variant_id
        or _sha256_json(row.get("binding")) != _sha256_json(binding.payload())
        or row.get("status") != "passed"
        or not isinstance(row.get("graph"), dict)
        or not isinstance(row.get("local_view_contract"), dict)
    ):
        raise WorkerGateError("P0 preflight matrix-row identity differs")
    genept = row.get("genept")
    expected_genept_status = "passed" if binding.genept_preflight_required else "not_required"
    if not isinstance(genept, dict) or genept.get("status") != expected_genept_status:
        raise WorkerGateError("P0 preflight GenePT row status differs")
    if binding.genept_preflight_required:
        receipt_identity = genept.get("receipt")
        receipt_size_bytes = (
            receipt_identity.get("size_bytes") if isinstance(receipt_identity, dict) else None
        )
        if (
            args.genept_preflight_receipt is None
            or args.genept_preflight_receipt_sha256 is None
            or not isinstance(receipt_identity, dict)
            or receipt_identity.get("path")
            != str(args.genept_preflight_receipt.resolve(strict=True))
            or receipt_identity.get("sha256") != args.genept_preflight_receipt_sha256
            or not isinstance(receipt_size_bytes, int)
            or isinstance(receipt_size_bytes, bool)
            or receipt_size_bytes <= 0
        ):
            raise WorkerGateError("P0 GenePT receipt differs from worker arguments")
    a0_rows = [
        value
        for value in rows
        if isinstance(value, dict) and value.get("variant_id") == census.A0_VARIANT_ID
    ]
    if len(a0_rows) != 1:
        raise WorkerGateError("P0 preflight lacks one exact A0 row")
    a0_graph = a0_rows[0].get("graph")
    if (
        not isinstance(a0_graph, dict)
        or a0_graph.get("manifest_file_sha256") != batch_manifest.runtime_graph_manifest_sha256
        or a0_graph.get("graph_gene_order_sha256") != batch_manifest.runtime_graph_gene_order_sha256
    ):
        raise WorkerGateError("P0 A0 graph differs from the frozen batch manifest")
    return {
        "receipt_path": str(resolved),
        "receipt_sha256": args.p0_preflight_receipt_sha256,
        "source": source,
        "data": data,
        "row_payload_sha256": _sha256_json(row),
        "row": row,
        "a0_row_payload_sha256": _sha256_json(a0_rows[0]),
        "a0_graph": a0_graph,
    }


def _resolve_stage_prerequisite(
    args: argparse.Namespace,
    *,
    binding: Any,
    batch_manifest: Any,
    p0_preflight: Mapping[str, object],
) -> dict[str, object] | None:
    required = args.stage_id in {"p2_timing", "diagnostic_profile"}
    if not required:
        if args.p1_receipt is not None or args.p1_receipt_sha256 is not None:
            raise WorkerGateError("P1 capacity stage cannot accept a P1 prerequisite")
        return None
    if args.p1_receipt is None or args.p1_receipt_sha256 is None:
        raise WorkerGateError(f"{args.stage_id} requires a hash-pinned completed P1 receipt")
    resolved, payload = _load_hash_pinned_json(
        args.p1_receipt,
        args.p1_receipt_sha256,
        label="P1 stage receipt",
    )
    expected_prefix_sha = census.batch_sequence_sha256(batch_manifest.batches[:1])
    batch_binding = payload.get("frozen_batch_manifest")
    p0_binding = payload.get("p0_preflight")
    repository = payload.get("repository_identity")
    final_repository = payload.get("final_repository_identity")
    resource = payload.get("resource_preflight")
    capacity = payload.get("capacity_evidence")
    instrumentation = payload.get("instrumentation")
    persistent_pkl = payload.get("persistent_pkl_scan")
    native_identity = payload.get("native_identity_receipts")
    immutable_inputs = payload.get("final_immutable_input_evidence")
    if (
        payload.get("schema_version") != "nadig-vnext-performance-stage-v1"
        or payload.get("status") != "complete"
        or payload.get("evidence_class") != "performance_training_only"
        or payload.get("scientific_completion") is not False
        or payload.get("stage_id") != "p1_capacity"
        or payload.get("protocol") != census.STAGE_PROTOCOLS["p1_capacity"].payload()
        or payload.get("variant_id") != binding.variant_id
        or payload.get("config_sha256") != binding.config_sha256
        or payload.get("matrix_sha256") != binding.matrix_sha256
        or _sha256_json(payload.get("binding")) != _sha256_json(binding.payload())
        or payload.get("development_commit") != args.development_commit
        or payload.get("attempted_batch_count") != 1
        or payload.get("completed_step_count") != 1
        or payload.get("observed_step_count") != 1
        or payload.get("batch_sequence_sha256") != expected_prefix_sha
        or not isinstance(batch_binding, dict)
        or batch_binding.get("receipt_sha256") != batch_manifest.sha256
        or batch_binding.get("receipt_path") != batch_manifest.path
        or batch_binding.get("expected_batch_count") != batch_manifest.frozen_prefix_count
        or batch_binding.get("expected_sequence_sha256") != batch_manifest.batch_sequence_sha256
        or batch_binding.get("observed_prefix_count") != 1
        or batch_binding.get("observed_prefix_sha256") != expected_prefix_sha
        or batch_binding.get("expected_prefix_sha256") != expected_prefix_sha
        or batch_binding.get("prefix_matches") is not True
        or not isinstance(p0_binding, dict)
        or p0_binding.get("receipt_sha256") != p0_preflight.get("receipt_sha256")
        or p0_binding.get("row_payload_sha256") != p0_preflight.get("row_payload_sha256")
        or not isinstance(repository, dict)
        or repository.get("head_commit") != args.development_commit
        or not isinstance(final_repository, dict)
        or final_repository.get("head_commit") != args.development_commit
        or not isinstance(persistent_pkl, dict)
        or persistent_pkl.get("passed") is not True
        or persistent_pkl.get("persistent_pkl_count") != 0
        or not isinstance(instrumentation, dict)
        or instrumentation.get("torch_profiler_enabled") is not False
        or instrumentation.get("timing_acceptance") is not False
        or instrumentation.get("heavy_capacity_instrumentation") is not True
        or instrumentation.get("step_timer") != "native_train_step_cuda_synchronized_step_wall_ms"
        or instrumentation.get("stage_observer") != "atomic_per_native_phase"
        or payload.get("torch_profiler_trace_sha256") is not None
        or payload.get("torch_profiler_table_sha256") is not None
        or payload.get("primary_failure") is not None
        or payload.get("teardown_failures") != []
        or payload.get("batch_gate_failure") is not None
        or not isinstance(native_identity, dict)
        or not isinstance(immutable_inputs, dict)
    ):
        raise WorkerGateError("P1 prerequisite identity or completion evidence differs")
    _require_native_identity_receipts(
        native_identity,
        genept_required=binding.genept_preflight_required,
    )
    immutable_files = immutable_inputs.get("files")
    if (
        immutable_inputs.get("schema_version") != "nadig-vnext-performance-immutable-input-audit-v1"
        or not isinstance(immutable_files, list)
        or immutable_inputs.get("file_count") != len(immutable_files)
        or not immutable_files
        or immutable_inputs.get("ordered_file_bindings_sha256") != _sha256_json(immutable_files)
    ):
        raise WorkerGateError("P1 prerequisite immutable-input evidence is malformed")
    for label, identity in (("initial", repository), ("final", final_repository)):
        predicates = identity.get("predicates")
        if (
            identity.get("schema_version") != "nadig-vnext-performance-repository-identity-v1"
            or not isinstance(predicates, dict)
            or set(predicates)
            != {
                "head_equals_development_commit",
                "worktree_clean",
                "formal_source_eligible",
                "published_commit_equals_development_commit",
                "remote_ref_equals_p0",
                "source_content_tree_equals_p0",
                "remote_url_equals_p0",
                "publication_receipt_equals_p0",
            }
            or not all(bool(value) for value in predicates.values())
        ):
            raise WorkerGateError(f"P1 prerequisite {label} repository predicates failed")
    for name in (
        "repository_root",
        "declared_development_commit",
        "head_commit",
        "head_tree",
        "source_tree_sha256",
        "remote_url",
        "remote_ref",
        "published_commit",
        "formal_eligible",
        "publication_receipt_path",
        "publication_receipt_sha256",
        "publication_receipt_size_bytes",
        "status_porcelain_sha256",
    ):
        if repository.get(name) != final_repository.get(name):
            raise WorkerGateError("P1 prerequisite source identity changed during execution")
    training_only = payload.get("training_only_evidence")
    if not isinstance(training_only, dict):
        raise WorkerGateError("P1 prerequisite lacks training-only evidence")
    try:
        census.require_training_only_evidence(training_only)
    except ValueError as error:
        raise WorkerGateError("P1 prerequisite accessed validation/test truth") from error
    if (
        not isinstance(resource, dict)
        or resource.get("schema_version") != "nadig-vnext-performance-resource-preflight-v1"
        or not isinstance(capacity, dict)
    ):
        raise WorkerGateError("P1 prerequisite lacks resource/capacity evidence")
    for name, evidence in (("resource", resource), ("capacity", capacity)):
        evidence_predicates = evidence.get("predicates")
        if (
            not isinstance(evidence_predicates, dict)
            or not evidence_predicates
            or not all(bool(value) for value in evidence_predicates.values())
        ):
            raise WorkerGateError(f"P1 prerequisite {name} predicates failed")
    if set(resource["predicates"]) != {
        "no_competing_compute_processes",
        "gpu_utilization_at_most_limit",
        "gpu_memory_used_at_most_limit",
        "disk_free_at_least_limit",
        "host_available_at_least_limit",
    } or set(capacity["predicates"]) != {
        "exact_observed_step_count",
        "zero_cuda_allocation_retries_or_ooms",
        "gpu_free_bytes_at_least_required_headroom",
    }:
        raise WorkerGateError("P1 prerequisite resource/capacity protocol differs")
    selected = resource.get("selected_physical_gpu")
    if not isinstance(selected, dict) or not isinstance(selected.get("uuid"), str):
        raise WorkerGateError("P1 prerequisite lacks a physical GPU UUID")
    if (
        persistent_pkl.get("schema_version") != "nadig-vnext-performance-zero-pkl-scan-v1"
        or persistent_pkl.get("ordered_relative_paths") != []
    ):
        raise WorkerGateError("P1 prerequisite zero-PKL evidence is malformed")
    batches = payload.get("batches")
    steps = payload.get("steps")
    stage_evidence = payload.get("stage_evidence")
    if (
        not isinstance(batches, list)
        or batches != [batch_manifest.batches[0].payload()]
        or not isinstance(steps, list)
        or len(steps) != 1
        or not isinstance(steps[0], dict)
        or steps[0].get("global_step") != 0
        or steps[0].get("phase") != "measured"
        or steps[0].get("batch_identity_sha256") != batch_manifest.batches[0].sha256
        or not isinstance(stage_evidence, dict)
        or stage_evidence.get("stage_observer_failures") != []
        or not isinstance(stage_evidence.get("terminal_stage_progress"), dict)
    ):
        raise WorkerGateError("P1 prerequisite step/stage evidence is malformed")
    return {
        "receipt_path": str(resolved),
        "receipt_sha256": args.p1_receipt_sha256,
        "physical_gpu_uuid": selected["uuid"],
        "receipt_payload_sha256": _sha256_json(payload),
    }


def _require_prerequisite_gpu(
    prerequisite: Mapping[str, object] | None,
    resource_preflight: Mapping[str, object],
) -> None:
    if prerequisite is None:
        return
    selected = resource_preflight.get("selected_physical_gpu")
    if not isinstance(selected, dict) or selected.get("uuid") != prerequisite.get(
        "physical_gpu_uuid"
    ):
        raise WorkerGateError("current physical GPU differs from the P1 prerequisite")


def _regular_file_evidence(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    allowed_root: Path | None = None,
    expected_size_bytes: int | None = None,
) -> dict[str, object]:
    unresolved = path
    if unresolved.is_symlink():
        raise WorkerGateError(f"{label} must not be a symlink")
    try:
        resolved = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkerGateError(f"{label} is missing or cannot be resolved") from error
    if allowed_root is not None and not resolved.is_relative_to(allowed_root):
        raise WorkerGateError(f"{label} escapes the frozen data root")
    if not resolved.is_file():
        raise WorkerGateError(f"{label} must be a regular file")
    observed_size_bytes = resolved.stat().st_size
    if expected_size_bytes is not None and (
        not isinstance(expected_size_bytes, int)
        or isinstance(expected_size_bytes, bool)
        or expected_size_bytes <= 0
        or observed_size_bytes != expected_size_bytes
    ):
        raise WorkerGateError(f"{label} size changed after P0 sealing")
    observed_sha256 = _sha256_file(resolved)
    if observed_sha256 != expected_sha256:
        raise WorkerGateError(f"{label} changed after P0/batch sealing")
    return {
        "label": label,
        "path": str(resolved),
        "sha256": observed_sha256,
        "size_bytes": observed_size_bytes,
    }


def _require_graph_artifacts(
    graph: Mapping[str, object],
    *,
    label: str,
    allowed_root: Path,
) -> list[dict[str, object]]:
    graph_axis_policy = graph.get("graph_axis_policy", "recomputed_hvg_union_candidate_targets")
    requested_graph_gene_count = graph.get("requested_graph_gene_count")
    legacy_requested_hvg_count = graph.get("requested_hvg_count")
    if requested_graph_gene_count is None:
        requested_graph_gene_count = legacy_requested_hvg_count
    elif (
        legacy_requested_hvg_count is not None
        and legacy_requested_hvg_count != requested_graph_gene_count
    ):
        raise WorkerGateError(f"{label} graph requested-count summaries differ")
    graph_root_value = graph.get("root_path")
    artifacts = graph.get("artifacts")
    if (
        not isinstance(requested_graph_gene_count, int)
        or isinstance(requested_graph_gene_count, bool)
        or not isinstance(graph_root_value, str)
        or not isinstance(artifacts, Mapping)
    ):
        raise WorkerGateError(f"{label} graph artifact contract is malformed")
    common_specifications = {
        "manifest": ("manifest.json", "runtime_graph_manifest"),
        "graph_gene_ids": ("graph_gene_ids.txt", "ordered_graph_gene_axis"),
        "go": ("go.npz", "pruned_go_graph"),
        "string": ("string.npz", "pruned_string_graph"),
    }
    if graph_axis_policy == "recomputed_hvg_union_candidate_targets":
        expected_specifications = {
            **common_specifications,
            "hvg_dispersion_ranking": (
                f"hvg{requested_graph_gene_count}_dispersion_ranking.json",
                "hvg_dispersion_ranking_receipt",
            ),
        }
    elif graph_axis_policy == "txpert_candidate_gene_universe":
        expected_specifications = common_specifications
    else:
        raise WorkerGateError(f"{label} graph-axis policy is unsupported")
    if set(artifacts) != set(expected_specifications):
        raise WorkerGateError(f"{label} graph artifact set differs from P0")
    graph_root = Path(graph_root_value).resolve(strict=True)
    if not graph_root.is_dir() or not graph_root.is_relative_to(allowed_root):
        raise WorkerGateError(f"{label} graph root escapes the frozen data root")
    evidence_by_id: dict[str, dict[str, object]] = {}
    for artifact_id, (filename, expected_role) in expected_specifications.items():
        artifact = artifacts.get(artifact_id)
        if not isinstance(artifact, Mapping):
            raise WorkerGateError(f"{label} graph artifact is malformed: {artifact_id}")
        path = artifact.get("path")
        sha256 = artifact.get("sha256")
        size_bytes = artifact.get("size_bytes")
        if (
            not isinstance(path, str)
            or not isinstance(sha256, str)
            or artifact.get("role") != expected_role
            or Path(path).resolve(strict=True) != (graph_root / filename).resolve(strict=True)
        ):
            raise WorkerGateError(f"{label} graph artifact binding differs: {artifact_id}")
        evidence_by_id[artifact_id] = _regular_file_evidence(
            Path(path),
            sha256,
            label=f"{label} graph {artifact_id} artifact",
            allowed_root=allowed_root,
            expected_size_bytes=size_bytes if isinstance(size_bytes, int) else None,
        )
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise WorkerGateError(f"{label} graph artifact size is malformed: {artifact_id}")
    manifest_path_value = graph.get("manifest_path")
    manifest_sha256 = graph.get("manifest_file_sha256")
    manifest_evidence = evidence_by_id["manifest"]
    if (
        not isinstance(manifest_path_value, str)
        or Path(manifest_path_value).resolve(strict=True) != Path(str(manifest_evidence["path"]))
        or manifest_sha256 != manifest_evidence["sha256"]
    ):
        raise WorkerGateError(f"{label} graph manifest summary differs from P0 artifacts")
    manifest_path = Path(str(manifest_evidence["path"]))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerGateError(f"{label} graph manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise WorkerGateError(f"{label} graph manifest must be an object")
    source_hashes = graph.get("source_artifact_sha256")
    manifest_source_hashes = manifest.get("source_artifact_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != {"go", "string"}
        or source_hashes != manifest_source_hashes
    ):
        raise WorkerGateError(f"{label} graph source-artifact identities differ")
    gene_order_sha256 = graph.get("graph_gene_order_sha256")
    topology_sha256 = graph.get("topology_content_sha256")
    if (
        not isinstance(gene_order_sha256, str)
        or manifest.get("graph_gene_order_sha256") != gene_order_sha256
        or not isinstance(topology_sha256, str)
        or manifest.get("topology_content_sha256") != topology_sha256
        or _sha256_json({"graph_gene_order_sha256": gene_order_sha256, "sources": source_hashes})
        != topology_sha256
    ):
        raise WorkerGateError(f"{label} graph semantic identity differs")
    if graph_axis_policy == "recomputed_hvg_union_candidate_targets":
        if manifest.get("requested_hvg_count") != requested_graph_gene_count:
            raise WorkerGateError(f"{label} graph HVG identity differs")
    else:
        try:
            candidate_manifest = TxPertCandidateGraphManifest.model_validate(manifest)
        except ValueError as error:
            raise WorkerGateError(f"{label} TxPert candidate graph manifest differs") from error
        if (
            requested_graph_gene_count != TXPERT_CANDIDATE_GENE_COUNT
            or candidate_manifest.requested_gene_count != requested_graph_gene_count
            or graph.get("graph_axis_source_sha256") != TXPERT_CANDIDATE_GENE_SET_SHA256
            or candidate_manifest.candidate_gene_set_sha256 != TXPERT_CANDIDATE_GENE_SET_SHA256
            or candidate_manifest.txpert_public_commit != TXPERT_PUBLIC_COMMIT
            or graph.get("candidate_target_order_sha256")
            != candidate_manifest.candidate_target_order_sha256
        ):
            raise WorkerGateError(f"{label} TxPert candidate graph identity differs")
    gene_axis = Path(str(evidence_by_id["graph_gene_ids"]["path"]))
    try:
        gene_ids = gene_axis.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise WorkerGateError(f"{label} graph gene axis is unreadable") from error
    if _sha256_json(gene_ids) != gene_order_sha256:
        raise WorkerGateError(f"{label} graph gene axis changed after P0 sealing")
    for source_name in ("go", "string"):
        expected = source_hashes.get(source_name)
        if not isinstance(expected, str) or evidence_by_id[source_name]["sha256"] != expected:
            raise WorkerGateError(f"{label} graph {source_name} SHA-256 is malformed")
    if graph_axis_policy == "txpert_candidate_gene_universe":
        candidate_evidence = _regular_file_evidence(
            Path(candidate_manifest.candidate_gene_set_path),
            candidate_manifest.candidate_gene_set_sha256,
            label=f"{label} TxPert candidate-gene source",
        )
        return [
            *(evidence_by_id[artifact_id] for artifact_id in expected_specifications),
            candidate_evidence,
        ]
    ranking_path = Path(str(evidence_by_id["hvg_dispersion_ranking"]["path"]))
    try:
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerGateError(f"{label} graph ranking receipt is unreadable") from error
    ordered_entries = ranking.get("ordered_entries") if isinstance(ranking, dict) else None
    ranked_ids = manifest.get("normalized_dispersion_ranked_hvg_gene_ids")
    ranking_sha256 = manifest.get("normalized_dispersion_ranking_sha256")
    if (
        not isinstance(ordered_entries, list)
        or not isinstance(ranked_ids, list)
        or not isinstance(ranking_sha256, str)
        or _sha256_json(ordered_entries) != ranking_sha256
        or [entry.get("gene_id") if isinstance(entry, dict) else None for entry in ordered_entries]
        != ranked_ids
    ):
        raise WorkerGateError(f"{label} graph ranking changed after P0 sealing")
    return [evidence_by_id[artifact_id] for artifact_id in expected_specifications]


def _require_immutable_inputs(
    *,
    binding: Any,
    p0_preflight: Mapping[str, object],
    batch_manifest: Any,
    stage_prerequisite: Mapping[str, object] | None,
    data_root: Path,
) -> dict[str, object]:
    allowed_root = data_root.resolve(strict=True)
    expected_files = [
        (Path(binding.matrix_path), binding.matrix_sha256, "matrix"),
        (Path(binding.config_path), binding.config_sha256, "variant config"),
        (
            Path(str(p0_preflight["receipt_path"])),
            str(p0_preflight["receipt_sha256"]),
            "P0 preflight receipt",
        ),
        (Path(batch_manifest.path), batch_manifest.sha256, "batch manifest"),
    ]
    if stage_prerequisite is not None:
        expected_files.append(
            (
                Path(str(stage_prerequisite["receipt_path"])),
                str(stage_prerequisite["receipt_sha256"]),
                "P1 prerequisite receipt",
            )
        )
    evidence = [
        _regular_file_evidence(path, expected_sha256, label=label)
        for path, expected_sha256, label in expected_files
    ]
    source = p0_preflight.get("source")
    if not isinstance(source, Mapping):
        raise WorkerGateError("P0 preflight lacks its source publication receipt")
    publication_receipt_path = source.get("publication_receipt_path")
    publication_receipt_sha256 = source.get("publication_receipt_sha256")
    publication_receipt_size_bytes = source.get("publication_receipt_size_bytes")
    if (
        not isinstance(publication_receipt_path, str)
        or not isinstance(publication_receipt_sha256, str)
        or not isinstance(publication_receipt_size_bytes, int)
        or isinstance(publication_receipt_size_bytes, bool)
        or publication_receipt_size_bytes <= 0
    ):
        raise WorkerGateError("P0 source publication receipt identity is malformed")
    evidence.append(
        _regular_file_evidence(
            Path(publication_receipt_path),
            publication_receipt_sha256,
            label="source publication receipt",
            expected_size_bytes=publication_receipt_size_bytes,
        )
    )
    data = p0_preflight.get("data")
    if not isinstance(data, Mapping):
        raise WorkerGateError("P0 preflight lacks live data artifact identities")
    data_artifacts = data.get("artifacts")
    expected_data_artifacts = {
        "canonical_manifest": (
            "canonical_manifest_path",
            "canonical_manifest_sha256",
            "canonical_manifest_size_bytes",
            "canonical_data_manifest",
        ),
        "canonical_h5ad": (
            "canonical_data_path",
            "canonical_data_sha256",
            "canonical_data_size_bytes",
            "canonical_expression_and_metadata",
        ),
        "split_manifest": (
            "split_manifest_path",
            "split_manifest_sha256",
            "split_manifest_size_bytes",
            "canonical_condition_split",
        ),
        "source_manifest": (
            "source_manifest_path",
            "source_manifest_sha256",
            "source_manifest_size_bytes",
            "source_data_manifest",
        ),
        "source_h5ad": (
            "source_h5ad_path",
            "source_h5ad_sha256",
            "source_h5ad_size_bytes",
            "source_expression_and_metadata",
        ),
    }
    if not isinstance(data_artifacts, Mapping) or set(data_artifacts) != set(
        expected_data_artifacts
    ):
        raise WorkerGateError("P0 preflight data artifact set differs")
    for artifact_id, (path_field, hash_field, size_field, role) in expected_data_artifacts.items():
        artifact = data_artifacts.get(artifact_id)
        if not isinstance(artifact, Mapping):
            raise WorkerGateError(f"P0 data artifact is malformed: {artifact_id}")
        path_value = artifact.get("path")
        hash_value = artifact.get("sha256")
        size_value = artifact.get("size_bytes")
        if (
            not isinstance(path_value, str)
            or not isinstance(hash_value, str)
            or not isinstance(size_value, int)
            or isinstance(size_value, bool)
            or size_value <= 0
            or artifact.get("role") != role
            or data.get(path_field) != path_value
            or data.get(hash_field) != hash_value
            or data.get(size_field) != size_value
        ):
            raise WorkerGateError(f"P0 data artifact binding differs: {artifact_id}")
        evidence.append(
            _regular_file_evidence(
                Path(path_value),
                hash_value,
                label=f"P0 data {artifact_id} artifact",
                allowed_root=allowed_root,
                expected_size_bytes=size_value,
            )
        )
    row = p0_preflight.get("row")
    a0_graph = p0_preflight.get("a0_graph")
    graph = row.get("graph") if isinstance(row, Mapping) else None
    if not isinstance(graph, Mapping) or not isinstance(a0_graph, Mapping):
        raise WorkerGateError("P0 preflight lacks selected/A0 graph artifact identities")
    evidence.extend(
        _require_graph_artifacts(graph, label="selected row", allowed_root=allowed_root)
    )
    if graph.get("manifest_path") != a0_graph.get("manifest_path"):
        evidence.extend(
            _require_graph_artifacts(a0_graph, label="A0 batch", allowed_root=allowed_root)
        )
    if str(Path(batch_manifest.runtime_graph_manifest_path).resolve(strict=True)) != str(
        Path(str(a0_graph["manifest_path"])).resolve(strict=True)
    ) or batch_manifest.runtime_graph_manifest_sha256 != a0_graph.get("manifest_file_sha256"):
        raise WorkerGateError("frozen batch manifest differs from the live A0 graph")
    genept = row.get("genept") if isinstance(row, Mapping) else None
    if isinstance(genept, Mapping) and genept.get("status") == "passed":
        receipt = genept.get("receipt")
        artifact = genept.get("artifact")
        if not isinstance(receipt, Mapping) or not isinstance(artifact, Mapping):
            raise WorkerGateError("P0 GenePT row lacks its receipt or selected source artifact")
        receipt_path = receipt.get("path")
        receipt_sha256 = receipt.get("sha256")
        receipt_size_bytes = receipt.get("size_bytes")
        artifact_path = artifact.get("path")
        artifact_sha256 = artifact.get("sha256")
        artifact_size_bytes = artifact.get("size_bytes")
        if (
            not isinstance(receipt_path, str)
            or not isinstance(receipt_sha256, str)
            or not isinstance(receipt_size_bytes, int)
            or isinstance(receipt_size_bytes, bool)
            or receipt_size_bytes <= 0
            or not isinstance(artifact_path, str)
            or not isinstance(artifact_sha256, str)
            or not isinstance(artifact_size_bytes, int)
            or isinstance(artifact_size_bytes, bool)
            or artifact_size_bytes <= 0
        ):
            raise WorkerGateError("P0 GenePT receipt or artifact identity is malformed")
        evidence.append(
            _regular_file_evidence(
                Path(receipt_path),
                receipt_sha256,
                label="GenePT preflight receipt",
                allowed_root=None,
                expected_size_bytes=receipt_size_bytes,
            )
        )
        evidence.append(
            _regular_file_evidence(
                Path(artifact_path),
                artifact_sha256,
                label="GenePT selected artifact",
                allowed_root=None,
                expected_size_bytes=artifact_size_bytes,
            )
        )
    return {
        "schema_version": "nadig-vnext-performance-immutable-input-audit-v1",
        "file_count": len(evidence),
        "files": evidence,
        "ordered_file_bindings_sha256": _sha256_json(evidence),
    }


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if completed.returncode != 0:
        raise WorkerGateError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _repository_identity_evidence(
    args: argparse.Namespace,
    *,
    expected_p0_source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = Path(args.repository_root).resolve(strict=True)
    inside = _git_output(root, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise WorkerGateError("repository root is not a Git worktree")
    commit = _git_output(root, "rev-parse", "HEAD")
    tree = _git_output(root, "rev-parse", "HEAD^{tree}")
    status = _git_output(root, "status", "--porcelain", "--untracked-files=normal")
    publication_receipt = _regular_file_evidence(
        args.source_publication_receipt,
        args.source_publication_receipt_sha256,
        label="source publication receipt",
    )
    try:
        source_identity = inspect_source_identity(
            root,
            formal=True,
            expected_repository="https://github.com/elan6666/GraD-Pert",
            development_commit=args.development_commit,
            remote_ref="refs/heads/codex/vnext-performance",
            publication_receipt=Path(str(publication_receipt["path"])),
            expected_publication_receipt_sha256=args.source_publication_receipt_sha256,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise WorkerGateError("source content-tree identity inspection failed") from error
    expected_tree_sha256 = (
        expected_p0_source.get("source_tree_sha256")
        if isinstance(expected_p0_source, Mapping)
        else source_identity.tree_sha256
    )
    expected_remote_url = (
        expected_p0_source.get("remote_url")
        if isinstance(expected_p0_source, Mapping)
        else source_identity.remote_url
    )
    expected_remote_ref = (
        expected_p0_source.get("remote_ref")
        if isinstance(expected_p0_source, Mapping)
        else source_identity.remote_ref
    )
    expected_published_commit = (
        expected_p0_source.get("published_commit")
        if isinstance(expected_p0_source, Mapping)
        else source_identity.published_commit
    )
    expected_publication_receipt_path = (
        expected_p0_source.get("publication_receipt_path")
        if isinstance(expected_p0_source, Mapping)
        else publication_receipt["path"]
    )
    expected_publication_receipt_sha256 = (
        expected_p0_source.get("publication_receipt_sha256")
        if isinstance(expected_p0_source, Mapping)
        else publication_receipt["sha256"]
    )
    expected_publication_receipt_size_bytes = (
        expected_p0_source.get("publication_receipt_size_bytes")
        if isinstance(expected_p0_source, Mapping)
        else publication_receipt["size_bytes"]
    )
    predicates = {
        "head_equals_development_commit": commit == args.development_commit,
        "worktree_clean": status == "",
        "formal_source_eligible": source_identity.formal_eligible is True,
        "published_commit_equals_development_commit": source_identity.published_commit
        == args.development_commit
        == expected_published_commit,
        "remote_ref_equals_p0": source_identity.remote_ref == expected_remote_ref,
        "source_content_tree_equals_p0": source_identity.tree_sha256 == expected_tree_sha256,
        "remote_url_equals_p0": source_identity.remote_url == expected_remote_url,
        "publication_receipt_equals_p0": (
            publication_receipt["path"] == expected_publication_receipt_path
            and publication_receipt["sha256"] == expected_publication_receipt_sha256
            and publication_receipt["size_bytes"] == expected_publication_receipt_size_bytes
            and source_identity.publication_receipt_sha256
            == args.source_publication_receipt_sha256
            == expected_publication_receipt_sha256
        ),
    }
    return {
        "schema_version": "nadig-vnext-performance-repository-identity-v1",
        "repository_root": str(root),
        "declared_development_commit": args.development_commit,
        "head_commit": commit,
        "head_tree": tree,
        "source_tree_sha256": source_identity.tree_sha256,
        "remote_url": source_identity.remote_url,
        "remote_ref": source_identity.remote_ref,
        "published_commit": source_identity.published_commit,
        "formal_eligible": source_identity.formal_eligible,
        "publication_receipt_path": publication_receipt["path"],
        "publication_receipt_sha256": publication_receipt["sha256"],
        "publication_receipt_size_bytes": publication_receipt["size_bytes"],
        "status_porcelain": status,
        "status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "predicates": predicates,
    }


def _record_final_repository_identity(
    args: argparse.Namespace,
    *,
    state: WorkerState,
    expected_p0_source: Mapping[str, object] | None = None,
) -> None:
    """Seal post-run source identity without obscuring an earlier training failure."""

    try:
        final = _repository_identity_evidence(
            args,
            expected_p0_source=expected_p0_source,
        )
        state.final_repository_identity = final
        predicates = final.get("predicates")
        initial = state.repository_identity
        identity_fields = (
            "repository_root",
            "declared_development_commit",
            "head_commit",
            "head_tree",
            "source_tree_sha256",
            "remote_url",
            "remote_ref",
            "published_commit",
            "formal_eligible",
            "publication_receipt_path",
            "publication_receipt_sha256",
            "publication_receipt_size_bytes",
            "status_porcelain_sha256",
        )
        passed = (
            isinstance(initial, dict)
            and isinstance(predicates, dict)
            and bool(predicates)
            and all(bool(value) for value in predicates.values())
            and all(initial.get(name) == final.get(name) for name in identity_fields)
        )
        if passed:
            return
        error: BaseException = WorkerGateError(
            "repository identity changed or became dirty during bounded execution"
        )
    except BaseException as observed_error:
        error = observed_error
    if state.primary_failure is None:
        state.primary_failure = error
    else:
        state.teardown_failures.append(
            {"stage": "final_repository_identity", **_failure_payload(error)}
        )


def _cpu_rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        fields = statm.read_text(encoding="ascii").split()
        if len(fields) < 2:
            raise WorkerGateError("/proc/self/statm is malformed")
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    scale = 1 if sys.platform == "darwin" else 1024
    return maximum_rss * scale


def _resolve_genept_preflight(
    args: argparse.Namespace,
    *,
    binding: Any,
) -> tuple[Path | None, str | None]:
    receipt = args.genept_preflight_receipt
    expected_sha256 = args.genept_preflight_receipt_sha256
    if binding.genept_preflight_required:
        if receipt is None or expected_sha256 is None:
            raise WorkerGateError("GenePT matrix row requires a hash-pinned preflight receipt")
        resolved = Path(receipt).resolve(strict=True)
        if _sha256_file(resolved) != expected_sha256:
            raise WorkerGateError("GenePT preflight receipt SHA-256 differs")
        return resolved, expected_sha256
    if receipt is not None or expected_sha256 is not None:
        raise WorkerGateError("non-GenePT matrix row cannot accept a GenePT preflight receipt")
    return None, None


def _host_available_bytes() -> int | None:
    return host_available_memory_bytes()


def _nvidia_rows(fields: Sequence[str], *, compute_apps: bool = False) -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise WorkerGateError("nvidia-smi is required for physical-GPU preflight")
    flag = "--query-compute-apps" if compute_apps else "--query-gpu"
    completed = subprocess.run(
        [executable, f"{flag}={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise WorkerGateError(f"nvidia-smi preflight failed: {completed.stderr.strip()}")
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise WorkerGateError("nvidia-smi returned a malformed row")
        rows.append(dict(zip(fields, values, strict=True)))
    return rows


def _physical_gpu_preflight(args: argparse.Namespace, attempt_root: Path) -> dict[str, object]:
    if not args.device.startswith("cuda:"):
        raise WorkerGateError("performance census requires an explicit cuda:N device")
    try:
        logical_index = int(args.device.split(":", 1)[1])
    except ValueError as error:
        raise WorkerGateError("performance census device must be cuda:N") from error
    gpu_fields = ("index", "uuid", "name", "utilization.gpu", "memory.used", "memory.total")
    app_fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    gpu_rows = _nvidia_rows(gpu_fields)
    app_rows = _nvidia_rows(app_fields, compute_apps=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selectors = [value.strip() for value in visible.split(",")] if visible else []
    if selectors and logical_index >= len(selectors):
        raise WorkerGateError("logical device is outside CUDA_VISIBLE_DEVICES")
    selector = selectors[logical_index] if selectors else str(logical_index)
    matches = [
        row
        for row in gpu_rows
        if row["index"] == selector
        or row["uuid"].startswith(selector)
        or selector.startswith(row["uuid"])
    ]
    if len(matches) != 1:
        raise WorkerGateError("logical device does not resolve to one physical GPU")
    selected = matches[0]
    competing = [
        row
        for row in app_rows
        if row["gpu_uuid"] == selected["uuid"] and row["pid"] != str(os.getpid())
    ]
    disk = shutil.disk_usage(attempt_root)
    host_available = _host_available_bytes()
    predicates = {
        "no_competing_compute_processes": not competing,
        "gpu_utilization_at_most_limit": float(selected["utilization.gpu"])
        <= args.maximum_idle_gpu_utilization_percent,
        "gpu_memory_used_at_most_limit": int(selected["memory.used"])
        <= args.maximum_idle_gpu_memory_mib,
        "disk_free_at_least_limit": disk.free >= args.minimum_disk_free_bytes,
        "host_available_at_least_limit": host_available is not None
        and host_available >= args.minimum_host_available_bytes,
    }
    return {
        "schema_version": "nadig-vnext-performance-resource-preflight-v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "selected_physical_gpu": selected,
        "competing_compute_processes": competing,
        "disk_free_bytes": disk.free,
        "host_available_bytes": host_available,
        "predicates": predicates,
    }


class _TrainingOnlyEvaluationGuard:
    def __init__(self, state: dict[str, object], *, split_name: str) -> None:
        self._state = state
        self._split_name = split_name

    def __enter__(self) -> _TrainingOnlyEvaluationGuard:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def configure_expression_cache(self, *, enabled: bool) -> float:
        self._state["validation_cache_requested"] = bool(enabled)
        self._state["validation_cache_materialized"] = False
        return 0.0

    def __getattr__(self, name: str) -> object:
        attempts = self._state["truth_access_attempts"]
        assert isinstance(attempts, list)
        attempts.append(f"{self._split_name}.{name}")
        raise TrainingOnlyAccessError(
            f"training-only census attempted evaluator access: {self._split_name}.{name}"
        )


def _evaluation_guard_factory(
    state: dict[str, object],
) -> Callable[..., _TrainingOnlyEvaluationGuard]:
    def factory(*_: object, split_name: str, **__: object) -> _TrainingOnlyEvaluationGuard:
        bindings = state["guard_bindings"]
        assert isinstance(bindings, list)
        bindings.append(split_name)
        return _TrainingOnlyEvaluationGuard(state, split_name=split_name)

    return factory


def _training_only_evidence(state: Mapping[str, object]) -> dict[str, object]:
    attempts = [str(value) for value in state["truth_access_attempts"]]
    return {
        "scope": "performance_training_only",
        "real_canonical_evaluation_constructor_count": int(
            state["real_canonical_evaluation_constructor_count"]
        ),
        "validation_cache_materialized": bool(state["validation_cache_materialized"]),
        "validation_callback_count": int(state["validation_callback_count"]),
        "validation_accessed": bool(state["validation_cache_materialized"])
        or int(state["real_canonical_evaluation_constructor_count"]) > 0
        or any(value.startswith("val.") for value in attempts),
        "test_truth_accessed": any(value.startswith("test.") for value in attempts),
        "truth_access_attempts": attempts,
        "guard_bindings": [str(value) for value in state["guard_bindings"]],
        "validation_cache_requested": bool(state["validation_cache_requested"]),
    }


def ordered_batch_identity(batch: Any, *, global_step: int, topology: Any) -> Any:
    condition_ids = tuple(str(value) for value in batch.condition_ids)
    anchors = batch.anchors_by_condition
    if set(anchors) != set(condition_ids):
        raise WorkerGateError("batch anchor conditions differ from ordered condition IDs")
    try:
        gene_ids = tuple(str(value) for value in topology.gene_ids)
    except (AttributeError, TypeError) as error:
        raise WorkerGateError("engine topology lacks an ordered gene axis") from error
    if not gene_ids or any(not value for value in gene_ids) or len(set(gene_ids)) != len(gene_ids):
        raise WorkerGateError("engine topology gene IDs must be nonempty and unique")
    active_anchor_ids: list[list[str]] = []
    for condition_id in condition_ids:
        stable_ids: list[str] = []
        for anchor in anchors[condition_id]:
            if isinstance(anchor, bool) or not isinstance(anchor, int):
                raise WorkerGateError("batch anchor indices must be integers")
            if anchor < 0 or anchor >= len(gene_ids):
                raise WorkerGateError("batch anchor index is outside the topology gene axis")
            stable_ids.append(gene_ids[anchor])
        active_anchor_ids.append(stable_ids)
    identity = census.OrderedBatchIdentity.create(
        global_step=global_step,
        row_ids=[str(value) for value in batch.perturbed_row_ids],
        condition_ids=condition_ids,
        control_row_ids=[str(value) for value in batch.control_row_ids],
        active_anchor_ids=active_anchor_ids,
        actual_batch_size=len(condition_ids),
        unique_condition_count=len(set(condition_ids)),
    )
    if identity.actual_batch_size != census.EXACT_TRAIN_BATCH_SIZE:
        raise WorkerGateError("performance census observed a non-256 training batch")
    return identity


def _metrics_payload(metrics: Any) -> dict[str, object]:
    if is_dataclass(metrics):
        payload = dict(asdict(metrics))
    elif isinstance(metrics, Mapping):
        if any(not isinstance(key, str) for key in metrics):
            raise WorkerGateError("native step metrics contain a non-string key")
        payload = dict(metrics)
    else:
        payload = dict(vars(metrics))
    _require_finite_json_payload(payload, label="native step metrics")
    return payload


def _cuda_telemetry(torch: Any, device: Any) -> dict[str, object]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    memory_stats = torch.cuda.memory_stats(device)
    return {
        "allocated_gpu_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_gpu_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu_free_bytes": int(free_bytes),
        "gpu_total_bytes": int(total_bytes),
        "cuda_retry_oom_counters": {
            str(key): int(value)
            for key, value in memory_stats.items()
            if "retry" in str(key).lower() or "oom" in str(key).lower()
        },
    }


def _resource_telemetry(torch: Any, device: Any) -> dict[str, object]:
    return {**_cuda_telemetry(torch, device), "cpu_rss_bytes": _cpu_rss_bytes()}


def _record_atomic_stage_event(
    observer: Any,
    *,
    event: Any,
    telemetry: Mapping[str, object],
) -> None:
    if event.status == "entered":
        observer.entered(event.phase_id, telemetry)
        return
    if event.status == "completed":
        observer.completed(event.phase_id, telemetry)
        return
    if event.status != "failure":
        raise WorkerGateError(f"unknown stage event status: {event.status}")
    events = observer.receipt["stage_events"]
    if not isinstance(events, list):
        raise WorkerGateError("atomic stage event collection is malformed")
    events.append({"event": "failure", "stage": event.phase_id, "telemetry": dict(telemetry)})
    observer.receipt["last_failed_stage"] = event.phase_id
    observer._write()


_NATIVE_IDENTITY_CANDIDATES = (
    "config.resolved.yaml",
    "source_identity.json",
    "environment.json",
    "resolved_local_view_contract.json",
    "genept_preflight.json",
    "genept_feature.json",
    "training_data.json",
    "run_meta.json",
)
_NATIVE_IDENTITY_REQUIRED = frozenset(
    {
        "config.resolved.yaml",
        "source_identity.json",
        "environment.json",
        "resolved_local_view_contract.json",
        "training_data.json",
        "run_meta.json",
    }
)
_GENEPT_NATIVE_IDENTITY_REQUIRED = frozenset({"genept_preflight.json", "genept_feature.json"})


def _collect_native_identity_receipts(attempt_root: Path) -> dict[str, object]:
    small_root = attempt_root / "native-run" / "small_results"
    bindings: list[dict[str, object]] = []
    for name in _NATIVE_IDENTITY_CANDIDATES:
        path = small_root / name
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise WorkerGateError(f"native identity receipt is not a regular file: {path}")
        bindings.append(
            {
                "relative_path": str(path.relative_to(attempt_root)),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "nadig-vnext-native-small-identity-bindings-v1",
        "candidate_names": list(_NATIVE_IDENTITY_CANDIDATES),
        "files": bindings,
        "ordered_bindings_sha256": _sha256_json(bindings),
    }


def _require_native_identity_receipts(
    evidence: Mapping[str, object],
    *,
    genept_required: bool,
) -> None:
    files = evidence.get("files")
    if (
        evidence.get("schema_version") != "nadig-vnext-native-small-identity-bindings-v1"
        or evidence.get("candidate_names") != list(_NATIVE_IDENTITY_CANDIDATES)
        or not isinstance(files, list)
        or any(not isinstance(value, dict) for value in files)
        or evidence.get("ordered_bindings_sha256") != _sha256_json(files)
    ):
        raise WorkerGateError("native identity bindings are malformed")
    names: set[str] = set()
    for value in files:
        relative_path = value.get("relative_path")
        sha256 = value.get("sha256")
        size_bytes = value.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).parts[:2] != ("native-run", "small_results")
            or Path(relative_path).name not in _NATIVE_IDENTITY_CANDIDATES
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise WorkerGateError("native identity binding entry is malformed")
        names.add(Path(relative_path).name)
    if len(names) != len(files):
        raise WorkerGateError("native identity bindings contain duplicate files")
    required = set(_NATIVE_IDENTITY_REQUIRED)
    if genept_required:
        required.update(_GENEPT_NATIVE_IDENTITY_REQUIRED)
    missing = sorted(required - names)
    if missing:
        raise WorkerGateError("native identity bindings are incomplete: " + ", ".join(missing))


def _scan_attempt_pkls(attempt_root: Path) -> dict[str, object]:
    relative_paths = [
        str(path.relative_to(attempt_root))
        for path in sorted(attempt_root.rglob("*.pkl"))
        if path.is_file() or path.is_symlink()
    ]
    return {
        "schema_version": "nadig-vnext-performance-zero-pkl-scan-v1",
        "attempt_root": str(attempt_root),
        "persistent_pkl_count": len(relative_paths),
        "ordered_relative_paths": relative_paths,
        "ordered_relative_paths_sha256": _sha256_json(relative_paths),
        "passed": not relative_paths,
    }


def _stop_profiler(
    profiler: Any | None,
    *,
    evidence_root: Path,
    state: WorkerState,
) -> None:
    if profiler is None:
        return
    try:
        profiler.stop()
        evidence_root.mkdir(parents=True, exist_ok=True)
        trace = evidence_root / "torch-profiler-trace.json"
        table = evidence_root / "torch-profiler-table.txt"
        profiler.export_chrome_trace(str(trace))
        table.write_text(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=200),
            encoding="utf-8",
        )
        state.profiler_trace = trace
        state.profiler_table = table
    except BaseException as error:
        state.teardown_failures.append({"stage": "profiler", **_failure_payload(error)})


def _execute_bounded_native(
    args: argparse.Namespace,
    *,
    binding: Any,
    attempt_root: Path,
    runtime: RuntimeModules,
    resource_preflight: Mapping[str, object],
    repository_identity: Mapping[str, object],
    genept_preflight: tuple[Path | None, str | None],
    batch_manifest: Any,
    p0_preflight: Mapping[str, object],
) -> WorkerState:
    protocol = census.STAGE_PROTOCOLS[args.stage_id]
    state = WorkerState()
    state.repository_identity = dict(repository_identity)
    native_run_root = attempt_root / "native-run"
    device = runtime.torch.device(args.device)
    original_train_step = runtime.engine_class.train_step
    original_evaluator = runtime.native_execution.CanonicalEvaluationData
    original_validation = runtime.native_execution.evaluate_validation_macro_delta
    atomic_observer: Any | None = None
    if protocol.heavy_capacity_instrumentation:
        atomic_observer = census.AtomicStageObserver(
            attempt_root / "stage-progress.json",
            {
                "variant_id": binding.variant_id,
                "stage_id": args.stage_id,
                "matrix_sha256": binding.matrix_sha256,
                "config_sha256": binding.config_sha256,
                "p0_preflight_sha256": p0_preflight["receipt_sha256"],
                "batch_manifest_sha256": batch_manifest.sha256,
            },
        )
    profiler: Any | None = None
    profiler_stopped = False

    def stage_callback(event: Any, engine: Any) -> None:
        if atomic_observer is None:
            return
        try:
            telemetry = {
                "stage_event": event.payload(),
                **_resource_telemetry(runtime.torch, device),
            }
            _record_atomic_stage_event(atomic_observer, event=event, telemetry=telemetry)
        except BaseException as error:
            state.stage_observer_failures.append(
                {
                    "schema_version": "nadig-vnext-worker-stage-observer-failure-v1",
                    "event": event.payload(),
                    **_failure_payload(error),
                }
            )

    def bounded_train_step(engine: Any, batch: Any, *, global_step: int) -> Any:
        nonlocal profiler
        if len(state.steps) >= protocol.total_steps:
            raise AssertionError("bounded census requested an N+1 training step")
        identity = ordered_batch_identity(
            batch,
            global_step=global_step,
            topology=engine.topology,
        )
        candidate_batches = (*state.batches, identity)
        try:
            census.require_batch_prefix(
                candidate_batches,
                batch_manifest.batches,
            )
        except ValueError as error:
            expected = batch_manifest.batches[len(state.batches)]
            state.batch_gate_failure = {
                "batch_index": len(state.batches),
                "expected_sha256": expected.sha256,
                "observed_sha256": identity.sha256,
                "expected": expected.payload(),
                "observed": identity.payload(),
                "optimizer_step_executed": False,
            }
            raise WorkerGateError(
                "training batch differs from the frozen semantic batch prefix"
            ) from error
        state.batches.append(identity)
        if not state.steps:
            runtime.torch.cuda.reset_peak_memory_stats(device)
            if protocol.torch_profiler_enabled:
                schedule = protocol.profiler_schedule
                assert schedule is not None
                profiler = runtime.torch.profiler.profile(
                    activities=[
                        runtime.torch.profiler.ProfilerActivity.CPU,
                        runtime.torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=runtime.torch.profiler.schedule(**schedule, repeat=1),
                    record_shapes=False,
                    profile_memory=True,
                    with_stack=False,
                )
                profiler.start()
        engine.stage_observer = stage_callback if atomic_observer is not None else None
        try:
            metrics = original_train_step(engine, batch, global_step=global_step)
        except BaseException:
            failures = getattr(engine, "stage_observer_failures", [])
            if failures:
                state.stage_observer_failures.extend(dict(value) for value in failures)
            raise
        runtime.torch.cuda.synchronize(device)
        if profiler is not None:
            profiler.step()
        metrics_payload = _metrics_payload(metrics)
        view_stats = getattr(engine, "last_view_stats", None)
        _require_finite_json_payload(view_stats, label="native step view statistics")
        step_payload = {
            "global_step": global_step,
            "phase": ("warmup" if len(state.steps) < protocol.warmup_steps else "measured"),
            "batch_identity_sha256": identity.sha256,
            "metrics": metrics_payload,
            "resource": _resource_telemetry(runtime.torch, device),
            "view_stats": view_stats,
        }
        _require_finite_json_payload(step_payload, label="native step receipt")
        state.steps.append(step_payload)
        if len(state.steps) > protocol.warmup_steps and protocol.timing_acceptance:
            value = float(metrics_payload["step_wall_ms"])
            if not math.isfinite(value) or value <= 0:
                raise WorkerGateError("native step timing is not finite and positive")
            state.timing_samples_ms.append(value)
        failures = getattr(engine, "stage_observer_failures", [])
        if failures:
            state.stage_observer_failures.extend(dict(value) for value in failures)
        if state.stage_observer_failures:
            raise WorkerGateError("native engine stage observer failed")
        if len(state.steps) == protocol.total_steps:
            raise BoundedCensusComplete(
                "bounded census completed immediately after its final training step"
            )
        return metrics

    def reject_validation(*_: object, **__: object) -> object:
        state.evaluation["validation_callback_count"] = (
            int(state.evaluation["validation_callback_count"]) + 1
        )
        raise TrainingOnlyAccessError("bounded census reached validation callback")

    runtime.engine_class.train_step = bounded_train_step
    runtime.native_execution.CanonicalEvaluationData = _evaluation_guard_factory(state.evaluation)
    runtime.native_execution.evaluate_validation_macro_delta = reject_validation
    try:
        runtime.native_execution.run_native_experiment(
            config_path=Path(binding.config_path),
            data_root=args.data_root,
            run_root=native_run_root,
            run_id=f"{binding.variant_id}-{args.stage_id}-{attempt_root.name}",
            run_seed=binding.run_seed,
            mode="pilot",
            device_name=args.device,
            repository_root=args.repository_root,
            formal=False,
            development_commit=args.development_commit,
            genept_preflight_receipt=genept_preflight[0],
            genept_preflight_receipt_sha256=genept_preflight[1],
        )
        state.primary_failure = RuntimeError(
            "bounded census unexpectedly reached validation/test lifecycle"
        )
    except BoundedCensusComplete:
        pass
    except BaseException as error:
        state.primary_failure = error
    finally:
        try:
            if profiler is not None and not profiler_stopped:
                _stop_profiler(
                    profiler,
                    evidence_root=attempt_root / "profile-evidence",
                    state=state,
                )
                profiler_stopped = True
        finally:
            runtime.engine_class.train_step = original_train_step
            runtime.native_execution.CanonicalEvaluationData = original_evaluator
            runtime.native_execution.evaluate_validation_macro_delta = original_validation

    try:
        state.native_identity_receipts = _collect_native_identity_receipts(attempt_root)
        _require_native_identity_receipts(
            state.native_identity_receipts,
            genept_required=binding.genept_preflight_required,
        )
    except BaseException as error:
        state.teardown_failures.append(
            {"stage": "native_identity_receipts", **_failure_payload(error)}
        )
        if state.primary_failure is None:
            state.primary_failure = WorkerGateError("native identity receipts are incomplete")
    try:
        state.persistent_pkl_scan = _scan_attempt_pkls(attempt_root)
        if not bool(state.persistent_pkl_scan["passed"]) and state.primary_failure is None:
            state.primary_failure = WorkerGateError(
                "bounded census attempt contains persistent PKL files"
            )
    except BaseException as error:
        state.teardown_failures.append({"stage": "persistent_pkl_scan", **_failure_payload(error)})

    training_only = _training_only_evidence(state.evaluation)
    try:
        census.require_training_only_evidence(training_only)
    except BaseException as error:
        if state.primary_failure is None:
            state.primary_failure = error
    capacity = _capacity_evidence(args, protocol=protocol, state=state)
    repository_predicates = (
        state.repository_identity.get("predicates")
        if isinstance(state.repository_identity, dict)
        else None
    )
    if state.primary_failure is None and (
        not isinstance(repository_predicates, dict)
        or not all(bool(value) for value in repository_predicates.values())
    ):
        state.primary_failure = WorkerGateError("repository identity predicate failed")
    if state.primary_failure is None and not bool(state.persistent_pkl_scan.get("passed")):
        state.primary_failure = WorkerGateError("whole-attempt zero-PKL predicate failed")
    capacity_predicates = capacity["predicates"]
    assert isinstance(capacity_predicates, dict)
    if state.primary_failure is None and not all(
        bool(value) for value in capacity_predicates.values()
    ):
        state.primary_failure = WorkerGateError("runtime capacity predicate failed")
    if atomic_observer is not None:
        atomic_observer.finalize(
            result={
                "observed_step_count": len(state.steps),
                "resource_preflight": dict(resource_preflight),
            },
            primary_failure=state.primary_failure,
            teardown_failures=state.teardown_failures,
        )
        progress_path = attempt_root / "stage-progress.json"
        try:
            state.terminal_stage_progress = {
                "relative_path": str(progress_path.relative_to(attempt_root)),
                "sha256": _sha256_file(progress_path),
                "size_bytes": progress_path.stat().st_size,
            }
        except BaseException as error:
            state.teardown_failures.append(
                {"stage": "terminal_stage_progress", **_failure_payload(error)}
            )
            if state.primary_failure is None:
                state.primary_failure = WorkerGateError(
                    "terminal stage-progress receipt could not be hash-pinned"
                )
    return state


def _capacity_evidence(
    args: argparse.Namespace,
    *,
    protocol: Any,
    state: WorkerState,
) -> dict[str, object]:
    resources = [step["resource"] for step in state.steps]
    typed = [value for value in resources if isinstance(value, dict)]
    total_bytes = max((int(value["gpu_total_bytes"]) for value in typed), default=0)
    minimum_free = min((int(value["gpu_free_bytes"]) for value in typed), default=0)
    required_free = max(
        args.minimum_gpu_free_bytes,
        math.ceil(total_bytes * args.minimum_gpu_headroom_fraction),
    )
    retry_oom = max(
        (
            sum(int(counter) for counter in value["cuda_retry_oom_counters"].values())
            for value in typed
        ),
        default=0,
    )
    predicates = {
        "exact_observed_step_count": len(state.steps) == protocol.total_steps,
        "zero_cuda_allocation_retries_or_ooms": retry_oom == 0,
        "gpu_free_bytes_at_least_required_headroom": minimum_free >= required_free,
    }
    return {
        "minimum_gpu_free_bytes": minimum_free,
        "gpu_total_bytes": total_bytes,
        "required_gpu_free_bytes": required_free,
        "cuda_retry_or_oom_counter_max": retry_oom,
        "predicates": predicates,
    }


def _build_stage_receipt(
    args: argparse.Namespace,
    *,
    binding: Any,
    attempt_root: Path,
    resource_preflight: Mapping[str, object],
    state: WorkerState,
    p0_preflight: Mapping[str, object],
    batch_manifest: Any,
    stage_prerequisite: Mapping[str, object] | None,
) -> dict[str, object]:
    protocol = census.STAGE_PROTOCOLS[args.stage_id]
    capacity = _capacity_evidence(args, protocol=protocol, state=state)
    preflight_predicates = resource_preflight.get("predicates")
    if not isinstance(preflight_predicates, dict):
        raise WorkerGateError("resource preflight lacks predicates")
    capacity_predicates = capacity["predicates"]
    assert isinstance(capacity_predicates, dict)
    if state.primary_failure is None and (
        not all(bool(value) for value in preflight_predicates.values())
        or not all(bool(value) for value in capacity_predicates.values())
    ):
        state.primary_failure = WorkerGateError("resource/capacity predicate failed")
    if state.teardown_failures and state.primary_failure is None:
        state.primary_failure = WorkerGateError("worker teardown failed")
    final_repository_predicates = (
        state.final_repository_identity.get("predicates")
        if isinstance(state.final_repository_identity, dict)
        else None
    )
    if state.primary_failure is None and (
        not isinstance(final_repository_predicates, dict)
        or not all(bool(value) for value in final_repository_predicates.values())
        or not isinstance(state.final_immutable_input_evidence, dict)
        or state.final_immutable_input_evidence.get("schema_version")
        != "nadig-vnext-performance-immutable-input-audit-v1"
    ):
        state.primary_failure = WorkerGateError(
            "terminal source or immutable-input evidence is incomplete"
        )
    training_only = _training_only_evidence(state.evaluation)
    batches = [batch.payload() for batch in state.batches]
    observed_prefix_sha256 = census.batch_sequence_sha256(state.batches)
    expected_prefix_sha256 = census.batch_sequence_sha256(
        batch_manifest.batches[: len(state.batches)]
    )
    batch_prefix_matches = observed_prefix_sha256 == expected_prefix_sha256
    if not batch_prefix_matches and state.primary_failure is None:
        state.primary_failure = WorkerGateError("observed batch prefix differs at receipt time")
    trace_sha = _sha256_file(state.profiler_trace) if state.profiler_trace else None
    table_sha = _sha256_file(state.profiler_table) if state.profiler_table else None
    return {
        "schema_version": "nadig-vnext-performance-stage-v1",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "variant_id": binding.variant_id,
        "config_sha256": binding.config_sha256,
        "matrix_sha256": binding.matrix_sha256,
        "binding": binding.payload(),
        "stage_id": args.stage_id,
        "protocol": protocol.payload(),
        "attempt_root": str(attempt_root),
        "native_run_root": str(attempt_root / "native-run"),
        "development_commit": args.development_commit,
        "repository_identity": state.repository_identity,
        "final_repository_identity": state.final_repository_identity,
        "final_immutable_input_evidence": state.final_immutable_input_evidence,
        "repository_root": str(args.repository_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "device": args.device,
        "p0_preflight": dict(p0_preflight),
        "frozen_batch_manifest": {
            "receipt_path": batch_manifest.path,
            "receipt_sha256": batch_manifest.sha256,
            "expected_batch_count": batch_manifest.frozen_prefix_count,
            "expected_sequence_sha256": batch_manifest.batch_sequence_sha256,
            "observed_prefix_count": len(state.batches),
            "observed_prefix_sha256": observed_prefix_sha256,
            "expected_prefix_sha256": expected_prefix_sha256,
            "prefix_matches": batch_prefix_matches,
        },
        "stage_prerequisite": (None if stage_prerequisite is None else dict(stage_prerequisite)),
        "genept_preflight": {
            "required": binding.genept_preflight_required,
            "receipt_path": (
                str(args.genept_preflight_receipt.resolve())
                if args.genept_preflight_receipt is not None
                else None
            ),
            "receipt_sha256": args.genept_preflight_receipt_sha256,
        },
        "status": "complete" if state.primary_failure is None else "failed",
        "resource_preflight_failure": (
            state.primary_failure is not None and not state.native_runtime_started
        ),
        "native_runtime_started": state.native_runtime_started,
        "training_only_evidence": training_only,
        "instrumentation": {
            "timing_acceptance": protocol.timing_acceptance,
            "heavy_capacity_instrumentation": protocol.heavy_capacity_instrumentation,
            "torch_profiler_enabled": protocol.torch_profiler_enabled,
            "step_timer": "native_train_step_cuda_synchronized_step_wall_ms",
            "stage_observer": (
                "atomic_per_native_phase" if protocol.heavy_capacity_instrumentation else "off"
            ),
        },
        "attempted_batch_count": len(state.batches),
        "completed_step_count": len(state.steps),
        "observed_step_count": len(state.steps),
        "batches": batches,
        "batch_gate_failure": state.batch_gate_failure,
        "batch_sequence_sha256": census.batch_sequence_sha256(state.batches),
        "timing_samples_ms": state.timing_samples_ms,
        "timing_summary_ms": (
            census.timing_summary(state.timing_samples_ms) if state.timing_samples_ms else None
        ),
        "steps": state.steps,
        "resource_preflight": dict(resource_preflight),
        "capacity_evidence": capacity,
        "stage_evidence": {
            "atomic_progress_receipt": (
                str(attempt_root / "stage-progress.json")
                if protocol.heavy_capacity_instrumentation
                else None
            ),
            "terminal_stage_progress": state.terminal_stage_progress,
            "stage_observer_failures": state.stage_observer_failures,
        },
        "native_identity_receipts": state.native_identity_receipts,
        "persistent_pkl_scan": state.persistent_pkl_scan,
        "torch_profiler_trace_sha256": trace_sha,
        "torch_profiler_table_sha256": table_sha,
        "primary_failure": (
            None if state.primary_failure is None else _failure_payload(state.primary_failure)
        ),
        "teardown_failures": state.teardown_failures,
    }


def _load_runtime() -> RuntimeModules:
    import torch

    import gradpert.execution.native as native_execution
    from gradpert.training.step import GraDPertStepEngine

    return RuntimeModules(
        torch=torch,
        native_execution=native_execution,
        engine_class=GraDPertStepEngine,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.source_publication_receipt is None or args.source_publication_receipt_sha256 is None:
        raise WorkerGateError("source publication receipt path/SHA are required")
    if not 0 < args.minimum_gpu_headroom_fraction < 1:
        raise WorkerGateError("minimum GPU headroom fraction must be in (0, 1)")
    numeric = (
        args.minimum_gpu_free_bytes,
        args.maximum_idle_gpu_utilization_percent,
        args.maximum_idle_gpu_memory_mib,
        args.minimum_disk_free_bytes,
        args.minimum_host_available_bytes,
    )
    if any(value < 0 for value in numeric):
        raise WorkerGateError("worker resource thresholds must be nonnegative")
    if args.stage_id not in SUPPORTED_STAGES:
        raise WorkerGateError("worker cannot execute P3 or an unknown stage")
    p1_pair_present = args.p1_receipt is not None and args.p1_receipt_sha256 is not None
    if (args.p1_receipt is None) != (args.p1_receipt_sha256 is None):
        raise WorkerGateError("P1 receipt path/SHA must be supplied together")
    if args.stage_id == "p1_capacity" and p1_pair_present:
        raise WorkerGateError("P1 capacity stage cannot accept a P1 prerequisite")
    if args.stage_id in {"p2_timing", "diagnostic_profile"} and not p1_pair_present:
        raise WorkerGateError(f"{args.stage_id} requires a P1 prerequisite")


def _resolve_preclaim_inputs(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, object], tuple[Path | None, str | None], Any]:
    _validate_args(args)
    binding = census.bind_matrix_variant(
        args.matrix,
        repository_root=args.repository_root,
        expected_matrix_sha256=args.expected_matrix_sha256,
        variant_id=args.variant_id,
    )
    try:
        census.require_performance_worker_variant(binding.variant_id, stage_id=args.stage_id)
    except ValueError as error:
        raise WorkerGateError(str(error)) from error
    a0_binding = census.bind_matrix_variant(
        args.matrix,
        repository_root=args.repository_root,
        expected_matrix_sha256=args.expected_matrix_sha256,
        variant_id=census.A0_VARIANT_ID,
    )
    batch_manifest = census.load_frozen_batch_manifest(
        args.batch_manifest,
        expected_sha256=args.batch_manifest_sha256,
        expected_matrix_sha256=binding.matrix_sha256,
        expected_config_sha256=a0_binding.config_sha256,
    )
    p0_preflight = _resolve_p0_preflight(
        args,
        binding=binding,
        batch_manifest=batch_manifest,
    )
    genept_preflight = _resolve_genept_preflight(args, binding=binding)
    stage_prerequisite = _resolve_stage_prerequisite(
        args,
        binding=binding,
        batch_manifest=batch_manifest,
        p0_preflight=p0_preflight,
    )
    immutable_input_evidence = _require_immutable_inputs(
        binding=binding,
        p0_preflight=p0_preflight,
        batch_manifest=batch_manifest,
        stage_prerequisite=stage_prerequisite,
        data_root=args.data_root,
    )
    p0_preflight["preclaim_immutable_input_evidence"] = immutable_input_evidence
    return (
        binding,
        a0_binding,
        batch_manifest,
        p0_preflight,
        genept_preflight,
        stage_prerequisite,
    )


def _write_preclaim_failure(args: argparse.Namespace, error: BaseException) -> Path:
    root = args.census_root.resolve() / "_preclaim_failures"
    root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f"{args.variant_id}-{args.stage_id}-",
        suffix=".json",
        dir=root,
    )
    path = Path(name)
    payload = {
        "schema_version": "nadig-vnext-performance-preclaim-failure-v1",
        "status": "failed",
        "evidence_class": "performance_preflight_only",
        "scientific_completion": False,
        "variant_id": args.variant_id,
        "stage_id": args.stage_id,
        "matrix_path": str(args.matrix),
        "expected_matrix_sha256": args.expected_matrix_sha256,
        "development_commit": args.development_commit,
        "cuda_runtime_loaded": False,
        "attempt_root_claimed": False,
        "primary_failure": _failure_payload(error),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def _write_terminal_stage_receipt(
    receipt_path: Path,
    args: argparse.Namespace,
    *,
    binding: Any,
    attempt_root: Path,
    resource_preflight: Mapping[str, object],
    state: WorkerState,
    p0_preflight: Mapping[str, object],
    batch_manifest: Any,
    stage_prerequisite: Mapping[str, object] | None,
) -> dict[str, object]:
    """Replace the running marker with complete evidence or a safe terminal failure."""

    try:
        receipt = _build_stage_receipt(
            args,
            binding=binding,
            attempt_root=attempt_root,
            resource_preflight=resource_preflight,
            state=state,
            p0_preflight=p0_preflight,
            batch_manifest=batch_manifest,
            stage_prerequisite=stage_prerequisite,
        )
        _require_finite_json_payload(receipt, label="terminal stage receipt")
        _atomic_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        if state.primary_failure is None:
            state.primary_failure = WorkerGateError(
                f"terminal stage receipt construction failed: {error}"
            )
        else:
            state.teardown_failures.append(
                {"stage": "terminal_stage_receipt", **_failure_payload(error)}
            )
        fallback = {
            "schema_version": "nadig-vnext-performance-stage-v1",
            "evidence_class": "performance_training_only",
            "scientific_completion": False,
            "variant_id": binding.variant_id,
            "config_sha256": binding.config_sha256,
            "matrix_sha256": binding.matrix_sha256,
            "binding": binding.payload(),
            "stage_id": args.stage_id,
            "protocol": census.STAGE_PROTOCOLS[args.stage_id].payload(),
            "attempt_root": str(attempt_root),
            "development_commit": args.development_commit,
            "status": "failed",
            "running_receipt_replaced": True,
            "completed_step_count": len(state.steps),
            "attempted_batch_count": len(state.batches),
            "primary_failure": _failure_payload(state.primary_failure),
            "teardown_failures": state.teardown_failures,
            "receipt_construction_failure": _failure_payload(error),
        }
        _require_finite_json_payload(fallback, label="fallback terminal stage receipt")
        _atomic_json(receipt_path, fallback)
        return fallback


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        (
            binding,
            _a0_binding,
            batch_manifest,
            p0_preflight,
            genept_preflight,
            stage_prerequisite,
        ) = _resolve_preclaim_inputs(args)
    except BaseException as error:
        path = _write_preclaim_failure(args, error)
        output = {
            "status": "failed_preclaim",
            "receipt_path": str(path),
            "receipt_sha256": _sha256_file(path),
        }
        print(json.dumps(output, sort_keys=True) if args.as_json else f"failed: {path}")
        return 1
    attempt_root = census.claim_fresh_attempt_root(
        args.census_root,
        variant_id=binding.variant_id,
        stage_id=args.stage_id,
    )
    receipt_path = attempt_root / "stage-receipt.json"
    _atomic_json(
        receipt_path,
        {
            "schema_version": "nadig-vnext-performance-stage-v1",
            "variant_id": binding.variant_id,
            "config_sha256": binding.config_sha256,
            "matrix_sha256": binding.matrix_sha256,
            "stage_id": args.stage_id,
            "protocol": census.STAGE_PROTOCOLS[args.stage_id].payload(),
            "status": "running",
        },
    )
    state = WorkerState()
    try:
        repository_identity = _repository_identity_evidence(
            args,
            expected_p0_source=p0_preflight["source"],
        )
        repository_predicates = repository_identity["predicates"]
        assert isinstance(repository_predicates, dict)
        state.repository_identity = repository_identity
        if not all(bool(value) for value in repository_predicates.values()):
            raise WorkerGateError("repository identity is dirty or differs from development commit")
        resource_preflight = _physical_gpu_preflight(args, attempt_root)
        predicates = resource_preflight["predicates"]
        assert isinstance(predicates, dict)
        if not all(bool(value) for value in predicates.values()):
            raise WorkerGateError("physical GPU/host/disk preflight failed")
        _require_prerequisite_gpu(stage_prerequisite, resource_preflight)
        runtime = _load_runtime()
        state.native_runtime_started = True
        state = _execute_bounded_native(
            args,
            binding=binding,
            attempt_root=attempt_root,
            runtime=runtime,
            resource_preflight=resource_preflight,
            repository_identity=repository_identity,
            genept_preflight=genept_preflight,
            batch_manifest=batch_manifest,
            p0_preflight=p0_preflight,
        )
        state.native_runtime_started = True
    except BaseException as error:
        resource_preflight = locals().get(
            "resource_preflight",
            {
                "schema_version": "nadig-vnext-performance-resource-preflight-v1",
                "predicates": {},
                "failure": _failure_payload(error),
            },
        )
        state.primary_failure = error
        state.repository_identity = locals().get("repository_identity")
        try:
            state.native_identity_receipts = _collect_native_identity_receipts(attempt_root)
        except BaseException as evidence_error:
            state.teardown_failures.append(
                {"stage": "native_identity_receipts", **_failure_payload(evidence_error)}
            )
        try:
            state.persistent_pkl_scan = _scan_attempt_pkls(attempt_root)
        except BaseException as evidence_error:
            state.teardown_failures.append(
                {"stage": "persistent_pkl_scan", **_failure_payload(evidence_error)}
            )
    _record_final_repository_identity(
        args,
        state=state,
        expected_p0_source=p0_preflight["source"],
    )
    try:
        state.final_immutable_input_evidence = _require_immutable_inputs(
            binding=binding,
            p0_preflight=p0_preflight,
            batch_manifest=batch_manifest,
            stage_prerequisite=stage_prerequisite,
            data_root=args.data_root,
        )
    except BaseException as evidence_error:
        if state.primary_failure is None:
            state.primary_failure = evidence_error
        else:
            state.teardown_failures.append(
                {"stage": "final_prerequisite_rehash", **_failure_payload(evidence_error)}
            )
    receipt = _write_terminal_stage_receipt(
        receipt_path,
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=resource_preflight,
        state=state,
        p0_preflight=p0_preflight,
        batch_manifest=batch_manifest,
        stage_prerequisite=stage_prerequisite,
    )
    output = {
        "attempt_root": str(attempt_root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "status": receipt["status"],
    }
    if args.as_json:
        print(json.dumps(output, sort_keys=True))
    else:
        print(f"{output['status']}: {receipt_path}")
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
