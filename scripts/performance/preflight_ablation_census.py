#!/usr/bin/env python3
"""Run the non-CUDA P0 closure for the exact 30-row ablation census."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import ablation_performance_census as census  # noqa: E402

from gradpert.config import load_experiment_config  # noqa: E402
from gradpert.config.native import NativeArchitectureOptions  # noqa: E402
from gradpert.contracts import (  # noqa: E402
    CanonicalDataManifest,
    SourceManifest,
    SplitManifest,
)
from gradpert.data import DatasetLayout  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.graphs import (  # noqa: E402
    resolve_legacy_local_view_contract,
    resolve_local_view_contract,
)
from gradpert.hashing import sha256_file  # noqa: E402
from gradpert.pilots import (  # noqa: E402
    GenePTSeedAvailabilityReceipt,
    load_vnext_runtime_graph_topology,
)

SUPPORTED_HVG_COUNTS = (512, 1024, 2048, 5000)
EXPECTED_REPOSITORY = "https://github.com/elan6666/GraD-Pert"
SOURCE_REMOTE_REF = "refs/heads/main"


class PreflightError(RuntimeError):
    """Raised when a global P0 identity cannot be trusted."""


@dataclass(frozen=True)
class DataIdentity:
    dataset_id: str
    protocol_id: str
    canonical_manifest_path: str
    canonical_manifest_sha256: str
    canonical_manifest_size_bytes: int
    canonical_data_path: str
    canonical_data_sha256: str
    canonical_data_size_bytes: int
    observation_order_sha256: str
    split_manifest_path: str
    split_manifest_sha256: str
    split_manifest_size_bytes: int
    split_content_sha256: str
    source_manifest_path: str
    source_manifest_sha256: str
    source_manifest_size_bytes: int
    source_h5ad_path: str
    source_h5ad_sha256: str
    source_h5ad_size_bytes: int

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["artifacts"] = {
            "canonical_manifest": {
                "path": self.canonical_manifest_path,
                "sha256": self.canonical_manifest_sha256,
                "size_bytes": self.canonical_manifest_size_bytes,
                "role": "canonical_data_manifest",
            },
            "canonical_h5ad": {
                "path": self.canonical_data_path,
                "sha256": self.canonical_data_sha256,
                "size_bytes": self.canonical_data_size_bytes,
                "role": "canonical_expression_and_metadata",
            },
            "split_manifest": {
                "path": self.split_manifest_path,
                "sha256": self.split_manifest_sha256,
                "size_bytes": self.split_manifest_size_bytes,
                "role": "canonical_condition_split",
            },
            "source_manifest": {
                "path": self.source_manifest_path,
                "sha256": self.source_manifest_sha256,
                "size_bytes": self.source_manifest_size_bytes,
                "role": "source_data_manifest",
            },
            "source_h5ad": {
                "path": self.source_h5ad_path,
                "sha256": self.source_h5ad_sha256,
                "size_bytes": self.source_h5ad_size_bytes,
                "role": "source_expression_and_metadata",
            },
        }
        return payload


@dataclass(frozen=True)
class PreflightDependencies:
    inspect_source: Callable[[Path, str, Path, str], Mapping[str, object]]
    load_data_identity: Callable[[Path, str], DataIdentity]
    load_graph: Callable[[Path], tuple[Any, Any]]
    verify_artifact: Callable[[Path, str, int], Mapping[str, object]]


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
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", type=_commit_argument, required=True)
    parser.add_argument("--source-publication-receipt", type=Path, required=True)
    parser.add_argument(
        "--source-publication-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--genept-preflight-receipt", type=Path)
    parser.add_argument("--genept-preflight-receipt-sha256", type=_sha256_argument)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


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


def _claim_json_output(path: Path, payload: object) -> Path:
    """Atomically reserve the one immutable P0 receipt destination."""

    parent = path.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
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
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _failure_payload(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _runtime_import_snapshot() -> dict[str, object]:
    torch_modules = sorted(
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    )
    cuda_modules = [
        name for name in torch_modules if name == "torch.cuda" or name.startswith("torch.cuda.")
    ]
    return {
        "measurement_method": "sys.modules_snapshot_without_importing_torch",
        "torch_loaded": bool(torch_modules),
        "torch_cuda_loaded": bool(cuda_modules),
        "torch_module_count": len(torch_modules),
        "torch_cuda_module_count": len(cuda_modules),
        "torch_modules": torch_modules,
        "torch_cuda_modules": cuda_modules,
    }


def _require_no_torch_imports(stage: str) -> dict[str, object]:
    snapshot = _runtime_import_snapshot()
    if snapshot["torch_loaded"] is not False or snapshot["torch_cuda_loaded"] is not False:
        raise PreflightError(f"P0 {stage} detected a loaded Torch/CUDA module")
    return snapshot


def _runtime_import_guard(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    before_names = before.get("torch_modules")
    after_names = after.get("torch_modules")
    if not isinstance(before_names, list) or not isinstance(after_names, list):
        raise PreflightError("P0 runtime import snapshots are malformed")
    return {
        "status": "passed",
        "measurement_method": "sys.modules_snapshot_without_importing_torch",
        "before": dict(before),
        "after": dict(after),
        "new_torch_modules": sorted(set(after_names) - set(before_names)),
    }


def _require_resolved_within(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise PreflightError(f"P0 {label} escapes its frozen root")
    if not resolved.is_file():
        raise PreflightError(f"P0 {label} is not a regular file")
    return resolved


def _normalized_repository_url(value: str) -> str:
    normalized = value.strip().removesuffix(".git").rstrip("/")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized


def _inspect_clean_source(
    repository_root: Path,
    expected_commit: str,
    publication_receipt: Path,
    expected_publication_receipt_sha256: str,
) -> dict[str, object]:
    root = repository_root.resolve(strict=True)
    if not publication_receipt.is_file() or publication_receipt.is_symlink():
        raise PreflightError("P0 source publication receipt must be a regular file")
    resolved_publication_receipt = publication_receipt.resolve(strict=True)
    observed_publication_receipt_sha256 = sha256_file(resolved_publication_receipt)
    if observed_publication_receipt_sha256 != expected_publication_receipt_sha256:
        raise PreflightError("P0 source publication receipt SHA-256 differs")
    identity = inspect_source_identity(
        root,
        formal=True,
        expected_repository=EXPECTED_REPOSITORY,
        remote_ref=SOURCE_REMOTE_REF,
        publication_receipt=resolved_publication_receipt,
        expected_publication_receipt_sha256=expected_publication_receipt_sha256,
    )
    head_tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if identity.commit != expected_commit:
        raise PreflightError("P0 source commit differs from the frozen commit")
    if identity.dirty:
        raise PreflightError("P0 requires a clean source worktree")
    if (
        identity.published_commit != expected_commit
        or identity.remote_ref != SOURCE_REMOTE_REF
        or identity.formal_eligible is not True
        or identity.formal_eligibility_reason is not None
        or identity.publication_receipt_sha256 != expected_publication_receipt_sha256
        or not isinstance(identity.remote_url, str)
        or _normalized_repository_url(identity.remote_url)
        != _normalized_repository_url(EXPECTED_REPOSITORY)
    ):
        raise PreflightError("P0 formal publication identity differs from the frozen source")
    return {
        "schema_version": "nadig-vnext-performance-source-identity-v1",
        "repository_root": str(root),
        "expected_repository": EXPECTED_REPOSITORY,
        "expected_commit": expected_commit,
        "observed_commit": identity.commit,
        "git_tree_object": head_tree,
        "source_tree_sha256": identity.tree_sha256,
        "source_tree_identity_method": "gradpert.execution.identity.inspect_source_identity",
        "remote_url": identity.remote_url,
        "remote_ref": identity.remote_ref,
        "published_commit": identity.published_commit,
        "formal_eligible": identity.formal_eligible,
        "formal_eligibility_reason": identity.formal_eligibility_reason,
        "publication_verification_method": "hash_pinned_source_publication_receipt",
        "publication_receipt_path": str(resolved_publication_receipt),
        "publication_receipt_sha256": identity.publication_receipt_sha256,
        "publication_receipt_size_bytes": resolved_publication_receipt.stat().st_size,
        "source_dirty": False,
    }


def _validate_source_evidence(
    source: Mapping[str, object],
    *,
    repository_root: Path,
    expected_commit: str,
    publication_receipt: Path,
    expected_publication_receipt_sha256: str,
) -> None:
    expected_root = str(repository_root.resolve(strict=True))
    if not publication_receipt.is_file() or publication_receipt.is_symlink():
        raise PreflightError("P0 source publication receipt must be a regular file")
    resolved_publication_receipt = publication_receipt.resolve(strict=True)
    observed_publication_receipt_sha256 = sha256_file(resolved_publication_receipt)
    observed_publication_receipt_size = resolved_publication_receipt.stat().st_size
    if observed_publication_receipt_sha256 != expected_publication_receipt_sha256:
        raise PreflightError("P0 source publication receipt SHA-256 differs")
    if (
        source.get("repository_root") != expected_root
        or source.get("expected_repository") != EXPECTED_REPOSITORY
        or source.get("expected_commit") != expected_commit
        or source.get("observed_commit") != expected_commit
        or source.get("published_commit") != expected_commit
        or source.get("remote_ref") != SOURCE_REMOTE_REF
        or source.get("formal_eligible") is not True
        or source.get("formal_eligibility_reason") is not None
        or source.get("publication_verification_method") != "hash_pinned_source_publication_receipt"
        or source.get("publication_receipt_path") != str(resolved_publication_receipt)
        or source.get("publication_receipt_sha256") != expected_publication_receipt_sha256
        or source.get("publication_receipt_size_bytes") != observed_publication_receipt_size
        or source.get("source_dirty") is not False
    ):
        raise PreflightError("P0 source identity evidence differs from the frozen source")
    remote_url = source.get("remote_url")
    if not isinstance(remote_url, str) or _normalized_repository_url(remote_url) != (
        _normalized_repository_url(EXPECTED_REPOSITORY)
    ):
        raise PreflightError("P0 source remote differs from the experiment repository")
    for field, length in (("git_tree_object", 40), ("source_tree_sha256", 64)):
        value = source.get(field)
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PreflightError(f"P0 source identity lacks a valid {field}")


def _validate_data_identity_evidence(
    identity: DataIdentity,
    *,
    protocol_id: str,
) -> None:
    if identity.dataset_id != "nadig_jurkat" or identity.protocol_id != protocol_id:
        raise PreflightError("P0 live data identity differs from the frozen dataset/protocol")
    for artifact_id, artifact in identity.payload()["artifacts"].items():
        if not isinstance(artifact, Mapping):
            raise PreflightError(f"P0 data artifact identity is malformed: {artifact_id}")
        path = artifact.get("path")
        sha256 = artifact.get("sha256")
        size_bytes = artifact.get("size_bytes")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise PreflightError(f"P0 data artifact path must be absolute: {artifact_id}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise PreflightError(f"P0 data artifact SHA-256 is invalid: {artifact_id}")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 1:
            raise PreflightError(f"P0 data artifact size is invalid: {artifact_id}")
    if (
        len(identity.observation_order_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in identity.observation_order_sha256
        )
        or len(identity.split_content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in identity.split_content_sha256)
    ):
        raise PreflightError("P0 semantic data content SHA-256 is invalid")


def _load_live_data_identity(data_root: Path, protocol_id: str) -> DataIdentity:
    resolved_data_root = data_root.resolve(strict=True)
    layout = DatasetLayout(resolved_data_root, "nadig_jurkat", protocol_id)
    canonical_path = _require_resolved_within(
        resolved_data_root,
        layout.canonical_manifest,
        label="canonical manifest",
    )
    split_path = _require_resolved_within(
        resolved_data_root,
        layout.manifests / "split.json",
        label="split manifest",
    )
    source_path = _require_resolved_within(
        resolved_data_root,
        layout.manifests / "source.json",
        label="source manifest",
    )
    canonical = CanonicalDataManifest.model_validate_json(
        canonical_path.read_text(encoding="utf-8")
    )
    split = SplitManifest.model_validate_json(split_path.read_text(encoding="utf-8"))
    source = SourceManifest.model_validate_json(source_path.read_text(encoding="utf-8"))
    if any(
        value != "nadig_jurkat"
        for value in (canonical.dataset_id, split.dataset_id, source.dataset_id)
    ) or any(
        value != protocol_id
        for value in (canonical.protocol_id, split.protocol_id, source.protocol_id)
    ):
        raise PreflightError("P0 canonical/source/split dataset identity differs")
    split_file_sha = sha256_file(split_path)
    source_file_sha = sha256_file(source_path)
    if (
        canonical.split_manifest_sha256 != split_file_sha
        or canonical.split_content_sha256 != split.split_content_sha256
        or canonical.source_manifest_sha256 != source_file_sha
    ):
        raise PreflightError("P0 canonical manifest links differ from live source/split files")
    canonical_adata = _require_resolved_within(
        resolved_data_root,
        layout.canonical_adata,
        label="canonical H5AD",
    )
    source_h5ad = _require_resolved_within(
        resolved_data_root,
        layout.source / source.filename,
        label="source H5AD",
    )
    canonical_data_sha = sha256_file(canonical_adata, chunk_size=8 * 1024 * 1024)
    source_h5ad_sha = sha256_file(source_h5ad, chunk_size=8 * 1024 * 1024)
    if canonical_data_sha != canonical.canonical_adata_sha256:
        raise PreflightError("P0 canonical H5AD SHA-256 differs")
    if source_h5ad_sha != source.source_sha256:
        raise PreflightError("P0 source H5AD SHA-256 differs")
    return DataIdentity(
        dataset_id="nadig_jurkat",
        protocol_id=protocol_id,
        canonical_manifest_path=str(canonical_path),
        canonical_manifest_sha256=sha256_file(canonical_path),
        canonical_manifest_size_bytes=canonical_path.stat().st_size,
        canonical_data_path=str(canonical_adata),
        canonical_data_sha256=canonical_data_sha,
        canonical_data_size_bytes=canonical_adata.stat().st_size,
        observation_order_sha256=canonical.observation_order_sha256,
        split_manifest_path=str(split_path),
        split_manifest_sha256=split_file_sha,
        split_manifest_size_bytes=split_path.stat().st_size,
        split_content_sha256=split.split_content_sha256,
        source_manifest_path=str(source_path),
        source_manifest_sha256=source_file_sha,
        source_manifest_size_bytes=source_path.stat().st_size,
        source_h5ad_path=str(source_h5ad),
        source_h5ad_sha256=source_h5ad_sha,
        source_h5ad_size_bytes=source_h5ad.stat().st_size,
    )


def _verify_live_artifact(
    path: Path,
    expected_sha256: str,
    expected_size: int,
) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    observed_size = resolved.stat().st_size
    observed_sha = sha256_file(resolved, chunk_size=8 * 1024 * 1024)
    if observed_size != expected_size or observed_sha != expected_sha256:
        raise PreflightError("GenePT source artifact size/SHA-256 differs from its receipt")
    return {
        "path": str(resolved),
        "size_bytes": observed_size,
        "sha256": observed_sha,
    }


def _parameter(config: Any, name: str) -> object:
    try:
        return config.model.parameters[name].value
    except KeyError as error:
        raise PreflightError(f"P0 config lacks required parameter: {name}") from error


def _resolve_local_contract(architecture: NativeArchitectureOptions, graph_node_count: int) -> Any:
    if architecture.legacy_local_view_node_budget is None:
        return resolve_local_view_contract(
            graph_node_count=graph_node_count,
            local_view_count=architecture.local_view_count,
            node_budget_ratio=(
                architecture.local_view_node_budget_ratio_numerator,
                architecture.local_view_node_budget_ratio_denominator,
            ),
            mask_view_ratio=(
                architecture.local_anchor_mask_view_ratio_numerator,
                architecture.local_anchor_mask_view_ratio_denominator,
            ),
        )
    if architecture.legacy_local_anchor_mask_count is None:
        raise PreflightError("legacy local budget lacks its mask-count contract")
    return resolve_legacy_local_view_contract(
        graph_node_count=graph_node_count,
        local_view_count=architecture.local_view_count,
        fixed_node_budget=architecture.legacy_local_view_node_budget,
        fixed_mask_view_count=architecture.legacy_local_anchor_mask_count,
    )


def _safe_runtime_graph_root(data_root: Path, label: object) -> Path:
    if not isinstance(label, str) or not label:
        raise PreflightError("P0 runtime_graph_root must be nonempty")
    relative = Path(label)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PreflightError("P0 runtime_graph_root must be a safe relative path")
    resolved_data_root = data_root.resolve(strict=True)
    root = resolved_data_root.joinpath(*relative.parts).resolve(strict=False)
    if not root.is_relative_to(resolved_data_root):
        raise PreflightError("P0 runtime graph escapes the data root")
    return root


def _ranking_receipt_filename(hvg_count: int) -> str:
    if hvg_count not in SUPPORTED_HVG_COUNTS:
        raise PreflightError("P0 graph artifact capture requires a supported HVG count")
    return f"hvg{hvg_count}_dispersion_ranking.json"


def _capture_graph_artifacts(
    graph_root: Path,
    hvg_count: int,
    *,
    graph_axis_policy: str = "recomputed_hvg_union_candidate_targets",
) -> dict[str, object]:
    specifications = {
        "manifest": ("manifest.json", "runtime_graph_manifest"),
        "graph_gene_ids": ("graph_gene_ids.txt", "ordered_graph_gene_axis"),
        "go": ("go.npz", "pruned_go_graph"),
        "string": ("string.npz", "pruned_string_graph"),
    }
    if graph_axis_policy == "recomputed_hvg_union_candidate_targets":
        specifications["hvg_dispersion_ranking"] = (
            _ranking_receipt_filename(hvg_count),
            "hvg_dispersion_ranking_receipt",
        )
    artifacts: dict[str, object] = {}
    for artifact_id, (filename, role) in specifications.items():
        resolved = _require_resolved_within(
            graph_root,
            graph_root / filename,
            label=f"runtime graph artifact {filename}",
        )
        artifacts[artifact_id] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved, chunk_size=8 * 1024 * 1024),
            "size_bytes": resolved.stat().st_size,
            "role": role,
        }
    return artifacts


def _artifact_sha256(artifacts: Mapping[str, object], artifact_id: str) -> str:
    artifact = artifacts.get(artifact_id)
    if not isinstance(artifact, Mapping):
        raise PreflightError(f"P0 graph artifact identity is missing: {artifact_id}")
    value = artifact.get("sha256")
    if not isinstance(value, str):
        raise PreflightError(f"P0 graph artifact SHA-256 is missing: {artifact_id}")
    return value


def _ordered_subsequence(smaller: Sequence[str], larger: Sequence[str]) -> bool:
    iterator = iter(larger)
    return all(any(candidate == gene for candidate in iterator) for gene in smaller)


def _cross_h_audit(
    manifests: Mapping[int, Any],
    manifest_hashes: Mapping[int, str],
) -> dict[str, object]:
    if set(manifests) != set(SUPPORTED_HVG_COUNTS):
        return {
            "status": "blocked_missing_graph",
            "required_hvg_counts": list(SUPPORTED_HVG_COUNTS),
            "available_hvg_counts": sorted(manifests),
        }
    shared_fields = (
        "dataset_id",
        "protocol_id",
        "canonical_data_sha256",
        "split_content_sha256",
        "source_h5ad_sha256",
        "source_registry_sha256",
        "hvg_method",
        "hvg_flavor",
        "normalize_total",
        "log1p",
        "hvg_subset",
        "expression_gene_count",
        "hvg_fit_scope",
        "hvg_fit_cell_count",
        "hvg_fit_condition_ids",
        "hvg_fit_condition_ids_sha256",
        "candidate_target_ids",
        "candidate_target_order_sha256",
        "gene_feature_policy",
    )
    reference = manifests[512]
    for hvg_count in SUPPORTED_HVG_COUNTS:
        manifest = manifests[hvg_count]
        if manifest.requested_hvg_count != hvg_count:
            raise PreflightError(f"H{hvg_count} graph count differs from its config slot")
        differing = [
            field
            for field in shared_fields
            if getattr(manifest, field) != getattr(reference, field)
        ]
        if differing:
            raise PreflightError(
                f"H{hvg_count} graph lineage differs from H512: {', '.join(differing)}"
            )
        if manifest.gene_feature_policy != "learned_id":
            raise PreflightError("P0 H closure requires learned-ID parent graph axes")
    nested_pairs: list[str] = []
    for smaller_count, larger_count in pairwise(SUPPORTED_HVG_COUNTS):
        smaller = manifests[smaller_count]
        larger = manifests[larger_count]
        if not set(smaller.direct_hvg_gene_ids) < set(larger.direct_hvg_gene_ids):
            raise PreflightError(f"H{smaller_count} direct HVGs are not nested")
        if not _ordered_subsequence(smaller.direct_hvg_gene_ids, larger.direct_hvg_gene_ids):
            raise PreflightError(f"H{smaller_count} direct-HVG order is not preserved")
        if larger.normalized_dispersion_ranked_hvg_gene_ids[:smaller_count] != (
            smaller.normalized_dispersion_ranked_hvg_gene_ids
        ):
            raise PreflightError(f"H{smaller_count} dispersion ranking is not a prefix")
        if not set(smaller.graph_gene_ids) < set(larger.graph_gene_ids):
            raise PreflightError(f"H{smaller_count} graph axis is not nested")
        nested_pairs.append(f"{smaller_count}<{larger_count}")
    return {
        "status": "passed",
        "required_hvg_counts": list(SUPPORTED_HVG_COUNTS),
        "available_hvg_counts": list(SUPPORTED_HVG_COUNTS),
        "manifest_sha256_by_hvg_count": {
            str(value): manifest_hashes[value] for value in SUPPORTED_HVG_COUNTS
        },
        "graph_gene_order_sha256_by_hvg_count": {
            str(value): manifests[value].graph_gene_order_sha256 for value in SUPPORTED_HVG_COUNTS
        },
        "topology_content_sha256_by_hvg_count": {
            str(value): manifests[value].topology_content_sha256 for value in SUPPORTED_HVG_COUNTS
        },
        "candidate_target_order_sha256": reference.candidate_target_order_sha256,
        "nested_pairs": nested_pairs,
    }


def _load_genept_receipt(
    path: Path | None,
    expected_file_sha256: str | None,
) -> tuple[GenePTSeedAvailabilityReceipt | None, dict[str, object] | None, str | None]:
    if (path is None) != (expected_file_sha256 is None):
        return None, None, "GenePT receipt path/SHA must be supplied together"
    if path is None:
        return None, None, "GenePT Seed preflight receipt is missing"
    try:
        resolved = path.resolve(strict=True)
        observed_sha = sha256_file(resolved)
        if observed_sha != expected_file_sha256:
            raise PreflightError("GenePT preflight receipt file SHA-256 differs")
        receipt = GenePTSeedAvailabilityReceipt.model_validate_json(
            resolved.read_text(encoding="utf-8")
        )
        return (
            receipt,
            {
                "path": str(resolved),
                "sha256": observed_sha,
                "size_bytes": resolved.stat().st_size,
            },
            None,
        )
    except BaseException as error:
        return None, None, str(error)


def _genept_row_evidence(
    *,
    config: Any,
    graph_root_label: str,
    graph_manifest_path: Path,
    graph_manifest: Any,
    receipt: GenePTSeedAvailabilityReceipt | None,
    receipt_identity: Mapping[str, object] | None,
    receipt_error: str | None,
    verify_artifact: Callable[[Path, str, int], Mapping[str, object]],
) -> tuple[dict[str, object], str | None]:
    if receipt is None:
        return {
            "status": "blocked_missing_or_invalid_preflight",
            "reason": receipt_error,
        }, receipt_error
    configured_path = _parameter(config, "genept_artifact_path")
    configured_sha = _parameter(config, "genept_expected_sha256")
    expected = {
        "genept_source_path": configured_path,
        "genept_source_sha256": configured_sha,
        "runtime_graph_root": graph_root_label,
        "parent_graph_manifest_sha256": sha256_file(graph_manifest_path),
        "parent_topology_content_sha256": graph_manifest.topology_content_sha256,
        "parent_graph_gene_order_sha256": graph_manifest.graph_gene_order_sha256,
        "selected_gene_count": graph_manifest.graph_gene_count,
        "selected_gene_order_sha256": graph_manifest.graph_gene_order_sha256,
        "candidate_target_order_sha256": graph_manifest.candidate_target_order_sha256,
        "perturbation_target_gene_count": len(graph_manifest.candidate_target_ids),
        "perturbation_target_gene_ids_sha256": graph_manifest.candidate_target_order_sha256,
    }
    observed = {name: getattr(receipt, name) for name in expected}
    differing = sorted(name for name in expected if observed[name] != expected[name])
    if differing:
        reason = "GenePT preflight differs from live inputs: " + ", ".join(differing)
        return {"status": "blocked_identity_mismatch", "reason": reason}, reason
    try:
        artifact = verify_artifact(
            Path(receipt.genept_source_path),
            receipt.genept_source_sha256,
            receipt.genept_source_size_bytes,
        )
    except BaseException as error:
        return {"status": "blocked_artifact_mismatch", "reason": str(error)}, str(error)
    return {
        "status": "passed",
        "receipt": dict(receipt_identity or {}),
        "prior_contract_id": receipt.prior_contract_id,
        "artifact": dict(artifact),
        "selected_gene_count": receipt.selected_gene_count,
        "selected_gene_order_sha256": receipt.selected_gene_order_sha256,
        "zero_vector_gene_count": receipt.zero_vector_gene_count,
    }, None


def build_preflight_receipt(
    *,
    matrix_path: Path,
    expected_matrix_sha256: str,
    repository_root: Path,
    expected_source_commit: str,
    source_publication_receipt: Path,
    source_publication_receipt_sha256: str,
    data_root: Path,
    genept_preflight_receipt: Path | None,
    genept_preflight_receipt_sha256: str | None,
    dependencies: PreflightDependencies | None = None,
) -> dict[str, object]:
    runtime_before = _require_no_torch_imports("entry")
    deps = dependencies or PreflightDependencies(
        inspect_source=_inspect_clean_source,
        load_data_identity=_load_live_data_identity,
        load_graph=load_vnext_runtime_graph_topology,
        verify_artifact=_verify_live_artifact,
    )
    bindings = census.bind_matrix_variants(
        matrix_path,
        repository_root=repository_root,
        expected_matrix_sha256=expected_matrix_sha256,
    )
    if len(bindings) != census.MATRIX_ROW_COUNT or [
        binding.matrix_row_index for binding in bindings
    ] != list(range(census.MATRIX_ROW_COUNT)):
        raise PreflightError(
            f"P0 requires all exact {census.MATRIX_ROW_COUNT} matrix rows in order"
        )
    source = dict(
        deps.inspect_source(
            repository_root,
            expected_source_commit,
            source_publication_receipt,
            source_publication_receipt_sha256,
        )
    )
    _validate_source_evidence(
        source,
        repository_root=repository_root,
        expected_commit=expected_source_commit,
        publication_receipt=source_publication_receipt,
        expected_publication_receipt_sha256=source_publication_receipt_sha256,
    )
    configs = {
        binding.variant_id: load_experiment_config(binding.config_path) for binding in bindings
    }
    protocol_ids = {config.data.protocol_id for config in configs.values()}
    if len(protocol_ids) != 1:
        raise PreflightError("P0 matrix rows do not share one protocol")
    protocol_id = next(iter(protocol_ids))
    data_identity = deps.load_data_identity(data_root, protocol_id)
    _validate_data_identity_evidence(data_identity, protocol_id=protocol_id)
    genept_receipt, genept_receipt_identity, genept_error = _load_genept_receipt(
        genept_preflight_receipt,
        genept_preflight_receipt_sha256,
    )

    graph_cache: dict[
        str,
        tuple[Any, Any, Path, str, dict[str, object]] | BaseException,
    ] = {}
    row_payloads: list[dict[str, object]] = []
    manifests_by_hvg: dict[int, Any] = {}
    manifest_hashes_by_hvg: dict[int, str] = {}
    for binding in bindings:
        config = configs[binding.variant_id]
        architecture = NativeArchitectureOptions.from_parameters(config.model.parameters)
        graph_root_label = _parameter(config, "runtime_graph_root")
        graph_root = _safe_runtime_graph_root(data_root, graph_root_label)
        graph_key = str(graph_root)
        if graph_key not in graph_cache:
            manifest_path = graph_root / "manifest.json"
            if not manifest_path.is_file():
                graph_cache[graph_key] = FileNotFoundError(
                    f"runtime graph manifest is missing: {manifest_path}"
                )
            else:
                try:
                    artifacts_before = _capture_graph_artifacts(
                        graph_root,
                        architecture.graph_hvg_count,
                        graph_axis_policy=architecture.graph_axis_policy,
                    )
                    topology, manifest = deps.load_graph(graph_root)
                    artifacts_after = _capture_graph_artifacts(
                        graph_root,
                        architecture.graph_hvg_count,
                        graph_axis_policy=architecture.graph_axis_policy,
                    )
                    if artifacts_after != artifacts_before:
                        raise PreflightError(
                            "P0 runtime graph artifacts changed while they were loaded"
                        )
                    for source_name in ("go", "string"):
                        if (
                            _artifact_sha256(artifacts_after, source_name)
                            != (manifest.source_artifact_sha256[source_name])
                        ):
                            raise PreflightError(
                                f"P0 runtime graph artifact SHA-256 differs: {source_name}"
                            )
                    manifest_identity = artifacts_after["manifest"]
                    if not isinstance(manifest_identity, Mapping):
                        raise PreflightError("P0 runtime graph manifest identity is malformed")
                    resolved_manifest_path = Path(str(manifest_identity["path"]))
                    manifest_file_sha = _artifact_sha256(artifacts_after, "manifest")
                    graph_cache[graph_key] = (
                        topology,
                        manifest,
                        resolved_manifest_path,
                        manifest_file_sha,
                        artifacts_after,
                    )
                except BaseException as error:
                    graph_cache[graph_key] = error
        cached = graph_cache[graph_key]
        row: dict[str, object] = {
            "matrix_row_index": binding.matrix_row_index,
            "variant_id": binding.variant_id,
            "binding": binding.payload(),
            "runtime_graph_root": graph_root_label,
            "runtime_graph_root_path": str(graph_root),
            "architecture": asdict(architecture),
            "data_binding": data_identity.payload(),
            "scientific_completion": False,
            "model_parameter_count": {
                "status": "not_computed_model_construction_forbidden",
                "value": None,
            },
        }
        if isinstance(cached, BaseException):
            row.update(
                {
                    "status": (
                        "blocked_missing_graph"
                        if isinstance(cached, FileNotFoundError)
                        else "blocked_invalid_graph"
                    ),
                    "reasons": [str(cached)],
                    "graph": None,
                    "local_view_contract": None,
                    "anchor_capacity": {
                        "status": "blocked_graph_unavailable",
                        "checked": False,
                    },
                    "genept": (
                        {"status": "blocked_graph_unavailable"}
                        if binding.genept_preflight_required
                        else {"status": "not_required"}
                    ),
                }
            )
            row_payloads.append(row)
            continue
        topology, manifest, manifest_path, manifest_file_sha, graph_artifacts = cached
        reasons: list[str] = []
        if (
            manifest.dataset_id != data_identity.dataset_id
            or manifest.protocol_id != data_identity.protocol_id
            or manifest.canonical_data_sha256 != data_identity.canonical_data_sha256
            or manifest.split_content_sha256 != data_identity.split_content_sha256
            or manifest.source_h5ad_sha256 != data_identity.source_h5ad_sha256
        ):
            reasons.append("runtime graph lineage differs from live canonical/source/split data")
        requested_count = (
            manifest.requested_gene_count
            if architecture.graph_axis_policy == "txpert_candidate_gene_universe"
            else manifest.requested_hvg_count
        )
        if requested_count != architecture.graph_hvg_count:
            reasons.append("runtime graph requested gene count differs from config")
        if architecture.graph_axis_policy == "txpert_candidate_gene_universe" and (
            manifest.candidate_gene_set_sha256 != architecture.graph_axis_source_sha256
        ):
            reasons.append("runtime graph candidate-gene source SHA-256 differs from config")
        if manifest.gene_feature_policy != "learned_id":
            reasons.append("runtime graph is not the learned-ID parent axis")
        if tuple(topology.gene_ids) != tuple(manifest.graph_gene_ids):
            reasons.append("runtime topology gene order differs from manifest")
        local_contract = _resolve_local_contract(architecture, manifest.graph_gene_count)
        genept_payload: dict[str, object] = {"status": "not_required"}
        if binding.genept_preflight_required:
            assert isinstance(graph_root_label, str)
            genept_payload, genept_reason = _genept_row_evidence(
                config=config,
                graph_root_label=graph_root_label,
                graph_manifest_path=manifest_path,
                graph_manifest=manifest,
                receipt=genept_receipt,
                receipt_identity=genept_receipt_identity,
                receipt_error=genept_error,
                verify_artifact=deps.verify_artifact,
            )
            if genept_reason is not None:
                reasons.append(genept_reason)
        row.update(
            {
                "status": "passed" if not reasons else "blocked_identity_mismatch",
                "reasons": reasons,
                "graph": {
                    "root_path": str(graph_root),
                    "manifest_path": str(manifest_path),
                    "manifest_file_sha256": manifest_file_sha,
                    "artifacts": graph_artifacts,
                    "dataset_id": manifest.dataset_id,
                    "protocol_id": manifest.protocol_id,
                    "canonical_data_sha256": manifest.canonical_data_sha256,
                    "split_content_sha256": manifest.split_content_sha256,
                    "source_h5ad_sha256": manifest.source_h5ad_sha256,
                    "source_registry_sha256": manifest.source_registry_sha256,
                    "requested_graph_gene_count": requested_count,
                    "graph_axis_policy": architecture.graph_axis_policy,
                    "graph_axis_source_sha256": architecture.graph_axis_source_sha256,
                    "graph_node_count": manifest.graph_gene_count,
                    "graph_gene_order_sha256": manifest.graph_gene_order_sha256,
                    "topology_content_sha256": manifest.topology_content_sha256,
                    "candidate_target_order_sha256": (manifest.candidate_target_order_sha256),
                    "gene_feature_policy": manifest.gene_feature_policy,
                    "source_artifact_sha256": dict(manifest.source_artifact_sha256),
                    "source_pruned_nonself_edge_count": dict(
                        manifest.source_pruned_nonself_edge_count
                    ),
                    "active_sources": list(architecture.graph_sources),
                },
                "local_view_contract": local_contract.payload(),
                "anchor_capacity": {
                    "status": "deferred_batch_gate",
                    "checked": False,
                    "reason": (
                        "anchors_by_condition belongs to training-only data; P0 does not "
                        "construct CanonicalTrainingData"
                    ),
                    "effective_node_budget": local_contract.effective_node_budget,
                },
                "genept": genept_payload,
            }
        )
        row_payloads.append(row)
        if (
            not reasons
            and architecture.graph_axis_policy == "recomputed_hvg_union_candidate_targets"
        ):
            manifests_by_hvg[architecture.graph_hvg_count] = manifest
            manifest_hashes_by_hvg[architecture.graph_hvg_count] = manifest_file_sha

    try:
        cross_h = _cross_h_audit(manifests_by_hvg, manifest_hashes_by_hvg)
    except BaseException as error:
        cross_h = {"status": "blocked_invalid_lineage", "reason": str(error)}
    passed_count = sum(row["status"] == "passed" for row in row_payloads)
    all_passed = passed_count == census.MATRIX_ROW_COUNT and cross_h.get("status") == "passed"
    runtime_after = _require_no_torch_imports("completion")
    return {
        "schema_version": "nadig-vnext-performance-p0-preflight-v1",
        "status": "passed" if all_passed else "blocked",
        "evidence_class": "performance_preflight_only",
        "scientific_completion": False,
        "matrix_path": str(Path(matrix_path).resolve(strict=True)),
        "matrix_sha256": expected_matrix_sha256,
        "matrix_row_count": len(bindings),
        "source": source,
        "data": data_identity.payload(),
        "cross_h_audit": cross_h,
        "row_status_counts": {
            "passed": passed_count,
            "blocked": len(row_payloads) - passed_count,
        },
        "rows": row_payloads,
        "runtime_import_guard": _runtime_import_guard(runtime_before, runtime_after),
        "forbidden_runtime": {
            "cuda_initialized": False,
            "model_constructed": False,
            "canonical_training_data_constructed": False,
            "canonical_validation_data_constructed": False,
            "canonical_test_data_constructed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = _claim_json_output(
            args.receipt,
            {
                "schema_version": "nadig-vnext-performance-p0-preflight-claim-v1",
                "status": "claimed",
                "evidence_class": "performance_preflight_only",
                "scientific_completion": False,
            },
        )
    except FileExistsError:
        output = {
            "receipt_path": str(args.receipt.resolve(strict=False)),
            "status": "failed_existing_receipt",
            "error": "P0 receipt destination must be new",
        }
        print(
            json.dumps(output, sort_keys=True)
            if args.as_json
            else f"{output['status']}: {args.receipt}"
        )
        return 1
    try:
        receipt = build_preflight_receipt(
            matrix_path=args.matrix,
            expected_matrix_sha256=args.expected_matrix_sha256,
            repository_root=args.repository_root,
            expected_source_commit=args.expected_source_commit,
            source_publication_receipt=args.source_publication_receipt,
            source_publication_receipt_sha256=args.source_publication_receipt_sha256,
            data_root=args.data_root,
            genept_preflight_receipt=args.genept_preflight_receipt,
            genept_preflight_receipt_sha256=args.genept_preflight_receipt_sha256,
        )
    except BaseException as error:
        runtime_snapshot = _runtime_import_snapshot()
        receipt = {
            "schema_version": "nadig-vnext-performance-p0-preflight-v1",
            "status": "failed",
            "evidence_class": "performance_preflight_only",
            "scientific_completion": False,
            "primary_failure": _failure_payload(error),
            "runtime_import_guard": {
                "status": "failed_before_completion",
                "measurement_method": "sys.modules_snapshot_without_importing_torch",
                "observed": runtime_snapshot,
            },
            "forbidden_runtime": {
                "cuda_initialized": False,
                "model_constructed": False,
                "canonical_training_data_constructed": False,
                "canonical_validation_data_constructed": False,
                "canonical_test_data_constructed": False,
            },
        }
    _atomic_json(destination, receipt)
    output = {
        "receipt_path": str(destination),
        "receipt_sha256": sha256_file(destination),
        "status": receipt["status"],
    }
    print(
        json.dumps(output, sort_keys=True)
        if args.as_json
        else f"{output['status']}: {args.receipt}"
    )
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
