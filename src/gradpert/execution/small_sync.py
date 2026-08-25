"""Fail-closed staging and verification for server-only small result files."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json

ALLOWED_SUFFIXES = frozenset({".txt", ".json", ".jsonl", ".csv", ".md", ".yaml", ".yml"})
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 128 * 1024 * 1024
MANIFEST_NAME = "small-sync-manifest.json"
SelectionScope = Literal["run-small-results", "explicit-root"]


@dataclass(frozen=True)
class SmallResultFile:
    relative_path: str
    size_bytes: int
    sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def discover_small_result_files(
    source_root: str | Path,
    *,
    selection_scope: SelectionScope = "run-small-results",
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> tuple[SmallResultFile, ...]:
    """List allowlisted regular files from a named or explicit small-file root."""

    root = Path(source_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("small-result source root must be a real directory")
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("small-result size limits must be positive")
    if selection_scope not in {"run-small-results", "explicit-root"}:
        raise ValueError(f"unsupported small-result selection scope: {selection_scope}")
    discovered: list[SmallResultFile] = []
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"small-result tree contains a symlink: {relative}")
        if not path.is_file():
            continue
        if selection_scope == "run-small-results" and "small_results" not in relative.parts:
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"small-result extension is forbidden: {relative}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(f"small-result file exceeds size limit: {relative}")
        total += size
        if total > max_total_bytes:
            raise ValueError("small-result selection exceeds total size limit")
        discovered.append(
            SmallResultFile(
                relative_path=relative.as_posix(),
                size_bytes=size,
                sha256=sha256_file(path),
            )
        )
    if not discovered:
        raise ValueError("no allowlisted small result files were found")
    return tuple(discovered)


def small_sync_plan(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    selection_scope: SelectionScope = "run-small-results",
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    files = discover_small_result_files(
        source_root,
        selection_scope=selection_scope,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    source = Path(source_root).resolve(strict=True)
    destination = Path(destination_root).resolve()
    file_payloads = [item.payload() for item in files]
    return {
        "schema_version": "small-sync-manifest-v1",
        "source_root": str(source),
        "destination_root": str(destination),
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        "selection_scope": selection_scope,
        "max_file_bytes": max_file_bytes,
        "max_total_bytes": max_total_bytes,
        "file_count": len(files),
        "total_bytes": sum(item.size_bytes for item in files),
        "files_sha256": sha256_json(file_payloads),
        "files": file_payloads,
    }


def stage_small_results(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    selection_scope: SelectionScope = "run-small-results",
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    """Copy an immutable allowlisted snapshot into a new empty staging root."""

    plan = small_sync_plan(
        source_root,
        destination_root,
        selection_scope=selection_scope,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    source = Path(plan["source_root"])
    destination = Path(plan["destination_root"])
    if destination == source or source in destination.parents:
        raise ValueError("small-result staging root must be outside the source tree")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("small-result staging root must be new and empty")
    destination.mkdir(parents=True, exist_ok=True)
    for item in plan["files"]:
        relative = Path(item["relative_path"])
        source_file = source / relative
        if not source_file.is_file() or source_file.is_symlink():
            raise ValueError(f"small-result source changed after planning: {relative}")
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file, follow_symlinks=False)
        if destination_file.is_symlink() or sha256_file(destination_file) != item["sha256"]:
            raise RuntimeError(f"staged file hash changed during copy: {relative}")
    atomic_json(destination / MANIFEST_NAME, plan)
    return plan


def verify_staged_small_results(destination_root: str | Path) -> dict[str, Any]:
    """Verify the transferred staging tree contains exactly the sealed snapshot."""

    destination = Path(destination_root).resolve(strict=True)
    manifest_path = destination / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("small-sync manifest is missing or unsafe")
    import json

    plan = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if plan.get("schema_version") != "small-sync-manifest-v1":
        raise ValueError("unsupported small-sync manifest schema")
    if plan.get("selection_scope") not in {"run-small-results", "explicit-root"}:
        raise ValueError("unsupported small-sync selection scope")
    file_payloads = plan.get("files")
    if not isinstance(file_payloads, list) or sha256_json(file_payloads) != plan.get(
        "files_sha256"
    ):
        raise ValueError("small-sync file list hash mismatch")
    expected_paths = {Path(item["relative_path"]) for item in file_payloads}
    observed_paths: set[Path] = set()
    for path in sorted(destination.rglob("*")):
        relative = path.relative_to(destination)
        if path.is_symlink():
            raise ValueError(f"staged small-result tree contains a symlink: {relative}")
        if not path.is_file() or relative == Path(MANIFEST_NAME):
            continue
        observed_paths.add(relative)
    if observed_paths != expected_paths:
        raise ValueError("staged small-result file set differs from manifest")
    total = 0
    for item in file_payloads:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("small-sync manifest contains an unsafe path")
        if relative.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"staged small-result extension is forbidden: {relative}")
        path = destination / relative
        size = path.stat().st_size
        if size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"staged small-result content mismatch: {relative}")
        total += size
    if total != plan.get("total_bytes") or len(file_payloads) != plan.get("file_count"):
        raise ValueError("staged small-result totals differ from manifest")
    return plan
