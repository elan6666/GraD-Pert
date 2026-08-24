"""Fail-closed source acquisition and safe archive handling."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

from gradpert.data.schema import DatasetRegistryEntry

SourceState = Literal["missing", "partial", "ready", "corrupt"]


@dataclass(frozen=True)
class SourceFileStatus:
    state: SourceState
    path: Path
    observed_size_bytes: int
    expected_size_bytes: int
    observed_checksum: str | None
    expected_checksum: str


def _digest_stream(stream: BinaryIO, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def checksum_file(path: str | Path, algorithm: Literal["md5", "sha256"]) -> str:
    with Path(path).open("rb") as stream:
        return _digest_stream(stream, algorithm)


def inspect_source_file(
    entry: DatasetRegistryEntry,
    source_root: str | Path,
) -> SourceFileStatus:
    """Inspect size first and hash only a size-complete source file."""

    path = Path(source_root) / entry.source.filename
    expected_size = entry.source.size_bytes
    expected_checksum = entry.source.checksum.value
    if not path.exists():
        return SourceFileStatus(
            state="missing",
            path=path,
            observed_size_bytes=0,
            expected_size_bytes=expected_size,
            observed_checksum=None,
            expected_checksum=expected_checksum,
        )
    if not path.is_file():
        raise ValueError(f"source path is not a regular file: {path}")
    size = path.stat().st_size
    if size < expected_size:
        return SourceFileStatus(
            state="partial",
            path=path,
            observed_size_bytes=size,
            expected_size_bytes=expected_size,
            observed_checksum=None,
            expected_checksum=expected_checksum,
        )
    if size > expected_size:
        return SourceFileStatus(
            state="corrupt",
            path=path,
            observed_size_bytes=size,
            expected_size_bytes=expected_size,
            observed_checksum=None,
            expected_checksum=expected_checksum,
        )
    observed_checksum = checksum_file(path, entry.source.checksum.algorithm)
    return SourceFileStatus(
        state="ready" if observed_checksum == expected_checksum else "corrupt",
        path=path,
        observed_size_bytes=size,
        expected_size_bytes=expected_size,
        observed_checksum=observed_checksum,
        expected_checksum=expected_checksum,
    )


def require_downloadable(entry: DatasetRegistryEntry) -> None:
    if entry.source.availability != "ready_for_download":
        reason = entry.source.blocked_reason or "unknown reason"
        raise RuntimeError(f"{entry.dataset_id} source is blocked: {reason}")


def _copy_response(
    response: BinaryIO,
    destination: Path,
    mode: Literal["ab", "wb"],
    limit: int,
) -> None:
    written = destination.stat().st_size if mode == "ab" and destination.exists() else 0
    with destination.open(mode) as stream:
        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
            written += len(chunk)
            if written > limit:
                raise ValueError("download exceeded frozen source size")
            stream.write(chunk)


def download_source(
    entry: DatasetRegistryEntry,
    source_root: str | Path,
    *,
    timeout_seconds: int = 120,
) -> SourceFileStatus:
    """Download to a partial path, resume when supported, then atomically seal."""

    require_downloadable(entry)
    root = Path(source_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / entry.source.filename
    current = inspect_source_file(entry, root)
    if current.state == "ready":
        return current
    if current.state == "corrupt":
        raise ValueError(f"refusing to overwrite corrupt frozen source: {destination}")

    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry.source.size_bytes:
        raise ValueError(f"partial file exceeds frozen source size: {partial}")

    headers = {"User-Agent": "GraD-Pert-source-acquisition/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(entry.source.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", 200)
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"resume response has unexpected Content-Range: {content_range}")
            _copy_response(response, partial, "ab", entry.source.size_bytes)
        elif status == 200:
            restart = partial.with_name(f"{partial.name}.restart")
            _copy_response(response, restart, "wb", entry.source.size_bytes)
            os.replace(restart, partial)
        else:
            raise RuntimeError(f"unexpected download HTTP status: {status}")

    if partial.stat().st_size != entry.source.size_bytes:
        raise RuntimeError(
            f"download remains partial: {partial.stat().st_size}/{entry.source.size_bytes} bytes"
        )
    observed_checksum = checksum_file(partial, entry.source.checksum.algorithm)
    if observed_checksum != entry.source.checksum.value:
        raise ValueError(
            f"source checksum mismatch: expected {entry.source.checksum.value}, "
            f"observed {observed_checksum}"
        )
    os.replace(partial, destination)
    return inspect_source_file(entry, root)


def _validate_zip_member(member: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(member.filename)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe zip member path: {member.filename}")
    mode = member.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"unsupported zip member type: {member.filename}")
    return path


def safe_extract_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    max_members: int = 100_000,
    max_uncompressed_bytes: int = 100 * 1024**3,
) -> list[Path]:
    """Extract only audited regular files/directories below one destination."""

    archive_path = Path(archive)
    destination_path = Path(destination)
    with zipfile.ZipFile(archive_path) as package:
        members = package.infolist()
        if len(members) > max_members:
            raise ValueError("zip member count exceeds safety limit")
        if sum(member.file_size for member in members) > max_uncompressed_bytes:
            raise ValueError("zip uncompressed size exceeds safety limit")
        validated = [(member, _validate_zip_member(member)) for member in members]
        extracted: list[Path] = []
        destination_path.mkdir(parents=True, exist_ok=True)
        resolved_root = destination_path.resolve()
        for member, relative in validated:
            target = destination_path.joinpath(*relative.parts)
            resolved_target = target.resolve()
            if not resolved_target.is_relative_to(resolved_root):
                raise ValueError(f"zip member escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            extracted.append(target)
    return extracted
