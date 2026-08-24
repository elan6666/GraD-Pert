"""Shared atomic and trusted-root file primitives for sealed artifacts."""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def resolve_trusted_file(path: str | Path, trusted_root: str | Path) -> Path:
    candidate = Path(path)
    root = Path(trusted_root).resolve(strict=True)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("artifact path is outside trusted_root")
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError("artifact must be a regular non-symlink file")
    return resolved


def load_hash_pinned_pickle(
    path: str | Path,
    *,
    expected_file_sha256: str,
    trusted_root: str | Path,
) -> tuple[Path, Any]:
    resolved = resolve_trusted_file(path, trusted_root)
    observed = sha256_file(resolved)
    if observed != expected_file_sha256:
        raise ValueError("artifact file SHA-256 mismatch")
    # Pickle is accepted only after the explicit trusted-root and content-hash gates.
    with resolved.open("rb") as handle:
        return resolved, pickle.load(handle)
