"""Source/publication and runtime identities for development and formal runs."""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json

_TREE_ROOTS = ("src", "benchmarks", "configs", "registry")
_TREE_FILES = ("pyproject.toml", "uv.lock", "AGENTS.md")
_GIT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class SourceIdentity:
    repository_root: str
    commit: str
    dirty: bool
    tree_sha256: str
    formal_eligible: bool
    formal_eligibility_reason: str | None
    remote_url: str | None
    remote_ref: str | None
    published_commit: str | None
    publication_receipt_sha256: str | None

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourcePublicationReceipt:
    schema_version: str
    repository: str
    remote_url: str
    remote_ref: str
    published_commit: str
    tree_sha256: str
    verification_method: str
    verified_unix: float

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentIdentity:
    python_version: str
    platform: str
    lock_file: str
    lock_file_sha256: str
    torch_version: str
    cuda_runtime: str | None
    cudnn_version: int | None
    device_name: str
    payload_sha256: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _git(root: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        operation = arguments[0] if arguments else "unknown"
        raise RuntimeError(
            f"Git {operation} timed out after {_GIT_TIMEOUT_SECONDS:g} seconds"
        ) from error
    return result.stdout.strip()


def _source_tree_sha256(root: Path) -> str:
    files: list[Path] = []
    for relative_root in _TREE_ROOTS:
        candidate = root / relative_root
        if candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and "__pycache__" not in path.parts
                and not path.name.endswith((".pyc", ".pyo"))
            )
    files.extend(
        candidate
        for relative in _TREE_FILES
        if (candidate := root / relative).is_file() and not candidate.is_symlink()
    )
    payload = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for path in sorted(set(files))
    ]
    if not payload:
        raise ValueError("source tree contains no identity files")
    return sha256_json(payload)


def _normalized_repository(value: str) -> str:
    normalized = value.strip().removesuffix(".git").rstrip("/")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a 64-character SHA-256") from error


def _load_publication_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_repository: str,
    remote_url: str,
    remote_ref: str,
    commit: str,
    tree_sha256: str,
) -> SourcePublicationReceipt:
    _require_sha256(expected_sha256, label="publication receipt hash")
    receipt_path = Path(path).resolve(strict=True)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("publication receipt must be a regular file")
    if sha256_file(receipt_path) != expected_sha256:
        raise ValueError("publication receipt hash differs from the frozen contract")
    payload: Any = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "repository",
        "remote_url",
        "remote_ref",
        "published_commit",
        "tree_sha256",
        "verification_method",
        "verified_unix",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("publication receipt schema is malformed")
    try:
        receipt = SourcePublicationReceipt(**payload)
    except TypeError as error:
        raise ValueError("publication receipt schema is malformed") from error
    string_fields = (
        receipt.schema_version,
        receipt.repository,
        receipt.remote_url,
        receipt.remote_ref,
        receipt.published_commit,
        receipt.tree_sha256,
        receipt.verification_method,
    )
    if (
        not all(isinstance(value, str) for value in string_fields)
        or receipt.schema_version != "source-publication-receipt-v1"
        or receipt.verification_method != "git_ls_remote_live"
        or not isinstance(receipt.verified_unix, (int, float))
        or isinstance(receipt.verified_unix, bool)
        or not math.isfinite(receipt.verified_unix)
        or receipt.verified_unix <= 0
    ):
        raise ValueError("publication receipt identity is malformed")
    if (
        _normalized_repository(receipt.repository) != _normalized_repository(expected_repository)
        or _normalized_repository(receipt.remote_url) != _normalized_repository(remote_url)
        or receipt.remote_ref != remote_ref
        or receipt.published_commit != commit
        or receipt.tree_sha256 != tree_sha256
    ):
        raise ValueError("publication receipt differs from the current formal source")
    return receipt


def create_source_publication_receipt(
    repository_root: str | Path,
    *,
    expected_repository: str,
    output_path: str | Path,
    remote_ref: str = "refs/heads/main",
) -> SourcePublicationReceipt:
    """Seal one live GitHub publication check for a hash-pinned formal queue."""

    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError("publication receipt destination must be new")
    identity = inspect_source_identity(
        repository_root,
        formal=True,
        expected_repository=expected_repository,
        remote_ref=remote_ref,
    )
    if identity.remote_url is None or identity.published_commit is None:
        raise AssertionError("formal publication inspection returned an incomplete identity")
    receipt = SourcePublicationReceipt(
        schema_version="source-publication-receipt-v1",
        repository=expected_repository,
        remote_url=identity.remote_url,
        remote_ref=remote_ref,
        published_commit=identity.published_commit,
        tree_sha256=identity.tree_sha256,
        verification_method="git_ls_remote_live",
        verified_unix=time.time(),
    )
    atomic_json(destination, receipt.payload())
    return receipt


def inspect_source_identity(
    repository_root: str | Path,
    *,
    formal: bool,
    expected_repository: str,
    development_commit: str | None = None,
    remote_ref: str = "refs/heads/main",
    publication_receipt: str | Path | None = None,
    expected_publication_receipt_sha256: str | None = None,
) -> SourceIdentity:
    """Inspect a development snapshot or prove clean local/server/GitHub parity."""

    root = Path(repository_root).resolve(strict=True)
    tree_sha256 = _source_tree_sha256(root)
    git_dir = root / ".git"
    if not formal:
        if publication_receipt is not None or expected_publication_receipt_sha256 is not None:
            raise ValueError("publication receipts are only valid for formal execution")
        commit = development_commit
        dirty = True
        remote_url: str | None = None
        if git_dir.exists():
            commit = _git(root, "rev-parse", "HEAD")
            if development_commit is not None and development_commit != commit:
                raise ValueError("declared development commit differs from Git HEAD")
            dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=normal"))
            remote_url = _git(root, "remote", "get-url", "origin")
        if commit is None or len(commit) != 40:
            raise ValueError("development snapshot requires a declared 40-character commit")
        return SourceIdentity(
            repository_root=str(root),
            commit=commit,
            dirty=dirty,
            tree_sha256=tree_sha256,
            formal_eligible=False,
            formal_eligibility_reason=(
                "development_snapshot_not_verified_against_published_remote"
            ),
            remote_url=remote_url,
            remote_ref=None,
            published_commit=None,
            publication_receipt_sha256=None,
        )

    if not git_dir.exists():
        raise ValueError("formal execution requires a Git worktree")
    commit = _git(root, "rev-parse", "HEAD")
    dirty_output = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if dirty_output:
        raise ValueError("formal execution requires a clean Git worktree")
    remote_url = _git(root, "remote", "get-url", "origin")
    if _normalized_repository(remote_url) != _normalized_repository(expected_repository):
        raise ValueError("formal source remote does not match experiment repository")
    if (publication_receipt is None) != (expected_publication_receipt_sha256 is None):
        raise ValueError("publication receipt path and hash must be provided together")
    receipt_sha256: str | None = None
    if publication_receipt is None:
        published = _git(root, "ls-remote", "--exit-code", "origin", remote_ref)
        fields = published.split()
        if len(fields) != 2 or fields[1] != remote_ref or fields[0] != commit:
            raise ValueError("formal source commit is not the published remote ref")
    else:
        assert expected_publication_receipt_sha256 is not None
        _load_publication_receipt(
            publication_receipt,
            expected_sha256=expected_publication_receipt_sha256,
            expected_repository=expected_repository,
            remote_url=remote_url,
            remote_ref=remote_ref,
            commit=commit,
            tree_sha256=tree_sha256,
        )
        receipt_sha256 = expected_publication_receipt_sha256
    return SourceIdentity(
        repository_root=str(root),
        commit=commit,
        dirty=False,
        tree_sha256=tree_sha256,
        formal_eligible=True,
        formal_eligibility_reason=None,
        remote_url=remote_url,
        remote_ref=remote_ref,
        published_commit=commit,
        publication_receipt_sha256=receipt_sha256,
    )


def inspect_environment(
    repository_root: str | Path,
    *,
    device_name: str,
    lock_file: str | Path | None = None,
    require_cuda: bool = True,
) -> EnvironmentIdentity:
    """Hash the resolved lock plus the exact server runtime used for a run."""

    root = Path(repository_root).resolve(strict=True)
    lock_path = root / "uv.lock" if lock_file is None else Path(lock_file).resolve(strict=True)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ValueError("runtime identity requires a regular uv.lock")
    torch = importlib.import_module("torch")
    device = torch.device(device_name)
    if require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("training and benchmark execution require an available CUDA device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested CUDA device is unavailable")
        torch.cuda.set_device(device)
    python_version = platform.python_version()
    platform_name = platform.platform()
    lock_file_name = str(lock_path)
    lock_file_sha256 = sha256_file(lock_path)
    torch_version = str(torch.__version__)
    cuda_runtime = (
        None if device.type != "cuda" or torch.version.cuda is None else str(torch.version.cuda)
    )
    cudnn_version = None if device.type != "cuda" else torch.backends.cudnn.version()
    observed_device_name = (
        platform.processor() or platform.machine()
        if device.type != "cuda"
        else str(torch.cuda.get_device_name(device))
    )
    core = {
        "python_version": python_version,
        "platform": platform_name,
        "lock_file": lock_file_name,
        "lock_file_sha256": lock_file_sha256,
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "cudnn_version": cudnn_version,
        "device_name": observed_device_name,
    }
    return EnvironmentIdentity(
        python_version=python_version,
        platform=platform_name,
        lock_file=lock_file_name,
        lock_file_sha256=lock_file_sha256,
        torch_version=torch_version,
        cuda_runtime=cuda_runtime,
        cudnn_version=cudnn_version,
        device_name=observed_device_name,
        payload_sha256=sha256_json(core),
    )
