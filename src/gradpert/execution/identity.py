"""Source/publication and runtime identities for development and formal runs."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

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


def inspect_source_identity(
    repository_root: str | Path,
    *,
    formal: bool,
    expected_repository: str,
    development_commit: str | None = None,
    remote_ref: str = "refs/heads/main",
) -> SourceIdentity:
    """Inspect a development snapshot or prove clean local/server/GitHub parity."""

    root = Path(repository_root).resolve(strict=True)
    tree_sha256 = _source_tree_sha256(root)
    git_dir = root / ".git"
    if not formal:
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
    published = _git(root, "ls-remote", "--exit-code", "origin", remote_ref)
    fields = published.split()
    if len(fields) != 2 or fields[1] != remote_ref or fields[0] != commit:
        raise ValueError("formal source commit is not the published remote ref")
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
