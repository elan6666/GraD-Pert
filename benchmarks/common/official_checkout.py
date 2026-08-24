"""Fail-closed import and Git checks for an isolated official checkout."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

from gradpert.hashing import sha256_json


@dataclass(frozen=True)
class OfficialCheckoutReceipt:
    checkout_root: str
    expected_commit: str
    observed_commit: str
    worktree_clean: bool
    module_origins: dict[str, str]
    receipt_sha256: str

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _git(checkout_root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(checkout_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def verify_official_checkout(checkout_root: str | Path, expected_commit: str) -> Path:
    root = Path(checkout_root).resolve(strict=True)
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError("official checkout root must be a Git worktree")
    observed = _git(root, "rev-parse", "HEAD")
    if observed != expected_commit:
        raise ValueError(
            f"official checkout commit mismatch: expected {expected_commit}, observed {observed}"
        )
    dirty = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise ValueError("official checkout worktree must be clean")
    return root


@contextmanager
def _isolated_import_path(root: Path, module_roots: Sequence[str]) -> Iterator[None]:
    original_path = list(sys.path)
    displaced = {
        name: module
        for name, module in list(sys.modules.items())
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in module_roots)
    }
    for name in displaced:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in module_roots):
                del sys.modules[name]
        sys.modules.update(displaced)
        sys.path[:] = original_path


@contextmanager
def official_module_session(
    *,
    checkout_root: str | Path,
    expected_commit: str,
    module_names: Sequence[str],
) -> Iterator[tuple[dict[str, ModuleType], OfficialCheckoutReceipt]]:
    """Keep the frozen checkout import boundary active for the whole API call session."""

    if not module_names or any(not name for name in module_names):
        raise ValueError("official module_names must be non-empty")
    root = verify_official_checkout(checkout_root, expected_commit)
    module_roots = tuple(sorted({name.split(".", 1)[0] for name in module_names}))
    imported: dict[str, ModuleType] = {}
    origins: dict[str, str] = {}
    with _isolated_import_path(root, module_roots):
        for name in module_names:
            module = importlib.import_module(name)
            origin_text = getattr(module, "__file__", None)
            if not origin_text:
                raise ValueError(f"official module has no file origin: {name}")
            origin = Path(origin_text).resolve(strict=True)
            if not origin.is_relative_to(root):
                raise ValueError(f"official module resolved outside checkout: {name} -> {origin}")
            imported[name] = module
            origins[name] = str(origin)
        receipt_core = {
            "checkout_root": str(root),
            "expected_commit": expected_commit,
            "observed_commit": expected_commit,
            "worktree_clean": True,
            "module_origins": origins,
        }
        receipt = OfficialCheckoutReceipt(
            checkout_root=str(root),
            expected_commit=expected_commit,
            observed_commit=expected_commit,
            worktree_clean=True,
            module_origins=origins,
            receipt_sha256=sha256_json(receipt_core),
        )
        yield imported, receipt
