from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.common import official_module_session, verify_official_checkout


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "official"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    package = root / "fakeofficial"
    package.mkdir()
    (package / "__init__.py").write_text(
        "OFFICIAL_SYMBOL = 7\n"
        "def lazy_symbol():\n"
        "    from fakeofficial.lazy import LAZY_SYMBOL\n"
        "    return LAZY_SYMBOL\n",
        encoding="utf-8",
    )
    (package / "lazy.py").write_text("LAZY_SYMBOL = 11\n", encoding="utf-8")
    _git(root, "add", "fakeofficial/__init__.py", "fakeofficial/lazy.py")
    _git(root, "commit", "-qm", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def test_official_import_is_commit_clean_and_origin_bound(tmp_path: Path) -> None:
    root, commit = _checkout(tmp_path)

    with official_module_session(
        checkout_root=root,
        expected_commit=commit,
        module_names=("fakeofficial",),
    ) as (modules, receipt):
        assert modules["fakeofficial"].OFFICIAL_SYMBOL == 7
        assert modules["fakeofficial"].lazy_symbol() == 11
        assert "fakeofficial.lazy" in sys.modules

    assert receipt.observed_commit == commit
    assert receipt.worktree_clean
    assert Path(receipt.module_origins["fakeofficial"]).is_relative_to(root)
    assert "fakeofficial" not in sys.modules
    assert "fakeofficial.lazy" not in sys.modules


def test_official_checkout_rejects_commit_mismatch_and_dirty_state(tmp_path: Path) -> None:
    root, commit = _checkout(tmp_path)
    with pytest.raises(ValueError, match="commit mismatch"):
        verify_official_checkout(root, "0" * 40)

    (root / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="must be clean"):
        verify_official_checkout(root, commit)
