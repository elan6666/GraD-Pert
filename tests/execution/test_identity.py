from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import gradpert.execution.identity as identity_module
from gradpert.execution.identity import inspect_source_identity

COMMIT = "a" * 40


def _write_source_tree(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("fixture\n", encoding="utf-8")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_development_snapshot_is_truthfully_ineligible(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)

    identity = inspect_source_identity(
        tmp_path,
        formal=False,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        development_commit=COMMIT,
    )

    assert identity.commit == COMMIT
    assert identity.dirty is True
    assert identity.formal_eligible is False
    assert identity.published_commit is None


def test_formal_source_requires_clean_published_head(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    remote = tmp_path / "remote.git"
    checkout.mkdir()
    _write_source_tree(checkout)
    _git(checkout, "init", "-b", "main")
    _git(checkout, "config", "user.name", "Fixture")
    _git(checkout, "config", "user.email", "fixture@example.com")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(checkout, "remote", "add", "origin", str(remote))
    _git(checkout, "push", "-u", "origin", "main")

    identity = inspect_source_identity(
        checkout,
        formal=True,
        expected_repository=str(remote),
    )
    assert identity.formal_eligible is True
    assert identity.published_commit == _git(checkout, "rev-parse", "HEAD")

    (checkout / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git worktree"):
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=str(remote),
        )


def test_development_worktree_rejects_declared_commit_drift(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.com")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "fixture")

    with pytest.raises(ValueError, match="differs from Git HEAD"):
        inspect_source_identity(
            tmp_path,
            formal=False,
            expected_repository="https://github.com/elan6666/GraD-Pert.git",
            development_commit="0" * 40,
        )


def test_git_identity_commands_are_noninteractive_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="observed\n", stderr="")

    monkeypatch.setattr(identity_module.subprocess, "run", fake_run)

    assert identity_module._git(tmp_path, "rev-parse", "HEAD") == "observed"
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert observed["timeout"] == 30.0


def test_git_identity_timeout_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        raise subprocess.TimeoutExpired(command, timeout=30.0)

    monkeypatch.setattr(identity_module.subprocess, "run", fake_timeout)

    with pytest.raises(RuntimeError, match="Git ls-remote timed out after 30 seconds"):
        identity_module._git(tmp_path, "ls-remote", "origin")
