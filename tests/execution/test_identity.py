from __future__ import annotations

import json
import ssl
import subprocess
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request

import pytest

import gradpert.execution.identity as identity_module
from gradpert.execution.identity import (
    create_source_publication_receipt,
    inspect_source_identity,
)
from gradpert.hashing import sha256_file

COMMIT = "a" * 40
REPOSITORY = "https://github.com/elan6666/GraD-Pert.git"
API_REPO = "https://api.github.com/repos/elan6666/GraD-Pert"


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


def test_hash_pinned_publication_receipt_replaces_only_redundant_remote_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    receipt_path = tmp_path / "publication.json"

    receipt = create_source_publication_receipt(
        checkout,
        expected_repository=str(remote),
        output_path=receipt_path,
    )
    receipt_sha256 = sha256_file(receipt_path)
    assert receipt.published_commit == _git(checkout, "rev-parse", "HEAD")

    original_run = subprocess.run

    def reject_live_remote(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "ls-remote" in command:
            raise AssertionError("a sealed queue row must not repeat the live remote check")
        return original_run(command, **kwargs)

    monkeypatch.setattr(identity_module.subprocess, "run", reject_live_remote)
    identity = inspect_source_identity(
        checkout,
        formal=True,
        expected_repository=str(remote),
        publication_receipt=receipt_path,
        expected_publication_receipt_sha256=receipt_sha256,
    )
    assert identity.formal_eligible is True
    assert identity.publication_receipt_sha256 == receipt_sha256


def test_publication_receipt_fails_closed_on_pairing_hash_and_source_drift(
    tmp_path: Path,
) -> None:
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
    receipt_path = tmp_path / "publication.json"
    create_source_publication_receipt(
        checkout,
        expected_repository=str(remote),
        output_path=receipt_path,
    )
    receipt_sha256 = sha256_file(receipt_path)

    with pytest.raises(ValueError, match="path and hash"):
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=str(remote),
            publication_receipt=receipt_path,
        )
    with pytest.raises(ValueError, match="hash differs"):
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=str(remote),
            publication_receipt=receipt_path,
            expected_publication_receipt_sha256="0" * 64,
        )

    (checkout / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "successor")
    with pytest.raises(ValueError, match="differs from the current formal source"):
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=str(remote),
            publication_receipt=receipt_path,
            expected_publication_receipt_sha256=receipt_sha256,
        )


def test_publication_receipt_destination_is_immutable(tmp_path: Path) -> None:
    destination = tmp_path / "publication.json"
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be new"):
        create_source_publication_receipt(
            tmp_path,
            expected_repository="https://github.com/elan6666/GraD-Pert.git",
            output_path=destination,
        )


@pytest.fixture
def github_checkout(tmp_path: Path) -> tuple[Path, str]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _write_source_tree(checkout)
    _git(checkout, "init", "-b", "main")
    _git(checkout, "config", "user.name", "Fixture")
    _git(checkout, "config", "user.email", "fixture@example.com")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")
    _git(checkout, "remote", "add", "origin", REPOSITORY)
    return checkout, _git(checkout, "rev-parse", "HEAD")


class RestResponse:
    def __init__(
        self,
        url: str,
        payload: object,
        *,
        status: int = 200,
        final_url: str | None = None,
        content_type: str = "application/json",
        body: bytes | None = None,
    ):
        self.status = status
        self.url = final_url or url
        self.body = json.dumps(payload).encode() if body is None else body
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def rest_payloads(commit: str) -> dict[str, dict]:
    return {
        API_REPO: dict(full_name="elan6666/GraD-Pert", private=False, visibility="public"),
        API_REPO + "/git/ref/heads/main": dict(
            ref="refs/heads/main", object=dict(type="commit", sha=commit)
        ),
    }


def mock_rest(monkeypatch, payloads, *, response_options=None, error=None):
    requests = []
    handlers = []

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        if error is not None:
            raise error
        return RestResponse(
            request.full_url, payloads[request.full_url], **(response_options or {})
        )

    def opener(*args):
        handlers.extend(args)
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(identity_module, "build_opener", opener)
    return requests, handlers


def test_explicit_rest_publication_uses_public_repo_exact_ref_and_verified_tls(
    github_checkout,
    tmp_path,
    monkeypatch,
):
    checkout, commit = github_checkout
    requests, handlers = mock_rest(monkeypatch, rest_payloads(commit))
    receipt_path = tmp_path / "rest-publication.json"
    receipt = create_source_publication_receipt(
        checkout,
        expected_repository=REPOSITORY,
        output_path=receipt_path,
        verification_method="github_rest_live",
    )
    assert receipt.verification_method == "github_rest_live" and receipt.published_commit == commit
    assert [r.full_url for r, _ in requests] == [API_REPO, API_REPO + "/git/ref/heads/main"]
    for request, timeout in requests:
        assert request.get_method() == "GET" and timeout == 30.0
        assert request.get_header("X-github-api-version") == "2026-03-10"
        assert request.get_header("Accept") == "application/vnd.github+json"
        assert not request.has_header("Authorization")
    https = next(h for h in handlers if isinstance(h, HTTPSHandler))
    assert https._context.verify_mode == ssl.CERT_REQUIRED and https._context.check_hostname
    redirect = next(
        h for h in handlers if isinstance(h, identity_module._RejectPublicationRedirects)
    )
    assert (
        redirect.redirect_request(Request(API_REPO), None, 302, "moved", {}, "http://evil.test")
        is None
    )

    monkeypatch.setattr(
        identity_module,
        "build_opener",
        lambda *_: pytest.fail("cached receipt must not call network"),
    )
    identity = inspect_source_identity(
        checkout,
        formal=True,
        expected_repository=REPOSITORY,
        publication_receipt=receipt_path,
        expected_publication_receipt_sha256=sha256_file(receipt_path),
        verification_method="github_rest_live",
    )
    assert identity.formal_eligible and identity.publication_receipt_sha256 == sha256_file(
        receipt_path
    )
    assert set(receipt.payload()) == {
        "schema_version",
        "repository",
        "remote_url",
        "remote_ref",
        "published_commit",
        "tree_sha256",
        "verification_method",
        "verified_unix",
    }
    with pytest.raises(ValueError, match="malformed"):
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=REPOSITORY,
            publication_receipt=receipt_path,
            expected_publication_receipt_sha256=sha256_file(receipt_path),
        )


def test_default_git_path_never_implicitly_uses_rest(github_checkout, monkeypatch):
    checkout, commit = github_checkout
    native_git = identity_module._git
    observed = []

    def git(root, *args):
        observed.append(args)
        return f"{commit}\trefs/heads/main" if args[0] == "ls-remote" else native_git(root, *args)

    monkeypatch.setattr(identity_module, "_git", git)
    monkeypatch.setattr(
        identity_module, "build_opener", lambda *_: pytest.fail("automatic REST fallback")
    )
    assert (
        inspect_source_identity(
            checkout, formal=True, expected_repository=REPOSITORY
        ).published_commit
        == commit
    )
    assert any(row[0] == "ls-remote" for row in observed)


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        URLError("offline"),
        HTTPError(API_REPO, 403, "rate limited", {}, None),
        HTTPError(API_REPO, 429, "rate limited", {}, None),
        HTTPError(API_REPO, 302, "redirect rejected", {}, None),
    ],
)
def test_rest_network_failures_do_not_fallback_or_create_receipt(
    github_checkout,
    tmp_path,
    monkeypatch,
    error,
):
    checkout, commit = github_checkout
    mock_rest(monkeypatch, rest_payloads(commit), error=error)
    native_git = identity_module._git

    def no_fallback(root, *args):
        assert args[0] != "ls-remote", "REST must never silently fallback to Git"
        return native_git(root, *args)

    monkeypatch.setattr(identity_module, "_git", no_fallback)
    destination = tmp_path / "never-created.json"
    with pytest.raises(RuntimeError, match="no fallback"):
        create_source_publication_receipt(
            checkout,
            expected_repository=REPOSITORY,
            output_path=destination,
            verification_method="github_rest_live",
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "options",
    [
        dict(status=201),
        dict(status=301),
        dict(content_type="text/html"),
        dict(final_url="https://evil.test/fake"),
        dict(final_url=API_REPO + "/other"),
        dict(body=b"not json"),
        dict(body=b"\xff"),
        dict(body=b"x" * (1024 * 1024 + 1)),
    ],
)
def test_rest_transport_rejects_status_redirect_content_and_unbounded_payload(monkeypatch, options):
    mock_rest(monkeypatch, {API_REPO: {}}, response_options=options)
    with pytest.raises((ValueError, RuntimeError)):
        identity_module._github_rest_json(API_REPO)


@pytest.mark.parametrize(
    "url",
    [
        "http://api.github.com/repos/o/r",
        "https://api.github.com.evil.test/repos/o/r",
        "https://api.github.com@evil.test/repos/o/r",
        "https://api.github.com:443/repos/o/r",
        API_REPO + "?token=x",
        API_REPO + "#fragment",
    ],
)
def test_rest_api_url_restrictions_precede_network(monkeypatch, url):
    monkeypatch.setattr(
        identity_module, "build_opener", lambda *_: pytest.fail("unapproved API endpoint")
    )
    with pytest.raises(ValueError, match=r"HTTPS api\.github\.com"):
        identity_module._github_rest_json(url)


@pytest.mark.parametrize(
    "payload",
    [
        dict(ref="refs/heads/main2", object=dict(type="commit", sha=COMMIT)),
        dict(ref="refs/heads/main", object=dict(type="tag", sha=COMMIT)),
        dict(ref="refs/heads/main", object=dict(type="commit", sha="a" * 39)),
        dict(ref="refs/heads/main", object=dict(type="commit", sha="b" * 40)),
        dict(ref="refs/heads/main"),
        {},
        [],
    ],
)
def test_rest_ref_payload_must_be_exact_commit(monkeypatch, payload):
    payloads = rest_payloads(COMMIT)
    payloads[API_REPO + "/git/ref/heads/main"] = payload
    mock_rest(monkeypatch, payloads)
    with pytest.raises(ValueError):
        identity_module._verify_github_rest_live(REPOSITORY, "refs/heads/main", COMMIT)


@pytest.mark.parametrize(
    "metadata",
    [
        dict(full_name="other/repo", private=False, visibility="public"),
        dict(full_name="elan6666/GraD-Pert", private=True, visibility="private"),
        dict(full_name="elan6666/GraD-Pert", private=False),
        {},
    ],
)
def test_rest_public_repository_identity_is_not_inferred_from_ref(monkeypatch, metadata):
    payloads = rest_payloads(COMMIT)
    payloads[API_REPO] = metadata
    requests, _ = mock_rest(monkeypatch, payloads)
    with pytest.raises(ValueError, match="exact public repository"):
        identity_module._verify_github_rest_live(REPOSITORY, "refs/heads/main", COMMIT)
    assert len(requests) == 1


@pytest.mark.parametrize(
    "reference",
    [
        "main",
        "refs/tags/main",
        "refs/heads/",
        "refs/heads/a..b",
        "refs/heads/.hidden",
        "refs/heads/a.lock",
        "refs/heads/a//b",
        "refs/heads/a?b",
        "refs/heads/a b",
        "refs/heads/a@{b",
    ],
)
def test_rest_requires_exact_valid_branch_ref(monkeypatch, reference):
    monkeypatch.setattr(
        identity_module, "build_opener", lambda *_: pytest.fail("invalid branch reached network")
    )
    with pytest.raises(ValueError, match="refs/heads branch"):
        identity_module._verify_github_rest_live(REPOSITORY, reference, COMMIT)


def test_rest_accepts_exact_nested_branch_and_origin_binding(github_checkout, monkeypatch):
    checkout, commit = github_checkout
    payloads = rest_payloads(commit)
    payloads[API_REPO + "/git/ref/heads/feature/topic"] = dict(
        ref="refs/heads/feature/topic", object=dict(type="commit", sha=commit)
    )
    requests, _ = mock_rest(monkeypatch, payloads)
    assert (
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=REPOSITORY,
            remote_ref="refs/heads/feature/topic",
            verification_method="github_rest_live",
        ).published_commit
        == commit
    )
    assert requests[-1][0].full_url.endswith("/git/ref/heads/feature/topic")
    _git(checkout, "remote", "set-url", "origin", "https://github.com/other/repo.git")
    with pytest.raises(ValueError, match="remote does not match"):
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=REPOSITORY,
            verification_method="github_rest_live",
        )


def test_unknown_verification_method_and_development_rest_are_rejected(tmp_path):
    _write_source_tree(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        inspect_source_identity(
            tmp_path, formal=True, expected_repository=REPOSITORY, verification_method="auto"
        )
    with pytest.raises(ValueError, match="requires formal"):
        inspect_source_identity(
            tmp_path,
            formal=False,
            expected_repository=REPOSITORY,
            development_commit=COMMIT,
            verification_method="github_rest_live",
        )


def test_rest_receipt_hash_and_current_tree_remain_strict(github_checkout, tmp_path, monkeypatch):
    checkout, commit = github_checkout
    mock_rest(monkeypatch, rest_payloads(commit))
    receipt_path = tmp_path / "receipt.json"
    create_source_publication_receipt(
        checkout,
        expected_repository=REPOSITORY,
        output_path=receipt_path,
        verification_method="github_rest_live",
    )
    original = json.loads(receipt_path.read_text())
    for key, value in (
        ("verification_method", "unknown"),
        ("tree_sha256", "0" * 64),
        ("published_commit", "b" * 40),
        ("repository", "https://github.com/other/repo"),
    ):
        receipt_path.write_text(json.dumps({**original, key: value}))
        with pytest.raises(ValueError):
            inspect_source_identity(
                checkout,
                formal=True,
                expected_repository=REPOSITORY,
                publication_receipt=receipt_path,
                expected_publication_receipt_sha256=sha256_file(receipt_path),
                verification_method="github_rest_live",
            )
    receipt_path.write_text(json.dumps(original))
    (checkout / "src/module.py").write_text("CHANGED = 2\n")
    with pytest.raises(ValueError, match="clean Git worktree"):
        inspect_source_identity(
            checkout,
            formal=True,
            expected_repository=REPOSITORY,
            publication_receipt=receipt_path,
            expected_publication_receipt_sha256=sha256_file(receipt_path),
            verification_method="github_rest_live",
        )
