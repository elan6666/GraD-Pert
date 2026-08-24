from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest

from gradpert.data.acquisition import (
    download_source,
    inspect_source_file,
    require_downloadable,
    safe_extract_zip,
)
from gradpert.data.registry import load_dataset_registry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = ROOT / "registry" / "datasets"


def _tiny_entry(payload: bytes):  # type: ignore[no-untyped-def]
    entry = load_dataset_registry(REGISTRY_ROOT / "nadig_hepg2.yaml")
    source = entry.source.model_copy(
        update={
            "filename": "tiny.bin",
            "size_bytes": len(payload),
            "checksum": entry.source.checksum.model_copy(
                update={"algorithm": "md5", "value": hashlib.md5(payload).hexdigest()}
            ),
        }
    )
    return entry.model_copy(update={"source": source})


def test_source_inspection_distinguishes_missing_partial_ready_and_corrupt(
    tmp_path: Path,
) -> None:
    payload = b"frozen-source"
    entry = _tiny_entry(payload)
    assert inspect_source_file(entry, tmp_path).state == "missing"

    path = tmp_path / entry.source.filename
    path.write_bytes(payload[:-1])
    assert inspect_source_file(entry, tmp_path).state == "partial"

    path.write_bytes(payload)
    assert inspect_source_file(entry, tmp_path).state == "ready"

    path.write_bytes(b"X" * len(payload))
    status = inspect_source_file(entry, tmp_path)
    assert status.state == "corrupt"
    assert status.observed_checksum is not None


def test_blocked_source_cannot_enter_downloader() -> None:
    entry = load_dataset_registry(REGISTRY_ROOT / "replogle_rpe1_essential.yaml")
    entry = entry.model_copy(
        update={
            "source": entry.source.model_copy(
                update={"availability": "blocked_upstream", "blocked_reason": "test block"}
            )
        }
    )
    with pytest.raises(RuntimeError, match="source is blocked"):
        require_downloadable(entry)


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, status: int, headers: dict[str, str]):
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def test_downloader_seals_a_complete_response_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"frozen-source"
    entry = _tiny_entry(payload)

    def open_full(request: object, timeout: int) -> _Response:
        assert timeout == 9
        assert request is not None
        return _Response(payload, 200, {})

    monkeypatch.setattr("gradpert.data.acquisition.urllib.request.urlopen", open_full)
    status = download_source(entry, tmp_path, timeout_seconds=9)
    assert status.state == "ready"
    assert status.path.read_bytes() == payload
    assert not (tmp_path / "tiny.bin.part").exists()


def test_downloader_resumes_only_from_matching_content_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"frozen-source"
    entry = _tiny_entry(payload)
    prefix = payload[:6]
    (tmp_path / "tiny.bin.part").write_bytes(prefix)

    def open_tail(request: Any, timeout: int) -> _Response:
        assert timeout == 120
        assert request.headers["Range"] == f"bytes={len(prefix)}-"
        return _Response(payload[len(prefix) :], 206, {"Content-Range": "bytes 6-12/13"})

    monkeypatch.setattr("gradpert.data.acquisition.urllib.request.urlopen", open_tail)
    status = download_source(entry, tmp_path)
    assert status.state == "ready"
    assert status.path.read_bytes() == payload


def test_safe_zip_extracts_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("nested/data.txt", "ok")
    output = tmp_path / "out"
    extracted = safe_extract_zip(archive, output)
    assert extracted == [output / "nested" / "data.txt"]
    assert extracted[0].read_text(encoding="utf-8") == "ok"


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "a/../../escape.txt"])
def test_safe_zip_rejects_path_traversal(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(member, "no")
    with pytest.raises(ValueError, match="unsafe zip member path"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_zip_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(link, "target")
    with pytest.raises(ValueError, match="unsupported zip member type"):
        safe_extract_zip(archive, tmp_path / "out")
