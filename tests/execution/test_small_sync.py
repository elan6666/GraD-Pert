from __future__ import annotations

import json
from pathlib import Path

import pytest

from gradpert.execution.small_sync import (
    MANIFEST_NAME,
    discover_small_result_files,
    small_sync_plan,
    stage_small_results,
    verify_staged_small_results,
)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "runs"
    small = source / "smoke" / "gradpert_b2" / "norman" / "seed-1" / "small_results"
    small.mkdir(parents=True)
    (small / "metrics.json").write_text('{"score": 1}\n', encoding="utf-8")
    (small / "metrics.csv").write_text("metric,value\nx,1\n", encoding="utf-8")
    (small / "config.resolved.yaml").write_text("model_id: gradpert_b2\n", encoding="utf-8")
    (source / "artifacts").mkdir()
    (source / "artifacts" / "prediction.pkl").write_bytes(b"server-only")
    return source


def test_small_sync_dry_run_does_not_create_destination(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "stage"
    plan = small_sync_plan(source, destination)
    assert plan["file_count"] == 3
    assert {item["relative_path"].split("/")[-1] for item in plan["files"]} == {
        "metrics.json",
        "metrics.csv",
        "config.resolved.yaml",
    }
    assert not destination.exists()


def test_stage_and_verify_exact_allowlisted_snapshot(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "stage"
    plan = stage_small_results(source, destination)
    assert (destination / MANIFEST_NAME).is_file()
    assert verify_staged_small_results(destination)["files_sha256"] == plan["files_sha256"]
    assert not any(path.suffix == ".pkl" for path in destination.rglob("*"))

    copied = destination / plan["files"][0]["relative_path"]
    copied.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content mismatch"):
        verify_staged_small_results(destination)


def test_forbidden_file_inside_small_results_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    forbidden = next(source.rglob("small_results")) / "prediction.pkl"
    forbidden.write_bytes(b"large artifact")
    with pytest.raises(ValueError, match="extension is forbidden"):
        discover_small_result_files(source)


def test_symlink_and_extra_transferred_file_fail_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    small = next(source.rglob("small_results"))
    link = small / "linked.json"
    try:
        link.symlink_to(small / "metrics.json")
    except OSError:
        pytest.skip("filesystem does not support symlinks")
    with pytest.raises(ValueError, match="contains a symlink"):
        discover_small_result_files(source)
    link.unlink()

    destination = tmp_path / "stage"
    stage_small_results(source, destination)
    (destination / "extra.json").write_text(json.dumps({"extra": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        verify_staged_small_results(destination)


def test_stage_destination_must_be_outside_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="outside the source tree"):
        stage_small_results(source, source / "stage")


def test_explicit_receipt_root_is_allowlisted_without_small_results_name(tmp_path: Path) -> None:
    source = tmp_path / "registry" / "prepared"
    source.mkdir(parents=True)
    (source / "CURRENT_STATE.json").write_text("{}\n", encoding="utf-8")
    (source / "README.md").write_text("# Receipts\n", encoding="utf-8")
    files = discover_small_result_files(source, selection_scope="explicit-root")
    assert {item.relative_path for item in files} == {"CURRENT_STATE.json", "README.md"}

    (source / "canonical.h5ad").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="extension is forbidden"):
        discover_small_result_files(source, selection_scope="explicit-root")
