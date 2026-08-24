from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gradpert.artifacts import (
    CatalogEntrySource,
    build_catalog_entry,
    load_result_catalog,
    seal_result_catalog,
)

SHA = "a" * 64
COMMIT = "b" * 40


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _files(root: Path, *, run_id: str = "run-a") -> CatalogEntrySource:
    run_path = root / "small" / run_id / "run_manifest.json"
    pointer_path = root / "small" / run_id / "server_pointer.json"
    metrics_path = root / "small" / run_id / "metrics.csv"
    _write_json(
        run_path,
        {
            "schema_version": "run-manifest-v1",
            "run_id": run_id,
            "model_id": "gradpert_b2",
            "dataset_id": "replogle_k562_essential",
            "protocol_id": "within_cell_unseen_single",
            "run_seed": 1,
            "source_commit": COMMIT,
            "source_dirty": False,
            "formal_eligible": True,
            "config_sha256": SHA,
            "environment_sha256": SHA,
            "canonical_data_sha256": SHA,
            "split_content_sha256": SHA,
            "control_manifest_sha256": SHA,
            "status": "evaluated",
            "best_checkpoint_sha256": SHA,
            "test_evaluations": 1,
        },
    )
    _write_json(
        pointer_path,
        {
            "schema_version": "server-artifact-pointer-v1",
            "run_id": run_id,
            "source_commit": COMMIT,
            "server_root": "/data/yilangliu/GraD-Pert/artifacts",
            "prediction_manifest_path": f"runs/{run_id}/prediction_manifest.json",
            "prediction_manifest_sha256": SHA,
            "evaluation_manifest_path": f"runs/{run_id}/evaluation_manifest.json",
            "evaluation_manifest_sha256": SHA,
            "synchronized_large_artifacts": False,
        },
    )
    metrics_path.write_text("metric_id,value\ntxpert_macro_pearson_delta,0.5\n", encoding="utf-8")
    return CatalogEntrySource(run_path, pointer_path, metrics_path)


def test_catalog_uses_explicit_hash_pinned_files_and_selects_deterministically(
    tmp_path: Path,
) -> None:
    entry = build_catalog_entry(_files(tmp_path), trusted_root=tmp_path)
    catalog_path = tmp_path / "result_catalog.json"
    catalog_hash = seal_result_catalog(catalog_path, catalog_id="five-dataset-v1", entries=[entry])

    catalog = load_result_catalog(
        catalog_path,
        expected_file_sha256=catalog_hash,
        trusted_root=tmp_path,
    )

    assert catalog.require_run("run-a").run_manifest.status == "evaluated"
    assert len(catalog.select(model_id="gradpert_b2")) == 1
    assert catalog.select(dataset_id="norman") == ()


def test_catalog_rejects_mutated_metrics_instead_of_falling_back(tmp_path: Path) -> None:
    source = _files(tmp_path)
    entry = build_catalog_entry(source, trusted_root=tmp_path)
    catalog_path = tmp_path / "result_catalog.json"
    catalog_hash = seal_result_catalog(catalog_path, catalog_id="five-dataset-v1", entries=[entry])
    Path(source.metrics_path).write_text("metric_id,value\nwrong,9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dependency SHA-256 mismatch"):
        load_result_catalog(
            catalog_path,
            expected_file_sha256=catalog_hash,
            trusted_root=tmp_path,
        )


def test_catalog_rejects_unevaluated_run_and_path_escape(tmp_path: Path) -> None:
    source = _files(tmp_path)
    run_path = Path(source.run_manifest_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["status"] = "predicted"
    run["test_evaluations"] = 0
    _write_json(run_path, run)
    with pytest.raises(ValueError, match="only evaluated runs"):
        build_catalog_entry(source, trusted_root=tmp_path)

    outside = tmp_path.parent / f"{tmp_path.name}-outside-metrics.csv"
    outside.write_text("metric_id,value\nx,1\n", encoding="utf-8")
    escaped = CatalogEntrySource(source.run_manifest_path, source.server_pointer_path, outside)
    with pytest.raises(ValueError, match="outside trusted_root"):
        build_catalog_entry(escaped, trusted_root=tmp_path)
    outside.unlink()


def test_catalog_rejects_tampered_catalog_before_parsing(tmp_path: Path) -> None:
    entry = build_catalog_entry(_files(tmp_path), trusted_root=tmp_path)
    catalog_path = tmp_path / "result_catalog.json"
    catalog_hash = seal_result_catalog(catalog_path, catalog_id="five-dataset-v1", entries=[entry])
    catalog_path.write_text("not-json", encoding="utf-8")
    assert hashlib.sha256(catalog_path.read_bytes()).hexdigest() != catalog_hash

    with pytest.raises(ValueError, match="dependency SHA-256 mismatch"):
        load_result_catalog(
            catalog_path,
            expected_file_sha256=catalog_hash,
            trusted_root=tmp_path,
        )


def test_catalog_rejects_development_evaluation(tmp_path: Path) -> None:
    source = _files(tmp_path)
    run_path = Path(source.run_manifest_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["source_dirty"] = True
    run["formal_eligible"] = False
    _write_json(run_path, run)

    with pytest.raises(ValueError, match="formal eligibility"):
        build_catalog_entry(source, trusted_root=tmp_path)
