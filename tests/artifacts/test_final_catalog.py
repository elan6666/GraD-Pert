from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gradpert.artifacts import (
    load_final_result_catalog,
    plan_final_result_catalog,
    seal_final_result_catalog_from_spec,
)
from gradpert.config.matrix import DATASET_IDS, MODEL_IDS

COMMIT = "a" * 40
METRIC_IDS = (
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metrics() -> str:
    lines = ["metric_id,available,macro_mean,reason,finite_condition_count,total_condition_count"]
    lines.extend(f"{metric_id},true,0.5,,5,5" for metric_id in METRIC_IDS)
    return "\n".join(lines) + "\n"


def _write_final_source_spec(root: Path) -> Path:
    coordinates = [
        (model_id, dataset_id, 1) for model_id in MODEL_IDS for dataset_id in DATASET_IDS
    ] + [("gradpert_b2", dataset_id, seed) for dataset_id in DATASET_IDS for seed in (2, 3, 4)]
    entries = []
    for model_id, dataset_id, seed in coordinates:
        run_id = f"formal__{model_id}__{dataset_id}__seed{seed}"
        small = root / "synced" / model_id / dataset_id / f"seed-{seed}" / "small_results"
        small.mkdir(parents=True, exist_ok=True)
        run_path = small / "run_manifest.json"
        pointer_path = small / "server_pointer.json"
        metrics_path = small / "metrics_summary.csv"
        run = {
            "schema_version": "run-manifest-v1",
            "run_id": run_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "protocol_id": f"protocol::{dataset_id}",
            "run_seed": seed,
            "source_commit": COMMIT,
            "source_dirty": False,
            "formal_eligible": True,
            "config_sha256": _hash(f"config::{model_id}::{dataset_id}"),
            "environment_sha256": _hash(f"environment::{model_id}"),
            "canonical_data_sha256": _hash(f"data::{dataset_id}"),
            "split_content_sha256": _hash(f"split::{dataset_id}"),
            "control_manifest_sha256": _hash(f"controls::{dataset_id}"),
            "status": "evaluated",
            "best_checkpoint_sha256": (
                _hash(f"checkpoint::{run_id}")
                if model_id in {"gradpert_b2", "gears", "txpert_public"}
                else None
            ),
            "test_evaluations": 1,
        }
        pointer = {
            "schema_version": "server-artifact-pointer-v1",
            "run_id": run_id,
            "source_commit": COMMIT,
            "server_root": f"/data/yilangliu/GraD-Pert/runs/{run_id}",
            "prediction_manifest_path": f"runs/{run_id}/prediction_manifest.json",
            "prediction_manifest_sha256": _hash(f"prediction::{run_id}"),
            "evaluation_manifest_path": f"runs/{run_id}/evaluation_manifest.json",
            "evaluation_manifest_sha256": _hash(f"evaluation::{run_id}"),
            "synchronized_large_artifacts": False,
        }
        run_path.write_text(json.dumps(run), encoding="utf-8")
        pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
        metrics_path.write_text(_metrics(), encoding="utf-8")
        entries.append(
            {
                "run_manifest_path": run_path.relative_to(root).as_posix(),
                "server_pointer_path": pointer_path.relative_to(root).as_posix(),
                "metrics_path": metrics_path.relative_to(root).as_posix(),
            }
        )
    spec = root / "final-catalog-source-spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "result-catalog-source-spec-v1",
                "catalog_id": "five-dataset-formal-v1",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return spec


def test_final_catalog_plan_seal_sidecar_and_round_trip(tmp_path: Path) -> None:
    spec = _write_final_source_spec(tmp_path)
    plan = plan_final_result_catalog(spec, trusted_root=tmp_path)
    assert plan.audit.entry_count == 45
    assert plan.audit.dataset_count == 5
    assert plan.audit.model_count == 6
    assert all(item["run_count"] == 9 for item in plan.audit.fairness_by_dataset.values())

    output = tmp_path / "result_catalog.json"
    assert not output.exists()
    catalog_hash, audit = seal_final_result_catalog_from_spec(
        output,
        source_spec_path=spec,
        trusted_root=tmp_path,
    )
    assert audit == plan.audit
    assert output.with_suffix(".json.sha256").read_text().split()[0] == catalog_hash
    loaded, loaded_audit = load_final_result_catalog(
        output,
        expected_file_sha256=catalog_hash,
        trusted_root=tmp_path,
    )
    assert len(loaded.entries) == 45
    assert loaded_audit == audit


def test_final_catalog_rejects_missing_coordinate_and_fairness_drift(tmp_path: Path) -> None:
    spec = _write_final_source_spec(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["entries"].pop()
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate mismatch"):
        plan_final_result_catalog(spec, trusted_root=tmp_path)

    spec = _write_final_source_spec(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    run_path = tmp_path / payload["entries"][0]["run_manifest_path"]
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["control_manifest_sha256"] = _hash("drift")
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="fairness hashes differ"):
        plan_final_result_catalog(spec, trusted_root=tmp_path)


def test_final_catalog_rejects_metric_schema_drift(tmp_path: Path) -> None:
    spec = _write_final_source_spec(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    metrics_path = tmp_path / payload["entries"][0]["metrics_path"]
    metrics_path.write_text("metric_id,value\nwrong,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        plan_final_result_catalog(spec, trusted_root=tmp_path)
