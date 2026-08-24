"""Deterministic small-file exports derived only from a sealed evaluation bundle."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gradpert.artifacts._io import atomic_bytes, sha256_file
from gradpert.hashing import sha256_json

if TYPE_CHECKING:
    from gradpert.artifacts.evaluation import EvaluationBundle


@dataclass(frozen=True)
class SmallMetricExports:
    root: Path
    files: dict[str, str]


def _csv_bytes(fieldnames: list[str], rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_small_metric_exports(
    bundle: EvaluationBundle,
    output_root: str | Path,
) -> SmallMetricExports:
    """Write notebook-friendly CSV/JSON summaries without server-only arrays."""

    root = Path(output_root).resolve()
    evaluator_state = {
        "manifest_file_sha256": bundle.manifest.evaluation_state_manifest_file_sha256,
        "arrays_sha256": bundle.manifest.evaluation_state_arrays_sha256,
        "condition_ids_sha256": bundle.manifest.evaluation_state_condition_ids_sha256,
        "de_gene_indices_sha256": bundle.manifest.de_gene_indices_sha256,
        "top_de_gene_indices_sha256": bundle.manifest.top_de_gene_indices_sha256,
        "de_unavailable_reasons_sha256": (bundle.manifest.de_unavailable_reasons_sha256),
        "de_method": bundle.manifest.de_method,
        "de_reference": bundle.manifest.de_reference,
        "de_source_commit": bundle.manifest.de_source_commit,
        "systema_reference_condition_ids": (bundle.manifest.systema_reference_condition_ids),
        "systema_reference_condition_ids_sha256": (
            bundle.manifest.systema_reference_condition_ids_sha256
        ),
        "metric_control_means_content_sha256": (
            bundle.manifest.metric_control_means_content_sha256
        ),
    }
    condition_rows: list[dict[str, object]] = []
    for condition_id in sorted(bundle.conditions):
        for result in bundle.conditions[condition_id].metrics.results:
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "metric_id": result.metric_id,
                    "value": "" if result.value is None else format(result.value, ".17g"),
                    "reason": result.reason or "",
                    "gene_count": result.gene_count,
                }
            )
    summary_rows = [
        {
            "metric_id": item.metric_id,
            "available": str(item.available).lower(),
            "macro_mean": "" if item.macro_mean is None else format(item.macro_mean, ".17g"),
            "reason": item.reason or "",
            "finite_condition_count": item.finite_condition_count,
            "total_condition_count": item.total_condition_count,
        }
        for item in bundle.manifest.metrics
    ]
    summary_json = {
        "schema_version": "metrics-summary-v1",
        "evaluation_bundle_sha256": bundle.file_sha256,
        "prediction_artifact_sha256": bundle.manifest.prediction_artifact_file_sha256,
        "evaluator_state": evaluator_state,
        "metrics": [item.model_dump(mode="json") for item in bundle.manifest.metrics],
    }
    availability_json = {
        "schema_version": "metric-availability-v1",
        "evaluation_bundle_sha256": bundle.file_sha256,
        "evaluator_state_manifest_file_sha256": (
            bundle.manifest.evaluation_state_manifest_file_sha256
        ),
        "metrics": [
            {
                "metric_id": item.metric_id,
                "available": item.available,
                "reason": item.reason,
                "finite_condition_count": item.finite_condition_count,
                "total_condition_count": item.total_condition_count,
            }
            for item in bundle.manifest.metrics
        ],
    }
    payloads = {
        "metrics_per_condition.csv": _csv_bytes(
            ["condition_id", "metric_id", "value", "reason", "gene_count"],
            condition_rows,
        ),
        "metrics_summary.csv": _csv_bytes(
            [
                "metric_id",
                "available",
                "macro_mean",
                "reason",
                "finite_condition_count",
                "total_condition_count",
            ],
            summary_rows,
        ),
        "metrics_summary.json": _json_bytes(summary_json),
        "metric_availability.json": _json_bytes(availability_json),
    }
    for filename, payload in payloads.items():
        atomic_bytes(root / filename, payload)
    file_hashes = {filename: sha256_file(root / filename) for filename in sorted(payloads)}
    export_manifest = {
        "schema_version": "small-metric-exports-v1",
        "evaluation_manifest_sha256": sha256_json(bundle.manifest.model_dump(mode="json")),
        "evaluation_bundle_sha256": bundle.file_sha256,
        "evaluator_state": evaluator_state,
        "files": file_hashes,
    }
    atomic_bytes(root / "metrics_export_manifest.json", _json_bytes(export_manifest))
    file_hashes["metrics_export_manifest.json"] = sha256_file(root / "metrics_export_manifest.json")
    return SmallMetricExports(root=root, files=file_hashes)
