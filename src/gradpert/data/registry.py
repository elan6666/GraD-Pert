"""Load and verify the exact five-entry dataset registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gradpert.config.loader import _reject_defaults_key, _reject_hidden_composition
from gradpert.config.schema import DatasetId
from gradpert.data.schema import DatasetRegistryEntry
from gradpert.hashing import sha256_file

DATASET_IDS: tuple[DatasetId, ...] = (
    "replogle_k562_essential",
    "replogle_rpe1_essential",
    "nadig_jurkat",
    "nadig_hepg2",
    "norman",
)


def load_dataset_registry(path: str | Path) -> DatasetRegistryEntry:
    registry_path = Path(path)
    text = registry_path.read_text(encoding="utf-8")
    _reject_hidden_composition(text)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset registry must be a mapping: {registry_path}")
    _reject_defaults_key(payload)
    entry = DatasetRegistryEntry.model_validate(payload)
    if registry_path.name != f"{entry.dataset_id}.yaml":
        raise ValueError("dataset registry identity/path mismatch")
    return entry


def verify_dataset_registry(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    expected = {base / f"{dataset_id}.yaml" for dataset_id in DATASET_IDS}
    discovered = set(base.glob("*.yaml")) | set(base.glob("*.yml"))
    missing = sorted(str(path) for path in expected - discovered)
    extra = sorted(str(path) for path in discovered - expected)
    if missing or extra:
        raise ValueError(f"Dataset registry mismatch: missing={missing}, extra={extra}")
    entries = []
    for path in sorted(expected):
        entry = load_dataset_registry(path)
        entries.append(
            {
                "dataset_id": entry.dataset_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "source_url": entry.source.url,
                "source_checksum": (
                    f"{entry.source.checksum.algorithm}:{entry.source.checksum.value}"
                ),
                "source_availability": entry.source.availability,
                "source_mapping_audit_state": entry.source_metadata.audit_state,
            }
        )
    return {"count": len(entries), "expected_count": 5, "entries": entries}
