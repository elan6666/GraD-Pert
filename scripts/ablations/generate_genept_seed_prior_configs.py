#!/usr/bin/env python3
"""Generate the three pinned GenePT text-prior comparison configs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/ablations/nadig_jurkat/a0_default/gradpert_b2/nadig_jurkat.yaml"
LOCK = ROOT / "configs/experiments/genept_seed_priors/artifacts.json"
OUTPUT = ROOT / "configs/experiments/genept_seed_priors"
REFERENCE = "docs/experiments/GENEPT_SEED_PRIOR_COMPARISON.md"
CONDITION_IDS = (
    "latest_genept_model3",
    "genept_seed",
    "genept_seed_goexp",
)
ALLOWED_PARAMETER_DIFFERENCES = {
    "performance_pilot_variant",
    "genept_expected_sha256",
    "genept_artifact_path",
}


def sourced(value: object) -> dict[str, object]:
    return {"value": value, "source": "user_locked", "reference": REFERENCE}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lock() -> dict[str, dict[str, object]]:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "genept-seed-prior-artifacts-v1":
        raise ValueError("unexpected prior artifact lock schema")
    raw = payload.get("conditions")
    if not isinstance(raw, list):
        raise ValueError("prior artifact lock conditions must be a list")
    conditions: dict[str, dict[str, object]] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("prior artifact lock row must be an object")
        condition_id = row.get("condition_id")
        artifact_path = row.get("artifact_path")
        artifact_sha256 = row.get("artifact_sha256")
        if (
            condition_id not in CONDITION_IDS
            or not isinstance(artifact_path, str)
            or not artifact_path.startswith("/data/yilangliu/")
            or not isinstance(artifact_sha256, str)
            or len(artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in artifact_sha256)
        ):
            raise ValueError("prior artifact lock row is invalid")
        conditions[str(condition_id)] = row
    if tuple(conditions) != CONDITION_IDS:
        raise ValueError("prior artifact lock condition order or membership differs")
    return conditions


def _common_contract(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    parameters = normalized["model"]["parameters"]
    for key in ALLOWED_PARAMETER_DIFFERENCES:
        parameters.pop(key)
    normalized["artifacts"]["root"] = "<condition-root>"
    return normalized


def render() -> None:
    conditions = _load_lock()
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    contract_hash: str | None = None
    for condition_id in CONDITION_IDS:
        artifact = conditions[condition_id]
        payload = copy.deepcopy(base)
        parameters = payload["model"]["parameters"]
        parameters["performance_pilot_variant"] = sourced(f"genept_prior_{condition_id}")
        parameters["runtime_graph_root"] = sourced(
            "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"
        )
        parameters["gene_feature_mode"] = sourced("genept_id_residual")
        parameters["genept_expected_sha256"] = sourced(artifact["artifact_sha256"])
        parameters["genept_artifact_path"] = sourced(artifact["artifact_path"])
        payload["artifacts"]["root"] = f"runs/experiments/genept_seed_priors/{condition_id}"
        destination = OUTPUT / condition_id / "gradpert_b2" / "nadig_jurkat.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        common = hashlib.sha256(
            json.dumps(
                _common_contract(payload),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if contract_hash is None:
            contract_hash = common
        elif common != contract_hash:
            raise ValueError("comparison configs differ outside the prior condition")
        rows.append(
            {
                **artifact,
                "config_path": str(destination.relative_to(ROOT)),
                "config_sha256": _sha256(destination),
                "common_contract_sha256": common,
            }
        )
    manifest = {
        "schema_version": "genept-seed-prior-configs-v1",
        "design_reference": REFERENCE,
        "dataset_id": "nadig_jurkat",
        "run_seed": 1,
        "max_epochs": 10,
        "common_contract_sha256": contract_hash,
        "allowed_differences": sorted(ALLOWED_PARAMETER_DIFFERENCES | {"artifacts.root"}),
        "conditions": rows,
    }
    (OUTPUT / "matrix.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render()
