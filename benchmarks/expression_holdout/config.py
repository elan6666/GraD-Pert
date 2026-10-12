"""Self-contained configs for officially trained expression-holdout baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from gradpert.hashing import sha256_file

OFFICIAL = {
    "scouter": (
        "https://github.com/PancakeZoy/scouter",
        "0cfddd000e19b72ff033ba67c8315f7bc3304932",
    ),
    "scdfm": (
        "https://github.com/AI4Science-WestlakeU/scDFM",
        "8de47e7d3d443939f93f768f7902c66e0e82eda2",
    ),
    "gears": ("https://github.com/snap-stanford/GEARS", "f374e43e197b295016d80395d7a54ddb81cc6769"),
}


def load_config(path: Path) -> dict:
    value = (
        yaml.safe_load(path.read_text()) if path.suffix != ".json" else json.loads(path.read_text())
    )
    required = {
        "schema",
        "model_id",
        "seed",
        "source_code",
        "data",
        "model",
        "training",
        "evaluation",
        "artifacts",
        "adaptations",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("self-contained official baseline config fields differ")
    model = value["model_id"]
    if value["schema"] != "official-expression-holdout-v1" or model not in OFFICIAL:
        raise ValueError("unsupported official baseline schema/model")
    if (value["source_code"]["repository"], value["source_code"]["commit"]) != OFFICIAL[model]:
        raise ValueError("official source/config identity mismatch")
    if value["seed"] != 1 or not value["data"]["exclude_test_target_expression"]:
        raise ValueError("comparison fixes seed1 and expression exclusion")
    ev = value["evaluation"]
    if (
        set(ev.get("expected_hashes", {})) != {"control_manifest_sha256", "evaluation_state_sha256"}
        or ev.get("expected_conditions") != 592
    ):
        raise ValueError("frozen BR evaluation identities required")
    if (
        ev["protocol"] != "v2"
        or ev["control_cells"] != 300
        or ev["checkpoint"] != "last"
        or ev["gene_groups"] != ["all", "seen_expression", "unseen_expression"]
        or ev["gene_sets"] != ["all", "deg"]
        or ev["query_count"] != 1000
        or type(ev["cell_batch"]) is not int
        or ev["cell_batch"] <= 0
    ):
        raise ValueError("common evaluation protocol changed")
    if value["artifacts"] != {"result_mode": "metrics_only", "zero_pkl": True}:
        raise ValueError("official baseline artifact contract changed")
    for key, digest in (value["data"]["expected_hashes"] | ev["expected_hashes"]).items():
        if not key or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("canonical identity must use explicit SHA256")
    for name in ["train_selection_path", "train_selection_sha256"]:
        if not value["data"].get(name):
            raise ValueError("cap40 selection requires explicit path/hash")
    return value


def require_evaluation_identity(data, reference, config: dict) -> None:
    actual = {
        "control_manifest_sha256": data.control_manifest_file_sha256,
        "evaluation_state_sha256": reference.manifest_file_sha256,
    }
    if actual != config["evaluation"]["expected_hashes"]:
        raise ValueError("frozen BR control/reference identity differs")
    if len(data.control_manifest.draws) != config["evaluation"]["expected_conditions"]:
        raise ValueError("frozen test condition count differs")


def require_data_identity(prepared, config: dict) -> None:
    for key, digest in config["data"]["expected_hashes"].items():
        if prepared.receipt.get(key) != digest:
            raise ValueError(f"canonical data identity differs: {key}")
    if (
        prepared.receipt["train_selection"]["manifest_sha256"]
        != config["data"]["train_selection_sha256"]
    ):
        raise ValueError("cap40 selection identity differs")


def validate_preflight(
    receipt: dict, *, config: Path, source_commit: str, official_commit: str, runtime_sha256: str
) -> None:
    frozen = load_config(config)
    if (
        receipt.get("schema") != "official-expression-holdout-preflight-v1"
        or receipt.get("status") != "passed"
        or receipt.get("config_sha256") != sha256_file(config)
        or receipt.get("source_commit") != source_commit
        or receipt.get("source_clean") is not True
        or receipt.get("official_commit") != official_commit
        or receipt.get("runtime_sha256") != runtime_sha256
        or receipt.get("optimizer_updates") != 10
        or receipt.get("restored_checkpoint_predictions_finite") is not True
        or receipt.get("controls_per_condition") != 300
        or receipt.get("zero_pkl") is not True
        or any(receipt.get(k) != v for k, v in frozen["evaluation"]["expected_hashes"].items())
    ):
        raise ValueError("real official 10-update/checkpoint/common-inference preflight mismatch")


def validate_terminal(root: Path, row: dict, source_commit: str) -> dict:
    """Bind scientific completion to every sealed input and actual artifact."""
    config = load_config(Path(row["config"]))
    receipt = json.loads((root / "receipt.json").read_text())
    expected = {
        "source_commit": source_commit,
        "source_clean": True,
        "config_sha256": row["config_sha256"],
        "runtime_sha256": row["runtime_sha256"],
        "official_commit": config["source_code"]["commit"],
        "checkpoint_role": "last",
        **config["evaluation"]["expected_hashes"],
    }
    if (
        any(receipt.get(k) != v for k, v in expected.items())
        or receipt.get("scientific_complete") is not True
        or receipt.get("zero_pkl") is not True
    ):
        raise ValueError("official terminal input identities differ")
    if (
        json.loads((root / "COMPLETE.json").read_text()) != receipt
        or list(root.rglob("*.pkl"))
        or (root / "FAILURE.json").exists()
    ):
        raise ValueError("official terminal artifact postcondition differs")
    if sha256_file(root / "fit/last.pt") != receipt.get("checkpoint_sha256") or sha256_file(
        root / "last-test.json"
    ) != receipt.get("last_test_sha256"):
        raise ValueError("official terminal checkpoint/test seal differs")
    if sha256_file(root / "inference-receipt.json") != receipt.get("inference_receipt_sha256"):
        raise ValueError("official terminal inference provenance differs")
    result = json.loads((root / "last-test.json").read_text())
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("official evaluation identity differs")
    for key, digest in config["data"]["expected_hashes"].items():
        if receipt.get("data_identity", {}).get(key) != digest:
            raise ValueError("official terminal canonical identities differ")
    metric_ids = {
        f"{name}_{genes}"
        for name in ("txpert_macro_pearson_delta", "trishift_pearson_delta", "systema_pearson")
        for genes in ("all", "deg")
    }
    groups = result.get("gene_group_summaries", {})
    if set(groups) != {"seen_expression", "unseen_expression"}:
        raise ValueError("official terminal lacks expression exposure groups")
    count = config["evaluation"]["expected_conditions"]
    for metrics in [result.get("summary"), *groups.values()]:
        if (
            not isinstance(metrics, list)
            or len(metrics) != 6
            or {m.get("metric_id") for m in metrics} != metric_ids
        ):
            raise ValueError("official terminal lacks all18 metrics")
        for m in metrics:
            finite = m.get("finite_condition_count")
            mean = m.get("macro_mean")
            if (
                m.get("total_condition_count") != count
                or type(finite) is not int
                or not 0 <= finite <= count
                or (finite == 0 and mean is not None)
                or (
                    finite > 0 and (mean is None or not math.isfinite(mean) or abs(mean) > 1.000001)
                )
            ):
                raise ValueError("official metric count/value invalid")
    conditions = result.get("condition_metrics", [])
    if (
        len(conditions) != count
        or len({r["condition_id"] for r in conditions}) != count
        or any(r["control_count"] != 300 for r in conditions)
    ):
        raise ValueError("official frozen condition/control coverage differs")
    return receipt
