from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from benchmarks.expression_holdout.config import load_config, validate_preflight
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/v2/official_expression_20261013/scouter_jurkat.yaml"


@pytest.mark.parametrize("name", ["scouter", "scdfm", "gears"])
def test_frozen_official_recipes_and_shared_data(name):
    cfg = load_config(ROOT / f"configs/v2/official_expression_20261013/{name}_jurkat.yaml")
    assert cfg["data"] == load_config(CONFIG)["data"]
    assert cfg["data"]["exclude_test_target_expression"] is True
    assert len(cfg["source_code"]["commit"]) == 40


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["data"].update(exclude_test_target_expression=False),
        lambda c: c["evaluation"].update(control_cells=128),
        lambda c: c["source_code"].update(commit="a" * 40),
    ],
)
def test_protocol_and_source_drift_rejected(tmp_path, mutation):
    cfg = deepcopy(load_config(CONFIG))
    mutation(cfg)
    path = tmp_path / "modified.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_probe_requires_actual10_updates_and_restore_with_same_identity():
    cfg = load_config(CONFIG)
    receipt = {
        "schema": "official-expression-holdout-preflight-v1",
        "status": "passed",
        "config_sha256": sha256_file(CONFIG),
        "source_commit": "s",
        "source_clean": True,
        "official_commit": cfg["source_code"]["commit"],
        "runtime_sha256": "r",
        "optimizer_updates": 10,
        "restored_checkpoint_predictions_finite": True,
        "controls_per_condition": 300,
        "zero_pkl": True,
        **cfg["evaluation"]["expected_hashes"],
    }
    kwargs = dict(
        config=CONFIG,
        source_commit="s",
        official_commit=cfg["source_code"]["commit"],
        runtime_sha256="r",
    )
    validate_preflight(receipt, **kwargs)
    for change in [
        {"optimizer_updates": 2},
        {"source_clean": False},
        {"zero_pkl": False},
        {"restored_checkpoint_predictions_finite": False},
        {"config_sha256": "x"},
    ]:
        with pytest.raises(ValueError):
            validate_preflight(receipt | change, **kwargs)


def test_frozen_control_reference_identity():
    from types import SimpleNamespace

    from benchmarks.expression_holdout.config import require_evaluation_identity

    cfg = load_config(CONFIG)
    data = SimpleNamespace(
        control_manifest_file_sha256=cfg["evaluation"]["expected_hashes"][
            "control_manifest_sha256"
        ],
        control_manifest=SimpleNamespace(draws=[None] * 592),
    )
    ref = SimpleNamespace(
        manifest_file_sha256=cfg["evaluation"]["expected_hashes"]["evaluation_state_sha256"]
    )
    require_evaluation_identity(data, ref, cfg)
    ref.manifest_file_sha256 = "a" * 64
    with pytest.raises(ValueError, match="identity differs"):
        require_evaluation_identity(data, ref, cfg)


def test_terminal_requires_sealed18_metrics_and_artifacts(tmp_path):
    import json

    from benchmarks.expression_holdout.config import validate_terminal

    cfg = load_config(CONFIG)
    root = tmp_path / "run"
    (root / "fit").mkdir(parents=True)
    (root / "fit/last.pt").write_bytes(b"checkpoint")
    (root / "inference-receipt.json").write_text("{}")
    ids = [
        f"{name}_{genes}"
        for name in ("txpert_macro_pearson_delta", "trishift_pearson_delta", "systema_pearson")
        for genes in ("all", "deg")
    ]
    metrics = [
        dict(metric_id=k, macro_mean=0.3, finite_condition_count=592, total_condition_count=592)
        for k in ids
    ]
    row = dict(config=str(CONFIG), config_sha256=sha256_file(CONFIG), runtime_sha256="r")
    expected = dict(
        source_commit="s",
        source_clean=True,
        config_sha256=row["config_sha256"],
        runtime_sha256="r",
        official_commit=cfg["source_code"]["commit"],
        checkpoint_role="last",
        **cfg["evaluation"]["expected_hashes"],
    )
    result = dict(
        expected,
        summary=metrics,
        gene_group_summaries={k: metrics for k in ("seen_expression", "unseen_expression")},
        condition_metrics=[dict(condition_id=str(i), control_count=300) for i in range(592)],
    )

    def save(value):
        (root / "last-test.json").write_text(json.dumps(value))
        receipt = dict(
            expected,
            inference_receipt_sha256=sha256_file(root / "inference-receipt.json"),
            scientific_complete=True,
            zero_pkl=True,
            checkpoint_sha256=sha256_file(root / "fit/last.pt"),
            last_test_sha256=sha256_file(root / "last-test.json"),
            data_identity=cfg["data"]["expected_hashes"],
        )
        for filename in ("receipt.json", "COMPLETE.json"):
            (root / filename).write_text(json.dumps(receipt))

    save(result)
    validate_terminal(root, row, "s")
    broken = deepcopy(result)
    broken["summary"] = metrics[:5]
    save(broken)
    with pytest.raises(ValueError, match="18 metrics"):
        validate_terminal(root, row, "s")
    save(result)
    (root / "fit/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="seal differs"):
        validate_terminal(root, row, "s")
