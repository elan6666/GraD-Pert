import importlib
import json
from pathlib import Path

import pytest

from gradpert.hashing import sha256_file
from gradpert.training.v2.holdout import make_partition

ROOT = Path(__file__).resolve().parents[2]


def test_full_matrix_is_reviewable_but_no_later_winner_is_invented(tmp_path, monkeypatch):
    from test_capacity_report import valid_receipt

    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    api = importlib.import_module("prepare_matrix")
    probes = []
    for batch in (32, 64):
        config = ROOT / f"configs/v2/exclusion_capacity/m{batch}/gradpert_v2/nadig_jurkat.yaml"
        receipt = tmp_path / f"synthetic-capacity-{batch}.json"
        payload = valid_receipt() | {
            "config_sha256": sha256_file(config),
            "world_size": 2,
            "gpu": "0,1",
        }
        receipt.write_text(json.dumps(payload))
        probes.append(
            {"config": str(config), "receipt": str(receipt), "receipt_sha256": sha256_file(receipt)}
        )
    partition = tmp_path / "synthetic-partition.json"
    partition.write_text(
        json.dumps(make_partition(tuple(f"g{i}" for i in range(2000)), heldout_count=1000, seed=1))
    )
    holdout = {"path": str(partition), "sha256": sha256_file(partition)}
    reference = probes[-1]
    output = tmp_path / "matrix"
    result = api.prepare(
        "nadig_jurkat",
        Path(reference["config"]),
        Path(reference["receipt"]),
        reference["receipt_sha256"],
        output,
        batch_probes=probes,
        holdout=holdout,
    )
    parent = output / result["parent"]
    assert result["measured_global_batches"] == [64, 128]
    assert result["total_training_rows"] == 45
    assert set(result["groups"]) == set(api.GROUPS)
    assert set(result["diagnostics"]) == {"G1_context", "D1"}
    for group, row in result["groups"].items():
        manifest = output / row["manifest"]
        assert sha256_file(manifest) == row["manifest_sha256"]
        api.verify_group(manifest, parent)
        if group in ("B0", "H1"):
            assert api.validate_dependency(manifest, parent) is None
        else:
            assert row["stage"] == "prospective"
            with pytest.raises(ValueError, match="validation-selected parent"):
                api.validate_dependency(manifest, parent)
    # Invalid capacity input must fail before making an apparently usable matrix.
    Path(reference["receipt"]).write_text("changed")
    with pytest.raises(ValueError, match="receipt changed"):
        api.prepare(
            "nadig_jurkat",
            Path(reference["config"]),
            Path(reference["receipt"]),
            reference["receipt_sha256"],
            tmp_path / "invalid",
            batch_probes=probes,
            holdout=holdout,
        )
    assert not (tmp_path / "invalid").exists()
