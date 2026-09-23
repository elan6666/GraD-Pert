import json
import runpy
from pathlib import Path

import pytest

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
collect = runpy.run_path(str(ROOT / "scripts/v2/collect_results.py"))["collect_run"]


def write(path, value):
    path.write_text(json.dumps(value))


def completed(root, epochs=50):
    fit = root / "fit"
    fit.mkdir()
    checkpoint = fit / f"epoch-{epochs:04d}.pt"
    checkpoint.write_bytes(b"synthetic checkpoint hash fixture")
    identity = {
        "source": {"dirty": False, "commit": "a" * 40, "published_commit": "a" * 40},
        "run_id": "synthetic",
        "config_sha256": "c" * 64,
        "resolved_config_sha256": "d" * 64,
    }
    selected = {
        "file": checkpoint.name,
        "sha256": sha256_file(checkpoint),
        "epoch": epochs,
        "prediction_loss": 1 / epochs,
    }
    journal = {
        "identity": identity,
        "epoch": epochs,
        "budget": [epochs, 2],
        "best": selected,
        "last": selected,
    }
    write(root / "run_manifest.json", identity)
    write(
        root / "resolved_config.json",
        {
            "training": {
                "formal_run_policy": f"v2_fixed_{epochs}",
                "max_epochs": {"value": epochs},
            }
        },
    )
    write(fit / "epoch_state.json", journal)
    write(
        fit / "history.json",
        [
            {"epoch": e, "validation": {"split": "val", "prediction_loss": 1 / e}}
            for e in range(1, epochs + 1)
        ],
    )
    for role in ("best", "last"):
        write(
            fit / f"{role}-test.json",
            {
                "identity": {
                    "training": identity,
                    "evaluation": identity,
                    "checkpoint": selected,
                    "role": role,
                },
                "result": {"split": "test", "prediction_loss": 0.3, "metrics": []},
            },
        )
    write(root / "COMPLETE.json", {**journal, "test_roles": ["best", "last"], "zero_pkl": True})


@pytest.mark.parametrize("epochs", [5, 50])
def test_keeps_both_roles_even_when_checkpoint_identical(tmp_path, epochs):
    completed(tmp_path, epochs)
    rows = collect(tmp_path)
    assert [r["role"] for r in rows] == ["best", "last"]
    assert all(r["status"] == "complete" for r in rows)
    assert rows[0]["checkpoint_sha256"] == rows[1]["checkpoint_sha256"]
    assert rows[0]["training_sha"] == rows[0]["evaluation_sha"]


def test_partial_run_reports_missing_tests(tmp_path):
    completed(tmp_path)
    (tmp_path / "COMPLETE.json").unlink()
    (tmp_path / "fit/last-test.json").unlink()
    assert [r["status"] for r in collect(tmp_path)] == ["tested_completion_missing", "missing_test"]


@pytest.mark.parametrize("corruption", ["checkpoint", "test", "history", "pkl"])
def test_completion_cannot_hide_missing_or_corrupt_evidence(tmp_path, corruption):
    completed(tmp_path)
    if corruption == "checkpoint":
        (tmp_path / "fit/epoch-0050.pt").write_bytes(b"changed")
    elif corruption == "test":
        (tmp_path / "fit/last-test.json").unlink()
    elif corruption == "history":
        write(tmp_path / "fit/history.json", [])
    else:
        (tmp_path / "unexpected.pkl").write_bytes(b"fixture")
    with pytest.raises(ValueError):
        collect(tmp_path)


def test_rejects_early_completion_under_five_epoch_contract(tmp_path):
    completed(tmp_path, 5)
    complete = json.loads((tmp_path / "COMPLETE.json").read_text())
    complete["epoch"] = 4
    write(tmp_path / "COMPLETE.json", complete)
    with pytest.raises(ValueError, match="fixed-epoch contract"):
        collect(tmp_path)
