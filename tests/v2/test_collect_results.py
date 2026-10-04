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


@pytest.mark.parametrize("epochs", [1, 3, 5, 6, 20, 50])
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


def test_joint_loss_selects_best_even_when_prediction_loss_prefers_last(tmp_path):
    completed(tmp_path, 3)
    fit = tmp_path / "fit"
    journal = json.loads((fit / "epoch_state.json").read_text())
    history = json.loads((fit / "history.json").read_text())
    for row, joint_loss in zip(history, (0.1, 0.3, 0.2), strict=True):
        row["validation"]["joint_loss"] = joint_loss
    write(fit / "history.json", history)
    best_checkpoint = fit / "epoch-0001.pt"
    best_checkpoint.write_bytes(b"joint-selected checkpoint")
    journal["selection_metric"] = "joint_loss"
    journal["best"] = {
        "file": best_checkpoint.name,
        "sha256": sha256_file(best_checkpoint),
        "epoch": 1,
        "joint_loss": 0.1,
        "prediction_loss": 1.0,
    }
    journal["last"]["joint_loss"] = 0.2
    write(fit / "epoch_state.json", journal)
    for role in ("best", "last"):
        test = json.loads((fit / f"{role}-test.json").read_text())
        test["identity"]["checkpoint"] = journal[role]
        write(fit / f"{role}-test.json", test)
    write(
        tmp_path / "COMPLETE.json",
        {**journal, "test_roles": ["best", "last"], "zero_pkl": True},
    )
    rows = collect(tmp_path)
    assert [row["checkpoint_epoch"] for row in rows] == [1, 3]
    assert [row["validation_selection_loss"] for row in rows] == [0.1, 0.2]
    assert all(row["validation_selection_metric"] == "joint_loss" for row in rows)


def test_joint_only_completion_retains_both_test_roles(tmp_path):
    test_joint_loss_selects_best_even_when_prediction_loss_prefers_last(tmp_path)
    fit = tmp_path / "fit"
    history = json.loads((fit / "history.json").read_text())
    for record in history:
        record["validation"].pop("prediction_loss")
        record["validation"]["validation_mode"] = "joint_only"
    write(fit / "history.json", history)
    journal = json.loads((fit / "epoch_state.json").read_text())
    for role in ("best", "last"):
        journal[role].pop("prediction_loss")
        test = json.loads((fit / f"{role}-test.json").read_text())
        test["identity"]["checkpoint"] = journal[role]
        write(fit / f"{role}-test.json", test)
    write(fit / "epoch_state.json", journal)
    write(tmp_path / "COMPLETE.json", {**journal, "test_roles": ["best", "last"], "zero_pkl": True})
    rows = collect(tmp_path)
    assert all(row["status"] == "complete" for row in rows)
    assert all(row["validation_prediction_loss"] is None for row in rows)
    assert all(row["test_prediction_loss"] == 0.3 for row in rows)


@pytest.mark.parametrize("epochs", [6, 20])
def test_fixed_epoch_joint_only_completion_advances_queue_without_retraining(
    tmp_path, monkeypatch, epochs
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    next_action = runpy.run_path(str(ROOT / "scripts/v2/run_group.py"))["next_action"]
    completed(tmp_path, epochs)
    fit = tmp_path / "fit"
    journal = json.loads((fit / "epoch_state.json").read_text())
    history = json.loads((fit / "history.json").read_text())
    journal["selection_metric"] = "joint_loss"
    for record in history:
        record["validation"]["joint_loss"] = record["validation"].pop("prediction_loss")
        record["validation"]["validation_mode"] = "joint_only"
    for role in ("best", "last"):
        journal[role]["joint_loss"] = journal[role].pop("prediction_loss", 1 / epochs)
        test = json.loads((fit / f"{role}-test.json").read_text())
        test["identity"]["checkpoint"] = journal[role]
        write(fit / f"{role}-test.json", test)
    write(fit / "history.json", history)
    write(fit / "epoch_state.json", journal)
    write(tmp_path / "COMPLETE.json", {**journal, "test_roles": ["best", "last"], "zero_pkl": True})
    plan = {"run_root": str(tmp_path), "config_sha256": "c" * 64, "source_commit": "a" * 40}
    write(tmp_path / "launch.json", plan)
    assert next_action(plan) == "skip_complete"
    assert all(row["checkpoint_epoch"] == epochs for row in collect(tmp_path))


@pytest.mark.parametrize("epochs", [6, 20])
@pytest.mark.parametrize("corruption", ["budget", "history", "checkpoint", "test"])
def test_fixed_epoch_policy_still_requires_exact_terminal_evidence(tmp_path, corruption, epochs):
    completed(tmp_path, epochs)
    if corruption == "budget":
        resolved = json.loads((tmp_path / "resolved_config.json").read_text())
        resolved["training"]["max_epochs"]["value"] = 5
        write(tmp_path / "resolved_config.json", resolved)
    elif corruption == "history":
        history = json.loads((tmp_path / "fit/history.json").read_text())
        write(tmp_path / "fit/history.json", history[:-1])
    elif corruption == "checkpoint":
        (tmp_path / f"fit/epoch-{epochs:04d}.pt").write_bytes(b"corrupt")
    else:
        (tmp_path / "fit/best-test.json").unlink()
    with pytest.raises(ValueError):
        collect(tmp_path)
