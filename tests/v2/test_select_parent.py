import json
import runpy
from pathlib import Path

import pytest

from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def selector(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    return runpy.run_path(str(ROOT / "scripts/v2/select_parent.py"))["choose"]


@pytest.fixture
def candidates(tmp_path):
    verified = {"group": "H1", "manifest_sha256": "manifest", "rows": []}
    runs = {}
    for index, name in enumerate(("a", "b")):
        verified["rows"].append({"name": name, "sha256": name, "config": name + ".yaml"})
        runs[name] = []
        for seed in (1, 2):
            root = tmp_path / name / str(seed)
            (root / "fit").mkdir(parents=True)
            checkpoint = root / "fit/best.pt"
            checkpoint.write_bytes(b"synthetic checkpoint evidence")
            identity = {
                "config_sha256": name,
                "source": {"commit": "source", "published_commit": "source", "dirty": False},
                "data": {"run_seed": seed, "split_sha256": "split"},
            }
            history = [
                {
                    "epoch": epoch,
                    "optimizer_steps": epoch,
                    "validation": {
                        "split": "val",
                        "prediction_loss": 1.0 + index,
                        "control_manifest_sha256": "controls",
                        "reference_sha256": "reference",
                        "query_recipe": {"query_count": 1000},
                    },
                }
                for epoch in range(1, 51)
            ]
            atomic_json(root / "run_manifest.json", identity)
            atomic_json(root / "fit/history.json", history)
            atomic_json(
                root / "fit/epoch_state.json",
                {
                    "identity": identity,
                    "epoch": 50,
                    "budget": [50, 1],
                    "best": {
                        "file": "best.pt",
                        "sha256": sha256_file(checkpoint),
                        "epoch": 1,
                        "prediction_loss": 1.0 + index,
                    },
                },
            )
            # Intentionally invalid test data: selection must never read it.
            (root / "test_best.json").write_text("not JSON; deliberately inaccessible as evidence")
            runs[name].append(str(root))
    return verified, runs


def test_parent_selection_reads_validation_only(selector, candidates):
    verified, runs = candidates
    result = selector(verified, runs, [1, 2])
    assert result["winner"]["name"] == "a"
    assert result["test_data_read"] is False
    assert len(result["scores"][0]["evidence"]) == 2


def test_missing_seed_cannot_win(selector, candidates):
    verified, runs = candidates
    runs["a"].pop()
    with pytest.raises(ValueError, match="missing paired seeds"):
        selector(verified, runs, [1, 2])


@pytest.mark.parametrize("mutation", ["unfinished", "test", "reference", "checkpoint"])
def test_invalid_scientific_evidence_rejected(selector, candidates, mutation):
    verified, runs = candidates
    root = Path(runs["a"][0])
    if mutation == "checkpoint":
        (root / "fit/best.pt").write_bytes(b"changed")
    elif mutation == "unfinished":
        path = root / "fit/epoch_state.json"
        value = json.loads(path.read_text())
        value["epoch"] = 49
        atomic_json(path, value)
    else:
        path = root / "fit/history.json"
        value = json.loads(path.read_text())
        value[-1]["validation"]["split" if mutation == "test" else "reference_sha256"] = (
            "changed-" + mutation
        )
        atomic_json(path, value)
    with pytest.raises(ValueError):
        selector(verified, runs, [1, 2])


@pytest.mark.parametrize("mutation", ["steps", "context", "population"])
def test_selection_rejects_changed_execution_evidence(selector, candidates, mutation):
    verified, runs = candidates
    path = Path(runs["a"][0]) / "fit/history.json"
    history = json.loads(path.read_text())
    if mutation == "steps":
        history[-1]["optimizer_steps"] -= 1
    elif mutation == "context":
        history[-1]["validation"]["query_recipe"]["query_count"] = 5000
    else:
        population = Path(runs["a"][0]) / "validation_population.json"
        population.write_text("changed")
        history[-1]["validation"]["population_receipt"] = {
            "file": population.name,
            "sha256": "original",
        }
    atomic_json(path, history)
    with pytest.raises(ValueError):
        selector(verified, runs, [1, 2])
