"""Inference scheduling fallback never changes training or launches a fit."""

import copy
import importlib
from pathlib import Path

import pytest
from test_br400_group import ROOT, execution_mocks
from test_br400_group import prepared as prepared

from gradpert.data._io import atomic_json, read_json

family = importlib.import_module("generate_ffn_five")
runner = importlib.import_module("run_br400_group")


@pytest.mark.parametrize("batch", [128, 192, 256])
def test_smaller_inference_batch_only_changes_execution_chunk(batch):
    old = family.configurations(ROOT)
    new = family.configurations(ROOT, eval_batch=batch)
    for name in family.PRIORITY:
        observed = copy.deepcopy(new[name])
        assert observed["training"]["eval_batch_size"]["value"] == batch
        observed["training"]["eval_batch_size"] = old[name]["training"]["eval_batch_size"]
        assert observed == old[name]
    manifest = family.verify_manifest(
        ROOT, ROOT / f"configs/v2/ffn_five_eval{batch}_20261014/manifest.json"
    )
    assert manifest["global_batch"] == 400 and manifest["epochs"] == 4


@pytest.mark.parametrize("batch", [True, None, 1, 301])
def test_undeclared_batch_rejected(batch):
    with pytest.raises(ValueError):
        family.configurations(ROOT, eval_batch=batch)


def test_eval_only_dispatch_cannot_train_or_create_scientific_completion(prepared, monkeypatch):
    old, output = prepared
    runtime = Path(old["runtime"])
    value = read_json(runtime)
    value["official_dependency"] = {"purpose": "synthetic_dependency_fixture"}
    atomic_json(runtime, value)
    policy = copy.deepcopy(old["shared_gpu_policy"])
    policy.update(
        additional_command_basenames=["run_totalvi_source_validation_development.py"],
        authorization_reference="2026-10-12-user-shared-memory90-coexistence",
        max_own_memory_fraction=0.90,
    )
    target = output.parent / "eval-only"
    queue = runner.prepare(
        ROOT, ROOT / "configs/v2/ffn_five_eval256_20261014/manifest.json", runtime, target, policy
    )
    queue, target, calls = execution_mocks((queue, target), monkeypatch)
    monkeypatch.setattr(runner, "official_dependency_ready", lambda *_: True)
    # Synthetic receipts intentionally exercise dispatch only; real gate tests
    # independently validate binding, state, RNG, tolerances and population.
    monkeypatch.setattr(runner, "validate_eval_gate", lambda *args: {})
    runner.execute(queue, target, eval_only_row="FB")
    assert [(name, stage) for name, stage, _, _ in calls] == [
        ("FB", "eval-capacity-g0"),
        ("FB", "eval-capacity-g1"),
    ]
    assert not (target / "COMPLETE.json").exists()
    assert not (target / "PREFLIGHT_COMPLETE.json").exists()
    assert not (target / "MAIN_ACCEPTANCE.json").exists()
    observed = read_json(target / "EVAL_CAPACITY_COMPLETE.json")
    assert observed["scientific_complete"] is False and observed["completed"] == {}
    assert observed["preflights"] == {}
