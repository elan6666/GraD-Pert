import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from test_collect_results import completed, write
from test_components import fixture

from gradpert.config import load_experiment_config
from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.data._io import read_json
from gradpert.training.v2.lifecycle import fit
from gradpert.training.v2.lifecycle import test_selected as evaluate_selected
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer
from gradpert.training.v2.reporting import export_curves

ROOT = Path(__file__).resolve().parents[2]


def test_train_only_resume_final_role_and_training_curves(tmp_path):
    model, batch = fixture()
    initial = copy.deepcopy(model.state_dict())
    identity = {"source": {"commit": "a" * 40}, "run_id": "synthetic"}

    def run(root, *, interrupt=False, resume=False):
        model.load_state_dict(initial)
        objective = JointObjective(model, lambda1=0, lambda2=0)
        optimizer = V2Optimizer(model, 0.001, 0)
        rng = np.random.default_rng(9)

        def batches(epoch):
            for step in range(2):
                rng.random()
                if interrupt and epoch == 1 and step == 1:
                    raise RuntimeError("interrupted train-only epoch")
                yield batch

        journal = fit(
            objective,
            optimizer,
            root=root,
            identity=identity,
            generator=rng,
            epochs=6,
            steps_per_epoch=2,
            batches=batches,
            validate=None,
            schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.16),
            teacher_start=0.99,
            teacher_end=1,
            microbatch=2,
            bf16=False,
            resume=resume,
        )
        assert journal["best"] is None and journal["selection_metric"] == "final_epoch"
        assert journal["last"]["epoch"] == 6 and "joint_loss" not in journal["last"]
        assert not (root / "best.pt").exists()
        assert [p.name for p in root.glob("epoch-*.pt")] == ["epoch-0006.pt"]
        calls = []
        receipts = evaluate_selected(
            objective,
            root=root,
            evaluation_identity=identity,
            test=lambda: {"split": "test"},
            on_role_start=calls.append,
        )
        assert list(receipts) == ["last"] and calls == ["last"]
        receipt = export_curves(root)
        assert receipt["validation_mode"] == "disabled"
        assert not list(root.rglob("validation.epoch-*.json"))
        assert (root / "training_loss.png").stat().st_size > 1000
        return copy.deepcopy(objective.state_dict()), optimizer.state_dict(), rng.random()

    from test_relay_method import assert_tree_close

    full = run(tmp_path / "full")
    with pytest.raises(RuntimeError, match="interrupted"):
        run(tmp_path / "resumed", interrupt=True)
    assert_tree_close(full, run(tmp_path / "resumed", resume=True))


def train_only_completion(root):
    completed(root, epochs=6)
    fit_root = root / "fit"
    resolved = json.loads((root / "resolved_config.json").read_text())
    resolved["model"] = {"parameters": {"validation_mode": {"value": "disabled"}}}
    resolved["training"]["monitor"] = "none"
    write(root / "resolved_config.json", resolved)
    journal = read_json(fit_root / "epoch_state.json")
    journal.update(best=None, selection_metric="final_epoch")
    journal["last"].pop("prediction_loss")
    write(fit_root / "epoch_state.json", journal)
    history = read_json(fit_root / "history.json")
    for row in history:
        row["validation"] = {"validation_mode": "disabled", "performed": False}
    write(fit_root / "history.json", history)
    test = read_json(fit_root / "last-test.json")
    test["identity"]["checkpoint"] = journal["last"]
    write(fit_root / "last-test.json", test)
    (fit_root / "best-test.json").unlink()
    write(root / "COMPLETE.json", {**journal, "test_roles": ["last"], "zero_pkl": True})


@pytest.mark.parametrize("corruption", [None, "best", "validation", "epoch", "test"])
def test_final_only_collector_and_queue_terminal_gates(tmp_path, monkeypatch, corruption):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    collect = runpy.run_path(str(ROOT / "scripts/v2/collect_results.py"))["collect_run"]
    train_only_completion(tmp_path)
    if corruption == "best":
        journal = read_json(tmp_path / "fit/epoch_state.json")
        journal["best"] = journal["last"]
        write(tmp_path / "fit/epoch_state.json", journal)
    elif corruption == "validation":
        history = read_json(tmp_path / "fit/history.json")
        history[0]["validation"]["joint_loss"] = 0
        write(tmp_path / "fit/history.json", history)
    elif corruption == "epoch":
        complete = read_json(tmp_path / "COMPLETE.json")
        complete["epoch"] = 5
        write(tmp_path / "COMPLETE.json", complete)
    elif corruption == "test":
        (tmp_path / "fit/last-test.json").unlink()
    if corruption:
        with pytest.raises(ValueError):
            collect(tmp_path)
    else:
        rows = collect(tmp_path)
        assert len(rows) == 1 and rows[0]["role"] == "last" and rows[0]["status"] == "complete"
        assert rows[0]["validation_selection_loss"] is None
        plan = {"run_root": str(tmp_path), "config_sha256": "c" * 64, "source_commit": "a" * 40}
        write(tmp_path / "launch.json", plan)
        assert (
            runpy.run_path(str(ROOT / "scripts/v2/run_group.py"))["next_action"](plan)
            == "skip_complete"
        )


def test_five_configs_are_self_contained_single_variable_arms(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/v2"))
    verify = runpy.run_path(str(ROOT / "scripts/v2/run_functional_group.py"))["verify_manifest"]
    manifest = verify(ROOT, ROOT / "configs/v2/cap40_functional_b0_ka/manifest.json")
    assert [r["name"] for r in manifest["rows"]] == ["B0", "K1", "K2", "A1", "A2"]
    for row in manifest["rows"]:
        config = load_experiment_config(ROOT / row["config"])
        assert config.training.monitor == "none"
        assert config.model.parameters["streams"].value == 4


def test_disabled_validation_requires_explicit_fresh_six_epoch_protocol():
    import yaml

    from gradpert.config.schema import ExperimentConfig

    payload = yaml.safe_load(
        (ROOT / "configs/v2/cap40_functional_b0_ka/B0/gradpert_v2/nadig_jurkat.yaml").read_text()
    )
    payload["model"]["parameters"]["validation_mode"]["value"] = "joint_only"
    with pytest.raises(ValueError, match="disabled validation"):
        ExperimentConfig.model_validate(payload)
