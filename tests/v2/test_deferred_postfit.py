"""Real full-state training is distinct from independent scientific completion."""

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
from test_components import fixture

from gradpert.config import load_experiment_config
from gradpert.config.step_schedule import EndpointLRWarmupCosine
from gradpert.data._io import atomic_json, read_json
from gradpert.execution import v2_deferred_postfit as postfit
from gradpert.execution.v2_training_stage import seal_training_stage, validate_training_stage
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.lifecycle import fit
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/v2/cap40_loss_ablations_20261007/L0/gradpert_v2/nadig_jurkat.yaml"


@pytest.fixture
def trained(tmp_path):
    torch.set_num_threads(2)
    model, batch = fixture()
    objective = JointObjective(model, lambda1=1, lambda2=1)
    optimizer = V2Optimizer(model, 0.001, 0)
    config = load_experiment_config(CFG)
    identity = {
        "run_id": "synthetic",
        "source": {"commit": "a" * 40, "published_commit": "a" * 40, "dirty": False},
        "config_sha256": sha256_file(CFG),
        "resolved_config_sha256": sha256_json(config.model_dump(mode="json")),
    }
    root = tmp_path / "run"
    root.mkdir()
    plan = {
        "run_root": str(root),
        "run_id": "synthetic",
        "config": str(CFG),
        "config_sha256": identity["config_sha256"],
        "source_commit": "a" * 40,
        "postfit_policy": "deferred",
        "resolved_config": config.model_dump(mode="json"),
    }
    atomic_json(root / "launch.json", plan)
    atomic_json(root / "run_manifest.json", identity)
    journal = fit(
        objective,
        optimizer,
        root=root / "fit",
        identity=identity,
        generator=np.random.default_rng(9),
        epochs=6,
        steps_per_epoch=2,
        batches=lambda epoch: iter([batch, batch]),
        validate=None,
        schedule=EndpointLRWarmupCosine(0.001, 0.0002, 0.16),
        teacher_start=0.99,
        teacher_end=1,
        microbatch=2,
        bf16=False,
    )
    receipt = seal_training_stage(root, identity, 6)
    return root, plan, objective, optimizer, journal, receipt


def test_full_nonzero_lr_checkpoint_can_advance_without_test(trained):
    root, plan, objective, optimizer, journal, receipt = trained
    assert optimizer.steps == 12
    assert receipt == validate_training_stage(plan)
    assert receipt["stage"] == "training_complete" and not receipt["scientific_complete"]
    assert receipt["test_roles"] == ["last"] and not (root / "COMPLETE.json").exists()
    payload = torch.load(root / "fit" / journal["last"]["file"], weights_only=False)
    assert payload["optimizer"]["steps"] == 12
    assert payload["rank_rng_states"] is None
    assert payload["architecture"] == objective.student.options.payload()
    assert any("teacher" in key for key in payload["objective"])
    assert payload["objective"]["ssl1_cls_center"].abs().sum() > 0
    assert "numpy_generator" in payload and "torch_rng" in payload
    with pytest.raises(FileNotFoundError):
        postfit.finalize_deferred_run(plan, receipt)
    assert not (root / "COMPLETE.json").exists()


def test_changed_history_and_checkpoint_cannot_advance(trained):
    root, plan, _, _, journal, _ = trained
    checkpoint = root / "fit" / journal["last"]["file"]
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_training_stage(plan)


def test_evaluation_failure_keeps_training_terminal(trained, monkeypatch):
    root, plan, _, _, _, receipt = trained
    plan = {**plan, "data_root": "unused", "runtime": "unused"}
    monkeypatch.setattr(postfit, "validate_training_stage", lambda plan: receipt)
    monkeypatch.setattr(postfit, "prepare_evaluation_state", lambda **kw: None)
    monkeypatch.setattr(postfit, "resolve_evaluation_plan", lambda args: {"bad": "plan"})
    monkeypatch.setattr(
        postfit,
        "execute_evaluation_plan",
        lambda plan: (_ for _ in ()).throw(RuntimeError("test worker failed")),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        postfit.run_deferred_postfit(plan)
    assert (root / "TRAIN_COMPLETE.json").exists()
    assert not (root / "COMPLETE.json").exists()
    assert read_json(root / "EVALUATION_FAILURE.json")["training_complete"]
    assert not (root / "FAILURE.json").exists()


def test_only_exact_test_role_can_finalize(trained):
    root, plan, _, _, _, receipt = trained
    path = root / "fit/last-test.json"
    test = {
        "identity": {
            "training": receipt["identity"],
            "checkpoint": receipt["last"],
            "role": "best",
            "evaluation": {"source": {"dirty": False}, "environment": {}},
        },
        "result": {"split": "test"},
    }
    independent = root / "postfit-last/COMPLETE.json"
    atomic_json(
        independent,
        {
            "plan": {
                "training_identity": receipt["identity"],
                "checkpoint_sha256": receipt["last"]["sha256"],
                "checkpoint_role": "last",
                "checkpoint_epoch": 6,
                "training_run_root": str(root.resolve()),
                "split": "test",
                "output_root": str(independent.parent),
                "evaluation_source": {"dirty": False},
            },
            "evaluation_environment": {},
            "zero_pkl": True,
            "result": test["result"],
        },
    )
    test["independent_evaluation_receipt"] = str(independent)
    test["independent_evaluation_sha256"] = sha256_file(independent)
    atomic_json(path, test)
    with pytest.raises(ValueError, match="role"):
        postfit.finalize_deferred_run(plan, receipt)
    test["identity"]["role"] = "last"
    atomic_json(path, test)
    final = postfit.finalize_deferred_run(plan, receipt)
    assert final["epoch"] == 6 and final["best"] is None and final["test_roles"] == ["last"]
    assert final["training_complete_sha256"] == sha256_file(root / "TRAIN_COMPLETE.json")
    test["independent_evaluation_sha256"] = "wrong"
    atomic_json(path, test)
    with pytest.raises(ValueError, match="hash"):
        postfit.finalize_deferred_run(plan, receipt)
    del test["independent_evaluation_receipt"]
    atomic_json(path, test)
    with pytest.raises(ValueError, match="evidence missing"):
        postfit.finalize_deferred_run(plan, receipt)


def test_separate_evaluation_release_does_not_rewrite_training_plan(trained, monkeypatch):
    root, plan, _, _, _, receipt = trained
    plan = {**plan, "data_root": "unused", "runtime": "original-release"}
    original = dict(plan)
    monkeypatch.setattr(postfit, "validate_training_stage", lambda plan: receipt)
    monkeypatch.setattr(postfit, "prepare_evaluation_state", lambda **kw: None)
    seen = {}

    def resolve(args):
        seen["runtime"] = args.runtime
        return {}

    def execute(evaluation_plan):
        seen.update(evaluation_plan)
        raise RuntimeError("stop before writing a scientific receipt")

    monkeypatch.setattr(postfit, "resolve_evaluation_plan", resolve)
    monkeypatch.setattr(postfit, "execute_evaluation_plan", execute)
    with pytest.raises(RuntimeError, match="scientific receipt"):
        postfit.run_deferred_postfit(
            plan, evaluation_runtime=Path("new-evaluator"), cuda_memory_fraction=0.19
        )
    assert seen == {
        "runtime": Path("new-evaluator"),
        "cpu_training_state": True,
        "cuda_memory_fraction": 0.19,
    }
    assert plan == original and (root / "TRAIN_COMPLETE.json").exists()
    assert not (root / "COMPLETE.json").exists()


def test_queue_trains_b_before_evaluating_a(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts/v2"))
    import run_cap40_ablations as queue

    import gradpert.execution.v2_training_stage as stage

    monkeypatch.setattr(Path, "is_relative_to", lambda *args: True)

    @contextmanager
    def lock(path):
        yield 42

    monkeypatch.setattr(queue, "lock", lock)
    monkeypatch.setattr(queue, "require_fresh_formal", lambda plan: None)
    monkeypatch.setattr(queue, "validate_preflight", lambda plan, entry: None)
    monkeypatch.setattr(stage, "validate_training_stage", lambda plan: {"last": "ok"})
    monkeypatch.setattr(queue, "next_action", lambda plan: "skip_complete")
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **k: "0, 0\n1, 0\n")
    events = []

    class Child:
        pid = 123

        def __init__(self, command, **kw):
            if "--output" in command:
                p = Path(command[command.index("--output") + 1])
                p.mkdir()
                atomic_json(p / "receipt.json", {})
                events.append((p.name.split("-")[0], "preflight"))
            else:
                arm = Path(command[-1]).name.split(".")[0]
                events.append((arm, "postfit" if "run_deferred_postfit" in command[2] else "fit"))

        def wait(self):
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    baseline = tmp_path / "base"
    baseline.mkdir()
    atomic_json(baseline / "COMPLETE.json", {})
    d = tmp_path / "queue"
    d.mkdir()
    rows = [
        {
            "name": name,
            "probe_kind": "preflight_only",
            "plan": {
                "run_id": name,
                "repository_root": str(ROOT),
                "source_commit": "a" * 40,
                "config_sha256": "c",
                "seed": 1,
                "postfit_policy": "deferred",
                "config": "unused",
                "data_root": "unused",
                "publication": "unused",
                "publication_sha256": "unused",
            },
        }
        for name in ("A", "B")
    ]
    q = {
        "schema": "synthetic",
        "rows": rows,
        "baseline": str(baseline),
        "baseline_complete_sha256": sha256_file(baseline / "COMPLETE.json"),
        "postfit_policy": "deferred",
    }
    queue.execute(q, d)
    assert events == [
        ("A", "preflight"),
        ("B", "preflight"),
        ("A", "fit"),
        ("B", "fit"),
        ("A", "postfit"),
        ("B", "postfit"),
    ]
    assert (d / "COMPLETE.json").exists()
