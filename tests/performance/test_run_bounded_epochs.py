"""Entry guards and method restoration, independent of CUDA availability."""

import json
from types import SimpleNamespace

import pytest

from scripts.performance import run_bounded_epochs as runner


@pytest.mark.parametrize("scenario", ["test", "unexpected_return", "boundary"])
def test_entry_guards_and_restores_hooks(tmp_path, monkeypatch, scenario):
    import gradpert.execution.identity as identity
    import gradpert.execution.native as native
    from gradpert.training.trainer import GraDPertTrainer

    config = tmp_path / "config.yaml"
    config.write_text("synthetic test config")
    args = SimpleNamespace(
        coordinate="r50_e3_batch512",
        phase="capacity",
        deterministic_algorithms=True,
        capture_exact_state=False,
        checkpoint_roundtrip_after_step=None,
        resume=False,
        stop_before_epoch=1,
        run_root=tmp_path / "run",
        config=config,
        repository_root=tmp_path,
        development_commit="a" * 40,
        run_seed=1,
        device="cpu",
        data_root=tmp_path,
        run_id="unit",
        genept_preflight_receipt=None,
        genept_preflight_receipt_sha256=None,
        minimum_disk_free_bytes=1,
        minimum_host_available_bytes=1,
    )
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("GRADPERT_SPARSE_UNION_IMPL", "cpu_vectorized")
    monkeypatch.setattr(
        runner.profile,
        "_parser",
        lambda: SimpleNamespace(add_argument=lambda *a, **kw: None, parse_args=lambda _: args),
    )
    monkeypatch.setattr(runner.profile, "_validate_thresholds", lambda _: None)
    monkeypatch.setattr(runner.profile, "_coordinate_preflight", lambda _: None)
    monkeypatch.setattr(runner.profile, "_require_reference_a0", lambda *a: (None, {}))
    monkeypatch.setattr(
        runner.profile,
        "_host_snapshot",
        lambda _: {"disk_free_bytes": 2, "host_available_bytes": 2},
    )
    monkeypatch.setattr(runner.profile, "_preflight_predicates", lambda *a: ({"idle": True}, {}))
    monkeypatch.setattr(runner.profile, "_exact_engine_state", lambda _: {"model": "unit"})
    monkeypatch.setattr(
        identity,
        "inspect_source_identity",
        lambda *a, **kw: SimpleNamespace(
            dirty=False, commit="a" * 40, payload=lambda: {"commit": "a" * 40}
        ),
    )
    monkeypatch.setattr(runner, "audit_trainer_boundary", lambda _, e: {"completed_epochs": e})

    def evaluator(*a, **kw):
        return None

    monkeypatch.setattr(native, "CanonicalEvaluationData", evaluator)

    def fake_fit(self, **kw):
        kw["train_epoch_factory"](0)
        kw["train_epoch_factory"](1)

    monkeypatch.setattr(GraDPertTrainer, "fit", fake_fit)
    original_test = GraDPertTrainer.test_best_once

    def fake_native(**kw):
        if scenario == "test":
            native.CanonicalEvaluationData(split_name="test")
        elif scenario == "boundary":
            native.CanonicalEvaluationData(split_name="val")
            trainer = SimpleNamespace(progress=SimpleNamespace(completed_epochs=0), engine=None)
            GraDPertTrainer.fit(trainer, train_epoch_factory=lambda _: ())

    monkeypatch.setattr(native, "run_native_experiment", fake_native)
    result = runner.main([])
    receipt = json.loads((args.run_root / "bounded_evidence/segment-0-to-1.json").read_text())
    assert result == (0 if scenario == "boundary" else 1)
    assert receipt["scientific_completion"] is False
    assert receipt["status"] == ("complete" if scenario == "boundary" else "failed")
    assert receipt["guard"]["test_access_attempts"] == (1 if scenario == "test" else 0)
    assert native.CanonicalEvaluationData is evaluator
    assert GraDPertTrainer.fit is fake_fit
    assert GraDPertTrainer.test_best_once is original_test
