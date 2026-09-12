import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from benchmarks.scouter import runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("smoke", [True, False])
def test_runner_smoke_never_opens_test_and_full_loads_both_states(tmp_path, monkeypatch, smoke):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    source = NS(commit="training-sha", dirty=False, formal_eligible=True, payload=lambda: {})
    env = NS(payload_sha256="environment", payload=lambda: {})
    monkeypatch.setattr(runner, "inspect_source_identity", lambda *a, **k: source)
    monkeypatch.setattr(runner, "inspect_environment", lambda *a, **k: env)
    data = NS(
        manifest=NS(canonical_adata_sha256="canonical"),
        split=NS(
            split_content_sha256="split",
            train_conditions=["A"],
            val_conditions=["B"],
            test_conditions=["C"],
        ),
        require_experiment_data_contract=lambda **k: None,
    )

    @contextmanager
    def training(**kwargs):
        yield data

    monkeypatch.setattr(runner, "CanonicalTrainingData", training)
    monkeypatch.setattr(runner, "write_training_data_receipt", lambda *a: None)
    monkeypatch.setattr(
        runner,
        "verify_text_prior_npz",
        lambda *a, **k: NS(
            source_sha256="prior",
            embedding_width=2,
            gene_order_sha256="genes",
            selected_matrix_sha256="matrix",
        ),
    )
    monkeypatch.setattr(runner, "build_training_validation_adata", lambda *a, **k: NS(receipt={}))
    model = NS(network=torch.nn.Linear(1, 1, bias=False))

    @contextmanager
    def official(**kwargs):
        yield {"scouter": NS(Scouter=lambda *a, **k: model)}, NS(payload=lambda: {})

    monkeypatch.setattr(runner, "official_module_session", official)
    monkeypatch.setattr(runner, "prepare_data", lambda *a: None)
    fit_calls = []

    def fit(m, config, *, epochs, progress_path, r50, last_checkpoint_path):
        fit_calls.append(epochs)
        assert r50
        last_checkpoint_path.parent.mkdir(parents=True)
        with torch.no_grad():
            m.network.weight.fill_(50)
        torch.save(m.network.state_dict(), last_checkpoint_path)
        with torch.no_grad():
            m.network.weight.fill_(1)
        return dict(epochs_completed=epochs, best_epoch=1, last_epoch=epochs, best_val_loss=0.4)

    monkeypatch.setattr(runner, "fit_official", fit)
    gate_calls = []
    monkeypatch.setattr(runner, "require_r50_smoke", lambda *a, **k: gate_calls.append(1) or {})
    test_calls = []

    @contextmanager
    def test(**kwargs):
        assert not smoke, "smoke must never open canonical test truth"
        test_calls.append(1)
        yield NS(control_manifest=NS(draws=[]))

    monkeypatch.setattr(runner, "CanonicalEvaluationData", test)
    captured = []

    def seal(**kwargs):
        captured.append(float(model.network.weight.item()))
        root = kwargs["destination"] / "small_results"
        root.mkdir()
        for name in ("run_manifest", "prediction_manifest", "evaluation_manifest"):
            (root / f"{name}.json").write_text("{}")
        return NS(
            run_manifest=NS(status="evaluated"),
            prediction_manifest_path=root / "prediction_manifest.json",
            evaluation_manifest_path=root / "evaluation_manifest.json",
        )

    monkeypatch.setattr(runner, "seal_evaluated_run", seal)
    args = NS(
        config=ROOT / "configs/r50-rerun/scouter_genept_seed/nadig_jurkat.yaml",
        repository_root=ROOT,
        publication_receipt=tmp_path / "pub.json",
        publication_receipt_sha256="pub",
        environment_lock=tmp_path / "env.json",
        device="cpu",
        run_root=tmp_path / "run",
        data_root=tmp_path,
        smoke=smoke,
        smoke_run_root=tmp_path / "smoke",
        genept_seed=tmp_path / "prior",
        official_checkout=tmp_path / "official",
        run_id="fresh",
    )
    result = runner.run(args)
    assert fit_calls == [1 if smoke else 50]
    assert len(gate_calls) == (0 if smoke else 1)
    assert len(test_calls) == (0 if smoke else 2)
    assert captured == ([] if smoke else [1.0, 50.0])
    assert result["status"] == ("trained_validation_only" if smoke else "evaluated")
    assert (args.run_root / "checkpoints/best.pt").exists()
    assert (args.run_root / "checkpoints/last.pt").exists()
    if not smoke:
        records = json.loads((args.run_root / "small_results/best_last_tests.json").read_text())
        assert set(records) == {"best", "last"}
        assert records["best"]["checkpoint_epoch"] == 1
        assert records["last"]["checkpoint_epoch"] == 50
        assert records["best"]["checkpoint_sha256"] != records["last"]["checkpoint_sha256"]
        for role, record in records.items():
            assert record["checkpoint_role"] == role
            assert record["training_git_sha"] == "training-sha"
            assert record["evaluation_git_sha"] == "training-sha"
