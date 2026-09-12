import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from benchmarks.gears import runner
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("smoke", [True, False])
def test_gears_r50_real_last_and_no_smoke_test(tmp_path, monkeypatch, smoke):
    source = NS(commit="train", dirty=False, formal_eligible=True, payload=lambda: {})
    env = NS(payload_sha256="env", payload=lambda: {})
    monkeypatch.setattr(runner, "inspect_source_identity", lambda *a, **k: source)
    monkeypatch.setattr(runner, "inspect_environment", lambda *a, **k: env)
    resources = tmp_path / "resources"
    resources.mkdir()
    for name in ("gene2go_all.pkl", "essential_all_data_pert_genes.pkl"):
        (resources / name).write_bytes(b"synthetic")
    policy = NS(
        official_commit="f374e43e197b295016d80395d7a54ddb81cc6769",
        gene2go_resource_sha256=sha256_file(resources / "gene2go_all.pkl"),
        essential_gene_resource_sha256=sha256_file(resources / "essential_all_data_pert_genes.pkl"),
        policy_id="policy",
        excluded_conditions=[],
        model_dump=lambda **k: {},
    )
    monkeypatch.setattr(
        runner, "load_dataset_registry", lambda *a: NS(benchmark_condition_policy=policy)
    )
    data = NS(
        manifest=NS(canonical_adata_sha256="data"),
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
        "build_training_validation_adata",
        lambda *a, **k: NS(adata=None, receipt={}, train_conditions=["A"], val_conditions=["B"]),
    )
    monkeypatch.setattr(runner, "_ensure_official_sparse_expression", lambda *a: {})
    model = NS(best_model=torch.nn.Linear(1, 1, bias=False))
    calls = []

    class API:
        def prepare_training_data(self, **kwargs):
            return NS(gradpert_de_ranking_receipt={})

        def require_perturbation_coverage(self, *args):
            return ["A", "B", "C"]

        def fit_one_epoch(self, **kwargs):
            assert kwargs["r50"]
            epochs = kwargs["epochs"]
            calls.append(epochs)
            best_dir = kwargs["checkpoint_dir"]
            best_dir.mkdir(parents=True)
            with torch.no_grad():
                model.best_model.weight.fill_(50)
            torch.save(model.best_model.state_dict(), kwargs["last_checkpoint_path"])
            with torch.no_grad():
                model.best_model.weight.fill_(1)
            torch.save(model.best_model.state_dict(), best_dir / "model.pt")
            (best_dir / "config.pkl").write_bytes(b"reconstructible")
            model.gradpert_validation_history = [
                {"epoch": n, "best": n == 1, "val_mse_de": float(n)} for n in range(1, epochs + 1)
            ]
            model.gradpert_last_epoch = epochs
            return model

    monkeypatch.setattr(runner, "OfficialGearsAPI", lambda *a: API())
    monkeypatch.setattr(runner, "GearsOfficialModules", NS(from_mapping=lambda m: m))

    @contextmanager
    def official(**kwargs):
        yield {}, NS(payload=lambda: {})

    monkeypatch.setattr(runner, "official_module_session", official)
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: NS() if name == "torch_geometric.loader" else original_import(name),
    )
    gate_calls = []
    monkeypatch.setattr(runner, "require_r50_smoke", lambda *a, **k: gate_calls.append(1) or {})
    test_calls = []

    @contextmanager
    def test(**kwargs):
        assert not smoke
        test_calls.append(1)
        yield NS(control_manifest=NS(draws=[]))

    monkeypatch.setattr(runner, "CanonicalEvaluationData", test)
    captured = []

    def seal(**kwargs):
        captured.append(float(model.best_model.weight.item()))
        small = kwargs["destination"] / "small_results"
        small.mkdir()
        for name in ("run_manifest", "prediction_manifest", "evaluation_manifest"):
            (small / f"{name}.json").write_text("{}")
        return NS(
            run_manifest=NS(status="evaluated", formal_eligible=True),
            prediction_manifest_path=small / "prediction_manifest.json",
            evaluation_manifest_path=small / "evaluation_manifest.json",
        )

    monkeypatch.setattr(runner, "seal_evaluated_run", seal)
    root = tmp_path / "run"
    result = runner.run_one_epoch(
        config_path=ROOT / "configs/r50-rerun/gears/nadig_jurkat.yaml",
        checkout_root=tmp_path / "official",
        data_root=tmp_path,
        official_data_root=resources,
        run_root=root,
        run_id="fresh",
        device="cpu",
        repository_root=ROOT,
        formal=True,
        development_commit=None,
        smoke=smoke,
        smoke_run_root=tmp_path / "smoke",
    )
    assert calls == [1 if smoke else 50]
    assert len(gate_calls) == (0 if smoke else 1)
    assert len(test_calls) == (0 if smoke else 2)
    assert captured == ([] if smoke else [1.0, 50.0])
    assert not list(root.rglob("*.pkl"))
    assert result["status"] == ("trained_validation_only" if smoke else "evaluated")
    if not smoke:
        records = json.loads((root / "small_results/best_last_tests.json").read_text())
        assert records["best"]["checkpoint_epoch"] == 1
        assert records["last"]["checkpoint_epoch"] == 50
        assert records["best"]["checkpoint_sha256"] != records["last"]["checkpoint_sha256"]
