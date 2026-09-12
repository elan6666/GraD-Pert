import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest
import torch

from benchmarks.txpert import runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("smoke", [True, False])
def test_txpert_r50_runner_checkpoint_roles_and_truth_scope(tmp_path, monkeypatch, smoke):
    source = NS(commit="train", dirty=False, formal_eligible=True, payload=lambda: {})
    env = NS(payload_sha256="env", payload=lambda: {})
    monkeypatch.setattr(runner, "inspect_source_identity", lambda *a, **k: source)
    monkeypatch.setattr(runner, "inspect_environment", lambda *a, **k: env)
    config_file = tmp_path / "official.yaml"
    config_file.write_text("{}")
    monkeypatch.setattr(
        runner, "_official_config", lambda *a: (config_file, {"model": {}, "graph": {}})
    )
    monkeypatch.setattr(runner, "load_runtime_contract", lambda **k: (config_file, {}))
    monkeypatch.setattr(runner, "inspect_cuda_runtime", lambda **k: {})
    monkeypatch.setattr(runner, "_register_anndata_null_reader", lambda: {})
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
        lambda *a, **k: NS(receipt={}, adata=NS(obs=pd.DataFrame({"cell_type": ["Jurkat"]}))),
    )
    monkeypatch.setattr(runner, "_write_official_cache", lambda **k: {})
    monkeypatch.setattr(runner, "_remove_official_split_pickles", lambda **k: {})
    model = torch.nn.Linear(1, 1, bias=False)
    calls = []

    class API:
        def prepare_training_data_module(self, **kwargs):
            return NS(pert2id={})

        def normalize_training_perturbation_indices(self, data):
            return {}

        def require_perturbation_coverage(self, *args):
            return ["A", "B", "C"]

        def build_model(self, **kwargs):
            return model

        def restore_post_fit_device(self, model, device):
            return {}

        def fit_with_validation(self, **kwargs):
            assert kwargs["r50"]
            epochs = kwargs["epochs"]
            calls.append(epochs)
            with torch.no_grad():
                model.weight.fill_(50)
            torch.save({"state_dict": model.state_dict()}, kwargs["last_checkpoint_path"])
            with torch.no_grad():
                model.weight.fill_(1)
            torch.save({"state_dict": model.state_dict()}, kwargs["checkpoint_path"])
            return NS(
                gradpert_validation_history=[
                    {"epoch": n, "val_pearson_delta": 0.4} for n in range(1, epochs + 1)
                ],
                gradpert_best_epoch=1,
                gradpert_last_epoch=epochs,
                global_step=epochs,
            )

    monkeypatch.setattr(runner, "OfficialPublicAPI", lambda *a: API())
    monkeypatch.setattr(runner, "OfficialPublicModules", NS(from_mapping=lambda m: m))

    @contextmanager
    def official(**kwargs):
        yield {}, NS(payload=lambda: {})

    monkeypatch.setattr(runner, "official_module_session", official)
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: NS() if name == "lightning" else original_import(name),
    )
    gates = []
    monkeypatch.setattr(runner, "require_r50_smoke", lambda *a, **k: gates.append(1) or {})
    captured = []

    @contextmanager
    def test(**kwargs):
        assert not smoke, "smoke opened test truth"
        yield NS(control_manifest=NS(draws=[]))

    monkeypatch.setattr(runner, "CanonicalEvaluationData", test)

    def seal(**kwargs):
        captured.append(float(model.weight.item()))
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
        config_path=ROOT / "configs/r50-rerun/txpert_public/nadig_jurkat.yaml",
        checkout_root=tmp_path / "official",
        data_root=tmp_path,
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
    assert len(gates) == (0 if smoke else 1)
    assert captured == ([] if smoke else [1.0, 50.0])
    assert result["status"] == ("trained_validation_only" if smoke else "evaluated")
    if not smoke:
        records = json.loads((root / "small_results/best_last_tests.json").read_text())
        assert records["best"]["checkpoint_epoch"] == 1
        assert records["last"]["checkpoint_epoch"] == 50
        assert records["best"]["checkpoint_sha256"] != records["last"]["checkpoint_sha256"]
