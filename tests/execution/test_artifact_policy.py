from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gradpert.config import load_experiment_config
from gradpert.execution import artifact_run

ROOT = Path(__file__).resolve().parents[2]


class _Contract:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self.payload


def _exercise_policy(monkeypatch: object, tmp_path: Path, *, mode: str) -> Path | None:
    config = load_experiment_config(
        ROOT / "configs/experiments/matched_control_mean/nadig_jurkat.yaml"
    )
    config = config.model_copy(
        update={"artifacts": config.artifacts.model_copy(update={"result_mode": mode})}
    )
    control_ids = tuple(f"control-{index % 7}" for index in range(300))

    def fake_prediction(path: Path, **_: object) -> SimpleNamespace:
        path.write_bytes(b"temporary prediction")
        return SimpleNamespace(
            manifest=_Contract({"schema_version": "prediction-artifact-v1"}),
            conditions={"PERT": SimpleNamespace(input_control_row_ids=control_ids)},
        )

    def fake_evaluation(path: Path, **_: object) -> SimpleNamespace:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"temporary or retained result")
        return SimpleNamespace(
            manifest=_Contract({"schema_version": "evaluation-bundle-v1"}),
            file_sha256="a" * 64,
            conditions={"PERT": SimpleNamespace(truth_row_ids=("truth-0", "truth-1"))},
        )

    monkeypatch.setattr(artifact_run, "seal_prediction_artifact", fake_prediction)
    monkeypatch.setattr(artifact_run, "load_evaluation_state", lambda **_: object())
    monkeypatch.setattr(artifact_run, "seal_frozen_evaluation_bundle", fake_evaluation)
    monkeypatch.setattr(artifact_run, "write_small_metric_exports", lambda *_: None)

    training_data = SimpleNamespace(
        layout=SimpleNamespace(root=tmp_path / "canonical", data_root=tmp_path / "data"),
        manifest=SimpleNamespace(
            canonical_adata_sha256="b" * 64,
            expression_gene_order_sha256="c" * 64,
            graph_gene_order_sha256="d" * 64,
        ),
        split=SimpleNamespace(split_content_sha256="e" * 64),
        expression_gene_ids=("G1", "G2"),
    )
    test_data = SimpleNamespace(control_manifest_file_sha256="f" * 64)
    source = SimpleNamespace(
        commit="1" * 40,
        dirty=False,
        formal_eligible=True,
        tree_sha256="2" * 64,
    )
    environment = SimpleNamespace(payload_sha256="3" * 64)
    result = artifact_run.seal_evaluation_outputs(
        destination=tmp_path / "run",
        config=config,
        config_sha256="4" * 64,
        run_id="artifact-policy-test",
        run_seed=1,
        source=source,
        environment=environment,
        training_data=training_data,
        test_data=test_data,
        predictions=[],
        checkpoint_sha256="5" * 64,
    )
    return result.result_pkl_path


def test_metrics_only_leaves_no_pkl(monkeypatch: object, tmp_path: Path) -> None:
    result_path = _exercise_policy(monkeypatch, tmp_path, mode="metrics_only")

    assert result_path is None
    assert list((tmp_path / "run").rglob("*.pkl")) == []
    assert (tmp_path / "run/small_results/inference_recipe.json").is_file()


def test_single_pkl_leaves_exactly_one_result(monkeypatch: object, tmp_path: Path) -> None:
    result_path = _exercise_policy(monkeypatch, tmp_path, mode="single_pkl")

    assert result_path == tmp_path / "run/artifacts/result.pkl"
    assert list((tmp_path / "run").rglob("*.pkl")) == [result_path]
