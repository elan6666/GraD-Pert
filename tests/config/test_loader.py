from __future__ import annotations

from pathlib import Path

import pytest

from gradpert.config.loader import load_experiment_config
from gradpert.config.matrix import verify_config_matrix

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "experiments"


def test_matrix_is_exact_and_valid() -> None:
    report = verify_config_matrix(CONFIG_ROOT)
    assert report["count"] == 30
    assert report["expected_count"] == 30
    assert len({entry["sha256"] for entry in report["entries"]}) == 30


def test_each_config_carries_its_own_execution_policy() -> None:
    for path in sorted(CONFIG_ROOT.glob("*/*.yaml")):
        config = load_experiment_config(path)
        assert config.artifacts.result_mode == "metrics_only"
        assert config.artifacts.result_pkl_name == "result.pkl"
        assert config.artifacts.inference_recipe_schema_version == "inference-recipe-v1"
        if config.model_id == "gradpert_b2":
            assert config.model.parameters["prototype_count"].value == 16384
            assert config.training.smoke_epochs.value == 1
            assert config.training.formal_run_policy == "smoke_then_full"
            assert config.training.max_epochs.value == 100
            assert config.training.train_batch_size.value == 256
            assert config.training.eval_batch_size.value == 256
            assert (
                config.model.parameters["cuda_allocator_config"].value == "expandable_segments:True"
            )
        elif config.model_id in {"gears", "txpert_public"}:
            assert config.training.smoke_epochs.value == 1
            assert config.training.formal_run_policy == "smoke_only"
            assert config.training.max_epochs.value == 1
            assert config.training.run_seeds == [1]
            assert not config.training.early_stopping
        else:
            assert config.training.smoke_epochs.value == 0
            assert config.training.formal_run_policy == "inference_only"


def test_external_configs_pin_official_package_and_values() -> None:
    adapter_parameters = {
        "architecture_profile",
        "custom_split_source",
        "prediction_adapter_average_controls",
        "retain_per_control_predictions",
    }
    for model_id in ("gears", "txpert_public"):
        for path in sorted((CONFIG_ROOT / model_id).glob("*.yaml")):
            config = load_experiment_config(path)
            assert config.source_code.execution == "isolated"
            assert (
                config.model.implementation
                == f"benchmarks.{model_id.removesuffix('_public')}.runner"
            )
            for name, value in config.model.parameters.items():
                if name not in adapter_parameters:
                    assert value.source == "official", (path, name, value.source)
            if model_id == "txpert_public":
                assert (
                    config.model.parameters["official_config_file"].value
                    == "configs/config-exphormer-mg.yaml"
                )
                assert (
                    config.model.parameters["official_config_sha256"].value
                    == "2991e704496809979a47c4fade782698f9e465f593e0bc7fb87412f0c416df21"
                )
            for name in (
                "train_batch_size",
                "eval_batch_size",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "scheduler",
            ):
                assert getattr(config.training, name).source == "official", (path, name)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("base: &base {x: 1}\ncopy: *base\n", "anchors and aliases"),
        ("defaults: [global]\n", "defaults key"),
        ("base: {x: 1}\ncopy:\n  <<: {x: 1}\n", "merge keys"),
    ],
)
def test_loader_rejects_hidden_composition(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "model" / "dataset.yaml"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("max_epochs:\n    value: 100", "max_epochs:\n    value: 101", "max_epochs=100"),
        (
            "n_controls_per_condition: 300",
            "n_controls_per_condition: 299",
            "300 controls",
        ),
        ("  min_delta: 0.0\n", "  min_delta: 0.0\n  unknown_option: true\n", "extra"),
    ],
)
def test_loader_rejects_protocol_drift(
    tmp_path: Path,
    needle: str,
    replacement: str,
    message: str,
) -> None:
    source = CONFIG_ROOT / "gradpert_b2" / "replogle_k562_essential.yaml"
    text = source.read_text(encoding="utf-8").replace(needle, replacement)
    path = tmp_path / "gradpert_b2" / "replogle_k562_essential.yaml"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


def test_loader_rejects_identity_path_mismatch(tmp_path: Path) -> None:
    source = CONFIG_ROOT / "gradpert_b2" / "replogle_k562_essential.yaml"
    path = tmp_path / "gears" / "replogle_k562_essential.yaml"
    path.parent.mkdir()
    path.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="identity/path mismatch"):
        load_experiment_config(path)


def test_single_pkl_is_an_explicit_valid_opt_in(tmp_path: Path) -> None:
    source = CONFIG_ROOT / "gradpert_b2" / "replogle_k562_essential.yaml"
    path = tmp_path / "gradpert_b2" / "replogle_k562_essential.yaml"
    path.parent.mkdir()
    path.write_text(
        source.read_text(encoding="utf-8").replace(
            "result_mode: metrics_only", "result_mode: single_pkl"
        ),
        encoding="utf-8",
    )
    assert load_experiment_config(path).artifacts.result_mode == "single_pkl"
