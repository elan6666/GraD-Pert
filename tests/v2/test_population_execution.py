"""CPU-only execution gates reject foreign/mutated population calibrations."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file, sha256_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))

from scripts.v2 import run_response_anchor as runner  # noqa: E402
from scripts.v2.generate_response_anchor import configuration  # noqa: E402

SOURCE = "e" * 40
MANIFEST = "a" * 64
ARRAYS = "b" * 64
POPULATION = "c" * 64
POPULATION_ARRAYS = "d" * 64
POPULATION_PATH = "/data/yilangliu/population/manifest.json"
WEIGHT = 0.05


def _write(path, payload):
    path.write_text(json.dumps(payload))


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    def make(arm="R16-GPM"):
        mode = {
            "R14-FRGA": "mse",
            "R15-DIR": "cosine_direction",
            "R16-GPM": "population_prediction",
        }[arm]
        population = (
            dict(
                population_path=POPULATION_PATH,
                population_sha256=POPULATION,
                population_arrays_sha256=POPULATION_ARRAYS,
            )
            if arm == "R16-GPM"
            else {}
        )
        kwargs = dict(
            bank_path="/data/yilangliu/bank/manifest.json",
            manifest_sha256=MANIFEST,
            arrays_sha256=ARRAYS,
            loss_mode=mode,
            **population,
        )
        probe = tmp_path / "calibration/gradpert_v2/nadig_jurkat.yaml"
        fit = tmp_path / "R16-GPM/gradpert_v2/nadig_jurkat.yaml"
        probe.parent.mkdir(parents=True, exist_ok=True)
        fit.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(yaml.safe_dump(configuration(ROOT, weight=1.0, **kwargs)))
        fit.write_text(yaml.safe_dump(configuration(ROOT, weight=WEIGHT, **kwargs)))
        metadata = [
            dict(
                step=i,
                ordered_identity_sha256=f"{i:064x}",
                query_ids_sha256="1" * 64,
                local_group_indices_sha256=f"{i + 8:064x}",
                global_rows=320,
                local_group_count=4,
            )
            for i in range(8)
        ]
        batches = [
            dict(row, prediction_gradient_norm=1.0, unit_anchor_gradient_norm=2.0)
            for row in metadata
        ]
        calibration = dict(
            calibration_complete=True,
            state_restored=True,
            forward_precision="bfloat16_autocast",
            world_size=2,
            rank=0,
            epoch=0,
            optimizer_updates=0,
            teacher_updates=0,
            center_updates=0,
            batches=batches,
            source=dict(commit=SOURCE, dirty=False, published_commit=SOURCE, formal_eligible=True),
            config_sha256=sha256_file(probe),
            resolved_config_sha256=sha256_json(
                load_experiment_config(probe).model_dump(mode="json")
            ),
            identity=dict(
                run_seed=1,
                response_anchor=dict(
                    manifest_sha256=MANIFEST,
                    arrays_sha256=ARRAYS,
                    weight=1.0,
                    seed=20261028,
                    loss=mode,
                    head="existing_native_prediction_softplus",
                    teacher_readout="absent_no_additional_forward",
                ),
            ),
            bank_manifest_sha256=MANIFEST,
            bank_arrays_sha256=ARRAYS,
            selected_weight=WEIGHT,
            ordered_batch_metadata_sha256=sha256_json(metadata),
            median_prediction_gradient_norm=1.0,
            median_unit_anchor_gradient_norm=2.0,
            anchor_coefficient=300 / 1322,
            rule="min(1,0.1*median(Gpred)/median(Ganchor))",
            parameter_scope="student.response.parameters; excludes response_anchor_readout",
            gradient_bank_dtype="float32",
            norm_accumulation_dtype="float64",
        )
        decision = dict(
            arm=arm,
            response_anchor_loss=mode,
            calibration_source=SOURCE,
            calibration_complete_sha256="f" * 64,
            calibration_forward_precision="bfloat16_autocast",
            selected_weight=WEIGHT,
            rule=calibration["rule"],
            median_prediction_gradient_norm=1.0,
            median_unit_anchor_gradient_norm=2.0,
            bank_manifest_sha256=MANIFEST,
            bank_arrays_sha256=ARRAYS,
            fit_config_sha256=sha256_file(fit),
            training_epochs=2,
            optimizer_updates=300,
            global_batch=320,
        )
        if arm == "R16-GPM":
            calibration.update(
                response_anchor_loss=mode,
                population_manifest_sha256=POPULATION,
                population_arrays_sha256=POPULATION_ARRAYS,
                input_files_sha256={
                    POPULATION_PATH: POPULATION,
                    "/data/yilangliu/population/targets.npz": POPULATION_ARRAYS,
                },
            )
            calibration["identity"]["response_anchor"].update(
                population_manifest_sha256=POPULATION,
                population_arrays_sha256=POPULATION_ARRAYS,
            )
            decision.update(
                population_manifest_sha256=POPULATION,
                population_arrays_sha256=POPULATION_ARRAYS,
            )
        if arm == "R15-DIR":
            calibration["response_anchor_loss"] = mode
            calibration["direction_diagnostics"] = dict(
                response_anchor_loss=mode,
                eps=1e-6,
                target_count=1322,
                target_clamped_count=0,
                head_observation_count=8,
                head_clamped_count=0,
                target_norm_min=1.0,
                target_norm_max=2.0,
                head_norm_min=0.1,
                head_norm_max=1.0,
            )
        decision_path = tmp_path / "DECISION.json"
        _write(decision_path, decision)
        monkeypatch.setattr(runner, "recipe", lambda root, selected: (probe, decision_path, mode))
        return fit, calibration, decision_path

    return make


def test_population_recipe_is_explicit_and_historical_paths_unchanged():
    for arm, family, mode in [
        ("R14-FRGA", "response_anchor_20261028", "mse"),
        ("R15-DIR", "response_anchor_direction_20261029", "cosine_direction"),
        ("R16-GPM", "response_population_prediction_20261029", "population_prediction"),
    ]:
        probe, decision, actual = runner.recipe(ROOT, arm)
        assert probe == ROOT / f"configs/v2/{family}/calibration/gradpert_v2/nadig_jurkat.yaml"
        assert decision == ROOT / f"configs/v2/{family}/CALIBRATION_DECISION.json"
        assert actual == mode
    with pytest.raises(ValueError, match="unknown"):
        runner.recipe(ROOT, "R16")


def test_population_accepts_own_first8_without_reading_any_sidecar(sealed):
    fit, calibration, _ = sealed()
    # No numerical sidecar/bank exists on this CPU host. Only sealed scalar IDs are consumed.
    assert not Path(POPULATION_PATH).exists()
    runner.validate_calibration(fit, calibration, "R16-GPM")


@pytest.mark.parametrize("arm", ["R14-FRGA", "R15-DIR"])
def test_historical_calibration_does_not_require_population_metadata(sealed, arm):
    fit, calibration, _ = sealed(arm)
    # Keep the historical gate's behavior: no newly imposed R16-only first8/decision fields.
    calibration["batches"] = [{} for _ in range(8)]
    for key in ["epoch", "rank", "rule", "parameter_scope", "input_files_sha256"]:
        calibration.pop(key, None)
    runner.validate_calibration(fit, calibration, arm)


@pytest.mark.parametrize("mode", ["mse", "cosine_direction"])
def test_population_arm_rejects_wrong_configured_mode(sealed, mode):
    fit, calibration, _ = sealed()
    raw = yaml.safe_load(fit.read_text())
    raw["model"]["parameters"]["response_anchor_loss"]["value"] = mode
    fit.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="configured objective"):
        runner.validate_calibration(fit, calibration, "R16-GPM")


@pytest.mark.parametrize("legacy_mode", [None, "mse", "cosine_direction"])
def test_old_calibration_cannot_supply_population_weight(sealed, legacy_mode):
    fit, calibration, _ = sealed()
    calibration["response_anchor_loss"] = legacy_mode
    calibration.pop("population_manifest_sha256")
    with pytest.raises(ValueError, match="own complete initialization"):
        runner.validate_calibration(fit, calibration, "R16-GPM")


@pytest.mark.parametrize(
    "where,key",
    [
        ("calibration", "population_manifest_sha256"),
        ("calibration", "population_arrays_sha256"),
        ("identity", "population_manifest_sha256"),
        ("identity", "population_arrays_sha256"),
        ("decision", "population_manifest_sha256"),
        ("decision", "population_arrays_sha256"),
        ("identity", "manifest_sha256"),
        ("identity", "arrays_sha256"),
    ],
)
def test_mismatched_sidecar_or_sampling_bank_identity_rejected(sealed, where, key):
    fit, calibration, decision_path = sealed()
    decision = json.loads(decision_path.read_text())
    target = {
        "calibration": calibration,
        "identity": calibration["identity"]["response_anchor"],
        "decision": decision,
    }[where]
    target[key] = "9" * 64
    _write(decision_path, decision)
    with pytest.raises(ValueError, match="hashes differ"):
        runner.validate_calibration(fit, calibration, "R16-GPM")


@pytest.mark.parametrize(
    "case", ["missing_manifest", "missing_arrays", "outside_sidecar", "wrong_path"]
)
def test_sidecar_file_audit_required_in_own_calibration(sealed, case):
    fit, calibration, _ = sealed()
    files = calibration["input_files_sha256"]
    if case == "missing_manifest":
        del files[POPULATION_PATH]
    elif case == "missing_arrays":
        del files["/data/yilangliu/population/targets.npz"]
    elif case == "outside_sidecar":
        files["/data/yilangliu/other/targets.npz"] = files.pop(
            "/data/yilangliu/population/targets.npz"
        )
    else:
        files["/data/yilangliu/foreign/manifest.json"] = files.pop(POPULATION_PATH)
    with pytest.raises(ValueError, match="own complete initialization"):
        runner.validate_calibration(fit, calibration, "R16-GPM")


@pytest.mark.parametrize(
    "case",
    [
        "wrong_order",
        "duplicate_identity",
        "query_identity",
        "metadata_sha",
        "nonfinite",
        "negative",
    ],
)
def test_population_requires_exact_first8_finite_gradient_receipt(sealed, case):
    fit, calibration, _ = sealed()
    if case == "wrong_order":
        calibration["batches"] = list(reversed(calibration["batches"]))
    elif case == "duplicate_identity":
        calibration["batches"][7]["ordered_identity_sha256"] = calibration["batches"][0][
            "ordered_identity_sha256"
        ]
    elif case == "query_identity":
        calibration["batches"][0].pop("query_ids_sha256")
    elif case == "metadata_sha":
        calibration["ordered_batch_metadata_sha256"] = "9" * 64
    elif case == "nonfinite":
        calibration["batches"][0]["unit_anchor_gradient_norm"] = float("nan")
    else:
        calibration["batches"][0]["prediction_gradient_norm"] = -1
    with pytest.raises(ValueError, match=r"first eight|first-eight"):
        runner.validate_calibration(fit, calibration, "R16-GPM")


@pytest.mark.parametrize(
    "key,value",
    [
        ("state_restored", False),
        ("world_size", 1),
        ("forward_precision", "float32"),
        ("optimizer_updates", 1),
        ("teacher_updates", 1),
        ("center_updates", 1),
        ("selected_weight", float("nan")),
        ("parameter_scope", "student.parameters"),
    ],
)
def test_population_rejects_wrong_runtime_or_gradient_scope(sealed, key, value):
    fit, calibration, _ = sealed()
    calibration[key] = value
    with pytest.raises(ValueError):
        runner.validate_calibration(fit, calibration, "R16-GPM")


def test_edited_fit_yaml_cannot_keep_previous_calibration_decision(sealed):
    fit, calibration, decision_path = sealed()
    before = copy.deepcopy(calibration)
    raw = yaml.safe_load(fit.read_text())
    raw["model"]["parameters"]["response_anchor_population_path"]["value"] = (
        "/data/yilangliu/foreign/manifest.json"
    )
    fit.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        runner.validate_calibration(fit, calibration, "R16-GPM")
    assert calibration == before
    assert json.loads(decision_path.read_text())["fit_config_sha256"] != sha256_file(fit)


def test_population_rejects_relabelled_old_readout_calibration(sealed):
    fit, calibration, _ = sealed()
    calibration["identity"]["response_anchor"]["head"] = (
        "Linear256_256_default_random_bias_true_private_rng"
    )
    with pytest.raises(ValueError, match="own complete initialization"):
        runner.validate_calibration(fit, calibration, "R16-GPM")


def test_historical_arm_rejects_population_fields_in_config(sealed):
    fit, calibration, _ = sealed("R14-FRGA")
    raw = yaml.safe_load(fit.read_text())
    raw["model"]["parameters"]["response_anchor_population_path"] = {
        "value": POPULATION_PATH,
        "source": "project_preregistered",
        "reference": "synthetic forbidden historical field",
    }
    fit.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="single calibrated"):
        runner.validate_calibration(fit, calibration)


def test_population_unit_calibration_weight_differs_from_selected_fit_weight(sealed):
    fit, calibration, _ = sealed()
    assert calibration["selected_weight"] != 1.0
    calibration["identity"]["response_anchor"]["weight"] = calibration["selected_weight"]
    with pytest.raises(ValueError, match="own complete initialization"):
        runner.validate_calibration(fit, calibration, "R16-GPM")


def test_population_rejects_changed_probe_even_if_its_receipt_hash_is_updated(sealed):
    fit, calibration, _ = sealed()
    probe, _, _ = runner.recipe(ROOT, "R16-GPM")
    raw = yaml.safe_load(probe.read_text())
    raw["model"]["parameters"]["response_anchor_weight"]["value"] = 0.5
    probe.write_text(yaml.safe_dump(raw))
    calibration["config_sha256"] = sha256_file(probe)
    calibration["resolved_config_sha256"] = sha256_json(
        load_experiment_config(probe).model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="unit-weight native probe"):
        runner.validate_calibration(fit, calibration, "R16-GPM")
