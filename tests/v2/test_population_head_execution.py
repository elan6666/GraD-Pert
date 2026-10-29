"""R17 recipe sealing with synthetic scalar calibration receipts only."""

import copy
import json
from pathlib import Path

import pytest
import yaml
from test_population_execution import sealed as _sealed
from test_population_head_calibration import native

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2 import run_response_anchor as runner
from scripts.v2.calibrate_response_anchor import audit_population_head_route
from scripts.v2.generate_response_anchor import configuration

ROOT = Path(__file__).resolve().parents[2]
ARM = "R17-GPM-HG0"
ROUTE = "gpm_head_parameters_detached_preserve_input_jacobian"
FAMILY = "response_population_head_protection_20261030"
sealed = _sealed


@pytest.fixture(scope="module")
def audit():
    runtime, controller, batch = native()
    result = audit_population_head_route(
        runtime, controller, batch, 0, bf16=False, world=1, global_rows=len(batch.control)
    )
    # Synthetic server receipt shape; native CPU comparisons above were exact.
    for row in result["gradient_comparisons"].values():
        row["atol"] = 2e-6
    # Contract fixture only: actual CUDA execution is a separate native gate.
    result["deterministic_audit"] = dict(
        scope="route_audit_only_not_norms_or_fit",
        cuda=True,
        cublas_workspace_config=":4096:8",
        before=dict(enabled=False, warn_only=False),
        inside=dict(enabled=True, warn_only=False),
        restored=dict(enabled=False, warn_only=False),
        settings_restored=True,
    )
    result["group_replays"] = 5
    result["additional_audit_replays"] = 2
    result["repeatability_comparisons"] = {
        route: copy.deepcopy(result["gradient_comparisons"])
        for route in ("true_true", "false_false")
    }
    return result


@pytest.fixture
def protected(sealed, audit):
    fit, calibration, decision_path = sealed()
    probe, _, _ = runner.recipe(ROOT, ARM)
    for path in (fit, probe):
        raw = yaml.safe_load(path.read_text())
        raw["model"]["parameters"]["response_anchor_prediction_head_grad"] = dict(
            value=False,
            source="project_preregistered",
            reference="docs/experiments/RESPONSE_POPULATION_HEAD_PROTECTION_20261030.md",
        )
        raw["model"]["parameters"]["response_anchor_weight"]["reference"] = (
            "docs/experiments/RESPONSE_POPULATION_HEAD_PROTECTION_20261030.md#first8-calibration-only"
        )
        path.write_text(yaml.safe_dump(raw))
    calibration.update(
        response_anchor_prediction_head_grad=False,
        gradient_routing=ROUTE,
        population_head_route_audit=copy.deepcopy(audit),
        population_head_route_audit_ranks=[copy.deepcopy(audit), copy.deepcopy(audit)],
        config_sha256=sha256_file(probe),
        resolved_config_sha256=sha256_json(load_experiment_config(probe).model_dump(mode="json")),
    )
    calibration["identity"]["response_anchor"].update(
        prediction_head_grad=False,
        gradient_routing=ROUTE,
    )
    decision = json.loads(decision_path.read_text())
    decision.update(arm=ARM, fit_config_sha256=sha256_file(fit))
    decision_path.write_text(json.dumps(decision))
    return fit, calibration, decision_path


def test_explicit_r17_recipe_family():
    probe, decision, mode = runner.recipe(ROOT, ARM)
    assert probe == ROOT / f"configs/v2/{FAMILY}/calibration/gradpert_v2/nadig_jurkat.yaml"
    assert decision == ROOT / f"configs/v2/{FAMILY}/CALIBRATION_DECISION.json"
    assert mode == "population_prediction"


def kwargs(raw):
    p = raw["model"]["parameters"]
    return dict(
        bank_path=p["response_anchor_path"]["value"],
        manifest_sha256=p["response_anchor_sha256"]["value"],
        arrays_sha256=p["response_anchor_arrays_sha256"]["value"],
        weight=p["response_anchor_weight"]["value"],
        loss_mode="population_prediction",
        population_path=p["response_anchor_population_path"]["value"],
        population_sha256=p["response_anchor_population_sha256"]["value"],
        population_arrays_sha256=p["response_anchor_population_arrays_sha256"]["value"],
    )


def test_generator_preserves_default_r16_and_exact_native_r17_unit_dict():
    r16 = yaml.safe_load(
        (
            ROOT
            / (
                "configs/v2/response_population_prediction_20261029/R16-GPM/"
                "gradpert_v2/nadig_jurkat.yaml"
            )
        ).read_text()
    )
    assert configuration(ROOT, **kwargs(r16)) == r16
    assert configuration(ROOT, prediction_head_grad=True, **kwargs(r16)) == r16
    r17 = yaml.safe_load(
        (ROOT / (f"configs/v2/{FAMILY}/{ARM}/gradpert_v2/nadig_jurkat.yaml")).read_text()
    )
    assert configuration(ROOT, prediction_head_grad=False, **kwargs(r17)) == r17


@pytest.mark.parametrize("value", [None, 0, 1, "False"])
def test_generator_rejects_nonboolean_optin(value):
    with pytest.raises(ValueError, match="boolean"):
        configuration(
            ROOT,
            bank_path="x",
            manifest_sha256="a",
            arrays_sha256="b",
            weight=1,
            loss_mode="population_prediction",
            prediction_head_grad=value,
        )


@pytest.mark.parametrize("mode", ["mse", "cosine_direction"])
def test_generator_false_requires_population(mode):
    with pytest.raises(ValueError, match="requires population_prediction"):
        configuration(
            ROOT,
            bank_path="x",
            manifest_sha256="a",
            arrays_sha256="b",
            weight=1,
            loss_mode=mode,
            prediction_head_grad=False,
        )


def test_r17_accepts_own_first_eight_and_preserves_calibration_source(protected):
    fit, calibration, decision_path = protected
    original = copy.deepcopy(calibration)
    runner.validate_calibration(fit, calibration, ARM)
    generated_kwargs = runner.validate_population_calibration(
        fit,
        yaml.safe_load(fit.read_text())["model"]["parameters"],
        calibration,
        json.loads(decision_path.read_text()),
        arm=ARM,
    )
    assert generated_kwargs["prediction_head_grad"] is False
    assert calibration == original
    assert calibration["source"]["commit"] == "e" * 40


def test_true_r17_and_false_r16_are_rejected(sealed, protected):
    fit, calibration, _ = protected
    with pytest.raises(ValueError, match="head gradient route"):
        runner.validate_calibration(fit, calibration, "R16-GPM")
    fit, calibration, _ = sealed()
    with pytest.raises(ValueError, match="head gradient route"):
        runner.validate_calibration(fit, calibration, ARM)


@pytest.mark.parametrize(
    "case",
    [
        "decision_arm",
        "identity_flag",
        "identity_route",
        "native_flag",
        "native_route",
        "missing_rank",
        "missing_audits",
        "audit_failure",
        "gradient_tolerance",
    ],
)
def test_r17_rejects_foreign_or_incomplete_route_receipts(protected, case):
    fit, calibration, decision_path = protected
    if case == "decision_arm":
        decision = json.loads(decision_path.read_text())
        decision["arm"] = "R16-GPM"
        decision_path.write_text(json.dumps(decision))
    elif case == "identity_flag":
        calibration["identity"]["response_anchor"]["prediction_head_grad"] = True
    elif case == "identity_route":
        calibration["identity"]["response_anchor"]["gradient_routing"] = "wrong"
    elif case == "native_flag":
        calibration["response_anchor_prediction_head_grad"] = True
    elif case == "native_route":
        calibration["gradient_routing"] = "wrong"
    elif case == "missing_rank":
        calibration["population_head_route_audit_ranks"].pop()
    elif case == "missing_audits":
        calibration.pop("population_head_route_audit_ranks")
    elif case == "audit_failure":
        calibration["population_head_route_audit_ranks"][1]["primary_head_gradient_preserved"] = (
            False
        )
    else:
        calibration["population_head_route_audit_ranks"][1]["gradient_comparisons"]["response"][
            "max_abs"
        ] = 3e-6
    with pytest.raises(ValueError):
        runner.validate_calibration(fit, calibration, ARM)


@pytest.mark.parametrize("case", ["first_eight", "weight_rule", "source", "state", "probe"])
def test_r17_retains_original_population_gates(protected, case):
    fit, calibration, _ = protected
    if case == "first_eight":
        calibration["batches"].reverse()
    elif case == "weight_rule":
        calibration["median_prediction_gradient_norm"] = 9.0
    elif case == "source":
        calibration["source"]["published_commit"] = "9" * 40
    elif case == "state":
        calibration["state_restored"] = False
    else:
        probe, _, _ = runner.recipe(ROOT, ARM)
        probe.write_text(probe.read_text() + "\n# byte mutation\n")
    with pytest.raises(ValueError):
        runner.validate_calibration(fit, calibration, ARM)


def test_r16_cannot_disguise_protected_receipt_with_true_fit_config(sealed):
    fit, calibration, _ = sealed()
    calibration["identity"]["response_anchor"]["prediction_head_grad"] = False
    with pytest.raises(ValueError, match="cannot reuse a protected"):
        runner.validate_calibration(fit, calibration, "R16-GPM")
