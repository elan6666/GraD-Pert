"""Scalar-only R18 calibration/fit sealing rejects historical or tampered receipts."""

import copy
import json

import pytest
import yaml
from test_population_execution import sealed as sealed

from gradpert.config import load_experiment_config
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.response_anchor import PAIR_MODE, pair_schedule, pair_schedule_identity
from scripts.v2 import run_response_anchor as runner


@pytest.fixture
def pair_sealed(sealed, monkeypatch):
    fit, c, decision_path = sealed()
    probe = fit.parents[2] / "calibration/gradpert_v2/nadig_jurkat.yaml"
    for path in (fit, probe):
        value = yaml.safe_load(path.read_text())
        value["model"]["parameters"]["response_anchor_loss"]["value"] = PAIR_MODE
        value["model"]["parameters"]["response_anchor_loss"]["reference"] = (
            "R18-CPD signed complete-condition native prediction contrast"
        )
        value["model"]["parameters"]["response_anchor_prediction_head_grad"] = {
            "value": True,
            "source": "project_preregistered",
            "reference": "R18-CPD signed complete-condition native prediction contrast",
        }
        path.write_text(yaml.safe_dump(value))
    schedule = pair_schedule_identity(1322, 150, 20261028)
    c.update(
        response_anchor_loss=PAIR_MODE,
        config_sha256=sha256_file(probe),
        resolved_config_sha256=sha256_json(load_experiment_config(probe).model_dump(mode="json")),
        anchor_coefficient=300 / 661,
        pair_schedule=schedule,
        response_anchor_prediction_head_grad=True,
    )
    c["identity"]["response_anchor"].update(
        loss=PAIR_MODE, pair_schedule=schedule, prediction_head_grad=True
    )
    for step, row in enumerate(c["batches"]):
        units = pair_schedule(1322, 150, 0, 20261028, 2, 0)[step]
        row.update(
            local_group_count=len(units), local_group_indices_sha256=sha256_json(list(units))
        )
    metadata = [
        {
            k: v
            for k, v in row.items()
            if k not in ("prediction_gradient_norm", "unit_anchor_gradient_norm")
        }
        for row in c["batches"]
    ]
    c["ordered_batch_metadata_sha256"] = sha256_json(metadata)
    c["pair_diagnostics_ranks"] = []
    c["pair_replay_audit_ranks"] = []
    for rank in range(2):
        rows = []
        for step in range(8):
            for pair in pair_schedule(1322, 150, 0, 20261028, 2, rank)[step]:
                rows.append(
                    dict(
                        step=step,
                        pair=list(pair),
                        query_ids_sha256="1" * 64,
                        group_rows=[40, 40],
                        target_contrast_rms=1.0,
                        prediction_contrast_rms=0.5,
                        contrast_error_rms=0.5,
                        response=dict(
                            pair_gradient_norm=1.0,
                            absolute_gradient_norm=1.0,
                            absolute_pair_gradient_cosine=0.3,
                        ),
                        prediction=dict(
                            pair_gradient_norm=1.0,
                            absolute_gradient_norm=1.0,
                            absolute_pair_gradient_cosine=0.4,
                        ),
                    )
                )
        c["pair_diagnostics_ranks"].append(rows)
        c["pair_replay_audit_ranks"].append(
            dict(
                schema="population-pair-replay-audit-v1",
                pair=rows[0]["pair"],
                query_ids_sha256="1" * 64,
                epoch=0,
                complete=True,
                state_restored=True,
                replays=2,
                extra_group_forwards=4,
                included_in_calibration_norms=False,
                optimizer_updates=0,
                teacher_updates=0,
                center_updates=0,
                gradient_comparison=dict(
                    atol=2e-6,
                    rtol=0,
                    tensor_count=1,
                    missing_mismatch=0,
                    shape_mismatch=0,
                    nonfinite_tensors=0,
                    max_abs=0,
                ),
                deterministic_audit=dict(
                    cuda=True,
                    inside=dict(enabled=True, warn_only=False),
                    before=dict(enabled=False, warn_only=False),
                    restored=dict(enabled=False, warn_only=False),
                    settings_restored=True,
                    cublas_workspace_config=":4096:8",
                ),
            )
        )
    decision = json.loads(decision_path.read_text())
    decision.update(
        arm="R18-CPD", response_anchor_loss=PAIR_MODE, fit_config_sha256=sha256_file(fit)
    )
    decision_path.write_text(json.dumps(decision))
    monkeypatch.setattr(runner, "recipe", lambda root, arm: (probe, decision_path, PAIR_MODE))
    return fit, c


def test_r18_sealed_calibration_accepts_exact_first8_pair_receipt(pair_sealed):
    runner.validate_calibration(*pair_sealed, arm="R18-CPD")


@pytest.mark.parametrize(
    "mutation",
    [
        "old_loss",
        "old_coefficient",
        "pair",
        "epoch",
        "query",
        "schedule",
        "rankmissing",
        "normzero",
        "batchpairhash",
        "headfalse",
    ],
)
def test_r18_calibration_tampering_rejected(pair_sealed, mutation):
    fit, c = pair_sealed
    c = copy.deepcopy(c)
    if mutation == "old_loss":
        c["response_anchor_loss"] = "population_prediction"
    elif mutation == "old_coefficient":
        c["anchor_coefficient"] = 300 / 1322
    elif mutation == "pair":
        c["pair_replay_audit_ranks"][0]["pair"].reverse()
    elif mutation == "epoch":
        c["pair_replay_audit_ranks"][0]["epoch"] = 1
    elif mutation == "query":
        c["pair_replay_audit_ranks"][0]["query_ids_sha256"] = "9" * 64
    elif mutation == "schedule":
        c["pair_schedule"]["schedule_sha256"] = "9" * 64
    elif mutation == "rankmissing":
        c["pair_diagnostics_ranks"].pop()
    elif mutation == "normzero":
        c["pair_diagnostics_ranks"][0][0]["prediction"]["pair_gradient_norm"] = 0
    elif mutation == "batchpairhash":
        c["batches"][0]["local_group_indices_sha256"] = "9" * 64
    else:
        c["response_anchor_prediction_head_grad"] = False
    with pytest.raises(ValueError):
        runner.validate_calibration(fit, c, arm="R18-CPD")
