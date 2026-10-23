"""Diagnostic coding-rate and opt-in trained-AU replay contract."""

import copy
import itertools
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/v2"))

from calibrate_alignment_uniformity import (
    MECHANISM_PROFILE,
    calibrate_batch,
    negative_coding_rate,
    preserved_observer,
)
from capacity_probe import validation_state
from test_au_calibration import runtime_fixture


def test_coding_rate_value_invariances_and_finite_gradient():
    x = torch.eye(3, dtype=torch.float64, requires_grad=True)
    expected = -1.5 * math.log(21)
    torch.testing.assert_close(negative_coding_rate(x, x), x.new_tensor(expected))
    torch.manual_seed(18)
    x = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    value = negative_coding_rate(x, x.flip(0))
    torch.testing.assert_close(value, negative_coding_rate(x @ q * 8, x * 0.3))
    assert torch.isfinite(torch.autograd.grad(value, x)[0]).all()
    with pytest.raises(ValueError, match="nonzero"):
        negative_coding_rate(torch.zeros(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match=r"0\.05"):
        negative_coding_rate(x, x, 0.05**2)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_trained_au_profile_retained_graph_matches_replay(checkpointed):
    from gradpert.training.v2.engine import slice_cells
    from gradpert.training.v2.reductions import population_weights

    torch.set_num_threads(1)
    runtime, batch = runtime_fixture(checkpointed)
    obj = runtime.objective
    obj.au_enabled = True
    obj.au_alignment_weight = 0.002
    obj.au_uniformity_weight = 0.001
    before = validation_state(runtime)
    config = (obj.au_enabled, obj.au_alignment_weight, obj.au_uniformity_weight)
    with preserved_observer(runtime, 912):
        result, banks = calibrate_batch(
            runtime,
            batch,
            microbatch=2,
            bf16=False,
            profile=MECHANISM_PROFILE,
            return_gradients=True,
        )
    assert validation_state(runtime) == before
    assert config == (obj.au_enabled, obj.au_alignment_weight, obj.au_uniformity_weight)
    assert set(result["losses"]) == {"A", "U", "CR", "MSE", "joint", "SSL1", "SSL2"}
    assert set(result["gradients"]) == {"graph", "response", "cell", "prediction", "response_cls"}
    params = [
        p
        for name, p in obj.student.named_parameters()
        if p.requires_grad
        and (
            name == "response_cls"
            or name.startswith(("response.", "graph.", "cell.", "prediction."))
        )
    ]
    with preserved_observer(runtime, 912):
        ids = batch.condition_index
        row = population_weights(ids, torch.ones_like(ids, dtype=torch.bool), "row_mean")
        valid = [v.mask.any(-1) for v in batch.cell_views[:2]]
        active = max(1, sum(bool(v.any()) for v in valid))
        nodes = torch.stack([population_weights(ids, v, "row_mean") / active for v in valid])
        collected, joints, mses = [], [], []
        for start in (0, 2):
            joint, terms = obj(
                slice_cells(batch, start, start + 2),
                include_ssl1=False,
                reduction_weights=(row[start : start + 2], nodes[:, start : start + 2]),
                prediction_weights=row[start : start + 2],
                row_weights=row[start : start + 2],
                deferred_au=collected,
                checkpoint_deferred_response=True,
            )
            joints.append(joint)
            mses.append(terms["prediction_mse"])
        z = [torch.cat([chunk[v] for chunk in collected]) for v in (0, 1)]
        graph_terms = obj.graph_loss(
            batch.graph_views,
            batch.condition_index,
            control=batch.control,
            query_gene_ids=batch.graph.ids[batch.query_positions],
        )
        ssl1 = obj.lambda1 * sum(
            w * v for w, v in zip(obj.weights[0], graph_terms.values(), strict=True)
        )
        scalars = dict(CR=negative_coding_rate(*z), SSL1=ssl1, SSL2=sum(joints) - sum(mses))
        for label, scalar in scalars.items():
            assert result["losses"][label] == pytest.approx(float(scalar.detach()), abs=2e-6)
            targets = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
            for actual, target in zip(banks[label], targets, strict=True):
                torch.testing.assert_close(
                    actual,
                    torch.zeros_like(actual) if target is None else target,
                    rtol=2e-5,
                    atol=2e-6,
                )
    assert validation_state(runtime) == before


def test_profile_receipt_rejects_missing_extra_and_wrong_profile():
    from calibrate_alignment_uniformity import MECHANISM_SCOPE, validate_batch_receipt
    from test_au_calibration import complete_receipt

    record, census = complete_receipt()
    record.update(profile=MECHANISM_PROFILE, loss_scope=MECHANISM_SCOPE)
    record["losses"].update(CR=-2.0, SSL1=1.0, SSL2=2.0)
    record["gradients"] = {
        group: dict(
            norms={key: 1.0 for key in record["losses"]},
            pairs={
                a + "__" + b: dict(dot=0.0, cosine=0.0, angle_degrees=90.0)
                for a, b in itertools.combinations(sorted(record["losses"]), 2)
            },
            parameter_count=1,
            inactive_parameters={key: [] for key in record["losses"]},
            zero_gradient_parameters={key: [] for key in record["losses"]},
        )
        for group in ("graph", "response", "cell", "prediction", "response_cls")
    }
    validate_batch_receipt(
        record, index=0, plan_hash="plan1", census=census, profile=MECHANISM_PROFILE
    )
    for section, key in (("losses", "CR"), ("gradients", "cell")):
        bad = copy.deepcopy(record)
        bad[section].pop(key)
        with pytest.raises(ValueError):
            validate_batch_receipt(
                bad, index=0, plan_hash="plan1", census=census, profile=MECHANISM_PROFILE
            )
        bad = copy.deepcopy(record)
        bad[section]["extra"] = 0
        with pytest.raises(ValueError):
            validate_batch_receipt(
                bad, index=0, plan_hash="plan1", census=census, profile=MECHANISM_PROFILE
            )
    with pytest.raises(ValueError, match="profile"):
        validate_batch_receipt(record, index=0, plan_hash="plan1", census=census)


def test_real_profile_receipt_roundtrip_and_resume_index(tmp_path):
    from calibrate_alignment_uniformity import validate_batch_receipt, verified_batch_index
    from cap40_posthoc import sha, write

    torch.set_num_threads(1)
    runtime, batch = runtime_fixture()
    runtime.objective.au_enabled = True
    runtime.objective.au_alignment_weight = 0.002
    with preserved_observer(runtime, 912) as before:
        record = calibrate_batch(
            runtime, batch, microbatch=2, bf16=False, profile=MECHANISM_PROFILE
        )
    census = dict(unique_count=record["unique_count"], pair_count=record["pair_count"])
    record.update(
        plan_sha256="profile-plan",
        batch_index=0,
        census=census,
        state_unchanged=True,
        frozen_state=before,
        environment={"torch": torch.__version__},
    )
    record = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    validate_batch_receipt(
        record, index=0, plan_hash="profile-plan", census=census, profile=MECHANISM_PROFILE
    )
    index = verified_batch_index(tmp_path, "profile-plan", [census] * 8, profile=MECHANISM_PROFILE)
    path = tmp_path / "batch-00.json"
    path.write_text(json.dumps(record, sort_keys=True, allow_nan=False))
    index["receipts"][path.name] = sha(path)
    write(tmp_path / "BATCH_INDEX.json", index)
    assert (
        verified_batch_index(tmp_path, "profile-plan", [census] * 8, profile=MECHANISM_PROFILE)
        == index
    )
    bad = copy.deepcopy(record)
    bad["loss_scope"].pop("SSL2_activity")
    with pytest.raises(ValueError, match="profile"):
        validate_batch_receipt(
            bad, index=0, plan_hash="profile-plan", census=census, profile=MECHANISM_PROFILE
        )
