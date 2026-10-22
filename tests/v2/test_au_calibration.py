"""Full-population no-update calibration must preserve gradients and state."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from calibrate_alignment_uniformity import (  # noqa: E402
    calibrate_batch,
    metadata_census,
    preserved_observer,
    require_raw_identity,
    spectrum,
    validate_batch_receipt,
    verified_batch_index,
)
from capacity_probe import validation_state  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "au_cal_fixture", ROOT / "tests/v2/test_relay_method.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def runtime_fixture(checkpointed=False):
    from gradpert.training.v2.optimizer import V2Optimizer

    objective, batch = fixture.relay_training_fixture(checkpointed=checkpointed)
    objective.weights = ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    batch = replace(batch, context_index=torch.tensor([0, 0, 0, 0]))
    runtime = SimpleNamespace(
        objective=objective,
        optimizer=V2Optimizer(objective.student, 0.001, 0),
        generator=np.random.default_rng(7),
        device=torch.device("cpu"),
    )
    return runtime, batch


def test_census_deduplicates_before_context_pair_weighting():
    result, indices = metadata_census(
        ["a", "b", "a", "c", "d"],
        ["x", "y", "x", "z", "w"],
        ["ctx1", "ctx1", "ctx1", "ctx2", "ctx2"],
    )
    assert indices == [0, 1, 3, 4]
    assert result["input_count"] == 5 and result["duplicate_count"] == 1
    assert result["unique_count"] == 4 and result["pair_count"] == 2
    assert [r["pair_weight"] for r in result["contexts"]] == [0.5, 0.5]
    empty, _ = metadata_census(["a", "a"], ["x", "y"], ["c", "c"])
    assert empty["pair_count"] == 0 and empty["contexts"][0]["pair_weight"] is None
    with pytest.raises(ValueError, match="multiple compatible contexts"):
        metadata_census(["a", "a"], ["x", "x"], ["c", "d"])


def test_spectrum_does_not_add_identity_noise_or_normalize_rows():
    values = torch.tensor([[10.0, -1.0], [10.0, 1.0], [10.0, -1.0], [10.0, 1.0]])
    actual = spectrum(values)
    assert actual["effective_rank"] == pytest.approx(1.0)
    assert actual["normalized_rank"] == pytest.approx(0.5)
    assert actual["relative_variance"] == pytest.approx(1 / 101)
    assert spectrum(torch.ones(4, 3))["effective_rank"] == 0.0
    assert spectrum(torch.ones(1, 3))["effective_rank"] is None


@pytest.mark.parametrize("checkpointed", [False, True])
def test_replay_matches_retained_graph_gradients_and_preserves_state(checkpointed):
    from gradpert.training.v2.alignment_uniformity import alignment_uniformity
    from gradpert.training.v2.engine import slice_cells
    from gradpert.training.v2.reductions import population_weights

    torch.set_num_threads(1)
    runtime, batch = runtime_fixture(checkpointed)
    before = validation_state(runtime)
    with preserved_observer(runtime, 912):
        result, gradients = calibrate_batch(
            runtime, batch, microbatch=2, bf16=False, return_gradients=True
        )
    assert validation_state(runtime) == before
    assert result["updates"] == 0 and result["pair_count"] == 4
    assert result["replay_max_abs_error"] <= 2e-6
    obj = runtime.objective
    parameters = [
        p
        for name, p in obj.student.named_parameters()
        if p.requires_grad and name.startswith(("response.", "graph."))
    ]
    with preserved_observer(runtime, 912):
        ids = batch.condition_index
        row = population_weights(ids, torch.ones_like(ids, dtype=torch.bool), "row_mean")
        valid = [view.mask.any(-1) for view in batch.cell_views[:2]]
        active = max(1, sum(bool(v.any()) for v in valid))
        nodes = torch.stack([population_weights(ids, v, "row_mean") / active for v in valid])
        collected, joints, mses = [], [], []
        for start in (0, 2):
            loss, terms = obj(
                slice_cells(batch, start, start + 2),
                include_ssl1=False,
                reduction_weights=(row[start : start + 2], nodes[:, start : start + 2]),
                prediction_weights=row[start : start + 2],
                row_weights=row[start : start + 2],
                deferred_au=collected,
                checkpoint_deferred_response=True,
            )
            joints.append(loss)
            mses.append(terms["prediction_mse"])
        values = [torch.cat([chunk[v] for chunk in collected]) for v in range(4)]
        au = alignment_uniformity(*values, temperature=2.0)
        graph_terms = obj.graph_loss(
            batch.graph_views,
            batch.condition_index,
            control=batch.control,
            query_gene_ids=batch.graph.ids[batch.query_positions],
        )
        graph_loss = obj.lambda1 * sum(
            w * v for w, v in zip(obj.weights[0], graph_terms.values(), strict=True)
        )
        scalars = dict(
            A=au.alignment, U=au.uniformity, MSE=sum(mses), joint=sum(joints) + graph_loss
        )
        for label, scalar in scalars.items():
            assert result["losses"][label] == pytest.approx(float(scalar.detach()), abs=2e-6)
            expected = torch.autograd.grad(scalar, parameters, retain_graph=True, allow_unused=True)
            for actual, target in zip(gradients[label], expected, strict=True):
                target = torch.zeros_like(actual) if target is None else target
                torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
    assert validation_state(runtime) == before


def test_observer_restores_rng_pending_modes_on_failure():
    runtime, _ = runtime_fixture()
    runtime.objective.pending["existing"] = [torch.tensor([1.0])]
    before = validation_state(runtime)
    with pytest.raises(RuntimeError, match="planned"), preserved_observer(runtime, 301):
        torch.rand(3)
        np.random.rand(3)
        runtime.generator.random(4)
        runtime.objective.pending["temporary"] = [torch.ones(4)]
        raise RuntimeError("planned failure")
    assert validation_state(runtime) == before


def test_observer_detects_parameter_change_instead_of_hiding_it():
    runtime, _ = runtime_fixture()
    with (
        pytest.raises(RuntimeError, match="mutated forbidden state: student"),
        preserved_observer(runtime, 301),
        torch.no_grad(),
    ):
        next(runtime.objective.student.parameters()).add_(1)


def test_calibration_rejects_missing_context_or_mixed_regularizers():
    runtime, batch = runtime_fixture()
    with pytest.raises(ValueError, match="context"):
        calibrate_batch(runtime, replace(batch, context_index=None), bf16=False)
    runtime.objective.weights = ((1.0, 0.0, 0.0), (1.0, 1.0, 0.1))
    with pytest.raises(ValueError, match="original RI"):
        calibrate_batch(runtime, batch, bf16=False)


def test_actual_batch_must_match_every_ordered_census_axis():
    spec = SimpleNamespace(
        control_row_ids=("c1", "c2"), perturbed_row_ids=("p1", "p2"), condition_ids=("a", "b")
    )
    require_raw_identity(spec, spec)
    for name in vars(spec):
        actual = SimpleNamespace(**vars(spec))
        setattr(actual, name, tuple(reversed(getattr(actual, name))))
        with pytest.raises(ValueError, match=name):
            require_raw_identity(actual, spec)


def complete_receipt():
    census = {"unique_count": 320, "pair_count": 200}
    result = dict(
        plan_sha256="plan1",
        batch_index=0,
        census=census,
        state_unchanged=True,
        updates=0,
        unique_count=320,
        pair_count=200,
        frozen_state={"student": "frozen_hash"},
        environment={"torch": "version"},
        replay_tolerance=2e-6,
        replay_max_abs_error=0,
        losses=dict(A=0.1, U=-0.4, MSE=0.8, joint=30.0),
        gradients={"graph": {}, "response": {}},
        spectrum={"global0": {}, "global1": {}},
    )
    return result, census


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_index", 1),
        ("plan_sha256", "oldrun"),
        ("updates", 1),
        ("state_unchanged", False),
        ("pair_count", 2),
        ("frozen_state", {}),
        ("replay_tolerance", 1e-4),
        ("replay_max_abs_error", 3e-6),
        ("losses", {"A": 0.1}),
        ("spectrum", {}),
    ],
)
def test_incomplete_observation_cannot_become_a_complete_receipt(field, value):
    record, census = complete_receipt()
    record[field] = value
    with pytest.raises(ValueError):
        validate_batch_receipt(record, index=0, plan_hash="plan1", census=census)


def test_receipt_resume_verifies_independent_hash_index(tmp_path):
    from cap40_posthoc import sha, write

    record, census = complete_receipt()
    index = verified_batch_index(tmp_path, "plan1", [census] * 8)
    write(tmp_path / "batch-00.json", record)
    with pytest.raises(ValueError, match="index and files differ"):
        verified_batch_index(tmp_path, "plan1", [census] * 8)
    index["receipts"]["batch-00.json"] = sha(tmp_path / "batch-00.json")
    write(tmp_path / "BATCH_INDEX.json", index)
    assert len(verified_batch_index(tmp_path, "plan1", [census] * 8)["receipts"]) == 1
    record["losses"]["A"] = 0.2
    write(tmp_path / "batch-00.json", record)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_batch_index(tmp_path, "plan1", [census] * 8)


def test_execution_failure_leaves_explicit_noncompletion_receipt(tmp_path, monkeypatch, capsys):
    import calibrate_alignment_uniformity as module
    import cap40_posthoc

    cap40_posthoc.write(tmp_path / "PLAN.json", dict(run_id="run1", source={"commit": "sha1"}))
    monkeypatch.setattr(cap40_posthoc, "server_path", lambda path: path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate",
            "--mode",
            "census",
            "--config",
            "config",
            "--data-root",
            "data",
            "--output",
            str(tmp_path),
            "--publication",
            "publication",
            "--publication-sha256",
            "hash",
            "--source-commit",
            "sha1",
            "--run-id",
            "run1",
        ],
    )

    def fail(_args):
        raise ValueError("planned replay mismatch")

    monkeypatch.setattr(module, "execute", fail)
    assert module.main() == 1
    result = cap40_posthoc.read(tmp_path / "FAILURE.json")
    assert result["failed"] is True and result["calibration_complete"] is False
    assert result["scientific_complete"] is False
    assert "planned replay mismatch" in capsys.readouterr().err
