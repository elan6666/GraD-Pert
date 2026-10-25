"""R13 native replay, immutable three-endpoint verification and coefficient fit."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import calibrate_alignment_uniformity as shared  # noqa: E402
import calibrate_relative_amplitude as amplitude  # noqa: E402
from cap40_posthoc import read, sha, write  # noqa: E402
from capacity_probe import validation_state  # noqa: E402
from test_au_calibration import runtime_fixture  # noqa: E402


@pytest.mark.parametrize("checkpointed", [False, True])
def test_amplitude_replay_matches_direct_gradients_and_preserves_all_state(checkpointed):
    from gradpert.training.v2.engine import slice_cells
    from gradpert.training.v2.reductions import population_weights
    from gradpert.training.v2.spectral_pair import spectral_pair

    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(334)
        runtime, batch = runtime_fixture(checkpointed)
    obj = runtime.objective
    old_forward = obj.forward

    def shifted(*args, **kwargs):
        # A controlled near-constant population activates the real floor.
        result = old_forward(*args, **kwargs)
        collected = kwargs.get("deferred_au")
        if collected:
            values = collected[-1]
            collected[-1] = tuple([25 + 0.1 * x for x in values[:2]] + list(values[2:]))
        return result

    obj.forward = shifted
    next(obj.student.parameters()).grad = torch.ones_like(next(obj.student.parameters()))
    before = validation_state(runtime)
    with shared.preserved_observer(runtime, 334):
        result, banks = shared.calibrate_batch(
            runtime,
            batch,
            microbatch=2,
            bf16=False,
            profile=shared.AMPLITUDE_PROFILE,
            return_gradients=True,
        )
    assert before == validation_state(runtime)
    assert result["losses"]["amplitude"] > 0
    assert result["relative_amplitude"]["active"] == [True, True]
    params = [p for p in obj.student.parameters() if p.requires_grad]
    with shared.preserved_observer(runtime, 334):
        row = population_weights(
            batch.condition_index,
            torch.ones_like(batch.condition_index, dtype=torch.bool),
            "row_mean",
        )
        valid = [v.mask.any(-1) for v in batch.cell_views[:2]]
        active = max(1, sum(bool(v.any()) for v in valid))
        nodes = torch.stack(
            [population_weights(batch.condition_index, v, "row_mean") / active for v in valid]
        )
        collect, joints, mses = [], [], []
        for start in (0, 2):
            joint, terms = obj(
                slice_cells(batch, start, start + 2),
                include_ssl1=False,
                reduction_weights=(row[start : start + 2], nodes[:, start : start + 2]),
                prediction_weights=row[start : start + 2],
                row_weights=row[start : start + 2],
                deferred_au=collect,
                checkpoint_deferred_response=True,
            )
            joints.append(joint)
            mses.append(terms["prediction_mse"])
        z = [torch.cat([chunk[v] for chunk in collect]) for v in (0, 1)]
        spectral = spectral_pair(*z, amplitude_target=0.0225)
        graph_terms = obj.graph_loss(
            batch.graph_views,
            batch.condition_index,
            control=batch.control,
            query_gene_ids=batch.graph.ids[batch.query_positions],
        )
        ssl1 = obj.lambda1 * sum(
            w * value for w, value in zip(obj.weights[0], graph_terms.values(), strict=True)
        )
        direct = dict(
            amplitude=spectral.amplitude,
            rank_bal=spectral.rank,
            cosine=spectral.cosine,
            MSE=sum(mses),
            SSL1=ssl1,
            SSL2=sum(joints) - sum(mses),
            joint=sum(joints) + ssl1,
        )
        direct.update(
            base_cos010=direct["joint"] + 0.02 * spectral.rank + 0.10 * spectral.cosine,
            base_cos000=direct["joint"] + 0.02 * spectral.rank,
        )
        for label, scalar in direct.items():
            target = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
            assert result["losses"][label] == pytest.approx(float(scalar.detach()), abs=2e-6)
            for observed, expected in zip(banks[label], target, strict=True):
                expected = torch.zeros_like(observed) if expected is None else expected
                torch.testing.assert_close(observed, expected, rtol=3e-5, atol=2e-6)
    assert before == validation_state(runtime)
    assert banks["amplitude"][0].dtype == torch.float32
    assert result["updates"] == 0
    assert set(result["gradients"]) == {"response", "graph", "full"}


def _record(plan_hash, census, index, *, inactive=False, amp=-0.0):
    names = ("response.weight", "graph.weight", "prediction.weight", "response_cls")
    scalars = dict(
        MSE=1.0,
        joint=3.0,
        amplitude=0.0 if inactive else (amp or 1.0),
        rank_bal=0.2,
        cosine=0.1,
        SSL1=1.0,
        SSL2=1.0,
    )
    scalars["base_cos010"] = scalars["joint"] + 0.02 * scalars["rank_bal"] + 0.1 * scalars["cosine"]
    scalars["base_cos000"] = scalars["joint"] + 0.02 * scalars["rank_bal"]
    banks = {
        label: [torch.tensor([value], dtype=torch.float64) for _ in names]
        for label, value in scalars.items()
    }
    activity = {label: [True] * len(names) for label in banks}
    gradients = shared.gradient_summary(
        banks, names, profile=shared.AMPLITUDE_PROFILE, activity=activity
    )
    q = 0.1 if inactive else 0.0001
    amp_loss = max(0.0, math.sqrt(0.0225) - math.sqrt(q + 1e-8))
    energies = dict(total_energy=256.0, centered_energy=256 * q, mean_energy=256 * (1 - q), q=q)
    frozen = {key: "d" * 64 for key in amplitude.FROZEN}
    return dict(
        plan_sha256=plan_hash,
        batch_index=index,
        census=census,
        updates=0,
        state_unchanged=True,
        unique_count=320,
        pair_count=census["pair_count"],
        frozen_state=frozen,
        frozen_state_after=dict(frozen),
        environment={"torch": "synthetic"},
        replay_tolerance=2e-6,
        replay_max_abs_error=0.0,
        losses={label: amp_loss if label == "amplitude" else 0.1 for label in amplitude.LOSSES},
        gradients=gradients,
        spectrum={
            f"global{v}": dict(sample_count=320, width=256, relative_variance=q) for v in range(2)
        },
        profile=shared.AMPLITUDE_PROFILE,
        loss_scope=shared.AMPLITUDE_SCOPE,
        observation_seed=41239 + index,
        observation_rng_sha256="f" * 64,
        assembled_batch_sha256="a" * 64,
        microbatch_rng_sha256=["c" * 64] * 4,
        spectral=dict(
            input_count=320,
            feature_count=256,
            max_rank=256,
            raw_rank_target_bal=0.2,
            softness=0.25,
            weighted_warmup_scale=1.0,
            first_update_scale=0.0,
            rho=[0.5, 0.5],
            traces=[256 * q, 256 * q],
        ),
        relative_amplitude=dict(
            target=0.0225,
            epsilon=1e-8,
            q=[q, q],
            total_energy=[256.0, 256.0],
            mean_energy=[256 * (1 - q)] * 2,
            active=[not inactive] * 2,
            gamma=dict(min=1.0, median=1.0, max=1.0, norm=16.0, mean_abs=1.0),
            fixed_row_rms=[energies, energies],
            affine_free=[energies, energies],
            affine_free_status="ok",
            raw_activations_saved=False,
        ),
    )


def _endpoint(root, name, *, inactive=False, amp=1.0):
    from gradpert.hashing import sha256_json

    root.mkdir()
    expected = amplitude.TRAINING["ri_last" if name == "init" else name]
    identity = (
        None
        if name == "init"
        else dict(
            source=dict(
                commit=expected["commit"],
                published_commit=expected["commit"],
                dirty=False,
                formal_eligible=True,
            ),
            config_sha256=expected["config_sha256"],
            resolved_config_sha256="6" * 64,
            data={"canonical_sha256": "7" * 64},
            run_id=name,
        )
    )
    plan = dict(
        schema="gradpert-relative-amplitude-calibration-1",
        mode="gradients",
        endpoint=name,
        profile=shared.AMPLITUDE_PROFILE,
        loss_scope=shared.AMPLITUDE_SCOPE,
        batches=8,
        effective_population=320,
        microbatch=80,
        updates=0,
        split="train_only",
        seed=1,
        softness=0.25,
        amplitude_target=0.0225,
        amplitude_epsilon=1e-8,
        topology="single_gpu_diagnostic_not_ddp_resume",
        rng_seed=41239,
        config_sha256=expected["config_sha256"],
        resolved_config_sha256="6" * 64,
        publication_sha256="2" * 64,
        shared_policy_sha256="3" * 64,
        training_manifest_sha256=None if name == "init" else "4" * 64,
        checkpoint_sha256=None if name == "init" else expected["checkpoint_sha256"],
        training_identity=identity,
        source=dict(commit="a" * 40, published_commit="a" * 40, dirty=False, formal_eligible=True),
    )
    plan_hash = sha256_json(plan)
    census = [
        dict(input_count=320, unique_count=320, pair_count=200, ordered_identity_sha256=f"{i:064x}")
        for i in range(8)
    ]
    write(root / "PLAN.json", plan)
    write(
        root / "CENSUS.json",
        dict(plan_sha256=plan_hash, batches=census, complete=True, model_forward=False, updates=0),
    )
    hashes = {}
    for i, population in enumerate(census):
        path = root / f"batch-{i:02d}.json"
        write(path, _record(plan_hash, population, i, inactive=inactive, amp=amp))
        hashes[path.name] = sha(path)
    write(root / "BATCH_INDEX.json", dict(plan_sha256=plan_hash, receipts=hashes))
    write(
        root / "COMPLETE.json",
        dict(
            calibration_complete=True,
            scientific_complete=False,
            updates=0,
            plan_sha256=plan_hash,
            batches_completed=8,
            expected=8,
            batch_receipts=hashes,
        ),
    )
    return root


def _inputs(tmp_path):
    return [
        _endpoint(tmp_path / name, name, inactive=name == "init")
        for name in ("init", "ri_last", "r12bal_last")
    ]


def _rewrite_record(root, index, edit):
    path = root / f"batch-{index:02d}.json"
    value = read(path)
    edit(value)
    write(path, value)
    checksum = sha(path)
    for filename in ("BATCH_INDEX.json", "COMPLETE.json"):
        doc = read(root / filename)
        doc["receipts" if filename == "BATCH_INDEX.json" else "batch_receipts"][path.name] = (
            checksum
        )
        write(root / filename, doc)


def test_three_endpoints_select_training_only_one_common_weight(tmp_path):
    inputs = _inputs(tmp_path)
    result = amplitude.verify_endpoints(*inputs, source_commit="a" * 40)
    assert result["calibration_complete"] and result["engineering_accepted"]
    assert not result["scientific_complete"] and not result["coefficient_yaml_frozen"]
    assert result["selection"]["selected_weight"] == pytest.approx(0.1)
    assert result["selection"]["eligible_batches"] == 8
    assert len(result["source_files"]) == 36
    assert len(result["selection"]["attempts"][0]["directions"]) == 144
    assert all(
        row["unit_amplitude_over_mse"] is None
        for row in result["selection"]["gradient_comparisons"]
        if row["endpoint"] == "init"
    )


@pytest.mark.parametrize(
    "mutation", ["checkpoint", "source", "config", "rng", "population", "state", "nan", "missing"]
)
def test_bad_or_mutated_endpoint_cannot_select(tmp_path, mutation):
    inputs = _inputs(tmp_path)
    root = inputs[-1]
    if mutation in {"checkpoint", "source", "config"}:
        path = root / "PLAN.json"
        value = read(path)
        if mutation == "checkpoint":
            value["checkpoint_sha256"] = "0" * 64
        elif mutation == "source":
            value["source"]["dirty"] = True
        else:
            value["config_sha256"] = "0" * 64
        write(path, value)
    elif mutation == "missing":
        (root / "batch-07.json").unlink()
    elif mutation == "nan":
        path = root / "batch-00.json"
        path.write_text(path.read_text().replace('"MSE": 0.1', '"MSE": NaN'))
    else:

        def edit(value):
            if mutation == "rng":
                value["microbatch_rng_sha256"][0] = "8" * 64
            elif mutation == "population":
                value["assembled_batch_sha256"] = "9" * 64
            else:
                value["frozen_state_after"]["teacher"] = "8" * 64

        _rewrite_record(root, 0, edit)
    with pytest.raises(ValueError):
        amplitude.verify_endpoints(*inputs, source_commit="a" * 40)


def test_receipt_hash_mismatch_fails_even_if_finite(tmp_path):
    inputs = _inputs(tmp_path)
    path = inputs[-1] / "batch-00.json"
    value = read(path)
    value["losses"]["MSE"] += 0.1
    write(path, value)
    with pytest.raises(ValueError, match="checksum"):
        amplitude.verify_endpoints(*inputs, source_commit="a" * 40)


def test_weight_fit_excludes_zero_mse_and_requires_four_finite_positive_ratios():
    record = _record("a" * 64, dict(pair_count=200), 0)
    records = [copy.deepcopy(record) for _ in range(8)]
    for row in records[:5]:
        row["gradients"]["response"]["norms"]["MSE"] = 0.0
    endpoints = {
        name: dict(records=copy.deepcopy(records)) for name in ("init", "ri_last", "r12bal_last")
    }
    result = amplitude.select_weight(endpoints)
    assert not result["accepted"] and result["selected_weight"] is None
    assert result["eligible_batches"] == 3
    assert any(row["unavailable_reason"] == "zero_MSE" for row in result["gradient_comparisons"])


def test_direction_flip_halves_at_most_twice_and_rejects_outside_band():
    record = _record("a" * 64, dict(pair_count=200), 0)
    endpoints = {
        name: dict(records=[copy.deepcopy(record) for _ in range(8)])
        for name in ("init", "ri_last", "r12bal_last")
    }
    for endpoint in endpoints.values():
        for row in endpoint["records"]:
            for bank in row["gradients"].values():
                for base in ("base_cos010", "base_cos000"):
                    bank["pairs"]["__".join(sorted(("MSE", base)))]["dot"] = 0.06
                bank["pairs"]["MSE__amplitude"]["dot"] = -1.0
    result = amplitude.select_weight(endpoints)
    assert result["accepted"] and len(result["attempts"]) == 2
    assert result["selected_weight"] == pytest.approx(0.05)
    for endpoint in endpoints.values():
        for row in endpoint["records"]:
            for bank in row["gradients"].values():
                for base in ("base_cos010", "base_cos000"):
                    bank["pairs"]["__".join(sorted(("MSE", base)))]["dot"] = 0.001
    result = amplitude.select_weight(endpoints)
    assert not result["accepted"] and result["selected_weight"] is None
    assert len(result["attempts"]) == 3
    assert result["attempts"][-1]["weight"] == pytest.approx(0.025)


def test_selection_fits_only_bal_and_no_mutation_of_inputs():
    record = _record("a" * 64, dict(pair_count=200), 0)
    endpoints = {
        name: dict(records=[copy.deepcopy(record) for _ in range(8)])
        for name in ("init", "ri_last", "r12bal_last")
    }
    for row in endpoints["ri_last"]["records"]:
        row["gradients"]["response"]["norms"]["amplitude"] = 1000.0
    before = copy.deepcopy(endpoints)
    result = amplitude.select_weight(endpoints)
    assert result["selected_weight"] == pytest.approx(0.1)
    assert endpoints == before


def test_input_mutation_during_verification_is_detected(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path)
    original = amplitude._read

    def mutate_after_last_read(path, hashes):
        result = original(path, hashes)
        if path == inputs[-1] / "batch-07.json":
            plan = inputs[0] / "PLAN.json"
            plan.write_bytes(plan.read_bytes() + b" ")
        return result

    monkeypatch.setattr(amplitude, "_read", mutate_after_last_read)
    with pytest.raises(ValueError, match="changed during verification"):
        amplitude.verify_endpoints(*inputs, source_commit="a" * 40)


def test_model_mutation_between_populations_fails_even_with_unchanged_each_observation(tmp_path):
    inputs = _inputs(tmp_path)

    def edit(value):
        value["frozen_state"]["student"] = "8" * 64
        value["frozen_state_after"]["student"] = "8" * 64

    _rewrite_record(inputs[-1], 1, edit)
    with pytest.raises(ValueError, match="changed between populations"):
        amplitude.verify_endpoints(*inputs, source_commit="a" * 40)


@pytest.mark.parametrize("field", ["profile", "nonfinite_gradient", "energy", "loss"])
def test_strict_profile_and_finite_energy_component_validation(tmp_path, field):
    inputs = _inputs(tmp_path)

    def edit(value):
        if field == "profile":
            value["profile"] = shared.SPECTRAL_PROFILE
        elif field == "nonfinite_gradient":
            value["gradients"]["response"]["norms"]["MSE"] = "not-a-number"
        elif field == "energy":
            value["relative_amplitude"]["total_energy"][0] = 10.0
        else:
            value["losses"]["amplitude"] = 0.2

    _rewrite_record(inputs[-1], 0, edit)
    with pytest.raises(ValueError):
        amplitude.verify_endpoints(*inputs, source_commit="a" * 40)


def test_rehashed_inactive_r12_hinge_with_forged_positive_gradient_is_rejected(tmp_path):
    inputs = _inputs(tmp_path)

    def forge_inactive_metadata(value):
        inactive = _record(
            value["plan_sha256"], value["census"], value["batch_index"], inactive=True
        )
        for key in ("relative_amplitude", "spectral", "spectrum"):
            value[key] = inactive[key]
        value["losses"]["amplitude"] = 0.0
        # Keep the old positive amplitude gradient bank and rehash all receipts.
        assert all(bank["norms"]["amplitude"] > 0 for bank in value["gradients"].values())

    for i in range(8):
        _rewrite_record(inputs[-1], i, forge_inactive_metadata)
    with pytest.raises(ValueError, match="inactive amplitude hinge"):
        amplitude.verify_endpoints(*inputs, source_commit="a" * 40)


def test_inactive_observations_are_valid_but_explicitly_excluded_from_weight_fit(tmp_path):
    inputs = _inputs(tmp_path)
    result = amplitude.verify_endpoints(*inputs, source_commit="a" * 40)
    initial = [
        row for row in result["selection"]["gradient_comparisons"] if row["endpoint"] == "init"
    ]
    assert result["calibration_complete"] and result["engineering_accepted"]
    assert initial and all(
        row["unit_amplitude_over_mse"] is None
        and row["unavailable_reason"] == "inactive_amplitude_hinge"
        for row in initial
    )
    # Even direct selection never fits impossible positive norms if the hinge is inactive.
    forged = _record("a" * 64, dict(pair_count=200), 0, inactive=True)
    forged["gradients"] = _record("a" * 64, dict(pair_count=200), 0)["gradients"]
    endpoints = {
        name: dict(records=[copy.deepcopy(forged) for _ in range(8)])
        for name in ("init", "ri_last", "r12bal_last")
    }
    selection = amplitude.select_weight(endpoints)
    assert not selection["accepted"] and selection["eligible_batches"] == 0
    assert all(
        row["unavailable_reason"] == "inactive_amplitude_hinge"
        for row in selection["gradient_comparisons"]
    )


def test_active_hinge_does_not_require_nonzero_parameter_gradients():
    active = _record("a" * 64, dict(pair_count=200), 0)
    inactive = _record("a" * 64, dict(pair_count=200), 0, inactive=True)
    active["gradients"] = inactive["gradients"]
    assert active["relative_amplitude"]["active"] == [True, True]
    assert active["losses"]["amplitude"] > 0
    amplitude.validate_amplitude_record(active)
