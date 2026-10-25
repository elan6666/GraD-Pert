"""Original spectral pair: exact replay/weighted gradient banks and immutable receipts."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import calibrate_alignment_uniformity as shared  # noqa: E402
from calibrate_spectral_pair import verify_endpoints  # noqa: E402
from capacity_probe import validation_state  # noqa: E402
from test_au_calibration import runtime_fixture  # noqa: E402


@pytest.mark.parametrize("checkpointed", [False, True])
def test_full_population_replay_and_weighted_banks_match_direct_autograd(checkpointed):
    from gradpert.training.v2.engine import slice_cells
    from gradpert.training.v2.reductions import population_weights
    from gradpert.training.v2.spectral_pair import spectral_pair

    torch.set_num_threads(1)
    runtime, batch = runtime_fixture(checkpointed)
    before = validation_state(runtime)
    with shared.preserved_observer(runtime, 345):
        result, banks = shared.calibrate_batch(
            runtime,
            batch,
            microbatch=2,
            bf16=False,
            profile=shared.SPECTRAL_PROFILE,
            return_gradients=True,
        )
    assert before == validation_state(runtime)
    assert result["unique_count"] == 4 and result["updates"] == 0
    assert result["spectral"]["first_update_scale"] == 0
    assert result["replay_max_abs_error"] <= 2e-6
    obj = runtime.objective
    params = [
        p
        for name, p in obj.student.named_parameters()
        if p.requires_grad and (name.startswith(("response.", "graph.")) or name == "response_cls")
    ]
    with shared.preserved_observer(runtime, 345):
        ids = batch.condition_index
        row = population_weights(ids, torch.ones_like(ids, dtype=torch.bool), "row_mean")
        valid = [v.mask.any(-1) for v in batch.cell_views[:2]]
        active = max(1, sum(bool(v.any()) for v in valid))
        nodes = torch.stack([population_weights(ids, v, "row_mean") / active for v in valid])
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
        graph_terms = obj.graph_loss(
            batch.graph_views,
            batch.condition_index,
            control=batch.control,
            query_gene_ids=batch.graph.ids[batch.query_positions],
        )
        graph_loss = obj.lambda1 * sum(
            w * value for w, value in zip(obj.weights[0], graph_terms.values(), strict=True)
        )
        z = [torch.cat([chunk[v] for chunk in collect]) for v in (0, 1)]
        balanced, aggressive = spectral_pair(*z, target=0.20), spectral_pair(*z, target=0.35)
        direct = dict(
            joint=sum(joints) + graph_loss,
            MSE=sum(mses),
            rank_bal=balanced.rank,
            rank_agg=aggressive.rank,
            cosine=balanced.cosine,
            R12_SRBAL=0.02 * balanced.rank + 0.10 * balanced.cosine,
            R12_SRAGG=0.04 * aggressive.rank + 0.20 * aggressive.cosine,
        )
        for name, scalar in direct.items():
            label = name.replace("R12_SRBAL", "R12-SRBAL_auxiliary").replace(
                "R12_SRAGG", "R12-SRAGG_auxiliary"
            )
            expected = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
            assert result["losses"][label] == pytest.approx(float(scalar.detach()), abs=2e-6)
            for actual, target in zip(banks[label], expected, strict=True):
                target = torch.zeros_like(actual) if target is None else target
                torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
    assert before == validation_state(runtime)
    for group in ("graph", "response"):
        assert "joint__rank_bal" in result["gradients"][group]["pairs"]
        assert "MSE__rank_agg" in result["gradients"][group]["pairs"]
        assert result["gradients"][group]["norms"]["R12-SRBAL_rank"] == pytest.approx(
            0.02 * result["gradients"][group]["norms"]["rank_bal"]
        )


def _endpoint(root, *, last=False):
    from cap40_posthoc import sha, write

    from gradpert.hashing import sha256_json

    root.mkdir()
    plan = dict(
        schema="gradpert-spectral-pair-calibration-1",
        mode="gradients",
        profile=shared.SPECTRAL_PROFILE,
        loss_scope=shared.SPECTRAL_SCOPE,
        batches=8,
        effective_population=320,
        microbatch=80,
        updates=0,
        split="train_only",
        seed=1,
        softness=0.25,
        rng_seed=37,
        config_sha256="config",
        source=dict(commit="a" * 40, published_commit="a" * 40, dirty=False, formal_eligible=True),
        checkpoint_sha256="b" * 64 if last else None,
        training_identity={"run_id": "original-ri"} if last else None,
    )
    plan_hash = sha256_json(plan)
    census = [
        dict(input_count=320, unique_count=320, pair_count=200, ordered_identity_sha256=str(i))
        for i in range(8)
    ]
    write(root / "PLAN.json", plan)
    write(
        root / "CENSUS.json",
        dict(plan_sha256=plan_hash, complete=True, model_forward=False, updates=0, batches=census),
    )
    losses = {"rank_bal": 0.2, "rank_agg": 0.5, "cosine": 0.1, "joint": 4, "MSE": 0.5}
    for arm in ("R12-SRBAL", "R12-SRAGG"):
        losses.update({arm + "_" + term: 0.1 for term in ("rank", "cosine", "auxiliary", "total")})
    import itertools

    gradient = dict(
        norms=dict.fromkeys(losses, 0.1),
        pairs={
            a + "__" + b: dict(dot=0.01, cosine=1, angle_degrees=0)
            for a, b in itertools.combinations(sorted(losses), 2)
        },
        parameter_count=1,
        inactive_parameters={key: [] for key in losses},
        zero_gradient_parameters={key: [] for key in losses},
    )
    hashes = {}
    for i in range(8):
        record = dict(
            plan_sha256=plan_hash,
            batch_index=i,
            census=census[i],
            updates=0,
            state_unchanged=True,
            unique_count=320,
            pair_count=200,
            frozen_state=dict(student="hash"),
            environment=dict(torch="cpu-synthetic"),
            replay_tolerance=2e-6,
            replay_max_abs_error=0,
            losses=losses,
            gradients={k: gradient for k in ("graph", "response")},
            spectrum={k: {} for k in ("global0", "global1")},
            profile=shared.SPECTRAL_PROFILE,
            loss_scope=shared.SPECTRAL_SCOPE,
            spectral=dict(
                input_count=320,
                softness=0.25,
                weighted_warmup_scale=1.0,
                first_update_scale=0,
                rho=[0.2, 0.2],
                traces=[1, 1],
            ),
        )
        filename = f"batch-{i:02d}.json"
        write(root / filename, record)
        hashes[filename] = sha(root / filename)
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


def test_two_endpoints_require_same_eight_ordered_populations_and_rng(tmp_path):
    init, last = _endpoint(tmp_path / "init"), _endpoint(tmp_path / "last", last=True)
    result = verify_endpoints(init, last, source_commit="a" * 40)
    assert result["calibration_complete"] and not result["scientific_complete"]
    path = last / "CENSUS.json"
    value = json.loads(path.read_text())
    value["batches"][0]["ordered_identity_sha256"] = "different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify_endpoints(init, last)


def test_changed_receipt_or_missing_batch_never_completes(tmp_path):
    init, last = _endpoint(tmp_path / "init"), _endpoint(tmp_path / "last", last=True)
    path = last / "batch-07.json"
    record = json.loads(path.read_text())
    record["updates"] = 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        verify_endpoints(init, last)
    path.unlink()
    with pytest.raises(ValueError, match="index and files"):
        verify_endpoints(init, last)
