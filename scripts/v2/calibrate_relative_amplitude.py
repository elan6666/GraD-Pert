"""R13 zero-update relative-amplitude observation and three-endpoint verification.

The normal driver reuses native source/resource/allocator gates. Supply the exact
sealed original configuration and manifest for each --endpoint init, ri_last,
or r12bal_last. No optimizer update, coefficient YAML, validation or test access.

Verify and select only after all three eight-population endpoints complete:
  python scripts/v2/calibrate_relative_amplitude.py verify --init INIT
    --ri-last RI --r12bal-last BAL --source-commit NEW_PUBLISHED_SHA
    --output NEW_SELECTION_JSON
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import calibrate_alignment_uniformity as shared

TARGET = 0.0225
EPSILON = 1e-8
TRAINING = {
    "ri_last": {
        "commit": "f0cb20ebb2efab749970f2bf5d205ed2f358592c",
        "config_sha256": "65304a02704561f9fbe3addb7be335fc88a51688f9cef69941ac81d1a920a4f7",
        "checkpoint_sha256": "96f6e718b4a3d2045ed4b0a5b3ba813a1fbc029aa3e1f0091d3ed68ef3f2d0d4",
        "config": "configs/v2/representation_round3_20261023/R3-RI/gradpert_v2/nadig_jurkat.yaml",
    },
    "r12bal_last": {
        "commit": "8f4fad3e7c9558b10e3a733be3e5e2fcf7046595",
        "config_sha256": "f90daed710f9333c42bca5d8290ba9e28ecebf042146a52e11ae5ec32c795acc",
        "checkpoint_sha256": "95d69d60f8469d0481692b1b78756d91fb27c32c056d487209673464b9d5ec29",
        "config": (
            "configs/v2/representation_round12_20261025/R12-SRBAL/gradpert_v2/nadig_jurkat.yaml"
        ),
    },
}
LOSSES = {
    "rank_bal",
    "cosine",
    "amplitude",
    "MSE",
    "joint",
    "SSL1",
    "SSL2",
    "base_cos010",
    "base_cos000",
}
GROUPS = {"response", "graph", "full"}
FROZEN = {
    "student",
    "teacher",
    "centers",
    "optimizer",
    "gradients",
    "numpy_rng",
    "python_rng",
    "torch_rng",
    "pending_statistics",
    "auxiliary_statistics",
    "module_modes",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, minimum=None, maximum=None):
    require(
        type(value) in (int, float) and math.isfinite(value), "finite numeric aggregate required"
    )
    require(minimum is None or value >= minimum, "aggregate below allowed range")
    require(maximum is None or value <= maximum, "aggregate above allowed range")
    return value


def digest(value, length=64):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def batch_identity(batch):
    """Hash actual assembled tensors/RNG-independent views, exporting no raw data."""
    from capacity_probe import state_fingerprint

    def tree(value):
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: tree(getattr(value, field.name)) for field in dataclasses.fields(value)
            }
        if isinstance(value, dict):
            return {k: tree(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [tree(v) for v in value]
        return value

    return state_fingerprint(tree(batch))


def _energy(values):
    x = values.detach().double()
    total = float(x.square().sum(-1).mean())
    variance = float((x - x.mean(0)).square().sum(-1).mean())
    return dict(
        total_energy=total,
        centered_energy=variance,
        mean_energy=float(x.mean(0).square().sum()),
        q=variance / total if total > 0 else None,
    )


def amplitude_statistics(z, result, student):
    """Raw and scale diagnostics are scalar aggregates; never activation exports."""
    import torch

    gamma = student.response.norm.weight.detach().double()
    fixed, affine_free = [], []
    for values in z:
        x = values.detach().double()
        row_rms = x.square().mean(-1, keepdim=True).sqrt()
        require(bool((row_rms > 0).all()), "zero row energy in fixed-RMS diagnostic")
        fixed.append(_energy(x / row_rms))
        affine_free.append(_energy(x / gamma) if bool((gamma != 0).all()) else None)
    require(bool(torch.isfinite(gamma).all()), "nonfinite final RMSNorm gamma")
    return dict(
        target=TARGET,
        epsilon=EPSILON,
        q=result.q.detach().cpu().tolist(),
        total_energy=result.total_energies.detach().cpu().tolist(),
        mean_energy=result.mean_energies.detach().cpu().tolist(),
        active=[bool(q + EPSILON < TARGET) for q in result.q.detach().cpu().tolist()],
        gamma=dict(
            min=float(gamma.min()),
            median=float(gamma.median()),
            max=float(gamma.max()),
            norm=float(gamma.norm()),
            mean_abs=float(gamma.abs().mean()),
        ),
        fixed_row_rms=fixed,
        affine_free=affine_free,
        affine_free_status="ok" if bool((gamma != 0).all()) else "unavailable_zero_gamma",
        raw_activations_saved=False,
    )


def _training_identity(identity, endpoint, checkpoint_sha256):
    if endpoint == "init":
        require(
            identity is None and checkpoint_sha256 is None, "init cannot use a trained checkpoint"
        )
        return
    require(
        endpoint in TRAINING and isinstance(identity, dict),
        "named original endpoint provenance required",
    )
    expected = TRAINING[endpoint]
    source = identity.get("source", {})
    require(
        checkpoint_sha256 == expected["checkpoint_sha256"],
        "checkpoint is not the sealed original endpoint",
    )
    require(
        source.get("commit") == source.get("published_commit") == expected["commit"]
        and source.get("dirty") is False
        and source.get("formal_eligible") is True,
        "original training source identity differs",
    )
    require(
        digest(identity.get("resolved_config_sha256"))
        and digest(identity.get("config_sha256"))
        and bool(identity.get("data"))
        and bool(identity.get("run_id")),
        "checkpoint training config/data/run identity missing",
    )


def validate_endpoint_inputs(args, config, training_identity):
    """Exact original config plus checkpoint source, before census/model preparation."""
    from gradpert.config import load_experiment_config
    from gradpert.hashing import sha256_json

    endpoint = getattr(args, "endpoint", None)
    require(
        endpoint in ("init", "ri_last", "r12bal_last"),
        "relative calibration requires explicit endpoint",
    )
    require(args.seed == 1, "original endpoint seed must be1")
    expected = TRAINING["ri_last" if endpoint == "init" else endpoint]
    from cap40_posthoc import sha

    require(sha(args.config) == expected["config_sha256"], "exact sealed config file hash required")
    original = load_experiment_config(Path(__file__).resolve().parents[2] / expected["config"])
    require(
        config.model_dump(mode="json") == original.model_dump(mode="json"),
        "relative calibration requires exact sealed endpoint configuration",
    )
    _training_identity(training_identity, endpoint, args.checkpoint_sha256)
    if training_identity:
        require(
            training_identity["resolved_config_sha256"]
            == sha256_json(config.model_dump(mode="json")),
            "original checkpoint resolved configuration differs",
        )


def _pair(bank, a, b):
    return bank["pairs"]["__".join(sorted((a, b)))]["dot"]


def validate_amplitude_record(record):
    """Supplement native common checks with strict profile and finite state proof."""

    def finite_tree(value):
        if type(value) in (int, float):
            finite(value)
        elif isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, list):
            for child in value:
                finite_tree(child)

    finite_tree(record)
    require(
        record.get("profile") == shared.AMPLITUDE_PROFILE
        and record.get("loss_scope") == shared.AMPLITUDE_SCOPE,
        "relative amplitude profile differs",
    )
    require(set(record.get("losses", {})) == LOSSES, "incomplete amplitude component losses")
    for value in record["losses"].values():
        finite(value)
    require(set(record.get("gradients", {})) == GROUPS, "shared and full gradient groups required")
    expected_pairs = {a + "__" + b for a, b in itertools.combinations(sorted(LOSSES), 2)}
    for bank in record["gradients"].values():
        require(
            set(bank)
            == {
                "norms",
                "pairs",
                "parameter_count",
                "inactive_parameters",
                "zero_gradient_parameters",
            }
            and set(bank["norms"]) == LOSSES
            and set(bank["pairs"]) == expected_pairs
            and set(bank["inactive_parameters"]) == LOSSES
            and set(bank["zero_gradient_parameters"]) == LOSSES,
            "incomplete component gradient banks",
        )
        require(
            type(bank["parameter_count"]) is int and bank["parameter_count"] > 0,
            "invalid gradient parameter count",
        )
        for value in bank["norms"].values():
            finite(value, 0)
        for a, b in itertools.combinations(sorted(LOSSES), 2):
            pair = bank["pairs"][a + "__" + b]
            require(
                set(pair) == {"dot", "cosine", "angle_degrees"}, "incomplete gradient comparison"
            )
            dot = finite(pair["dot"])
            product = bank["norms"][a] * bank["norms"][b]
            if product == 0:
                require(
                    dot == 0 and pair["cosine"] is None and pair["angle_degrees"] is None,
                    "zero gradients must have NA angles and exact zero dot",
                )
            else:
                cosine = finite(pair["cosine"], -1, 1)
                angle = finite(pair["angle_degrees"], 0, 180)
                require(
                    abs(dot) <= product * (1 + 1e-7)
                    and math.isclose(cosine, max(-1, min(1, dot / product)), abs_tol=1e-7)
                    and math.isclose(angle, math.degrees(math.acos(cosine)), abs_tol=1e-6),
                    "gradient dot/norm/angle inconsistency",
                )
    require(
        set(record.get("frozen_state", {})) >= FROZEN
        and all(digest(record["frozen_state"][k]) for k in FROZEN)
        and record.get("frozen_state_after") == record["frozen_state"],
        "all model/optimizer/gradient/center/RNG states must remain unchanged",
    )
    require(
        digest(record.get("assembled_batch_sha256"))
        and digest(record.get("observation_rng_sha256"))
        and type(record.get("observation_seed")) is int
        and len(record.get("microbatch_rng_sha256", [])) == 4
        and all(digest(x) for x in record["microbatch_rng_sha256"]),
        "actual population/view/RNG identity missing",
    )
    spectral = record.get("spectral", {})
    require(
        spectral.get("input_count") == 320
        and spectral.get("raw_rank_target_bal") == 0.20
        and spectral.get("softness") == 0.25
        and spectral.get("weighted_warmup_scale") == 1
        and spectral.get("first_update_scale") == 0
        and len(spectral.get("rho", [])) == len(spectral.get("traces", [])) == 2,
        "full original spectral population required",
    )
    stats = record.get("relative_amplitude", {})
    require(
        stats.get("target") == TARGET
        and stats.get("epsilon") == EPSILON
        and stats.get("raw_activations_saved") is False,
        "relative amplitude formula/export differs",
    )
    for key in ("q", "total_energy", "mean_energy", "active", "fixed_row_rms", "affine_free"):
        require(
            isinstance(stats.get(key), list) and len(stats[key]) == 2,
            "missing two Global energy diagnostics",
        )
    require(
        set(record.get("spectrum", {})) == {"global0", "global1"}, "two raw Global spectra required"
    )
    for v in range(2):
        finite(spectral["rho"][v], 0, 1 + 1e-7)
        variance = finite(spectral["traces"][v], 1e-24)
        total = finite(stats["total_energy"][v], 1e-24)
        mean = finite(stats["mean_energy"][v], 0)
        q = finite(stats["q"][v], 0, 1)
        require(
            math.isclose(q, variance / total, rel_tol=1e-7, abs_tol=1e-12)
            and math.isclose(total, variance + mean, rel_tol=1e-7, abs_tol=1e-12)
            and stats["active"][v] is (q + EPSILON < TARGET),
            "inconsistent energy decomposition",
        )
        raw = record["spectrum"][f"global{v}"]
        require(
            raw.get("sample_count") == 320
            and raw.get("width") == spectral.get("feature_count")
            and math.isclose(
                finite(raw.get("relative_variance"), 0, 1), q, rel_tol=1e-7, abs_tol=1e-12
            ),
            "raw energy/spectrum population mismatch",
        )
        for label in ("fixed_row_rms", "affine_free"):
            value = stats[label][v]
            if value is None:
                require(
                    label == "affine_free"
                    and stats.get("affine_free_status") == "unavailable_zero_gamma",
                    "unexpected missing amplitude diagnostics",
                )
            else:
                for key in ("total_energy", "centered_energy", "mean_energy", "q"):
                    finite(value.get(key), 0)
    require(
        math.isclose(
            record["losses"]["amplitude"],
            statistics.mean(max(0, math.sqrt(TARGET) - math.sqrt(q + EPSILON)) for q in stats["q"]),
            abs_tol=1e-10,
        ),
        "amplitude loss formula differs",
    )
    if not any(stats["active"]):
        require(
            record["losses"]["amplitude"] == 0, "inactive amplitude hinge requires exact zero loss"
        )
        for bank in record["gradients"].values():
            require(
                bank["norms"]["amplitude"] == 0,
                "inactive amplitude hinge requires exact zero gradient norms",
            )
            require(
                all(
                    pair["dot"] == 0 and pair["cosine"] is None and pair["angle_degrees"] is None
                    for key, pair in bank["pairs"].items()
                    if "amplitude" in key.split("__")
                ),
                "inactive amplitude hinge requires zero cross-dots and NA angles",
            )
    gamma = stats.get("gamma", {})
    require(set(gamma) == {"min", "median", "max", "norm", "mean_abs"}, "gamma diagnostics missing")
    for value in gamma.values():
        finite(value)
    require(
        gamma["min"] <= gamma["median"] <= gamma["max"]
        and gamma["norm"] >= 0
        and gamma["mean_abs"] >= 0,
        "invalid gamma distribution",
    )


def _read(path, hashes):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)

    import hashlib

    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=reject)
    hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return value


def _load_endpoint(root, name, source_commit, hashes):
    from gradpert.hashing import sha256_json

    root = Path(root).resolve(strict=True)
    plan, complete, census, index = (
        _read(root / f, hashes)
        for f in ("PLAN.json", "COMPLETE.json", "CENSUS.json", "BATCH_INDEX.json")
    )
    expected = dict(
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
        amplitude_target=TARGET,
        amplitude_epsilon=EPSILON,
        topology="single_gpu_diagnostic_not_ddp_resume",
    )
    require(
        all(plan.get(k) == v for k, v in expected.items()), "calibration endpoint protocol differs"
    )
    source = plan.get("source", {})
    require(
        digest(source_commit, 40)
        and source.get("commit") == source.get("published_commit") == source_commit
        and source.get("dirty") is False
        and source.get("formal_eligible") is True,
        "evaluation source must be the same clean published commit",
    )
    for key in (
        "config_sha256",
        "resolved_config_sha256",
        "publication_sha256",
        "shared_policy_sha256",
    ):
        require(digest(plan.get(key)), "missing immutable input hash: " + key)
    expected_config = TRAINING["ri_last" if name == "init" else name]
    require(
        plan["config_sha256"] == expected_config["config_sha256"],
        "sealed endpoint config file hash differs",
    )
    _training_identity(plan.get("training_identity"), name, plan.get("checkpoint_sha256"))
    if name != "init":
        require(
            digest(plan.get("training_manifest_sha256"))
            and plan["resolved_config_sha256"]
            == plan["training_identity"]["resolved_config_sha256"],
            "missing original training manifest/config identity",
        )
    plan_hash = sha256_json(plan)
    require(
        census.get("plan_sha256") == plan_hash
        and census.get("complete") is True
        and census.get("model_forward") is False
        and census.get("updates") == 0
        and len(census.get("batches", [])) == 8,
        "eight sealed training populations required",
    )
    expected_names = {f"batch-{i:02d}.json" for i in range(8)}
    require(
        index.get("plan_sha256") == plan_hash
        and set(index.get("receipts", {})) == expected_names
        and {p.name for p in root.glob("batch-*.json")} == expected_names,
        "eight indexed batch files required",
    )
    require(
        complete
        == dict(
            calibration_complete=True,
            scientific_complete=False,
            updates=0,
            plan_sha256=plan_hash,
            batches_completed=8,
            expected=8,
            batch_receipts=index["receipts"],
        ),
        "endpoint completion receipt differs",
    )
    require(
        len({row.get("ordered_identity_sha256") for row in census["batches"]}) == 8,
        "eight distinct ordered populations required",
    )
    records = []
    for i, row in enumerate(census["batches"]):
        require(
            row.get("input_count") == row.get("unique_count") == 320
            and digest(row.get("ordered_identity_sha256")),
            "full unique320 ordered population required",
        )
        path = root / f"batch-{i:02d}.json"
        record = _read(path, hashes)
        require(
            hashes[str(path.resolve())] == index["receipts"][path.name],
            "batch output checksum mismatch",
        )
        shared.validate_batch_receipt(
            record, index=i, plan_hash=plan_hash, census=row, profile=shared.AMPLITUDE_PROFILE
        )
        require(
            record["observation_seed"] == plan["rng_seed"] + i,
            "observed RNG seed differs from plan",
        )
        records.append(record)
    stable = FROZEN - {"numpy_rng", "python_rng", "torch_rng"}
    require(
        all(
            all(record["frozen_state"][key] == records[0]["frozen_state"][key] for key in stable)
            for record in records
        ),
        "endpoint model/state changed between populations",
    )
    return dict(name=name, root=str(root), plan=plan, census=census["batches"], records=records)


def select_weight(endpoints):
    """Fit one common weight on R12BAL train-only response norms, then audit all."""
    require(set(endpoints) == {"init", "ri_last", "r12bal_last"}, "three named endpoints required")
    ratios, comparisons = [], []
    for name, endpoint in endpoints.items():
        require(len(endpoint["records"]) == 8, "selection requires eight populations per endpoint")
        for i, record in enumerate(endpoint["records"]):
            active = any(record["relative_amplitude"]["active"])
            for group, bank in record["gradients"].items():
                mse = finite(bank["norms"]["MSE"], 0)
                amp = finite(bank["norms"]["amplitude"], 0)
                ratio = amp / mse if active and mse > 0 and amp > 0 else None
                if ratio is not None:
                    finite(ratio)
                    require(ratio > 0, "positive gradient ratio underflowed")
                comparisons.append(
                    dict(
                        endpoint=name,
                        batch_index=i,
                        group=group,
                        unit_amplitude_over_mse=ratio,
                        unavailable_reason="inactive_amplitude_hinge"
                        if not active
                        else "zero_MSE"
                        if mse == 0
                        else "zero_amplitude"
                        if amp == 0
                        else None,
                    )
                )
                if name == "r12bal_last" and group == "response" and ratio is not None:
                    ratios.append(ratio)
    if len(ratios) < 4:
        return dict(
            accepted=False,
            selected_weight=None,
            reason="fewer_than_four_positive_finite_R12_response_ratios",
            eligible_batches=len(ratios),
            gradient_comparisons=comparisons,
            attempts=[],
        )
    median = statistics.median(ratios)
    weight = 0.10 / median
    finite(weight)
    require(weight > 0, "invalid amplitude coefficient")
    attempts = []
    for attempt in range(3):
        directions, flips = [], []
        for name, endpoint in endpoints.items():
            for i, record in enumerate(endpoint["records"]):
                for group, bank in record["gradients"].items():
                    for base in ("base_cos010", "base_cos000"):
                        dot_before = _pair(bank, "MSE", base)
                        dot_after = dot_before + weight * _pair(bank, "MSE", "amplitude")
                        norm2 = (
                            bank["norms"][base] ** 2
                            + weight**2 * bank["norms"]["amplitude"] ** 2
                            + 2 * weight * _pair(bank, base, "amplitude")
                        )
                        norm = math.sqrt(max(0, norm2))
                        denominator = bank["norms"]["MSE"] * norm
                        cosine = max(-1, min(1, dot_after / denominator)) if denominator else None
                        row = dict(
                            endpoint=name,
                            batch_index=i,
                            group=group,
                            arm=base,
                            mse_dot_before=dot_before,
                            mse_dot_after=dot_after,
                            combined_norm=norm,
                            mse_cosine_after=cosine,
                            flips_positive_descent=dot_before > 0 and dot_after < 0,
                        )
                        directions.append(row)
                        if row["flips_positive_descent"]:
                            flips.append(row)
        ratio = weight * median
        attempts.append(
            dict(
                attempt=attempt,
                weight=weight,
                median_weighted_ratio=ratio,
                review_band_pass=0.05 <= ratio <= 0.20,
                direction_flip_count=len(flips),
                directions=directions,
            )
        )
        if not flips or attempt == 2:
            break
        weight /= 2
    accepted = not attempts[-1]["direction_flip_count"] and attempts[-1]["review_band_pass"]
    return dict(
        accepted=accepted,
        selected_weight=weight if accepted else None,
        proposed_weight=weight,
        reason="accepted_engineering_only"
        if accepted
        else "direction_flip_or_gradient_budget_outside_review_band",
        fitting_endpoint="r12bal_last",
        fitting_group="response",
        target_median_ratio=0.10,
        review_band=[0.05, 0.20],
        eligible_batches=len(ratios),
        median_unit_ratio=median,
        max_preregistered_halvings=2,
        attempts=attempts,
        gradient_comparisons=comparisons,
    )


def verify_endpoints(init, ri_last, r12bal_last, *, source_commit):
    """Read-only strict provenance, full-state, schedule and checksum verification."""
    hashes = {}
    endpoints = {
        name: _load_endpoint(path, name, source_commit, hashes)
        for name, path in (("init", init), ("ri_last", ri_last), ("r12bal_last", r12bal_last))
    }
    first = endpoints["init"]
    for endpoint in endpoints.values():
        require(
            endpoint["census"] == first["census"], "endpoint ordered training populations differ"
        )
        for key in ("seed", "rng_seed", "source", "publication_sha256", "shared_policy_sha256"):
            require(
                endpoint["plan"].get(key) == first["plan"].get(key),
                "endpoint source/RNG/resource identity differs",
            )
        for a, b in zip(first["records"], endpoint["records"], strict=True):
            for key in (
                "assembled_batch_sha256",
                "observation_seed",
                "observation_rng_sha256",
                "microbatch_rng_sha256",
            ):
                require(a[key] == b[key], "endpoint actual view/population/RNG identity differs")
    selection = select_weight(endpoints)
    # Recheck all inputs after verification/selection, including plan/census/index.
    import hashlib

    require(
        all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest_value
            for path, digest_value in hashes.items()
        ),
        "calibration inputs changed during verification",
    )
    return dict(
        schema="gradpert-relative-amplitude-selection-1",
        profile=shared.AMPLITUDE_PROFILE,
        calibration_complete=True,
        engineering_accepted=selection["accepted"],
        scientific_complete=False,
        updates=0,
        source_commit=source_commit,
        no_validation_or_test=True,
        coefficient_yaml_frozen=False,
        source_files=hashes,
        endpoints={
            name: {k: v for k, v in value.items() if k != "records"}
            for name, value in endpoints.items()
        },
        selection=selection,
    )


def main():
    if sys.argv[1:2] != ["verify"]:
        require("--profile" not in sys.argv, "relative amplitude driver fixes its profile")
        sys.argv.extend(["--profile", shared.AMPLITUDE_PROFILE])
        return shared.main()
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("init", "ri-last"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument(
        "--r12-bal-last", "--r12bal-last", dest="r12bal_last", type=Path, required=True
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(sys.argv[2:])
    result = verify_endpoints(
        args.init, args.ri_last, args.r12bal_last, source_commit=args.source_commit
    )
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(serialized)
    print(serialized, end="")
    return 0 if result["engineering_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
