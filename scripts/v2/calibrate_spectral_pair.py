"""Server-only original soft-rank/cosine no-update calibration.

Uses the shared AU driver's exact eight train populations, two-pass replay,
immutable resume receipts, all-state/RNG preservation and resource/source gates.
Run once at init and once with the original RI-last checkpoint + exact original
RI config/training manifest. Both strengths share the same forward populations;
weighted banks show the fixed fully-warmed coefficients, and first-step weight0
is explicit. No optimizer/Teacher/center updates or validation/test access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import calibrate_alignment_uniformity as shared


def main():
    if sys.argv[1:2] == ["verify"]:
        import argparse
        import json

        parser = argparse.ArgumentParser(description="Verify both immutable eight-batch endpoints")
        parser.add_argument("--init", type=Path, required=True)
        parser.add_argument("--ri-last", type=Path, required=True)
        parser.add_argument("--source-commit", required=True)
        args = parser.parse_args(sys.argv[2:])
        print(
            json.dumps(
                verify_endpoints(args.init, args.ri_last, source_commit=args.source_commit),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
        )
        return 0
    if "--profile" in sys.argv:
        raise ValueError("spectral calibration fixes its diagnostic profile")
    sys.argv.extend(["--profile", shared.SPECTRAL_PROFILE])
    return shared.main()


def verify_endpoints(init, ri_last, *, source_commit=None):
    """Read-only acceptance of both checksum-bound eight-batch endpoints."""
    from cap40_posthoc import read, sha

    from gradpert.hashing import sha256_json

    endpoints = []
    for name, root in (("init", init), ("ri_last", ri_last)):
        root = Path(root)
        for filename in ("PLAN.json", "COMPLETE.json", "CENSUS.json", "BATCH_INDEX.json"):
            if not (root / filename).is_file():
                raise ValueError("calibration requires existing complete endpoint: " + name)
        plan, complete, census = (
            read(root / filename) for filename in ("PLAN.json", "COMPLETE.json", "CENSUS.json")
        )
        expected = dict(
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
        )
        source = plan.get("source", {})
        if (
            any(plan.get(k) != v for k, v in expected.items())
            or source.get("dirty") is not False
            or source.get("formal_eligible") is not True
            or source.get("commit") != source.get("published_commit")
            or (source_commit is not None and source.get("commit") != source_commit)
        ):
            raise ValueError("calibration source/no-update/train-only protocol differs")
        if name == "init":
            if (
                plan.get("checkpoint_sha256") is not None
                or plan.get("training_identity") is not None
            ):
                raise ValueError("init endpoint cannot use a trained checkpoint")
        elif not plan.get("checkpoint_sha256") or not plan.get("training_identity"):
            raise ValueError("RI-last endpoint requires original checkpoint/training provenance")
        plan_hash = sha256_json(plan)
        if (
            census.get("plan_sha256") != plan_hash
            or census.get("complete") is not True
            or census.get("model_forward") is not False
            or census.get("updates") != 0
            or len(census.get("batches", [])) != 8
            or any(
                row.get("input_count") != 320 or row.get("unique_count") != 320
                for row in census["batches"]
            )
        ):
            raise ValueError("eight complete unique 320-row populations required")
        index = shared.verified_batch_index(
            root, plan_hash, census["batches"], profile=shared.SPECTRAL_PROFILE
        )
        if (
            complete
            != dict(
                calibration_complete=True,
                scientific_complete=False,
                updates=0,
                plan_sha256=plan_hash,
                batches_completed=8,
                expected=8,
                batch_receipts=index["receipts"],
            )
            or len(index["receipts"]) != 8
        ):
            raise ValueError("calibration requires all eight checksum-bound completed batches")
        endpoints.append(
            dict(
                name=name,
                plan=plan,
                census=census["batches"],
                root=str(root.resolve()),
                complete_sha256=sha(root / "COMPLETE.json"),
            )
        )
    first, last = endpoints
    if first["census"] != last["census"] or any(
        first["plan"].get(k) != last["plan"].get(k)
        for k in ("rng_seed", "seed", "config_sha256", "source")
    ):
        raise ValueError(
            "both endpoints/strengths require identical ordered train populations and RNG"
        )
    rows = []
    for endpoint in endpoints:
        for index in range(8):
            record = read(Path(endpoint["root"]) / f"batch-{index:02d}.json")
            for group, bank in record["gradients"].items():
                norms = bank["norms"]
                for arm in ("R12-SRBAL", "R12-SRAGG"):
                    auxiliary = norms[arm + "_auxiliary"]
                    rows.append(
                        dict(
                            endpoint=endpoint["name"],
                            batch_index=index,
                            group=group,
                            arm=arm,
                            auxiliary_norm=auxiliary,
                            auxiliary_over_joint=auxiliary / norms["joint"]
                            if norms["joint"]
                            else None,
                            auxiliary_over_mse=auxiliary / norms["MSE"] if norms["MSE"] else None,
                            min_trace=min(record["spectral"]["traces"]),
                            min_rho=min(record["spectral"]["rho"]),
                        )
                    )
    return dict(
        calibration_complete=True,
        scientific_complete=False,
        updates=0,
        endpoints=endpoints,
        gradient_comparisons=rows,
        coefficients="original fixed BAL/AGG retained; no adaptive tuning",
    )


if __name__ == "__main__":
    raise SystemExit(main())
