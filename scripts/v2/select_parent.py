"""Choose a hyperparameter parent using complete, matched validation histories only."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from generate_group import verify_group

from gradpert.hashing import sha256_file


def choose(verified: dict, runs: dict[str, list[str]], seeds: list[int]) -> dict:
    if verified["group"] not in ("H1", "H2", "H3"):
        raise ValueError("automatic parent selection is restricted to hyperparameter groups")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("selection needs an explicit unique seed set")
    if set(runs) != {r["name"] for r in verified["rows"]}:
        raise ValueError("all registered candidates must have training evidence")
    common = None
    scores = []
    roots = set()
    for row in verified["rows"]:
        evidence = []
        observed = set()
        for run in runs[row["name"]]:
            root = Path(run).resolve()
            if root in roots:
                raise ValueError("a run cannot represent multiple candidates or seeds")
            roots.add(root)
            manifest_path, journal_path = root / "run_manifest.json", root / "fit/epoch_state.json"
            history_path = root / "fit/history.json"
            identity = json.loads(manifest_path.read_text())
            journal = json.loads(journal_path.read_text())
            history = json.loads(history_path.read_text())
            seed = identity["data"]["run_seed"]
            if seed not in seeds or seed in observed or identity["config_sha256"] != row["sha256"]:
                raise ValueError("candidate config or paired seed population mismatch")
            observed.add(seed)
            source = identity["source"]
            if source["dirty"] or source["commit"] != source["published_commit"]:
                raise ValueError("selection requires clean published training source")
            if (
                journal["identity"] != identity
                or journal["epoch"] != 50
                or journal["budget"][0] != 50
                or [h["epoch"] for h in history] != list(range(1, 51))
                or journal["budget"][1] <= 0
                or any(h["optimizer_steps"] != h["epoch"] * journal["budget"][1] for h in history)
            ):
                raise ValueError("every candidate must finish all fifty committed epochs")
            validations = [h["validation"] for h in history]
            if any(
                v["split"] != "val" or not math.isfinite(v["prediction_loss"]) for v in validations
            ):
                raise ValueError("parent selection accepts finite validation losses only")
            contract = {
                "training_sha": source["commit"],
                "data": {k: v for k, v in identity["data"].items() if k != "run_seed"},
                "control_manifest_sha256": validations[0]["control_manifest_sha256"],
                "reference_sha256": validations[0]["reference_sha256"],
                "query_recipe": validations[0]["query_recipe"],
            }
            if any(
                v["control_manifest_sha256"] != contract["control_manifest_sha256"]
                or v["reference_sha256"] != contract["reference_sha256"]
                or v["query_recipe"] != contract["query_recipe"]
                for v in validations
            ):
                raise ValueError("validation protocol changed within a run")
            for validation in validations:
                population = validation.get("population_receipt")
                if population is not None:
                    path = (root / population["file"]).resolve()
                    if not path.is_relative_to(root) or sha256_file(path) != population["sha256"]:
                        raise ValueError("validation population receipt differs from history")
            if common is None:
                common = contract
            elif common != contract:
                raise ValueError("candidate source/data/validation contracts differ")
            best = min(history, key=lambda h: h["validation"]["prediction_loss"])
            selected = journal["best"]
            checkpoint = (root / "fit" / selected["file"]).resolve()
            if (
                selected["epoch"] != best["epoch"]
                or selected["prediction_loss"] != best["validation"]["prediction_loss"]
                or not checkpoint.is_relative_to(root / "fit")
                or sha256_file(checkpoint) != selected["sha256"]
            ):
                raise ValueError("selected checkpoint differs from validation history or hash")
            evidence.append(
                {
                    "seed": seed,
                    "run_root": str(root),
                    "validation_prediction_loss": selected["prediction_loss"],
                    "history_sha256": sha256_file(history_path),
                    "journal_sha256": sha256_file(journal_path),
                    "run_manifest_sha256": sha256_file(manifest_path),
                    "checkpoint_sha256": selected["sha256"],
                }
            )
        if observed != set(seeds):
            raise ValueError("cannot select with missing paired seeds")
        scores.append(
            {
                "name": row["name"],
                "config": row["config"],
                "sha256": row["sha256"],
                "mean_validation_prediction_loss": statistics.mean(
                    e["validation_prediction_loss"] for e in evidence
                ),
                "evidence": evidence,
            }
        )
    winner = min(scores, key=lambda r: r["mean_validation_prediction_loss"])
    return {
        "schema_version": "gradpert-v2-validation-selection-1",
        "group": verified["group"],
        "group_manifest_sha256": verified["manifest_sha256"],
        "seeds": seeds,
        "rule": "mean best validation prediction loss across paired seeds; ties use manifest order",
        "test_data_read": False,
        "contract": common,
        "scores": scores,
        "winner": {k: winner[k] for k in ("name", "config", "sha256")},
    }


def verify_selection(selection_path: Path, manifest: Path, parent: Path) -> dict:
    """Recompute a frozen winner from its original evidence before using it."""
    recorded = json.loads(selection_path.read_text())
    runs = {
        row["name"]: [entry["run_root"] for entry in row["evidence"]] for row in recorded["scores"]
    }
    if len(runs) != len(recorded["scores"]):
        raise ValueError("selection receipt repeats a candidate")
    actual = choose(verify_group(manifest, parent), runs, recorded["seeds"])
    if actual != recorded:
        raise ValueError("selection receipt differs from recomputed validation evidence")
    winner = Path(actual["winner"]["config"])
    if sha256_file(winner) != actual["winner"]["sha256"]:
        raise ValueError("selected parent configuration changed")
    return {
        "selection_sha256": sha256_file(selection_path),
        "manifest_sha256": sha256_file(manifest),
        "winner": actual["winner"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--runs", type=Path, help="JSON mapping row names to run-root lists")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-selection", type=Path)
    args = parser.parse_args()
    if args.verify_selection:
        if args.runs or args.seeds or args.output:
            parser.error("selection verification does not accept generation arguments")
        print(json.dumps(verify_selection(args.verify_selection, args.manifest, args.parent)))
        return
    if not args.runs or not args.seeds or not args.output:
        parser.error("selection requires --runs, --seeds and --output")
    result = choose(
        verify_group(args.manifest, args.parent), json.loads(args.runs.read_text()), args.seeds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["winner"]))


if __name__ == "__main__":
    main()
