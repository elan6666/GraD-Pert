"""Seal loss-arm identities and reuse the existing capacity/training queue."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from generate_loss_group import ARMS, FIELDS, PARENT, PARENT_SHA256
from run_cap40_ablations import execute

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.train_entry import resolve_plan
from gradpert.hashing import sha256_file


def verify_manifest(source: Path, path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != "gradpert-v2-loss-six-1"
        or manifest.get("parent") != PARENT
        or manifest.get("parent_sha256") != PARENT_SHA256
        or [row["name"] for row in manifest["rows"]] != list(ARMS)
        or manifest.get("epochs") != 6
        or manifest.get("validation") != "disabled"
        or manifest.get("test_roles") != ["last"]
        or manifest.get("common_microbatch") != 68
        or manifest.get("global_batch") != 272
        or manifest.get("seed") != 1
        or manifest.get("primary_metric") != "txpert_macro_pearson_delta_deg"
        or sha256_file(source / PARENT) != PARENT_SHA256
    ):
        raise ValueError("loss queue differs from the authorized six-arm design")
    baseline = load_experiment_config(source / PARENT).model_dump(mode="json")
    for row in manifest["rows"]:
        config_path = (source / row["config"]).resolve()
        if (
            not config_path.is_relative_to(source.resolve())
            or sha256_file(config_path) != row["sha256"]
        ):
            raise ValueError("loss config escaped source or changed checksum")
        expected = dict(zip(FIELDS, ARMS[row["name"]], strict=True))
        value = load_experiment_config(config_path).model_dump(mode="json")
        parameters = value["model"]["parameters"]
        if row["changes"] != expected or any(
            parameters[name]["value"] != setting for name, setting in expected.items()
        ):
            raise ValueError("loss arm settings changed")
        unchanged = copy.deepcopy(value)
        for name in FIELDS:
            unchanged["model"]["parameters"].pop(name)
        if unchanged != baseline:
            raise ValueError("loss arm changed an unrelated data/model/training setting")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--prior-complete-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("required allocator contract is missing")
    if not all(
        p.resolve().is_relative_to("/data/yilangliu") for p in (args.queue_root, args.prior_run)
    ):
        parser.error("scientific work and queue evidence stay on server")
    source = Path(__file__).resolve().parents[2]
    manifest = verify_manifest(source, args.manifest)
    if sha256_file(args.prior_run / "COMPLETE.json") != args.prior_complete_sha256:
        parser.error("preceding A2 terminal checksum changed")
    prior = json.loads((args.prior_run / "fit/epoch_state.json").read_text())
    if prior["epoch"] != 6 or prior["best"] is not None or prior["last"]["epoch"] != 6:
        parser.error("preceding A2 final-only six-epoch prerequisite changed")
    rows = [
        {
            "name": row["name"],
            "plan": resolve_plan(
                argparse.Namespace(
                    config=source / row["config"],
                    runtime=args.runtime,
                    data_root=None,
                    gpu="0,1",
                    seed=1,
                )
            ),
            "probe_kind": "preflight_only",
        }
        for row in manifest["rows"]
    ]
    queue = {
        "schema": "gradpert-v2-loss-six-queue-1",
        "baseline": str(args.prior_run),
        "baseline_complete_sha256": args.prior_complete_sha256,
        "rows": rows,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
    }
    args.queue_root.mkdir(parents=True, exist_ok=False)
    atomic_json(args.queue_root / "queue.json", queue)
    if args.execute:
        execute(queue, args.queue_root)
    else:
        print(json.dumps({"queue_root": str(args.queue_root), "status": "planned_not_launched"}))


if __name__ == "__main__":
    main()
