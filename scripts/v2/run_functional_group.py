"""Guarded B0 -> K1 -> K2 -> A1 -> A2 queue; all capacity gates precede training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from generate_functional_group import ARMS, PARENT, PARENT_SHA256
from run_cap40_ablations import execute

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.train_entry import resolve_plan
from gradpert.hashing import sha256_file


def verify_manifest(source: Path, path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != "gradpert-v2-functional-b0-ka-1"
        or manifest.get("parent") != PARENT
        or manifest.get("parent_sha256") != PARENT_SHA256
        or [r["name"] for r in manifest["rows"]] != list(ARMS)
        or manifest.get("epochs") != 6
        or manifest.get("validation") != "disabled"
        or manifest.get("test_roles") != ["last"]
    ):
        raise ValueError("functional queue differs from the five authorized arms")
    baseline = load_experiment_config(source / PARENT).model_dump(mode="json")
    for row in manifest["rows"]:
        config_path = (source / row["config"]).resolve()
        if (
            not config_path.is_relative_to(source.resolve())
            or sha256_file(config_path) != row["sha256"]
        ):
            raise ValueError("functional config escaped source or changed checksum")
        config = load_experiment_config(config_path).model_dump(mode="json")
        if row["changes"] != ARMS[row["name"]]:
            raise ValueError("functional arm identity changed")
        p = config["model"]["parameters"]
        expected = {
            "validation_mode": "disabled",
            "microbatch": manifest["common_microbatch"],
            **ARMS[row["name"]],
        }
        parent_p = baseline["model"]["parameters"]
        changed = {
            k
            for k in set(p) | set(parent_p)
            if p.get(k, {}).get("value") != parent_p.get(k, {}).get("value")
        }
        if not changed <= set(expected) or any(p[k]["value"] != v for k, v in expected.items()):
            raise ValueError("functional arm changed an unrelated model setting")
        if (
            config["data"] != baseline["data"]
            or config["evaluation"] != baseline["evaluation"]
            or config["model"]["exclude_test_target_expression"]
            != baseline["model"]["exclude_test_target_expression"]
        ):
            raise ValueError("functional arm changed data or evaluation protocol")
        training = config["training"]
        if (
            training["formal_run_policy"] != "v2_fixed_6"
            or training["max_epochs"]["value"] != 6
            or training["monitor"] != "none"
            or training["monitor_mode"] != "none"
            or training["train_batch_size"]["value"] != manifest["global_batch"]
            or training["run_seeds"] != [1]
            or "continuation" in config
        ):
            raise ValueError("functional fresh training protocol changed")
        for key in set(training) - {
            "formal_run_policy",
            "max_epochs",
            "monitor",
            "monitor_mode",
            "train_batch_size",
        }:
            if training[key] != baseline["training"][key]:
                raise ValueError("functional arm changed optimizer or training settings")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--prior-evaluation", type=Path, required=True)
    parser.add_argument("--prior-complete-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("required allocator contract is missing")
    if not all(
        p.resolve().is_relative_to("/data/yilangliu")
        for p in (args.queue_root, args.prior_evaluation)
    ):
        parser.error("scientific work and queue evidence stay on server")
    source = Path(__file__).resolve().parents[2]
    manifest = verify_manifest(source, args.manifest)
    if sha256_file(args.prior_evaluation / "COMPLETE.json") != args.prior_complete_sha256:
        parser.error("accepted preceding evaluation checksum changed")
    rows = []
    for row in manifest["rows"]:
        plan = resolve_plan(
            argparse.Namespace(
                config=source / row["config"],
                runtime=args.runtime,
                data_root=None,
                gpu="0,1",
                seed=1,
            )
        )
        rows.append({"name": row["name"], "plan": plan, "probe_kind": "capacity_only"})
    queue = {
        "schema": "gradpert-v2-functional-b0-ka-queue-1",
        "baseline": str(args.prior_evaluation),
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
