"""Replay v2 training view sampling to certify expression-gene exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.hashing import sha256_file, sha256_json
from gradpert.training.v2.runtime import prepare_runtime
from gradpert.training.v2.views import assemble_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-root", "training-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--role", choices=("best", "last"), default="best")
    args = parser.parse_args()
    if any(
        not path.resolve().is_relative_to("/data/yilangliu")
        for path in (args.data_root, args.training_run, args.output)
    ):
        raise ValueError("scientific inputs and receipt must stay on the server")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.training_run / "run_manifest.json").read_text())
    journal = json.loads((args.training_run / "fit/epoch_state.json").read_text())
    selected = journal[args.role]
    if journal["identity"] != manifest or journal["epoch"] < selected["epoch"]:
        raise ValueError("selected checkpoint is not committed in the training journal")
    if sha256_file(args.config) != manifest["config_sha256"]:
        raise ValueError("training configuration differs from the checkpoint")
    checkpoint = args.training_run / "fit" / selected["file"]
    if sha256_file(checkpoint) != selected["sha256"]:
        raise ValueError("selected checkpoint hash differs")
    config = load_experiment_config(args.config)
    with prepare_runtime(
        config,
        data_root=args.data_root,
        run_seed=manifest["data"]["run_seed"],
        device=torch.device("cpu"),
        purpose="evaluation",
    ) as runtime:
        if runtime.identity != manifest["data"]:
            raise ValueError("replay data and view configuration differ from training")
        if runtime.allowed_expression_ids is None:
            raise ValueError("this analysis requires an explicit expression eligibility pool")
        all_ids = set(range(len(runtime.data.expression_gene_ids)))
        seen: set[int] = set()
        first_full: list[int] | None = None
        steps = 0
        for epoch in range(selected["epoch"]):
            for raw in runtime.data.iter_train_epoch(
                epoch=epoch,
                device=torch.device("cpu"),
                batch_size=runtime.batch_size,
                max_unique_conditions=(
                    0
                    if runtime.options.max_conditions == 0
                    else min(runtime.options.max_conditions, runtime.batch_size)
                ),
            ):
                batch = assemble_batch(
                    raw,
                    runtime.index,
                    runtime.options,
                    runtime.generator,
                    allowed_expression_ids=runtime.allowed_expression_ids,
                )
                queries = batch.graph.ids[batch.query_positions].tolist()
                if first_full is None:
                    first_full = queries
                seen.update(int(i) for i in queries)
                steps += 1
                if seen == set(map(int, runtime.allowed_expression_ids)):
                    break
                if steps >= (epoch + 1) * journal["budget"][1]:
                    break
            if seen == set(map(int, runtime.allowed_expression_ids)):
                break
        if steps > selected["epoch"] * journal["budget"][1]:
            raise ValueError("replay exceeded checkpoint step budget")
        complete = len(seen) == len(runtime.allowed_expression_ids) or steps == (
            selected["epoch"] * journal["budget"][1]
        )
        if not complete:
            raise ValueError("exposure replay stopped before it could classify every gene")
        axes = runtime.data.expression_gene_ids
        seen_ids = [axes[i] for i in sorted(seen)]
        unseen_ids = [axes[i] for i in sorted(all_ids - seen)]
        allowed = [axes[int(i)] for i in runtime.allowed_expression_ids]
        if sha256_json(allowed) != manifest["data"]["effective_training_expression_ids_sha256"]:
            raise ValueError("allowed expression gene identity changed")
        atomic_json(
            args.output,
            {
                "schema_version": "v2-expression-exposure-replay-1",
                "training_identity": manifest,
                "checkpoint_role": args.role,
                "checkpoint": selected,
                "replayed_steps": steps,
                "selected_checkpoint_steps": selected["epoch"] * journal["budget"][1],
                "all_allowed_covered": len(seen) == len(allowed),
                "exposure_complete": complete,
                "first_query_gene_ids_sha256": sha256_json([axes[i] for i in first_full or []]),
                "seen_expression_gene_ids": seen_ids,
                "unseen_expression_gene_ids": unseen_ids,
                "seen_expression_gene_ids_sha256": sha256_json(seen_ids),
                "unseen_expression_gene_ids_sha256": sha256_json(unseen_ids),
                "scope": (
                    "numeric train control/truth expression query columns only; "
                    "excludes validation, test and gene-identity priors"
                ),
            },
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "replayed_steps": steps,
                "seen": len(seen_ids),
                "unseen": len(unseen_ids),
            }
        )
    )


if __name__ == "__main__":
    main()
