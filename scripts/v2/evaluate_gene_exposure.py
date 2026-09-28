"""Re-evaluate one v2 checkpoint by actual training-expression exposure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-root", "training-run", "exposure", "publication", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exposure-sha256", required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    args = parser.parse_args()
    if any(
        not path.resolve().is_relative_to("/data/yilangliu")
        for path in (args.data_root, args.training_run, args.exposure, args.output)
    ):
        raise ValueError("scientific inputs and receipt must stay on the server")
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    import torch

    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json
    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.evaluation.state import load_evaluation_state
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.evaluation import evaluate
    from gradpert.training.v2.runtime import prepare_runtime

    if sha256_file(args.exposure) != args.exposure_sha256:
        raise ValueError("exposure replay receipt hash differs")
    exposure = json.loads(args.exposure.read_text())
    training = json.loads((args.training_run / "run_manifest.json").read_text())
    journal = json.loads((args.training_run / "fit/epoch_state.json").read_text())
    selected = journal["best"]
    if (
        journal["identity"] != training
        or exposure["training_identity"] != training
        or exposure["checkpoint"] != selected
        or exposure["checkpoint_role"] != "best"
        or not exposure["exposure_complete"]
        or sha256_file(args.config) != training["config_sha256"]
    ):
        raise ValueError("exposure proof, checkpoint and training configuration differ")
    checkpoint = args.training_run / "fit" / selected["file"]
    if sha256_file(checkpoint) != selected["sha256"]:
        raise ValueError("selected checkpoint checksum differs")
    config = load_experiment_config(args.config)
    source = inspect_source_identity(
        Path(__file__).resolve().parents[2],
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    environment = inspect_environment(Path(__file__).resolve().parents[2], device_name="cuda:0")
    with prepare_runtime(
        config,
        data_root=args.data_root,
        run_seed=training["data"]["run_seed"],
        device=torch.device("cuda:0"),
        purpose="evaluation",
    ) as runtime:
        if runtime.identity != training["data"]:
            raise ValueError("evaluation data/graph/GenePT identity differs from training")
        load_evaluation_checkpoint(
            checkpoint,
            runtime.objective,
            training_identity=training,
            checkpoint_sha256=selected["sha256"],
        )
        with CanonicalEvaluationData(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=args.data_root,
            split_name="test",
        ) as data:
            all_genes = set(data.expression_gene_ids)
            seen_ids = exposure["seen_expression_gene_ids"]
            unseen_ids = exposure["unseen_expression_gene_ids"]
            if (
                set(seen_ids).isdisjoint(unseen_ids)
                and set(seen_ids) | set(unseen_ids) == all_genes
                and sha256_json(seen_ids) == exposure["seen_expression_gene_ids_sha256"]
                and sha256_json(unseen_ids) == exposure["unseen_expression_gene_ids_sha256"]
            ) is False:
                raise ValueError("exposure groups do not partition the frozen expression axis")
            positions = {gene: i for i, gene in enumerate(data.expression_gene_ids)}
            groups = {
                "seen_expression": tuple(positions[g] for g in seen_ids),
                "unseen_expression": tuple(positions[g] for g in unseen_ids),
            }
            reference = load_evaluation_state(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                data_root=args.data_root,
            )
            result = evaluate(
                runtime.objective.student,
                runtime.index,
                data,
                reference,
                expected_split="test",
                device=torch.device("cuda:0"),
                cell_batch=int(config.training.eval_batch_size.value),
                query_count=runtime.options.eval_query_count,
                metric_gene_groups=groups,
            )
    baseline = json.loads((args.training_run / "fit/best-test.json").read_text())
    if (
        baseline["identity"]["checkpoint"] != selected
        or baseline["result"]["query_recipe"] != result["query_recipe"]
        or baseline["result"]["control_manifest_sha256"] != result["control_manifest_sha256"]
        or baseline["result"]["reference_sha256"] != result["reference_sha256"]
    ):
        raise ValueError("full-gene comparison is not on the same evaluation protocol")
    previous = {m["metric_id"]: m for m in baseline["result"]["metrics"]}
    for metric in result["metrics"]:
        old = previous[metric["metric_id"]]
        if metric["finite_condition_count"] != old["finite_condition_count"]:
            raise ValueError("repeated full-gene evaluation changed valid condition count")
        if abs(metric["macro_mean"] - old["macro_mean"]) > 1e-5:
            raise ValueError("repeated full-gene evaluation differs from original best test")
    atomic_json(
        args.output,
        {
            "status": "passed",
            "kind": "v2_best_test_expression_exposure_stratification",
            "training_identity": training,
            "evaluation_source": source.payload(),
            "evaluation_environment": environment.payload(),
            "checkpoint": selected,
            "exposure_receipt_sha256": args.exposure_sha256,
            "original_best_test_sha256": sha256_file(args.training_run / "fit/best-test.json"),
            "result": result,
        },
    )
    print(json.dumps({"status": "passed", "output": str(args.output)}))


if __name__ == "__main__":
    main()
