"""Run fixed-axis G1 context evaluation on a selected completed-run checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def selected_checkpoint(root: Path, role: str) -> tuple[dict, dict, Path]:
    from gradpert.hashing import sha256_file

    if role not in ("best", "last"):
        raise ValueError("checkpoint role must be best or last")
    manifest = json.loads((root / "run_manifest.json").read_text())
    journal = json.loads((root / "fit/epoch_state.json").read_text())
    if journal["identity"] != manifest or journal["epoch"] != 50 or journal["budget"][0] != 50:
        raise ValueError("context evaluation requires a completed fixed-50 training journal")
    selected = journal[role]
    checkpoint = (root / "fit" / selected["file"]).resolve()
    if (
        not checkpoint.is_relative_to((root / "fit").resolve())
        or sha256_file(checkpoint) != selected["sha256"]
    ):
        raise ValueError("selected checkpoint path or hash differs from training journal")
    return manifest, selected, checkpoint


def main(kind: str = "context") -> None:
    parser = argparse.ArgumentParser(
        description=(
            __doc__
            if kind == "context"
            else "Run fixed-population D1 response diagnostics on a completed-run checkpoint."
        )
    )
    for name in ("config", "data-root", "training-run", "protocol", "publication", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--role", choices=("best", "last"), required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    args = parser.parse_args()
    for path in (args.data_root, args.training_run, args.output):
        if not path.resolve().is_relative_to("/data/yilangliu"):
            parser.error("scientific data, checkpoints and evaluation outputs stay on the server")
    if args.output.exists():
        parser.error("evaluation output must be new")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json
    from gradpert.evaluation.data import CanonicalEvaluationData
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.context_evaluation import evaluate_contexts
    from gradpert.training.v2.runtime import prepare_runtime

    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError("evaluation protocol checksum mismatch")
    protocol = json.loads(args.protocol.read_text())
    fields = (
        {"evaluation_gene_ids", "budgets", "context_seed", "split"}
        if kind == "context"
        else {"query_gene_ids", "condition_id", "alternative_condition_id", "split"}
    )
    if set(protocol) != fields or protocol["split"] not in ("val", "test"):
        raise ValueError("context protocol requires explicit fixed genes, budgets, seed and split")
    config = load_experiment_config(args.config)
    training, selected, checkpoint = selected_checkpoint(args.training_run, args.role)
    if training["config_sha256"] != sha256_file(args.config):
        raise ValueError("evaluation config differs from checkpoint training config")
    repository = Path(__file__).resolve().parents[2]
    source = inspect_source_identity(
        repository,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    environment = inspect_environment(repository, device_name="cuda:0")
    with prepare_runtime(
        config,
        data_root=args.data_root,
        run_seed=training["data"]["run_seed"],
        device=torch.device("cuda:0"),
    ) as runtime:
        if runtime.identity != training["data"]:
            raise ValueError("evaluation canonical data/graph/GenePT differs from training")
        progress = load_evaluation_checkpoint(
            checkpoint,
            runtime.objective,
            training_identity=training,
            checkpoint_sha256=selected["sha256"],
        )
        gene_positions = {g: i for i, g in enumerate(runtime.data.expression_gene_ids)}

        with CanonicalEvaluationData(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=args.data_root,
            split_name=protocol["split"],
        ) as data:
            if kind == "context":
                result = evaluate_contexts(
                    runtime.objective.student,
                    runtime.index,
                    data,
                    evaluation_ids=tuple(
                        gene_positions[g] for g in protocol["evaluation_gene_ids"]
                    ),
                    budgets=tuple(protocol["budgets"]),
                    context_seed=protocol["context_seed"],
                    expected_split=protocol["split"],
                    device=torch.device("cuda:0"),
                    cell_batch=int(config.training.eval_batch_size.value),
                )
            else:
                from gradpert.training.v2.diagnostics import evaluate_response_diagnostics

                result = evaluate_response_diagnostics(
                    runtime.objective.student,
                    runtime.index,
                    data,
                    query_gene_ids=tuple(protocol["query_gene_ids"]),
                    condition_id=protocol["condition_id"],
                    alternative_condition_id=protocol["alternative_condition_id"],
                    expected_split=protocol["split"],
                    device=torch.device("cuda:0"),
                    cell_batch=int(config.training.eval_batch_size.value),
                )
        receipt = {
            "status": "passed",
            "kind": kind + "_evaluation",
            "training": training,
            "evaluation_source": source.payload(),
            "evaluation_environment": environment.payload(),
            "checkpoint": selected,
            "checkpoint_role": args.role,
            "checkpoint_progress_sha256": sha256_json(progress),
            "protocol": protocol,
            "protocol_sha256": args.protocol_sha256,
            "result": result,
        }
        atomic_json(args.output, receipt)
    print(json.dumps({"output": str(args.output), "status": "passed"}))


if __name__ == "__main__":
    main()
