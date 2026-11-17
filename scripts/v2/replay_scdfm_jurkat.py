"""Restore the sealed scDFM cap40 fit and evaluate an ordered Jurkat test slice."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from benchmarks.expression_holdout import scdfm  # noqa: E402
from benchmarks.expression_holdout.contract import FittedAdapter  # noqa: E402
from benchmarks.expression_holdout.evaluation import evaluate  # noqa: E402
from benchmarks.expression_holdout.runner import restore_last  # noqa: E402
from gradpert.data._io import atomic_json, read_json  # noqa: E402
from gradpert.evaluation.data import CanonicalEvaluationData  # noqa: E402
from gradpert.evaluation.state import load_evaluation_state  # noqa: E402
from gradpert.execution.identity import inspect_source_identity  # noqa: E402
from gradpert.hashing import sha256_file, sha256_json  # noqa: E402
from scripts.v2.run_group import lock  # noqa: E402
from scripts.v2.run_small_softmax import admit_selected_idle  # noqa: E402
from scripts.v2.shared_gpu import process_info, snapshot  # noqa: E402

SERVER = Path("/data/yilangliu/GraD-Pert")
DATA = SERVER / "data-vnext-a942114"
CHECKOUT = SERVER / "upstreams/scdfm-8de47e7"
STUDY = SERVER / "development/perturbation-holdout-study-20261117-a1"
FIT = (
    SERVER
    / "runs-official-recovery-9607981"
    / "scdfm-jurkat-seed1-20261013T103748Z-1cba65fce8c941a49ea7cdf6a6f08455"
)
FIT_HASHES = {
    "fit/last.pt": "9dff79f7e2836482ea6f272dd619c9df967e73f8e5b16f911679c3d557c62a8c",
    "fit/coexpression-mask.pt": "0b3db555c298bcddbee726e0c90b727b8c8e0e54b2a2dd1a2330d0ca7133a4e4",
    "fit-receipt.json": "aa00d18a0f49fba76bd8035134f71653728580000054eb29790602fcfcb4160c",
    "data-receipt.json": "89ae735cb13081774cf8355c855a7c2c3091b9a68d7a7bd1167ed3682e2d9bf2",
}
COHORT_SHA = "543a5b5638030da5df5790423ad7efec636e298e12089b5d98b534688cb9d4f4"
DATASET = "nadig_jurkat"
PROTOCOL = "within_cell_unseen_single"


def evaluate_slice(args, output):
    for filename, digest in FIT_HASHES.items():
        if sha256_file(FIT / filename) != digest:
            raise ValueError(f"historical scDFM input changed: {filename}")
    cohort_path = STUDY / "frozen/cohorts.json"
    if sha256_file(cohort_path) != COHORT_SHA:
        raise ValueError("Jurkat cohort receipt changed")
    cohort = read_json(cohort_path)["jurkat"]
    fit_receipt = read_json(FIT / "fit-receipt.json")
    data_receipt = read_json(FIT / "data-receipt.json")
    if (
        not fit_receipt["training_complete"]
        or fit_receipt["completed_updates"] != 3988
        or fit_receipt["n_train_perturbed_rows"] != 47836
        or fit_receipt["official_commit"] != scdfm.OFFICIAL_SHA
        or data_receipt["train_selection"]["cap"] != 40
        or not data_receipt["expression_policy"]["exclude_test_target_expression"]
        or data_receipt["test_truth_present_during_fit"]
        or data_receipt["canonical_data_sha256"] != cohort["canonical_h5ad_sha256"]
        or data_receipt["split_content_sha256"] != cohort["split_content_sha256"]
        or data_receipt["test_condition_ids_sha256"] != cohort["condition_ids_sha256"]["test"]
    ):
        raise ValueError("historical scDFM fitting exposure differs from cap40 Jurkat")
    reference = load_evaluation_state(
        dataset_id=DATASET, protocol_id=PROTOCOL, data_root=DATA, evaluation_protocol="v2"
    )
    with CanonicalEvaluationData(dataset_id=DATASET, protocol_id=PROTOCOL, data_root=DATA) as data:
        if (
            data.control_manifest_file_sha256 != fit_receipt["control_manifest_sha256"]
            or data.control_manifest_file_sha256 != cohort["evaluation_controls_sha256"]
            or reference.manifest_file_sha256 != fit_receipt["evaluation_state_sha256"]
            or data.manifest.expression_gene_order_sha256 != cohort["expression_gene_order_sha256"]
            or len(data.split.test_conditions) != 592
        ):
            raise ValueError("frozen Jurkat evaluation inputs changed")
        genes = tuple(data.expression_gene_ids)
        graph_genes = (
            (DATA / DATASET / PROTOCOL / "canonical/graph_gene_ids.txt").read_text().splitlines()
        )
        if (
            graph_genes[: len(genes)] != list(genes)
            or sha256_json(graph_genes) != data_receipt["graph_gene_order_sha256"]
        ):
            raise ValueError("scDFM graph and vocabulary axis changed")
        seen = tuple(fit_receipt["observed_training_expression_gene_indices"])
        held = tuple(g for i, g in enumerate(genes) if i not in set(seen))
        if held != tuple(cohort["masked_genes_in_expression_order"]):
            raise ValueError("numerical fit exposure differs from frozen mask")
        official = scdfm.load_official(CHECKOUT)
        device = torch.device("cuda:0")
        with scdfm.official_namespace(CHECKOUT, official.modules):
            vocab = official.tokenizer.GeneVocab(
                graph_genes, specials=["<pad>", "<cls>", "<mask>", "control"]
            )
            shape = fit_receipt["actual_model"]
            if len(vocab) != shape["ntoken"] or shape["nlayers"] != 4:
                raise ValueError("scDFM official checkpoint architecture differs")
            model = official.origin.model(
                ntoken=len(vocab),
                nlayers=shape["nlayers"],
                fusion_method="differential_perceiver",
                perturbation_function="crisper",
                mask_path=str(FIT / "fit/coexpression-mask.pt"),
            ).to(device)
            state = scdfm.ScDFMState(
                model, official, CHECKOUT, vocab, genes, device, fit_receipt["resolved_training"]
            )
            fitted = FittedAdapter(state, {})
            restore_last(fitted, "scdfm", FIT / "fit/last.pt")
        model.eval().requires_grad_(False)

        def predictor(*inputs, **kwargs):
            with torch.no_grad():
                return scdfm.predict(fitted, *inputs, **kwargs)

        started = time.monotonic()

        def progress(done, total, condition):
            atomic_json(
                output / "progress.json",
                {
                    "stage": "frozen_inference",
                    "conditions_completed": done,
                    "conditions_total": total,
                    "latest_condition": condition,
                    "evaluation_seconds": time.monotonic() - started,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
                },
            )

        if not 0 <= args.start < args.stop <= len(data.split.test_conditions):
            raise ValueError("requested interval falls outside the locked test draw order")
        atomic_json(
            output / "progress.json",
            {
                "stage": "checkpoint_restored",
                "conditions_completed": 0,
                "conditions_total": args.stop - args.start,
                "first_condition": data.split.test_conditions[args.start],
            },
        )
        result = evaluate(
            data,
            reference,
            predictor,
            seen_indices=seen,
            cell_batch=128,
            query_count=1000,
            seed=1,
            progress=progress,
            condition_slice=slice(args.start, args.stop),
        )
        atomic_json(output / "last-test.json", result)
        atomic_json(output / "noise-receipt.json", fitted.receipt["inference_noise"])
        return {
            "checkpoint_sha256": FIT_HASHES["fit/last.pt"],
            "checkpoint_origin": str(FIT),
            "fit_updates": fit_receipt["completed_updates"],
            "optimizer_updates": 0,
            "cap": 40,
            "condition_start": args.start,
            "condition_stop": args.stop,
            "conditions": len(result["condition_metrics"]),
            "expected_full_conditions": len(data.split.test_conditions),
            "frozen_evaluation_sha256": sha256_file(output / "last-test.json"),
            "noise_receipt_sha256": sha256_file(output / "noise-receipt.json"),
            "evaluation_seconds": time.monotonic() - started,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--gpu", required=True, choices=("0", "1"))
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--stop", required=True, type=int)
    args = parser.parse_args()
    if any(not p.resolve().is_relative_to(STUDY) for p in (args.output, args.publication)):
        raise ValueError("Jurkat replay artifacts stay on the study server")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu
        or os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("CUDA execution envelope changed")
    identity = inspect_source_identity(
        ROOT,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), int(args.gpu), args.uuid)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        args.output.mkdir(exist_ok=False)
        owner = {
            "source_commit": identity.commit,
            "cohorts_sha256": COHORT_SHA,
            "checkout_commit": scdfm.OFFICIAL_SHA,
            "owner": process_info(os.getpid()),
            "gpu": int(args.gpu),
            "gpu_uuid": args.uuid,
        }
        atomic_json(args.output / "OWNER.json", owner)
        try:
            receipt = evaluate_slice(args, args.output)
            if list(args.output.rglob("*.pkl")):
                raise ValueError("frozen replay produced a scientific pickle")
            atomic_json(args.output / "COMPLETE.json", {**owner, **receipt})
        except BaseException as error:
            atomic_json(
                args.output / "FAILURE.json",
                {**owner, "error_type": type(error).__name__, "error": str(error)},
            )
            raise


if __name__ == "__main__":
    main()
