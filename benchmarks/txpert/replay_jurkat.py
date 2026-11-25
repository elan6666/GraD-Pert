"""Replay official TxPert Jurkat checkpoints using the common frozen evaluator."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import lightning
import torch
from gspp import predictor
from gspp.data import datamodule, graphmodule

from benchmarks.common.condition_response import FrozenConditionResponse, summarize_response
from benchmarks.crosscell.study_paths import COHORT_SHA, DATA, DATASET, PROTOCOL, SERVER, STUDY
from benchmarks.crosscell.study_paths import REPO as SOURCE
from benchmarks.expression_holdout.evaluation import evaluate
from benchmarks.txpert.official_api import OfficialPublicAPI, OfficialPublicModules
from benchmarks.txpert.runner import _official_config, _register_anndata_null_reader
from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.data import CanonicalEvaluationData
from gradpert.evaluation.state import load_evaluation_state
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file
from scripts.v2.shared_gpu import process_info, snapshot

CHECKOUT = SERVER / "upstreams/txpert"
COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
FIT = SERVER / "runs/r50-tx64-7005aa7-v1/full"
FIT_HASHES = {
    "checkpoints/last.ckpt": "b88784e54473c7e28e722d908e17567e76bb19c1537df7f23f0e3207f9b04f1f",
    "small_results/config.resolved.yaml": (
        "593f449ffd3bec72d60bb471502b83fb5c6d2cdeb6224d430cd44910bb16e12e"
    ),
    "small_results/training_receipt.json": (
        "ba4f8107ea8ad80f81be4680f6d8d17d1774936760152e6a1b82c3bc97d24645"
    ),
    "small_results/official_checkout.json": (
        "fbddce44ecce55c7305732c0de56e43b7b0121c3fdf13651fa5dcc602c316190"
    ),
    "small_results/official_data_adapter.json": (
        "5f57913cc2afb5a85d0dd7ba748f57f6d9fa6765623702baf847b6d908198d1f"
    ),
    "small_results/training_data.json": (
        "8b50801097f94daaa20e9d9303fc7b974e128f0ed58172e9bcf5190c5f7ee538"
    ),
    "evaluations/last/small_results/checkpoint_role.json": (
        "6d62a0b441c44ce3fe5b13ae2e30523d8579df71c5bef92d57955bbdc914a397"
    ),
}


J2_FIT = STUDY / "txpert-jurkat-j2-released-scratch-full-a1"
J2_FIT_HASHES = {
    "checkpoints/best.ckpt": "783ad20fedad57818ad6f3be418549ad67068eedea24fceebe6388613799b4ef",
    "small_results/config.resolved.yaml": (
        "7f6464011c7848f2debc73fcc21495d553e3a40d47416b2f151ed25983afa45c"
    ),
    "small_results/training_receipt.json": (
        "8f2f71d88f51a29b4cafea2755bf7603afe74046809668c011dad85561c5e475"
    ),
    "small_results/double_holdout_fit.json": (
        "4d58111505399e2b049c7006852420ae0f8a0b8c767b2ec62e27cfc44098ce48"
    ),
    "small_results/official_data_adapter.json": (
        "2da6884cf314a37ab2c15c982577f7113152b6fa9d079c06c04fb7465c7a2d74"
    ),
    "small_results/training_data.json": FIT_HASHES["small_results/training_data.json"],
    "evaluations/best/small_results/checkpoint_role.json": (
        "7ac80c418074f97a578c16a5b48b6f33fe6f13ac0c0b633a8011ca2ba80bbbc4"
    ),
}


def evaluate_frozen(output, stop, *, double_holdout=False, common_panel=False):
    fit, hashes, checkpoint_role = (
        (J2_FIT, J2_FIT_HASHES, "best") if double_holdout else (FIT, FIT_HASHES, "last")
    )
    checkpoint = f"checkpoints/{checkpoint_role}.ckpt"
    for name, digest in hashes.items():
        if sha256_file(fit / name) != digest:
            raise ValueError(f"historical TxPert fit input changed: {name}")
    if (
        not Path(predictor.__file__).resolve().is_relative_to(CHECKOUT)
        or subprocess.check_output(
            ["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True
        ).strip()
        != COMMIT
        or subprocess.check_output(
            ["git", "-C", str(CHECKOUT), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("the official TxPert source differs from the historical fit")
    cohort_path = STUDY / "frozen/cohorts.json"
    if sha256_file(cohort_path) != COHORT_SHA:
        raise ValueError("frozen Jurkat cohort identity changed")
    cohort = read_json(cohort_path)["jurkat"]
    fitted_data = read_json(fit / "small_results/training_data.json")
    fitted_adapter = read_json(fit / "small_results/official_data_adapter.json")
    training = read_json(fit / "small_results/training_receipt.json")
    role = read_json(fit / f"evaluations/{checkpoint_role}/small_results/checkpoint_role.json")
    if (
        training["epochs_completed"] != 50
        or role["checkpoint_epoch"] != 50
        or role["checkpoint_sha256"] != hashes[checkpoint]
        or fitted_data["train_perturbed_cell_count"] != 128266
        or fitted_data["canonical_data_sha256"] != cohort["canonical_h5ad_sha256"]
        or fitted_data["split_content_sha256"] != cohort["split_content_sha256"]
        or fitted_adapter[
            "test_conditions_sha256"
            if double_holdout
            else "canonical_test_conditions_excluded_sha256"
        ]
        != cohort["condition_ids_sha256"]["test"]
        or fitted_adapter["canonical_test_truth_present"]
    ):
        raise ValueError("last checkpoint or full training exposure has changed")
    held_genes = ()
    if double_holdout:
        policy = read_json(fit / "small_results/double_holdout_fit.json")
        if (
            policy["cohort_sha256"] != COHORT_SHA
            or policy["held_genes"] != cohort["masked_genes_in_expression_order"]
            or policy["test_truth_present_during_fit"]
            or policy["fit_allowed_gene_count"] != 4775
            or training["optimizer_steps"] != 109550
        ):
            raise ValueError("Jurkat J2 checkpoint does not have the accepted double-holdout fit")
        held_genes = tuple(policy["held_genes"])
    archived_config = fit / "small_results/config.resolved.yaml"
    config_path = output / "config/txpert_public/nadig_jurkat.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(archived_config.read_bytes())
    if sha256_file(config_path) != role["config_sha256"]:
        raise ValueError("archived TxPert experiment configuration changed")
    config = load_experiment_config(config_path)
    _, official = _official_config(config, CHECKOUT)
    dims = cohort["expression_panel_genes"] if double_holdout else fitted_adapter["gene_count"]
    mapping = datamodule.PertDataModule(
        batch_size=64,
        match_cntr=True,
        avg_cntr=True,
        embed_cntr=True,
        obsm_key="raw",
        task_type="gradpert_canonical_adapter",
        train_cell_types=[fitted_adapter["cell_type"]],
        test_cell_type=fitted_adapter["cell_type"],
        suppress_cell_type_validation=True,
    )
    mapping.input_dim = mapping.output_dim = mapping.adata_output_dim = dims
    api = OfficialPublicAPI(
        OfficialPublicModules(predictor, datamodule, graphmodule, lightning, torch)
    )
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    model = api.build_model(
        data_module=mapping,
        model_args=official["model"],
        graph_args=official["graph"],
        learning_rate=float(config.training.learning_rate.value),
        weight_decay=float(config.training.weight_decay.value),
        device="cuda:0",
        match_control_for_eval=True,
    )
    state = torch.load(fit / checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    if model.basal_mode != "construct_control":
        raise ValueError("forward and sample-inference recipes differ")
    reference = load_evaluation_state(
        dataset_id=DATASET, protocol_id=PROTOCOL, data_root=DATA, evaluation_protocol="v2"
    )
    train_reference = load_evaluation_state(
        dataset_id=DATASET,
        protocol_id=PROTOCOL,
        data_root=DATA,
        evaluation_protocol="v2",
        validation_only=True,
    )
    reader_compatibility = _register_anndata_null_reader()
    if common_panel:
        from benchmarks.jurkat_j2.study import evaluate as evaluate_common

        if not double_holdout or stop != 592:
            raise ValueError("common-panel replay is defined on the complete frozen J2 test")
        result = evaluate_common(
            lambda condition, controls, _rows, axis: api.predict_exact_controls(
                trained_model=model,
                perturbation_genes=(*[g for g in condition.split("+") if g != "ctrl"], "ctrl"),
                perturbation_to_id=mapping.pert2id,
                input_controls=controls,
                batch_size=int(config.training.eval_batch_size.value),
            )[:, axis],
            output,
            journal_identity=hashes[checkpoint],
        )
        result.update(
            checkpoint_sha256=hashes[checkpoint],
            checkpoint_epoch=50,
            model="txpert",
            scenario="J2",
            status="evaluated",
            training_source_commit=role["training_git_sha"],
            run_id=output.name,
        )
        atomic_json(output / "result.json", result)
        return result

    with CanonicalEvaluationData(
        dataset_id=DATASET, protocol_id=PROTOCOL, split_name="test", data_root=DATA
    ) as data:
        if (
            data.control_manifest_file_sha256 != cohort["evaluation_controls_sha256"]
            or data.manifest.expression_gene_order_sha256 != cohort["expression_gene_order_sha256"]
            or len(data.expression_gene_ids) != dims
            or not 0 < stop <= len(data.split.test_conditions)
        ):
            raise ValueError("TxPert frozen Jurkat inputs differ")
        atomic_json(output / "progress.json", {"stage": "reference", "conditions_total": stop})
        response = FrozenConditionResponse(data)

        def predict_for_benchmark(controls, targets, _genes, *, cell_batch, **_):
            return api.predict_exact_controls(
                trained_model=model,
                perturbation_genes=(*targets, "ctrl"),
                perturbation_to_id=mapping.pert2id,
                input_controls=controls,
                batch_size=cell_batch,
            )

        def progress(done, total, condition):
            if done == 1 or done % 5 == 0 or done == total:
                atomic_json(
                    output / "progress.json",
                    {
                        "stage": "inference",
                        "conditions_completed": done,
                        "conditions_total": total,
                        "latest_condition": condition,
                    },
                )

        result = evaluate(
            data,
            reference,
            predict_for_benchmark,
            seen_indices=tuple(
                i for i, gene in enumerate(data.expression_gene_ids) if gene not in held_genes
            ),
            cell_batch=int(config.training.eval_batch_size.value),
            query_count=dims,
            seed=1,
            progress=progress,
            condition_slice=slice(0, stop),
            response_scorer=response,
            train_systema_state=train_reference,
        )
    result["canonical_reader_compatibility"] = reader_compatibility
    result["checkpoint_role"] = checkpoint_role
    result["prediction_condition_encoding"] = "official_single_target_and_ctrl"
    result["response_summary"] = summarize_response(result["condition_metrics"])
    atomic_json(output / f"{checkpoint_role}-test.json", result)
    return {
        "conditions": len(result["condition_metrics"]),
        "optimizer_updates": 0,
        "fit_optimizer_updates": training["optimizer_steps"],
        "checkpoint_epoch": 50,
        "checkpoint_role": checkpoint_role,
        "checkpoint_sha256": hashes[checkpoint],
        "training_source_commit": role["training_git_sha"],
        "evaluation_sha256": sha256_file(output / f"{checkpoint_role}-test.json"),
        "retrieval_bank_conditions": len(response.conditions),
        "systema_reference_scope": result["systema_reference_scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop", type=int, default=592)
    parser.add_argument("--double-holdout", action="store_true")
    parser.add_argument("--common-panel", action="store_true")
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    parser.add_argument("--uuid", required=True)
    args = parser.parse_args()
    if (
        any(not p.resolve().is_relative_to(STUDY) for p in (args.output, args.publication))
        or os.getenv("CUDA_VISIBLE_DEVICES") != args.gpu
        or os.getenv("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.getenv("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("server data and CUDA execution envelope changed")
    source = inspect_source_identity(
        SOURCE,
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
        started = time.monotonic()
        owner = {
            "source_commit": source.commit,
            "checkout_commit": COMMIT,
            "gpu": int(args.gpu),
            "gpu_uuid": args.uuid,
            "owner": process_info(os.getpid()),
            "cohorts_sha256": COHORT_SHA,
        }
        atomic_json(args.output / "OWNER.json", owner)
        try:
            receipt = evaluate_frozen(
                args.output,
                args.stop,
                double_holdout=args.double_holdout,
                common_panel=args.common_panel,
            )
            atomic_json(
                args.output / "COMPLETE.json",
                {
                    **owner,
                    **receipt,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        except BaseException as error:
            atomic_json(
                args.output / "FAILURE.json",
                {
                    **owner,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    main()
