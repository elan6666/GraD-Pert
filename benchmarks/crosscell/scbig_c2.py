"""Continue the released scBIG C2 flow and evaluate the frozen 128-condition cohort."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from benchmarks.common import official_module_session
from benchmarks.crosscell.scbig_c2_flow_fit import (
    COVERAGE,
    COVERAGE_SHA,
    GENE_LOSS_PERIOD,
    STAGE1_RECEIPT_SHA,
    restore_stage1,
    verify_assets,
)
from benchmarks.crosscell.scbig_c2_flow_fit import prepare_latents as prepare_source_latents
from benchmarks.crosscell.scbig_c2_flow_probe import (
    ASSETS_RECEIPT,
    COMMIT,
    RESOURCES,
    SUPPORT_SHA,
    UPSTREAM,
    initialize_flow,
)
from benchmarks.crosscell.scbig_c2_full_axis_smoke import full_axis_grc
from benchmarks.crosscell.study_data import cohort_inputs, score
from benchmarks.crosscell.study_paths import (
    COHORT_SHA,
    SERVER,
    STUDY,
    TXPERT_CHECKOUT,
    TXPERT_COMMIT,
)
from benchmarks.expression_holdout.data import read_allowed_rows
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

PREFIX = STUDY / "scbig-crosscell-c2-budget3h-flow-prefix-a2"
PREFIX_SHA = "123370270b76a7903f5806222983814d7648edb97ae64adfbb36d15ceb7af713"
RELEASE_PROGRAM = UPSTREAM / "scbig/experiments/train_norman_additive.py"
RELEASE_PROGRAM_SHA = "074bfb1669bafa14a27386e2f791c099c330a82b9d96e9d2874e44de056abae3"
STEP_CAP = 20_000
FIT_SECONDS = 4_500
WALL_SECONDS = 9_900


def verify_prefix():
    manifest = PREFIX / "result.json"
    if sha256_file(manifest) != PREFIX_SHA:
        raise ValueError("the accepted step-999 receipt changed")
    receipt = read_json(manifest)
    required = {
        "status": "official_ot_before_first_gene_update",
        "flow_microsteps": 999,
        "flow_optimizer_updates": 49,
        "flow_accumulated_microsteps": 19,
        "next_gene_space_update_at": GENE_LOSS_PERIOD,
        "gene_space_updates": 0,
        "test_conditions_accessed": 0,
        "encoder_receipt_sha256": STAGE1_RECEIPT_SHA,
        "upstream_commit": COMMIT,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("the scBIG prefix did not end immediately before the gene loss")
    for name, field in (
        ("flow_train.msgpack", "flow_train_sha256"),
        ("flow_inference.msgpack", "flow_inference_sha256"),
        ("flow_rng.json", "flow_rng_sha256"),
    ):
        if sha256_file(PREFIX / name) != receipt[field]:
            raise ValueError(f"step-999 artifact changed: {name}")
    return receipt


def restore_flow(flow):
    import jax.numpy as jnp
    from flax import serialization

    solver = flow.solver
    solver.vf_state = serialization.from_bytes(
        solver.vf_state, (PREFIX / "flow_train.msgpack").read_bytes()
    )
    solver.vf_state_inference = serialization.from_bytes(
        solver.vf_state_inference, (PREFIX / "flow_inference.msgpack").read_bytes()
    )
    state = solver.vf_state
    if (int(state.step), int(state.opt_state.gradient_step), int(state.opt_state.mini_step)) != (
        999,
        49,
        19,
    ):
        raise ValueError("the restored Optax accumulation state is not step 999")
    rng = read_json(PREFIX / "flow_rng.json")
    generator = np.random.default_rng()
    generator.bit_generator.state = rng["sampler_rng"]
    return jnp.asarray(rng["jax_key"], dtype=jnp.uint32), generator


def loss_priors(panel, order, keep, compass):
    from scbig.models.gcae import load_pathway_gene_matrix

    names = (
        "9606.protein.info.v12.0.txt.gz",
        "9606.protein.links.v12.0.txt.gz",
        "ReactomePathways.gmt.zip",
    )
    manifest = RESOURCES / "release-assets-a1.json"
    if sha256_file(manifest) != ASSETS_RECEIPT:
        raise ValueError("the scBIG asset manifest changed")
    known = read_json(manifest)
    assets = tuple(RESOURCES / name for name in names)
    for asset in assets:
        if sha256_file(asset) != known[asset.name]["sha256"]:
            raise ValueError(f"the scBIG loss prior changed: {asset.name}")
    canonical = list(np.asarray(panel)[np.argsort(order)])
    info, links, reactome = map(str, assets)
    genes, labels, reproduced_order, _, _ = full_axis_grc(canonical, str(compass), info, links)
    if genes != panel or not np.array_equal(order, reproduced_order):
        raise ValueError("reproduced GRC clusters no longer match the stage-1 encoder order")
    labels = np.asarray(labels, dtype=np.int32)[keep]
    if not np.array_equal(np.unique(labels), np.arange(32)):
        raise ValueError("the visible response genes must cover all 32 released GRC groups")
    pathways, _ = load_pathway_gene_matrix(reactome, list(np.asarray(panel)[keep]), is_zip=True)
    if pathways.shape[0] == 0 or pathways.shape[1] != len(labels):
        raise ValueError("the Reactome pathway loss is missing its visible response axis")
    return labels, pathways


def official_step(flow, encoder, keep, optimizer):
    """Load only the released function bodies, without the Norman script's top-level I/O."""
    import diffrax
    import jax
    import jax.numpy as jnp
    import optax
    from scbig.losses import gene_losses

    if sha256_file(RELEASE_PROGRAM) != RELEASE_PROGRAM_SHA:
        raise ValueError("the official training-program bytes changed")
    tree = ast.parse(RELEASE_PROGRAM.read_text(), filename=str(RELEASE_PROGRAM))
    names = (
        "solve_ode_differentiable",
        "compute_gene_loss_batch",
        "e2e_gene_loss_fn_flow_only",
        "e2e_gene_loss_step_flow_only",
    )
    released_functions = {}
    for name in names:
        definitions = [
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if len(definitions) != 1:
            raise ValueError(f"the official gene-loss function is no longer unique: {name}")
        released_functions[name] = definitions[0]
    namespace = {
        "partial": partial,
        "jax": jax,
        "jnp": jnp,
        "optax": optax,
        "diffrax": diffrax,
        "cf": flow,
        "encoder": encoder,
        "flow_optimizer": optimizer,
    }
    for name in (
        "cluster_correlation_loss",
        "ppi_masked_correlation_loss",
        "gene_correlation_loss",
        "pathway_ot_loss",
        "graph_smoothness_loss",
        "direction_loss",
    ):
        namespace[name] = getattr(gene_losses, name)
    program = ast.Module(body=list(released_functions.values()), type_ignores=[])
    exec(compile(program, str(RELEASE_PROGRAM), "exec"), namespace)
    solve = jax.vmap(namespace["solve_ode_differentiable"], in_axes=(0, None, None, None, None))
    namespace["solve_ode_batch"] = solve
    loss = namespace["compute_gene_loss_batch"]
    keep_cols = np.flatnonzero(keep)

    def visible_loss(prediction, truth, *args):
        return loss(prediction[:, keep_cols], truth, *args)

    namespace["compute_gene_loss_batch"] = visible_loss
    return namespace["e2e_gene_loss_step_flow_only"]


def gene_update(step, flow, encoder, source, adata, columns, labels, pathways, optimizer, rng):
    import jax.numpy as jnp

    data = flow.train_data
    contexts = rng.choice(data.n_perturbations, size=3, replace=False)
    means = []
    for idx in contexts:
        condition = {key: jnp.asarray(value[[idx]]) for key, value in data.condition_data.items()}
        control_context = next(
            key for key, values in data.control_to_perturbation.items() if idx in values
        )
        controls = np.flatnonzero(data.split_covariates_mask == control_context)
        control_positions = rng.choice(controls, size=min(64, len(controls)), replace=False)
        treated = np.flatnonzero(data.perturbation_covariates_mask == idx)
        # The release draws these indices before its covariate-matched expression draw.
        rng.choice(treated, size=min(len(control_positions), len(treated)), replace=False)
        keys = ["pert1", "pert2", "cell_line"]
        grouping = source.obs.iloc[treated][keys].drop_duplicates()
        if len(grouping) != 1:
            raise ValueError("the released sampler mixed perturbation contexts")
        exact = (source.obs[keys].to_numpy() == grouping.to_numpy()[0]).all(axis=1)
        if not np.array_equal(np.flatnonzero(exact), treated):
            raise ValueError("the released perturbation mask and fit covariates disagree")
        selected = treated[
            rng.choice(len(treated), size=min(len(control_positions), len(treated)), replace=False)
        ]
        rows = source.obs_names.to_numpy()
        truth = read_allowed_rows(
            adata, rows[selected].astype(np.int64), columns, adata.n_vars
        ).toarray()
        state = flow.solver.vf_state
        params, _, _, optimizer, _, _, total, parts = step(
            state.params,
            optimizer,
            jnp.asarray(data.cell_data[control_positions]),
            condition,
            jnp.asarray(truth),
            encoder.state.params,
            encoder.X_mean,
            encoder.ppi_mask,
            encoder.scgpt_pe,
            pathways,
            None,
            None,
            None,
            labels,
            None,
            0.1,
            0.1,
            0.0,
            0.0,
            False,
            True,
            32,
            0.0,
        )
        record = {name: float(parts[name]) for name in ("corr", "ot")}
        record["total"] = float(total)
        if not np.isfinite(list(record.values())).all():
            raise ValueError("the official gene-space loss is nonfinite")
        flow.solver.vf_state = state.replace(params=params)
        flow.solver.vf_state_inference = flow.solver.vf_state_inference.replace(params=params)
        means.append(record)
    return optimizer, means


def fit(output, flow, encoder, data, source_path, panel, keep, grc, pathways, deadline):
    import jax
    import jax.numpy as jnp
    import optax
    from cellflow.data._dataloader import TrainSampler
    from flax import serialization

    rng, host_rng = restore_flow(flow)
    adam = optax.adam(5e-5)
    optimizer = adam.init(flow.solver.vf_state.params)
    step = official_step(flow, encoder, keep, adam)
    sampler = TrainSampler(data=flow.train_data, batch_size=1024)
    if sampler.n_source_dists != 3 or sampler.n_target_dists != 4333:
        raise ValueError("the released sampler differs from the accepted prefix")
    log = output / "fit-losses.jsonl"
    adata = anndata.read_h5ad(source_path, backed="r")
    try:
        columns = adata.var_names.get_indexer(np.asarray(panel)[keep]).tolist()
        if min(columns) < 0 or len(columns) != 3224:
            raise ValueError("invalid stage 2 expression visibility")
        completed = 999
        gene_updates = 0
        start = time.monotonic()
        with log.open("x", encoding="utf-8") as history:
            for update in range(1000, STEP_CAP + 1):
                if time.monotonic() >= deadline:
                    break
                rng, key = jax.random.split(rng)
                loss = float(flow.solver.step_fn(key, sampler.sample(host_rng)))
                if not np.isfinite(loss):
                    raise ValueError("the official OT flow loss is nonfinite")
                components = []
                if update % GENE_LOSS_PERIOD == 0:
                    if update == 1000 and int(flow.solver.vf_state.opt_state.gradient_step) != 50:
                        raise ValueError(
                            "the first resumed update did not finish the saved accumulation"
                        )
                    flow.solver.is_trained = True
                    optimizer, components = gene_update(
                        step,
                        flow,
                        encoder,
                        data,
                        adata,
                        columns,
                        jnp.asarray(grc),
                        jnp.asarray(pathways),
                        optimizer,
                        host_rng,
                    )
                    gene_updates += len(components)
                    print(f"microstep={update} released_gene_losses={components}", flush=True)
                completed = update
                history.write(
                    json.dumps(
                        {
                            "microstep": update,
                            "ot_flow_loss": loss,
                            "gene_space_updates": gene_updates,
                            "gene_space_losses": components,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                if update == 1000 or update % 100 == 0:
                    history.flush()
                    atomic_json(
                        output / "training_progress.json",
                        {
                            "microsteps": update,
                            "planned": STEP_CAP,
                            "gene_space_updates": gene_updates,
                            "ot_flow_loss": loss,
                            "latest_gene_space_losses": components,
                        },
                    )
    finally:
        adata.file.close()
    state = flow.solver.vf_state
    if (
        completed < 1000
        or gene_updates != 3 * (completed // GENE_LOSS_PERIOD)
        or int(state.step) != completed
        or int(state.opt_state.gradient_step) != completed // 20
    ):
        raise ValueError("resumed scBIG did not complete the released update schedule")
    flow.solver.is_trained = True
    checkpoint = output / "flow_inference.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(flow.solver.vf_state_inference))
    flow.solver.vf_state_inference = serialization.from_bytes(
        flow.solver.vf_state_inference, checkpoint.read_bytes()
    )
    return {
        "source_prefix_sha256": PREFIX_SHA,
        "encoder_receipt_sha256": STAGE1_RECEIPT_SHA,
        "resumed_at_microstep": 1000,
        "flow_microsteps": completed,
        "planned_microsteps": STEP_CAP,
        "flow_optimizer_updates": int(state.opt_state.gradient_step),
        "gene_space_updates": gene_updates,
        "gene_space_update_period": GENE_LOSS_PERIOD,
        "fit_seconds": time.monotonic() - start,
        "fit_stop_reason": "microstep_cap" if completed == STEP_CAP else "time_cap",
        "checkpoint_sha256": sha256_file(checkpoint),
        "fit_history_sha256": sha256_file(log),
        "fit_expression_columns": len(columns),
        "gene_loss_program_sha256": RELEASE_PROGRAM_SHA,
        "gene_loss_grc_labels_sha256": sha256_json(grc.tolist()),
        "gene_loss_pathways": len(pathways),
    }


def test_prior(esm, expected):
    import torch

    if sha256_file(COVERAGE) != COVERAGE_SHA:
        raise ValueError("the scBIG test embedding audit changed")
    audit = read_json(COVERAGE)
    mapped = audit["test_resolved_aliases"]
    bank = torch.load(esm, map_location="cpu", weights_only=True)
    targets = {c.removesuffix("+ctrl") for c in expected}
    if len(targets) != 128 or set(mapped) != targets - bank.keys():
        raise ValueError("the sealed K562 test prior coverage changed")
    prior = {
        gene: np.asarray(bank[mapped.get(gene, gene)], dtype=np.float32).reshape(-1)
        for gene in targets
    }
    prior["ctrl"] = np.zeros_like(next(iter(prior.values())))
    return prior


def frozen_predictor(flow, encoder, order, prior, target_line):
    import jax

    # Compile the released ODE prediction once for the frozen 300-cell shape.
    solve = jax.jit(flow.solver._predict_jit)
    mean_identity = np.mean(np.stack(list(flow.adata.uns["cell_line_embeddings"].values())), axis=0)
    frozen_uns = {"cell_line_embeddings": {target_line: mean_identity}, "pert_embeddings": prior}
    inverse = np.argsort(order)

    def predict(condition, controls):
        from cellflow.data._dataloader import PredictionSampler

        obs = pd.DataFrame({"cell_line": target_line, "is_control": [True] * len(controls)})
        latent = np.asarray(encoder.transform(controls[:, order]), dtype=np.float32)
        adata = anndata.AnnData(X=latent, obs=obs)
        adata.obsm["X_pca"] = latent
        adata.uns.update(frozen_uns)
        covariates = pd.DataFrame(
            {
                "condition": [condition],
                "pert1": [condition.removesuffix("+ctrl")],
                "pert2": ["ctrl"],
                "cell_line": [target_line],
                "is_control": [False],
            }
        )
        prediction = flow.data_manager.get_prediction_data(
            adata, "X_pca", covariates, condition_id_key="condition"
        )
        if prediction.n_perturbations != 1 or not np.all(prediction.split_covariates_mask == 0):
            raise ValueError("released scBIG did not match all ordered K562 controls")
        batch = PredictionSampler(prediction).sample()
        (source_key,) = batch["source"]
        latent = solve(batch["source"][source_key], batch["condition"][source_key])
        return np.asarray(encoder.inverse_transform(np.asarray(latent)))[:, inverse]

    return predict


def run(output, gpu):
    import jax
    from flax import serialization

    started = time.monotonic()
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("scBIG C2 requires one CUDA device")
    prefix = verify_prefix()
    compass, esm = verify_assets()
    encoder, inputs = restore_stage1(compass)
    source_path, cross, cohort, rows = cohort_inputs()
    panel = inputs["grc_gene_order"]
    order = np.asarray(inputs["grc_indices"])
    keep = ~np.isin(panel, cohort["masked_genes_in_expression_order"])
    labels, pathways = loss_priors(panel, order, keep, compass)
    training, aliases = prepare_source_latents(encoder, inputs, esm)
    if aliases != prefix["esm2_resolved_aliases"] or training.n_obs != prefix["source_rows"]:
        raise ValueError("the source-tensor reconstruction differs from the accepted flow prefix")
    support = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support) != SUPPORT_SHA:
        raise ValueError("shared C2 fitting support changed")
    flow = initialize_flow(training)
    fitted = fit(
        output,
        flow,
        encoder,
        training,
        source_path,
        panel,
        keep,
        labels,
        pathways,
        started + FIT_SECONDS,
    )
    identity = {
        **fitted,
        "model": "scbig",
        "scenario": "C2",
        "run_id": output.name,
        "gpu": gpu,
        "cohort_sha256": COHORT_SHA,
        "shared_fit_support_file_sha256": SUPPORT_SHA,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "official_commit": COMMIT,
        "fit_test_conditions_accessed": 0,
        "unseen_cell_line_policy": "uniform_mean_of_the_three_fitted_source_cell_line_embeddings",
    }
    atomic_json(output / "training_result.json", identity)
    prior = test_prior(esm, cohort["condition_ids"])
    predict = frozen_predictor(flow, encoder, order, prior, cross["target_cell_line"])
    atomic_json(
        output / "worker.json",
        {"stage": "evaluate", "owner": process_info(os.getpid()), "physical_gpu": gpu},
    )
    start_eval = time.monotonic()
    with official_module_session(
        checkout_root=TXPERT_CHECKOUT, expected_commit=TXPERT_COMMIT, module_names=("gspp.metrics",)
    ) as (modules, _):
        result = score(
            predict,
            modules["gspp.metrics"],
            source_path,
            cross,
            cohort,
            rows,
            read_json(support),
            output,
            journal_key=sha256_json(identity),
            stop_starting_conditions_at=started + WALL_SECONDS,
        )
    checkpoint = output / "flow_inference.msgpack"
    if (
        result.get("conditions") != 128
        or sha256_file(checkpoint) != identity["checkpoint_sha256"]
        or serialization.to_bytes(flow.solver.vf_state_inference) != checkpoint.read_bytes()
        or list(output.rglob("*.pkl"))
    ):
        raise RuntimeError("frozen scoring did not complete with the same checkpoint and zero PKLs")
    atomic_json(
        output / "result.json",
        {
            **identity,
            **result,
            "status": "evaluated",
            "formal_eligible": True,
            "evaluation_seconds": time.monotonic() - start_eval,
            "continuation_seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != STUDY or any(
        os.getenv(key) != value
        for key, value in {
            "CUDA_VISIBLE_DEVICES": "1",
            "JAX_PLATFORMS": "cuda",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("the scBIG continuation requires the sealed GPU 1 environment")
    git = ["git", "-C", str(UPSTREAM)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("the pinned scBIG upstream checkout changed")
    with lock(SERVER / "runtime/v2-gpu-1.lock"):
        admit_selected_idle(snapshot(os.getpid()), 1, args.gpu_uuid)
        output.mkdir()
        atomic_json(
            output / "worker.json",
            {"stage": "restore_step_999", "owner": process_info(os.getpid()), "physical_gpu": 1},
        )
        sys.path.insert(0, str(UPSTREAM))
        run(output, 1)


if __name__ == "__main__":
    main()
