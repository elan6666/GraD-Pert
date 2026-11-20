"""Bounded GPU fit probes of upstream PerturbDiff and PRESCRIBE on C2 source rows."""

import argparse
import logging
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import torch

from benchmarks.common.study_fit_partition import (
    allowed_expression_indices,
    crosscell_fit_rows,
)
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import SERVER, STUDY
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.crosscell import read_rows
from gradpert.execution.gpu_lease import lock
from gradpert.hashing import sha256_file, sha256_json

SUPPORT_SHA = "e00c407acc221206d27cc2aeab0f9ea73a2dc0989cdf4a33852d7be9b6030964"
COMMITS = {
    "perturbdiff": "31be89cb8cce5dddef0782bac29d5c748ce55931",
    "prescribe": "6f7264a205aaff654a9594863c5c10b656f88ebe",
}
PRIOR_SHA = {
    "perturbdiff": "8c43bce711e34b4b7cb27541e1dc654d54d01a0efb225993b5c396124c4a7574",
    "prescribe": "011230c0767bb0eeb730b787b7ade2a815567249bf2b2c108188734514778f23",
}
CELL_PRIOR_SHA = "e29b1213fc95a72d5f4402920291d26125f07b22b3ab74d414f500fe8811d99a"
STEPS = 4


def source_batches(
    role_rows, conditions, contexts, support, random, groups, cells, *, replace_short=False
):
    grouped = defaultdict(list)
    for row in role_rows:
        condition = conditions[row]
        if condition in support:
            grouped[(condition, contexts[row])].append(row)
    candidates = sorted(
        key for key, values in grouped.items() if (replace_short or len(values) >= cells)
    )
    if len(candidates) < groups:
        raise ValueError("not enough represented C2 source groups for the bounded probe")
    chosen = random.choice(len(candidates), groups, replace=False)
    selected = [candidates[i] for i in chosen]
    treated = np.stack(
        [random.choice(grouped[key], cells, replace=len(grouped[key]) < cells) for key in selected]
    )
    return treated, selected


def prepare(path, rows, cohort, priors, seed, output, *, full_axis=False):
    support_path = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support_path) != SUPPORT_SHA:
        raise ValueError("C2 shared fit support changed")
    adata = anndata.read_h5ad(path, backed="r")
    try:
        genes = list(map(str, adata.var_names))
        allowed = list(
            allowed_expression_indices(
                genes, cohort["masked_genes_in_expression_order"], masked=True
            )
        )
        conditions = adata.obs["condition"].astype(str).to_numpy()
        lines = adata.obs["cell_line"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        contexts = [pair for pair in zip(lines, batches, strict=True)]
        fit = crosscell_fit_rows(rows, conditions, cohort, compatibility=read_json(support_path))
        controls = defaultdict(list)
        for row in fit["source_control"]:
            controls[contexts[row]].append(row)
        permitted = {
            cond
            for cond in set(conditions[fit["train"]]) | set(conditions[fit["val"]])
            if all(g in priors for g in cond.split("+") if g != "ctrl")
            and cond.count("+") == 1
            and cond.endswith("+ctrl")
        }
        rng = np.random.default_rng(seed)
        train, train_keys = source_batches(
            fit["train"], conditions, contexts, permitted, rng, 4 * STEPS, 32
        )
        val, val_keys = source_batches(
            fit["val"], conditions, contexts, permitted, rng, 4, 32, replace_short=True
        )
        train_controls = np.stack(
            [rng.choice(controls[key[1]], 32, replace=True) for key in train_keys]
        )
        val_controls = np.stack(
            [rng.choice(controls[key[1]], 32, replace=True) for key in val_keys]
        )
        all_ids = np.concatenate(
            [train.ravel(), train_controls.ravel(), val.ravel(), val_controls.ravel()]
        )
        if not set(lines[all_ids]) <= {"RPE1", "hepg2", "jurkat"}:
            raise ValueError("a target-line row crossed the fit boundary")
        if np.intersect1d(all_ids, rows["test"]).size:
            raise ValueError("a test row crossed the fit boundary")

        def read(ids):
            values = read_rows(adata, ids.ravel())[:, allowed]
            if full_axis:
                # Only observed values cross the fit boundary.
                packed = np.zeros((len(values), len(genes)), dtype=values.dtype)
                packed[:, allowed] = values
                values = packed
            return torch.from_numpy(values.reshape(*ids.shape, -1))

        atomic_json(
            output / "fit_inputs.json",
            {
                "fit_support_sha256": SUPPORT_SHA,
                "frozen_expression_axis": len(genes),
                "visible_expression_axis": len(allowed),
                **(
                    {"masked_fit_input_axis": len(genes), "visible_indices": allowed}
                    if full_axis
                    else {}
                ),
                "train_row_sha256": sha256_json(train.tolist()),
                "val_row_sha256": sha256_json(val.tolist()),
                "val_rows_unique": int(np.unique(val).size),
                "train_control_row_sha256": sha256_json(train_controls.tolist()),
                "val_control_row_sha256": sha256_json(val_controls.tolist()),
                "test_expression_rows_read": 0,
                "train_conditions": sorted({key[0] for key in train_keys}),
                "val_conditions": sorted({key[0] for key in val_keys}),
                "released_prior_missing_fit_targets": len(
                    {
                        g
                        for c in set(conditions[fit["train"]]) | set(conditions[fit["val"]])
                        for g in c.split("+")
                        if g != "ctrl" and g not in priors
                    }
                ),
                "released_prior_missing_test_targets": sorted(
                    {
                        g
                        for c in cohort["condition_ids"]
                        for g in c.split("+")
                        if g not in priors and g != "ctrl"
                    }
                ),
            },
        )
        return (read(train), read(train_controls), train_keys), (
            read(val),
            read(val_controls),
            val_keys,
        )
    finally:
        adata.file.close()


def build_perturbdiff(upstream, asset, train, val):
    from hydra import compose, initialize_config_dir
    from src.models.lightning.lightning_module import PlModel

    genes = train[0].shape[-1]
    keys = train[2] + val[2]
    pert = {
        g: i for i, g in enumerate(sorted({"non-targeting", *(c.split("+")[0] for c, _ in keys)}))
    }
    lines = {name: i for i, name in enumerate(sorted({context[0].lower() for _, context in keys}))}
    batches = {name: i for i, name in enumerate(sorted({context[1] for _, context in keys}))}
    with initialize_config_dir(version_base=None, config_dir=str(upstream / "configs")):
        cfg = compose(
            config_name="rawdata_diffusion_training",
            overrides=[
                "data=replogle_finetune",
                "optimization.optimizer.lr=0.002",
                "trainer.max_steps=200000",
                "cov_encoding.replogle_gene_encoding=genept",
                "model.p_drop_control=0",
                f"model.input_dim={genes}",
                f"model.hidden_num=[{genes},512]",
                f"data.pad_length={genes}",
                "cov_encoding.celltype_encoding=llm",
            ],
        )
    cell_prior = asset.with_name("celltype.pkl")
    if sha256_file(cell_prior) != CELL_PRIOR_SHA:
        raise ValueError("official released cell-type embeddings changed")
    cfg.cov_encoding.celltype_embedding_path = str(cell_prior)
    cfg.cov_encoding.replogle_gene_embedding_path = str(asset)
    cfg.cov_encoding.pert_dict = pert
    cfg.cov_encoding.cell_type_dict = lines
    cfg.cov_encoding.batch_dict = batches
    cfg.cov_encoding.num_pert = len(pert)
    cfg.cov_encoding.num_celltype = len(lines)
    cfg.cov_encoding.num_batch = len(batches)
    cfg.model.dataset_dict = ["replogle"]
    mod = PlModel(
        cfg.cov_encoding,
        cfg.model,
        logging.getLogger(__name__),
        cfg.optimization,
        cfg.trainer,
        ["val"],
    ).cuda()
    return mod, pert, lines, batches


def fit_perturbdiff(upstream, asset, train, val, visible_indices=None):
    mod, pert, lines, batches = build_perturbdiff(upstream, asset, train, val)
    optimizers, schedules = mod.configure_optimizers()
    opt, schedule = optimizers[0], schedules[0]["scheduler"]

    loss_scale = 1.0
    projection = None
    if visible_indices is not None:
        keep = torch.zeros(train[0].shape[-1], dtype=torch.bool, device="cuda")
        keep[visible_indices] = True
        loss_scale = len(visible_indices) / len(keep)

        def hide_unobserved_response(_layer, _inputs, values):
            return {
                **values,
                "x": values["x"].masked_fill(~keep, 0),
                "x_control": values["x_control"].masked_fill(~keep, 0),
            }

        projection = mod.model.register_forward_hook(hide_unobserved_response)
        # Energy distance is unaffected by padded zeros. Preserve its relative
        # weight after the official MSE averages over the wider tensor.
        mod.optimizer_cfg.MMD_loss_factor *= loss_scale

    def compute(batch, start):
        y, x, contexts = batch
        keys = contexts[start : start + 4]

        # The released cell-set collator emits one covariate index per cell.
        def per_cell(values):
            return torch.tensor(values, device="cuda")[:, None].expand(-1, y.shape[1])

        loss = mod._compute_loss(
            {
                "pert_emb": y[start : start + 4].cuda() / 10.0,
                "cont_emb": x[start : start + 4].cuda() / 10.0,
                "cov_pert": per_cell([pert[c.split("+")[0]] for c, _ in keys]),
                "cov_celltype": per_cell([lines[line.lower()] for _, (line, _) in keys]),
                "cov_batch": per_cell([batches[b] for _, (_, b) in keys]),
                "ds_name": ["replogle"] * len(keys),
            }
        )
        return (loss["loss"] * loss["weights"]).mean() / loss_scale

    loss_values = []
    for index in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = compute(train, index * 4)
        if not torch.isfinite(loss).item():
            raise ValueError("official PerturbDiff fit loss is not finite")
        loss.backward()
        opt.step()
        schedule.step()
        loss_values.append(float(loss.detach()))
        print(f"fit_update={index + 1}/{STEPS} loss={loss_values[-1]:.8g}", flush=True)
    mod.eval()
    with torch.no_grad():
        valid = float(compute(val, 0))
    if projection is not None:
        projection.remove()
    return (
        mod,
        loss_values,
        valid,
        {
            "celltype_encoding": "llm",
            "group_size": 32,
            "batch_cells": 128,
            "input_divisor": 10.0,
            "full_output_axis": visible_indices is not None,
            "fit_loss_visible_only": True,
        },
    )


def time_perturbdiff(upstream, asset, train, val, source, rows, cohort, previous, output):
    """Replay one condition through the official sampler with 300 source controls."""
    from hydra import compose, initialize_config_dir
    from src.apps.sampling.sampling_generation_helpers import resolve_sampling_runner

    fit_receipt = read_json(previous / "result.json")
    checkpoint = previous / "fit_probe.pt"
    fit_inputs = output / "fit_inputs.json"
    if (
        fit_receipt["status"] != "bounded_fit_probe"
        or fit_receipt["model"] != "perturbdiff"
        or fit_receipt["upstream_commit"] != COMMITS["perturbdiff"]
        or fit_receipt["test_conditions_accessed"] != 0
        or fit_receipt["checkpoint_sha256"] != sha256_file(checkpoint)
        or fit_receipt["fit_inputs_sha256"] != sha256_file(fit_inputs)
    ):
        raise ValueError("the probe checkpoint and reconstructed fitting metadata differ")
    model, pert, lines, batches = build_perturbdiff(upstream, asset, train, val)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    adata = anndata.read_h5ad(source, backed="r")
    try:
        gene_ids = list(map(str, adata.var_names))
        allowed = list(
            allowed_expression_indices(
                gene_ids, cohort["masked_genes_in_expression_order"], masked=True
            )
        )
        if train[0].shape[-1] == len(gene_ids):
            allowed = list(range(len(gene_ids)))
        available = rows["source_control"]
        control_obs = adata.obs.iloc[available]
        ctl_lines = control_obs["cell_line"].astype(str).to_numpy()
        ctl_batches = control_obs["batch"].astype(str).to_numpy()
        compatible = np.isin(ctl_batches, list(batches))
        candidates = [
            (c, line, available[compatible & (ctl_lines == line)]) for c, (line, _) in val[2]
        ]
        condition, line, positions = max(candidates, key=lambda item: len(item[-1]))
        if len(positions) < 300 or line.lower() == "k562":
            raise ValueError("the checkpoint lacks a source line with 300 representable controls")

        # Each cell set stays within a batch, as in the released Replogle sampler.
        within_line = adata.obs.iloc[positions]["batch"].astype(str).to_numpy()
        by_batch = [(b, positions[within_line == b]) for b in np.unique(within_line)]
        rng = np.random.default_rng(42)
        selected_groups = []
        remaining = 300
        for batch_id, candidates in sorted(by_batch, key=lambda item: -len(item[1])):
            count = min(len(candidates), remaining)
            sampled = np.sort(rng.choice(candidates, count, replace=False))
            for offset in range(0, count, 32):
                group = sampled[offset : offset + 32]
                selected_groups.append((batch_id, group))
            remaining -= count
            if remaining == 0:
                break
        selected = np.concatenate([group for _, group in selected_groups])
        if (
            len(np.unique(selected)) != 300
            or not (adata.obs.iloc[selected]["condition"].astype(str) == "ctrl").all()
        ):
            raise ValueError("the sampling benchmark requires 300 distinct source controls")
        controls = [
            (
                batch_id,
                len(group),
                read_rows(adata, np.pad(group, (0, 32 - len(group)), mode="edge"))[:, allowed],
            )
            for batch_id, group in selected_groups
        ]
    finally:
        adata.file.close()

    with initialize_config_dir(version_base=None, config_dir=str(upstream / "configs")):
        config = compose(
            config_name="rawdata_diffusion_sampling", overrides=["data=replogle_finetune"]
        )
    sample, kwargs = resolve_sampling_runner(config, model.diffusion, config.sampling.use_ddim)
    part_seconds = []
    completed = 0
    with torch.inference_mode():
        for begin in range(0, len(controls), 4):
            chunk = controls[begin : begin + 4]
            encoded = torch.from_numpy(np.stack([x for _, _, x in chunk])).cuda() / 10.0
            group_shape = encoded.shape[:2]
            batch_indices = torch.tensor([batches[b] for b, _, _ in chunk], device="cuda")
            ids = {
                "cov_pert": torch.full(group_shape, pert[condition.split("+")[0]], device="cuda"),
                "cov_celltype": torch.full(group_shape, lines[line.lower()], device="cuda"),
                "cov_batch": batch_indices[:, None].expand(group_shape),
            }
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction, _ = sample(
                model.model,
                tuple(encoded.shape),
                self_condition={
                    "batch_emb": model._encode_covariates(ids),
                    "cont_emb": encoded,
                    "gene_emb": None,
                    "ds_name": ["replogle"] * group_shape[0],
                },
                clip_denoised=config.sampling.clip_denoised,
                device=encoded.device,
                progress=False,
                **kwargs,
            )
            torch.cuda.synchronize()
            part_seconds.append(time.perf_counter() - start)
            if prediction.shape != encoded.shape or not torch.isfinite(prediction).all().item():
                raise ValueError("official DDIM did not return a finite visible-axis prediction")
            completed += sum(count for _, count, _ in chunk)
            print(f"sampled_rows={completed}/300 seconds={part_seconds[-1]:.2f}", flush=True)
    return {
        "model": "perturbdiff",
        "scenario": "C2",
        "status": "source_ddim_timing",
        "formal_eligible": False,
        "optimizer_updates": 0,
        "test_conditions_accessed": 0,
        "upstream_commit": COMMITS["perturbdiff"],
        "checkpoint_sha256": fit_receipt["checkpoint_sha256"],
        "fit_probe_receipt_sha256": sha256_file(previous / "result.json"),
        "control_row_sha256": sha256_json(selected.tolist()),
        "condition_id": condition,
        "source_cell_line": line,
        "output_shape": [completed, len(allowed)],
        "padded_rows": len(controls) * 32 - completed,
        "ddim_steps": kwargs["start_time"],
        "guidance_strength": kwargs["guidance_strength"],
        "batch_seconds": part_seconds,
        "total_sample_seconds": sum(part_seconds),
        "projected_128_conditions_sample_seconds": 128 * sum(part_seconds),
    }


def fit_prescribe(asset, train, val, output):
    from types import SimpleNamespace

    from sklearn.decomposition import PCA
    from src.model.lightening_module_flow import NaturalPosteriorNetworkFlowLightningModule
    from src.nn import PRESCRIBE
    from src.nn.encoder import Encoder
    from src.nn.flow import MixFlow
    from src.nn.loss import BayesianLoss
    from src.nn.output import MultiNormalOutput

    target, controls, _ = train
    genes = target.shape[-1]
    pca = PCA(n_components=10, random_state=3407).fit(target.reshape(-1, genes).numpy())
    x_prior = pca.transform(controls.reshape(-1, genes).numpy())
    prior_dir = output / "data/c2/data_pyg"
    prior_dir.mkdir(parents=True, exist_ok=True)
    np.save(prior_dir / "mean.npy", x_prior.mean(0))
    np.save(prior_dir / "cov.npy", np.cov(x_prior, rowvar=False))
    np.savez(output / "fit_pca.npz", mean=pca.mean_, components=pca.components_)
    embedding_dir = output / "scLLM_weights/scGPT"
    embedding_dir.mkdir(parents=True)
    (embedding_dir / "embedding.pkl").symlink_to(asset)
    mod = PRESCRIBE(
        latent_dim=64,
        output_dim=10,
        encoder=Encoder(latent_dim=64, backbone="scGPT", pert_num=genes),
        flow=MixFlow(dim=64, num_layers=10, flow_size=0.774, n_hidden=2, maf_layers=10),
        output=MultiNormalOutput(latent_dim=64, output_dim=10, use_prior=True, data_name="c2"),
        num_perts=len(train[2]),
        certainty_budget="exp",
        bound=30.0,
    ).cuda()
    # Use the upstream Adam warmup settings. This probe invokes the Bayesian term
    # directly, before the rank-target construction required by the full runner.
    opt = NaturalPosteriorNetworkFlowLightningModule(
        mod, learning_rate=1e-3
    ).configure_optimizers()["optimizer"]
    criterion = BayesianLoss(1e-7)

    def compute(batch, start):
        y, x, keys = batch
        selected = keys[start : start + 4]
        y = y[start : start + 4].reshape(-1, genes)
        x = x[start : start + 4].reshape(-1, genes)
        data = SimpleNamespace(
            x=x.cuda(), pert=[key[0] for key in selected for _ in range(32)], num_graphs=len(x)
        )
        posterior, _ = mod(data)
        coords = torch.from_numpy(pca.transform(y.numpy())).cuda().float()
        return criterion(posterior, coords)[0]

    loss_values = []
    for index in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = compute(train, index * 4)
        if not torch.isfinite(loss).item():
            raise ValueError("official PRESCRIBE Bayesian loss is not finite")
        loss.backward()
        opt.step()
        loss_values.append(float(loss.detach()))
        print(f"fit_update={index + 1}/{STEPS} loss={loss_values[-1]:.8g}", flush=True)
    mod.eval()
    with torch.no_grad():
        valid = float(compute(val, 0))
    return (
        mod,
        loss_values,
        valid,
        {"pca_dimensions": 10, "batch_cells": 128, "loss_scope": "Bayesian term only"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(COMMITS))
    parser.add_argument("gpu", type=int, choices=(0, 1))
    parser.add_argument("output", type=Path)
    parser.add_argument("--timing-from", type=Path)
    parser.add_argument("--full-axis-probe", action="store_true")
    args = parser.parse_args()
    if args.full_axis_probe and args.model != "perturbdiff":
        raise ValueError("the full-axis probe requires PerturbDiff")
    if args.timing_from is not None:
        args.timing_from = args.timing_from.resolve(strict=True)
        if args.model != "perturbdiff" or args.timing_from.parent != STUDY:
            raise ValueError("DDIM timing requires a server PerturbDiff fit probe")
    args.output = args.output.resolve()
    if args.output.parent != STUDY or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("pin one physical GPU and a fresh server study output")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("expected the allocated CUDA allocator setting")
    upstream = SERVER / "upstreams" / f"{args.model}-c2-study"
    head = subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", upstream, "status", "--porcelain"], text=True
    ).strip()
    if head != COMMITS[args.model] or dirty:
        raise ValueError("official upstream checkout changed")
    asset = upstream.with_name(upstream.name + "-assets") / (
        "replogle-prior.pkl" if args.model == "perturbdiff" else "scgpt-primary.pkl"
    )
    if sha256_file(asset) != PRIOR_SHA[args.model]:
        raise ValueError("official released prior changed")
    seed = 42 if args.model == "perturbdiff" else 3407
    args.output.mkdir()
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        if torch.cuda.device_count() != 1:
            raise ValueError("probe did not receive exactly one CUDA device")
        torch.manual_seed(seed)
        with asset.open("rb") as stream:
            priors = pickle.load(stream)
        source, _, cohort, rows = cohort_inputs()
        train, val = prepare(
            source, rows, cohort, priors, seed, args.output, full_axis=args.full_axis_probe
        )
        del priors
        sys.path.insert(0, str(upstream))
        os.chdir(args.output)
        if args.timing_from is not None:
            receipt = time_perturbdiff(
                upstream, asset, train, val, source, rows, cohort, args.timing_from, args.output
            )
            atomic_json(args.output / "result.json", {**receipt, "gpu": args.gpu})
            return
        begin = time.monotonic()
        if args.model == "perturbdiff":
            visible = read_json(args.output / "fit_inputs.json").get("visible_indices")
            mod, losses, valid, changes = fit_perturbdiff(upstream, asset, train, val, visible)
        else:
            mod, losses, valid, changes = fit_prescribe(asset, train, val, args.output)
        if not np.isfinite(valid):
            raise ValueError("nonfinite validation loss")
        checkpoint = args.output / "fit_probe.pt"
        torch.save(mod.state_dict(), checkpoint)
        atomic_json(
            args.output / "result.json",
            {
                "model": args.model,
                "scenario": "C2",
                "status": "bounded_fit_probe",
                "formal_eligible": False,
                "optimizer_updates": len(losses),
                "gpu": args.gpu,
                "upstream_commit": head,
                "official_prior_sha256": PRIOR_SHA[args.model],
                "fit_inputs_sha256": sha256_file(args.output / "fit_inputs.json"),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_losses": losses,
                "validation_loss": valid,
                "test_conditions_accessed": 0,
                "test_updates": 0,
                "elapsed_seconds": time.monotonic() - begin,
                "probe_configuration": changes,
            },
        )


if __name__ == "__main__":
    main()
