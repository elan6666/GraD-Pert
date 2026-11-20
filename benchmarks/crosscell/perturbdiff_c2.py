"""Run released PerturbDiff model code against the frozen Cross-cell C2 protocol."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import torch
import yaml

from benchmarks.common import official_module_session
from benchmarks.common.study_fit_partition import allowed_expression_indices, crosscell_fit_rows
from benchmarks.crosscell.official_fit_probe import (
    COMMITS,
    PRIOR_SHA,
    SUPPORT_SHA,
    build_perturbdiff,
    install_perturbdiff_fit_mask,
)
from benchmarks.crosscell.study_data import cohort_inputs, score
from benchmarks.crosscell.study_paths import (
    COHORT_SHA,
    REPO,
    SERVER,
    STUDY,
    TXPERT_CHECKOUT,
    TXPERT_COMMIT,
)
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.crosscell import read_rows
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

CONFIG = REPO / "configs/experiments/perturbdiff/crosscell_k562.yaml"
UPSTREAM = SERVER / "upstreams/perturbdiff-c2-study"
ASSETS = UPSTREAM.with_name(UPSTREAM.name + "-assets")
GENEPT = Path(
    "/data/yilangliu/DinoGenePT/data/genept/GenePT_gene_protein_embedding_model_3_text.pickle"
)
GENEPT_SHA = "cd64611edab310b517cbb461295605a37821f6cd4f8ddae90ba18a2f861a6618"
UNSEEN_BATCH = "__C2_UNSEEN__"


def load_prior(perturbations, fit_targets, test_targets, destination, missing_fit):
    released = ASSETS / "replogle-prior.pkl"
    if sha256_file(released) != PRIOR_SHA["perturbdiff"] or sha256_file(GENEPT) != GENEPT_SHA:
        raise ValueError("released perturbation prior or GenePT revision changed")
    with released.open("rb") as stream:
        small = pickle.load(stream)
    with GENEPT.open("rb") as stream:
        large = pickle.load(stream)
    overlap = small.keys() & large.keys()
    if len(overlap) != 2009 or any(not np.array_equal(small[g], large[g]) for g in overlap):
        raise ValueError("official priors differ across their shared genes")
    bank = {
        g: small[g] if g in small else large[g] for g in perturbations if g in small or g in large
    }
    if fit_targets - bank.keys() != {missing_fit} or test_targets - bank.keys():
        raise ValueError("the sealed missing-fit or test target coverage changed")
    with destination.open("xb") as stream:
        pickle.dump(bank, stream)
    return sha256_file(destination)


def groups_for(rows, condition, line, batch, rng, size):
    contexts = defaultdict(list)
    for i in rows:
        name = condition[i]
        if not name.endswith("+ctrl") or name.count("+") != 1:
            raise ValueError("released Replogle adapter requires a single target per treatment")
        contexts[(name.removesuffix("+ctrl"), line[i].lower(), batch[i])].append(i)
    grouped = []
    for key, positions in contexts.items():
        order = rng.permutation(positions)
        grouped.extend((*key, order[i : i + size]) for i in range(0, len(order), size))
    return [grouped[i] for i in rng.permutation(len(grouped))]


def pad_indices(indices, size):
    return np.pad(indices, (0, size - len(indices)), mode="edge")


def tensor_batch(adata, groups, controls, visible, vocab, cfg, rng):
    size = cfg["cell_set_size"]
    response = np.stack([pad_indices(rows, size) for _, _, _, rows in groups])
    control = np.stack(
        [rng.choice(controls[(line, batch)], size, replace=True) for _, line, batch, _ in groups]
    )

    def expression(ids):
        flat = read_rows(adata, ids.ravel())[:, visible]
        packed = np.zeros((flat.shape[0], adata.n_vars), dtype=np.float32)
        packed[:, visible] = flat
        return torch.from_numpy(packed.reshape(*ids.shape, -1)).cuda() / cfg["input_divisor"]

    def code(key, names):
        indices = torch.tensor([vocab[key][n] for n in names], device="cuda")
        return indices[:, None].expand(-1, size)

    return {
        "pert_emb": expression(response),
        "cont_emb": expression(control),
        "cov_pert": code("pert", [g for g, _, _, _ in groups]),
        "cov_celltype": code("cell_line", [line for _, line, _, _ in groups]),
        "cov_batch": code("batch", [batch for _, _, batch, _ in groups]),
        "ds_name": ["replogle"] * len(groups),
    }


def fit(adata, train, validation, controls, visible, model, vocab, cfg, output, limit, deadline):
    loss_scale, projection = install_perturbdiff_fit_mask(model, adata.n_vars, visible)
    optimizers, schedules = model.configure_optimizers()
    optimizer, scheduler = optimizers[0], schedules[0]["scheduler"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    groups_per_batch = cfg["batch_cells"] // cfg["cell_set_size"]

    def objective(groups):
        batch = tensor_batch(adata, groups, controls, visible, vocab, cfg, rng)
        loss = model._compute_loss(batch)
        value = (loss["loss"] * loss["weights"]).mean() / loss_scale
        if not torch.isfinite(value).item():
            raise ValueError("non-finite released fitting or validation loss")
        return value

    completed = 0
    exposed = 0
    for offset in range(0, len(train), groups_per_batch):
        if completed == limit or time.monotonic() >= deadline:
            break
        group_batch = train[offset : offset + groups_per_batch]
        optimizer.zero_grad(set_to_none=True)
        loss = objective(group_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        completed += 1
        exposed += sum(len(rows) for _, _, _, rows in group_batch)
        print(f"fit_update={completed}/{limit} loss={float(loss.detach()):.8g}", flush=True)
        if completed == 1 or completed % 25 == 0:
            atomic_json(output / "training_progress.json", {"updates": completed, "planned": limit})
    if completed == 0:
        raise RuntimeError("no PerturbDiff optimizer update completed")
    model.eval()
    with torch.no_grad():
        n = min(len(validation), cfg["validation_batches"] * groups_per_batch)
        validation_loss = float(
            np.mean(
                [
                    float(objective(validation[start : min(n, start + groups_per_batch)]))
                    for start in range(0, n, groups_per_batch)
                ]
            )
        )
    projection.remove()
    return completed, exposed, validation_loss


def predict(model, vocab, cfg, condition, line, controls, control_batches, *, unseen_batch=False):
    size = cfg["cell_set_size"]
    bins = defaultdict(list)
    for index, name in enumerate(control_batches):
        bins[name].append(index)
    grouped = [
        (batch, np.asarray(rows[offset : offset + size]))
        for batch, rows in bins.items()
        for offset in range(0, len(rows), size)
    ]
    output = np.empty_like(controls, dtype=np.float32)
    batch_groups = cfg["batch_cells"] // size
    with torch.inference_mode():
        for offset in range(0, len(grouped), batch_groups):
            groups = grouped[offset : offset + batch_groups]
            rows = np.stack([pad_indices(indices, size) for _, indices in groups])
            values = torch.from_numpy(controls[rows]).cuda() / cfg["input_divisor"]
            shape = values.shape[:2]
            indexes = torch.tensor(
                [vocab["batch"][UNSEEN_BATCH if unseen_batch else name] for name, _ in groups],
                device="cuda",
            )
            encoded = model._encode_covariates(
                {
                    "cov_pert": torch.full(shape, vocab["pert"][condition], device="cuda"),
                    "cov_celltype": torch.full(shape, vocab["cell_line"][line], device="cuda"),
                    "cov_batch": indexes[:, None].expand(shape),
                }
            )
            prediction, _ = model.diffusion.ddim_sample_loop(
                model.model,
                tuple(values.shape),
                self_condition={
                    "batch_emb": encoded,
                    "cont_emb": values,
                    "gene_emb": None,
                    "ds_name": ["replogle"] * len(groups),
                },
                clip_denoised=True,
                device=values.device,
                start_time=cfg["ddim_steps"],
                eta=0.0,
                guidance_strength=cfg["guidance_strength"],
                sample_kwargs={},
            )
            pred = prediction.cpu().numpy() * cfg["input_divisor"]
            for i, (_, indices) in enumerate(groups):
                output[indices] = pred[i, : len(indices)]
    if not np.isfinite(output).all():
        raise ValueError("nonfinite frozen DDIM response")
    return output


def run(output, support, cfg, identity, smoke, started):
    source, cross, cohort, rows = cohort_inputs()
    adata = anndata.read_h5ad(source, backed="r")
    try:
        visible = list(
            allowed_expression_indices(
                list(adata.var_names), cohort["masked_genes_in_expression_order"], masked=True
            )
        )
        conditions = adata.obs["condition"].astype(str).to_numpy()
        lines = adata.obs["cell_line"].astype(str).to_numpy()
        batches = adata.obs["batch"].astype(str).to_numpy()
        roles = crosscell_fit_rows(rows, conditions, cohort, compatibility=support)
        treatment = np.r_[roles["train"], roles["val"]]
        source_lines = set(lines[treatment])
        if source_lines != {"RPE1", "hepg2", "jurkat"} or adata.n_vars != cfg["expression_genes"]:
            raise ValueError("the C2 expression axis or fitting cell lines changed")
        if np.intersect1d(treatment, np.r_[rows["test"], rows["target_control"]]).size:
            raise ValueError("held-out rows crossed the fitting boundary")
        controls = defaultdict(list)
        for i in roles["source_control"]:
            if conditions[i] != "ctrl" or lines[i] not in source_lines:
                raise ValueError("invalid fitting control")
            controls[(lines[i].lower(), batches[i])].append(i)
        rng = np.random.default_rng(cfg["seed"])
        train = groups_for(roles["train"], conditions, lines, batches, rng, cfg["cell_set_size"])
        val = groups_for(roles["val"], conditions, lines, batches, rng, cfg["cell_set_size"])
        fit_targets = {g for g, _, _, _ in train + val}
        test_targets = {c.removesuffix("+ctrl") for c in cohort["condition_ids"]}
        if fit_targets & test_targets or len(test_targets) != 128:
            raise ValueError("held-out perturbation identities crossed the fitting boundary")
        vocab = {
            "pert": {
                g: i for i, g in enumerate(sorted(fit_targets | test_targets | {"non-targeting"}))
            },
            "cell_line": {
                c: i for i, c in enumerate(sorted({s.lower() for s in source_lines} | {"k562"}))
            },
            "batch": {b: i for i, b in enumerate([*sorted(set(batches[treatment])), UNSEEN_BATCH])},
        }
        limit = cfg["smoke_updates"] if smoke else cfg["max_updates"]
        with tempfile.TemporaryDirectory(dir=output, prefix="official-prior-") as tmp:
            prior = Path(tmp) / "replogle.pkl"
            prior_sha = load_prior(
                vocab["pert"], fit_targets, test_targets, prior, cfg["allowed_missing_fit_target"]
            )
            model, _, _, _ = build_perturbdiff(
                UPSTREAM,
                prior,
                vocabulary=vocab,
                genes=adata.n_vars,
                cell_prior=ASSETS / "celltype.pkl",
                max_steps=cfg["max_updates"],
            )
            completed, exposed, validation_loss = fit(
                adata,
                train,
                val,
                controls,
                visible,
                model,
                vocab,
                cfg,
                output,
                limit,
                started + cfg["max_fit_seconds"],
            )
        with torch.no_grad():
            weight = model.cov_encoder.batch_encoder.weight
            weight[vocab["batch"][UNSEEN_BATCH]].copy_(weight[:-1].mean(0))
        checkpoint = output / "checkpoint.pt"
        torch.save(model.state_dict(), checkpoint)
        fit_receipt = {
            **identity,
            "optimizer_updates": completed,
            "planned_updates": limit,
            "seen_training_rows_excluding_padding": exposed,
            "validation_loss": validation_loss,
            "checkpoint_sha256": sha256_file(checkpoint),
            "prior_sha256": prior_sha,
            "missing_fit_target": cfg["allowed_missing_fit_target"],
            "missing_fit_rows": int(np.count_nonzero(conditions[roles["train"]] == "CCDC130+ctrl")),
            "missing_fit_policy": "released PerturbDiff zero prior fallback; fitted rows preserved",
            "fit_support_row_sha256": support["fit_row_sha256"],
            "fit_expression_columns": len(visible),
            "fit_seconds": time.monotonic() - started,
            "test_updates": 0,
            "budget_stop_reason": "update_cap" if completed == limit else "fit_time_cap",
        }
        atomic_json(output / "vocabulary.json", vocab)
        atomic_json(output / "training_result.json", fit_receipt)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        model.eval().requires_grad_(False)
        if smoke:
            target, line, _, _ = val[0]
            sample_rows = np.concatenate(
                [r for (source_line, _), r in controls.items() if source_line == line]
            )[:300]
            if len(np.unique(sample_rows)) != 300:
                raise ValueError("smoke requires 300 different source controls")
            sample = predict(
                model, vocab, cfg, target, line, read_rows(adata, sample_rows), batches[sample_rows]
            )
            atomic_json(
                output / "result.json",
                {
                    **fit_receipt,
                    "status": "trained_validation_only",
                    "test_conditions_accessed": 0,
                    "validation_prediction_shape": list(sample.shape),
                },
            )
            return
    finally:
        adata.file.close()
    journal = sha256_json(
        {
            **identity,
            "checkpoint": fit_receipt["checkpoint_sha256"],
            "paired_rows": cohort["per_condition_pairing_sha256"],
        }
    )

    def apply(condition, input_controls, control_rows):
        torch.manual_seed(cfg["evaluation_seed"] + cohort["condition_ids"].index(condition))
        return predict(
            model,
            vocab,
            cfg,
            condition.removesuffix("+ctrl"),
            "k562",
            input_controls,
            batches[control_rows],
            unseen_batch=True,
        )

    atomic_json(
        output / "worker.json",
        {
            "stage": "evaluate",
            "owner": process_info(os.getpid()),
            "physical_gpu": identity["gpu"],
        },
    )
    with official_module_session(
        checkout_root=TXPERT_CHECKOUT, expected_commit=TXPERT_COMMIT, module_names=("gspp.metrics",)
    ) as (modules, _):
        result = score(
            apply,
            modules["gspp.metrics"],
            source,
            cross,
            cohort,
            rows,
            support,
            output,
            journal_key=journal,
            include_control_row_ids=True,
            stop_starting_conditions_at=started + cfg["max_wall_seconds"],
        )
    if result.get("conditions") != 128:
        raise RuntimeError("formal frozen evaluation did not complete all 128 conditions")
    if list(output.rglob("*.pkl")):
        raise ValueError("successful metrics-only output must contain zero pickles")
    atomic_json(
        output / "result.json",
        {
            **fit_receipt,
            **result,
            "status": "evaluated",
            "unseen_batch_policy": cfg["unseen_batch"],
        },
    )


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    if args.output.resolve().parent != STUDY or any(
        os.getenv(k) != v
        for k, v in {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("invalid PerturbDiff CUDA admission envelope")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        verification_method="github_rest_live",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    support_path = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support_path) != SUPPORT_SHA:
        raise ValueError("sealed C2 population changed")
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        upstream_commit != COMMITS["perturbdiff"]
        or subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("released PerturbDiff source changed")
    cfg = yaml.safe_load(CONFIG.read_text())
    if (cfg["learning_rate"], cfg["batch_cells"], cfg["cell_set_size"], cfg["input_divisor"]) != (
        0.002,
        128,
        32,
        10.0,
    ):
        raise ValueError("released Replogle fitting settings changed")
    identity = {
        "scenario": "C2",
        "formal_eligible": source.formal_eligible,
        "source_commit": source.commit,
        "cohort_sha256": COHORT_SHA,
        "shared_fit_support_file_sha256": SUPPORT_SHA,
        "config_sha256": sha256_file(CONFIG),
        "official_commit": upstream_commit,
        "model": "perturbdiff",
        "gpu": args.gpu,
        "run_id": args.output.name,
    }
    if not args.smoke:
        if args.smoke_run is None or args.smoke_run.resolve().parent != STUDY:
            raise ValueError("formal run needs the same-source validation receipt")
        smoke = read_json(args.smoke_run / "result.json")
        for key in (
            "cohort_sha256",
            "shared_fit_support_file_sha256",
            "config_sha256",
            "source_commit",
        ):
            if smoke.get(key) != identity[key]:
                raise ValueError(f"smoke {key} differs from the queued formal run")
        if (
            smoke["status"] != "trained_validation_only"
            or smoke["test_updates"] != 0
            or smoke["optimizer_updates"] != cfg["smoke_updates"]
            or smoke["validation_prediction_shape"] != [300, cfg["expression_genes"]]
        ):
            raise ValueError("validation-only gate did not pass")
    with lock(SERVER / "runtime" / f"v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        if torch.cuda.device_count() != 1:
            raise ValueError("PerturbDiff requires exactly one CUDA device")
        torch.manual_seed(cfg["seed"])
        torch.cuda.set_per_process_memory_fraction(0.9)
        args.output.mkdir()
        atomic_json(
            args.output / "worker.json",
            {
                "stage": "fit",
                "owner": process_info(os.getpid()),
                "physical_gpu": args.gpu,
            },
        )
        sys.path.insert(0, str(UPSTREAM))
        run(args.output, read_json(support_path), cfg, identity, args.smoke, started)


if __name__ == "__main__":
    main()
