"""Adapt the released PRESCRIBE model and training modules to frozen C2."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

import anndata
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from benchmarks.common import official_module_session
from benchmarks.common.full_gene_evaluation import replay_fields, verify_replay
from benchmarks.common.study_fit_partition import allowed_expression_indices, crosscell_fit_rows
from benchmarks.crosscell.common_score_panel import AUDIT, AUDIT_SHA, PANEL, load_panel
from benchmarks.crosscell.official_fit_probe import COMMITS, PRIOR_SHA, SUPPORT_SHA
from benchmarks.crosscell.prescribe_gpu import SharedGpu
from benchmarks.crosscell.study_data import cohort_inputs, score
from benchmarks.crosscell.study_paths import (
    COHORT_SHA,
    REPO,
    SERVER,
    STUDY,
    TXPERT_CHECKOUT,
    TXPERT_COMMIT,
)
from benchmarks.expression_holdout.data import read_allowed_rows
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

CONFIG = REPO / "configs/experiments/prescribe/crosscell_k562.yaml"
UPSTREAM = SERVER / "upstreams/prescribe-c2-study"
ASSETS = UPSTREAM.with_name(UPSTREAM.name + "-assets")
SUPPORT = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
PANEL_SHA = "931d25e3153c220b07b3ca9c1410b90d3f66c851b8259c2a3cfd0cd48f2074ca"
MISSING = "TBCE"


def condition_target(condition, aliases):
    if not condition.endswith("+ctrl") or condition.count("+") != 1:
        raise ValueError(f"C2 requires one target per treatment: {condition}")
    target = condition.removesuffix("+ctrl")
    return aliases.get(target, target)


def build_embedding_bank(train_names, val_names, test_names, aliases, *, missing=(MISSING,)):
    path = ASSETS / "scgpt-primary.pkl"
    if sha256_file(path) != PRIOR_SHA["prescribe"]:
        raise ValueError("released scGPT embedding bank changed")
    with path.open("rb") as file:
        prior = pickle.load(file)
    fit = {condition_target(c, aliases) for c in (*train_names, *val_names)}
    test = {condition_target(c, aliases) for c in test_names}
    if fit & test or fit - prior.keys() != set(missing) or test - prior.keys():
        raise ValueError("frozen released-prior coverage changed")
    bank = {
        gene: np.asarray(prior[gene], dtype=np.float32)
        for gene in sorted((fit | test) - set(missing))
    }
    supported_train = sorted({condition_target(c, aliases) for c in train_names} & bank.keys())
    fallback = np.stack([bank[gene] for gene in supported_train]).mean(0, dtype=np.float64)
    bank.update({gene: fallback.astype(np.float32) for gene in missing})
    if not np.isfinite(np.stack(list(bank.values()))).all():
        raise ValueError("the selected scGPT embeddings must be finite")
    return bank, supported_train


def sample_corrected_edistance(treated, controls):
    """Compute the released squared-Euclidean E-test statistic without permutations."""
    treated = np.asarray(treated, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    delta = treated.mean(0) - controls.mean(0)
    # For singleton fit conditions there is no within-condition pair.
    treated_noise = treated.var(0, ddof=1).sum() / len(treated) if len(treated) > 1 else 0.0
    control_noise = controls.var(0, ddof=1).sum() / len(controls) if len(controls) > 1 else 0.0
    return 2 * (float(delta @ delta) - treated_noise - control_noise)


def fit_data(source, cohort, rows, compatibility, cfg):
    data = anndata.read_h5ad(source, backed="r")
    try:
        names = data.obs["condition"].astype(str).to_numpy()
        lines = data.obs["cell_line"].astype(str).to_numpy()
        batches = data.obs["batch"].astype(str).to_numpy()
        roles = crosscell_fit_rows(rows, names, cohort, compatibility=compatibility)
        fit = np.sort(np.concatenate(list(roles.values())))
        if (
            set(lines[fit]) != {"RPE1", "hepg2", "jurkat"}
            or np.intersect1d(fit, np.r_[rows["test"], rows["target_control"]]).size
        ):
            raise ValueError("the C2 fit contains an unexpected cell line or test row")
        cols = allowed_expression_indices(
            list(data.var_names), cohort["masked_genes_in_expression_order"], masked=True
        )
        matrix = read_allowed_rows(data, fit, cols, data.n_vars)
        visible = data.var_names[list(cols)]
    finally:
        data.file.close()
    roles = {role: np.searchsorted(fit, values) for role, values in roles.items()}
    names, lines, batches = names[fit], lines[fit], batches[fit]
    train, val, controls = [roles[role] for role in ("train", "val", "source_control")]
    train_names, val_names = [sorted(set(names[role])) for role in (train, val)]
    if sha256_file(AUDIT) != AUDIT_SHA:
        raise ValueError("the C2 vocabulary audit changed")
    coverage = read_json(AUDIT)["prescribe"]
    fit_aliases, test_aliases = (coverage[role]["aliases"] for role in ("training", "test"))
    if any(
        fit_aliases[key] != test_aliases[key] for key in fit_aliases.keys() & test_aliases.keys()
    ):
        raise ValueError("C2 vocabulary aliases disagree")
    aliases = {**fit_aliases, **test_aliases}
    bank, supported_train = build_embedding_bank(
        train_names, val_names, cohort["condition_ids"], aliases
    )
    missing_counts = {
        role: int((names[indices] == "TBCE+ctrl").sum())
        for role, indices in (("train", train), ("val", val))
    }
    if missing_counts != {"train": 113, "val": 12} or not np.all(names[controls] == "ctrl"):
        raise ValueError("the prior omission or control identities changed")
    return fit_projection(
        matrix,
        cols,
        visible,
        roles,
        names,
        lines,
        batches,
        train_names,
        aliases,
        bank,
        supported_train,
        missing_counts,
        cfg,
    )


def fit_projection(
    matrix,
    cols,
    visible,
    roles,
    names,
    lines,
    batches,
    train_names,
    aliases,
    bank,
    supported_train,
    missing_counts,
    cfg,
):
    train, val, controls = [roles[role] for role in ("train", "val", "source_control")]
    print(
        f"PCA train_rows={len(train)} val_rows={len(val)} "
        f"source_controls={len(controls)} genes={len(cols)}",
        flush=True,
    )
    fit_matrix = matrix[train].toarray()
    pca = PCA(n_components=cfg["output_dim"], random_state=cfg["seed"], svd_solver="randomized")
    pca.fit(fit_matrix)
    del fit_matrix
    pca_rows = matrix @ pca.components_.T - (pca.mean_ @ pca.components_.T)
    control_coords = pca_rows[controls]
    ranks = {
        name: float(
            sample_corrected_edistance(pca_rows[train[names[train] == name]], control_coords)
        )
        for name in train_names
    }
    if not np.isfinite(list(ranks.values())).all():
        raise ValueError("the C2 rank targets must be finite")
    # Reuse source controls matched by cell line and batch, with a fixed pairing.
    groups = defaultdict(list)
    for row in controls:
        groups[(lines[row], batches[row])].append(row)
    rng = np.random.default_rng(cfg["seed"])
    paired = {
        int(row): int(rng.choice(groups[(lines[row], batches[row])])) for row in np.r_[train, val]
    }
    receipt = {
        "fit_counts": {role: len(indices) for role, indices in roles.items()},
        "fit_gene_order_sha256": sha256_json(list(visible)),
        "fit_gene_count": len(visible),
        "missing_fit_prior_counts": missing_counts,
        "missing_fit_prior_policy": "mean_of_distinct_supported_training_target_embeddings",
        "prior_contributors_sha256": sha256_json(supported_train),
        "rank_target": "released_squared_euclidean_sample_corrected_e_distance",
        "rank_target_scope": "train_treatment_and_source_controls_only",
        "rank_targets_sha256": sha256_json(ranks),
        "target_aliases": aliases,
    }
    return matrix, cols, visible, roles, names, pca, pca_rows, ranks, paired, bank, receipt


class Batch(dict):
    @property
    def num_graphs(self):
        return len(self.x)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)


def prepare_batch(controls, conditions, aliases, **targets):
    return Batch(
        x=torch.as_tensor(controls, dtype=torch.float32),
        pert=[f"{condition_target(c, aliases)}+ctrl" for c in conditions],
        **targets,
    )


def run(output, identity, cfg, smoke, cpu_check, started, shared=None, *, j2=False, replay=None):
    if j2:
        from benchmarks.jurkat_j2.prescribe import fit_inputs

        scores, load_fitting = fit_inputs()
        prepared = load_fitting(cfg)
    else:
        source, cross, cohort, rows = cohort_inputs()
        panel = load_panel(source, cross, cohort)
        if sha256_file(PANEL) != PANEL_SHA or sha256_file(SUPPORT) != SUPPORT_SHA:
            raise ValueError("the C2 score panel or fit support changed")
        scores = panel["score_genes"]
        prepared = fit_data(source, cohort, rows, read_json(SUPPORT), cfg)
    matrix, cols, visible, roles, names, pca, pca_rows, ranks, paired, bank, receipt = prepared
    projection = visible.get_indexer(scores)
    if np.any(projection < 0):
        raise ValueError("PRESCRIBE cannot produce the common output panel")
    if replay is not None:
        original = verify_replay(replay, identity, "best.ckpt", conditions=592 if j2 else 128)
        if original["config_sha256"] != identity["config_sha256"]:
            raise ValueError("PRESCRIBE replay configuration changed")
        fit_receipt = read_json(replay / "training_result.json")
        archive = replay / "fit_projection.npz"
        if sha256_file(archive) != fit_receipt["fit_projection_sha256"]:
            raise ValueError("frozen PRESCRIBE projection changed")
        with np.load(archive, allow_pickle=False) as frozen_projection:
            if (
                not np.array_equal(cols, frozen_projection["visible_indices"])
                or not np.array_equal(projection, frozen_projection["projection"])
                or not np.allclose(pca.components_, frozen_projection["components"], atol=1e-6)
                or not np.allclose(pca.mean_, frozen_projection["mean"], atol=1e-6)
            ):
                raise ValueError("recreated PRESCRIBE inputs differ from the saved projection")
            pca.components_ = frozen_projection["components"]
            pca.mean_ = frozen_projection["mean"]
        projection = slice(None)
    atomic_json(output / "fit_data_receipt.json", {**identity, **receipt})
    if cpu_check:
        atomic_json(
            output / "result.json",
            {
                **identity,
                **receipt,
                "status": "fit_data_preflight",
                "formal_eligible": False,
                "test_conditions_accessed": 0,
            },
        )
        return
    fit_deadline = started + cfg["fit_seconds"]
    sys.path[:0] = [str(UPSTREAM), str(ASSETS / "python-pyblaze-2.3.2")]
    from lightning.pytorch import Callback, Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, Timer
    from src.model.lightening_module import NaturalPosteriorNetworkLightningModule
    from src.model.lightening_module_flow import NaturalPosteriorNetworkFlowLightningModule
    from src.nn import PRESCRIBE
    from src.nn.encoder import Encoder
    from src.nn.flow import MixFlow
    from src.nn.output import MultiNormalOutput

    class ShareGuard(Callback):
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            shared.check()
            self.started = time.monotonic()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            shared.finish_batch(self.started)

        def on_validation_batch_start(self, trainer, module, batch, batch_idx, dataloader_idx=0):
            shared.check()
            self.started = time.monotonic()

        def on_validation_batch_end(
            self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0
        ):
            shared.finish_batch(self.started)

    def collate(indices):
        examples = names[indices]
        return prepare_batch(
            matrix[[paired[row] for row in indices]].toarray(),
            examples,
            receipt["target_aliases"],
            pert_idx=[[0] for _ in indices],
            y_pca=torch.from_numpy(pca_rows[indices]).float(),
            y_n=torch.tensor([ranks.get(c, 0.0) for c in examples], dtype=torch.float32),
        )

    def loader(role):
        return DataLoader(
            roles[role].tolist(),
            batch_size=cfg["batch_size"],
            shuffle=role == "train",
            collate_fn=collate,
        )

    def predict(condition, input_controls):
        if shared is not None:
            shared.check()
        started_prediction = time.monotonic()
        with torch.inference_mode():
            batch = prepare_batch(
                input_controls, [condition] * len(input_controls), receipt["target_aliases"]
            )
            batch["x"] = batch.x.cuda()
            posterior, _ = model(batch)
            pca_prediction = posterior.maximum_a_posteriori().loc.cpu().numpy()
            prediction = (pca_prediction @ pca.components_ + pca.mean_)[:, projection]
        if shared is not None:
            shared.finish_batch(started_prediction)
        return prediction

    def trainer(epochs, **kwargs):
        duration = int(fit_deadline - time.monotonic())
        if duration <= 0:
            raise RuntimeError("PRESCRIBE fit budget expired before the next phase")
        if shared is not None:
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), ShareGuard()]
        return Trainer(
            accelerator="gpu",
            devices=1,
            deterministic=True,
            logger=False,
            max_epochs=epochs,
            accumulate_grad_batches=cfg["accumulate_grad_batches"],
            num_sanity_val_steps=0,
            limit_train_batches=4 if smoke else 1.0,
            limit_val_batches=1 if smoke else 1.0,
            **kwargs,
        ), Timer(duration={"seconds": duration}, interval="step")

    seed_everything(cfg["seed"], workers=True)
    with tempfile.TemporaryDirectory(prefix="official-input-", dir=output) as temporary:
        temp = Path(temporary)
        embedding_dir = temp / "scLLM_weights/scGPT"
        embedding_dir.mkdir(parents=True)
        with (embedding_dir / "embedding.pkl").open("xb") as f:
            pickle.dump(bank, f)
        data_dir = temp / "data/c2"
        (data_dir / "data_pyg").mkdir(parents=True)
        controls = pca_rows[roles["source_control"]]
        np.save(data_dir / "data_pyg/mean.npy", controls.mean(0))
        np.save(data_dir / "data_pyg/cov.npy", np.cov(controls, rowvar=False))
        metadata = anndata.AnnData()
        metadata.uns.update(pca_components=pca.components_, pca_mean=pca.mean_)
        metadata.write_h5ad(data_dir / "perturb_processed.h5ad")
        os.chdir(temp)
        try:
            model = PRESCRIBE(
                latent_dim=cfg["latent_dim"],
                output_dim=cfg["output_dim"],
                encoder=Encoder(latent_dim=cfg["latent_dim"], backbone="scGPT", pert_num=len(cols)),
                flow=MixFlow(
                    dim=cfg["latent_dim"],
                    num_layers=cfg["flow_layers"],
                    flow_size=cfg["flow_size"],
                    n_hidden=cfg["flow_n_hidden"],
                    maf_layers=cfg["maf_layers"],
                ),
                output=MultiNormalOutput(
                    latent_dim=cfg["latent_dim"],
                    output_dim=cfg["output_dim"],
                    use_prior=True,
                    data_name="c2",
                ),
                num_perts=len(ranks),
                certainty_budget=cfg["certainty_budget"],
                bound=cfg["bound"],
            )
            if replay is None:
                warmup = NaturalPosteriorNetworkFlowLightningModule(
                    model,
                    learning_rate=cfg["warmup_lr"],
                    early_stopping=False,
                    stage="warmup",
                    log_prob_positive=False,
                )
                warm, warm_timer = trainer(
                    1 if smoke else cfg["warmup_epochs"], enable_checkpointing=False
                )
                warm.callbacks.append(warm_timer)
                worker(output, "warmup", identity["gpu"])
                warm.fit(warmup, loader("train"), loader("val"))
                if warm.global_step < 1 or warm.current_epoch != (
                    1 if smoke else cfg["warmup_epochs"]
                ):
                    raise RuntimeError("PRESCRIBE warmup did not finish its configured epochs")
            main_module = NaturalPosteriorNetworkLightningModule(
                model=model,
                data_name="c2",
                learning_rate=cfg["lr"],
                learning_rate_decay=True,
                save_value_only=False,
                **{
                    k: cfg[k]
                    for k in (
                        "lam1",
                        "lam2",
                        "lam3",
                        "scheduler",
                        "interval",
                        "change_step",
                        "reduce_rate",
                        "patience",
                        "warmup_steps",
                        "warmup_max_steps",
                    )
                },
            )
            main_module.hparams.pop("model", None)
            if replay is None:
                checkpoint = ModelCheckpoint(
                    dirpath=output,
                    filename="best",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_weights_only=True,
                )
                main, main_timer = trainer(1 if smoke else cfg["epochs"], callbacks=[checkpoint])
                main.callbacks.append(main_timer)
                worker(output, "fit", identity["gpu"])
                main.fit(main_module, loader("train"), loader("val"))
                if (
                    main.global_step < 1
                    or not checkpoint.best_model_path
                    or not torch.isfinite(checkpoint.best_model_score).item()
                ):
                    raise RuntimeError(
                        "official PRESCRIBE did not produce a finite validation checkpoint"
                    )
                checkpoint_path = checkpoint.best_model_path
            else:
                checkpoint_path = replay / "best.ckpt"
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            main_module.load_state_dict(saved["state_dict"], strict=True)
            model.cuda().eval().requires_grad_(False)
            if replay is None:
                val_row = roles["val"][0]
                sample = roles["source_control"][:300]
                value = predict(names[val_row], matrix[sample].toarray())
                if value.shape != (300, len(scores)) or not np.isfinite(value).all():
                    raise ValueError(
                        "PRESCRIBE validation inference did not match the common panel"
                    )
                receipt.update(
                    warmup_optimizer_updates=warm.global_step,
                    optimizer_updates=main.global_step,
                    best_epoch=int(saved["epoch"]) + 1,
                    best_val_loss=float(checkpoint.best_model_score),
                    checkpoint_sha256=sha256_file(checkpoint.best_model_path),
                    validation_prediction_shape=list(value.shape),
                )
        finally:
            os.chdir(REPO)
    if replay is not None:
        receipt = {**fit_receipt, **replay_fields(replay, identity)}
    else:
        np.savez(
            output / "fit_projection.npz",
            components=pca.components_,
            mean=pca.mean_,
            visible_indices=cols,
            projection=projection,
            control_pca_mean=controls.mean(0),
            control_pca_cov=np.cov(controls, rowvar=False),
        )
        receipt["fit_projection_sha256"] = sha256_file(output / "fit_projection.npz")
        receipt["main_epochs_completed"] = main.current_epoch
        if shared is not None:
            receipt["shared_gpu_evidence"] = shared.evidence()
        atomic_json(output / "training_result.json", {**identity, **receipt})
    if smoke:
        atomic_json(
            output / "result.json",
            {
                **identity,
                **receipt,
                "status": "trained_validation_only",
                "test_conditions_accessed": 0,
                "test_updates": 0,
            },
        )
        return
    worker(output, "evaluate", identity["gpu"])
    if j2:
        from benchmarks.jurkat_j2.study import evaluate

        result = evaluate(
            lambda condition, controls, _rows, _axis: predict(condition, controls[:, cols]),
            output,
            deadline=started + cfg["wall_seconds"],
            journal_identity=receipt["checkpoint_sha256"],
            full_axis=replay is not None,
            native_columns=cols if replay is not None else None,
        )
    else:
        with official_module_session(
            checkout_root=TXPERT_CHECKOUT,
            expected_commit=TXPERT_COMMIT,
            module_names=("gspp.metrics",),
        ) as (modules, _):
            result = score(
                lambda condition, controls: predict(condition, controls[:, cols]),
                modules["gspp.metrics"],
                source,
                cross,
                cohort,
                rows,
                read_json(SUPPORT),
                output,
                journal_key=receipt["checkpoint_sha256"],
                score_genes=scores if replay is None else None,
                prediction_on_score_panel=replay is None,
                full_axis=replay is not None,
                native_columns=cols if replay is not None else None,
                stop_starting_conditions_at=started + cfg["wall_seconds"],
            )
    if result.get("conditions") != (592 if j2 else 128) or list(output.rglob("*.pkl")):
        raise RuntimeError("PRESCRIBE did not finish a complete metrics-only evaluation")
    if shared is not None:
        receipt["shared_gpu_evidence"] = shared.evidence()
    if replay is not None and sha256_file(replay / "best.ckpt") != original["checkpoint_sha256"]:
        raise ValueError("the frozen PRESCRIBE checkpoint changed during scoring")
    atomic_json(output / "result.json", {**identity, **receipt, **result, "status": "evaluated"})


def worker(output, stage, gpu):
    atomic_json(
        output / "worker.json",
        {"stage": stage, "physical_gpu": gpu, "owner": process_info(os.getpid())},
    )


@contextmanager
def wait_for_gpu(output, gpu, uuid, shared=None):
    worker(output, "awaiting_gpu", gpu)
    while True:
        try:
            with ExitStack() as resources:
                resources.enter_context(lock(SERVER / "runtime" / f"v2-gpu-{gpu}.lock"))
                if shared is None:
                    admit_selected_idle(snapshot(os.getpid()), gpu, uuid)
                else:
                    shared.admit()
                lease = resources.pop_all()
        except RuntimeError as error:
            reason = str(error)
            if not reason.startswith(
                (
                    "another process owns lease",
                    "selected GPU has foreign work",
                    "selected GPU free-memory reserve exhausted",
                    "selected GPU coexistence unavailable",
                )
            ):
                raise
            print(f"awaiting_gpu={gpu}: {reason}", flush=True)
            time.sleep(45)
            continue
        break
    with lease:
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--shared-gpu", action="store_true")
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--cpu-check", action="store_true")
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--smoke-run", type=Path)
    phase.add_argument("--replay-full", type=Path)
    args = parser.parse_args()
    if args.replay_full is not None:
        args.replay_full = args.replay_full.resolve()
        if args.replay_full.parent != STUDY or args.replay_full == args.output.resolve():
            raise ValueError("frozen replay requires a distinct original study run")
        if (args.output / "result.json").exists():
            raise FileExistsError("the full-gene replay already completed")
    if args.output.resolve().parent != STUDY:
        raise ValueError("PRESCRIBE output must stay within the research study")
    if args.cpu_check:
        if args.shared_gpu or os.getenv("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("data preflight must be CPU-only")
    elif (
        args.gpu is None
        or not args.gpu_uuid
        or any(
            os.getenv(key) != value
            for key, value in {
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }.items()
        )
    ):
        raise ValueError("invalid PRESCRIBE CUDA admission envelope")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        verification_method="git_ls_remote_live" if args.replay_full else "github_rest_live",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    git = ["git", "-C", str(UPSTREAM)]
    upstream = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if (
        upstream != COMMITS["prescribe"]
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("the official PRESCRIBE source changed")
    identity = {
        "model": "prescribe",
        "scenario": "C2",
        "formal_eligible": source.formal_eligible,
        "run_id": args.output.name,
        "gpu": args.gpu,
        "gpu_admission": "capped_shared" if args.shared_gpu else "exclusive",
        "source_commit": source.commit,
        "official_commit": upstream,
        "cohort_sha256": COHORT_SHA,
        "shared_fit_support_file_sha256": SUPPORT_SHA,
        "config_sha256": sha256_file(CONFIG),
        "score_panel_sha256": PANEL_SHA,
    }
    if args.smoke_run:
        if args.smoke_run.resolve().parent != STUDY:
            raise ValueError("smoke receipt must belong to this study")
        smoke = read_json(args.smoke_run / "result.json")
        if any(smoke.get(key) != value for key, value in identity.items() if key != "run_id"):
            raise ValueError("PRESCRIBE smoke protocol differs from formal run")
        if (
            smoke.get("status") != "trained_validation_only"
            or smoke.get("test_conditions_accessed") != 0
            or smoke.get("validation_prediction_shape") != [300, 3194]
            or smoke.get("optimizer_updates") != 1
            or smoke.get("warmup_optimizer_updates") != 1
        ):
            raise ValueError("PRESCRIBE official GPU smoke did not pass")
    cfg = yaml.safe_load(CONFIG.read_text())
    args.output.mkdir(exist_ok=args.replay_full is not None)
    if args.cpu_check:
        run(args.output, identity, cfg, False, True, time.monotonic())
    else:
        shared = (
            SharedGpu(
                args.gpu,
                args.gpu_uuid,
                cfg["shared_gpu"],
                allow_same_account_resident=args.replay_full is not None,
            )
            if args.shared_gpu
            else None
        )
        with wait_for_gpu(args.output, args.gpu, args.gpu_uuid, shared):
            if torch.cuda.device_count() != 1:
                raise ValueError("PRESCRIBE requires exactly one admitted CUDA device")
            if shared is not None:
                shared.configure_allocator()
            worker(args.output, "fit_data", args.gpu)
            run(
                args.output,
                identity,
                cfg,
                args.smoke,
                False,
                time.monotonic(),
                shared,
                replay=args.replay_full,
            )


if __name__ == "__main__":
    main()
