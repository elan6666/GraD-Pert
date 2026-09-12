"""Data, selection and prediction adapters; all model/loss/training is official."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.common import AdaptedCanonicalData
from gradpert.data._io import atomic_json


@contextmanager
def independent_best_snapshot(network: Any, *, before_restore: Any = None):
    """Fix upstream's shallow state_dict alias without modifying its checkout."""
    original = network.state_dict
    had_override = "state_dict" in network.__dict__
    previous = network.__dict__.get("state_dict")
    original_load = network.load_state_dict
    had_load_override = "load_state_dict" in network.__dict__
    previous_load = network.__dict__.get("load_state_dict")
    network.state_dict = lambda *args, **kwargs: deepcopy(original(*args, **kwargs))

    def load(state, *args, **kwargs):
        before_restore(deepcopy(original()))
        return original_load(state, *args, **kwargs)

    if before_restore is not None:
        network.load_state_dict = load
    try:
        yield
    finally:
        if had_override:
            network.state_dict = previous
        else:
            del network.state_dict
        if before_restore is not None:
            if had_load_override:
                network.load_state_dict = previous_load
            else:
                del network.load_state_dict


def prepare_data(package: Any, adapted: AdaptedCanonicalData, prior: Any) -> Any:
    """Keep canonical split, all target embeddings and only train/val expression."""
    adata = adapted.adata
    adata.X = sparse.csr_matrix(adata.X)
    adata.obs["condition"] = adata.obs["condition"].astype(str).astype("category")
    before_rows = tuple(adata.obs_names)
    embeddings = pd.DataFrame(
        np.concatenate([np.zeros((1, prior.embedding_width), dtype=np.float32), prior.values]),
        index=["ctrl", *prior.gene_ids],
    )
    fit_targets = {
        gene for condition in adata.obs["condition"].astype(str) for gene in condition.split("+")
    }
    if fit_targets - set(embeddings.index):
        raise ValueError("official Scouter would silently remove a canonical target")
    data = package.ScouterData(adata, embeddings, "condition", "gene_name")
    data.setup_ad("embd_index", slim=False)
    if tuple(data.adata.obs_names) != before_rows:
        raise ValueError("Scouter changed canonical training/validation rows")

    # The official loss consumes gene_idx_non_zeros, not DE rankings. Empty
    # rankings ask the official helper to compute its exact nonzero masks but
    # skip unused top20 bookkeeping (including undefined singleton rankings).
    # These empty lists must never be presented as scientific DE results.
    data.adata.uns["rank_genes_groups"] = {
        condition: []
        for condition in data.adata.obs["condition"].astype(str).unique()
        if condition != "ctrl"
    }
    data.get_dropout_non_zero_genes()
    condition = data.adata.obs["condition"].astype(str)
    data.train_conds = [*adapted.train_conditions, "ctrl"]
    data.val_conds = [*adapted.val_conditions, "ctrl"]
    data.train_adata = data.adata[condition.isin(data.train_conds)].copy()
    data.val_adata = data.adata[condition.isin(data.val_conds)].copy()
    data.test_conds = []
    data.test_adata = data.adata[:0].copy()
    return data


def fit_official(
    model: Any,
    config: Any,
    *,
    epochs: int,
    progress_path: Any,
    r50: bool = False,
    last_checkpoint_path: Path | None = None,
) -> dict:
    """Call one continuous official train invocation, including its scheduler."""
    if r50 and (epochs not in {1, 50} or last_checkpoint_path is None):
        raise ValueError("R50 Scouter requires one or50 epochs and an explicit last path")
    if not r50 and last_checkpoint_path is not None:
        raise ValueError("dual checkpoint capture requires explicit R50 policy")
    if last_checkpoint_path is not None and last_checkpoint_path.exists():
        raise FileExistsError("last checkpoint must be a fresh artifact")
    captures = 0

    def capture_last(state):
        import torch

        nonlocal captures
        if captures or last_checkpoint_path is None:
            raise RuntimeError("unexpected repeated official best restoration")
        last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, last_checkpoint_path)
        captures += 1

    def p(name):
        return config.model.parameters[name].value

    model.model_init(
        n_encoder=tuple(int(x) for x in p("encoder").split(",")),
        n_out_encoder=int(p("bottleneck")),
        n_decoder=tuple(int(x) for x in p("generator").split(",")),
        use_batch_norm=p("batch_norm"),
        use_layer_norm=p("layer_norm"),
        dropout_rate=p("dropout"),
    )
    steps = 0

    def count_forward(module, args, output):
        nonlocal steps
        if module.training:
            steps += 1
            if steps == 1 or steps % 50 == 0:
                atomic_json(
                    progress_path,
                    {
                        "stage": "training",
                        "training_forwards": steps,
                        "completed_validation_losses": list(model.loss_history["val_loss"]),
                        "epochs_requested": epochs,
                    },
                )

    handle = model.network.register_forward_hook(count_forward)
    try:
        with independent_best_snapshot(model.network, before_restore=capture_last if r50 else None):
            model.train(
                batch_size=int(config.training.train_batch_size.value),
                loss_gamma=float(p("loss_gamma")),
                loss_lambda=float(p("loss_lambda")),
                lr=float(config.training.learning_rate.value),
                sched_gamma=float(p("scheduler_gamma")),
                n_epochs=epochs,
                patience=epochs + 1 if r50 else int(config.training.early_stopping_patience.value),
            )
    finally:
        handle.remove()
    losses = model.loss_history
    val = losses["val_loss"]
    if not val or len(val) != len(losses["train_loss"]) or not np.isfinite(val).all():
        raise RuntimeError("official Scouter validation is absent or nonfinite")
    if r50 and (len(val) != epochs or captures != 1):
        raise RuntimeError("R50 Scouter did not preserve exact epochs and actual final state")
    best, best_epoch = float("inf"), None
    for epoch, value in enumerate(val, 1):
        if best - value > 0.001:
            best, best_epoch = float(value), epoch
    if best_epoch is None or best != model.best_val_loss:
        raise RuntimeError("Scouter best selection does not match official min_delta")
    receipt = {
        "epochs_requested": epochs,
        "epochs_completed": len(val),
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "training_forwards": steps,
        "loss_history": losses,
        "patience": config.training.early_stopping_patience.value,
        "min_delta": 0.001,
        "best_snapshot_adapter": "deepcopy_state_dict_instance_scope",
        "official_training_api": "scouter.Scouter.train",
        "canonical_test_truth_present_during_fit": False,
    }
    if r50:
        from gradpert.hashing import sha256_file

        assert last_checkpoint_path is not None
        receipt.update(
            r50=True,
            early_stopping=False,
            patience=epochs + 1,
            last_epoch=epochs,
            last_checkpoint_sha256=sha256_file(last_checkpoint_path),
            last_capture="before_official_best_restore",
        )
    atomic_json(progress_path, {"stage": "fit_complete", **receipt})
    return receipt


def predict_exact_controls(
    model: Any, torch: Any, condition: str, controls: Any, batch_size: int
) -> np.ndarray:
    controls = np.asarray(controls, dtype=np.float32)
    if controls.ndim != 2 or len(controls) != 300:
        raise ValueError("Scouter prediction requires ordered [300,G] controls")
    indices = [model.embd_idx_dict[gene] for gene in condition.split("+")]
    model.network.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, 300, batch_size):
            x = torch.as_tensor(controls[start : start + batch_size], device=model.device)
            idx = torch.tensor([indices] * len(x), dtype=torch.long, device=model.device)
            outputs.append(model.network(idx, x).detach().cpu().numpy())
    result = np.concatenate(outputs)
    if result.shape != controls.shape or not np.isfinite(result).all():
        raise RuntimeError("invalid Scouter prediction")
    return result
