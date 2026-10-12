"""Frozen Scouter training with expression-axis and context adapters only."""

from __future__ import annotations

import importlib
import pickle
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_SHA = "0cfddd000e19b72ff033ba67c8315f7bc3304932"


def _allowed(prepared: Any) -> np.ndarray:
    indices = np.asarray(prepared.allowed_indices, dtype=np.int64)
    excluded = np.asarray(prepared.excluded_indices, dtype=np.int64)
    if not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("allowed expression axis is empty or duplicated")
    if set(indices) & set(excluded):
        raise ValueError("allowed/excluded expression axes overlap")
    if set(indices) | set(excluded) != set(range(prepared.adata.n_vars)):
        raise ValueError("allowed/excluded axes do not partition the expression axis")
    return indices


def require_sanitized_expression(adata: Any, excluded: Any):
    """Reject auxiliary expression, allowing AnnData0.13's primary-X alias."""
    if adata.raw is not None or any(key is not None for key in adata.layers):
        raise ValueError("unsanitized expression side channels")
    # AnnData0.13 stores its primary X in layers[None]. It is not a separate
    # expression source; only accept this representation when it aliases X.
    if None in adata.layers and adata.layers[None] is not adata.X:
        raise ValueError("None layer is not the sanitized primary expression matrix")
    if adata.X[:, excluded].nnz:
        raise ValueError("excluded expression is visible during fit")


def load_official_prior(runtime: dict, fit_targets: set[str]):
    """Read a hashed official prior and only materialize audited target vectors."""
    import pandas as pd

    from gradpert.hashing import sha256_file, sha256_json

    prior_path = Path(runtime["prior_path"])
    digest = sha256_file(prior_path)
    if digest != runtime["prior_sha256"]:
        raise ValueError("official prior digest mismatch")
    required = set(map(str, runtime["required_target_genes"])) | fit_targets
    required.discard("ctrl")
    aliases = runtime.get("prior_aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in aliases.items()
    ):
        raise ValueError("official prior aliases must be an audited gene/source mapping")
    if aliases and sha256_json(aliases) != runtime.get("prior_aliases_sha256"):
        raise ValueError("official prior alias digest mismatch")
    if set(aliases) - required:
        raise ValueError("prior aliases contain noncanonical target identities")
    if prior_path.suffix in {".pickle", ".pkl"}:
        with prior_path.open("rb") as handle:
            values = pickle.load(handle)
        if not isinstance(values, dict):
            raise ValueError("official prior must be a gene/vector dictionary")
        missing = sorted(gene for gene in required if aliases.get(gene, gene) not in values)
        if missing:
            raise ValueError(f"official prior misses canonical fit/test targets: {missing}")
        embeddings = pd.DataFrame.from_dict(
            {gene: values[aliases.get(gene, gene)] for gene in sorted(required)}, orient="index"
        )
        if "ctrl" in values:
            embeddings.loc["ctrl"] = values["ctrl"]
    else:
        source = pd.read_csv(prior_path, index_col=0)
        if not source.index.is_unique:
            raise ValueError("official prior gene index is duplicated")
        missing = sorted(gene for gene in required if aliases.get(gene, gene) not in source.index)
        if missing:
            raise ValueError(f"official prior misses canonical fit/test targets: {missing}")
        embeddings = source.loc[[aliases.get(gene, gene) for gene in sorted(required)]].copy()
        embeddings.index = sorted(required)
        if "ctrl" in source.index:
            embeddings.loc["ctrl"] = source.loc["ctrl"]
    if "ctrl" not in embeddings.index:
        embeddings.loc["ctrl"] = np.zeros(embeddings.shape[1], dtype=np.float32)
    if not embeddings.index.is_unique or not np.isfinite(embeddings.to_numpy()).all():
        raise ValueError("invalid official prior vectors")
    return embeddings, {
        "prior_sha256": digest,
        "prior_aliases": aliases,
        "prior_aliases_sha256": sha256_json(aliases),
        "required_target_gene_count": len(required),
        "prior_subset": "all canonical train/validation/test targets plus ctrl",
    }


def masked_official_loss(original: Any, allowed: np.ndarray):
    """Restrict the tensors, then execute the unchanged official loss."""
    remap = {int(index): position for position, index in enumerate(allowed)}

    def loss(pred, truth, ctrl, group, nonzero_idx_dict, gamma, lambda_):
        masks = {
            condition: [remap[int(i)] for i in indices if int(i) in remap]
            for condition, indices in nonzero_idx_dict.items()
        }
        if any(not masks.get(condition) for condition in set(group)):
            raise ValueError("condition has no allowed official nonzero loss genes")
        return original(
            pred[:, allowed],
            truth[:, allowed],
            ctrl[:, allowed],
            group,
            nonzero_idx_dict=masks,
            gamma=gamma,
            lambda_=lambda_,
        )

    return loss


def official_nonzero_masks(data: Any, allowed: np.ndarray):
    """Run upstream dropout statistics solely on observed expression columns."""
    full = data.adata
    subset = full[:, allowed].copy()
    data.adata = subset
    try:
        data.get_dropout_non_zero_genes()
    finally:
        data.adata = full
    for name, values in subset.uns.items():
        if name in {"gene_idx_non_zeros", "gene_idx_non_dropout"}:
            full.uns[name] = {
                condition: allowed[np.asarray(indices, dtype=np.int64)]
                for condition, indices in values.items()
            }
    return full.uns["gene_idx_non_zeros"]


def contextual_dataset_factory(original: Any, prepared: Any, pair_audit: list):
    """Keep official Dataset/ordering; restrict basal pair choices by context."""
    row_names = tuple(map(str, prepared.adata.obs_names))
    row_lookup = {name: index for index, name in enumerate(row_names)}
    maps = {**prepared.train_control_indices, **prepared.val_control_indices}

    def factory(adata, key_label, key_embd_index, shuffle=True, seed=24):
        dataset = original(adata, key_label, key_embd_index, shuffle=shuffle, seed=seed)
        ctrl_names = tuple(map(str, adata.obs_names[adata.obs[key_label].astype(str) == "ctrl"]))
        local_ctrl = {name: index for index, name in enumerate(ctrl_names)}
        rng = np.random.RandomState(seed)
        # The official dataset pairs once, then reuses those pairs each epoch.
        # Preserve cycling/shuffling within each compatible candidate pool.
        pools: dict[tuple[int, ...], list[int]] = {}
        offsets: dict[tuple[int, ...], int] = {}
        paired = []
        for pert_index in dataset.PertCellIndx:
            row = str(dataset.bcode[int(pert_index)])
            candidates = tuple(int(i) for i in maps.get(row, ()))
            if not candidates:
                raise ValueError(f"no compatible training controls for {row}")
            if any(
                row_names[i] not in local_ctrl
                or prepared.row_context_ids[i] != prepared.row_context_ids[row_lookup[row]]
                for i in candidates
            ):
                raise ValueError("context pairing candidates violate the canonical context")
            if candidates not in pools:
                pool = list(candidates)
                if shuffle:
                    rng.shuffle(pool)
                pools[candidates] = pool
                offsets[candidates] = 0
            offset = offsets[candidates]
            chosen = pools[candidates][offset % len(candidates)]
            offsets[candidates] = offset + 1
            paired.append(local_ctrl[row_names[chosen]])
            pair_audit.append((row, row_names[chosen]))
        dataset.CtrlCellIndx = np.asarray(paired, dtype=np.int64)
        return dataset

    return factory


class _UpdateLimitReached(Exception):
    pass


@contextmanager
def bounded_official_updates(module: Any, limit: int | None, counter: dict):
    """Diagnostic only: stop after a completed official Adam step, no copied loop."""
    if limit is None:
        yield
        return
    if limit <= 0:
        raise ValueError("max_updates must be positive")
    original = module.optim.Adam

    class CountingAdam(original):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            counter["updates"] += 1
            if counter["updates"] >= limit:
                raise _UpdateLimitReached
            return result

    module.optim.Adam = CountingAdam
    try:
        yield
    finally:
        module.optim.Adam = original


def fit(prepared, config: dict, runtime: dict, output: Path, max_updates=None):
    import torch
    from scipy import sparse

    from benchmarks.common.official_checkout import verify_official_checkout
    from benchmarks.expression_holdout.contract import FittedAdapter
    from benchmarks.scouter.official_api import independent_best_snapshot
    from gradpert.hashing import sha256_json

    root = verify_official_checkout(runtime["official_checkout"], OFFICIAL_SHA)
    package = importlib.import_module("scouter")
    module = importlib.import_module("scouter.Scouter")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError("Scouter import is not from the frozen official checkout")
    allowed = _allowed(prepared)
    adata = prepared.adata.copy()
    adata.X = sparse.csr_matrix(adata.X)
    require_sanitized_expression(adata, prepared.excluded_indices)
    required = {
        gene
        for condition in (*prepared.train_conditions, *prepared.val_conditions)
        for gene in condition.split("+")
    }
    embeddings, prior_receipt = load_official_prior(runtime, required)
    data = package.ScouterData(adata, embeddings, "condition", "gene_name")
    rows = tuple(map(str, adata.obs_names))
    data.setup_ad("embd_index", slim=False)
    if tuple(map(str, data.adata.obs_names)) != rows:
        raise ValueError("Scouter silently changed fit rows")
    # Loss needs only the official nonzero masks, not DE rankings. Empty rankings
    # suppress unused internal DE summaries; common evaluation supplies frozen DEGs.
    data.adata.uns["rank_genes_groups"] = {
        str(c): [] for c in data.adata.obs.condition.unique() if str(c) != "ctrl"
    }
    masks = official_nonzero_masks(data, allowed)
    for condition in (*prepared.train_conditions, *prepared.val_conditions):
        if not set(masks.get(condition, ())) & set(allowed):
            raise ValueError(f"no allowed nonzero loss genes: {condition}")
    train_rows = [*prepared.train_row_ids, *prepared.control_row_ids]
    val_rows = [*prepared.val_row_ids, *prepared.control_row_ids]
    data.train_adata = data.adata[list(dict.fromkeys(train_rows))].copy()
    data.val_adata = data.adata[list(dict.fromkeys(val_rows))].copy()
    data.test_adata = data.adata[:0].copy()
    data.train_conds = [*prepared.train_conditions, "ctrl"]
    data.val_conds = [*prepared.val_conditions, "ctrl"]
    data.test_conds = []
    torch.manual_seed(int(runtime["seed"]))
    np.random.seed(int(runtime["seed"]))
    model = package.Scouter(data, device=runtime["device"])
    model.model_init(**config["model"])
    training = dict(config["training"])
    if len(prepared.train_row_ids) < int(training["batch_size"]):
        raise ValueError("official drop_last loader would contain zero training batches")
    original_dataset, original_loss = module.BalancedDataset, module.gears_loss
    pairs: list = []
    module.BalancedDataset = contextual_dataset_factory(original_dataset, prepared, pairs)
    module.gears_loss = masked_official_loss(original_loss, allowed)
    final_states: list = []
    counter = {"updates": 0}
    stopped = False
    try:
        with (
            independent_best_snapshot(model.network, before_restore=final_states.append),
            bounded_official_updates(module, max_updates, counter),
        ):
            try:
                model.train(**training)
            except _UpdateLimitReached:
                stopped = True
        if final_states:
            model.network.load_state_dict(final_states[-1])
    finally:
        module.BalancedDataset, module.gears_loss = original_dataset, original_loss
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(deepcopy(model.network.state_dict()), output / "last.pt")
    return FittedAdapter(
        model=model,
        receipt={
            "official_sha": OFFICIAL_SHA,
            "official_training_api": "scouter.Scouter.train",
            **prior_receipt,
            "prior_kind": runtime.get("prior_kind", "official-supported raw prior"),
            "ctrl_embedding": "official zero vector if absent in prior",
            "checkpoint_role": "last",
            "last_capture": "before_official_best_restore" if final_states else "final_state",
            "official_selection": "validation allowed-expression loss; min_delta=0.001",
            "official_dropout_statistics_axis": "allowed expression genes only",
            "loss_history": model.loss_history,
            "epochs_requested": training["n_epochs"],
            "epochs_completed": len(model.loss_history["train_loss"]),
            "pair_count": len(pairs),
            "pair_sha256": sha256_json(pairs),
            "context_pairing": "official fixed pairs; balanced within compatible pools",
            "diagnostic_update_limit": max_updates,
            "diagnostic_updates_completed": counter["updates"] if max_updates else None,
            "diagnostic_limit_reached": stopped,
            "seen_expression_gene_indices": allowed.tolist(),
            "adaptations": [
                "input expression exclusion",
                "allowed official loss axis",
                "context-compatible fixed control pairs",
                "independent best snapshot",
            ],
        },
    )


def predict(fitted, controls, target_genes, gene_ids, *, cell_batch, query_count, seed):
    import torch

    controls = np.asarray(controls, dtype=np.float32)
    if controls.ndim != 2 or controls.shape[1] != len(gene_ids):
        raise ValueError("prediction control gene axis mismatch")
    if cell_batch <= 0 or not len(controls):
        raise ValueError("invalid inference batch")
    model = fitted.model
    if tuple(map(str, model.all_adata.var.gene_name)) != tuple(gene_ids):
        raise ValueError("prediction gene identity/order mismatch")
    targets = [model.embd_idx_dict[g] for g in (*target_genes, "ctrl")]
    # Single targets use the same +ctrl representation as official setup_ad.
    if len(target_genes) > 1:
        targets = [model.embd_idx_dict[g] for g in target_genes]
    outputs = []
    model.network.eval()
    with torch.no_grad():
        for offset in range(0, len(controls), cell_batch):
            x = torch.as_tensor(controls[offset : offset + cell_batch], device=model.device)
            idx = torch.tensor([targets] * len(x), dtype=torch.long, device=model.device)
            outputs.append(model.network(idx, x).cpu().numpy())
    result = np.concatenate(outputs)
    if result.shape != controls.shape or not np.isfinite(result).all():
        raise ValueError("invalid full-axis Scouter prediction")
    return result
