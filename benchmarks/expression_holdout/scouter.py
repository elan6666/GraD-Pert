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
    kind = runtime.get("prior_kind", "official-supported raw prior")
    representation = runtime.get("prior_representation", "raw")
    if representation != "raw":
        raise ValueError("Scouter requires raw source prior vectors, not PCA/learned projections")
    expected_dimension = runtime.get("prior_dimension")
    if kind in {"project_genept_seed", "official_genept"} and (
        not isinstance(expected_dimension, int) or expected_dimension <= 0
    ):
        raise ValueError("named GenePT comparison requires an explicit prior_dimension")
    excluded_conditions = tuple(runtime.get("excluded_conditions", ()))
    if any(not isinstance(condition, str) or not condition for condition in excluded_conditions):
        raise ValueError("authorized excluded conditions must be explicit condition identities")
    excluded_targets = {
        gene for condition in excluded_conditions for gene in condition.split("+") if gene != "ctrl"
    }
    aliases = runtime.get("prior_aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in aliases.items()
    ):
        raise ValueError("official prior aliases must be an audited gene/source mapping")
    if aliases and sha256_json(aliases) != runtime.get("prior_aliases_sha256"):
        raise ValueError("official prior alias digest mismatch")
    if set(aliases) - (required | excluded_targets):
        raise ValueError("prior aliases contain noncanonical target identities")
    source_model = runtime.get("prior_source_model")
    if prior_path.suffix in {".pickle", ".pkl"}:
        with prior_path.open("rb") as handle:
            values = pickle.load(handle)
        if not isinstance(values, dict):
            raise ValueError("official prior must be a gene/vector dictionary")
        widths = set()
        for name, vector in values.items():
            v = np.asarray(vector)
            if not isinstance(name, str) or not name or v.ndim != 1 or not len(v):
                raise ValueError("invalid raw prior gene/vector identity")
            if v.dtype.kind not in "fiu" or not np.isfinite(v).all():
                raise ValueError("raw prior contains nonfinite or nonnumeric vectors")
            widths.add(len(v))
        if len(widths) != 1:
            raise ValueError("raw prior mixes embedding dimensions")
        requested_sources = {aliases.get(gene, gene) for gene in required} | {"ctrl"}
        source = pd.DataFrame.from_dict(
            {name: vector for name, vector in values.items() if name in requested_sources},
            orient="index",
            columns=range(next(iter(widths))),
        )
    elif prior_path.suffix == ".npz":
        gene_key = runtime.get("prior_gene_key", "genes")
        embedding_key = runtime.get("prior_embedding_key", "embeddings")
        with np.load(prior_path, allow_pickle=False) as archive:
            genes = archive[gene_key]
            vectors = archive[embedding_key]
            if genes.ndim != 1 or vectors.ndim != 2 or len(genes) != len(vectors):
                raise ValueError("raw NPZ prior gene/vector axes disagree")
            if genes.dtype.kind not in "US" or vectors.dtype.kind not in "fiu":
                raise ValueError("raw NPZ prior requires string identities and numeric vectors")
            names = [
                name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in genes
            ]
            source = pd.DataFrame(vectors, index=names)
            if "model" in archive:
                model = archive["model"]
                if model.ndim != 0 or model.dtype.kind not in "US":
                    raise ValueError("raw NPZ model provenance must be a scalar string")
                raw_model = model.item()
                observed_model = (
                    raw_model.decode("utf-8") if isinstance(raw_model, bytes) else str(raw_model)
                )
                if source_model is not None and observed_model != source_model:
                    raise ValueError("raw NPZ source-model provenance mismatch")
                source_model = observed_model
    else:
        source = pd.read_csv(prior_path, index_col=0)
    if not source.index.is_unique or any(
        not isinstance(name, str) or not name for name in source.index
    ):
        raise ValueError("official prior gene index is duplicated or invalid")
    if not source.shape[1] or (
        expected_dimension is not None and source.shape[1] != expected_dimension
    ):
        raise ValueError("raw prior dimension differs from its frozen declaration")
    if source.to_numpy().dtype.kind not in "fiu" or not np.isfinite(source.to_numpy()).all():
        raise ValueError("raw prior contains nonfinite or nonnumeric vectors")
    missing = {gene for gene in required if aliases.get(gene, gene) not in source.index}
    authorized_missing = missing & excluded_targets
    if authorized_missing & fit_targets:
        raise ValueError("excluded missing-prior target remains in prepared fit data")
    if len(authorized_missing) > 1:
        raise ValueError("authorized condition exclusion must identify one unique missing gene")
    if missing - authorized_missing:
        raise ValueError(f"official prior misses canonical fit/test targets: {sorted(missing)}")
    required -= authorized_missing
    if not required:
        raise ValueError("raw prior has no remaining canonical targets")
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
        "prior_kind": kind,
        "prior_dimension": source.shape[1],
        "prior_representation": representation,
        "prior_source_model": source_model,
        "authorized_exclusions": list(excluded_conditions),
        "excluded_missing_target_genes": sorted(authorized_missing),
    }


def resolve_training_recipe(training: dict) -> dict:
    """Retain the official train API while preventing stop before the fixed budget."""
    resolved = dict(training)
    epochs, patience = int(resolved["n_epochs"]), int(resolved["patience"])
    if epochs <= 0 or patience <= 0:
        raise ValueError("Scouter epoch and patience budgets must be positive")
    resolved["patience"] = max(epochs, patience)
    return resolved


def declared_prior_runtime(config: dict, runtime: dict) -> dict:
    """Only the sealed config may authorize omitting canonical conditions."""
    exclusions = config.get("data", {}).get("omit_condition_ids", ())
    if not isinstance(exclusions, (tuple, list)):
        raise ValueError("config condition exclusions must be an explicit identity list")
    declared = tuple(exclusions)
    runtime_exclusions = runtime.get("excluded_conditions")
    if runtime_exclusions is not None and (
        not isinstance(runtime_exclusions, (tuple, list)) or tuple(runtime_exclusions) != declared
    ):
        raise ValueError("config/runtime authorized condition exclusions disagree")
    return {**runtime, "excluded_conditions": declared}


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
    prior_runtime = declared_prior_runtime(config, runtime)
    if set((*prepared.train_conditions, *prepared.val_conditions)) & set(
        prior_runtime.get("excluded_conditions", ())
    ):
        raise ValueError("authorized excluded conditions remain in prepared fit data")
    embeddings, prior_receipt = load_official_prior(prior_runtime, required)
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
    training = resolve_training_recipe(config["training"])
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
    if max_updates is None and len(model.loss_history["train_loss"]) != training["n_epochs"]:
        raise RuntimeError("Scouter did not complete the authorized fixed epoch budget")
    torch.save(deepcopy(model.network.state_dict()), output / "last.pt")
    return FittedAdapter(
        model=model,
        receipt={
            "official_sha": OFFICIAL_SHA,
            "official_training_api": "scouter.Scouter.train",
            **prior_receipt,
            "ctrl_embedding": "official zero vector if absent in prior",
            "checkpoint_role": "last",
            "last_capture": "before_official_best_restore" if final_states else "final_state",
            "official_selection": "validation allowed-expression loss; min_delta=0.001",
            "official_dropout_statistics_axis": "allowed expression genes only",
            "loss_history": model.loss_history,
            "epochs_requested": training["n_epochs"],
            "epochs_completed": len(model.loss_history["train_loss"]),
            "early_stopping": False,
            "configured_patience": config["training"]["patience"],
            "resolved_patience": training["patience"],
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
