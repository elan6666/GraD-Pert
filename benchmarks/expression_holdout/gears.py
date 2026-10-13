"""Frozen GEARS training; canonical expression and basal-context adaptation."""

from __future__ import annotations

import importlib
import inspect
import pickle
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

OFFICIAL_SHA = "f374e43e197b295016d80395d7a54ddb81cc6769"
CACHE_SERVER_ROOT = Path("/data/yilangliu")


def copy_official_assets(runtime: dict, destination: Path) -> dict[str, str]:
    """Check all pinned inputs before copying, then check the actual copies."""
    from gradpert.hashing import sha256_file

    names = ("gene2go_all.pkl", "essential_all_data_pert_genes.pkl")
    expected = dict(runtime["gears_asset_sha256"])
    if set(expected) != set(names):
        raise ValueError("GEARS asset digest manifest must name the two official graph assets")
    go_csv = "go_essential_all/go_essential_all.csv"
    expected[go_csv] = runtime["gears_go_csv_sha256"]
    names = (*names, go_csv)
    assets = Path(runtime["gears_assets"])
    observed = {}
    for filename in names:
        source = assets / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing frozen official GEARS graph asset: {source}")
        observed[filename] = sha256_file(source)
        if observed[filename] != expected[filename]:
            raise ValueError(f"GEARS official graph asset digest mismatch: {filename}")
    for filename in names:
        target = destination / filename
        if target.exists():
            raise FileExistsError(f"GEARS asset copy would overwrite an existing file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets / filename, target)
        if sha256_file(target) != expected[filename]:
            raise ValueError(f"GEARS official asset changed during copying: {filename}")
    return observed


def official_control_metadata(adata: Any):
    """Match frozen upstream integer control labels before its .loc assignment."""
    # GEARS get_DE_genes derives 0 for gene+ctrl/combo and 1 for ctrl. Native
    # sanitation uses a bool flag; pandas strict dtype assignment rejects ints
    # into that column. This conversion changes labels only, never expression.
    adata.obs["control"] = (adata.obs.condition.astype(str) == "ctrl").astype(np.int64)
    return adata


def masked_official_loss(original: Any, allowed: np.ndarray):
    """Official fourth-power/direction loss on the allowed axis, including ctrl."""
    remap = {int(index): position for position, index in enumerate(allowed)}

    def loss(pred, truth, perts, ctrl=None, direction_lambda=1e-3, dict_filter=None):
        masks = {
            condition: [remap[int(i)] for i in indices if int(i) in remap]
            for condition, indices in (dict_filter or {}).items()
        }
        if any(p != "ctrl" and not masks.get(p) for p in set(perts)):
            raise ValueError("condition has no allowed official nonzero loss genes")
        return original(
            pred[:, allowed],
            truth[:, allowed],
            perts,
            ctrl=ctrl[allowed] if ctrl is not None else None,
            direction_lambda=direction_lambda,
            dict_filter=masks,
        )

    return loss


def context_graph_builder(original: Any, prepared: Any, pair_audit: list):
    """Delegate every graph to upstream, caching identical compatible pools.

    Upstream only reads the supplied AnnData views. Reusing a pool and avoiding
    per-cell deep copies does not change its sampler, graph tensors or row order.
    """
    names = tuple(map(str, prepared.adata.obs_names))
    lookup = {name: index for index, name in enumerate(names)}
    maps = {**prepared.train_control_indices, **prepared.val_control_indices}
    control_flags = prepared.adata.obs.condition.astype(str).to_numpy() == "ctrl"
    pools: dict[str, list[tuple[tuple[int, ...], Any]]] = {}
    statistics = {"pool_materializations": 0, "pool_hits": 0, "official_row_calls": 0}

    def build(instance, split_adata, pert_category, num_samples=1):
        if pert_category == "ctrl":
            return original(split_adata, pert_category, num_samples)
        rows = split_adata.obs_names[split_adata.obs.condition.astype(str) == pert_category]
        original_controls = instance.ctrl_adata
        graphs = []
        try:
            for row in rows:
                row = str(row)
                candidate_ids = maps.get(row, ())
                context = prepared.row_context_ids[lookup[row]]
                cached = next(
                    (entry for entry in pools.get(context, ()) if entry[0] == candidate_ids), None
                )
                if cached is None:
                    candidates = tuple(int(i) for i in candidate_ids)
                    if not candidates or any(
                        i < 0
                        or i >= len(names)
                        or prepared.row_context_ids[i] != context
                        or not control_flags[i]
                        for i in candidates
                    ):
                        raise ValueError(f"invalid canonical control candidates for {row}")
                    pool_view = prepared.adata[[names[i] for i in candidates]]
                    cached = (candidates, pool_view)
                    pools.setdefault(context, []).append(cached)
                    statistics["pool_materializations"] += 1
                else:
                    statistics["pool_hits"] += 1
                candidates, pool_view = cached
                # The original sampler draws uniformly with replacement. Keep that
                # sampler, but give it only the row's compatible candidate pool.
                instance.ctrl_adata = pool_view
                target = split_adata[[row]]
                original_randint = np.random.randint
                sampled: list[int] = []

                def record_randint(
                    low,
                    high=None,
                    size=None,
                    dtype=int,
                    *,
                    pool=candidates,
                    sampler=original_randint,
                    recorded=sampled,
                ):
                    if low != 0 or high != len(pool) or size != num_samples:
                        raise ValueError("frozen GEARS basal sampler signature changed")
                    chosen = sampler(low, high=high, size=size, dtype=dtype)
                    recorded.extend(map(int, np.asarray(chosen).reshape(-1)))
                    return chosen

                np.random.randint = record_randint
                try:
                    built = original(target, pert_category, num_samples)
                    statistics["official_row_calls"] += 1
                finally:
                    np.random.randint = original_randint
                if len(sampled) != len(built):
                    raise ValueError("official basal sampler/graph count mismatch")
                pair_audit.extend((row, names[candidates[i]]) for i in sampled)
                graphs.extend(built)
        finally:
            instance.ctrl_adata = original_controls
        return graphs

    build.cache_statistics = statistics
    return build


@contextmanager
def allowed_training_statistics(module: Any, allowed: np.ndarray):
    """Limit official coexpression and validation metrics to visible genes."""
    original_network, original_metrics = module.get_similarity_network, module.compute_metrics

    def network(*args, **kwargs):
        network_type = kwargs.get("network_type", args[0] if args else None)
        if network_type == "co-express":
            if "adata" not in kwargs:
                raise ValueError("unexpected frozen GEARS network-call signature")
            kwargs["adata"] = kwargs["adata"][:, allowed].copy()
        return original_network(*args, **kwargs)

    def metrics(results):
        scoped = dict(results)
        scoped["pred"] = results["pred"][:, allowed]
        scoped["truth"] = results["truth"][:, allowed]
        return original_metrics(scoped)

    module.get_similarity_network, module.compute_metrics = network, metrics
    try:
        yield
    finally:
        module.get_similarity_network, module.compute_metrics = original_network, original_metrics


def _ranking_metadata(adata, utils, allowed):
    """Use the official ranking helper with only allowed numeric coordinates."""
    ranking_adata = adata[:, allowed].copy()
    counts = ranking_adata.obs.condition_name.astype(str).value_counts()
    singleton = set(counts[counts < 2].index)
    control_names = set(
        ranking_adata.obs.loc[ranking_adata.obs.condition == "ctrl", "condition_name"]
    )
    if singleton & control_names:
        raise ValueError("GEARS DE ranking needs at least two compatible control rows")
    rankable = ranking_adata[~ranking_adata.obs.condition_name.astype(str).isin(singleton)].copy()
    rankings = utils.rank_genes_groups_by_cov(
        rankable,
        groupby="condition_name",
        covariate="cell_type",
        control_group="ctrl_1",
        n_genes=len(allowed),
        key_added="rank_genes_groups_cov_all",
        return_dict=True,
    )
    for name in singleton:
        rankings[str(name)] = list(map(str, ranking_adata.var_names))
    adata.uns["rank_genes_groups_cov_all"] = rankings
    # Call official dropout logic on allowed expression, then map its numeric
    # indices back to the full identity axis consumed by its unchanged model.
    ranking_adata.uns["rank_genes_groups_cov_all"] = rankings
    filtered = utils.get_dropout_non_zero_genes(ranking_adata)
    for name, value in filtered.uns.items():
        if name in {"non_zeros_gene_idx", "non_dropout_gene_idx"}:
            adata.uns[name] = {
                key: allowed[np.asarray(indices, dtype=int)] for key, indices in value.items()
            }
    # GEARS constructor expects condition-name keyed nonzero masks.
    for condition in adata.obs.condition.astype(str).unique():
        if condition == "ctrl":
            continue
        subset = adata[adata.obs.condition.astype(str) == condition]
        condition_names = subset.obs.condition_name.astype(str).unique()
        if len(condition_names) != 1:
            raise ValueError("GEARS condition has ambiguous context-qualified DE identity")
        mean = np.asarray(subset.X[:, allowed].mean(axis=0)).reshape(-1)
        indices = allowed[np.flatnonzero(mean != 0)]
        if not len(indices):
            raise ValueError("GEARS condition has no allowed nonzero genes")
        adata.uns.setdefault("non_zeros_gene_idx", {})[str(condition_names[0])] = indices
    return {"singleton_rank_fallbacks": sorted(map(str, singleton)), "ranking_axis": "allowed_only"}


def reuse_ranking_metadata(adata, allowed: np.ndarray, runtime: dict):
    """Reuse sealed official statistics only after exact sanitized-input checks."""
    cache = runtime.get("gears_ranking_cache")
    if cache is None:
        return None
    import hashlib

    import anndata
    import pandas as pd

    from gradpert.hashing import sha256_file

    path = Path(cache["path"])
    if not path.resolve().is_relative_to(CACHE_SERVER_ROOT):
        raise ValueError("GEARS ranking cache must stay on the server")
    method_hash = hashlib.sha256(inspect.getsource(_ranking_metadata).encode()).hexdigest()
    if method_hash != cache["ranking_method_sha256"] or sha256_file(path) != cache["sha256"]:
        raise ValueError("official ranking method/cache identity changed")
    parent = Path(cache["parent_run_root"])
    if (
        path.resolve()
        != (parent / "fit/official_data/expression_holdout/perturb_processed.h5ad").resolve()
    ):
        raise ValueError("ranking cache must be the parent's actual official output")
    for file, key in (
        ("RUN.json", "parent_run_sha256"),
        ("data-receipt.json", "data_receipt_sha256"),
    ):
        if sha256_file(parent / file) != cache[key]:
            raise ValueError("ranking cache parent identity changed")
    import json

    parent_identity = json.loads((parent / "RUN.json").read_text())
    if (
        parent_identity["source_commit"] != cache["source_commit"]
        or parent_identity["config_sha256"] != cache["config_sha256"]
        or parent_identity["environment_lock_sha256"] != runtime["environment_lock_sha256"]
    ):
        raise ValueError("ranking cache source/config/environment differs")
    cached = anndata.read_h5ad(path)
    # H5AD writing categorizes string columns; compare their exact values/axes.
    pd.testing.assert_frame_equal(cached.obs.astype(object), adata.obs.astype(object))
    pd.testing.assert_frame_equal(cached.var.astype(object), adata.var.astype(object))
    if cached.X.shape != adata.X.shape or (cached.X != adata.X).nnz:
        raise ValueError("ranking cache sanitized expression differs")
    allowed_names = set(map(str, adata.var_names[allowed]))
    for values in cached.uns["rank_genes_groups_cov_all"].values():
        if not set(map(str, values)) <= allowed_names:
            raise ValueError("ranking cache includes excluded expression genes")
    allowed_set = set(map(int, allowed))
    for key in ("non_zeros_gene_idx", "non_dropout_gene_idx"):
        if any(not set(map(int, ids)) <= allowed_set for ids in cached.uns[key].values()):
            raise ValueError("ranking cache numeric mask includes excluded genes")
    for key in ("rank_genes_groups_cov_all", "non_zeros_gene_idx", "non_dropout_gene_idx"):
        adata.uns[key] = cached.uns[key]
    counts = adata.obs.condition_name.astype(str).value_counts()
    return {
        "singleton_rank_fallbacks": sorted(map(str, counts[counts < 2].index)),
        "ranking_axis": "allowed_only",
        "cache_reused": True,
        "cache_sha256": cache["sha256"],
        "ranking_method_sha256": method_hash,
        "cache_source_commit": cache["source_commit"],
    }


def fit(prepared, config: dict, runtime: dict, output: Path, max_updates=None):
    import torch
    from scipy import sparse

    from benchmarks.common.official_checkout import verify_official_checkout
    from benchmarks.expression_holdout.contract import FittedAdapter
    from benchmarks.expression_holdout.scouter import (
        _allowed,
        _UpdateLimitReached,
        bounded_official_updates,
        require_sanitized_expression,
    )
    from gradpert.hashing import sha256_file, sha256_json

    root = verify_official_checkout(runtime["official_checkout"], OFFICIAL_SHA)
    package = importlib.import_module("gears")
    module = importlib.import_module("gears.gears")
    data_utils = importlib.import_module("gears.data_utils")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError("GEARS import is not from frozen checkout")
    if config["model"].get("uncertainty", False):
        raise ValueError("this frozen adaptation supports official default uncertainty=False only")
    allowed = _allowed(prepared)
    if len(allowed) < 20:
        raise ValueError("GEARS official DE selection requires at least20 allowed genes")
    adata = prepared.adata.copy()
    adata.X = sparse.csr_matrix(adata.X)
    require_sanitized_expression(adata, prepared.excluded_indices)
    # Frozen GEARS parses single targets as gene+ctrl and combo targets as gene+gene.
    if any(c != "ctrl" and len(c.split("+")) != 2 for c in adata.obs.condition.astype(str)):
        raise ValueError("GEARS requires canonical gene+ctrl/combo condition strings")
    torch.manual_seed(int(runtime["seed"]))
    np.random.seed(int(runtime["seed"]))
    output = Path(output)
    data_root = output / "official_data"
    data_root.mkdir(parents=True, exist_ok=False)
    asset_hashes = copy_official_assets(runtime, data_root)
    data = package.PertData(str(data_root))
    adata = data_utils.get_DE_genes(official_control_metadata(adata), skip_calc_de=True)
    ranking_receipt = reuse_ranking_metadata(adata, allowed, runtime)
    if ranking_receipt is None:
        ranking_receipt = _ranking_metadata(adata, data_utils, allowed)
    pairs = []
    original_builder = data.create_cell_graph_dataset
    builder = context_graph_builder(original_builder, prepared, pairs)
    data.create_cell_graph_dataset = MethodType(builder, data)
    try:
        data.new_data_process("expression_holdout", adata=adata, skip_calc_de=True)
    finally:
        del data.create_cell_graph_dataset
    if tuple(map(str, data.adata.obs_names)) != tuple(map(str, prepared.adata.obs_names)):
        raise ValueError("GEARS filtered canonical fit rows")
    # Ctrl DE arrays are not used for selection but must not point into the
    # excluded axis through the upstream [-1] sentinel.
    for graph in data.dataset_processed.get("ctrl", ()):
        graph.de_idx = list(map(int, allowed[:20]))
    for condition, graphs in data.dataset_processed.items():
        for graph in graphs:
            if set(map(int, graph.de_idx)) & set(prepared.excluded_indices):
                raise ValueError(f"GEARS DE evaluation includes excluded expression: {condition}")
    split_path = data_root / "canonical_split.pkl"
    split = {
        "train": [*prepared.train_conditions, "ctrl"],
        "val": [*prepared.val_conditions, "ctrl"],
        "test": [],
    }
    with split_path.open("wb") as handle:
        pickle.dump(split, handle)
    data.prepare_split(split="custom", seed=int(runtime["seed"]), split_dict_path=str(split_path))
    train_cfg = dict(config["training"])
    train_batch = int(train_cfg.pop("batch_size"))
    eval_batch = int(train_cfg.pop("eval_batch_size"))
    data.get_dataloader(batch_size=train_batch, test_batch_size=eval_batch)
    data.dataloader.pop("test_loader", None)
    if not len(data.dataloader["train_loader"]):
        raise ValueError("GEARS official drop_last loader is empty")
    original_loss = module.loss_fct
    module.loss_fct = masked_official_loss(original_loss, allowed)
    counter = {"updates": 0}
    stopped = False
    try:
        with allowed_training_statistics(module, allowed):
            model = package.GEARS(data, device=runtime["device"], weight_bias_track=False)
            model.model_initialize(**config["model"])
            with bounded_official_updates(module, max_updates, counter):
                try:
                    model.train(**train_cfg)
                except _UpdateLimitReached:
                    stopped = True
    finally:
        module.loss_fct = original_loss
    # Frozen train leaves final model in .model and best in .best_model.
    torch.save(model.model.state_dict(), output / "last.pt")
    # Remove only exact reconstructible official scratch PKLs, preserving their
    # hashes before deletion. Official graph asset provenance is already above.
    cleanup = []
    for path in (
        data_root / "gene2go_all.pkl",
        data_root / "essential_all_data_pert_genes.pkl",
        split_path,
        data_root / "expression_holdout/data_pyg/cell_graphs.pkl",
    ):
        cleanup.append({"path": str(path.relative_to(output)), "sha256": sha256_file(path)})
        path.unlink()
    return FittedAdapter(
        model=model,
        receipt={
            "official_sha": OFFICIAL_SHA,
            "official_training_api": "gears.GEARS.train",
            "checkpoint_role": "last",
            "last_capture": "official model.model final state",
            "official_selection": "validation top20 DE MSE on allowed genes",
            "epochs_requested": train_cfg["epochs"],
            "epochs_completed": None if stopped else train_cfg["epochs"],
            "validation_batch_size": train_batch,
            "official_test_loader_batch_size": eval_batch,
            "official_asset_sha256": asset_hashes,
            "scratch_cleanup": cleanup,
            "pair_count": len(pairs),
            "pair_sha256": sha256_json(pairs),
            "graph_builder_cache": dict(builder.cache_statistics),
            "context_pairing": "official uniform replacement within compatible pools",
            "diagnostic_update_limit": max_updates,
            "diagnostic_updates_completed": counter["updates"] if max_updates else None,
            "diagnostic_limit_reached": stopped,
            "seen_expression_gene_indices": allowed.tolist(),
            "ranking_receipt": ranking_receipt,
            "control_metadata_encoding": "official 0=perturbed,1=control int labels",
            "adaptations": [
                "input expression exclusion",
                "allowed official loss including ctrl",
                "allowed coexpression/DE/validation statistics",
                "context control pools",
            ],
        },
    )


def predict(fitted, controls, target_genes, gene_ids, *, cell_batch, query_count, seed):
    import torch
    from gears.utils import create_cell_graph_for_prediction
    from torch_geometric.loader import DataLoader

    controls = np.asarray(controls, dtype=np.float32)
    model = fitted.model
    if (
        controls.ndim != 2
        or controls.shape[1] != len(gene_ids)
        or not len(controls)
        or cell_batch <= 0
    ):
        raise ValueError("invalid full-axis GEARS inference controls")
    if tuple(map(str, model.gene_list)) != tuple(gene_ids):
        raise ValueError("GEARS inference gene identity/order mismatch")
    perturbation_indices = [list(model.pert_list).index(gene) for gene in target_genes]
    graphs = [
        create_cell_graph_for_prediction(row, perturbation_indices, list(target_genes))
        for row in controls
    ]
    network = model.model.to(model.device)
    network.eval()
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=cell_batch, shuffle=False):
            outputs.append(network(batch.to(model.device)).detach().cpu().numpy())
    result = np.concatenate(outputs)
    if result.shape != controls.shape or not np.isfinite(result).all():
        raise ValueError("invalid GEARS full-axis predictions")
    return result
