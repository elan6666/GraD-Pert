"""Adapt pinned GEARS batches to the frozen J2 and C2 fit partitions."""

from __future__ import annotations

import math
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from benchmarks.expression_holdout.contract import FittedAdapter
from benchmarks.expression_holdout.gears import (
    OFFICIAL_SHA,
    allowed_training_statistics,
    copy_official_assets,
    masked_official_loss,
    official_control_metadata,
)
from benchmarks.expression_holdout.scouter import FitBudgetReached, require_sanitized_expression
from gradpert.hashing import sha256_file, sha256_json

ASSETS = Path("/data/yilangliu/GraD-Pert/benchmark-data/v2-baseline-7bdfb7b/gears")
ASSET_HASHES = {
    "gene2go_all.pkl": "f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771",
    "essential_all_data_pert_genes.pkl": (
        "46c3dfe354d8ad5c0da22c69f3d0ca451987b1a61ed9d984279b22b9565ff8d7"
    ),
}
GO_SHA = "99622d9215462e7bbea99a5f6c0b1e86febfef9f57226269418053484fcb3f9f"


class GearsCSR(sparse.csr_matrix):
    """Bridge the pinned GEARS pandas mask to SciPy sparse indexing."""

    def __getitem__(self, index):
        if isinstance(index, pd.Series):
            index = index.to_numpy()
        return super().__getitem__(index)


class Batch:
    """Tensor shape used by the released GEARS PyG loader."""

    def __init__(self, x, pert_idx, pert, y=None, de_idx=None):
        self.x = torch.from_numpy(x.reshape(-1, 1))
        self.batch = torch.arange(len(x)).repeat_interleave(x.shape[1])
        self.pert_idx = pert_idx
        self.pert = pert
        self.y = torch.from_numpy(y) if y is not None else None
        self.de_idx = de_idx

    def to(self, device):
        self.batch = self.batch.to(device)
        self.x = self.x.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self


class FitLoader:
    """Materialize the current batch from frozen fit rows and paired controls."""

    def __init__(self, matrix, rows, paired, names, perturbations, ranks, batch_size, train):
        self.matrix = matrix
        self.rows = rows
        self.paired = paired
        self.names = names
        self.perturbations = perturbations
        self.ranks = ranks
        self.batch_size = batch_size
        self.train = train

    def __len__(self):
        count = len(self.rows) / self.batch_size
        return math.floor(count) if self.train else math.ceil(count)

    def __iter__(self):
        count = len(self.rows)
        order = torch.randperm(count).numpy() if self.train else np.arange(count)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            rows = self.rows[order[start : start + self.batch_size]]
            names = self.names[rows].tolist()
            yield Batch(
                self.matrix[self.paired[rows]].toarray(),
                [self.perturbations[name] for name in names],
                names,
                self.matrix[rows].toarray(),
                [self.ranks[name] for name in names],
            )


def role_statistics(matrix, names, genes, rows, controls, allowed):
    """Use the released absolute t-test ranking on each role's visible axis."""
    from gears import data_utils

    take = np.concatenate((rows, controls))
    role = anndata.AnnData(
        matrix[take][:, allowed],
        obs=pd.DataFrame(
            {"condition": names[take], "cell_type": "frozen_source"}, index=take.astype(str)
        ),
        var=pd.DataFrame(index=np.asarray(genes)[allowed]),
    )
    role = data_utils.get_DE_genes(official_control_metadata(role), skip_calc_de=True)
    counts = role.obs.condition_name.value_counts()
    keep = counts[counts >= 2].index
    rankable = role[role.obs.condition_name.isin(keep)]
    groups = role.obs.condition.to_numpy()
    control_mean = np.asarray(role.X[groups == "ctrl"].mean(axis=0)).ravel()
    rankings = data_utils.rank_genes_groups_by_cov(
        rankable,
        groupby="condition_name",
        control_group="ctrl_1",
        covariate="cell_type",
        n_genes=len(allowed),
        return_dict=True,
    )
    by_gene = {gene: i for i, gene in enumerate(role.var_names)}
    nonzero, ranks, fallbacks = {}, {"ctrl": allowed[:20]}, []
    order = np.argsort(groups, kind="stable")
    unique, starts = np.unique(groups[order], return_index=True)
    for name, members in zip(unique, np.split(order, starts[1:]), strict=True):
        if name == "ctrl":
            continue
        mean = np.asarray(role.X[members].mean(axis=0)).ravel()
        nonzero[name] = allowed[mean != 0]
        name_key = str(role.obs.condition_name.iloc[members[0]])
        if name_key in rankings:
            ranking = np.asarray([by_gene[gene] for gene in rankings[name_key]])
        else:
            ranking = np.arange(len(allowed))
            fallbacks.append(name)
        non_dropout = (mean != 0) | (control_mean == 0)
        ranked = ranking[non_dropout[ranking]]
        if len(ranked) < 20:
            rest = ranking[~non_dropout[ranking]]
            ranked = np.concatenate((ranked, rest))
        ranks[name] = allowed[ranked[:20]]
    return nonzero, ranks, fallbacks


def fit(prepared, config, runtime, output, *, on_epoch):
    import gears
    import gears.gears as module

    matrix = prepared.adata.X
    allowed = np.asarray(prepared.allowed_indices)
    require_sanitized_expression(prepared.adata, prepared.excluded_indices)
    names = prepared.adata.obs.condition.astype(str).to_numpy()
    row_names = tuple(map(str, prepared.adata.obs_names))
    lookup = {row: i for i, row in enumerate(row_names)}
    genes = list(prepared.gene_ids)
    train, val, controls = (
        np.asarray([lookup[row] for row in group])
        for group in (prepared.train_row_ids, prepared.val_row_ids, prepared.control_row_ids)
    )
    data_root = output / "official_data"
    (data_root / "frozen").mkdir(parents=True)
    asset_hashes = copy_official_assets(
        {
            "gears_assets": ASSETS,
            "gears_asset_sha256": ASSET_HASHES,
            "gears_go_csv_sha256": GO_SHA,
        },
        data_root,
    )
    data = gears.PertData(str(data_root))
    data.set_pert_genes()
    targets = {
        name: [gene for gene in name.split("+") if gene != "ctrl"]
        for name in (*prepared.train_conditions, *prepared.val_conditions, "ctrl")
    }
    required = {gene for genes_in_condition in targets.values() for gene in genes_in_condition}
    if (required | set(runtime["required_target_genes"])) - data.node_map_pert.keys():
        raise ValueError("released GEARS graph is missing a frozen condition identity")
    perturbations = {
        name: [data.node_map_pert[gene] for gene in values] or [-1]
        for name, values in targets.items()
    }
    train_nonzero, train_ranks, train_fallbacks = role_statistics(
        matrix, names, genes, train, controls, allowed
    )
    _, val_ranks, val_fallbacks = role_statistics(matrix, names, genes, val, controls, allowed)
    if any(not len(indices) for indices in train_nonzero.values()):
        raise ValueError("a GEARS train condition has no nonzero allowed expression")
    rng = np.random.RandomState(runtime["seed"])
    paired = np.arange(len(row_names))
    for mapping in (prepared.train_control_indices, prepared.val_control_indices):
        for row, candidates in mapping.items():
            position = lookup[row]
            # A random position in the tuple is equivalent to random choice of an element.
            chosen = candidates[rng.randint(len(candidates))]
            if prepared.row_context_ids[position] != prepared.row_context_ids[chosen]:
                raise ValueError("GEARS control pair escaped the fit context")
            paired[position] = chosen
    train_cfg = dict(config["training"])
    batch_size = train_cfg.pop("batch_size")
    eval_batch = train_cfg.pop("eval_batch_size")
    rows_for_graph = np.sort(np.concatenate((train, controls)))
    graph_adata = prepared.adata[rows_for_graph].copy()
    graph_adata.X = GearsCSR(graph_adata.X)
    graph_adata.obs["condition_name"] = graph_adata.obs.condition
    graph_adata.uns["non_zeros_gene_idx"] = train_nonzero
    data.adata = graph_adata
    data.node_map = {name: i for i, name in enumerate(genes)}
    data.gene_names = graph_adata.var.gene_name
    data.dataset_name = "frozen"
    data.split = "frozen"
    data.seed = runtime["seed"]
    data.train_gene_set_size = 0.75
    data.set2conditions = {"train": [*prepared.train_conditions, "ctrl"]}
    data.subgroup = None
    data.dataloader = {
        "train_loader": FitLoader(
            matrix,
            np.concatenate((train, controls)),
            paired,
            names,
            perturbations,
            train_ranks,
            batch_size,
            True,
        ),
        "val_loader": FitLoader(
            matrix,
            np.concatenate((val, controls)),
            paired,
            names,
            perturbations,
            val_ranks,
            batch_size,
            False,
        ),
    }
    model = gears.GEARS(data, device=runtime["device"])
    validation_history = []
    budget_reached = False
    original_loss, original_metrics = module.loss_fct, module.compute_metrics
    masked_loss = masked_official_loss(original_loss, allowed, static_filter=True)
    calls = 0

    def track_metrics(results):
        nonlocal calls
        metrics, by_condition = original_metrics(results)
        calls += 1
        # Upstream GEARS evaluates train and then validation after each epoch.
        if calls % 2 == 0:
            value = float(metrics["mse_de"])
            if not math.isfinite(value):
                raise ValueError("GEARS produced a nonfinite validation top20 MSE")
            validation_history.append(value)
            if on_epoch is not None:
                on_epoch(len(validation_history), value)
        return metrics, by_condition

    module.compute_metrics = track_metrics
    module.loss_fct = masked_loss
    try:
        with allowed_training_statistics(module, allowed):
            model.model_initialize(**config["model"])
            try:
                model.train(**train_cfg)
            except FitBudgetReached:
                budget_reached = True
    finally:
        module.compute_metrics = original_metrics
        module.loss_fct = original_loss
    if not validation_history:
        raise RuntimeError("GEARS did not complete validation of an epoch")
    if not budget_reached and len(validation_history) != train_cfg["epochs"]:
        raise RuntimeError("GEARS did not complete the configured epoch count")
    torch.save(
        {"model_state_dict": model.model.state_dict(), "model_config": model.config},
        output / "last.pt",
    )
    validation_rows = data.dataloader["val_loader"].rows[:300]
    chosen_name = str(names[validation_rows[0]])
    prediction = predict(
        FittedAdapter(model, {}),
        matrix[paired[validation_rows]].toarray(),
        chosen_name.split("+"),
        genes,
        cell_batch=eval_batch,
        query_count=300,
        seed=runtime["seed"],
    )
    model.dataloader = {}
    return FittedAdapter(
        model=model,
        receipt={
            "official_sha": OFFICIAL_SHA,
            "official_training_api": "gears.GEARS.train",
            "epochs_completed": len(validation_history),
            "epochs_requested": train_cfg["epochs"],
            "loss_history": {"val_top20_mse": validation_history},
            "wall_budget_reached": budget_reached,
            "checkpoint_role": "last",
            "official_asset_sha256": asset_hashes,
            "output_validation_shape": list(prediction.shape),
            "context_pairing": "fixed uniform replacement within compatible source control pools",
            "paired_row_positions_sha256": sha256_json(paired.tolist()),
            "native_test_targets_sha256": sha256_json(runtime["required_target_genes"]),
            "ranking_scope": "train and validation separately, pooled source lines, allowed genes",
            "ranking_singleton_fallbacks": {"train": train_fallbacks, "val": val_fallbacks},
            "coexpression_sha256": sha256_file(
                data_root / "frozen/frozen_1_0.75_0.4_20_co_expression_network.csv"
            ),
        },
    )


def predict(fitted, controls, target_genes, gene_ids, *, cell_batch, query_count, seed):
    model = fitted.model
    controls = np.asarray(controls, dtype=np.float32)
    if controls.ndim != 2 or controls.shape[1] != len(gene_ids) or not len(controls):
        raise ValueError("GEARS needs nonempty controls on the complete expression axis")
    if list(gene_ids) != model.gene_list or cell_batch <= 0:
        raise ValueError("GEARS frozen output order or prediction batch changed")
    indices = [model.node_map_pert[gene] for gene in target_genes if gene != "ctrl"]
    model.model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(controls), cell_batch):
            x = controls[start : start + cell_batch]
            batch = Batch(x, [indices] * len(x), []).to(model.device)
            outputs.append(model.model(batch).cpu().numpy())
    result = np.concatenate(outputs)
    if result.shape != controls.shape or not np.isfinite(result).all():
        raise ValueError("GEARS changed the frozen full-axis output shape or returned invalid data")
    return result
