"""Run released scFoundation/GEARS with frozen C2 fitting and scoring inputs."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from benchmarks.common import official_module_session
from benchmarks.common.study_fit_partition import crosscell_fit_rows
from benchmarks.crosscell.common_score_panel import AUDIT, AUDIT_SHA, PANEL, load_panel
from benchmarks.crosscell.official_fit_probe import SUPPORT_SHA
from benchmarks.crosscell.scfoundation_c2_weight_probe import (
    COMMIT,
    FOUNDATION_WEIGHTS,
    FOUNDATION_WEIGHTS_SHA,
    UPSTREAM,
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
from benchmarks.expression_holdout.data import read_allowed_rows
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.gpu_lease import admit_selected_idle, lock
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json
from scripts.v2.shared_gpu import process_info, snapshot

SUPPORT = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
COEX = STUDY / "scfoundation-crosscell-c2-coexpression-e41d27a-a2/coexpression.npz"
GO = UPSTREAM / "GEARS/data/gene2go.pkl"
GO_SHA = "f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771"
COEX_SHA = "aa9ec8578389fedbb8572763f7d69c9d19c91d524f97f76f5023fb47fe000f67"
GENES, BATCH, ACCUMULATION, SEED = 19264, 6, 5, 1
OFFICIAL_EPOCHS, WALL_SECONDS = 15, 10800


def released_module():
    git = ["git", "-C", str(UPSTREAM)]
    if (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != COMMIT
        or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
    ):
        raise ValueError("released scFoundation source changed")
    for path, expected in (
        (FOUNDATION_WEIGHTS, FOUNDATION_WEIGHTS_SHA),
        (GO, GO_SHA),
        (COEX, COEX_SHA),
        (AUDIT, AUDIT_SHA),
        (SUPPORT, SUPPORT_SHA),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen scFoundation input changed: {path.name}")
    # The official module resolves paths from its own GEARS directory.
    os.chdir(UPSTREAM / "GEARS")
    sys.path[:0] = [str(UPSTREAM / "GEARS"), str(UPSTREAM / "model")]
    from gears import GEARS
    from gears.utils import GeneSimNetwork

    return GEARS, GeneSimNetwork


def go_neighbors(ontology, count=21):
    """Exact released Jaccard and top-k rule over the declared C2 target graph."""
    terms = sorted({term for values in ontology.values() for term in values})
    term_index = {term: i for i, term in enumerate(terms)}
    nodes, cols = zip(
        *(
            (i, term_index[t])
            for i, terms_for_gene in enumerate(ontology.values())
            for t in set(terms_for_gene)
        ),
        strict=True,
    )
    incidence = sparse.csr_matrix(
        (np.ones(len(nodes), dtype=np.int32), (nodes, cols)), shape=(len(ontology), len(terms))
    )
    sizes = np.asarray(incidence.sum(axis=1)).ravel()
    if not sizes.size or (sizes == 0).any():
        raise ValueError("a C2 GO node has no annotation")
    names = np.asarray(list(ontology))
    edges = []
    for j, target in enumerate(names):
        intersection = (incidence @ incidence[j].T).toarray().ravel()
        jaccard = intersection / (sizes + sizes[j] - intersection)
        # The released groupby sorts targets and nlargest keeps the first tie.
        selected = np.argsort(-jaccard, kind="stable")[:count]
        edges.extend((names[i], target, float(jaccard[i])) for i in selected)
    return pd.DataFrame(edges, columns=["source", "target", "importance"]).sort_values(
        "target", kind="stable", ignore_index=True
    )


def fitting_data(source, rows, cohort, compatibility, panel):
    data = anndata.read_h5ad(source, backed="r")
    try:
        conditions = data.obs["condition"].astype(str).to_numpy()
        lines = data.obs["cell_line"].astype(str).to_numpy()
        batches = data.obs["batch"].astype(str).to_numpy()
        roles = crosscell_fit_rows(rows, conditions, cohort, compatibility=compatibility)
        fitting = np.sort(np.concatenate(list(roles.values())))
        columns = data.var_names.get_indexer(panel["score_genes"])
        if (
            (columns < 0).any()
            or set(data.var_names[columns]) & set(cohort["masked_genes_in_expression_order"])
            or np.intersect1d(fitting, np.r_[rows["test"], rows["target_control"]]).size
            or set(lines[fitting]) != {"RPE1", "hepg2", "jurkat"}
        ):
            raise ValueError("the C2 fit roles exposed held data")
        matrix = read_allowed_rows(data, fitting, columns, data.n_vars).tocsr()
    finally:
        data.file.close()
    roles = {k: np.searchsorted(fitting, v) for k, v in roles.items()}
    conditions, lines, batches = conditions[fitting], lines[fitting], batches[fitting]
    contexts = defaultdict(list)
    for row in roles["source_control"]:
        contexts[(lines[row], batches[row])].append(row)
    rng = np.random.default_rng(SEED)
    paired = np.full(len(fitting), -1, dtype=np.int64)
    for row in np.r_[roles["train"], roles["val"]]:
        paired[row] = rng.choice(contexts[(lines[row], batches[row])])
    return matrix, conditions, roles, paired


class Batch:
    def __init__(self, controls, positions, perturbations, names, truth=None, de=None):
        native = np.zeros((len(controls), GENES + 1), dtype=np.float32)
        native[:, positions] = controls
        native[:, -1] = native[:, :-1].sum(1, dtype=np.float32)
        if not np.isfinite(native).all() or (native[:, -1] <= 0).any():
            raise ValueError("invalid released pre-normalized scFoundation input")
        self.x = torch.from_numpy(native.ravel())
        self.batch = torch.arange(len(controls), dtype=torch.int64)
        self.pert, self.pert_idx, self.de_idx = names, perturbations, de
        self.y = None
        if truth is not None:
            target = np.zeros((len(controls), GENES), dtype=np.float32)
            target[:, positions] = truth
            self.y = torch.from_numpy(target)

    def to(self, device):
        self.x = self.x.to(device)
        self.batch = self.batch.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self


class FittingLoader:
    def __init__(self, matrix, conditions, paired, positions, targets, de, rows, output=None):
        self.matrix, self.conditions, self.paired, self.positions = (
            matrix,
            conditions,
            paired,
            positions,
        )
        self.targets, self.de, self.rows, self.output = targets, de, rows, output
        self.visits = 0

    def __len__(self):
        return (len(self.rows) + BATCH - 1) // BATCH

    def __iter__(self):
        self.visits += 1
        for i in range(0, len(self.rows), BATCH):
            idx = self.rows[i : i + BATCH]
            names = self.conditions[idx].tolist()
            if self.output is not None and (i == 0 or i % (20 * BATCH) == 0):
                atomic_json(
                    self.output / "training_progress.json",
                    {"pass": self.visits, "batch": i // BATCH + 1, "batches": len(self)},
                )
            yield Batch(
                self.matrix[self.paired[idx]].toarray(),
                self.positions,
                [self.targets[name] for name in names],
                names,
                self.matrix[idx].toarray(),
                [self.de[name] for name in names],
            )


def initialize(output, source, rows, cohort, compatibility, panel, limit, val_limit):
    Gears, network = released_module()
    audit = read_json(AUDIT)["scfoundation"]
    vocabulary = UPSTREAM / "model/OS_scRNA_gene_index.19264.tsv"
    if sha256_file(vocabulary) != audit["vocabulary_sha256"]:
        raise ValueError("released gene vocabulary changed")
    with vocabulary.open() as stream:
        native = [x["gene_name"] for x in csv.DictReader(stream, delimiter="\t")]
    node = {gene: i for i, gene in enumerate(native)}
    aliases = audit["expression"]["aliases"]
    positions = np.array([node[aliases.get(g, g)] for g in panel["score_genes"]])
    if len(node) != GENES or len(set(positions)) != len(positions):
        raise ValueError("ambiguous C2 vocabulary projection")
    matrix, conditions, roles, paired = fitting_data(source, rows, cohort, compatibility, panel)
    fit_names = np.unique(conditions[roles["train"]]).tolist()
    if not set(conditions[roles["val"]]) <= set(fit_names):
        raise ValueError("validation would require DE selection from an unseen fit condition")
    # The official graph uses external annotations, with nodes restricted to the
    # declared C2 perturbation identities. No K562 expression enters graph fitting.
    all_names = [*fit_names, *cohort["condition_ids"]]
    required = {g for name in all_names for g in name.split("+") if g != "ctrl"}
    with GO.open("rb") as file:
        stock = pickle.load(file)
    ontology = {g: v for g, v in stock.items() if g in required}
    if set(ontology) != required:
        raise ValueError("a frozen condition is missing from the audited GO prior")
    perts = sorted(ontology)
    pert_map = {g: i for i, g in enumerate(perts)}
    targets = {name: [pert_map[g] for g in name.split("+") if g != "ctrl"] for name in all_names}
    go = network(go_neighbors(ontology), perts, pert_map)
    with np.load(COEX, allow_pickle=False) as graph:
        coex_edges = torch.from_numpy(graph["edge_index"])
        coex_weights = torch.from_numpy(graph["edge_weight"])
    train = roles["train"]
    ordered_train = train[np.argsort(conditions[train], kind="stable")]
    subgroups = np.split(
        ordered_train,
        np.flatnonzero(conditions[ordered_train][1:] != conditions[ordered_train][:-1]) + 1,
    )
    ctrl_mean = np.asarray(matrix[roles["source_control"]].mean(0)).ravel()
    de, nonzero = {}, {}
    for group in subgroups:
        name = conditions[group[0]]
        mean = np.asarray(matrix[group].mean(0)).ravel()
        present = np.flatnonzero(mean != 0)
        if len(present) < 20:
            raise ValueError(f"too few expressed training genes: {name}")
        top = present[np.argsort(-np.abs(mean[present] - ctrl_mean[present]), kind="stable")[:20]]
        de[name], nonzero[name] = positions[top].tolist(), positions[present].tolist()

    rng = np.random.default_rng(SEED)
    train_selected = rng.permutation(train)[: limit * BATCH]
    val_selected = rng.permutation(roles["val"])[: val_limit * BATCH]
    train_loader = FittingLoader(
        matrix, conditions, paired, positions, targets, de, train_selected, output
    )
    val_loader = FittingLoader(matrix, conditions, paired, positions, targets, de, val_selected)
    # GEARS uses adata here for control mean and condition-keyed loss masks only.
    x = sparse.csr_matrix(
        (ctrl_mean, (np.zeros(len(positions), dtype=int), positions)),
        shape=(len(fit_names) + 1, GENES),
    )
    labels = ["ctrl", *fit_names]
    info = anndata.AnnData(
        x, obs=pd.DataFrame({"condition": labels, "condition_name": labels}, index=labels)
    )
    info.uns["non_zeros_gene_idx"] = nonzero
    pert_data = SimpleNamespace(
        dataloader={"train_loader": train_loader, "val_loader": val_loader},
        adata=info,
        node_map=node,
        node_map_pert=pert_map,
        data_path=str(output),
        dataset_name="C2",
        split="frozen",
        seed=SEED,
        train_gene_set_size=0.75,
        set2conditions={"train": fit_names},
        subgroup=None,
        gi_go=False,
        gene_names=pd.Series(native),
        pert_names=np.asarray(perts),
    )

    class CheckpointedGears(Gears):
        def save_model(self, path):
            torch.save(self.best_model.state_dict(), Path(path) / "checkpoint.pt")

    model = CheckpointedGears(pert_data, device="cuda")
    model.model_initialize(
        hidden_size=512,
        model_type="maeautobin",
        bin_set="autobin_resolution_append",
        load_path=str(FOUNDATION_WEIGHTS),
        finetune_method="frozen",
        accumulation_steps=ACCUMULATION,
        mode="v1",
        highres=0,
        G_go=go.edge_index,
        G_go_weight=go.edge_weight,
        G_coexpress=coex_edges,
        G_coexpress_weight=coex_weights,
    )
    return (
        model,
        positions,
        targets,
        val_loader,
        len(train_loader),
        len(val_loader),
        columns_hash(panel),
    )


def columns_hash(panel):
    return sha256_json(panel["score_genes"])


def run(output, identity, args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.monotonic()
    source, cross, cohort, rows = cohort_inputs()
    panel = load_panel(source, cross, cohort)
    compatibility = read_json(SUPPORT)
    limit, val_limit = 5, 2
    if not args.smoke:
        smoke = read_json(args.smoke_run / "result.json")
        if smoke["status"] != "trained_validation_only" or any(
            smoke[k] != identity[k]
            for k in ("source_commit", "shared_fit_support_file_sha256", "cohort_sha256")
        ):
            raise ValueError("formal run needs this source's validation-only smoke")
        # Reserve full frozen evaluation before choosing a bounded fit prefix.
        budget = (
            WALL_SECONDS - smoke["setup_seconds"] - 128 * smoke["condition_seconds"] * 1.20 - 180
        )
        if budget < 120:
            raise RuntimeError("smoke projects that frozen scoring exceeds the three-hour envelope")
        limit = max(5, min(1500, int(budget / max(smoke["fit_seconds"] / 5, 0.001))))
        limit -= limit % ACCUMULATION
        val_limit = 20
    model, positions, targets, val_loader, fit_batches, val_batches, axis = initialize(
        output, source, rows, cohort, compatibility, panel, limit, val_limit
    )
    setup_seconds = time.monotonic() - start
    fit_start = time.monotonic()
    model.train(epochs=1, result_dir=str(output), lr=0.01, weight_decay=5e-4)
    fit_seconds = time.monotonic() - fit_start
    checkpoint = output / "checkpoint.pt"
    checkpoint_sha = sha256_file(checkpoint)
    frozen = model.best_model
    frozen.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    frozen.requires_grad_(False).eval()
    fit = dict(
        identity,
        setup_seconds=setup_seconds,
        fit_seconds=fit_seconds,
        checkpoint_sha256=checkpoint_sha,
        fit_microbatches=fit_batches,
        val_microbatches=val_batches,
        optimizer_updates=(fit_batches + ACCUMULATION - 1) // ACCUMULATION,
        epochs=1,
        official_epoch_reference=OFFICIAL_EPOCHS,
        batch=BATCH,
        accumulation_steps=ACCUMULATION,
        validation_de_basis="top20_absolute_training_mean_delta_over_source_controls",
        budget_policy="single reduced epoch with a seeded fitting prefix and held validation",
        fit_test_conditions_accessed=0,
        test_panel_gene_order_sha256=axis,
    )
    atomic_json(output / "fit_receipt.json", fit)

    @torch.inference_mode()
    def predict(name, controls):
        y = []
        for i in range(0, len(controls), BATCH):
            values = controls[i : i + BATCH]
            batch = Batch(values, positions, [targets[name]] * len(values), [name] * len(values))
            y.append(frozen(batch.to("cuda"))[:, positions].cpu().numpy())
        return np.concatenate(y)

    if args.smoke:
        # Estimate frozen scoring using genuine source validation pairing, never K562.
        sample = next(iter(val_loader))
        visible = sample.x.reshape(-1, GENES + 1).numpy()[:, positions]
        t = time.monotonic()
        prediction = predict(sample.pert[0], np.resize(visible, (300, len(positions))))
        if prediction.shape != (300, len(positions)) or not np.isfinite(prediction).all():
            raise ValueError("invalid scFoundation source-only frozen predictions")
        result = dict(
            fit,
            status="trained_validation_only",
            condition_seconds=time.monotonic() - t,
            test_conditions_accessed=0,
            test_updates=0,
            validation_prediction_shape=list(prediction.shape),
        )
    else:
        data = anndata.read_h5ad(source, backed="r")
        try:
            projection = data.var_names.get_indexer(panel["score_genes"])
        finally:
            data.file.close()
        if (projection < 0).any():
            raise ValueError("common score columns not present in the frozen C2 axis")
        # Input still uses the masked common panel. The scorer checks the original frozen pairings.
        with official_module_session(
            checkout_root=TXPERT_CHECKOUT,
            expected_commit=TXPERT_COMMIT,
            module_names=("gspp.metrics",),
        ) as (modules, _):
            measured = score(
                lambda name, controls: predict(name, controls[:, projection]),
                modules["gspp.metrics"],
                source,
                cross,
                cohort,
                rows,
                compatibility,
                output,
                journal_key=checkpoint_sha,
                score_genes=panel["score_genes"],
                prediction_on_score_panel=True,
                stop_starting_conditions_at=start + WALL_SECONDS,
            )
        if measured.get("conditions") != 128 or sha256_file(checkpoint) != checkpoint_sha:
            raise RuntimeError("full frozen scoring did not finish within its budget")
        result = dict(fit, **measured, status="evaluated")
    if list(output.rglob("*.pkl")):
        raise RuntimeError("scFoundation output unexpectedly contains PKLs")
    atomic_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--publication-sha", required=True)
    parser.add_argument("--gpu", required=True, type=int, choices=(0, 1))
    parser.add_argument("--gpu-uuid", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--smoke", action="store_true")
    choice.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != STUDY or any(
        os.getenv(k) != v
        for k, v in {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }.items()
    ):
        raise ValueError("scFoundation requires the study and admitted CUDA environment")
    source = inspect_source_identity(
        REPO,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/codex/perturbation-holdout-study",
        verification_method="github_rest_live",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha,
    )
    identity = dict(
        model="scfoundation",
        scenario="C2",
        run_id=output.name,
        source_commit=source.commit,
        official_commit=COMMIT,
        formal_eligible=source.formal_eligible,
        cohort_sha256=COHORT_SHA,
        shared_fit_support_file_sha256=SUPPORT_SHA,
        weights_sha256=FOUNDATION_WEIGHTS_SHA,
        score_panel_sha256=sha256_file(PANEL),
    )
    with lock(SERVER / f"runtime/v2-gpu-{args.gpu}.lock"):
        admit_selected_idle(snapshot(os.getpid()), args.gpu, args.gpu_uuid)
        output.mkdir()
        atomic_json(output / "worker.json", {"owner": process_info(os.getpid()), "gpu": args.gpu})
        run(output, identity, args)


if __name__ == "__main__":
    main()
