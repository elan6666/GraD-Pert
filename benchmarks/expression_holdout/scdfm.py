"""Truth-free scDFM adapter calling the frozen upstream model and update routines.

Only data/vocabulary/query selection, checkpoint IO and common-control inference
are adapted. No velocity model, flow path, MMD or ODE solver is implemented here.
The official training recipe is run.sh, not the differently valued CLI defaults.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

OFFICIAL_SHA = "8de47e7d3d443939f93f768f7902c66e0e82eda2"
OFFICIAL_URL = "https://github.com/AI4Science-WestlakeU/scDFM"
OFFICIAL_RECIPE = {
    "batch_size": 48,
    "steps": 200000,
    "lr": 5e-5,
    "eta_min": 1e-6,
    "mode": "predict_y",
    "noise_type": "Gaussian",
    "infer_top_gene": 1000,
    "use_mmd_loss": True,
    "gamma": 0.5,
    "topk": 30,
    "use_negative_edge": True,
}
_ROOTS = ("src", "config")


def _upstream_module(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _ROOTS)


@contextlib.contextmanager
def official_namespace(checkout: Path, cached: dict[str, Any] | None = None):
    """Temporarily install upstream's generic src/config namespaces, restoring ours.

    This context is process-local and non-thread-safe by design; each official
    adapter executes in its own runner process. No fake dependency modules or
    copied/extracted upstream functions are installed.
    """
    saved = {name: module for name, module in sys.modules.items() if _upstream_module(name)}
    old_path = sys.path[:]
    for name in saved:
        del sys.modules[name]
    if cached:
        sys.modules.update(cached)
    sys.path.insert(0, str(checkout))
    try:
        yield
    finally:
        if cached is not None:
            cached.update(
                {name: module for name, module in sys.modules.items() if _upstream_module(name)}
            )
        for name in list(sys.modules):
            if _upstream_module(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = old_path


def _verify_checkout(checkout: Path) -> dict[str, str]:
    sha = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if sha != OFFICIAL_SHA:
        raise ValueError(f"scDFM frozen version mismatch: {sha} != {OFFICIAL_SHA}")
    status = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise ValueError("scDFM official checkout must be clean")
    return {
        name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
        for name in (
            "run.sh",
            "src/script/run.py",
            "src/models/origin/model.py",
            "src/models/origin/layers.py",
            "src/utils/utils.py",
            "src/tokenizer/gene_tokenizer.py",
        )
    }


def load_official(checkout: Path) -> SimpleNamespace:
    """Import the full official run module, including its real dependencies."""
    hashes = _verify_checkout(checkout)
    modules: dict[str, Any] = {}
    with official_namespace(checkout, modules):
        run = importlib.import_module("src.script.run")
        origin = importlib.import_module("src.models.origin.model")
        utils = importlib.import_module("src.utils.utils")
        tokenizer = importlib.import_module("src.tokenizer.gene_tokenizer")
        for module in (run, origin, utils, tokenizer):
            if not Path(module.__file__).resolve().is_relative_to(checkout.resolve()):
                raise ValueError("scDFM module origin escaped frozen checkout")
    return SimpleNamespace(
        run=run,
        origin=origin,
        utils=utils,
        tokenizer=tokenizer,
        modules=modules,
        source_hashes=hashes,
    )


def _matrix_rows(matrix: Any, indices: Any) -> np.ndarray:
    values = matrix[indices]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def _targets(condition: str) -> tuple[str, ...]:
    return tuple(g for g in condition.split("+") if g not in ("ctrl", "control"))


def perturbation_tokens(vocab: Any, target_genes: tuple[str, ...]) -> list[int]:
    if not 1 <= len(target_genes) <= 2 or len(set(target_genes)) != len(target_genes):
        raise ValueError("scDFM CRISPR adapter requires one or two distinct targets")
    tokens = list(target_genes)
    # Official Norman single-target conditions retain their +control component.
    if len(tokens) == 1:
        tokens.append("control")
    if any(token not in vocab for token in tokens):
        raise ValueError("scDFM perturbation identity is missing from full vocabulary")
    return vocab.encode(tokens)


def _training_groups(prepared: Any) -> dict[str, dict[str, list[int]]]:
    """Keep every canonical train row; adapt official same-condition sampling to context."""
    rows = {str(row): i for i, row in enumerate(prepared.adata.obs_names)}
    groups: dict[str, dict[str, list[int]]] = {}
    allowed_conditions = set(prepared.train_conditions)
    for row_id in prepared.train_row_ids:
        index = rows[str(row_id)]
        condition = str(prepared.adata.obs.iloc[index]["condition"])
        if condition not in allowed_conditions or not _targets(condition):
            raise ValueError("scDFM train row has an unauthorized/control condition")
        context = str(prepared.row_context_ids[index])
        candidates = prepared.train_control_indices[str(row_id)]
        if len(candidates) == 0:
            raise ValueError("scDFM target has no compatible training control")
        for control in candidates:
            if str(prepared.row_context_ids[int(control)]) != context:
                raise ValueError("scDFM control/target context mismatch")
        groups.setdefault(condition, {}).setdefault(context, []).append(index)
    if not groups or set(groups) != allowed_conditions:
        raise ValueError("scDFM training condition coverage differs from canonical split")
    return groups


def sample_batch(prepared: Any, groups: dict, batch_size: int, rng: Any):
    conditions = tuple(groups)
    condition = conditions[int(rng.integers(len(conditions)))]
    contexts = tuple(groups[condition])
    sizes = np.asarray([len(groups[condition][ctx]) for ctx in contexts], dtype=float)
    context = contexts[int(rng.choice(len(contexts), p=sizes / sizes.sum()))]
    target_rows = rng.choice(groups[condition][context], size=batch_size, replace=True)
    control_rows = np.asarray(
        [
            rng.choice(prepared.train_control_indices[str(prepared.adata.obs_names[int(row)])])
            for row in target_rows
        ],
        dtype=int,
    )
    allowed = np.asarray(prepared.allowed_indices, dtype=int)
    source = _matrix_rows(prepared.adata.X, control_rows)[:, allowed]
    target = _matrix_rows(prepared.adata.X, target_rows)[:, allowed]
    return source, target, condition, control_rows, target_rows


def _safe_graph_matrix(prepared: Any, graph_genes: tuple[str, ...]) -> np.ndarray:
    names = tuple(str(g) for g in prepared.adata.var_names)
    rows = {str(row): i for i, row in enumerate(prepared.adata.obs_names)}
    # Graph statistics have no validation or test numeric access.
    train = [rows[str(row)] for row in prepared.train_row_ids]
    control = sorted({int(i) for values in prepared.train_control_indices.values() for i in values})
    values = _matrix_rows(prepared.adata.X, sorted(set(train) | set(control)))
    excluded = np.asarray(prepared.excluded_indices, dtype=int)
    if excluded.size and np.any(values[:, excluded] != 0):
        raise ValueError("scDFM fitter received unsanitized excluded expression")
    if not np.isfinite(values).all():
        raise ValueError("scDFM fitter received nonfinite expression")
    result = np.zeros((len(values), len(graph_genes)), dtype=np.float32)
    locations = {gene: i for i, gene in enumerate(graph_genes)}
    for i, gene in enumerate(names):
        if gene not in locations:
            raise ValueError("expression gene absent from graph vocabulary")
        result[:, locations[gene]] = values[:, i]
    return result


@dataclass
class ScDFMState:
    vf: Any
    official: Any
    checkout: Path
    vocab: Any
    gene_ids: tuple[str, ...]
    device: Any
    training: dict
    used_gene_indices: set[int] = field(default_factory=set)


@dataclass
class _NoiseTorch:
    """Delegate torch except the frozen generate_sample Gaussian allocation."""

    original: Any
    noise: Any
    device: Any

    def __getattr__(self, name):
        return getattr(self.original, name)

    def randn(self, *shape, **kwargs):
        if tuple(shape) != tuple(self.noise.shape) or kwargs != {"device": self.device}:
            raise ValueError("scDFM official noise allocation contract changed")
        return self.noise.clone()


def resolve_training_budget(prepared: Any, requested: dict) -> tuple[dict, dict]:
    """Resolve mutually exclusive step/epoch budgets without changing the sampler.

    An epoch is an equivalent sample budget: ceil(N_train_pert / batch_size)
    official condition/population-sampled updates, not a traversal of each row.
    The old explicit steps budget (and official default) remains supported.
    """
    if "epochs" in requested and "steps" in requested:
        raise ValueError("scDFM epochs and steps budgets are mutually exclusive")
    training = {**OFFICIAL_RECIPE, **requested}
    batch = training["batch_size"]
    if type(batch) is not int or batch <= 0:
        raise ValueError("scDFM batch_size must be a positive integer")
    n_train = len(prepared.train_row_ids)
    if n_train < 1 or len(set(prepared.train_row_ids)) != n_train:
        raise ValueError("scDFM requires distinct canonical train perturbation rows")
    steps_per_epoch = (n_train + batch - 1) // batch
    if "epochs" in requested:
        epochs = requested["epochs"]
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("scDFM epochs must be a positive integer")
        training.pop("steps")  # Do not silently keep the official 200k default.
        total = epochs * steps_per_epoch
        mode = "project_epoch_equivalent_sample_budget"
        deviation = "official 200000-update budget replaced by requested epoch-equivalent budget"
    else:
        total = training["steps"]
        if type(total) is not int or total <= 0:
            raise ValueError("scDFM steps must be a positive integer")
        epochs = None
        mode = "steps"
        deviation = (
            None
            if total == OFFICIAL_RECIPE["steps"]
            else "explicit project update budget differs from official 200000 updates"
        )
    return training, {
        "budget_mode": mode,
        "n_train_perturbed_rows": n_train,
        "training_batch_size": batch,
        "steps_per_epoch": steps_per_epoch,
        "epochs_requested": epochs,
        "total_optimizer_updates": total,
        "official_optimizer_updates": OFFICIAL_RECIPE["steps"],
        "budget_formula": (
            "s=ceil(N_train_pert/batch_size); total=epochs*s"
            if epochs is not None
            else "total=steps; s=ceil(N_train_pert/batch_size)"
        ),
        "epoch_semantics": "equivalent sampled-cell budget; not exactly one traversal per cell",
        "budget_deviation": deviation,
        "cosine_t_max": total,
    }


def budget_progress(budget: dict, completed_updates: int) -> dict:
    """Report actual progress; a ten-update preflight cannot claim full epochs."""
    if not 0 <= completed_updates <= budget["total_optimizer_updates"]:
        raise ValueError("scDFM completed updates exceed the resolved budget")
    per_epoch = budget["steps_per_epoch"]
    return {
        **budget,
        "completed_updates": completed_updates,
        "epochs_completed": completed_updates // per_epoch,
        "epoch_equivalent_progress": completed_updates / per_epoch,
        "sampled_target_cell_draws": completed_updates * budget["training_batch_size"],
        "training_complete": completed_updates == budget["total_optimizer_updates"],
    }


def _require_accelerator_device(accelerator: Any, requested: Any):
    """Compare physical CUDA indices, retaining the isolated single-device guard."""
    import torch

    if accelerator.num_processes != 1:
        raise ValueError("scDFM external runner is single-device; do not launch with torchrun")
    requested = torch.device(requested)
    actual = torch.device(accelerator.device)
    if requested.type != actual.type:
        raise ValueError(
            "scDFM Accelerator device differs from requested isolated device: "
            f"{actual} != {requested}"
        )
    if actual.type == "cuda":
        # Accelerate can report 'cuda' for a single-process job. Its device is
        # the current visible CUDA index, not necessarily index zero.
        current = torch.cuda.current_device() if None in (actual.index, requested.index) else None
        actual = torch.device("cuda", current if actual.index is None else actual.index)
        requested = torch.device("cuda", current if requested.index is None else requested.index)
    if actual != requested:
        raise ValueError(
            "scDFM Accelerator device differs from requested isolated device: "
            f"{actual} != {requested}"
        )
    return requested


def fit(prepared: Any, config: dict, runtime: dict, output: Path, max_updates: int | None = None):
    """Use upstream train_step and exact Adam -> cosine update ordering."""
    import torch
    from accelerate import Accelerator

    from .contract import FittedAdapter

    training, budget = resolve_training_budget(prepared, config["training"])
    model_config = config["model"]
    if model_config.get("model_type", "origin") != "origin":
        raise ValueError("scDFM adapter frozen to official origin recipe")
    if model_config.get("fusion_method", "differential_perceiver") != "differential_perceiver":
        raise ValueError("scDFM adapter frozen to official differential_perceiver recipe")
    if training["noise_type"] != "Gaussian" or training["mode"] != "predict_y":
        raise ValueError("scDFM adapter frozen to official Gaussian predict_y recipe")
    steps = budget["total_optimizer_updates"]
    if max_updates is not None and (type(max_updates) is not int or max_updates <= 0):
        raise ValueError("scDFM diagnostic update count must be a positive integer")
    count = steps if max_updates is None else min(max_updates, steps)
    query_count = min(int(training["infer_top_gene"]), len(prepared.allowed_indices))
    if query_count < 1:
        raise ValueError("scDFM has no allowed expression query genes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(runtime["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(runtime["device"])
    checkout = Path(runtime["official_checkout"]).resolve()
    official = load_official(checkout)
    names = tuple(str(g) for g in prepared.adata.var_names)
    graph_names = tuple(str(g) for g in prepared.graph_gene_ids)
    if len(set(graph_names)) != len(graph_names):
        raise ValueError("scDFM graph vocabulary contains duplicate identities")
    groups = _training_groups(prepared)
    with official_namespace(checkout, official.modules):
        vocab = official.tokenizer.GeneVocab(
            list(graph_names), specials=["<pad>", "<cls>", "<mask>", "control"]
        )
        safe = _safe_graph_matrix(prepared, graph_names)
        mask = official.utils.build_gene_coexpression_graph(
            safe,
            method="pearson",
            wgcna_beta=None,
            sparsify="topk",
            k=int(training["topk"]),
            use_negative_edge=bool(training["use_negative_edge"]),
        )
        mask = official.utils.sorted_pad_mask(mask, pad_size=4, gene_names=list(graph_names))
        mask_path = output / "coexpression-mask.pt"
        torch.save(mask, mask_path)
        # instantiate_model ignores d_model and ntoken. Invoke its actual official
        # OriginModel constructor directly, changing only task vocabulary capacity.
        vf = official.origin.model(
            ntoken=len(vocab),
            nlayers=4,
            fusion_method="differential_perceiver",
            perturbation_function="crisper",
            mask_path=str(mask_path),
        ).to(device)
        if vf.encoder.embedding.embedding_dim != 512 or len(vf.blocks) != 4:
            raise ValueError("scDFM real official architecture changed")
        optimizer = torch.optim.Adam(vf.parameters(), lr=float(training["lr"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=float(training["eta_min"])
        )
        accelerator = Accelerator(cpu=device.type == "cpu")
        device = _require_accelerator_device(accelerator, device)
        vf, optimizer, scheduler = accelerator.prepare(vf, optimizer, scheduler)
        official.run.config = SimpleNamespace(**{**training, "infer_top_gene": query_count})
        official.run.device = device
        allowed = np.asarray(prepared.allowed_indices, dtype=int)
        official.run.gene_ids = torch.tensor(
            vocab.encode([names[i] for i in allowed]), dtype=torch.long, device=device
        )
        criterion = torch.nn.MSELoss()
        vf.train()
        sampled_rows: set[int] = set()
        sampled_gene_indices: set[int] = set()
        token_to_expression = {int(vocab.encode(gene)): i for i, gene in enumerate(names)}

        def observe_queries(_module, args):
            sampled_gene_indices.update(
                token_to_expression[int(token)] for token in args[0][0].detach().cpu().tolist()
            )

        query_hook = accelerator.unwrap_model(vf).register_forward_pre_hook(observe_queries)
        losses = []
        for step in range(count):
            source, target, condition, _, target_rows = sample_batch(
                prepared, groups, int(training["batch_size"]), rng
            )
            sampled_rows.update(int(row) for row in target_rows)
            perturbation = torch.tensor(
                perturbation_tokens(vocab, _targets(condition)), dtype=torch.long, device=device
            ).repeat(len(source), 1)
            official.utils.set_requires_grad_for_p_only(vf, p_only=training["mode"])
            loss = official.run.train_step(
                torch.tensor(source, device=device),
                torch.tensor(target, device=device),
                perturbation,
                vf,
                criterion,
                accelerator,
                noise_type=training["noise_type"],
                mode=training["mode"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"scDFM nonfinite loss at update {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step == 0 or (step + 1) % int(training.get("log_every", 100)) == 0:
                print(
                    json.dumps(
                        {
                            "adapter": "scdfm",
                            "update": step + 1,
                            "total_updates": count,
                            "planned_updates": steps,
                            "epoch_equivalent_progress": (step + 1) / budget["steps_per_epoch"],
                            "loss": losses[-1],
                        }
                    ),
                    flush=True,
                )
        vf = accelerator.unwrap_model(vf)
        query_hook.remove()
        official.utils.save_checkpoint(
            model=vf,
            optimizer=optimizer,
            scheduler=scheduler,
            iteration=count,
            eval_score=None,
            save_path=str(output),
            is_best=False,
        )
        (output / "checkpoint.pt").rename(output / "last.pt")
    state = ScDFMState(vf, official, checkout, vocab, names, device, training, sampled_gene_indices)
    receipt = {
        "official_repository": OFFICIAL_URL,
        "official_sha": OFFICIAL_SHA,
        "official_source_hashes": official.source_hashes,
        "training_routine": "src.script.run.train_step",
        "inference_routine": "src.script.run.generate_sample",
        "official_recipe": OFFICIAL_RECIPE,
        "resolved_training": training,
        "actual_model": {"d_model": 512, "nhead": 8, "nlayers": 4, "ntoken": len(vocab)},
        **budget_progress(budget, count),
        "last_loss": losses[-1],
        "sampled_target_rows": len(sampled_rows),
        "allowed_expression_gene_indices": allowed.tolist(),
        "observed_training_expression_gene_indices": sorted(sampled_gene_indices),
        "training_query_count": query_count,
        "checkpoint": str(output / "last.pt"),
        "adaptations": [
            "canonical data/split replaces upstream dataset loading and split/HVG recomputation",
            "full graph vocabulary capacity only; actual official512/8heads/4blocks preserved",
            "same-condition replacement sampling restricted to compatible control/target context",
            "train_step/flow/noise/MMD use allowed expression columns only",
            "coexpression built only from sanitized train targets and compatible train controls",
            "official periodic cell_eval validation omitted; common frozen last-only evaluation",
            "full expression inference uses fixed query blocks and exact supplied controls",
            "epoch budget uses ceil(train perturbation rows / official batch) sampled updates",
        ],
    }
    return FittedAdapter(model=state, receipt=receipt)


def predict(
    fitted: Any,
    controls: np.ndarray,
    target_genes: tuple[str, ...],
    gene_ids: tuple[str, ...],
    *,
    cell_batch: int,
    query_count: int,
    seed: int,
) -> np.ndarray:
    """ODE-generate every requested gene; never fill missing outputs with zeros/control."""
    import torch

    state = fitted.model
    if not isinstance(state, ScDFMState) or tuple(gene_ids) != state.gene_ids:
        raise ValueError("scDFM inference gene axis/model identity mismatch")
    values = np.asarray(controls, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(gene_ids) or not np.isfinite(values).all():
        raise ValueError("scDFM controls must be a finite cells-by-full-gene matrix")
    if cell_batch < 1 or query_count < 1:
        raise ValueError("scDFM inference batch/query sizes must be positive")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    device = state.device
    with official_namespace(state.checkout, state.official.modules):
        state.vf.eval()
        token_ids = state.vocab.encode(list(gene_ids))
        condition = torch.tensor(
            perturbation_tokens(state.vocab, target_genes), dtype=torch.long, device=device
        )
        # Preserve run.py's gene_input == gene_all policy (the model currently
        # ignores its gene_id_all argument). Each query block gets matching source.
        for start in range(0, len(gene_ids), query_count):
            stop = min(start + query_count, len(gene_ids))
            state.official.run.config = SimpleNamespace(
                **{**state.training, "infer_top_gene": stop - start}
            )
            state.official.run.device = device
            block_ids = torch.tensor(token_ids[start:stop], dtype=torch.long, device=device)
            noise_seed = int.from_bytes(
                hashlib.sha256(
                    json.dumps(
                        [seed, list(target_genes), list(gene_ids[start:stop])],
                        separators=(",", ":"),
                    ).encode()
                ).digest()[:8],
                "little",
            ) % (2**63 - 1)
            generator = torch.Generator(device=device).manual_seed(noise_seed)
            noise = torch.randn(len(values), stop - start, device=device, generator=generator)
            noise_sha = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            fitted.receipt.setdefault("inference_noise", []).append(
                {
                    "targets": list(target_genes),
                    "gene_start": start,
                    "gene_stop": stop,
                    "seed": noise_seed,
                    "sha256": noise_sha,
                    "rows": len(values),
                }
            )
            for row in range(0, len(values), cell_batch):
                source = torch.tensor(values[row : row + cell_batch, start:stop], device=device)
                # Only this upstream module's torch binding is scoped. Its actual
                # Gaussian-allocation call receives the exact control-row noise;
                # the global torch module and ODE/model math remain untouched.
                original_torch = state.official.run.torch

                state.official.run.torch = _NoiseTorch(
                    original_torch, noise[row : row + len(source)], device
                )
                try:
                    generated = state.official.run.generate_sample(
                        state.official.run.wrapped_vf,
                        source,
                        condition_vec=condition.repeat(len(source), 1),
                        vf=state.vf,
                        gene_ids=block_ids,
                        gene_all=block_ids,
                        steps=20,
                        method="rk4",
                    )
                finally:
                    state.official.run.torch = original_torch
                result[row : row + len(source), start:stop] = generated.cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("scDFM inference did not produce every requested gene")
    return result
