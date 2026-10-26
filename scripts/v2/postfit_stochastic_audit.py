"""Zero-update Cell/Response stochasticity audit with frozen native eval graphs."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from postfit_spectral_audit import (
    checked,
    digest,
    frozen_observation,
    load_panel,
    observed_cls,
    physical_data,
    read,
    sha,
    spectral_statistics,
    verify_source,
    write_new,
)

SCHEMA = "postfit-stochastic-audit-input-1"
NAMES = ("RI", "R13SRAMP", "NC")
TRAINING_COMMITS = dict(
    RI="f0cb20ebb2efab749970f2bf5d205ed2f358592c",
    R13SRAMP="dc79777827fb763e7b37b709d0149bbc1e9e4a9e",
    NC="2a2fbceb84519a458a4099ee7489a4ed38c5449c",
)
PROTOCOL = dict(
    scope="fixed_native_eval_graph_cell_response_v1",
    repeat_seeds=[2026102601, 2026102602, 2026102603, 2026102604],
    views=["main", "global0", "global1"],
    factors=[
        dict(dropout=False, random_order=False),
        dict(dropout=True, random_order=False),
        dict(dropout=False, random_order=True),
        dict(dropout=True, random_order=True),
    ],
    rng="per_encode_sha256_base_seed_group_index_view_no_endpoint_v1",
    teacher="unobserved",
    corrected_rank="not_computed",
)


def validate_spec(spec, *, server=True):
    fields = {
        "schema",
        "analysis_source",
        "parent",
        "panel",
        "resource_queue",
        "data_root",
        "output",
        "endpoints",
        "protocol",
    }
    if (
        set(spec) != fields
        or spec["schema"] != SCHEMA
        or digest(spec["protocol"]) != digest(PROTOCOL)
    ):
        raise ValueError("stochastic audit schema or frozen protocol differs")
    if [e.get("name") for e in spec["endpoints"]] != list(NAMES):
        raise ValueError("exact ordered RI/R13SRAMP/NC endpoints required")
    protected = [Path(spec["data_root"]).resolve()]
    paths = [Path(spec["output"]).resolve(), *protected]
    for key in ("parent", "panel", "resource_queue"):
        paths.append(checked(spec[key]))
    for source in [spec["analysis_source"], *(e["source"] for e in spec["endpoints"])]:
        if set(source) != {"root", "commit", "publication", "publication_sha256"}:
            raise ValueError("complete source/publication binding required")
        if len(source["commit"]) != 40 or any(
            c not in "0123456789abcdef" for c in source["commit"]
        ):
            raise ValueError("full source SHA required")
        root = Path(source["root"]).resolve(strict=True)
        protected.append(root)
        paths.extend(
            (root, checked(dict(path=source["publication"], sha256=source["publication_sha256"])))
        )
    for endpoint in spec["endpoints"]:
        if set(endpoint) != {
            "name",
            "source",
            "run_root",
            "run_manifest_sha256",
            "config",
            "config_sha256",
            "checkpoint_sha256",
        }:
            raise ValueError("complete original endpoint binding required")
        if endpoint["source"]["commit"] != TRAINING_COMMITS[endpoint["name"]]:
            raise ValueError("endpoint must use its retained original training source")
        root = Path(endpoint["run_root"]).resolve(strict=True)
        config = checked(dict(path=endpoint["config"], sha256=endpoint["config_sha256"]))
        if not config.is_relative_to(Path(endpoint["source"]["root"]).resolve() / "configs"):
            raise ValueError("original config must remain inside original source")
        checked(dict(path=root / "run_manifest.json", sha256=endpoint["run_manifest_sha256"]))
        protected.append(root)
        paths.extend((root, config))
    output = Path(spec["output"]).resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("fresh output must not overlap source/data/training roots")
    if server and any(not p.is_relative_to("/data/yilangliu") for p in paths):
        raise ValueError("scientific materialization is server-only")


def fingerprint(value):
    """Materialize tensor strides before the unchanged native byte fingerprint."""
    import torch
    from capacity_probe import state_fingerprint

    def concrete(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().clone(memory_format=torch.contiguous_format)
        if dataclasses.is_dataclass(item):
            return {f.name: concrete(getattr(item, f.name)) for f in dataclasses.fields(item)}
        if isinstance(item, dict):
            return {k: concrete(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return type(item)(concrete(v) for v in item)
        return item

    return state_fingerprint(concrete(value))


def native_compatibility(source, analysis):
    methods = {
        "model.py": {
            "GraDPertV2": (
                "encode_control",
                "encode_response",
                "train",
                "set_relay_order_randomization",
            ),
            "RelayResponseEncoder": ("forward",),
        },
        "operators.py": {
            None: ("relay_order",),
            "RelayDeltaAttention": ("forward", "cross", "graph", "random_order_enabled"),
            "LatentAttention": ("forward",),
            "LatentCrossAttention": ("forward",),
            "TokenEncoder": ("forward",),
        },
    }

    def signatures(root):
        result = {}
        for file, classes in methods.items():
            path = Path(root) / "src/gradpert/modeling/v2" / file
            tree = ast.parse(path.read_text())
            for name, functions in classes.items():
                container = (
                    tree
                    if name is None
                    else next(
                        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name
                    )
                )
                for function in functions:
                    node = next(
                        n
                        for n in container.body
                        if isinstance(n, ast.FunctionDef) and n.name == function
                    )
                    result[f"{file}:{name or 'module'}.{function}"] = digest(
                        ast.dump(node, include_attributes=False)
                    )
        return result

    original, current = signatures(source), signatures(analysis)
    if original != current:
        raise ValueError("native encoder/order methods differ from the approved analysis API")
    return dict(method_ast_sha256=original, unchanged=True)


def inventory(model):
    import torch

    options = model.options
    expected = dict(
        attention="relay_full",
        control_conditioned_graph=False,
        fixed_gene_order=False,
        random_gene_order=True,
        attention_replacement="none",
    )
    if any(getattr(options, k) != v for k, v in expected.items()) or options.dropout != 0.1:
        raise ValueError("endpoint architecture is outside the scoped Cell/Response factorial")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError("stateful BatchNorm is outside this frozen observation protocol")
    owners = []
    for name, module in model.named_modules():
        scoped = name.startswith(("cell.", "response."))
        probability = (
            module.p if isinstance(module, torch.nn.Dropout) else getattr(module, "dropout", None)
        )
        if isinstance(probability, (float, int)) and probability > 0:
            if not isinstance(module, torch.nn.Dropout) and type(module).__name__ not in {
                "LatentAttention",
                "LatentCrossAttention",
                "IndexedLatentAttention",
                "ReplacementAttention",
            }:
                raise ValueError(
                    f"unreviewed functional dropout owner: {name}:{type(module).__name__}"
                )
            owners.append(
                dict(
                    name=name,
                    module=type(module).__name__,
                    probability=probability,
                    scoped=scoped,
                    training=module.training,
                    implementation="nn.Dropout"
                    if isinstance(module, torch.nn.Dropout)
                    else "functional_or_SDPA",
                )
            )
    if not any(r["scoped"] for r in owners):
        raise ValueError("factor D has no active Cell/Response dropout owners")
    return dict(
        options={
            **expected,
            "dropout": options.dropout,
            "ffn_type": options.ffn_type,
            "kda_layers": options.kda_layers,
            "relay_eval_seed": options.relay_eval_seed,
            "width": options.width,
        },
        dropout_owners=owners,
    )


def set_factor(model, factor):
    from gradpert.modeling.v2.operators import RelayDeltaAttention

    model.cell.train(factor["dropout"])
    model.response.train(factor["dropout"])
    model.randomize_relay_order = factor["random_order"]
    for branch in (model.cell, model.response):
        for module in branch.modules():
            if isinstance(module, RelayDeltaAttention):
                module.randomize_order = factor["random_order"]
    model.graph.eval()
    for module in model.graph.modules():
        if isinstance(module, RelayDeltaAttention):
            module.randomize_order = False
    result = inventory(model)
    if any(
        r["training"] != (factor["dropout"] if r["scoped"] else False)
        for r in result["dropout_owners"]
    ):
        raise ValueError("actual dropout owner flags differ from the factorial setting")
    return result


class OrderObserver:
    """Observe native generated orders without modifying attention or RNG."""

    def __init__(self, model):
        self.values = {}
        self.handles = [
            model.cell.register_forward_pre_hook(self.cell, with_kwargs=True),
            model.response.register_forward_pre_hook(self.response, with_kwargs=True),
            model.graph.register_forward_pre_hook(self.forbid_graph),
        ]

    @staticmethod
    def forbid_graph(*_):
        raise ValueError("graph forward attempted after native eval cache was frozen")

    def cell(self, _module, _args, kwargs):
        if "cell" in self.values or kwargs.get("order") is None:
            raise ValueError("one native Cell order tuple per encode required")
        self.values["cell"] = fingerprint(kwargs["order"])

    def response(self, _module, args, kwargs):
        order = kwargs.get("order", args[3] if len(args) > 3 else None)
        if "response" in self.values or order is None:
            raise ValueError("one native Response order tuple per encode required")
        self.values["response"] = fingerprint(order)

    def receipt(self):
        if (
            set(self.values) != {"cell", "response"}
            or self.values["cell"] != self.values["response"]
        ):
            raise ValueError("native Response did not reuse the observed Cell order")
        return self.values.copy()

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def order_observer(model):
    observer = OrderObserver(model)
    try:
        yield observer
    finally:
        observer.close()


def encode_seed(base_seed, group, view):
    return int(digest([base_seed, group, view])[:16], 16) % (2**63 - 1)


def seed_encode(seed, device):
    """Seed CPU and this device only; torch.manual_seed would seed every GPU."""
    import torch

    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def zero_pkl(output):
    found = [
        str(p.relative_to(output))
        for p in Path(output).rglob("*")
        if p.suffix.lower() in (".pkl", ".pickle")
    ]
    if found:
        raise ValueError(f"zero-PKL postcondition failed: {found}")
    return dict(checked=True, count=0)


@contextmanager
def control_only_reads(runtime, panel):
    """Guard actual assembly reads against the sealed panel control whitelist."""
    import numpy as np

    data = runtime.data
    allowed = {data._row_index[r["control_row_id"]] for g in panel["schedule"] for r in g["rows"]}
    name = "_read_expression_indices"
    owned, original = name in vars(data), getattr(data, name)
    own_value = vars(data).get(name)
    proof = dict(calls=0, rows=0, forbidden=0, allowed_indices_sha256=digest(sorted(allowed)))

    def guarded(indices):
        indices = list(indices)
        if any(
            isinstance(i, bool) or not isinstance(i, (int, np.integer)) or int(i) not in allowed
            for i in indices
        ):
            proof["forbidden"] += 1
            raise ValueError("expression read outside sealed control-only panel whitelist")
        proof["calls"] += 1
        proof["rows"] += len(indices)
        return original(indices)

    setattr(data, name, guarded)
    try:
        yield proof
    finally:
        if owned:
            setattr(data, name, own_value)
        else:
            delattr(data, name)


@contextmanager
def forbidden_paths(runtime):
    """Fail closed on forbidden modules and process-local update entry points."""
    import torch

    counts = dict.fromkeys(
        (
            "student_encode",
            "native_graph",
            "objective",
            "teacher",
            "projector",
            "decoder",
            "backward",
            "autograd_grad",
            "optimizer_step",
            "statistics_commit",
        ),
        0,
    )
    handles, patched = [], []

    def hook(kind, forbidden=True):
        def invoke(*_args, **_kwargs):
            counts[kind] += 1
            if forbidden:
                raise ValueError(f"forbidden audit execution path: {kind}")

        return invoke

    def patch(owner, name, kind):
        if not hasattr(owner, name):
            raise ValueError(f"required native forbidden entry point is absent: {name}")
        owned = name in vars(owner)
        old = vars(owner).get(name) if owned else None
        patched.append((owner, name, owned, old))
        setattr(owner, name, hook(kind))

    obj = runtime.objective
    try:
        handles.append(obj.register_forward_pre_hook(hook("objective")))
        handles.append(
            obj.student.response.register_forward_pre_hook(hook("student_encode", False))
        )
        handles.append(obj.student.graph.register_forward_pre_hook(hook("native_graph", False)))
        for module in obj.teacher.modules():
            handles.append(module.register_forward_pre_hook(hook("teacher")))
        for name in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node", "prediction"):
            for module in getattr(obj.student, name).modules():
                handles.append(
                    module.register_forward_pre_hook(
                        hook("decoder" if name == "prediction" else "projector")
                    )
                )
        patch(runtime.optimizer, "step", "optimizer_step")
        patch(torch.autograd, "backward", "backward")
        patch(torch.autograd, "grad", "autograd_grad")
        patch(obj, "commit_statistics", "statistics_commit")
        yield counts
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, owned, old in reversed(patched):
            if owned:
                setattr(owner, name, old)
            else:
                delattr(owner, name)


def native_graph_reference(objective, model, batch):
    import torch

    with torch.random.fork_rng(
        devices=[batch.control.device.index or 0] if batch.control.is_cuda else []
    ):
        before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if batch.control.is_cuda else []
        cached = objective._batch_graph(model, batch)
        after = fingerprint(
            (torch.get_rng_state(), torch.cuda.get_rng_state_all() if cuda_before else [])
        )
        torch.set_rng_state(before)
        if cuda_before:
            torch.cuda.set_rng_state_all(cuda_before)
        reference = objective._batch_graph(model, batch)
        reference_after = fingerprint(
            (torch.get_rng_state(), torch.cuda.get_rng_state_all() if cuda_before else [])
        )
    maximum = max(float((a - b).abs().max()) for a, b in zip(cached, reference, strict=True))
    if not math.isfinite(maximum) or maximum > 2e-6 or after != reference_after:
        raise ValueError("independent native graph/condition/RNG reference parity failed")
    return cached, reference, maximum


def reference_parity(model, args, kwargs, *, reference_args=None, native_graph_max_abs=None):
    import torch

    cached_args = tuple(v.detach().clone() for v in args)
    cached_kwargs = {k: v.detach().clone() for k, v in kwargs.items()}
    with torch.random.fork_rng(devices=[args[1].device.index or 0] if args[1].is_cuda else []):
        before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if args[1].is_cuda else []
        original = model.encode_response(
            *(args if reference_args is None else reference_args), **kwargs, decode=False
        )
        after = fingerprint(
            (torch.get_rng_state(), torch.cuda.get_rng_state_all() if cuda_before else [])
        )
        torch.set_rng_state(before)
        if cuda_before:
            torch.cuda.set_rng_state_all(cuda_before)
        replay = model.encode_response(*cached_args, **cached_kwargs, decode=False)
        replay_after = fingerprint(
            (torch.get_rng_state(), torch.cuda.get_rng_state_all() if cuda_before else [])
        )
    maximum = max(float((original[k] - replay[k]).abs().max()) for k in original)
    graph_diff = max(
        float((a - b).abs().max())
        for a, b in zip(
            (args if reference_args is None else reference_args)[:3], cached_args[:3], strict=True
        )
    )
    graph_diff = max(graph_diff, native_graph_max_abs or 0.0)
    if (
        not math.isfinite(maximum)
        or not math.isfinite(graph_diff)
        or maximum > 2e-6
        or graph_diff > 2e-6
        or after != replay_after
    ):
        raise ValueError("native-reference graph/response/RNG parity failed")
    return (cached_args, cached_kwargs), dict(
        graph_max_abs=graph_diff,
        response_max_abs=maximum,
        final_rng_equal=True,
        tolerance=2e-6,
        cell_chunk_used=False,
        used_path_passed=True,
        native_graph_reference=reference_args is not None and native_graph_max_abs is not None,
    )


def prepare_inputs(runtime, panel, admission):
    from representation_panel import MatchedCellViews, assemble_control_only

    from gradpert.training.v2.runtime import _move_batch_tree

    obj, model = runtime.objective, runtime.objective.student
    matched, inputs, recipes, parities = MatchedCellViews(), [], [], []
    for number, group in enumerate(panel["schedule"]):
        admission()
        shallow = assemble_control_only(
            runtime, group["rows"], group["queries"], group["view_seed"], deep=False
        )
        deep = assemble_control_only(
            runtime, group["rows"], group["queries"], group["view_seed"], deep=True
        )
        deep, recipe = matched.bind(deep, group["rows"])
        original = dict(shallow=fingerprint(shallow), deep=fingerprint(deep))
        batches = {
            k: _move_batch_tree(v, runtime.device)
            for k, v in (("shallow", shallow), ("deep", deep))
        }
        cached, parity = {}, {}
        for key, batch in batches.items():
            if batch.graph.ids[batch.query_positions].tolist() != group["queries"]:
                raise ValueError("native assembled query order differs")
            (graph, condition), (ref_graph, ref_condition), graph_maximum = native_graph_reference(
                obj, model, batch
            )
            views = (
                [("main", None)]
                if key == "shallow"
                else [("global0", batch.cell_views[0]), ("global1", batch.cell_views[1])]
            )
            for view_name, view in views:
                p = batch.query_positions if view is None else batch.query_positions[view.positions]
                control = batch.control if view is None else batch.control[:, view.positions]
                args = (obj._select_graph(graph, p), control, condition)
                ref_args = (obj._select_graph(ref_graph, p), control, ref_condition)
                if view is not None:
                    args += (view.mask,)
                    ref_args += (view.mask,)
                metadata = obj.response_metadata(batch, None if view is None else view.positions)
                cached[view_name], parity[view_name] = reference_parity(
                    model,
                    args,
                    metadata,
                    reference_args=ref_args,
                    native_graph_max_abs=graph_maximum,
                )
        if original != dict(shallow=fingerprint(shallow), deep=fingerprint(deep)):
            raise ValueError("native input changed during cache construction")
        inputs.append(cached)
        recipes.append(
            dict(
                group=number,
                rows=group["rows"],
                condition_id=group["condition_id"],
                native_inputs=original,
                **recipe,
            )
        )
        parities.append(parity)
    return inputs, recipes, parities


def repeat_statistics(values, *, expected_rows):
    import torch

    if len(values) != 4:
        raise ValueError("exactly four whole-population repeats required")
    z = torch.stack(values).detach().to(device="cpu", dtype=torch.float64)
    if (
        z.ndim != 3
        or z.shape[1] != expected_rows
        or min(z.shape) < 1
        or not torch.isfinite(z).all()
    ):
        raise ValueError("same finite N-row population required; never pool512")
    k, n, _ = z.shape
    mean = z.mean(0)
    residual = z - mean
    centered_residual = residual - residual.mean(1, keepdim=True)
    between = (mean - mean.mean(0)).square().sum().item() / n
    wraw = residual.square().sum().item() / (n * (k - 1))
    wcenter = centered_residual.square().sum().item() / (n * (k - 1))
    bcorr = between - wcenter / k
    ecorr = mean.square().sum().item() / n - wraw / k
    raw = [spectral_statistics(row) for row in z]
    total_centered = sum(row["trace"] for row in raw) / k
    total_energy = sum(row["total_energy"] for row in raw) / k
    errors = dict(centered=total_centered - bcorr - wcenter, total=total_energy - ecorr - wraw)
    if max(abs(v) for v in errors.values()) > 128 * torch.finfo(torch.float64).eps * max(
        total_energy, 1e-300
    ):
        raise ValueError("whole-replicate energy decomposition failed")
    reason = (
        "nonpositive_corrected_total_energy"
        if ecorr <= 0
        else "negative_corrected_between"
        if bcorr < 0
        else "corrected_between_exceeds_total"
        if bcorr > ecorr
        else None
    )
    return dict(
        repeats=k,
        rows=n,
        raw=raw,
        repeat_sha256=[fingerprint(row) for row in z],
        repeat_mean=spectral_statistics(mean),
        B=between,
        Wraw=wraw,
        Wcenter=wcenter,
        Bcorr=bcorr,
        Ecorr=ecorr,
        q_corr=bcorr / ecorr if reason is None else None,
        q_corr_unavailable_reason=reason,
        corrected_rank=None,
        corrected_rank_status="NA_not_computed_nonPSD_risk",
        decomposition_residuals=errors,
        inference="K4 finite signed energy estimates; no biological replication or CI",
    )


def crossed_statistics(values, panel):
    import torch

    rows = [r for group in panel["schedule"] for r in group["rows"]]
    z = torch.stack(values).double()
    pairs = {"different_p_same_control": [], "same_p_different_control": []}
    for i, a in enumerate(rows):
        for j, b in enumerate(rows[:i]):
            if any(a[key] != b[key] for key in ("cell_line", "cell_type", "context_id", "batch")):
                continue
            if (
                a["condition_id"] != b["condition_id"]
                and a["control_row_id"] == b["control_row_id"]
            ):
                pairs["different_p_same_control"].append((i, j))
            if (
                a["condition_id"] == b["condition_id"]
                and a["control_row_id"] != b["control_row_id"]
            ):
                pairs["same_p_different_control"].append((i, j))
    result = {}
    for kind, indices in pairs.items():
        if not indices:
            result[kind] = dict(pairs=0, status="NA_no_eligible_pairs")
            continue
        delta = torch.stack([z[:, i] - z[:, j] for i, j in indices], dim=1)
        mean = delta.mean(0)
        w = (delta - mean).square().sum().item() / (len(indices) * 3)
        b = mean.square().sum().item() / len(indices)
        result[kind] = dict(
            pairs=len(indices),
            repeat_mean_squared_distance=b,
            repeat_noise=w,
            corrected_squared_distance=b - w / 4,
            status="signed_K4_estimate_not_independent_biological_pairs",
        )
    return result


def endpoint_worker(spec, endpoint):
    import torch
    from cap40_posthoc import runtime_args, write
    from checkpoint_endpoint import select_checkpoint
    from representation_panel import validate_panel_runtime
    from shared_gpu import admit, snapshot

    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint

    source, analysis = verify_source(endpoint["source"]), verify_source(spec["analysis_source"])
    compatibility = native_compatibility(
        endpoint["source"]["root"], spec["analysis_source"]["root"]
    )
    _, config, identity, prepare, root = runtime_args(
        SimpleNamespace(
            method_sha=endpoint["source"]["commit"],
            config=Path(endpoint["config"]),
            run_root=Path(endpoint["run_root"]),
        )
    )
    if root != Path(endpoint["source"]["root"]).resolve() or identity["source"] != source:
        raise ValueError("original training imports/source identity differ")
    cp, cp_endpoint = select_checkpoint(Path(endpoint["run_root"]), "last")
    checkpoint = Path(endpoint["run_root"]) / "fit" / cp["file"]
    if cp["sha256"] != endpoint["checkpoint_sha256"]:
        raise ValueError("last checkpoint differs from sealed endpoint")
    panel = load_panel(spec)
    output = Path(spec["output"]) / endpoint["name"]
    output.mkdir()
    owner = read(Path(spec["output"]) / "OWNER.json")
    policy = read(checked(spec["resource_queue"]))["shared_gpu_policy"]

    def admission():
        observed = snapshot(owner["pid"])
        admit(policy, observed)
        return observed

    observed = admission()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.cuda.current_stream(0) != torch.cuda.default_stream(0):
        raise ValueError("native default stream0 required")
    write_new(output / "ADMISSION.json", dict(policy=policy, observed=observed, gpu=0))
    start = time.perf_counter()
    banks, order_rows, factor_states, modes, timing = {}, [], {}, {}, []
    with prepare(
        config,
        data_root=Path(spec["data_root"]),
        run_seed=1,
        device=torch.device("cuda:0"),
        purpose="evaluation",
    ) as rt:
        canonical_name = "RI" if endpoint["name"] == "RI" else "R13SRAMP"
        if rt.identity != identity["data"] or physical_data(
            rt.identity, canonical_name
        ) != physical_data(read(checked(spec["parent"]))["data"], "RI"):
            raise ValueError("physical data or original native runtime identity differs")
        validate_panel_runtime(panel, rt)
        load_evaluation_checkpoint(
            checkpoint, rt.objective, training_identity=identity, checkpoint_sha256=cp["sha256"]
        )
        if getattr(rt.objective, "training_stage", "joint") != cp_endpoint["training_stage"]:
            raise ValueError("retained checkpoint phase differs")
        initial_inventory = inventory(rt.objective.student)
        with (
            forbidden_paths(rt) as calls,
            frozen_observation(rt) as state,
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            cache_start = time.perf_counter()
            with control_only_reads(rt, panel) as read_proof:
                inputs, recipes, parities = prepare_inputs(rt, panel, admission)
            torch.cuda.synchronize(0)
            cache_seconds = time.perf_counter() - cache_start
            cache_hashes = [fingerprint(row) for row in inputs]
            with order_observer(rt.objective.student) as observer:
                for factor in PROTOCOL["factors"]:
                    label = f"d{int(factor['dropout'])}o{int(factor['random_order'])}"
                    banks[label] = {view: [] for view in PROTOCOL["views"]}
                    with frozen_observation(rt) as factor_state:
                        modes[label] = set_factor(rt.objective.student, factor)
                        for repeat, seed in enumerate(PROTOCOL["repeat_seeds"]):
                            pieces = {view: [] for view in PROTOCOL["views"]}
                            for number, row in enumerate(inputs):
                                admission()
                                torch.cuda.synchronize(0)
                                group_start = time.perf_counter()
                                for view, (args, kwargs) in row.items():
                                    observer.values.clear()
                                    actual_seed = encode_seed(seed, number, view)
                                    with torch.random.fork_rng(devices=[0]):
                                        seed_encode(actual_seed, rt.device)
                                        values = observed_cls(
                                            rt.objective.student,
                                            partial(
                                                rt.objective.student.encode_response,
                                                *args,
                                                **kwargs,
                                                decode=False,
                                            ),
                                        )
                                    if values.shape[0] != 4:
                                        raise ValueError("sealed four-cell group changed")
                                    pieces[view].append(values)
                                    order_rows.append(
                                        dict(
                                            factor=label,
                                            repeat=repeat,
                                            group=number,
                                            view=view,
                                            seed=actual_seed,
                                            **observer.receipt(),
                                        )
                                    )
                                if fingerprint(row) != cache_hashes[number]:
                                    raise ValueError(
                                        "frozen graph/control/view encoder inputs changed"
                                    )
                                torch.cuda.synchronize(0)
                                timing.append(time.perf_counter() - group_start)
                                if number == 1:
                                    write(
                                        output / "progress.json",
                                        dict(
                                            factor=label,
                                            repeat=repeat,
                                            groups_completed=len(timing),
                                            total_groups=512,
                                            elapsed_seconds=time.perf_counter() - start,
                                            first_two_group_seconds=sum(timing[-2:]),
                                            cache_seconds=cache_seconds,
                                        ),
                                    )
                            for view in PROTOCOL["views"]:
                                banks[label][view].append(torch.cat(pieces[view]))
                    factor_states[label] = factor_state
            cache_after = [fingerprint(row) for row in inputs]
            if cache_after != cache_hashes:
                raise ValueError("frozen encoder cache changed")
        statistics = {
            f: {v: repeat_statistics(x, expected_rows=128) for v, x in views.items()}
            for f, views in banks.items()
        }
        crossed = {
            f: {v: crossed_statistics(x, panel) for v, x in views.items()}
            for f, views in banks.items()
        }
    validate_spec(spec)
    if (
        sha(checkpoint) != cp["sha256"]
        or verify_source(endpoint["source"]) != source
        or verify_source(spec["analysis_source"]) != analysis
    ):
        raise ValueError("immutable source/checkpoint drift during observation")
    result = dict(
        schema="postfit-stochastic-audit-result-1",
        endpoint=endpoint,
        source=source,
        analysis_source=analysis,
        compatibility=compatibility,
        training_identity=identity,
        protocol=PROTOCOL,
        checkpoint=cp,
        checkpoint_endpoint=cp_endpoint,
        panel_sha256=spec["panel"]["sha256"],
        panel_schedule_sha256=panel["panel_sha256"],
        query_sha256=panel["query_sha256"],
        matched_views=recipes,
        matched_views_sha256=digest(recipes),
        cache_sha256=cache_hashes,
        cache_after_sha256=cache_after,
        parity=parities,
        initial_inventory=initial_inventory,
        factor_modes=modes,
        actual_orders=order_rows,
        state=state,
        factor_states=factor_states,
        statistics=statistics,
        crossed=crossed,
        updates=0,
        backward_calls=calls["backward"],
        objective_calls=calls["objective"],
        observed_calls=calls,
        control_expression_reads=read_proof,
        teacher=None,
        teacher_status="NA_not_observed",
        corrected_rank_status="NA_not_computed",
        raw_activations_saved=False,
        timing=dict(
            cache_seconds=cache_seconds,
            factor_group_seconds=timing,
            total_seconds=time.perf_counter() - start,
            student_encode_calls=1536,
            parity_encode_calls=192,
            native_graph_calls=128,
        ),
        interpretation=(
            "fixed eval graph; Cell/Response only; K4 signed estimates; no task acceptance"
        ),
    )
    result["zero_pkl"] = zero_pkl(output)
    validate_result(result)
    write_new(output / "COMPLETE.json", result)


def validate_result(row):
    labels = {f"d{int(f['dropout'])}o{int(f['random_order'])}" for f in PROTOCOL["factors"]}
    if any(
        set(row[key]) != labels
        for key in ("statistics", "factor_states", "factor_modes", "crossed")
    ):
        raise ValueError("exact complete factorial result keys required")
    if row["cache_sha256"] != row["cache_after_sha256"] or len(row["cache_sha256"]) != 32:
        raise ValueError("frozen graph/control cache hash protection failed")
    if row["zero_pkl"] != dict(checked=True, count=0):
        raise ValueError("whole endpoint zero-PKL evidence missing")
    if {k: row["control_expression_reads"][k] for k in ("calls", "rows", "forbidden")} != dict(
        calls=64, rows=256, forbidden=0
    ):
        raise ValueError("actual control-only assembly read evidence differs")
    counts = row["observed_calls"]
    if (
        counts["student_encode"] != 1728
        or counts["native_graph"] != 128
        or any(
            value != 0
            for key, value in counts.items()
            if key not in ("student_encode", "native_graph")
        )
    ):
        raise ValueError("observed encoder/reference counts or forbidden execution differ")
    for label, views in row["statistics"].items():
        if set(views) != set(PROTOCOL["views"]):
            raise ValueError("complete main/G0/G1 statistics required")
        for stats in views.values():
            if stats["rows"] != 128 or stats["repeats"] != 4 or len(stats["raw"]) != 4:
                raise ValueError("whole128 population and K4 required")
            for raw in [*stats["raw"], stats["repeat_mean"]]:
                if raw["n"] != 128 or raw["rank_ceiling"] != min(raw["dimension"], 127):
                    raise ValueError("wrong spectrum population/ceiling")
            if label == "d0o0" and (
                stats["Wraw"] != 0 or stats["Wcenter"] != 0 or len(set(stats["repeat_sha256"])) != 1
            ):
                raise ValueError("Doff/Ofixed repeats are not exactly deterministic")
    expected = {
        (f, r, g, v) for f in labels for r in range(4) for g in range(32) for v in PROTOCOL["views"]
    }
    observed = {}
    for order in row["actual_orders"]:
        key = (order["factor"], order["repeat"], order["group"], order["view"])
        if key not in expected or key in observed or order["cell"] != order["response"]:
            raise ValueError("duplicate/missing/misaligned native order observations")
        if order["seed"] != encode_seed(PROTOCOL["repeat_seeds"][key[1]], key[2], key[3]):
            raise ValueError("native order seed identity differs")
        observed[key] = order["cell"]
    if set(observed) != expected:
        raise ValueError("incomplete factorial/seed/group/view order set")
    for group in range(32):
        for view in PROTOCOL["views"]:
            fixed = {observed[(f, r, group, view)] for f in ("d0o0", "d1o0") for r in range(4)}
            if len(fixed) != 1:
                raise ValueError("native fixed order changed between repeats/dropout factors")


def combine(results):
    if [r["endpoint"]["name"] for r in results] != list(NAMES):
        raise ValueError("complete ordered endpoint bank required")
    baseline = results[0]
    for row in results:
        validate_result(row)
        if any(
            row[k] != baseline[k]
            for k in (
                "panel_sha256",
                "panel_schedule_sha256",
                "query_sha256",
                "matched_views_sha256",
            )
        ):
            raise ValueError("actual frozen population/view inputs differ between endpoints")
        name = "RI" if row["endpoint"]["name"] == "RI" else "R13SRAMP"
        if physical_data(row["training_identity"]["data"], name) != physical_data(
            baseline["training_identity"]["data"], "RI"
        ):
            raise ValueError("physical data differ beyond declared spectral metadata")
        if digest(row["protocol"]) != digest(PROTOCOL) or any(
            row[k] != 0 for k in ("updates", "backward_calls", "objective_calls")
        ):
            raise ValueError("protocol or zero-update evidence differs")
        for state in [row["state"], *row["factor_states"].values()]:
            if not state["unchanged"] or state["before"] != state["after"]:
                raise ValueError("protected state changed")
        if len(row["parity"]) != 32 or len(row["actual_orders"]) != 1536:
            raise ValueError("incomplete native parity/order observations")
        for parity in row["parity"]:
            if set(parity) != set(PROTOCOL["views"]):
                raise ValueError("native main/Global parity required")
            for value in parity.values():
                if (
                    value["used_path_passed"] is not True
                    or value["cell_chunk_used"] is not False
                    or value["final_rng_equal"] is not True
                    or value["native_graph_reference"] is not True
                ):
                    raise ValueError("native reference/RNG parity missing")
                if any(
                    not math.isfinite(value[k]) or not 0 <= value[k] <= 2e-6
                    for k in ("graph_max_abs", "response_max_abs")
                ):
                    raise ValueError("native reference parity exceeds tolerance")
    return dict(
        schema="postfit-stochastic-audit-complete-1",
        updates=0,
        endpoints=list(NAMES),
        protocol=PROTOCOL,
        panel_sha256=baseline["panel_sha256"],
        matched_views_sha256=baseline["matched_views_sha256"],
        scientific_complete=False,
        fit_authorized=False,
        interpretation=(
            "zero-update fixed-graph observation complete; no required noise/SNR outcome"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--spec-sha256")
    parser.add_argument("--endpoint", choices=NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--print-protocol", action="store_true")
    args = parser.parse_args()
    if args.print_protocol:
        import json

        print(json.dumps(PROTOCOL, indent=2))
        return
    if args.spec is None or args.spec_sha256 is None:
        parser.error("--spec and --spec-sha256 are required")
    if sha(args.spec) != args.spec_sha256:
        raise ValueError("sealed input specification hash differs")
    spec = read(args.spec)
    validate_spec(spec)
    if Path(spec["analysis_source"]["root"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("executed script is outside the sealed analysis source")
    if (
        os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0")
    ):
        raise ValueError("native allocator/determinism/GPU0 environment contract differs")
    from run_group import lock
    from shared_gpu import descendant, parent_map, process_info

    output = Path(spec["output"])
    if args.endpoint:
        if sha(output / "INPUT.json") != args.spec_sha256:
            raise ValueError("worker input differs from the coordinator sealed input")
        owner = read(output / "OWNER.json")
        if process_info(owner["pid"]) != owner or not descendant(
            os.getpid(), owner["pid"], parent_map()
        ):
            raise ValueError("worker lacks live exact coordinator/GPU lease ownership")
        endpoint_worker(spec, next(e for e in spec["endpoints"] if e["name"] == args.endpoint))
        return
    verify_source(spec["analysis_source"])
    # The detached coordinator must already be adopted before its PPID is sealed.
    deadline = time.monotonic() + 10
    owner = None
    while time.monotonic() < deadline:
        candidate = process_info(os.getpid())
        if candidate is not None and candidate["ppid"] == 1:
            owner = candidate
            break
        time.sleep(0.1)
    if owner is None:
        raise ValueError("detached coordinator has not reached stable PPID1 adoption")
    with lock(Path("/data/yilangliu/GraD-Pert/runtime/v2-gpu-0.lock")):
        output.mkdir(parents=True, exist_ok=False)
        (output / "INPUT.json").write_bytes(args.spec.read_bytes())
        write_new(output / "OWNER.json", owner)
        try:
            for endpoint in spec["endpoints"]:
                env = dict(os.environ, PYTHONPATH=str(Path(endpoint["source"]["root"]) / "src"))
                with (output / f"{endpoint['name']}.log").open("x") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--spec",
                            str(args.spec.resolve()),
                            "--spec-sha256",
                            args.spec_sha256,
                            "--endpoint",
                            endpoint["name"],
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            result = combine([read(output / name / "COMPLETE.json") for name in NAMES])
            validate_spec(spec)
            if sha(args.spec) != args.spec_sha256:
                raise ValueError("input specification drift before completion")
            verify_source(spec["analysis_source"])
            for endpoint in spec["endpoints"]:
                verify_source(endpoint["source"])
            result["spec_sha256"] = args.spec_sha256
            result["result_sha256"] = {name: sha(output / name / "COMPLETE.json") for name in NAMES}
            result["zero_pkl"] = zero_pkl(output)
            write_new(output / "COMPLETE.json", result)
        except BaseException as error:
            write_new(
                output / "FAILURE.json",
                dict(type=type(error).__name__, error=str(error), updates=0),
            )
            raise


if __name__ == "__main__":
    main()
