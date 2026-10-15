"""Server-only, read-only, population/grouped last-checkpoint diagnostics.

Import gradpert from the immutable TRAINING checkout (PYTHONPATH), not this
analysis checkout. The two identities are verified and recorded independently.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from cap40_posthoc_stats import (
    Distribution,
    Moments,
    condition_summary,
    digest_json,
    query_panels,
    seeded_int,
    select_balanced,
)

HEADS = ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node")
METHOD_SHA = "420d3a1ecf0ca05a88859ae2c2457fa85a126ef0"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source_identity(root):
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if dirty:
        raise ValueError(f"source is not clean: {root}")
    return {"commit": commit, "dirty": False}


def server_path(path):
    if not Path(path).resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("scientific materialization is server-only")


def method_imports():
    import torch

    from gradpert.config import load_experiment_config
    from gradpert.hashing import sha256_json
    from gradpert.training.v2 import objective as module
    from gradpert.training.v2.runtime import prepare_runtime

    root = Path(module.__file__).resolve().parents[4]
    identity = source_identity(root)
    if identity["commit"] != METHOD_SHA:
        raise ValueError("method imports must resolve to original 420d3a1")
    return torch, load_experiment_config, sha256_json, prepare_runtime, root


def runtime_args(args):
    torch, load, json_hash, prepare, method_root = method_imports()
    config = load(args.config)
    identity = read(args.run_root / "run_manifest.json")
    if (
        identity["source"]["commit"] != METHOD_SHA
        or json_hash(config.model_dump(mode="json")) != identity["resolved_config_sha256"]
    ):
        raise ValueError("checkpoint/config/method mismatch")
    return torch, config, identity, prepare, method_root


def records_from_data(data):
    records = []
    seen = set()
    for spec in data.training_batch_identity_specs(epoch=0, batch_size=64, max_unique_conditions=0):
        for truth, control, p in zip(
            spec.perturbed_row_ids, spec.control_row_ids, spec.condition_ids, strict=True
        ):
            i, ci = data._row_index[truth], data._row_index[control]
            context = str(data.context_ids[i])
            if truth in seen or str(data.context_ids[ci]) != context:
                raise ValueError("duplicate truth or unmatched control context")
            seen.add(truth)
            cell_type = str(data._adata.obs.iloc[i]["cell_type"])
            records.append(
                {
                    "truth_row_id": truth,
                    "control_row_id": control,
                    "condition_id": p,
                    "context_id": context,
                    "cell_line": "Jurkat",
                    "cell_type": cell_type,
                    "batch": str(data.batch_ids[i]),
                }
            )
    expected = {data.row_ids[i] for i in data.train_row_indices}
    if seen != expected:
        raise ValueError("full cap40 row coverage differs from runtime training selection")
    return records


def make_raw(runtime, records, *, dummy=False):
    torch = __import__("torch")
    from gradpert.training.batch import GraDPertTrainingBatch

    d = runtime.data
    if dummy:
        control = truth = torch.zeros(len(records), d.manifest.n_expression_genes)
    else:
        control = torch.from_numpy(
            d._read_expression_indices([d._row_index[r["control_row_id"]] for r in records])
        )
        truth = torch.from_numpy(
            d._read_expression_indices([d._row_index[r["truth_row_id"]] for r in records])
        )
    ps = tuple(r["condition_id"] for r in records)
    return GraDPertTrainingBatch(
        control_expression=control,
        target_expression=truth,
        condition_ids=ps,
        anchors_by_condition={p: d.anchors_by_condition[p] for p in set(ps)},
        perturbed_row_ids=tuple(r["truth_row_id"] for r in records),
        control_row_ids=tuple(r["control_row_id"] for r in records),
    )


def assemble(runtime, records, queries, seed, *, deep, dummy=False):
    from gradpert.training.v2.views import assemble_batch

    options = (
        runtime.options if deep else dataclasses.replace(runtime.options, lambda1=0.0, lambda2=0.0)
    )
    return assemble_batch(
        make_raw(runtime, records, dummy=dummy),
        runtime.index,
        options,
        np.random.default_rng(seed),
        allowed_expression_ids=np.array(queries),
    )


def recipe(batch):
    def graph(v):
        return {
            "ids": v.ids.tolist(),
            "mask": v.masked_positions.tolist(),
            "edges_sha256": digest_json(
                [v.neighbors.tolist(), v.valid.tolist(), v.sources.tolist()]
            ),
            "context_sha256": digest_json(
                [
                    v.context.ids.tolist(),
                    v.context.neighbors.tolist(),
                    v.context.valid.tolist(),
                    v.context.sources.tolist(),
                ]
            )
            if v.context is not None
            else None,
        }

    return {
        "queries": batch.graph.ids[batch.query_positions].tolist(),
        "main": graph(batch.graph),
        "graph_views": [graph(v) for v in batch.graph_views],
        "cell_views": [
            {"positions": v.positions.tolist(), "mask": v.mask.tolist()} for v in batch.cell_views
        ],
    }


def seal(args):
    server_path(args.output)
    if args.output.exists():
        raise ValueError("manifest requires a fresh output ID")
    torch, config, identity, prepare, _ = runtime_args(args)
    analysis = source_identity(Path(__file__).resolve().parents[2])
    with prepare(
        config,
        data_root=args.data_root,
        run_seed=1,
        device=torch.device("cpu"),
        purpose="evaluation",
    ) as rt:
        if rt.identity != identity["data"]:
            raise ValueError("data identity mismatch")
        records = records_from_data(rt.data)
        groups = defaultdict(list)
        for row in records:
            groups[row["condition_id"]].append(row)
        if len(records) != 47836 or len(groups) != 1335:
            raise ValueError("unexpected cap40 population")
        panels = query_panels(rt.allowed_expression_ids, rt.options.query_count, args.seed)
        gradient_conditions = set(sorted(groups, key=lambda p: seeded_int(p, args.seed + 3))[:64])
        args.output.mkdir(parents=True)
        schedule = []
        for i, p in enumerate(sorted(groups)):
            rows = sorted(groups[p], key=lambda r: r["truth_row_id"])
            deep = select_balanced(rows.copy(), 8, args.seed)
            queries = panels[i % len(panels)]
            seed = seeded_int(p, args.seed) % (2**32)
            batch = assemble(rt, deep, queries, seed, deep=True, dummy=True)
            rec = recipe(batch)
            path = args.output / "recipes" / f"{i:04d}.json"
            write(path, rec)
            schedule.append(
                {
                    "index": i,
                    "condition_id": p,
                    "rows": rows,
                    "deep_rows": deep,
                    "gradient_rows": deep[:4] if p in gradient_conditions else [],
                    "queries": queries,
                    "view_seed": seed,
                    "recipe_file": str(path),
                    "recipe_sha256": sha(path),
                }
            )
            if i % 100 == 0:
                print(json.dumps({"seal_conditions": i + 1}), flush=True)
        manifest = {
            "schema": "cap40-posthoc-manifest-1",
            "analysis": analysis,
            "method_sha": METHOD_SHA,
            "data": rt.identity,
            "seed": args.seed,
            "rows": len(records),
            "conditions": len(groups),
            "independent_controls": len({r["control_row_id"] for r in records}),
            "deep_rows": sum(len(g["deep_rows"]) for g in schedule),
            "gradient_rows": sum(len(g["gradient_rows"]) for g in schedule),
            "allowed_expression_ids": list(map(int, rt.allowed_expression_ids)),
            "query_panels": panels,
            "schedule": schedule,
            "order_policy": (
                "original model eval order/eval_seed; training randomization disabled in eval"
            ),
            "selection_use": "none",
            "numeric_visibility": "train exclusions retained",
            "scope": (
                "full cap40 train rows; one original context-matched control "
                "per truth; no Cartesian expansion"
            ),
        }
        write(args.output / "manifest.json", manifest)
        write(
            args.output / "COMPLETE.json",
            {
                "status": "completed",
                "manifest_sha256": sha(args.output / "manifest.json"),
                "rows": len(records),
                "conditions": len(groups),
            },
        )
        print(
            json.dumps(
                {
                    "manifest": str(args.output / "manifest.json"),
                    "sha256": sha(args.output / "manifest.json"),
                }
            ),
            flush=True,
        )


def npval(x):
    return x.detach().float().cpu().numpy()


def feature(value):
    """Match encoder's stream-mean readout; do not pool cells into gene axes."""
    if isinstance(value, tuple):
        value = value[0]
    if value.ndim == 4:
        value = value.mean(-2)
    return value


class Collector:
    def __init__(self, batch, records, seed):
        self.batch, self.records, self.seed = batch, records, seed
        self.features = defaultdict(lambda: None)
        self.distributions = {}
        self.distribution_objects = {}
        self.pairs, self.outputs, self.per_cell = [], [], []
        self.logits = {}
        self.seen_pairs = set()
        self.graph_count = defaultdict(int)
        self.current = None
        self.response_count = defaultdict(int)
        self.head_count = defaultdict(int)
        self.response_cache = {}
        self.handles, self.wrappers = [], []
        self.same_gene = []
        self.cross_view = []

    def add_features(self, key, value):
        x = npval(feature(value))
        if x.ndim == 3:
            # Sample actual identities by a shared seeded hash, never a prefix.
            ids = (
                self.current[2]
                if self.current is not None and len(self.current[2]) == x.shape[1]
                else np.arange(x.shape[1])
            )
            selected = np.argsort([seeded_int(int(g), self.seed) for g in ids])[
                : min(128, len(ids))
            ]
            sampled = x[:, selected]
            # Same-gene differences across controls; gene identity is held fixed.
            for context in sorted({r["context_id"] for r in self.records[: len(x)]}):
                indices = [
                    i for i, r in enumerate(self.records[: len(x)]) if r["context_id"] == context
                ]
                unique = {}
                for i in indices:
                    unique.setdefault(self.records[i]["control_row_id"], i)
                indices = list(unique.values())
                within = (
                    float(sampled[indices].var(0).sum(-1).mean()) if len(indices) >= 2 else None
                )
                self.same_gene.append(
                    {
                        "feature": key,
                        "context_id": context,
                        "independent_controls": len(indices),
                        "genes": len(selected),
                        "same_gene_across_cells_variance": within,
                        "gene_ids_sha256": digest_json(np.asarray(ids)[selected].tolist()),
                    }
                )
            for i, row in enumerate(self.records[: len(x)]):
                m = Moments(x.shape[-1])
                m.add(sampled[i])
                self.per_cell.append(
                    {
                        "feature": key,
                        "truth_row_id": row["truth_row_id"],
                        "control_row_id": row["control_row_id"],
                        "context_id": row["context_id"],
                        "cell_type": row["cell_type"],
                        "batch": row["batch"],
                        "gene_ids_sha256": digest_json(np.asarray(ids)[selected].tolist()),
                        **m.result(),
                    }
                )
            x = sampled.reshape(-1, x.shape[-1])
        else:
            x = x.reshape(-1, x.shape[-1])
        if self.features[key] is None:
            self.features[key] = Moments(x.shape[-1])
        self.features[key].add(x)

    def install(self, obj):
        import torch

        from gradpert.training.v2 import objective as module

        self.original_ce = module.cross_entropy_terms
        self.module = module
        for role in ("student", "teacher"):
            model = getattr(obj, role)

            def graph_hook(mod, inputs, output, role=role):
                index = self.graph_count[role]
                self.graph_count[role] += 1
                label = (
                    "main"
                    if role == "student" and index == 0
                    else f"view{index - 1 if role == 'student' else index}"
                )
                ids = inputs[0].detach().cpu().numpy()
                selected = np.argsort([seeded_int(int(g), self.seed) for g in ids])[:512]
                self.add_features(f"{role}.graph.{label}", output[selected])

            self.handles.append(model.graph.register_forward_hook(graph_hook))
            original = model.encode_response
            self.wrappers.append((model, original))

            def wrapped(*values, _role=role, _original=original, **kw):
                index = self.response_count[_role]
                self.response_count[_role] += 1
                view = (
                    "main"
                    if _role == "student" and index == 0
                    else f"view{index - 1 if _role == 'student' else index}"
                )
                pos = None if view == "main" else self.batch.cell_views[int(view[4:])].positions
                ids = (
                    self.batch.graph.ids[
                        self.batch.query_positions
                        if pos is None
                        else self.batch.query_positions[pos]
                    ]
                    .cpu()
                    .numpy()
                )
                self.current = (_role, view, ids)
                output = _original(*values, **kw)
                for name in ("control_cls", "response_cls", "response_tokens"):
                    self.add_features(f"{_role}.{view}.{name}", output[name])
                self.response_cache[(_role, view)] = output["response_cls"].detach().float().cpu()
                self.current = None
                return output

            model.encode_response = wrapped
            for path in ("cell", "response"):
                encoder = getattr(model, path, None)
                if encoder is None:
                    continue
                for name, layer in encoder.named_modules():
                    # Component outputs, not all internal Linear activations.
                    if (
                        name
                        and name.split(".")[0]
                        in ("layers", "self_layers", "cross_layers", "ffn_layers", "injections")
                        and name.count(".") == 1
                    ):

                        def hook(mod, inputs, output, role=role, path=path, name=name):
                            if self.current is not None and self.current[1] == "main":
                                val = feature(output)
                                if val.ndim == 3:
                                    # Last token is CLS for these encoder components.
                                    ids = self.current[2]
                                    self.add_features(
                                        f"{role}.main.{path}.{name}.genes", val[:, : len(ids)]
                                    )
                                    self.add_features(f"{role}.main.{path}.{name}.cls", val[:, -1])

                        self.handles.append(layer.register_forward_hook(hook))
            for head_name in HEADS:
                head = getattr(model, head_name)

                def head_hook(mod, inputs, output, role=role, head_name=head_name):
                    k = (role, head_name)
                    index = self.head_count[k]
                    self.head_count[k] += 1
                    view = f"view{index}"
                    key = f"{role}.{head_name}.{view}"
                    self.logits[id(output)] = (key, output)
                    center = getattr(obj, head_name + "_center")
                    scaled = (
                        output.float() / 0.1
                        if role == "student"
                        else (output.float() - center) / 0.04
                    )
                    dist = Distribution(output.shape[-1])
                    for chunk in scaled.reshape(-1, scaled.shape[-1]).split(128):
                        dist.add(npval(chunk))
                    self.distributions[key] = dist.result()
                    self.distribution_objects[key] = dist
                    # Cell/gene rows already correctly selected by the objective.
                    self.add_features(key + ".input", inputs[0])

                self.handles.append(head.register_forward_hook(head_hook))

                def bottleneck_hook(mod, inputs, output, role=role, head_name=head_name):
                    index = self.head_count[(role, head_name)]
                    self.add_features(f"{role}.{head_name}.view{index}.bottleneck", output)

                self.handles.append(head.features.register_forward_hook(bottleneck_hook))

        def observed_ce(student, teacher, center):
            value = self.original_ce(student, teacher, center)
            sx = self.logits.get(id(student))
            tx = self.logits.get(id(teacher))
            if sx is None or tx is None:
                raise ValueError("unmapped CE head output")
            pair_identity = (id(student), id(teacher), id(center))
            if pair_identity in self.seen_pairs:
                return value
            self.seen_pairs.add(pair_identity)
            values = []
            for s, t in zip(
                student.detach().float().split(128),
                teacher.detach().float().split(128),
                strict=True,
            ):
                p = ((t - center) / 0.04).softmax(-1)
                h = -(p * p.clamp_min(1e-30).log()).sum(-1)
                ce = -(p * (s / 0.1).log_softmax(-1)).sum(-1)
                values.append(torch.stack((ce, h, ce - h), -1).cpu())
            a = torch.cat(values).numpy()
            self.pairs.append(
                {
                    "student": sx[0],
                    "teacher": tx[0],
                    "pairs": len(a),
                    "CE": float(a[:, 0].mean()),
                    "H_pair": float(a[:, 1].mean()),
                    "KL": float(a[:, 2].mean()),
                }
            )
            return value

        module.cross_entropy_terms = observed_ce

    def close(self):
        for model, original in self.wrappers:
            model.encode_response = original
        for h in self.handles:
            h.remove()
        self.module.cross_entropy_terms = self.original_ce

    def result(self):
        import torch

        for role in ("student", "teacher"):
            available = [
                (v, x) for (r, v), x in self.response_cache.items() if r == role and v != "main"
            ]
            for i, (va, a) in enumerate(available):
                for vb, b in available[i + 1 :]:
                    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                    self.cross_view.append(
                        {
                            "role": role,
                            "view_a": va,
                            "view_b": vb,
                            "independent_cells": len(a),
                            "mean_cosine": float(cosine.mean()),
                            "mean_squared_difference": float((a - b).square().mean()),
                        }
                    )
        return {
            "features": {k: v.result() for k, v in self.features.items() if v is not None},
            "heads": self.distributions,
            "pairs": self.pairs,
            "per_cell": self.per_cell,
            "same_gene": self.same_gene,
            "cross_view": self.cross_view,
        }


def output_rows(response, batch, records):
    a, y, c = (npval(x) for x in (response["prediction"], batch.truth, batch.control))
    d, effect = a - c, y - c
    rows = []
    for i, meta in enumerate(records):
        target, pred = effect[i], d[i]
        denom = float(target @ target)
        rows.append(
            {
                **meta,
                "queries": a.shape[1],
                "prediction_mse": float(((a[i] - y[i]) ** 2).mean()),
                "control_copy_mse": float(((c[i] - y[i]) ** 2).mean()),
                "prediction_negative_fraction": float((a[i] < 0).mean()),
                "prediction_zero_fraction": float((a[i] == 0).mean()),
                "prediction_min": float(a[i].min()),
                "prediction_max": float(a[i].max()),
                "prediction_mean": float(a[i].mean()),
                "truth_mean": float(y[i].mean()),
                "delta_rms": float(np.sqrt((pred**2).mean())),
                "effect_rms": float(np.sqrt((target**2).mean())),
                "effect_scale_slope_no_intercept": float(pred @ target / denom)
                if denom > 0
                else None,
                "effect_sign_accuracy_nonzero": float(
                    (np.sign(pred[target != 0]) == np.sign(target[target != 0])).mean()
                )
                if (target != 0).any()
                else None,
            }
        )
    return rows


def static_graph(rt):
    """Full unmasked context once; selected queries are subsequently indexed by ID."""
    view = rt.index.view(
        np.arange(rt.index.n_nodes),
        [(0,)],
        rng=np.random.default_rng(0),
        device=rt.device,
        induced=False,
    )
    graph, _ = rt.objective._graph(rt.objective.student, view, False)
    return graph


def main_response(rt, batch, cached):
    obj = rt.objective
    ids = batch.graph.ids
    target_ids = ids[batch.graph.target_positions.clamp_min(0)]
    target = cached[target_ids] * batch.graph.target_valid.unsqueeze(-1)
    condition = target.sum(1) / batch.graph.target_valid.sum(1).clamp_min(1).unsqueeze(-1)
    condition = condition[batch.condition_index]
    return obj.student.encode_response(
        cached[ids[batch.query_positions]], batch.control, condition, **obj.response_metadata(batch)
    )


def verify_cache(rt, batch, cached):
    import torch

    obj = rt.objective
    graph, condition = obj._batch_graph(obj.student, batch)
    selected = cached[batch.graph.ids]
    difference = float((selected - graph).abs().max())
    original = obj.student.encode_response(
        graph[batch.query_positions], batch.control, condition, **obj.response_metadata(batch)
    )
    optimized = main_response(rt, batch, cached)
    maximum = max(float((original[k] - optimized[k]).abs().max()) for k in original)
    if difference > 2e-6 or maximum > 2e-6:
        raise ValueError(
            f"static context cache parity failed graph={difference} response={maximum}"
        )
    # Cell chunk changes must not alter values or identity alignment.
    from gradpert.training.v2.engine import slice_cells

    pieces = [
        main_response(rt, slice_cells(batch, i, i + 1), cached) for i in range(len(batch.control))
    ]
    chunk = max(
        float((original[k] - torch.cat([p[k] for p in pieces])).abs().max()) for k in original
    )
    if chunk > 2e-6:
        raise ValueError(f"cell-chunk parity failed: {chunk}")
    return {
        "graph_max_abs": difference,
        "response_max_abs": maximum,
        "cell_chunk_max_abs": chunk,
        "tolerance": 2e-6,
    }


def csv_write(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    from capacity_probe import validation_state
    from shared_gpu import admit, snapshot

    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.process_diagnostics import evaluate_graph_gradient_diagnostics
    from gradpert.training.v2.runtime import _move_batch_tree

    server_path(args.output)
    manifest = read(args.manifest)
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError("shared manifest hash differs")
    torch, config, identity, prepare, _ = runtime_args(args)
    journal = read(args.run_root / "fit/epoch_state.json")
    cp = journal["last"]
    checkpoint = args.run_root / "fit" / cp["file"]
    if sha(checkpoint) != cp["sha256"]:
        raise ValueError("checkpoint hash differs")
    if args.output.exists():
        raise ValueError("fresh analysis output required")
    args.output.mkdir(parents=True)
    analysis = source_identity(Path(__file__).resolve().parents[2])
    policy = read(args.resource_queue)["shared_gpu_policy"]
    admit(policy, snapshot(os.getpid()))
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
    start = time.perf_counter()
    with prepare(
        config,
        data_root=args.data_root,
        run_seed=1,
        device=torch.device("cuda:0"),
        purpose="evaluation",
    ) as rt:
        if rt.identity != manifest["data"] or rt.identity != identity["data"]:
            raise ValueError("data binding mismatch")
        load_evaluation_checkpoint(
            checkpoint, rt.objective, training_identity=identity, checkpoint_sha256=cp["sha256"]
        )
        obj = rt.objective
        obj.eval()
        before = validation_state(rt)
        rows_out, basic_features, deep_results = [], [], []
        global_heads = {}
        global_features = {}
        basic_global = {}
        basic_between = {}
        basic_within_trace = defaultdict(float)
        basic_within_count = defaultdict(int)
        gradients = []
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(manifest["seed"])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cached = static_graph(rt)
                for key, x in (
                    ("student.genept.all_graph_genes", obj.student.graph.identities()),
                    ("student.graph.all_graph_genes", cached),
                ):
                    m = Moments(x.shape[-1])
                    m.add(npval(x))
                    global_features[key] = m
            schedule = manifest["schedule"][: args.limit] if args.limit else manifest["schedule"]
            for number, g in enumerate(schedule):
                admit(policy, snapshot(os.getpid()))
                t = time.perf_counter()
                raw_batch = assemble(rt, g["rows"], g["queries"], g["view_seed"], deep=False)
                batch = _move_batch_tree(raw_batch, rt.device)
                local_features = {}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    parity = verify_cache(rt, batch, cached) if number == 0 else None
                    response = main_response(rt, batch, cached)
                output = output_rows(response, batch, g["rows"])
                rows_out.extend(output)
                for key in ("control_cls", "response_cls"):
                    val = npval(response[key])
                    for context in sorted({r["context_id"] for r in g["rows"]}):
                        indices = [i for i, r in enumerate(g["rows"]) if r["context_id"] == context]
                        # Repeated controls within one condition are not independent features.
                        unique = {}
                        [unique.setdefault(g["rows"][i]["control_row_id"], i) for i in indices]
                        indices = list(unique.values())
                        m = Moments(val.shape[-1])
                        m.add(val[indices])
                        entry = {
                            "condition_id": g["condition_id"],
                            "context_id": context,
                            "feature": key,
                            "truth_rows": len(g["rows"]),
                            "cell_line": "Jurkat",
                            **m.result(),
                        }
                        basic_features.append(entry)
                        label = key + ":" + context
                        global_m = basic_global.setdefault(label, Moments(val.shape[-1]))
                        global_m.merge(m.n, m.mean, m.m2, m.norm_sum)
                        basic_between.setdefault(label, Moments(val.shape[-1])).add(m.mean[None])
                        basic_within_trace[label] += float(np.trace(m.m2))
                        basic_within_count[label] += m.n
                        local_features[label] = {
                            "count": m.n,
                            "mean": m.mean.tolist(),
                            "variance": float(np.trace(m.m2) / max(m.n, 1)),
                        }
                deep_cpu = assemble(rt, g["deep_rows"], g["queries"], g["view_seed"], deep=True)
                if sha(g["recipe_file"]) != g["recipe_sha256"] or digest_json(
                    recipe(deep_cpu)
                ) != digest_json(read(g["recipe_file"])):
                    raise ValueError(
                        "actual view/mask/gene/edge recipe differs from common manifest"
                    )
                deep_batch = _move_batch_tree(deep_cpu, rt.device)
                collector = Collector(deep_batch, g["deep_rows"], manifest["seed"])
                collector.install(obj)
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, terms = obj(deep_batch)
                    deep = collector.result()
                    for key, dist in collector.distribution_objects.items():
                        total = global_heads.setdefault(key, Distribution(len(dist.sum)))
                        total.n += dist.n
                        total.sum += dist.sum
                        total.hits += dist.hits
                        total.entropy_sum += dist.entropy_sum
                        total.top1_sum += dist.top1_sum
                        total.logit_std_sum += dist.logit_std_sum
                    for key, m in collector.features.items():
                        if m is not None:
                            global_features.setdefault(key, Moments(len(m.mean))).merge(
                                m.n, m.mean, m.m2, m.norm_sum
                            )
                finally:
                    collector.close()
                    obj.pending.clear()
                deep.update(
                    {
                        "condition_id": g["condition_id"],
                        "independent_truth_rows": len(g["deep_rows"]),
                        "independent_controls": len({r["control_row_id"] for r in g["deep_rows"]}),
                        "joint": float(loss),
                        "loss_terms": {k: float(v) for k, v in terms.items()},
                    }
                )
                deep_results.append(deep)
                if g["gradient_rows"] or args.limit:
                    grad_cpu = assemble(
                        rt,
                        g["gradient_rows"] or g["deep_rows"][:4],
                        g["queries"],
                        g["view_seed"],
                        deep=True,
                    )
                    gradient = evaluate_graph_gradient_diagnostics(
                        obj, _move_batch_tree(grad_cpu, rt.device), bf16=True
                    )
                    gradients.append({"condition_id": g["condition_id"], **gradient})
                # Save condition evidence atomically; no per-cell activation matrix persists.
                write(
                    args.output / "condition_states" / f"{g['index']:04d}.json",
                    {
                        "condition_id": g["condition_id"],
                        "basic_moments": local_features,
                        "deep": deep,
                        "parity": parity,
                    },
                )
                torch.cuda.synchronize()
                progress = {
                    "arm": args.arm,
                    "conditions_done": number + 1,
                    "conditions_total": len(schedule),
                    "rows_done": len(rows_out),
                    "rows_total": manifest["rows"],
                    "elapsed_seconds": time.perf_counter() - start,
                    "condition_seconds": time.perf_counter() - t,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                }
                write(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        if validation_state(rt) != before or sha(checkpoint) != cp["sha256"] or obj.pending:
            raise ValueError(
                "posthoc mutated model/teacher/center/optimizer/gradients/RNG/checkpoint"
            )
        for h in HEADS:
            if not torch.isfinite(getattr(obj, h + "_center")).all():
                raise ValueError("nonfinite center")
        if not args.limit and (
            len(rows_out) != manifest["rows"]
            or len({r["truth_row_id"] for r in rows_out}) != manifest["rows"]
        ):
            raise ValueError("unique row coverage incomplete")
        csv_write(args.output / "outputs_full.csv", rows_out)
        csv_write(args.output / "cls_within_condition.csv", basic_features)
        write(args.output / "deep.json", deep_results)
        write(args.output / "gradients.json", gradients)
        summary = {
            "schema": "cap40-posthoc-1",
            "arm": args.arm,
            "analysis": analysis,
            "method_sha": METHOD_SHA,
            "checkpoint_epoch": cp["epoch"],
            "checkpoint_sha256": cp["sha256"],
            "run_root": str(args.run_root),
            "run_manifest_sha256": sha(args.run_root / "run_manifest.json"),
            "config_sha256": sha(args.config),
            "resolved_config_sha256": identity["resolved_config_sha256"],
            "manifest_sha256": args.manifest_sha256,
            "data": rt.identity,
            "scope": "gate_subset" if args.limit else "full_cap40_train",
            "conditions": len(schedule),
            "rows": len(rows_out),
            "deep_rows": sum(d["independent_truth_rows"] for d in deep_results),
            "global_heads": {k: v.result() for k, v in global_heads.items()},
            "global_features": {k: v.result() for k, v in global_features.items()},
            "cls_decomposition": {
                k: {
                    "total": v.result(),
                    "between_condition_mean": basic_between[k].result(),
                    "between_condition_variance_row_weighted": float(
                        np.trace(v.m2) / v.n - basic_within_trace[k] / basic_within_count[k]
                    ),
                    "within_condition_variance_row_weighted": basic_within_trace[k]
                    / basic_within_count[k],
                }
                for k, v in basic_global.items()
            },
            "metrics": {
                k: condition_summary(rows_out, k)
                for k in (
                    "prediction_mse",
                    "control_copy_mse",
                    "delta_rms",
                    "effect_rms",
                    "prediction_negative_fraction",
                    "effect_scale_slope_no_intercept",
                    "effect_sign_accuracy_nonzero",
                )
            },
            "state_before": before,
            "state_after": validation_state(rt),
            "updates": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "limits": [
                "Frozen eval-mode checkpoint probes; not historical training activations.",
                (
                    "Numeric train exclusions retained; unseen-expression features not collected; "
                    "original 18 test scores reused separately."
                ),
                (
                    "Per-cell covariance samples up to128 hash-selected genes; "
                    "rank ceiling<=127; no prefix sampling."
                ),
                "SSL1 condition/node units are conditions/graph nodes, not cell replicates.",
                "Single Jurkat cell line cannot estimate cross-cell-line differences.",
            ],
        }
        write(args.output / "summary.json", summary)
        outputs = {
            str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in args.output.iterdir()
            if p.is_file()
        }
        write(
            args.output / "COMPLETE.json",
            {
                "status": "completed",
                "scope": summary["scope"],
                "manifest_sha256": args.manifest_sha256,
                "checkpoint_sha256": cp["sha256"],
                "outputs": outputs,
                "state_unchanged": True,
            },
        )


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("seal", "run"))
    for name in ("run-root", "config", "data-root", "output", "manifest", "resource-queue"):
        p.add_argument(
            "--" + name, type=Path, required=name in ("run-root", "config", "data-root", "output")
        )
    p.add_argument("--manifest-sha256")
    p.add_argument("--arm", default="DP")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.mode == "seal":
        seal(args)
    else:
        from run_group import lock

        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if gpu not in ("0", "1"):
            raise ValueError("one isolated physical GPU per posthoc worker")
        with lock(Path(f"/data/yilangliu/GraD-Pert/runtime/v2-gpu-{gpu}.lock")):
            run(args)


if __name__ == "__main__":
    cli()
