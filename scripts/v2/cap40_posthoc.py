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
    cell_balanced_terms,
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


def select_schedule(schedule, limit=0, gate_indices=""):
    if gate_indices:
        indices = [int(i) for i in gate_indices.split(",")]
        if (
            not limit
            or len(set(indices)) != len(indices)
            or any(i < 0 or i >= len(schedule) for i in indices)
        ):
            raise ValueError("explicit gates need a bounded, unique in-range selection")
        return [schedule[i] for i in indices]
    return schedule[:limit] if limit else schedule


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


def medium_cells(rows, limit, seed):
    """Seeded context order per condition avoids always taking lexically first batches."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["context_id"]].append(row)
    for values in groups.values():
        values.sort(key=lambda r: seeded_int(r["truth_row_id"], seed))
    condition = rows[0]["condition_id"]
    order = sorted(groups, key=lambda c: seeded_int([condition, c], seed))
    selected = []
    for rank in range(max(map(len, groups.values()))):
        for context in order:
            if rank < len(groups[context]):
                selected.append(groups[context][rank])
                if len(selected) == limit:
                    return selected
    return selected


def medium_schedule(parent, *, cells=4, deep_conditions=128, gradient_conditions=32):
    """Outcome-independent context/size stratification, exact parent row/control IDs."""
    if not (0 < gradient_conditions <= deep_conditions <= len(parent["schedule"])):
        raise ValueError("invalid medium condition budgets")
    if cells < 2:
        raise ValueError("medium within-condition diagnostics need at least two cells")
    seed = parent["seed"]
    strata = defaultdict(list)
    for group in parent["schedule"]:
        rows = group["rows"]
        size = len(rows)
        size_bin = (
            "singleton" if size == 1 else "2-7" if size < 8 else "8-19" if size < 20 else "20+"
        )
        key = (tuple(sorted({r["context_id"] for r in rows})), size_bin)
        strata[key].append(group)
    for values in strata.values():
        values.sort(key=lambda g: seeded_int(g["condition_id"], seed + 11))
    keys = sorted(strata, key=lambda k: seeded_int(k, seed + 12))
    ordered = [
        strata[k][rank]["condition_id"]
        for rank in range(max(map(len, strata.values())))
        for k in keys
        if rank < len(strata[k])
    ]
    deep_ids, grad_ids = set(ordered[:deep_conditions]), set(ordered[:gradient_conditions])
    schedule = []
    for group in parent["schedule"]:
        rows = [
            {**r, "control_treatment": "control", "paired_perturbation": group["condition_id"]}
            for r in medium_cells(group["rows"], cells, seed + 13)
        ]
        schedule.append(
            {
                **group,
                "rows": rows,
                "deep_rows": rows if group["condition_id"] in deep_ids else [],
                "gradient_rows": rows if group["condition_id"] in grad_ids else [],
                "recipe_file": None,
                "recipe_sha256": None,
            }
        )
    return schedule


def seal_medium(args):
    server_path(args.output)
    if args.output.exists():
        raise ValueError("medium manifest needs a fresh output ID")
    if not args.parent_manifest or sha(args.parent_manifest) != args.parent_manifest_sha256:
        raise ValueError("parent manifest identity differs")
    parent = read(args.parent_manifest)
    torch, config, identity, prepare, _ = runtime_args(args)
    schedule = medium_schedule(parent)
    args.output.mkdir(parents=True)
    with prepare(
        config,
        data_root=args.data_root,
        run_seed=1,
        device=torch.device("cpu"),
        purpose="evaluation",
    ) as rt:
        if rt.identity != parent["data"] or rt.identity != identity["data"]:
            raise ValueError("medium/parent/runtime data differ")
        for number, g in enumerate(schedule):
            if g["deep_rows"]:
                batch = assemble(
                    rt, g["deep_rows"], g["queries"], g["view_seed"], deep=True, dummy=True
                )
                path = args.output / "recipes" / f"{g['index']:04d}.json"
                write(path, recipe(batch))
                g.update(recipe_file=str(path), recipe_sha256=sha(path))
            if number % 100 == 0:
                print(json.dumps({"medium_seal_conditions": number + 1}), flush=True)
    records = [r for g in schedule for r in g["rows"]]
    manifest = {
        **parent,
        "analysis": source_identity(Path(__file__).resolve().parents[2]),
        "analysis_scope": "medium_cap40_stratified",
        "parent_manifest": str(args.parent_manifest),
        "parent_manifest_sha256": args.parent_manifest_sha256,
        "rows": len(records),
        "conditions": len(schedule),
        "independent_controls": len({r["control_row_id"] for r in records}),
        "deep_rows": sum(len(g["deep_rows"]) for g in schedule),
        "deep_conditions": sum(bool(g["deep_rows"]) for g in schedule),
        "gradient_rows": sum(len(g["gradient_rows"]) for g in schedule),
        "gradient_conditions": sum(bool(g["gradient_rows"]) for g in schedule),
        "schedule": schedule,
        "scope": (
            "medium: every cap40 condition, at most four context-balanced paired cells; "
            "128 context/condition-size stratified deep conditions, 32 gradient conditions; "
            "not full train-population estimates; rare singleton conditions retained"
        ),
    }
    write(args.output / "manifest.json", manifest)
    write(
        args.output / "COMPLETE.json",
        {
            "status": "completed",
            "scope": manifest["analysis_scope"],
            "manifest_sha256": sha(args.output / "manifest.json"),
            **{
                k: manifest[k]
                for k in (
                    "rows",
                    "conditions",
                    "deep_rows",
                    "deep_conditions",
                    "gradient_rows",
                    "gradient_conditions",
                )
            },
        },
    )
    print(json.dumps(read(args.output / "COMPLETE.json")), flush=True)


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
        self.graph_view_names = {id(batch.graph.ids): "main"}
        self.graph_view_names.update(
            {id(v.ids): f"view{i}" for i, v in enumerate(batch.graph_views)}
        )
        self.current = None
        self.response_count = defaultdict(int)
        self.head_count = defaultdict(int)
        self.node_view_ids = {
            "ssl1_node": [
                i for i, v in enumerate(batch.graph_views[:2]) if len(v.masked_positions)
            ],
            "ssl2_node": [i for i, v in enumerate(batch.cell_views[:2]) if bool(v.mask.any())],
        }
        self.response_cache = {}
        self.handles, self.wrappers = [], []
        self.same_gene = []
        self.cross_view = []

    def head_view_index(self, head_name, call_index):
        if head_name not in self.node_view_ids:
            return call_index
        active = self.node_view_ids[head_name]
        if not 0 <= call_index < len(active):
            raise ValueError("unexpected node-head call beyond active masked views")
        return active[call_index]

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
                label = self.graph_view_names[id(inputs[0])]
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

            def cell_hook(mod, inputs, output, role=role):
                if self.current is not None:
                    ids = self.current[2]
                    val = feature(output)
                    self.add_features(
                        f"{role}.{self.current[1]}.control_tokens", val[:, : len(ids)]
                    )

            self.handles.append(model.cell.register_forward_hook(cell_hook))
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
                    view = f"view{self.head_view_index(head_name, index)}"
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
                    view = self.head_view_index(head_name, index)
                    self.add_features(f"{role}.{head_name}.view{view}.bottleneck", output)

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
            averaged = a
            aggregation = "head_rows"
            if ".ssl2_node." in sx[0]:
                index = int(sx[0].rsplit("view", 1)[1])
                mask = self.batch.cell_views[index].mask.cpu().numpy()
                averaged = cell_balanced_terms(a, np.nonzero(mask)[0], len(mask))
                aggregation = "masked_token_mean_then_valid_cell_mean"
            self.pairs.append(
                {
                    "student": sx[0],
                    "teacher": tx[0],
                    "pairs": len(a),
                    "CE": float(averaged[:, 0].mean()),
                    "H_pair": float(averaged[:, 1].mean()),
                    "KL": float(averaged[:, 2].mean()),
                    "raw_token_CE": float(a[:, 0].mean()),
                    "raw_token_H": float(a[:, 1].mean()),
                    "raw_token_KL": float(a[:, 2].mean()),
                    "aggregation": aggregation,
                    "effective_loss_rows": len(averaged),
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
            "mask_eligibility": [
                {
                    "head": "ssl2_node",
                    "view": i,
                    "cells": len(v.mask),
                    "valid_cells": int(v.mask.any(-1).sum()),
                    "masked_tokens": int(v.mask.sum()),
                    "status": "active" if bool(v.mask.any()) else "skipped_no_valid_mask",
                }
                for i, v in enumerate(self.batch.cell_views[:2])
            ],
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
                "truth_min": float(y[i].min()),
                "truth_max": float(y[i].max()),
                "truth_negative_fraction": float((y[i] < 0).mean()),
                "error_top1pct_fraction": float(
                    np.sort((a[i] - y[i]) ** 2)[-max(1, a.shape[1] // 100) :].sum()
                    / max(float(((a[i] - y[i]) ** 2).sum()), 1e-30)
                ),
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
    differences = {
        k: float((original[k] - torch.cat([p[k] for p in pieces])).abs().max()) for k in original
    }
    chunk = max(differences.values())
    # This is an optional, rejected execution transformation, not the reference
    # path. All actual diagnostics keep each sealed condition group intact.
    return {
        "graph_max_abs": difference,
        "response_max_abs": maximum,
        "used_path_passed": True,
        "cell_chunk_max_abs": chunk,
        "cell_chunk_by_tensor": differences,
        "cell_chunk_passed": chunk <= 2e-6,
        "cell_chunk_used": False,
        "cell_batch_policy": "sealed groups unchanged; never split/rebatch cells",
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
    peer = None
    if getattr(args, "peer_receipt", None) is not None:
        if sha(args.peer_receipt) != args.peer_receipt_sha256:
            raise ValueError("analysis peer receipt hash differs")
        peer = read(args.peer_receipt)
        if peer["manifest_sha256"] != args.manifest_sha256:
            raise ValueError("analysis peer manifest differs")
    observed = snapshot(os.getpid())
    if peer is None:
        admit(policy, observed)
    else:
        from cap40_analysis_admission import admit_analysis

        admit_analysis(policy, observed, int(os.environ["CUDA_VISIBLE_DEVICES"]), peer)
    write(
        args.output / "ADMISSION.json",
        {"passed": True, "peer": peer, "policy": policy, "observed": observed},
    )
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
                numeric = np.arange(rt.data.manifest.n_expression_genes)
                groups = {
                    "all_graph_genes": np.arange(rt.index.n_nodes),
                    "seen_expression_genes": rt.allowed_expression_ids,
                    "unseen_expression_genes": np.setdiff1d(numeric, rt.allowed_expression_ids),
                }
                for module, x in (("genept", obj.student.graph.identities()), ("graph", cached)):
                    values = npval(x)
                    for name, indices in groups.items():
                        m = Moments(x.shape[-1])
                        m.add(values[indices])
                        global_features[f"student.{module}.{name}"] = m
            schedule = select_schedule(manifest["schedule"], args.limit, args.gate_indices)
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
                            "query_panel_sha256": digest_json(g["queries"]),
                            "control_treatment": "control",
                            "paired_perturbation": g["condition_id"],
                            **m.result(),
                        }
                        basic_features.append(entry)
                        label = key + ":" + context + ":panel=" + digest_json(g["queries"])
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
                deep = None
                if g["deep_rows"]:
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
                            "independent_controls": len(
                                {r["control_row_id"] for r in g["deep_rows"]}
                            ),
                            "joint": float(loss),
                            "loss_terms": {k: float(v) for k, v in terms.items()},
                        }
                    )
                    deep_results.append(deep)
                if g["gradient_rows"] or (args.limit and g["deep_rows"]):
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
                    gradients.append(
                        {
                            "condition_id": g["condition_id"],
                            "cells": len(g["gradient_rows"] or g["deep_rows"][:4]),
                            **gradient,
                        }
                    )
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
            "scope": "gate_subset"
            if args.limit
            else manifest.get("analysis_scope", "full_cap40_train"),
            "conditions": len(schedule),
            "deep_conditions": len(deep_results),
            "gradient_conditions": len(gradients),
            "gradient_rows": sum(g["cells"] for g in gradients),
            "manifest_scope": manifest["scope"],
            "rows": len(rows_out),
            "deep_rows": sum(d["independent_truth_rows"] for d in deep_results),
            "centers": {
                h: {
                    "l2": float(getattr(obj, h + "_center").norm()),
                    "std": float(getattr(obj, h + "_center").std(unbiased=False)),
                    "min": float(getattr(obj, h + "_center").min()),
                    "max": float(getattr(obj, h + "_center").max()),
                }
                for h in HEADS
            },
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
                    "prediction_zero_fraction",
                    "prediction_mean",
                    "prediction_min",
                    "prediction_max",
                    "truth_mean",
                    "truth_min",
                    "truth_max",
                    "truth_negative_fraction",
                    "error_top1pct_fraction",
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
                    "Numeric train exclusions retained; unseen graph features are identity-only; "
                    "original 18 test scores reused separately."
                ),
                (
                    "Per-cell covariance samples up to128 hash-selected genes; "
                    "rank ceiling<=127; no prefix sampling."
                ),
                "SSL1 condition/node units are conditions/graph nodes, not cell replicates.",
                "Single Jurkat cell line cannot estimate cross-cell-line differences.",
                (
                    "BF16 cell-chunk transformation failed parity and is NOT used; "
                    "sealed grouping stays fixed."
                ),
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
    p.add_argument("mode", choices=("seal", "seal-medium", "run"))
    for name in ("run-root", "config", "data-root", "output", "manifest", "resource-queue"):
        p.add_argument(
            "--" + name, type=Path, required=name in ("run-root", "config", "data-root", "output")
        )
    p.add_argument("--manifest-sha256")
    p.add_argument("--peer-receipt", type=Path)
    p.add_argument("--peer-receipt-sha256")
    p.add_argument("--parent-manifest", type=Path)
    p.add_argument("--parent-manifest-sha256")
    p.add_argument("--arm", default="DP")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gate-indices", default="")
    args = p.parse_args()
    if args.mode == "seal":
        seal(args)
    elif args.mode == "seal-medium":
        seal_medium(args)
    else:
        from run_group import lock

        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if gpu not in ("0", "1"):
            raise ValueError("one isolated physical GPU per posthoc worker")
        with lock(Path(f"/data/yilangliu/GraD-Pert/runtime/v2-gpu-{gpu}.lock")):
            run(args)


if __name__ == "__main__":
    cli()
