"""Frozen original B0/O1 Student CLS audit on the retained R13 crossed panel.

Two fresh workers import their original training sources. Only aggregate scalars
and identity receipts are exported; default native decoding is counted honestly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
from itertools import combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace

from postfit_spectral_audit import (
    checked,
    digest,
    frozen_observation,
    load_panel,
    observed_cls,
    read,
    sha,
    spectral_statistics,
    verify_source,
    write_new,
)
from postfit_stochastic_audit import (
    control_only_reads,
    fingerprint,
    native_graph_reference,
    verify_analysis_source,
    zero_pkl,
)

SCHEMA = "postfit-matched-cls-audit-input-1"
RESULT_SCHEMA = "postfit-matched-cls-audit-result-1"
NAMES = ("B0", "O1")
VIEWS = ("main", "global0", "global1", "local0", "local1")
INPUT_HASHES = dict(
    parent="19c814aac4642ae74b6d9d3d8830d0ca24a2725c8ecd112c7165f6d8fb87b1d8",
    panel="d4455c25a767218513f6967d744eb1867ad7f4161762473f3ea806ba5fc76c31",
    matched_views="a76b57ca79fb9855a41bb1f3c9dabb989631bd24cbcf97fafb106b84d2dccb6f",
)
ENDPOINTS = {
    "B0": dict(
        commit="c7660ad44eee422e29c72fbf1c149d4827e04d75",
        checkpoint="44abff5326b47f9b010f2b1fc9c25998295905ce2f026c726298ec99561e5e5c",
        config="80f5ee1d6c4d826d9d2c023f304c08f1734b5a05acd28d565f7053e367117830",
        queue="a808fc8cab24fab412e5f93f644785e786cfe31206aa92c62cac2604f66aacd1",
        manifest="1b9e37341e0dd5d2eb490595a6cc5ad4e9599e61d5dc868c20f719dc698482c1",
        publication="765844dc4845423b97196b6b2ed75bdb5650f753c8081a1320f7a90b431fde72",
        run_id="nadig_jurkat-seed1-20261015T165647Z-b5e5d795f82b45e9915efca980ef7596",
    ),
    "O1": dict(
        commit="ca09419603e9b065706613f4b9bce4a78525803f",
        checkpoint="7109f9e7938db3acd0db12e47a54c3c10a647f221aba3e6a4b00cf7529c5c7e9",
        config="76d9c3d2ede6646405ffde1b583449b9ad7f4867436405ab914a692b3c197a07",
        queue="6ea1c0c24bacbbeb201a4ff38cc5779d7dae3eecd0f090c7b9b3eb64364d6dea",
        manifest="09edae36f499beed30ec9a1fabacb37da3312e940e734e27ea5357fc02fff6ee",
        publication="d78e56b7a2d6d9e23f1e062d4d7601b63a790fc72521601d18a831467f513a99",
        run_id="nadig_jurkat-seed1-20261026T140644Z-e109c924379d452e88b3194447eaf4e5",
    ),
}
PROTOCOL = dict(
    scope="retained_R13_crossed128_native_eval_Student_ResponseCLS",
    views=list(VIEWS),
    groups=32,
    cells_per_group=4,
    conditions=32,
    contexts=16,
    teacher="unobserved",
    decode="original_native_default_both_endpoints; decoder_calls_observed",
    graph="original_batch_graph_per_shallow_deep_group; no_fullaxis_substitute",
    reference="first_group_all5_independent_native_graph_response_RNG_2e-6",
    truth="control_whitelist_only; representative_truth_ID_metadata_only",
    relation_centering="separate main centroid; augmented shared Global centroid per context",
    relation_spectrum=(
        "native GroupedViewBank eigclip0; rank trace>max(E,1e-30)*1e-12; "
        "RV denominator max(E,1e-30)"
    ),
    global_spectrum="FP64 centered SVD no jitter; q=trace/E without floor; ceiling=min(D,N-1)",
)
FORBIDDEN = (
    "objective",
    "teacher",
    "projector",
    "backward",
    "autograd_grad",
    "optimizer_step",
    "statistics_commit",
)


def validate_spec(spec, *, server=True):
    if (
        set(spec)
        != {
            "schema",
            "analysis_source",
            "parent",
            "panel",
            "matched_views",
            "resource_queue",
            "data_root",
            "output",
            "endpoints",
            "protocol",
        }
        or spec["schema"] != SCHEMA
        or spec["protocol"] != PROTOCOL
    ):
        raise ValueError("matched CLS schema/protocol differs")
    if not isinstance(spec["endpoints"], list) or [
        e.get("name") for e in spec["endpoints"]
    ] != list(NAMES):
        raise ValueError("exact original B0/O1 endpoints required")
    paths, protected = [], [Path(spec["data_root"]).resolve(strict=True)]
    if spec["resource_queue"] != spec["endpoints"][1]["queue"]:
        raise ValueError("resource admission must use the exact original O1 queue")
    for key in ("parent", "panel", "matched_views", "resource_queue"):
        if set(spec[key]) != {"path", "sha256"}:
            raise ValueError("complete immutable file binding required")
        paths.append(checked(spec[key]))
        if key in INPUT_HASHES and spec[key]["sha256"] != INPUT_HASHES[key]:
            raise ValueError("retained panel/parent/recipe identity differs")
    for source in [spec["analysis_source"], *(e["source"] for e in spec["endpoints"])]:
        if set(source) != {"root", "commit", "publication", "publication_sha256"}:
            raise ValueError("complete source/publication binding required")
        if len(source["commit"]) != 40 or any(
            c not in "0123456789abcdef" for c in source["commit"]
        ):
            raise ValueError("full source commit required")
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
            "queue",
        }:
            raise ValueError("complete historical endpoint binding required")
        fixed = ENDPOINTS[endpoint["name"]]
        if any(
            (
                endpoint["source"]["commit"] != fixed["commit"],
                endpoint["source"]["publication_sha256"] != fixed["publication"],
                endpoint["checkpoint_sha256"] != fixed["checkpoint"],
                endpoint["config_sha256"] != fixed["config"],
                endpoint["run_manifest_sha256"] != fixed["manifest"],
                endpoint["queue"]["sha256"] != fixed["queue"],
            )
        ):
            raise ValueError("original endpoint source/config/checkpoint differs")
        root = Path(endpoint["run_root"]).resolve(strict=True)
        config = checked(dict(path=endpoint["config"], sha256=endpoint["config_sha256"]))
        if root.name != fixed["run_id"] or not config.is_relative_to(
            Path(endpoint["source"]["root"]).resolve() / "configs"
        ):
            raise ValueError("original run/config path differs")
        checked(dict(path=root / "run_manifest.json", sha256=endpoint["run_manifest_sha256"]))
        paths.extend((root, config, checked(endpoint["queue"])))
        protected.append(root)
    output = Path(spec["output"]).resolve()
    inputs = paths.copy()
    paths.extend((output, *protected))
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("fresh output overlaps protected source/data/run")
    if any(p == output or p.is_relative_to(output) for p in inputs):
        raise ValueError("output contains an immutable audit input")
    if server and any(not p.is_relative_to("/data/yilangliu") for p in paths):
        raise ValueError("scientific materialization is server-only")


def load_inputs(spec):
    panel = load_panel(spec)
    expected = read(checked(spec["matched_views"]))
    if len(panel["schedule"]) != 32 or len(expected) != 32:
        raise ValueError("complete32-group panel/recipe receipt required")
    keys = []
    contexts = {}
    for index, (group, receipt) in enumerate(zip(panel["schedule"], expected, strict=True)):
        context = tuple(group["context"])
        if group["index"] != index or len(group["rows"]) != 4:
            raise ValueError("sealed four-cell group order differs")
        if receipt["condition_id"] != group["condition_id"] or tuple(receipt["context"]) != context:
            raise ValueError("old recipe/group identity differs")
        if receipt["query_sha256"] != panel["query_sha256"] or receipt["controls_sha256"] != digest(
            [r["control_row_id"] for r in group["rows"]]
        ):
            raise ValueError("old recipe query/control order differs")
        key = (group["condition_id"], context)
        if key in keys:
            raise ValueError("duplicate condition/context group")
        keys.append(key)
        contexts.setdefault(context, []).append(group)
    if len({c[:3] for c in contexts}) != len(contexts):
        raise ValueError("GroupedViewBank context key would merge distinct batches")
    if len(contexts) != 16 or any(len(g) != 2 for g in contexts.values()):
        raise ValueError("retained16-context crossed population required")
    return panel, expected


def used_recipe_digest(recipes):
    """Hash only the bound recipe actually used, retaining full raw receipts."""
    fields = (
        "condition_id",
        "context",
        "query_sha256",
        "controls_sha256",
        "matched_cell_views_sha256",
    )
    return digest([{key: recipe[key] for key in fields} for recipe in recipes])


def validate_recipe(actual, expected):
    for key in ("context", "query_sha256", "controls_sha256", "matched_cell_views_sha256"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"actual matched expression recipe differs: {key}")


def bind_endpoint(spec, endpoint, identity):
    from checkpoint_endpoint import select_checkpoint

    root = Path(endpoint["run_root"]).resolve()
    fixed = ENDPOINTS[endpoint["name"]]
    if (
        identity["run_id"] != fixed["run_id"]
        or identity["data"] != read(checked(spec["parent"]))["data"]
    ):
        raise ValueError("original run or complete physical data identity differs")
    if identity["source"]["commit"] != fixed["commit"]:
        raise ValueError("training source identity differs")
    queue = read(checked(endpoint["queue"]))
    rows = [r for r in queue["rows"] if Path(r["plan"]["run_root"]).resolve() == root]
    if len(rows) != 1:
        raise ValueError("exact one original queue/run binding required")
    row, plan = rows[0], rows[0]["plan"]
    if (
        row["name"] != endpoint["name"]
        or queue["source_commit"] != fixed["commit"]
        or plan["run_id"] != identity["run_id"]
        or Path(plan["config"]).resolve() != Path(endpoint["config"]).resolve()
        or plan["config_sha256"] != endpoint["config_sha256"]
    ):
        raise ValueError("original queue/config/source binding differs")
    cp, cp_endpoint = select_checkpoint(root, "last")
    if cp["sha256"] != endpoint["checkpoint_sha256"] or cp_endpoint != dict(
        role="last", epoch=4, training_stage="joint", full_training_epochs=4
    ):
        raise ValueError("original epoch4 joint last checkpoint required")
    return cp, cp_endpoint


def checkpoint_journal_digest(endpoint, checkpoint):
    """Seal parsed retained journal bytes without loading a model/checkpoint."""
    path = Path(endpoint["run_root"]) / "fit/epoch_state.json"
    data = path.read_bytes()
    journal = json.loads(data)
    if (
        type(journal.get("epoch")) is not int
        or journal["epoch"] != 4
        or journal.get("last") != checkpoint
    ):
        raise ValueError("checkpoint journal is not the completed epoch4 retained last")
    return hashlib.sha256(data).hexdigest()


def revalidate_endpoints(spec, results):
    """Rebind all completed workers to fresh journal, streamed cp and source evidence."""
    if [row["endpoint"] for row in results] != spec["endpoints"]:
        raise ValueError("final checkpoint closure endpoint bank differs")
    proofs = {}
    for endpoint, result in zip(spec["endpoints"], results, strict=True):
        manifest_path = checked(
            dict(
                path=Path(endpoint["run_root"]) / "run_manifest.json",
                sha256=endpoint["run_manifest_sha256"],
            )
        )
        identity = read(manifest_path)
        source = verify_source(endpoint["source"])
        if (
            source != result["source"]
            or identity["source"] != source
            or identity["data"] != result["training_data"]
        ):
            raise ValueError("original source/publication/run identity changed before closure")
        # Native selection streams the actual checkpoint bytes and checks the
        # selected epoch/config; this must happen again after the other worker.
        checkpoint, phase = bind_endpoint(spec, endpoint, identity)
        journal_sha = checkpoint_journal_digest(endpoint, checkpoint)
        if (
            checkpoint != result["checkpoint"]
            or phase != result["checkpoint_endpoint"]
            or journal_sha != result["checkpoint_journal_sha256"]
        ):
            raise ValueError("retained checkpoint/journal changed after worker completion")
        proofs[endpoint["name"]] = dict(
            run_manifest_sha256=endpoint["run_manifest_sha256"],
            checkpoint_sha256=checkpoint["sha256"],
            checkpoint_journal_sha256=journal_sha,
            checkpoint_endpoint=phase,
            source=source,
            streamed_checkpoint_rechecked=True,
        )
    return proofs


@contextmanager
def observed_paths(runtime):
    """Count actual encoders/graphs/decoder heads and forbid update/target paths."""
    import torch

    counts = dict.fromkeys((*FORBIDDEN, "student_encode", "native_graph", "decoder"), 0)
    handles, patched = [], []

    def hook(kind, forbidden=True):
        def invoke(*_a, **_k):
            counts[kind] += 1
            if forbidden:
                raise ValueError(f"forbidden matched audit path: {kind}")

        return invoke

    def patch(owner, name, kind):
        if not hasattr(owner, name):
            raise ValueError(f"required native update entry missing: {name}")
        owned = name in vars(owner)
        patched.append((owner, name, owned, vars(owner).get(name)))
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
        for name in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node"):
            for module in getattr(obj.student, name).modules():
                handles.append(module.register_forward_pre_hook(hook("projector")))
        # Both original GELU/Softplus endpoints call their final prediction head.
        handles.append(obj.student.prediction[-1].register_forward_pre_hook(hook("decoder", False)))
        patch(runtime.optimizer, "step", "optimizer_step")
        patch(obj, "commit_statistics", "statistics_commit")
        patch(torch.autograd, "backward", "backward")
        patch(torch.autograd, "grad", "autograd_grad")
        yield counts
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, owned, value in reversed(patched):
            if owned:
                setattr(owner, name, value)
            else:
                delattr(owner, name)


@contextmanager
def protected_observation(runtime):
    """Reuse native restoration; additionally protect objective buffers/clocks."""

    def snapshot():
        obj = runtime.objective
        return fingerprint(
            (
                obj.state_dict(),
                {
                    k: getattr(obj, k, None)
                    for k in ("_temperature_updates", "auxiliary_rng_counter")
                },
            )
        )

    before = snapshot()
    original = None
    proof = {}
    try:
        with frozen_observation(runtime) as proof:
            yield proof
    except BaseException as error:
        original = error
        raise
    finally:
        after = snapshot()
        proof["objective_state_before"] = before
        proof["objective_state_after"] = after
        if before != after:
            raise ValueError(
                f"protected objective state/clocks changed; original={original!r}"
            ) from original


def native_encode(model, args, kwargs):
    """No decode kw: c766 retains its original default decoder semantics."""
    return model.encode_response(*args, **kwargs)


def reference_parity(model, args, kwargs, reference_args, graph_difference):
    import torch

    devices = [args[1].device.index or 0] if args[1].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state() if devices else None
        original = native_encode(model, reference_args, kwargs)
        after = fingerprint(
            (torch.get_rng_state(), torch.cuda.get_rng_state() if devices else None)
        )
        torch.set_rng_state(cpu)
        if devices:
            torch.cuda.set_rng_state(cuda)
        replay = native_encode(model, args, kwargs)
        replay_after = fingerprint(
            (torch.get_rng_state(), torch.cuda.get_rng_state() if devices else None)
        )
    if set(original) != set(replay):
        raise ValueError("native response output keys differ")
    maximum = max(float((original[k] - replay[k]).abs().max()) for k in original)
    row = dict(
        graph_max_abs=graph_difference,
        response_max_abs=maximum,
        final_rng_equal=after == replay_after,
        used_path_passed=True,
        cell_chunk_used=False,
        native_graph_reference=True,
    )
    validate_parity(row)
    return row


def validate_parity(row):
    if any(
        row.get(k) is not v
        for k, v in dict(
            final_rng_equal=True,
            used_path_passed=True,
            cell_chunk_used=False,
            native_graph_reference=True,
        ).items()
    ):
        raise ValueError("native reference/RNG/grouping parity proof missing")
    if any(
        isinstance(row.get(k), bool)
        or not isinstance(row.get(k), (int, float))
        or not math.isfinite(row[k])
        or not 0 <= row[k] <= 2e-6
        for k in ("graph_max_abs", "response_max_abs")
    ):
        raise ValueError("native graph/response parity exceeds2e-6")


def prepare_recipes(runtime, panel, expected, admission):
    """Validate every old actual recipe on CPU before any model forward."""
    from representation_panel import MatchedCellViews, assemble_control_only

    matched, receipts = MatchedCellViews(), []
    for group, old in zip(panel["schedule"], expected, strict=True):
        admission()
        deep = assemble_control_only(
            runtime, group["rows"], group["queries"], group["view_seed"], deep=True
        )
        if deep.graph.ids[deep.query_positions].tolist() != group["queries"]:
            raise ValueError("native ordered query recipe differs")
        _, actual = matched.bind(deep, group["rows"])
        validate_recipe(actual, old)
        receipts.append(dict(condition_id=group["condition_id"], **actual))
    return receipts


def collect_inputs(runtime, group, batch, *, first=False):
    obj, model = runtime.objective, runtime.objective.student
    if first:
        cached, reference, difference = native_graph_reference(obj, model, batch)
    else:
        cached = obj._batch_graph(model, batch)
        reference, difference = None, None
    graph, condition = cached
    if batch.cell_views and len(batch.cell_views) != 4:
        raise ValueError("exact native2G+2L recipe required")
    positions = [None] if not batch.cell_views else [v.positions for v in batch.cell_views]
    result = {}
    for i, p in enumerate(positions):
        name = "main" if p is None else f"global{i}" if i < 2 else f"local{i - 2}"
        args = (
            obj._select_graph(
                graph, batch.query_positions if p is None else batch.query_positions[p]
            ),
            batch.control if p is None else batch.control[:, p],
            condition,
        )
        if p is not None:
            args += (batch.cell_views[i].mask,)
        kwargs = obj.response_metadata(batch, p)
        parity = None
        if first:
            rg, rc = reference
            reference_args = (
                obj._select_graph(
                    rg, batch.query_positions if p is None else batch.query_positions[p]
                ),
                args[1],
                rc,
                *args[3:],
            )
            parity = reference_parity(model, args, kwargs, reference_args, difference)
        before = fingerprint((args, kwargs))
        values = observed_cls(model, partial(native_encode, model, args, kwargs))
        after = fingerprint((args, kwargs))
        if before != after:
            raise ValueError("native cached encoder inputs mutated")
        result[name] = (values, parity, before)
    if len(group["rows"]) != 4 or any(v[0].shape[0] != 4 for v in result.values()):
        raise ValueError("native four-cell group changed")
    return result


def expected_pairs(panel):
    from collections import defaultdict

    contexts = defaultdict(list)
    for group in panel["schedule"]:
        for row in group["rows"]:
            contexts[tuple(str(row[k]) for k in ("cell_line", "cell_type", "context_id"))].append(
                row
            )
    result = {}
    for context, rows in contexts.items():
        counts = dict(
            same_sample=len(rows),
            same_condition_different_control=0,
            different_condition_same_control=0,
            different_condition_different_control=0,
        )
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                p, c = (
                    a["condition_id"] == b["condition_id"],
                    a["control_row_id"] == b["control_row_id"],
                )
                label = (
                    "same_condition_different_control"
                    if p and not c
                    else "different_condition_same_control"
                    if not p and c
                    else "different_condition_different_control"
                    if not p and not c
                    else None
                )
                if label:
                    counts[label] += 1
        for va, vb in [("main", "main"), *combinations_with_replacement(VIEWS[1:], 2)]:
            for label, count in counts.items():
                if va == vb and label == "same_sample":
                    continue
                result[(*context, va, vb, label)] = count
    return result


def summarize(panel, pieces):
    import torch
    from grouped_view_diagnostics import GroupedViewBank

    main_bank, augmented_bank = GroupedViewBank(), GroupedViewBank()
    for group, features in zip(panel["schedule"], pieces, strict=True):
        if set(features) != set(VIEWS):
            raise ValueError("exact main/2G/2L feature banks required")
        for view in VIEWS:
            bank = main_bank if view == "main" else augmented_bank
            bank.add("ssl2_response_cls", "student", view, group["rows"], features[view].numpy())
    main, augmented = main_bank.result(), augmented_bank.result()
    relations = dict(
        schema="matched-cls-separate-main-augmented-1",
        pairs=main["pairs"] + augmented["pairs"],
        features=main["features"] + augmented["features"],
        limits=main["limits"],
    )
    for row in relations["pairs"]:
        a, b = row["view_a"], row["view_b"]
        row["view_family"] = "MM" if a == b == "main" else row["view_family"]
    return dict(
        statistics={v: spectral_statistics(torch.cat([p[v] for p in pieces])) for v in VIEWS},
        relations=relations,
    )


def validate_statistics(row):
    if (
        type(row.get("n")) is not int
        or row["n"] != 128
        or type(row.get("dimension")) is not int
        or row["dimension"] < 1
    ):
        raise ValueError("complete128-row scalar dimensions required")
    ceiling = min(row["dimension"], 127)
    if row.get("rank_ceiling") != ceiling or row.get("covariance_denominator") != "N":
        raise ValueError("population covariance/rank ceiling differs")
    for key in ("trace", "total_energy", "mean_energy", "effective_rank", "rho"):
        value = row.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("finite nonnegative scalar statistics required")
    trace, energy, rank = row["trace"], row["total_energy"], row["effective_rank"]
    if not math.isclose(trace + row["mean_energy"], energy, rel_tol=1e-10, abs_tol=1e-24):
        raise ValueError("centered/mean energy decomposition differs")
    if energy == 0:
        if row.get("q") is not None:
            raise ValueError("zero total energy requires NA q")
    elif (
        not isinstance(row.get("q"), (int, float))
        or isinstance(row["q"], bool)
        or not math.isfinite(row["q"])
        or not -1e-12 <= row["q"] <= 1 + 1e-12
        or not math.isclose(row["q"], trace / energy, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError("relative variance scalar differs from trace/energy")
    if (
        rank > ceiling + 1e-10
        or (trace == 0 and rank != 0)
        or (trace > 0 and rank < 1 - 1e-10)
        or not math.isclose(row["rho"], rank / ceiling, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError("effective rank/rho scalar differs")


def validate_result(result, panel):
    if result.get("schema") != RESULT_SCHEMA or result.get("protocol") != PROTOCOL:
        raise ValueError("result schema/protocol differs")
    if (
        result["updates"] != 0
        or result["raw_activations_saved"] is not False
        or result.get("scientific_complete") is not False
        or result.get("teacher") is not None
        or result.get("corrected_rank") is not None
    ):
        raise ValueError("audit update/export contract differs")
    endpoint = result["endpoint"]
    if result.get("checkpoint") != dict(
        epoch=4, file="epoch-0004.pt", sha256=endpoint["checkpoint_sha256"]
    ) or result.get("checkpoint_endpoint") != dict(
        role="last", epoch=4, training_stage="joint", full_training_epochs=4
    ):
        raise ValueError("result checkpoint differs from original epoch4 joint last")
    state = result["state"]
    if (
        state.get("unchanged") is not True
        or state["before"] != state["after"]
        or state["objective_state_before"] != state["objective_state_after"]
    ):
        raise ValueError("protected state proof differs")
    counts = result["observed_calls"]
    if (
        set(counts) != set((*FORBIDDEN, "student_encode", "native_graph", "decoder"))
        or any(counts[k] != 0 for k in FORBIDDEN)
        or counts["student_encode"] != 170
        or counts["decoder"] != 170
        or counts["native_graph"] <= 0
    ):
        raise ValueError("actual execution counters differ from160 observations+10 references")
    if set(result["parity"]) != set(VIEWS):
        raise ValueError("complete first-group native parity required")
    for row in result["parity"].values():
        validate_parity(row)
    if (
        result["zero_pkl"] != dict(checked=True, count=0)
        or result["control_expression_reads"]["forbidden"] != 0
        or result["control_expression_reads"]["calls"] != 96
        or result["control_expression_reads"]["rows"] != 384
    ):
        raise ValueError("output/truth-read closure failed")
    if set(result["statistics"]) != set(VIEWS) or any(
        r["n"] != 128 or r["rank_ceiling"] != min(r["dimension"], 127)
        for r in result["statistics"].values()
    ):
        raise ValueError("complete128-row scalar bank required")
    for row in result["statistics"].values():
        validate_statistics(row)
    observed = {}
    for row in result["relations"]["pairs"]:
        key = tuple(
            str(row[k])
            for k in ("cell_line", "cell_type", "context_id", "view_a", "view_b", "stratum")
        )
        if key in observed:
            raise ValueError("duplicate relation row")
        observed[key] = row["pair_count"]
        if not row["pair_count"] and (
            row["status"] != "undefined:no_eligible_pairs" or row["cosine_pair_mean"] is not None
        ):
            raise ValueError("empty relation must retain NA")
    if observed != expected_pairs(panel):
        raise ValueError("actual eligible pair population differs")
    if (
        len(result["inputs"]) != 32
        or [r["group"] for r in result["inputs"]] != list(range(32))
        or any(set(r["cache_sha256"]) != set(VIEWS) for r in result["inputs"])
    ):
        raise ValueError("complete native input/cache identities required")
    if len(result["matched_views"]) != 32 or result["matched_views_sha256"] != used_recipe_digest(
        result["matched_views"]
    ):
        raise ValueError("actual recipe receipts incomplete")
    for group, actual, expected in zip(
        panel["schedule"], result["matched_views"], result["expected_views"], strict=True
    ):
        if actual["condition_id"] != group["condition_id"]:
            raise ValueError("recipe/group identity differs")
        validate_recipe(actual, expected)
    json.dumps(result, allow_nan=False)


def worker(spec, endpoint):
    import torch
    from cap40_posthoc import runtime_args
    from representation_panel import MatchedCellViews, assemble_control_only, validate_panel_runtime
    from shared_gpu import admit, snapshot

    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.runtime import _move_batch_tree

    source = verify_source(endpoint["source"])
    analysis = verify_analysis_source(spec["analysis_source"])
    _, config, identity, prepare, method_root = runtime_args(
        SimpleNamespace(
            method_sha=endpoint["source"]["commit"],
            config=Path(endpoint["config"]),
            run_root=Path(endpoint["run_root"]),
        )
    )
    if method_root != Path(endpoint["source"]["root"]).resolve() or identity["source"] != source:
        raise ValueError("original native imports/source identity differs")
    cp, cp_endpoint = bind_endpoint(spec, endpoint, identity)
    checkpoint = Path(endpoint["run_root"]) / "fit" / cp["file"]
    journal_sha = checkpoint_journal_digest(endpoint, cp)
    panel, expected = load_inputs(spec)
    policy = read(checked(spec["resource_queue"]))["shared_gpu_policy"]
    owner = read(Path(spec["output"]) / "OWNER.json")

    def admission():
        observed = snapshot(owner["pid"])
        admit(policy, observed)
        return observed

    observed = admission()
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.cuda.current_stream() != torch.cuda.default_stream():
        raise ValueError("native default CUDA stream0 required")
    output = Path(spec["output"]) / endpoint["name"]
    output.mkdir()
    write_new(output / "ADMISSION.json", dict(policy=policy, observed=observed, gpu=0))
    try:
        with prepare(
            config,
            data_root=Path(spec["data_root"]),
            run_seed=1,
            device=torch.device("cuda:0"),
            purpose="evaluation",
        ) as runtime:
            if runtime.identity != identity["data"]:
                raise ValueError("runtime physical data identity differs")
            validate_panel_runtime(panel, runtime)
            load_evaluation_checkpoint(
                checkpoint,
                runtime.objective,
                training_identity=identity,
                checkpoint_sha256=cp["sha256"],
            )
            options = runtime.objective.student.options
            if (
                options.attention != "relay_full"
                or options.control_conditioned_graph
                or bool(options.fixed_gene_order) != (endpoint["name"] == "O1")
                or bool(options.random_gene_order) != (endpoint["name"] == "B0")
            ):
                raise ValueError("original B0/O1 eval architecture/order policy differs")
            pieces, recipes, parity, input_rows = [], [], {}, []
            group_seconds = []
            matched = MatchedCellViews()
            with (
                protected_observation(runtime) as state,
                observed_paths(runtime) as calls,
                control_only_reads(runtime, panel) as reads,
                torch.autocast("cuda", dtype=torch.bfloat16),
            ):
                prepared = prepare_recipes(runtime, panel, expected, admission)
                torch.random.default_generator.manual_seed(panel["seed"])
                torch.cuda.manual_seed(panel["seed"])
                for index, group in enumerate(panel["schedule"]):
                    started = time.monotonic()
                    admission()
                    shallow = _move_batch_tree(
                        assemble_control_only(
                            runtime, group["rows"], group["queries"], group["view_seed"], deep=False
                        ),
                        runtime.device,
                    )
                    deep = _move_batch_tree(
                        assemble_control_only(
                            runtime, group["rows"], group["queries"], group["view_seed"], deep=True
                        ),
                        runtime.device,
                    )
                    if any(
                        b.graph.ids[b.query_positions].tolist() != group["queries"]
                        for b in (shallow, deep)
                    ):
                        raise ValueError("native ordered query recipe differs")
                    deep, recipe = matched.bind(deep, group["rows"])
                    validate_recipe(recipe, expected[index])
                    validate_recipe(recipe, prepared[index])
                    recipes.append(dict(condition_id=group["condition_id"], **recipe))
                    values = collect_inputs(runtime, group, shallow, first=index == 0)
                    values.update(collect_inputs(runtime, group, deep, first=index == 0))
                    if set(values) != set(VIEWS):
                        raise ValueError("native view bank differs")
                    pieces.append({k: v[0] for k, v in values.items()})
                    input_rows.append(
                        dict(
                            group=index,
                            raw=dict(
                                shallow_topology=fingerprint(shallow.graph),
                                deep_topology=fingerprint(deep.graph),
                                control=fingerprint(shallow.control),
                                deep_control=fingerprint(deep.control),
                                ordered_queries=digest(group["queries"]),
                            ),
                            cache_sha256={k: v[2] for k, v in values.items()},
                        )
                    )
                    if index == 0:
                        parity = {k: v[1] for k, v in values.items()}
                    group_seconds.append(time.monotonic() - started)
                    print(
                        json.dumps(
                            dict(
                                endpoint=endpoint["name"],
                                groups_completed=index + 1,
                                total_groups=32,
                                elapsed_group_seconds=group_seconds[-1],
                                observed_calls=calls,
                            )
                        ),
                        flush=True,
                    )
            aggregate = summarize(panel, pieces)
        validate_spec(spec)
        if (
            sha(checkpoint) != cp["sha256"]
            or checkpoint_journal_digest(endpoint, cp) != journal_sha
            or verify_source(endpoint["source"]) != source
            or verify_analysis_source(spec["analysis_source"]) != analysis
        ):
            raise ValueError("immutable source/checkpoint drift")
        result = dict(
            schema=RESULT_SCHEMA,
            endpoint=endpoint,
            source=source,
            analysis_source=analysis,
            training_data=identity["data"],
            checkpoint=cp,
            checkpoint_endpoint=cp_endpoint,
            checkpoint_journal_sha256=journal_sha,
            protocol=PROTOCOL,
            panel_sha256=spec["panel"]["sha256"],
            panel_schedule_sha256=panel["panel_sha256"],
            query_sha256=panel["query_sha256"],
            inputs=input_rows,
            matched_views=recipes,
            expected_views=expected,
            matched_views_sha256=used_recipe_digest(recipes),
            state=state,
            parity=parity,
            observed_calls=calls,
            control_expression_reads=reads,
            updates=0,
            raw_activations_saved=False,
            teacher=None,
            corrected_rank=None,
            group_seconds=group_seconds,
            scientific_complete=False,
            **aggregate,
        )
        result["zero_pkl"] = zero_pkl(output)
        validate_result(result, panel)
        result["artifact_sha256"] = artifact_closure(output, endpoint=True, pending=True)
        write_new(output / "COMPLETE.json", result)
    except BaseException as error:
        write_new(
            output / "FAILURE.json",
            dict(
                type=type(error).__name__,
                error=str(error),
                updates=0,
                state=locals().get("state"),
                observed_calls=locals().get("calls"),
            ),
        )
        raise


def combine(spec, results):
    panel, expected = load_inputs(spec)
    if [r["endpoint"]["name"] for r in results] != list(NAMES):
        raise ValueError("complete ordered B0/O1 bank required")
    for endpoint, result in zip(spec["endpoints"], results, strict=True):
        if (
            result["endpoint"] != endpoint
            or result["analysis_source"]["commit"] != spec["analysis_source"]["commit"]
        ):
            raise ValueError("sealed endpoint/analysis identity differs")
        if (
            result["expected_views"] != expected
            or result["panel_sha256"] != spec["panel"]["sha256"]
            or result["panel_schedule_sha256"] != panel["panel_sha256"]
            or result["query_sha256"] != panel["query_sha256"]
        ):
            raise ValueError("result differs from sealed original panel/recipe")
        manifest = read(
            checked(
                dict(
                    path=Path(endpoint["run_root"]) / "run_manifest.json",
                    sha256=endpoint["run_manifest_sha256"],
                )
            )
        )
        if result["source"] != manifest["source"] or result["training_data"] != manifest["data"]:
            raise ValueError("result source/data differs from original sealed run")
        validate_result(result, panel)
    baseline = results[0]
    for result in results[1:]:
        if [r["raw"] for r in result["inputs"]] != [r["raw"] for r in baseline["inputs"]]:
            raise ValueError("actual raw control/query/topology differs across endpoints")
        for key in (
            "panel_sha256",
            "panel_schedule_sha256",
            "query_sha256",
            "matched_views_sha256",
            "training_data",
        ):
            if result[key] != baseline[key]:
                raise ValueError("actual matched input population or physical data differs")
    return dict(
        schema="postfit-matched-cls-audit-complete-1",
        endpoints=list(NAMES),
        updates=0,
        scientific_complete=False,
        native_gpu_parity="observed_in_endpoint_receipts",
        panel_sha256=baseline["panel_sha256"],
        matched_views_sha256=baseline["matched_views_sha256"],
    )


def artifact_closure(output, *, endpoint=False, pending=False):
    output = Path(output)
    zero_pkl(output)
    wanted = (
        {"ADMISSION.json", "COMPLETE.json"}
        if endpoint
        else {
            "INPUT.json",
            "OWNER.json",
            "B0.log",
            "O1.log",
            "B0/ADMISSION.json",
            "B0/COMPLETE.json",
            "O1/ADMISSION.json",
            "O1/COMPLETE.json",
            "COMPLETE.json",
        }
    )
    if pending:
        wanted.remove("COMPLETE.json")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual != wanted:
        raise ValueError(
            f"artifact closure differs: missing={sorted(wanted - actual)}; "
            f"extra={sorted(actual - wanted)}"
        )
    return {name: sha(output / name) for name in sorted(actual) if name != "COMPLETE.json"}


def write_root_complete(spec, results, result):
    """Publish success only after every original endpoint is freshly revalidated."""
    output = Path(spec["output"])
    result["final_endpoint_verification"] = revalidate_endpoints(spec, results)
    result["artifact_sha256"] = artifact_closure(output, pending=True)
    write_new(output / "COMPLETE.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--spec-sha256")
    parser.add_argument("--endpoint", choices=NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--print-protocol", action="store_true")
    args = parser.parse_args()
    if args.print_protocol:
        print(json.dumps(PROTOCOL, indent=2))
        return
    if args.spec is None or args.spec_sha256 is None:
        parser.error("--spec and --spec-sha256 required")
    if sha(args.spec) != args.spec_sha256:
        raise ValueError("sealed input spec hash differs")
    spec = read(args.spec)
    validate_spec(spec)
    if Path(spec["analysis_source"]["root"]).resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError("script outside sealed analysis checkout")
    if (
        os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0")
    ):
        raise ValueError("GPU0 allocator/deterministic runtime contract differs")
    from run_group import lock
    from shared_gpu import descendant, parent_map, process_info

    output = Path(spec["output"])
    if args.endpoint:
        if sha(output / "INPUT.json") != args.spec_sha256:
            raise ValueError("worker input differs from coordinator")
        owner = read(output / "OWNER.json")
        if process_info(owner["pid"]) != owner or not descendant(
            os.getpid(), owner["pid"], parent_map()
        ):
            raise ValueError("worker lacks exact live coordinator/GPU lease ownership")
        worker(spec, next(e for e in spec["endpoints"] if e["name"] == args.endpoint))
        return
    verify_analysis_source(spec["analysis_source"])
    deadline = time.monotonic() + 10
    owner = None
    while time.monotonic() < deadline:
        candidate = process_info(os.getpid())
        if candidate is not None and candidate["ppid"] == 1:
            owner = candidate
            break
        time.sleep(0.1)
    if owner is None:
        raise ValueError("detached coordinator not yet adopted by PPID1")
    with lock(Path("/data/yilangliu/GraD-Pert/runtime/v2-gpu-0.lock")):
        output.mkdir(parents=True, exist_ok=False)
        (output / "INPUT.json").write_bytes(args.spec.read_bytes())
        write_new(output / "OWNER.json", owner)
        try:
            for endpoint in spec["endpoints"]:
                env = dict(
                    os.environ,
                    PYTHONPATH=str(Path(endpoint["source"]["root"]) / "src"),
                    CUDA_VISIBLE_DEVICES="0",
                )
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
            results = [read(output / n / "COMPLETE.json") for n in NAMES]
            result = combine(spec, results)
            validate_spec(spec)
            if sha(args.spec) != args.spec_sha256:
                raise ValueError("spec drift before completion")
            verify_analysis_source(spec["analysis_source"])
            result["spec_sha256"] = args.spec_sha256
            result["result_sha256"] = {n: sha(output / n / "COMPLETE.json") for n in NAMES}
            result["zero_pkl"] = zero_pkl(output)
            write_root_complete(spec, results, result)
        except BaseException as error:
            write_new(
                output / "FAILURE.json",
                dict(type=type(error).__name__, error=str(error), updates=0),
            )
            raise


if __name__ == "__main__":
    main()
