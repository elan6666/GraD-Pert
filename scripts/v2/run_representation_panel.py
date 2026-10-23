"""Server-only immutable-checkpoint mechanism probe, supplemental to medium.

Use the original training checkout on PYTHONPATH. No fit, gradient update,
test truth, or prediction-score selection is performed by this command.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from cap40_posthoc import (
    Collector,
    csv_write,
    main_response,
    npval,
    read,
    runtime_args,
    server_path,
    sha,
    source_identity,
    static_graph,
    verify_cache,
    write,
)
from checkpoint_endpoint import select_checkpoint
from grouped_view_diagnostics import GroupedViewBank
from representation_panel import (
    MatchedCellViews,
    assemble_control_only,
    panel_schedule,
    validate_panel_runtime,
)


def add_prototype_rows(collector, bank):
    obj = collector.objective
    for label, logits in collector.logits.values():
        role, head, view = label.split(".")
        if head not in ("ssl1_cls", "ssl2_cls"):
            continue
        view_index = int(view[4:])
        view = f"global{view_index}" if view_index < 2 else f"local{view_index - 2}"
        scaled = (
            logits.float() / 0.1
            if role == "student"
            else (logits.float() - getattr(obj, head + "_center")) / obj.teacher_temperature
        )
        unit = "control_condition"
        records = collector.records
        if head == "ssl1_cls" and not obj.student.options.control_conditioned_graph:
            unit = "perturbation_condition"
            records = [
                {
                    **records[
                        next(
                            i
                            for i, c in enumerate(collector.batch.condition_index.tolist())
                            if c == j
                        )
                    ],
                    "context_id": "static_graph",
                    "cell_type": "gene_identity_graph",
                    "control_row_id": "",
                }
                for j in range(len(logits))
            ]
        bank.add(head, role, view, records, npval(scaled), unit=unit)


def grouped_main_result(bank):
    """Label main-path interventions independently of augmented GG/GL/LL pairs."""
    result = bank.result()
    for row in result["pairs"]:
        if row["view_a"] != "main" or row["view_b"] != "main":
            raise ValueError("crossed main bank contains an augmented view")
        row["view_family"] = "main/main"
    return result


def prototype_result(bank, objective):
    result = bank.result()
    active = {
        "ssl1_cls": bool(objective.lambda1 and objective.weights[0][0]),
        "ssl2_cls": bool(objective.lambda2 and objective.weights[1][0]),
    }
    result["head_status"] = {}
    for head, enabled in active.items():
        rows = [r for r in result["pairs"] if r["branch"] == head]
        if enabled and not any(r["pair_count"] for r in rows):
            raise ValueError("active prototype head has no observed paired diagnostics: " + head)
        if not enabled and rows:
            raise ValueError("disabled head unexpectedly produced prototype diagnostics")
        result["head_status"][head] = (
            "observed" if enabled else "unavailable:ssl_disabled_in_checkpoint_stage_or_head"
        )
    result["status"] = (
        "completed" if any(active.values()) else "unavailable:no_active_cls_distillation_head"
    )
    return result


def collect_panel_views(objective, batch, records, seed, views, prototypes):
    """Observe fixed probes without inventing their unavailable global KoLeo loss."""
    from gradpert.training.v2.diagnostic_objective import diagnostic_objective

    collector = Collector(batch, records, seed, stats_backend="fast")
    collector.install(objective)
    try:
        _, _, scope = diagnostic_objective(objective, batch)
        collector.complete_feature_views()
        collector.add_to_view_bank(views)
        add_prototype_rows(collector, prototypes)
        return scope
    finally:
        collector.close()
        objective.pending.clear()


def panel_parent_binding(parent, identity, queue, args):
    """Bind frozen population-auxiliary metadata to the exact queued endpoint."""
    from generate_representation_round3 import SCHEMA
    from generate_representation_round4 import SCHEMA as CR_SCHEMA
    from run_fgpo_medium import au_parent_binding, cr_parent_binding

    rows = [
        row
        for row in queue["rows"]
        if Path(row["plan"]["run_root"]).resolve() == args.run_root.resolve()
    ]
    if len(rows) != 1:
        raise ValueError("panel requires one exact resource-queue run binding")
    row = rows[0]
    plan = row["plan"]
    if (
        plan["run_id"] != identity["run_id"]
        or Path(plan["config"]).resolve() != args.config.resolve()
        or plan["config_sha256"] != sha(args.config)
        or queue["source_commit"] != args.method_sha
        or identity["source"]["commit"] != args.method_sha
    ):
        raise ValueError("panel queue/run/config binding differs")
    if row["name"] in ("R3-AUB", "R3-AUS", "R4-CR"):
        spec = queue["medium_analysis"]
        family = CR_SCHEMA if row["name"] == "R4-CR" else SCHEMA
        if (
            queue.get("family") != family
            or Path(spec["parent_manifest"]).resolve() != args.parent_manifest.resolve()
            or spec["parent_manifest_sha256"] != args.parent_manifest_sha256
        ):
            raise ValueError("population auxiliary panel requires its exact original family parent")
        bind_parent = cr_parent_binding if row["name"] == "R4-CR" else au_parent_binding
        bound = bind_parent(
            parent,
            identity,
            arm=row["name"],
            config=args.config,
            parent_path=args.parent_manifest,
            parent_sha256=args.parent_manifest_sha256,
        )
        return bound["training_protocol_binding"]
    if parent["data"] != identity["data"]:
        raise ValueError("medium parent and checkpoint data differ")
    return None


def execute(args):
    from capacity_probe import validation_state
    from prototype_pair_diagnostics import PrototypePairBank
    from response_stage_diagnostics import ResponseStageObserver
    from shared_gpu import admit, snapshot

    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.runtime import _move_batch_tree

    for path in (
        args.output,
        args.run_root,
        args.data_root,
        args.parent_manifest,
        args.resource_queue,
    ):
        server_path(path)
    if sha(args.parent_manifest) != args.parent_manifest_sha256:
        raise ValueError("frozen medium parent identity differs")
    if sha(args.resource_queue) != args.resource_queue_sha256:
        raise ValueError("resource policy receipt identity differs")
    parent = read(args.parent_manifest)
    panel = panel_schedule(
        parent, seed=args.seed, conditions=args.conditions, controls=args.controls
    )
    torch, config, identity, prepare, _ = runtime_args(args)
    analysis_source = source_identity(Path(__file__).resolve().parents[2])
    queue = read(args.resource_queue)
    parent_binding = panel_parent_binding(parent, identity, queue, args)
    cp, endpoint = select_checkpoint(args.run_root, args.checkpoint_role)
    checkpoint = args.run_root / "fit" / cp["file"]
    if sha(checkpoint) != cp["sha256"]:
        raise ValueError("checkpoint hash differs")
    if args.output.exists():
        raise ValueError("fresh immutable diagnostic output required")
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise ValueError("allocator contract missing")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic probe requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    policy = queue["shared_gpu_policy"]
    observed = snapshot(os.getpid())
    admit(policy, observed)
    torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    write(args.output / "ADMISSION.json", dict(passed=True, policy=policy, observed=observed))
    write(args.output / "panel.json", panel)
    started = time.perf_counter()
    stage_rows = {name: [] for name in ("cell_statistics", "group_statistics", "pair_statistics")}
    matched_cell_views, view_receipts, diagnostic_scopes = MatchedCellViews(), [], []
    views, prototypes, main_views = GroupedViewBank(), PrototypePairBank(), GroupedViewBank()
    with prepare(
        config,
        data_root=args.data_root,
        run_seed=1,
        device=torch.device("cuda:0"),
        purpose="evaluation",
    ) as rt:
        if rt.identity != identity["data"]:
            raise ValueError("runtime data mismatch")
        validate_panel_runtime(panel, rt)
        load_evaluation_checkpoint(
            checkpoint, rt.objective, training_identity=identity, checkpoint_sha256=cp["sha256"]
        )
        obj = rt.objective
        obj.eval()
        if getattr(obj, "training_stage", "joint") != endpoint["training_stage"]:
            raise ValueError("loaded checkpoint phase differs from its diagnostic endpoint")
        before = validation_state(rt)
        with (
            torch.random.fork_rng(devices=[0]),
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            torch.manual_seed(args.seed)
            cached = static_graph(rt)
            for number, group in enumerate(panel["schedule"]):
                admit(policy, snapshot(os.getpid()))
                batch = _move_batch_tree(
                    assemble_control_only(
                        rt, group["rows"], group["queries"], group["view_seed"], deep=False
                    ),
                    rt.device,
                )
                actual_ids = batch.graph.ids[batch.query_positions]
                if actual_ids.tolist() != group["queries"]:
                    raise ValueError("assembled common query panel order changed")
                if number == 0:
                    parity = verify_cache(rt, batch, cached)
                with ResponseStageObserver(
                    obj.student.response, gene_ids=actual_ids, records=group["rows"], seed=args.seed
                ) as observer:
                    response = main_response(rt, batch, cached)
                stages = observer.result()
                for name in stage_rows:
                    stage_rows[name].extend(stages[name])
                for feature in ("control_cls", "response_cls"):
                    main_views.add(
                        feature, "student", "main", group["rows"], npval(response[feature])
                    )
                # These are predicted responses to crossed inputs, not observed
                # control/perturbed cell pairs; do not score against copied truth.
                main_views.add(
                    "predicted_delta", "student", "main", group["rows"], npval(response["delta"])
                )
                deep = _move_batch_tree(
                    assemble_control_only(
                        rt, group["rows"], group["queries"], group["view_seed"], deep=True
                    ),
                    rt.device,
                )
                deep, view_receipt = matched_cell_views.bind(deep, group["rows"])
                view_receipts.append({"condition_id": group["condition_id"], **view_receipt})
                scope = collect_panel_views(obj, deep, group["rows"], args.seed, views, prototypes)
                diagnostic_scopes.append({"condition_id": group["condition_id"], **scope})
                write(
                    args.output / "progress.json",
                    dict(
                        condition_context_groups_done=number + 1,
                        total=len(panel["schedule"]),
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
        after = validation_state(rt)
        if before != after or sha(checkpoint) != cp["sha256"]:
            raise ValueError(
                "diagnostic changed model/teacher/center/optimizer/gradients/RNG/checkpoint"
            )
    for name, rows in stage_rows.items():
        csv_write(args.output / ("stage_" + name + ".csv"), rows)
    write(args.output / "matched_expression_views.json", view_receipts)
    for name, bank in (
        ("crossed_views", views),
        ("crossed_main", main_views),
        ("prototype_pairs", prototypes),
    ):
        result = (
            grouped_main_result(bank)
            if name == "crossed_main"
            else prototype_result(bank, obj)
            if name == "prototype_pairs"
            else bank.result()
        )
        write(args.output / (name + ".json"), result)
    result = dict(
        schema="representation-mechanism-probe-1",
        status="completed",
        updates=0,
        analysis_source=analysis_source,
        method_sha=args.method_sha,
        run_id=identity.get("run_id"),
        checkpoint_sha256=cp["sha256"],
        checkpoint_endpoint=endpoint,
        config_sha256=sha(args.config),
        parent_manifest=str(args.parent_manifest),
        parent_manifest_sha256=args.parent_manifest_sha256,
        training_protocol_binding=parent_binding,
        data=identity["data"],
        panel_sha256=panel["panel_sha256"],
        panel_rows=panel["rows"],
        diagnostic_objective_scopes=diagnostic_scopes,
        state_before=before,
        state_after=after,
        state_unchanged=True,
        parity=parity,
        elapsed_seconds=time.perf_counter() - started,
        interpretation=(
            "Supplemental train-only crossed inputs; not historical medium; "
            "no observed paired counterfactual truth; no task-success claim from rank alone"
        ),
    )
    result["outputs"] = {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(args.output.iterdir())
        if p.is_file()
    }
    write(args.output / "COMPLETE.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "config", "data-root", "output", "parent-manifest", "resource-queue"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("method-sha", "parent-manifest-sha256", "resource-queue-sha256"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--checkpoint-role", choices=("last", "pretrain"), default="last")
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--conditions", type=int, default=32)
    p.add_argument("--controls", type=int, default=4)
    args = p.parse_args()
    from run_group import lock

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1"):
        raise ValueError("one physical GPU required")
    with lock(Path(f"/data/yilangliu/GraD-Pert/runtime/v2-gpu-{gpu}.lock")):
        execute(args)


if __name__ == "__main__":
    main()
