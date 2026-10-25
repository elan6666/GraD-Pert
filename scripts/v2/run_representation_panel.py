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
    from generate_representation_round5 import SCHEMA as PROJ_SCHEMA
    from generate_representation_round6 import SCHEMA as EMA_SCHEMA
    from generate_representation_round7 import SCHEMA as DOSE_SCHEMA
    from generate_representation_round8 import SCHEMA as CR0_SCHEMA
    from generate_representation_round9 import SCHEMA as JOINT_SCHEMA
    from generate_representation_round11 import SCHEMA as READOUT_SCHEMA
    from generate_representation_round12 import SCHEMA as SPECTRAL_SCHEMA
    from run_fgpo_medium import (
        au_parent_binding,
        cr_parent_binding,
        frozen_readout_parent_binding,
        spectral_pair_parent_binding,
        staged_joint_parent_binding,
    )

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
    if row["name"] in (
        "R3-AUB",
        "R3-AUS",
        "R4-CR",
        "R5-PROJ",
        "R6-EMA99",
        "R7-S1D05",
        "R8-CR0",
        "R9-SSL1J2",
        "R11-READOUT",
        "R12-SRBAL",
        "R12-SRAGG",
    ):
        spec = queue["medium_analysis"]
        family = {
            "R4-CR": CR_SCHEMA,
            "R5-PROJ": PROJ_SCHEMA,
            "R6-EMA99": EMA_SCHEMA,
            "R7-S1D05": DOSE_SCHEMA,
            "R8-CR0": CR0_SCHEMA,
            "R9-SSL1J2": JOINT_SCHEMA,
            "R11-READOUT": READOUT_SCHEMA,
            "R12-SRBAL": SPECTRAL_SCHEMA,
            "R12-SRAGG": SPECTRAL_SCHEMA,
        }.get(row["name"], SCHEMA)
        if (
            queue.get("family") != family
            or Path(spec["parent_manifest"]).resolve() != args.parent_manifest.resolve()
            or spec["parent_manifest_sha256"] != args.parent_manifest_sha256
        ):
            raise ValueError("population auxiliary panel requires its exact original family parent")
        bind_parent = (
            spectral_pair_parent_binding
            if row["name"] in ("R12-SRBAL", "R12-SRAGG")
            else frozen_readout_parent_binding
            if row["name"] == "R11-READOUT"
            else staged_joint_parent_binding
            if row["name"] == "R9-SSL1J2"
            else cr_parent_binding
            if row["name"] in ("R4-CR", "R5-PROJ", "R6-EMA99", "R7-S1D05", "R8-CR0")
            else au_parent_binding
        )
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


def clean_global_expression_views(batch):
    """Remove only augmentation masks from the two existing Global inputs."""
    from dataclasses import replace

    import torch

    if len(batch.cell_views) < 2:
        raise ValueError("two original Global expression views required")
    return replace(
        batch,
        cell_views=tuple(
            replace(view, mask=torch.zeros_like(view.mask)) if i < 2 else view
            for i, view in enumerate(batch.cell_views)
        ),
    )


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
    clean_globals = bool(getattr(args, "teacher_signal_clean_globals", False))
    if clean_globals and not getattr(args, "teacher_signal_audit", False):
        raise ValueError("clean Global input intervention requires teacher signal audit")
    truth_requested = bool(getattr(args, "truth_path_audit", False))
    if truth_requested and (
        getattr(args, "path_audit", False) or getattr(args, "teacher_signal_audit", False)
    ):
        raise ValueError(
            "truth path, response path and teacher signal audits require separate outputs"
        )
    if truth_requested and (args.conditions, args.controls, args.checkpoint_role) != (
        32,
        4,
        "last",
    ):
        raise ValueError("truth path audit requires frozen 32 x 4 final-checkpoint panel")
    if truth_requested and panel["conditions"] != 32:
        raise ValueError("truth path audit requires exactly 32 distinct frozen conditions")
    truth_audit = None
    teacher_signal = None
    if getattr(args, "teacher_signal_audit", False):
        from teacher_signal_diagnostics import TeacherSignalBank

        if getattr(args, "path_audit", False):
            raise ValueError("teacher signal and response path audits require separate outputs")
        from generate_representation_round9 import SCHEMA as JOINT_SCHEMA

        signal_queue = read(args.resource_queue)
        prefix = args.checkpoint_role == "pretrain" and signal_queue.get("family") == JOINT_SCHEMA
        if (
            (args.conditions, args.controls) != (32, 4)
            or (args.checkpoint_role != "last" and not prefix)
            or (prefix and clean_globals)
        ):
            raise ValueError("teacher signal audit requires frozen32x4 last or sealed R9 prefix")
        teacher_signal = TeacherSignalBank()
    path_audit = None
    if getattr(args, "path_audit", False):
        from response_path_diagnostics import PathAuditBank

        if (args.conditions, args.controls, args.checkpoint_role) != (32, 4, "last") or panel[
            "conditions"
        ] != 32:
            raise ValueError("path audit requires the frozen 32 x 4 last-checkpoint panel")
        path_audit = PathAuditBank(panel, required_shape=(16, 2, 4))
    torch, config, identity, prepare, _ = runtime_args(args)
    analysis_source = source_identity(Path(__file__).resolve().parents[2])
    queue = read(args.resource_queue)
    if truth_requested:
        from generate_representation_round9 import SCHEMA as TRUTH_JOINT_SCHEMA

        if queue.get("family") != TRUTH_JOINT_SCHEMA:
            raise ValueError("truth path audit requires the R9 staged-joint final checkpoint")
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
        if truth_requested:
            from truth_path_diagnostics import TruthPathBank

            truth_audit = TruthPathBank(panel, rt)
        load_evaluation_checkpoint(
            checkpoint, rt.objective, training_identity=identity, checkpoint_sha256=cp["sha256"]
        )
        obj = rt.objective
        obj.eval()
        if endpoint["training_stage"] == "frozen_readout":
            proof = read(args.run_root / "fit/frozen_readout_proof.json")
            from gradpert.training.v2.readout import frozen_digest

            if proof.get("frozen_equal") is not True or frozen_digest(obj) != proof.get(
                "frozen_sha256"
            ):
                raise ValueError("R11 diagnostic checkpoint differs from final frozen proof")
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
                if truth_audit is not None:
                    truth_audit.add(rt, batch, cached, group)
                    write(
                        args.output / "progress.json",
                        dict(
                            condition_context_groups_done=number + 1,
                            total=len(panel["schedule"]),
                            elapsed_seconds=time.perf_counter() - started,
                        ),
                    )
                    continue
                if path_audit is not None:
                    path_audit.add(rt, batch, cached, group)
                    write(
                        args.output / "progress.json",
                        dict(
                            condition_context_groups_done=number + 1,
                            total=len(panel["schedule"]),
                            path_audit_contexts_done=len(path_audit.receipts),
                            elapsed_seconds=time.perf_counter() - started,
                        ),
                    )
                    continue
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
                if clean_globals:
                    before_masks = [int(v.mask.sum()) for v in deep.cell_views[:2]]
                    deep = clean_global_expression_views(deep)
                    view_receipt = {
                        **view_receipt,
                        "input_intervention": "clean_globals_preserve_positions_and_control",
                        "student_global_mask_entries_before": before_masks,
                        "student_global_mask_entries_after": [
                            int(v.mask.sum()) for v in deep.cell_views[:2]
                        ],
                        "effective_cell_views_sha256": MatchedCellViews.signature(deep.cell_views),
                    }
                view_receipts.append({"condition_id": group["condition_id"], **view_receipt})
                if teacher_signal is None:
                    scope = collect_panel_views(
                        obj, deep, group["rows"], args.seed, views, prototypes
                    )
                else:
                    with teacher_signal.capture(obj, group["rows"]):
                        scope = collect_panel_views(
                            obj, deep, group["rows"], args.seed, views, prototypes
                        )
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
    if teacher_signal is not None:
        write(args.output / "teacher_signal.json", teacher_signal.result())
    path_result = path_audit.result() if path_audit is not None else None
    truth_result = truth_audit.result() if truth_audit is not None else None
    if truth_result is not None:
        write(args.output / "truth_path_audit.json", truth_result)
        csv_write(args.output / "truth_path_errors.csv", truth_result["rows"])
        csv_write(args.output / "truth_path_context_means.csv", truth_result["context_means"])
    if path_result is not None:
        write(args.output / "path_audit.json", path_result)
        for name in ("decomposition", "pairs", "interventions", "context_means"):
            csv_write(args.output / ("path_" + name + ".csv"), path_result[name])
    elif truth_result is None:
        for name, rows in stage_rows.items():
            csv_write(args.output / ("stage_" + name + ".csv"), rows)
        write(args.output / "matched_expression_views.json", view_receipts)
    for name, bank in (
        ("crossed_views", views),
        ("crossed_main", main_views),
        ("prototype_pairs", prototypes),
    ):
        if path_audit is not None or truth_audit is not None:
            continue
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
        representation_training_status=(
            "frozen_R9_backbone_and_Teacher; prediction_head_only; SSL_inactive"
            if endpoint["training_stage"] == "frozen_readout"
            else "native_training"
        ),
        diagnostic_mode=(
            "truth_path_audit"
            if truth_audit is not None
            else "response_path_audit"
            if path_audit is not None
            else "teacher_signal_audit"
            if teacher_signal is not None
            else "crossed_views"
        ),
        path_audit=(
            {
                "schema": path_result["schema"],
                "contexts": path_result["contexts"],
                "input_rows": path_result["input_rows"],
                "first_context_seconds": path_result["receipts"][0]["elapsed_seconds"],
                "forward_row_evaluations": path_result["forward_row_evaluations"],
            }
            if path_result is not None
            else None
        ),
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
    if truth_result is not None:
        result["truth_path_audit"] = {
            key: truth_result[key]
            for key in (
                "schema",
                "contexts",
                "input_rows",
                "query_count",
                "forward_row_evaluations",
            )
        }
    if teacher_signal is not None:
        result["teacher_signal_input_mode"] = (
            "clean_globals" if clean_globals else "native_student_masked_teacher_clean"
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
    p.add_argument(
        "--truth-path-audit", action="store_true", help="train population truth F/B errors"
    )
    p.add_argument("--path-audit", action="store_true", help="truth-free F/B/decoder-swap audit")
    p.add_argument(
        "--teacher-signal-audit", action="store_true", help="frozen SSL2 encoder/head crossover"
    )
    p.add_argument(
        "--teacher-signal-clean-globals",
        action="store_true",
        help="same clean Global expressions for Student and Teacher; frozen input intervention",
    )
    args = p.parse_args()
    from run_group import lock

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1"):
        raise ValueError("one physical GPU required")
    with lock(Path(f"/data/yilangliu/GraD-Pert/runtime/v2-gpu-{gpu}.lock")):
        execute(args)


if __name__ == "__main__":
    main()
