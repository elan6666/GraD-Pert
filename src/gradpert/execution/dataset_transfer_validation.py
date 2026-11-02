"""Last-300 validation planning; inference and metrics remain in native v2 APIs."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.data import DatasetLayout
from gradpert.data._io import atomic_json, read_json
from gradpert.execution.v2_training_stage import validate_training_stage
from gradpert.hashing import sha256_file, sha256_json

SCHEMA = "dataset-transfer-last300-validation-v1"
HOOK = "dataset_transfer_validation"
DATASETS = {"replogle_k562_essential", "replogle_rpe1_essential", "nadig_hepg2", "norman"}
METRICS = {
    f"{name}_{axis}"
    for name in ("txpert_macro_pearson_delta", "trishift_pearson_delta", "systema_pearson")
    for axis in ("all", "deg")
}


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}


def checked(value: dict[str, str]) -> Path:
    path = Path(value["path"]).resolve(strict=True)
    from gradpert.execution.v2_checkpoint_eval import SERVER_ROOT

    if not path.is_relative_to(SERVER_ROOT) or sha256_file(path) != value["sha256"]:
        raise ValueError("validation input path or hash changed")
    return path


def protected_roots(training: dict[str, Any], identity: dict[str, Any]) -> tuple[Path, ...]:
    """Original TRAIN source is mandatory, independent of the evaluation source."""
    value = identity.get("source", {}).get("repository_root")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("original training source repository_root metadata is required")
    original = Path(value).resolve(strict=True)
    if not original.is_dir():
        raise ValueError("original training source repository_root must be a directory")
    return (
        original,
        Path(training["run_root"]).resolve(strict=True),
        Path(training["data_root"]).resolve(strict=True),
        Path(__file__).resolve().parents[3],
    )


def validate_output_path(
    path: Path, roots: tuple[Path, ...], *, prospective_output: Path | None = None
) -> Path:
    """Apply the same server/protected-root fence to artifacts and plan files."""
    from gradpert.execution.v2_checkpoint_eval import SERVER_ROOT

    target = path.resolve()
    if not target.is_relative_to(SERVER_ROOT):
        raise ValueError("validation output paths must stay on the server")
    if any(target.is_relative_to(root) or root.is_relative_to(target) for root in roots):
        raise ValueError("independent validation output overlaps a protected input/source")
    if prospective_output is not None:
        output = prospective_output.resolve()
        if target.is_relative_to(output) or output.is_relative_to(target):
            raise ValueError("plan output overlaps prospective evaluation output")
    return target


def training_terminal(training: dict[str, Any]) -> dict[str, Any]:
    """Require the exact terminal 300-update protocol, including partial epochs."""
    config = load_experiment_config(training["config"])
    arch, options = V2Options.parse_parameters(config.model.parameters)
    if (
        config.dataset_id not in DATASETS
        or config.model_id != "gradpert_v2"
        or config.training.formal_run_policy != "v2_updates_300"
        or not config.model.excludes_test_target_expression
        or options.training_mode != "joint"
        or options.validation_mode != "disabled"
        or training.get("postfit_policy") != "deferred"
        or config.continuation is not None
        or any(
            getattr(arch, name) != "autocast"
            for name in ("ffn_compute_precision", "ssl_head_precision", "prediction_precision")
        )
    ):
        raise ValueError("validation requires the explicit within-cell native last300 protocol")
    terminal = validate_training_stage(training)
    root = Path(training["run_root"])
    identity = terminal["identity"]
    journal, history = (
        read_json(root / "fit/epoch_state.json"),
        read_json(root / "fit/history.json"),
    )
    epochs, steps = journal["budget"]
    if type(epochs) is not int or type(steps) is not int or steps < 1:
        raise ValueError("invalid sealed data-epoch budget")
    expected = [min(steps, 300 - i * steps) for i in range(epochs)]
    budget = identity.get("training_budget", {})
    if (
        epochs != (300 + steps - 1) // steps
        or epochs != config.training.max_epochs.value
        or budget
        != {
            "unit": "optimizer_update",
            "count": 300,
            "steps_per_data_epoch": steps,
            "partial_final_epoch": 300 % steps != 0,
        }
        or identity.get("run_id") != training["run_id"]
        or identity["resolved_config_sha256"] != sha256_json(config.model_dump(mode="json"))
        or training["resolved_config"] != config.model_dump(mode="json")
        or read_json(root / "resolved_config.json") != config.model_dump(mode="json")
        or [row["updates_this_data_epoch"] for row in history] != expected
        or [row["optimizer_steps"] for row in history]
        != [min((i + 1) * steps, 300) for i in range(epochs)]
        or any(
            row.get("data_epoch_complete") is not (count == steps)
            or row.get("data_epoch_fraction") != count / steps
            for row, count in zip(history, expected, strict=True)
        )
    ):
        raise ValueError("last300 terminal counters, identity or partial epoch differ")
    return terminal


def frozen_axis(training: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    """Read metadata only; never materialize expression or prepare a reference."""
    from gradpert.evaluation.state import EvaluationStateLayout
    from gradpert.training.expression_policy import expression_policy

    config = load_experiment_config(training["config"])
    layout = DatasetLayout(Path(training["data_root"]), config.dataset_id, config.data.protocol_id)
    genes_file, split_file = (
        layout.canonical / "expression_gene_ids.txt",
        layout.manifests / "split.json",
    )
    genes, split = tuple(genes_file.read_text().splitlines()), read_json(split_file)
    if (split["dataset_id"], split["protocol_id"]) != (config.dataset_id, config.data.protocol_id):
        raise ValueError("validation split dataset identity differs")
    allowed, policy = expression_policy(
        genes, tuple(split["test_conditions"]), split["control_condition_id"], enabled=True
    )
    data = identity["data"]
    if (
        policy != data["training_expression_policy"]
        or policy["allowed_expression_gene_ids_sha256"]
        != data["effective_training_expression_ids_sha256"]
    ):
        raise ValueError("frozen expression policy differs from training identity")
    allowed_ids = [int(i) for i in allowed]
    allowed_set = set(allowed_ids)
    heldout = [i for i in range(len(genes)) if i not in allowed_set]
    reference = EvaluationStateLayout(
        Path(training["data_root"]),
        config.dataset_id,
        config.data.protocol_id,
        validation_only=True,
        evaluation_protocol="v2",
    ).manifest
    # Missing reference is a separate explicit preparation stage, never silently generated.
    return dict(
        dataset_id=config.dataset_id,
        gene_axis=binding(genes_file),
        split=binding(split_file),
        val_controls=binding(layout.manifests / "evaluation_controls.val.json"),
        reference_manifest=binding(reference),
        conditions=split["val_conditions"],
        expression_gene_order_sha256=sha256_json(list(genes)),
        groups={"allowed_expression": allowed_ids, "heldout_expression": heldout},
        policy=policy,
    )


def _compatibility(plan: dict[str, Any], proof: dict[str, str] | None) -> None:
    train, evaluation = (
        plan["training_identity"]["source"]["commit"],
        plan["evaluation_source"]["commit"],
    )
    if train == evaluation:
        return
    if proof is None:
        raise ValueError("cross-version evaluation requires independent compatibility approval")
    receipt = read_json(checked(proof))
    if (
        receipt.get("approved") is not True
        or receipt.get("model_checkpoint_compatible") is not True
        or receipt.get("training_commit") != train
        or receipt.get("evaluation_commit") != evaluation
        or not isinstance(receipt.get("checks"), list)
        or not receipt["checks"]
    ):
        raise ValueError("cross-version compatibility approval differs")


def build_validation_plan(
    *,
    training_plan: Path,
    training_plan_sha256: str,
    runtime: Path,
    output: Path,
    gpu: str,
    compatibility: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Plan only; reference preparation, resource admission and execution are explicit."""
    from gradpert.execution.v2_checkpoint_eval import resolve_evaluation_plan

    original = {"path": str(training_plan.resolve(strict=True)), "sha256": training_plan_sha256}
    training = read_json(checked(original))
    terminal = training_terminal(training)
    root = Path(training["run_root"])
    last = terminal["last"]
    plan = resolve_evaluation_plan(
        argparse.Namespace(
            config=Path(training["config"]),
            training_run_root=root,
            checkpoint=root / "fit" / last["file"],
            checkpoint_sha256=last["sha256"],
            checkpoint_role="last",
            runtime=runtime,
            output_root=output,
            gpu=gpu,
            split="val",
            evaluation_precision="native",
        )
    )
    if Path(plan["data_root"]).resolve() != Path(training["data_root"]).resolve():
        raise ValueError("evaluation data root differs from training")
    _compatibility(plan, compatibility)
    paths = [
        root / name
        for name in (
            "launch.json",
            "run_manifest.json",
            "resolved_config.json",
            "TRAIN_COMPLETE.json",
            "fit/epoch_state.json",
            "fit/history.json",
        )
    ]
    plan[HOOK] = dict(
        schema=SCHEMA,
        training_plan=original,
        runtime=binding(runtime),
        terminal_inputs=[binding(p) for p in paths],
        frozen_axis=frozen_axis(training, terminal["identity"]),
        compatibility=compatibility,
        precision="native_unautocast",
    )
    validate_hook(plan)
    return plan


def validate_hook(plan: dict[str, Any]) -> bool:
    """No key means the historical path; any unknown or mismatched key fails closed."""
    if HOOK not in plan:
        return False
    hook = plan[HOOK]
    if (
        not isinstance(hook, dict)
        or hook.get("schema") != SCHEMA
        or plan.get("split") != "val"
        or plan.get("checkpoint_role") != "last"
        or plan.get("evaluation_precision", "native") != "native"
        or hook.get("precision") != "native_unautocast"
        or any(key in plan for key in ("diagnostic_only", "diagnostic_condition_count"))
    ):
        raise ValueError("unknown or truncated last300 validation policy")
    training = read_json(checked(hook["training_plan"]))
    for value in hook["terminal_inputs"]:
        checked(value)
    runtime = checked(hook["runtime"])
    if sha256_file(runtime) != plan["runtime_sha256"]:
        raise ValueError("evaluation runtime changed")
    terminal = training_terminal(training)
    config = load_experiment_config(training["config"])
    validate_output_path(Path(plan["output_root"]), protected_roots(training, terminal["identity"]))
    from gradpert.execution.v2_checkpoint_eval import inspect_source_identity

    runtime_config = read_json(runtime)
    expected_root = Path(__file__).resolve().parents[3]
    if Path(plan["evaluation_source"]["repository_root"]).resolve() != expected_root:
        raise ValueError("evaluation import root differs from the sealed source")
    source = inspect_source_identity(
        expected_root,
        formal=True,
        expected_repository=config.source_code.repository,
        publication_receipt=runtime_config["publication_receipt"],
        expected_publication_receipt_sha256=runtime_config["publication_sha256"],
    ).payload()
    if source != plan["evaluation_source"]:
        raise ValueError("evaluation publication/source identity changed")
    if (
        terminal["identity"] != plan["training_identity"]
        or plan["training_run_root"] != training["run_root"]
        or plan["config"] != training["config"]
        or plan["config_sha256"] != training["config_sha256"]
        or plan["checkpoint_sha256"] != terminal["last"]["sha256"]
        or plan["checkpoint_epoch"] != terminal["epoch"]
        or Path(plan["checkpoint"]).resolve()
        != (Path(training["run_root"]) / "fit" / terminal["last"]["file"]).resolve()
        or Path(plan["data_root"]).resolve() != Path(training["data_root"]).resolve()
        or config.dataset_id != hook["frozen_axis"]["dataset_id"]
        or frozen_axis(training, terminal["identity"]) != hook["frozen_axis"]
    ):
        raise ValueError("last300 evaluation plan differs from sealed training or frozen axis")
    _compatibility(plan, hook.get("compatibility"))
    return True


def validation_groups(
    plan: dict[str, Any],
    *,
    history: list[dict[str, Any]],
    genes: tuple[str, ...],
    allowed: tuple[int, ...],
    test_conditions: tuple[str, ...],
    control_condition: str,
    condition_order: tuple[str, ...],
    reference_hash: str,
    controls_hash: str,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    from gradpert.training.expression_policy import expression_policy
    from gradpert.training.v2.exposure import checkpoint_expression_groups

    frozen = plan[HOOK]["frozen_axis"]
    if history != read_json(Path(plan["training_run_root"]) / "fit/history.json"):
        raise ValueError("loaded checkpoint history differs from sealed terminal history")
    _, current_policy = expression_policy(genes, test_conditions, control_condition, enabled=True)
    groups = {key: tuple(value) for key, value in frozen["groups"].items()}
    if (
        current_policy != frozen["policy"]
        or condition_order != tuple(frozen["conditions"])
        or reference_hash != frozen["reference_manifest"]["sha256"]
        or controls_hash != frozen["val_controls"]["sha256"]
        or allowed != groups["allowed_expression"]
        or sha256_json(list(genes)) != frozen["expression_gene_order_sha256"]
    ):
        raise ValueError("runtime validation axis, conditions or reference differ")
    actual, exposure = checkpoint_expression_groups(
        history,
        checkpoint_epoch=plan["checkpoint_epoch"],
        expression_gene_ids=genes,
        allowed_gene_indices=allowed,
    )
    unobserved = sorted(set(allowed) - set(actual["seen_expression"]))
    exposure.update(
        definition="actual_query_exposure_separate_from_frozen_allowed_heldout",
        frozen_groups={
            key: dict(gene_count=len(ids), gene_ids_sha256=sha256_json([genes[i] for i in ids]))
            for key, ids in groups.items()
        },
        allowed_but_not_observed=dict(
            gene_count=len(unobserved), gene_ids_sha256=sha256_json([genes[i] for i in unobserved])
        ),
        actual_seen_equals_allowed=not unobserved,
    )
    return groups, exposure


def validation_context(plan: dict[str, Any], device_type: str) -> AbstractContextManager:
    if HOOK not in plan:
        return nullcontext()
    import torch

    return torch.autocast(device_type=device_type, enabled=False)


def validate_merged(plan: dict[str, Any], result: dict[str, Any], orders: tuple[str, ...]) -> None:
    if orders != tuple(plan[HOOK]["frozen_axis"]["conditions"]):
        raise ValueError("last300 validation did not cover the complete frozen order")
    if result["split"] != "val" or set(result["metric_gene_groups"]) != {
        "allowed_expression",
        "heldout_expression",
    }:
        raise ValueError("last300 requires all/allowed/heldout validation groups")
    for metrics in [
        result["metrics"],
        *[group["metrics"] for group in result["metric_gene_groups"].values()],
    ]:
        if {m["metric_id"] for m in metrics} != METRICS or len(metrics) != 6:
            raise ValueError("last300 validation must retain all18 native metrics, including NA")
        if any(
            m["total_condition_count"] != len(orders)
            or type(m["finite_condition_count"]) is not int
            or not 0 <= m["finite_condition_count"] <= len(orders)
            for m in metrics
        ):
            raise ValueError("validation metric support differs from full frozen population")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--training-plan-sha256", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--compatibility", type=Path)
    parser.add_argument("--compatibility-sha256")
    parser.add_argument("--plan-output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.compatibility) != bool(args.compatibility_sha256):
        parser.error("compatibility path and explicit hash must be supplied together")
    plan = build_validation_plan(
        training_plan=args.training_plan,
        training_plan_sha256=args.training_plan_sha256,
        runtime=args.runtime,
        output=args.output,
        gpu=args.gpu,
        compatibility=(
            {"path": str(args.compatibility), "sha256": args.compatibility_sha256}
            if args.compatibility
            else None
        ),
    )
    training = read_json(checked(plan[HOOK]["training_plan"]))
    destination = validate_output_path(
        args.plan_output,
        protected_roots(training, plan["training_identity"]),
        prospective_output=Path(plan["output_root"]),
    )
    if args.plan_output.exists() or args.plan_output.is_symlink():
        raise FileExistsError("validation plan output must be new")
    atomic_json(destination, plan)


if __name__ == "__main__":
    main()
