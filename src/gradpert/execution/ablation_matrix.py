"""Hash-pinned orchestration for the Nadig Jurkat B2-vNext config matrix.

This module never implements training.  Every executable row is delegated to
the one native ``gradpert model pilot`` entrypoint after the complete matrix,
source, config, and GenePT preflight contracts have been validated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from gradpert.config import load_experiment_config
from gradpert.data._io import atomic_json
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file
from gradpert.pilots import GenePTAvailabilityReceipt, GenePTSeedAvailabilityReceipt

SUCCESSOR_MATRIX_ID = "nadig_jurkat_vnext_ratio_graph_v4"
SUCCESSOR_CONTRACT: dict[str, tuple[str, frozenset[str]]] = {
    "a0_ratio_ring_half": ("reference", frozenset()),
    "h1_hvg1024_ratio_half": (
        "graph_hvg_count",
        frozenset({"graph_hvg_count", "runtime_graph_root"}),
    ),
    "h2_hvg2048_ratio_half": (
        "graph_hvg_count",
        frozenset({"graph_hvg_count", "runtime_graph_root"}),
    ),
    "h3_hvg5000_ratio_half": (
        "graph_hvg_count",
        frozenset({"graph_hvg_count", "runtime_graph_root"}),
    ),
    "l1_fanout_ratio_half": ("local_view_builder", frozenset({"local_view_builder"})),
    "l2_ring_half_count8": ("local_view_count", frozenset({"local_view_count"})),
    "l3_ring_quarter": (
        "local_view_node_budget_ratio",
        frozenset({"local_view_node_budget_ratio"}),
    ),
    "l4_ring_half_mask_half": (
        "local_anchor_mask_view_ratio",
        frozenset({"local_anchor_mask_view_ratio"}),
    ),
    "l5_ring_half_mask_quarter": (
        "local_anchor_mask_view_ratio",
        frozenset({"local_anchor_mask_view_ratio"}),
    ),
    "m1_single_string_gat": (
        "graph_encoder_family",
        frozenset({"graph_sources", "graph_encoder_family", "graph_encoder_dropout"}),
    ),
    "m2_single_string_transformer": (
        "graph_encoder_family",
        frozenset({"graph_sources", "graph_encoder_family"}),
    ),
    "m4_adaptive_source_gat": (
        "graph_encoder_family",
        frozenset({"graph_encoder_family", "graph_encoder_dropout"}),
    ),
    "w1_string_edge_feature": (
        "string_weight_mode",
        frozenset(
            {
                "graph_sources",
                "graph_encoder_family",
                "graph_encoder_dropout",
                "string_weight_mode",
            }
        ),
    ),
    "w2_string_fixed_prior": (
        "string_weight_mode",
        frozenset(
            {
                "graph_sources",
                "graph_encoder_family",
                "graph_encoder_dropout",
                "string_weight_mode",
            }
        ),
    ),
    "w3_string_prior_residual": (
        "string_weight_mode",
        frozenset(
            {
                "graph_sources",
                "graph_encoder_family",
                "graph_encoder_dropout",
                "string_weight_mode",
            }
        ),
    ),
    "ws_string_weight_shuffle": (
        "string_weight_mode",
        frozenset(
            {
                "graph_sources",
                "graph_encoder_family",
                "graph_encoder_dropout",
                "string_weight_mode",
            }
        ),
    ),
    "d1_control_mlp": ("decoder_mode", frozenset({"decoder_mode"})),
    "d2_control_transformer": ("decoder_mode", frozenset({"decoder_mode"})),
    "d3_concat_p64": ("decoder_fusion", frozenset({"decoder_mode"})),
    "d4_concat_transformer_p64": ("decoder_fusion", frozenset({"decoder_mode"})),
    "d5_concat_p256": (
        "decoder_fusion_x_perturbation_width",
        frozenset({"decoder_mode", "graph_tower_output_dim"}),
    ),
    "d6_concat_transformer_p256": (
        "decoder_fusion_x_perturbation_width",
        frozenset({"decoder_mode", "graph_tower_output_dim"}),
    ),
    "e1_frozen_genept": (
        "gene_feature_mode",
        frozenset(
            {
                "gene_feature_mode",
                "genept_artifact_path",
                "genept_expected_sha256",
            }
        ),
    ),
    "e2_genept_id_residual": (
        "gene_feature_mode",
        frozenset(
            {
                "gene_feature_mode",
                "genept_artifact_path",
                "genept_expected_sha256",
            }
        ),
    ),
    "e3_genept_initialized": (
        "gene_feature_mode",
        frozenset(
            {
                "gene_feature_mode",
                "genept_artifact_path",
                "genept_expected_sha256",
            }
        ),
    ),
    "es_genept_shuffle": (
        "gene_feature_mode",
        frozenset(
            {
                "gene_feature_mode",
                "genept_artifact_path",
                "genept_expected_sha256",
            }
        ),
    ),
    "o1_no_condition": (
        "condition_consistency_loss_weight",
        frozenset({"condition_consistency_loss_weight"}),
    ),
    "o2_no_masked_node": ("masked_node_loss_weight", frozenset({"masked_node_loss_weight"})),
    "o3_no_spread": ("spread_loss_weight", frozenset({"spread_loss_weight"})),
}


@dataclass(frozen=True)
class AblationMatrixRow:
    variant_id: str
    config_path: Path
    config_sha256: str
    run_seed: int
    genept_preflight_required: bool
    matrix_schema_version: Literal["1", "2"]
    semantic_factor: str | None
    declared_parameter_diffs: tuple[str, ...]
    genept_artifact_path: str | None
    genept_expected_sha256: str | None
    runtime_graph_root: str | None


@dataclass(frozen=True)
class AblationLaunchPlanRow:
    variant_id: str
    config_path: str
    config_sha256: str
    run_root: str
    run_id: str
    run_seed: int
    device: str
    disposition: Literal["run", "skip_genept_missing_target"]
    matrix_schema_version: Literal["1", "2"]
    semantic_factor: str | None
    declared_parameter_diffs: tuple[str, ...]
    genept_preflight_receipt_path: str | None
    genept_preflight_receipt_sha256: str | None
    genept_preflight_schema_version: str | None
    genept_preflight_status: str | None
    genept_preflight_artifact_sha256: str | None
    genept_preflight_missing_target_ids_sha256: str | None


def _require_git_source(repository_root: Path, expected_commit: str) -> None:
    if len(expected_commit) != 40:
        raise ValueError("expected source commit must contain 40 hexadecimal characters")
    try:
        int(expected_commit, 16)
    except ValueError as error:
        raise ValueError("expected source commit must contain 40 hexadecimal characters") from error
    observed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected_commit:
        raise ValueError("ablation launcher source commit differs from the frozen contract")
    dirty = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("ablation launcher requires a clean source worktree")


def load_ablation_matrix(
    matrix_path: str | Path,
    *,
    repository_root: str | Path,
) -> tuple[AblationMatrixRow, ...]:
    """Validate the frozen one-dataset/one-split/one-seed/10-epoch matrix."""

    matrix_file = Path(matrix_path).resolve(strict=True)
    root = Path(repository_root).resolve(strict=True)
    payload = json.loads(matrix_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ablation matrix must be a JSON object")
    schema_version = payload.get("schema_version")
    if (
        schema_version not in {"1", "2"}
        or payload.get("dataset_id") != "nadig_jurkat"
        or payload.get("canonical_split_count") != 1
        or payload.get("run_seeds") != [1]
        or payload.get("max_epochs") != 10
    ):
        raise ValueError("ablation matrix experiment identity differs from the frozen design")
    if schema_version == "2" and payload.get("matrix_id") != SUCCESSOR_MATRIX_ID:
        raise ValueError("schema-v2 ablation matrix id differs from the successor contract")
    if schema_version == "1" and payload.get("matrix_id") == SUCCESSOR_MATRIX_ID:
        raise ValueError("schema-v1 matrix cannot claim the successor schema-v2 identity")
    raw_rows = payload.get("rows")
    expected_row_count = {"1": 22, "2": 29}[str(schema_version)]
    if (
        not isinstance(raw_rows, list)
        or payload.get("row_count") != len(raw_rows)
        or len(raw_rows) != expected_row_count
    ):
        raise ValueError(f"ablation matrix must contain exactly {expected_row_count} declared rows")

    if schema_version == "2":
        raw_variant_ids = [
            raw.get("variant_id") if isinstance(raw, dict) else None for raw in raw_rows
        ]
        if (
            any(not isinstance(variant_id, str) for variant_id in raw_variant_ids)
            or len(set(raw_variant_ids)) != len(raw_variant_ids)
            or set(raw_variant_ids) != set(SUCCESSOR_CONTRACT)
        ):
            raise ValueError("schema-v2 matrix variant set differs from the successor contract")

    baseline_parameters: dict[str, object] | None = None
    if schema_version == "2":
        baseline_relative = (
            "configs/ablations/nadig_jurkat/a0_ratio_ring_half/gradpert_b2/nadig_jurkat.yaml"
        )
        baseline_config = load_experiment_config((root / baseline_relative).resolve(strict=True))
        baseline_parameters = {
            name: parameter.value for name, parameter in baseline_config.model.parameters.items()
        }

    rows: list[AblationMatrixRow] = []
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("every ablation matrix row must be a JSON object")
        variant_id = raw.get("variant_id")
        relative_config = raw.get("config_path")
        expected_hash = raw.get("config_sha256")
        requires_genept = raw.get("genept_preflight_required")
        semantic_factor = raw.get("semantic_factor")
        declared_parameter_diffs = raw.get("declared_parameter_diffs")
        if (
            not isinstance(variant_id, str)
            or not variant_id
            or variant_id in seen
            or not isinstance(relative_config, str)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not isinstance(requires_genept, bool)
        ):
            raise ValueError("ablation matrix row identity is malformed or duplicated")
        if schema_version == "2":
            expected_semantic_factor, expected_diffs = SUCCESSOR_CONTRACT[variant_id]
            if (
                semantic_factor != expected_semantic_factor
                or not isinstance(declared_parameter_diffs, list)
                or any(not isinstance(name, str) for name in declared_parameter_diffs)
                or declared_parameter_diffs != sorted(set(declared_parameter_diffs))
                or frozenset(declared_parameter_diffs) != expected_diffs
            ):
                raise ValueError(f"schema-v2 semantic declaration differs: {variant_id}")
            expected_relative_config = (
                f"configs/ablations/nadig_jurkat/{variant_id}/gradpert_b2/nadig_jurkat.yaml"
            )
            if relative_config != expected_relative_config:
                raise ValueError(f"schema-v2 config path does not match variant id: {variant_id}")
        else:
            semantic_factor = None
            declared_parameter_diffs = []
        config_path = (root / relative_config).resolve(strict=True)
        if not config_path.is_relative_to(root) or config_path.is_symlink():
            raise ValueError("ablation config must be a regular file inside the repository")
        if sha256_file(config_path) != expected_hash:
            raise ValueError(f"ablation config hash mismatch: {variant_id}")
        if (
            raw.get("dataset_id") != "nadig_jurkat"
            or raw.get("split_policy") != "frozen_canonical"
            or raw.get("run_seed") != 1
            or raw.get("max_epochs") != 10
            or raw.get("result_mode") != "metrics_only"
        ):
            raise ValueError(f"ablation row execution contract differs: {variant_id}")
        config = load_experiment_config(config_path)
        if (
            config.model.model_id != "gradpert_b2"
            or config.dataset_id != "nadig_jurkat"
            or config.training.max_epochs.value != 10
            or config.training.run_seeds != [1]
            or config.training.early_stopping
            or config.artifacts.result_mode != "metrics_only"
        ):
            raise ValueError(f"resolved config differs from matrix row: {variant_id}")
        if schema_version == "2":
            assert baseline_parameters is not None
            resolved_variant = config.model.parameters.get("performance_pilot_variant")
            if resolved_variant is None or resolved_variant.value != f"vnext_{variant_id}":
                raise ValueError(f"schema-v2 config identity differs from variant id: {variant_id}")
            observed_parameters = {
                name: parameter.value for name, parameter in config.model.parameters.items()
            }
            observed_diffs = {
                name
                for name in set(baseline_parameters) | set(observed_parameters)
                if baseline_parameters.get(name, "<missing>")
                != observed_parameters.get(name, "<missing>")
            }
            observed_diffs.discard("performance_pilot_variant")
            expected_diffs = SUCCESSOR_CONTRACT[variant_id][1]
            if observed_diffs != expected_diffs:
                raise ValueError(f"schema-v2 resolved parameter diff differs: {variant_id}")
        genept_artifact_path: str | None = None
        genept_expected_sha256: str | None = None
        runtime_graph_root: str | None = None
        if requires_genept:
            resolved_genept_values: dict[str, str] = {}
            for parameter_name in (
                "genept_artifact_path",
                "genept_expected_sha256",
                "runtime_graph_root",
            ):
                parameter = config.model.parameters.get(parameter_name)
                if parameter is None or not isinstance(parameter.value, str) or not parameter.value:
                    raise ValueError(f"GenePT row lacks a sealed {parameter_name}: {variant_id}")
                resolved_genept_values[parameter_name] = parameter.value
            genept_artifact_path = resolved_genept_values["genept_artifact_path"]
            genept_expected_sha256 = resolved_genept_values["genept_expected_sha256"]
            runtime_graph_root = resolved_genept_values["runtime_graph_root"]
        seen.add(variant_id)
        rows.append(
            AblationMatrixRow(
                variant_id=variant_id,
                config_path=config_path,
                config_sha256=expected_hash,
                run_seed=1,
                genept_preflight_required=requires_genept,
                matrix_schema_version=schema_version,
                semantic_factor=semantic_factor,
                declared_parameter_diffs=tuple(declared_parameter_diffs),
                genept_artifact_path=genept_artifact_path,
                genept_expected_sha256=genept_expected_sha256,
                runtime_graph_root=runtime_graph_root,
            )
        )
    return tuple(rows)


def build_ablation_launch_plan(
    rows: Sequence[AblationMatrixRow],
    *,
    selected_variants: Sequence[str],
    runs_root: str | Path,
    device: str,
    genept_availability_receipt: str | Path | None,
    data_root: str | Path | None = None,
) -> tuple[AblationLaunchPlanRow, ...]:
    """Resolve selected rows and fail closed before any training process starts."""

    if not device.startswith("cuda:"):
        raise ValueError("formal ablation execution requires an explicit cuda:N device")
    by_id = {row.variant_id: row for row in rows}
    selected = tuple(selected_variants) if selected_variants else tuple(by_id)
    if len(set(selected)) != len(selected):
        raise ValueError("selected ablation variants must be unique")
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")

    needs_genept = any(by_id[variant].genept_preflight_required for variant in selected)
    genept_status: str | None = None
    genept_receipt_path: Path | None = None
    genept_receipt_sha256: str | None = None
    genept_receipt_schema: str | None = None
    genept_artifact_sha256: str | None = None
    genept_missing_target_ids_sha256: str | None = None
    genept_receipt: GenePTAvailabilityReceipt | GenePTSeedAvailabilityReceipt | None = None
    if needs_genept:
        if genept_availability_receipt is None:
            raise ValueError("GenePT variants require a sealed availability receipt")
        genept_receipt_path = Path(genept_availability_receipt).resolve(strict=True)
        genept_receipt_sha256 = sha256_file(genept_receipt_path)
        receipt_text = genept_receipt_path.read_text(encoding="utf-8")
        raw_receipt = json.loads(receipt_text)
        if not isinstance(raw_receipt, dict):
            raise ValueError("GenePT availability receipt must be a JSON object")
        genept_receipt_schema = raw_receipt.get("schema_version")
        receipt: GenePTAvailabilityReceipt | GenePTSeedAvailabilityReceipt
        if genept_receipt_schema == "genept-vnext-availability-v1":
            receipt = GenePTAvailabilityReceipt.model_validate_json(receipt_text)
            genept_missing_target_ids_sha256 = receipt.missing_perturbation_target_gene_ids_sha256
        elif genept_receipt_schema == "genept-seed-go-protein-pathway-availability-v2":
            receipt = GenePTSeedAvailabilityReceipt.model_validate_json(receipt_text)
        else:
            raise ValueError("unsupported GenePT availability receipt schema")
        genept_status = receipt.status
        genept_artifact_sha256 = receipt.genept_source_sha256
        genept_receipt = receipt
        if isinstance(receipt, GenePTSeedAvailabilityReceipt):
            if data_root is None:
                raise ValueError("GenePT Seed launch planning requires the live data root")
            relative_graph_root = Path(receipt.runtime_graph_root)
            if relative_graph_root.is_absolute() or ".." in relative_graph_root.parts:
                raise ValueError("GenePT Seed receipt runtime graph root is unsafe")
            live_manifest_path = (
                Path(data_root)
                .resolve(strict=True)
                .joinpath(*relative_graph_root.parts)
                .joinpath("manifest.json")
                .resolve(strict=True)
            )
            live_manifest = json.loads(live_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(live_manifest, dict):
                raise ValueError("live GenePT parent graph manifest must be a JSON object")
            candidate_targets = live_manifest.get("candidate_target_ids")
            if (
                sha256_file(live_manifest_path) != receipt.parent_graph_manifest_sha256
                or live_manifest.get("topology_content_sha256")
                != receipt.parent_topology_content_sha256
                or live_manifest.get("graph_gene_order_sha256")
                != receipt.parent_graph_gene_order_sha256
                or live_manifest.get("graph_gene_count") != receipt.requested_runtime_gene_count
                or live_manifest.get("graph_gene_count") != receipt.selected_gene_count
                or receipt.ignored_missing_non_perturbation_gene_count != 0
                or receipt.result_topology_content_sha256 != receipt.parent_topology_content_sha256
                or live_manifest.get("candidate_target_order_sha256")
                != receipt.candidate_target_order_sha256
                or not isinstance(candidate_targets, list)
                or len(candidate_targets) != receipt.perturbation_target_gene_count
            ):
                raise ValueError("GenePT Seed preflight differs from the live graph manifest")

    root = Path(runs_root).resolve()
    plan: list[AblationLaunchPlanRow] = []
    for variant_id in selected:
        row = by_id[variant_id]
        run_root = root / variant_id / "gradpert_b2" / "nadig_jurkat" / "seed-1"
        if run_root.exists() and any(run_root.iterdir()):
            raise ValueError(f"refusing to overwrite existing ablation run: {run_root}")
        disposition: Literal["run", "skip_genept_missing_target"] = "run"
        if row.genept_preflight_required:
            assert genept_receipt is not None
            if isinstance(genept_receipt, GenePTSeedAvailabilityReceipt):
                if (
                    row.genept_artifact_path != genept_receipt.genept_source_path
                    or row.genept_expected_sha256 != genept_receipt.genept_source_sha256
                    or row.runtime_graph_root != genept_receipt.runtime_graph_root
                ):
                    raise ValueError(
                        f"GenePT Seed preflight identity differs from config: {variant_id}"
                    )
            elif row.genept_artifact_path is not None and row.genept_artifact_path.endswith(".npz"):
                raise ValueError(
                    f"GenePT NPZ variant requires the Seed availability schema: {variant_id}"
                )
            if genept_status != "available":
                disposition = "skip_genept_missing_target"
        plan.append(
            AblationLaunchPlanRow(
                variant_id=variant_id,
                config_path=str(row.config_path),
                config_sha256=row.config_sha256,
                run_root=str(run_root),
                run_id=f"ablation/nadig_jurkat/{variant_id}/seed-1",
                run_seed=row.run_seed,
                device=device,
                disposition=disposition,
                matrix_schema_version=row.matrix_schema_version,
                semantic_factor=row.semantic_factor,
                declared_parameter_diffs=row.declared_parameter_diffs,
                genept_preflight_receipt_path=(
                    str(genept_receipt_path) if row.genept_preflight_required else None
                ),
                genept_preflight_receipt_sha256=(
                    genept_receipt_sha256 if row.genept_preflight_required else None
                ),
                genept_preflight_schema_version=(
                    genept_receipt_schema if row.genept_preflight_required else None
                ),
                genept_preflight_status=(genept_status if row.genept_preflight_required else None),
                genept_preflight_artifact_sha256=(
                    genept_artifact_sha256 if row.genept_preflight_required else None
                ),
                genept_preflight_missing_target_ids_sha256=(
                    genept_missing_target_ids_sha256 if row.genept_preflight_required else None
                ),
            )
        )
    return tuple(plan)


def _command(
    row: AblationLaunchPlanRow,
    *,
    python_executable: str,
    data_root: Path,
    repository_root: Path,
    source_publication_receipt: Path | None,
    source_publication_receipt_sha256: str | None,
    source_publication_remote_ref: str,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "gradpert",
        "model",
        "pilot",
        "--config",
        row.config_path,
        "--data-root",
        str(data_root),
        "--run-root",
        row.run_root,
        "--run-id",
        row.run_id,
        "--run-seed",
        str(row.run_seed),
        "--device",
        row.device,
        "--repository-root",
        str(repository_root),
        "--formal",
        "--json",
    ]
    if source_publication_receipt is not None:
        assert source_publication_receipt_sha256 is not None
        command.extend(
            [
                "--source-publication-receipt",
                str(source_publication_receipt),
                "--source-publication-receipt-sha256",
                source_publication_receipt_sha256,
                "--source-publication-remote-ref",
                source_publication_remote_ref,
            ]
        )
    if row.genept_preflight_receipt_path is not None:
        assert row.genept_preflight_receipt_sha256 is not None
        command.extend(
            [
                "--genept-preflight-receipt",
                row.genept_preflight_receipt_path,
                "--genept-preflight-receipt-sha256",
                row.genept_preflight_receipt_sha256,
            ]
        )
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--genept-availability-receipt", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--source-publication-receipt", type=Path)
    parser.add_argument("--source-publication-receipt-sha256")
    parser.add_argument(
        "--source-publication-remote-ref",
        default="refs/heads/main",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve(strict=True)
    if (args.source_publication_receipt is None) != (
        args.source_publication_receipt_sha256 is None
    ):
        raise ValueError("publication receipt path and hash must be provided together")
    publication_receipt = (
        None
        if args.source_publication_receipt is None
        else args.source_publication_receipt.resolve(strict=True)
    )
    rows = load_ablation_matrix(args.matrix, repository_root=repository_root)
    plan = build_ablation_launch_plan(
        rows,
        selected_variants=args.variant,
        runs_root=args.runs_root,
        device=args.device,
        genept_availability_receipt=args.genept_availability_receipt,
        data_root=args.data_root,
    )
    plan_payload: dict[str, Any] = {
        "schema_version": "nadig-vnext-ablation-launch-plan-v1",
        "matrix_path": str(args.matrix.resolve(strict=True)),
        "matrix_sha256": sha256_file(args.matrix.resolve(strict=True)),
        "expected_source_commit": args.expected_source_commit,
        "source_publication_receipt": (
            None if publication_receipt is None else str(publication_receipt)
        ),
        "source_publication_receipt_sha256": args.source_publication_receipt_sha256,
        "source_publication_remote_ref": args.source_publication_remote_ref,
        "row_count": len(plan),
        "rows": [asdict(row) for row in plan],
    }
    if args.dry_run:
        print(json.dumps(plan_payload, sort_keys=True, separators=(",", ":")))
        return 0

    _require_git_source(repository_root, args.expected_source_commit)
    if publication_receipt is not None:
        expected_repository = load_experiment_config(plan[0].config_path).source_code.repository
        inspect_source_identity(
            repository_root,
            formal=True,
            expected_repository=expected_repository,
            publication_receipt=publication_receipt,
            expected_publication_receipt_sha256=args.source_publication_receipt_sha256,
            remote_ref=args.source_publication_remote_ref,
        )
    receipt_root = args.receipt_root.resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_root / "launch-plan.json", plan_payload)
    environment = os.environ.copy()
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    for row in plan:
        started = time.time()
        if row.disposition == "skip_genept_missing_target":
            atomic_json(
                receipt_root / f"{row.variant_id}.json",
                {
                    "schema_version": "nadig-vnext-ablation-row-v1",
                    **asdict(row),
                    "status": "skipped_before_model_construction",
                    "returncode": None,
                    "started_unix": started,
                    "completed_unix": time.time(),
                },
            )
            continue
        command = _command(
            row,
            python_executable=args.python_executable,
            data_root=args.data_root.resolve(strict=True),
            repository_root=repository_root,
            source_publication_receipt=publication_receipt,
            source_publication_receipt_sha256=args.source_publication_receipt_sha256,
            source_publication_remote_ref=args.source_publication_remote_ref,
        )
        result = subprocess.run(command, env=environment, check=False)
        atomic_json(
            receipt_root / f"{row.variant_id}.json",
            {
                "schema_version": "nadig-vnext-ablation-row-v1",
                **asdict(row),
                "status": "complete" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "command": command,
                "started_unix": started,
                "completed_unix": time.time(),
            },
        )
        if result.returncode != 0:
            return result.returncode
    return 0


__all__ = [
    "AblationLaunchPlanRow",
    "AblationMatrixRow",
    "build_ablation_launch_plan",
    "load_ablation_matrix",
    "main",
]
