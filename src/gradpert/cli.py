"""Command-line entry point for validated GraD-Pert workflows."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from gradpert._version import __version__
from gradpert.config import verify_config_matrix
from gradpert.contracts import DatasetGraphManifest
from gradpert.data import (
    dataset_status,
    load_dataset_registry,
    prepare_dataset,
    refresh_dataset_protocol,
    verify_dataset_registry,
    verify_prepared_dataset,
)
from gradpert.data.registry import DATASET_IDS
from gradpert.graphs import (
    DatasetGraphLayout,
    load_graph_source_registry,
    materialize_dataset_graphs,
    verify_dataset_graphs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gradpert", description="GraD-Pert research CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Report the local runtime without changing it")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    config = subparsers.add_parser("config", help="Validate resolved experiment configs")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    verify = config_subparsers.add_parser("verify", help="Verify the exact config matrix")
    verify.add_argument("--all", action="store_true", required=True, dest="verify_all")
    verify.add_argument("--json", action="store_true", dest="as_json")
    verify.add_argument(
        "--root", type=Path, default=Path("configs/experiments"), help="Config matrix root"
    )

    data = subparsers.add_parser("data", help="Inspect the five-dataset registry")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    registry = data_subparsers.add_parser("registry", help="Verify frozen dataset sources")
    registry.add_argument("--verify", action="store_true", required=True)
    registry.add_argument("--json", action="store_true", dest="as_json")
    registry.add_argument(
        "--root", type=Path, default=Path("registry/datasets"), help="Dataset registry root"
    )
    for command_name, help_text in (
        ("status", "Report source and canonical readiness without mutation"),
        ("prepare", "Materialize receipt-backed canonical datasets"),
        ("refresh-protocol", "Refresh split/control receipts from the registry policy"),
        ("verify", "Recompute all canonical data integrity gates"),
    ):
        command = data_subparsers.add_parser(command_name, help=help_text)
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument("--all", action="store_true")
        selection.add_argument("--dataset", choices=DATASET_IDS)
        command.add_argument("--json", action="store_true", dest="as_json")
        command.add_argument("--data-root", type=Path, default=Path("data"))
        command.add_argument("--registry-root", type=Path, default=Path("registry/datasets"))
        if command_name == "prepare":
            command.add_argument(
                "--download",
                action="store_true",
                help="Download a missing source through the frozen registry URL",
            )

    graph = subparsers.add_parser("graph", help="Materialize frozen GO/STRING graphs")
    graph_subparsers = graph.add_subparsers(dest="graph_command", required=True)
    for command_name, help_text in (
        ("status", "Report graph readiness without mutation"),
        ("prepare", "Build independent dataset Top-20 graph artifacts"),
        ("verify", "Verify upstream sources and sealed graph artifacts"),
    ):
        command = graph_subparsers.add_parser(command_name, help=help_text)
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument("--all", action="store_true")
        selection.add_argument("--dataset", choices=DATASET_IDS)
        command.add_argument("--json", action="store_true", dest="as_json")
        command.add_argument("--data-root", type=Path, default=Path("data"))
        command.add_argument(
            "--dataset-registry-root",
            type=Path,
            default=Path("registry/datasets"),
        )
        command.add_argument(
            "--source-registry",
            type=Path,
            default=Path("registry/graphs/public_string_go.yaml"),
        )
        if command_name in {"prepare", "verify"}:
            command.add_argument("--official-checkout", type=Path, required=True)

    model = subparsers.add_parser("model", help="Run native model integration gates")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    fit_head = model_subparsers.add_parser(
        "fit-head",
        help="Choose the global prototype width from sustained real-step GPU memory",
    )
    fit_head.add_argument("--data-root", type=Path, required=True)
    fit_head.add_argument(
        "--dataset-registry-root",
        type=Path,
        default=Path("registry/datasets"),
    )
    fit_head.add_argument("--output", type=Path, required=True)
    fit_head.add_argument("--device", default="cuda:0")
    fit_head.add_argument("--batch-size", type=int, choices=(64, 256), default=256)
    fit_head.add_argument("--json", action="store_true", dest="as_json")
    for command_name in ("smoke", "pilot", "full"):
        run = model_subparsers.add_parser(
            command_name,
            help=f"Run the native {command_name} lifecycle on a compute server",
        )
        run.add_argument("--config", type=Path, required=True)
        run.add_argument("--data-root", type=Path, required=True)
        run.add_argument("--run-root", type=Path, required=True)
        run.add_argument("--run-id", required=True)
        run.add_argument("--run-seed", type=int, required=True)
        run.add_argument("--device", default="cuda:0")
        run.add_argument("--repository-root", type=Path, required=True)
        run.add_argument("--formal", action="store_true")
        run.add_argument("--development-commit")
        run.add_argument("--resume", action="store_true")
        run.add_argument("--json", action="store_true", dest="as_json")

    baseline = subparsers.add_parser(
        "baseline",
        help="Run one nonlearned baseline through the common evaluator",
    )
    baseline.add_argument("--config", type=Path, required=True)
    baseline.add_argument("--data-root", type=Path, required=True)
    baseline.add_argument("--run-root", type=Path, required=True)
    baseline.add_argument("--run-id", required=True)
    baseline.add_argument("--repository-root", type=Path, required=True)
    baseline.add_argument("--formal", action="store_true")
    baseline.add_argument("--development-commit")
    baseline.add_argument("--json", action="store_true", dest="as_json")

    evaluation = subparsers.add_parser(
        "evaluation",
        help="Materialize model-independent evaluator state",
    )
    evaluation_subparsers = evaluation.add_subparsers(
        dest="evaluation_command",
        required=True,
    )
    for command_name, help_text in (
        ("prepare-state", "Build frozen DE masks and reference arrays"),
        ("verify-state", "Verify frozen DE masks and reference arrays"),
    ):
        command = evaluation_subparsers.add_parser(command_name, help=help_text)
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument("--all", action="store_true")
        selection.add_argument("--dataset", choices=DATASET_IDS)
        command.add_argument("--data-root", type=Path, default=Path("data"))
        command.add_argument(
            "--dataset-registry-root",
            type=Path,
            default=Path("registry/datasets"),
        )
        command.add_argument("--json", action="store_true", dest="as_json")

    pilot = subparsers.add_parser("pilot", help="Prepare explicit performance-pilot inputs")
    pilot_subparsers = pilot.add_subparsers(dest="pilot_command", required=True)
    reduced_graph = pilot_subparsers.add_parser(
        "prepare-top500-graph",
        help="Directly recompute Nadig Jurkat Top-500 HVGs and build the reduced graph",
    )
    reduced_graph.add_argument("--data-root", type=Path, required=True)
    reduced_graph.add_argument(
        "--dataset-registry",
        type=Path,
        default=Path("registry/datasets/nadig_jurkat.yaml"),
    )
    reduced_graph.add_argument(
        "--source-registry",
        type=Path,
        default=Path("registry/graphs/public_string_go.yaml"),
    )
    reduced_graph.add_argument("--official-checkout", type=Path, required=True)
    reduced_graph.add_argument("--output", type=Path, required=True)
    reduced_graph.add_argument("--json", action="store_true", dest="as_json")
    vnext_graph = pilot_subparsers.add_parser(
        "prepare-hvg512-graph",
        help="Build the TxPert-style pre-split Nadig Jurkat HVG512-plus-target graph",
    )
    vnext_graph.add_argument("--data-root", type=Path, required=True)
    vnext_graph.add_argument(
        "--dataset-registry",
        type=Path,
        default=Path("registry/datasets/nadig_jurkat.yaml"),
    )
    vnext_graph.add_argument(
        "--source-registry",
        type=Path,
        default=Path("registry/graphs/public_string_go.yaml"),
    )
    vnext_graph.add_argument("--official-checkout", type=Path, required=True)
    vnext_graph.add_argument("--output", type=Path, required=True)
    vnext_graph.add_argument("--json", action="store_true", dest="as_json")
    genept_graph = pilot_subparsers.add_parser(
        "prepare-genept-graph",
        help="Verify GenePT coverage and re-prune an HVG512 runtime graph",
    )
    genept_graph.add_argument("--parent", type=Path, required=True)
    genept_graph.add_argument("--genept-artifact", type=Path, required=True)
    genept_graph.add_argument("--availability-receipt", type=Path, required=True)
    genept_graph.add_argument(
        "--source-registry",
        type=Path,
        default=Path("registry/graphs/public_string_go.yaml"),
    )
    genept_graph.add_argument("--official-checkout", type=Path, required=True)
    genept_graph.add_argument("--output", type=Path, required=True)
    genept_graph.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _doctor(as_json: bool) -> int:
    payload = {
        "gradpert_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "formal_compute_allowed": False,
        "note": "Local doctor is read-only; formal compute is server-only.",
    }
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def _selected_dataset_ids(args: argparse.Namespace) -> tuple[str, ...]:
    if args.all:
        return tuple(DATASET_IDS)
    if args.dataset is None:  # pragma: no cover - argparse requires one selector
        raise AssertionError("dataset selection is missing")
    return (str(args.dataset),)


def _data_operation(args: argparse.Namespace) -> int:
    entries = [
        load_dataset_registry(args.registry_root / f"{dataset_id}.yaml")
        for dataset_id in _selected_dataset_ids(args)
    ]
    if args.data_command == "status":
        payloads = [dataset_status(entry, args.data_root) for entry in entries]
    elif args.data_command == "prepare":
        payloads = [
            prepare_dataset(entry, args.data_root, allow_download=args.download).__dict__
            for entry in entries
        ]
    elif args.data_command == "refresh-protocol":
        payloads = [refresh_dataset_protocol(entry, args.data_root).__dict__ for entry in entries]
    elif args.data_command == "verify":
        payloads = [verify_prepared_dataset(entry, args.data_root).__dict__ for entry in entries]
    else:  # pragma: no cover - closed by parser choices
        raise AssertionError(f"unsupported data operation: {args.data_command}")
    report = {"count": len(payloads), "entries": payloads}
    if args.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        for payload in payloads:
            state = payload.get("canonical_state", payload.get("state", "unknown"))
            print(f"{payload['dataset_id']} {state}")
    return 0


def _graph_operation(args: argparse.Namespace) -> int:
    entries = [
        load_dataset_registry(args.dataset_registry_root / f"{dataset_id}.yaml")
        for dataset_id in _selected_dataset_ids(args)
    ]
    payloads: list[dict[str, object]] = []
    if args.graph_command == "status":
        for entry in entries:
            layout = DatasetGraphLayout(
                args.data_root,
                entry.dataset_id,
                entry.protocol_id,
            )
            state = "missing"
            if layout.manifest.is_file():
                manifest = DatasetGraphManifest.model_validate_json(
                    layout.manifest.read_text(encoding="utf-8")
                )
                state = manifest.state
            payloads.append(
                {
                    "dataset_id": entry.dataset_id,
                    "protocol_id": entry.protocol_id,
                    "state": state,
                }
            )
    else:
        source_registry = load_graph_source_registry(args.source_registry)
        for entry in entries:
            common = {
                "dataset_id": entry.dataset_id,
                "protocol_id": entry.protocol_id,
                "data_root": args.data_root,
                "source_registry_path": args.source_registry,
                "source_registry": source_registry,
                "official_checkout": args.official_checkout,
            }
            if args.graph_command == "prepare":
                manifest = materialize_dataset_graphs(**common)
            elif args.graph_command == "verify":
                manifest = verify_dataset_graphs(**common)
            else:  # pragma: no cover - closed by parser choices
                raise AssertionError(f"unsupported graph operation: {args.graph_command}")
            payloads.append(manifest.model_dump(mode="json"))
    report = {"count": len(payloads), "entries": payloads}
    if args.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        for payload in payloads:
            print(f"{payload['dataset_id']} {payload['state']}")
    return 0


def _evaluation_operation(args: argparse.Namespace) -> int:
    from gradpert.evaluation import load_evaluation_state, prepare_evaluation_state

    entries = [
        load_dataset_registry(args.dataset_registry_root / f"{dataset_id}.yaml")
        for dataset_id in _selected_dataset_ids(args)
    ]
    manifests: list[dict[str, object]] = []
    for entry in entries:
        if args.evaluation_command == "prepare-state":
            manifest = prepare_evaluation_state(
                dataset_id=entry.dataset_id,
                protocol_id=entry.protocol_id,
                data_root=args.data_root,
            )
        else:
            manifest = load_evaluation_state(
                dataset_id=entry.dataset_id,
                protocol_id=entry.protocol_id,
                data_root=args.data_root,
            ).manifest
        manifests.append(manifest.model_dump(mode="json"))
    report = {"count": len(manifests), "entries": manifests}
    if args.as_json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        for payload in manifests:
            print(f"{payload['dataset_id']} evaluation_state_ready")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process status code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.as_json)
    if args.command == "config" and args.config_command == "verify":
        report = verify_config_matrix(args.root)
        if args.as_json:
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        else:
            print(f"verified {report['count']} self-contained experiment configs")
            for entry in report["entries"]:
                print(f"{entry['model_id']} {entry['dataset_id']} {entry['sha256']}")
        return 0
    if args.command == "data" and args.data_command == "registry":
        report = verify_dataset_registry(args.root)
        if args.as_json:
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        else:
            print(f"verified {report['count']} frozen dataset source entries")
            for entry in report["entries"]:
                print(
                    f"{entry['dataset_id']} {entry['source_availability']} "
                    f"{entry['source_checksum']} {entry['source_url']}"
                )
        return 0
    if args.command == "data" and args.data_command in {
        "status",
        "prepare",
        "refresh-protocol",
        "verify",
    }:
        return _data_operation(args)
    if args.command == "graph" and args.graph_command in {"status", "prepare", "verify"}:
        return _graph_operation(args)
    if args.command == "pilot" and args.pilot_command == "prepare-top500-graph":
        from gradpert.pilots import materialize_recomputed_top500_graph

        reduced_manifest = materialize_recomputed_top500_graph(
            entry=load_dataset_registry(args.dataset_registry),
            data_root=args.data_root,
            destination=args.output,
            source_registry_path=args.source_registry,
            source_registry=load_graph_source_registry(args.source_registry),
            official_checkout=args.official_checkout,
        )
        payload = reduced_manifest.model_dump(mode="json")
        if args.as_json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(
                f"{reduced_manifest.dataset_id} "
                f"graph_gene_count={reduced_manifest.graph_gene_count} "
                f"top500_sha256={reduced_manifest.direct_top500_gene_order_sha256}"
            )
        return 0
    if args.command == "pilot" and args.pilot_command == "prepare-hvg512-graph":
        from gradpert.pilots import materialize_vnext_hvg512_graph

        vnext_manifest = materialize_vnext_hvg512_graph(
            entry=load_dataset_registry(args.dataset_registry),
            data_root=args.data_root,
            destination=args.output,
            source_registry_path=args.source_registry,
            source_registry=load_graph_source_registry(args.source_registry),
            official_checkout=args.official_checkout,
        )
        payload = vnext_manifest.model_dump(mode="json")
        if args.as_json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(
                f"{vnext_manifest.dataset_id} "
                f"graph_gene_count={vnext_manifest.graph_gene_count} "
                f"hvg_count={vnext_manifest.requested_hvg_count}"
            )
        return 0
    if args.command == "pilot" and args.pilot_command == "prepare-genept-graph":
        from gradpert.pilots import materialize_genept_vnext_graph

        genept_manifest = materialize_genept_vnext_graph(
            parent_root=args.parent,
            destination=args.output,
            genept_artifact_path=args.genept_artifact,
            availability_receipt_path=args.availability_receipt,
            source_registry=load_graph_source_registry(args.source_registry),
            official_checkout=args.official_checkout,
        )
        payload = genept_manifest.model_dump(mode="json")
        if args.as_json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(
                f"{genept_manifest.dataset_id} "
                f"genept_graph_gene_count={genept_manifest.graph_gene_count} "
                "removed_non_targets="
                f"{len(genept_manifest.genept_removed_non_target_gene_ids)}"
            )
        return 0
    if args.command == "model" and args.model_command == "fit-head":
        from gradpert.training.capacity import fit_global_prototype_head

        report = fit_global_prototype_head(
            data_root=args.data_root,
            dataset_registry_root=args.dataset_registry_root,
            output_path=args.output,
            device_name=args.device,
            batch_size=args.batch_size,
        )
        if args.as_json:
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        else:
            print(f"selected prototype_count={report['selected_prototype_count']}")
        return 0
    if args.command == "model" and args.model_command in {"smoke", "pilot", "full"}:
        from gradpert.execution.native import run_native_experiment

        if args.formal and args.development_commit is not None:
            parser.error("--development-commit is forbidden with --formal")
        if not args.formal and args.development_commit is None:
            parser.error("development execution requires --development-commit")
        result = run_native_experiment(
            config_path=args.config,
            data_root=args.data_root,
            run_root=args.run_root,
            run_id=args.run_id,
            run_seed=args.run_seed,
            mode=args.model_command,
            device_name=args.device,
            repository_root=args.repository_root,
            formal=args.formal,
            development_commit=args.development_commit,
            resume=args.resume,
        )
        payload = {
            "run_id": result.run_id,
            "run_root": str(result.run_root),
            "formal_eligible": result.run_manifest.formal_eligible,
            "status": result.run_manifest.status,
        }
        if args.as_json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(
                f"{result.run_id} {result.run_manifest.status} "
                f"formal_eligible={result.run_manifest.formal_eligible}"
            )
        return 0
    if args.command == "evaluation" and args.evaluation_command in {
        "prepare-state",
        "verify-state",
    }:
        return _evaluation_operation(args)
    if args.command == "baseline":
        from gradpert.execution.nonlearned import run_nonlearned_experiment

        if args.formal and args.development_commit is not None:
            parser.error("--development-commit is forbidden with --formal")
        if not args.formal and args.development_commit is None:
            parser.error("development execution requires --development-commit")
        baseline_result = run_nonlearned_experiment(
            config_path=args.config,
            data_root=args.data_root,
            run_root=args.run_root,
            run_id=args.run_id,
            repository_root=args.repository_root,
            formal=args.formal,
            development_commit=args.development_commit,
        )
        payload = {
            "run_id": baseline_result.run_id,
            "run_root": str(baseline_result.run_root),
            "formal_eligible": baseline_result.run_manifest.formal_eligible,
            "status": baseline_result.run_manifest.status,
        }
        if args.as_json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(
                f"{baseline_result.run_id} {baseline_result.run_manifest.status} "
                f"formal_eligible={baseline_result.run_manifest.formal_eligible}"
            )
        return 0
    parser.print_help(sys.stdout)
    return 0
