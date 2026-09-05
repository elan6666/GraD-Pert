from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/performance/run_ablation_census_worker.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_ablation_census_worker_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load ablation census worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker() -> ModuleType:
    return _load_script()


@dataclass(frozen=True)
class _Metrics:
    step_wall_ms: float
    view_build_ms: float = 2.0


class _Event:
    def __init__(self, *, global_step: int, status: str) -> None:
        self.global_step = global_step
        self.phase_id = "views"
        self.status = status

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "gradpert-stage-event-v1",
            "global_step": self.global_step,
            "phase_id": self.phase_id,
            "status": self.status,
            "failure_type": "RuntimeError" if self.status == "failure" else None,
            "failure_message": "synthetic first-step OOM" if self.status == "failure" else None,
        }


class _Engine:
    def __init__(self) -> None:
        self.stage_observer = None
        self.stage_observer_failures: list[dict[str, object]] = []
        self.last_view_stats = {"effective_node_budget": 1405}
        self.topology = SimpleNamespace(gene_ids=tuple(f"gene-{index}" for index in range(256)))

    def train_step(self, batch, *, global_step: int):
        if self.stage_observer is not None:
            self.stage_observer(_Event(global_step=global_step, status="entered"), self)
            self.stage_observer(_Event(global_step=global_step, status="completed"), self)
        return _Metrics(step_wall_ms=float(global_step + 1))


class _Cuda:
    def reset_peak_memory_stats(self, _device) -> None:
        return None

    def synchronize(self, _device) -> None:
        return None

    def mem_get_info(self, _device) -> tuple[int, int]:
        return (12 * 1024**3, 20 * 1024**3)

    def memory_allocated(self, _device) -> int:
        return 3 * 1024**3

    def memory_reserved(self, _device) -> int:
        return 4 * 1024**3

    def max_memory_allocated(self, _device) -> int:
        return 5 * 1024**3

    def max_memory_reserved(self, _device) -> int:
        return 6 * 1024**3

    def memory_stats(self, _device) -> dict[str, int]:
        return {"num_alloc_retries": 0, "num_ooms": 0}


class _ProfilerInstance:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.steps = 0

    def start(self) -> None:
        self.started = True

    def step(self) -> None:
        self.steps += 1

    def stop(self) -> None:
        self.stopped = True

    def export_chrome_trace(self, path: str) -> None:
        Path(path).write_text('{"traceEvents":[]}\n', encoding="utf-8")

    def key_averages(self):
        return SimpleNamespace(table=lambda **_kwargs: "fake profiler table\n")


class _Profiler:
    ProfilerActivity = SimpleNamespace(CPU="cpu", CUDA="cuda")

    def __init__(self) -> None:
        self.instance: _ProfilerInstance | None = None
        self.schedule_payload: dict[str, int] | None = None

    def schedule(self, **kwargs):
        self.schedule_payload = kwargs
        return kwargs

    def profile(self, **_kwargs):
        self.instance = _ProfilerInstance()
        return self.instance


class _Torch:
    def __init__(self) -> None:
        self.cuda = _Cuda()
        self.profiler = _Profiler()

    def device(self, value: str) -> str:
        return value


class _Batch:
    def __init__(self, global_step: int) -> None:
        self.condition_ids = tuple(f"condition-{index % 5}" for index in range(256))
        self.perturbed_row_ids = tuple(f"perturbed-{global_step}-{index}" for index in range(256))
        self.control_row_ids = tuple(f"control-{global_step}-{index}" for index in range(256))
        self.anchors_by_condition = {
            f"condition-{index}": (index, index + 100) for index in range(5)
        }


class _Native:
    def __init__(self, *, engine_class: type[_Engine] = _Engine) -> None:
        self.CanonicalEvaluationData = object
        self.evaluate_validation_macro_delta = lambda: None
        self.requested_steps = 0
        self.validation_calls = 0
        self.engine_class = engine_class

    def run_native_experiment(self, **_kwargs) -> None:
        small_root = Path(_kwargs["run_root"]) / "small_results"
        small_root.mkdir(parents=True, exist_ok=True)
        names = [
            "config.resolved.yaml",
            "source_identity.json",
            "environment.json",
            "resolved_local_view_contract.json",
            "training_data.json",
            "run_meta.json",
        ]
        if _kwargs.get("genept_preflight_receipt") is not None:
            names.extend(["genept_preflight.json", "genept_feature.json"])
        for name in names:
            (small_root / name).write_text('{"sealed":true}\n', encoding="utf-8")
        with self.CanonicalEvaluationData(split_name="val") as validation:
            validation.configure_expression_cache(enabled=True)
            engine = self.engine_class()
            for global_step in range(100):
                self.requested_steps += 1
                engine.train_step(_Batch(global_step), global_step=global_step)
            self.validation_calls += 1
            self.evaluate_validation_macro_delta()


def _clean_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    if not repository.exists():
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        (repository / "tracked.txt").write_text("sealed\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-q",
                "-m",
                "sealed",
            ],
            check=True,
        )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def _args(tmp_path: Path, *, stage_id: str) -> Namespace:
    repository, commit = _clean_repository(tmp_path)
    source_publication_receipt = tmp_path / "source-publication-receipt.json"
    source_publication_receipt.write_text(
        '{"schema_version":"source-publication-receipt-v1"}\n',
        encoding="utf-8",
    )
    return Namespace(
        matrix=tmp_path / "matrix.json",
        expected_matrix_sha256="c" * 64,
        variant_id="a0_ratio_ring_half",
        stage_id=stage_id,
        device="cuda:0",
        data_root=tmp_path / "data",
        census_root=tmp_path / "census",
        repository_root=repository,
        development_commit=commit,
        source_publication_receipt=source_publication_receipt,
        source_publication_receipt_sha256=hashlib.sha256(
            source_publication_receipt.read_bytes()
        ).hexdigest(),
        p0_preflight_receipt=tmp_path / "p0.json",
        p0_preflight_receipt_sha256="d" * 64,
        batch_manifest=tmp_path / "batches.json",
        batch_manifest_sha256="e" * 64,
        p1_receipt=None,
        p1_receipt_sha256=None,
        genept_preflight_receipt=None,
        genept_preflight_receipt_sha256=None,
        minimum_gpu_headroom_fraction=0.15,
        minimum_gpu_free_bytes=4 * 1024**3,
        maximum_idle_gpu_utilization_percent=5.0,
        maximum_idle_gpu_memory_mib=1024,
        minimum_disk_free_bytes=20 * 1024**3,
        minimum_host_available_bytes=16 * 1024**3,
        as_json=False,
    )


def _repository_identity(worker: ModuleType, args: Namespace) -> dict[str, object]:
    identity = SimpleNamespace(
        commit=args.development_commit,
        dirty=False,
        tree_sha256="a" * 64,
        remote_url="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/main",
        published_commit=args.development_commit,
        formal_eligible=True,
        publication_receipt_sha256=args.source_publication_receipt_sha256,
    )
    original = worker.inspect_source_identity
    worker.inspect_source_identity = lambda *_args, **_kwargs: identity
    try:
        evidence = worker._repository_identity_evidence(args)
    finally:
        worker.inspect_source_identity = original
    assert all(evidence["predicates"].values())
    return evidence


def _source_publication_binding(args: Namespace) -> dict[str, object]:
    return {
        "publication_receipt_path": str(args.source_publication_receipt.resolve()),
        "publication_receipt_sha256": args.source_publication_receipt_sha256,
        "publication_receipt_size_bytes": args.source_publication_receipt.stat().st_size,
    }


def _binding(tmp_path: Path):
    return SimpleNamespace(
        matrix_path=str(tmp_path / "matrix.json"),
        matrix_row_index=0,
        variant_id="a0_ratio_ring_half",
        config_path=str(tmp_path / "config.yaml"),
        config_sha256="b" * 64,
        matrix_sha256="c" * 64,
        run_seed=1,
        genept_preflight_required=False,
        payload=lambda: {"variant_id": "a0_ratio_ring_half"},
    )


def _preflight() -> dict[str, object]:
    return {"predicates": {"all_safe": True}}


def _batch_manifest(worker: ModuleType, tmp_path: Path):
    topology = SimpleNamespace(gene_ids=tuple(f"gene-{index}" for index in range(256)))
    batches = tuple(
        worker.ordered_batch_identity(_Batch(step), global_step=step, topology=topology)
        for step in range(worker.census.EXACT_FROZEN_BATCH_COUNT)
    )
    return worker.census.FrozenBatchManifest(
        path=str(tmp_path / "batches.json"),
        sha256="e" * 64,
        matrix_path=str(tmp_path / "matrix.json"),
        matrix_sha256="c" * 64,
        config_path=str(tmp_path / "config.yaml"),
        config_sha256="b" * 64,
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        run_seed=1,
        epoch=0,
        batch_size=256,
        max_unique_conditions=8,
        epoch_step_count=582,
        frozen_prefix_count=110,
        batch_order_policy=worker.census.EXACT_BATCH_ORDER_POLICY,
        control_pairing_policy=worker.census.EXACT_CONTROL_PAIRING_POLICY,
        canonical_data_sha256="1" * 64,
        observation_order_sha256="2" * 64,
        split_content_sha256="3" * 64,
        ordered_training_row_ids_sha256="4" * 64,
        ordered_control_pools_sha256="5" * 64,
        runtime_graph_root="graphs/hvg512",
        runtime_graph_manifest_path=str(tmp_path / "data/graphs/hvg512/manifest.json"),
        runtime_graph_manifest_sha256="6" * 64,
        runtime_graph_gene_order_sha256="7" * 64,
        batch_sequence_sha256=worker.census.batch_sequence_sha256(batches),
        batches=batches,
    )


def _p0_binding() -> dict[str, object]:
    return {
        "receipt_path": "/sealed/p0.json",
        "receipt_sha256": "d" * 64,
        "row_payload_sha256": "8" * 64,
    }


def _training_only() -> dict[str, object]:
    return {
        "scope": "performance_training_only",
        "real_canonical_evaluation_constructor_count": 0,
        "validation_cache_materialized": False,
        "validation_callback_count": 0,
        "validation_accessed": False,
        "test_truth_accessed": False,
        "truth_access_attempts": [],
    }


def _native_identity_bindings(worker: ModuleType) -> dict[str, object]:
    names = (
        "config.resolved.yaml",
        "source_identity.json",
        "environment.json",
        "resolved_local_view_contract.json",
        "training_data.json",
        "run_meta.json",
    )
    files = [
        {
            "relative_path": f"native-run/small_results/{name}",
            "sha256": f"{index + 1:064x}",
            "size_bytes": 1,
        }
        for index, name in enumerate(names)
    ]
    return {
        "schema_version": "nadig-vnext-native-small-identity-bindings-v1",
        "candidate_names": list(worker._NATIVE_IDENTITY_CANDIDATES),
        "files": files,
        "ordered_bindings_sha256": worker._sha256_json(files),
    }


def _immutable_inputs_fixture(worker: ModuleType, tmp_path: Path):
    data_root = tmp_path / "data"
    protocol_root = data_root / "nadig_jurkat" / "within_cell_unseen_single"
    manifests_root = protocol_root / "manifests"
    canonical_root = protocol_root / "canonical"
    source_root = protocol_root / "source"
    graph_root = data_root / "graphs" / "hvg512"
    for root in (manifests_root, canonical_root, source_root, graph_root):
        root.mkdir(parents=True, exist_ok=True)

    def write(path: Path, payload: bytes) -> str:
        path.write_bytes(payload)
        return worker._sha256_file(path)

    canonical_manifest = manifests_root / "canonical.json"
    split_manifest = manifests_root / "split.json"
    source_manifest = manifests_root / "source.json"
    canonical_h5ad = canonical_root / "adata.h5ad"
    source_h5ad = source_root / "source.h5ad"
    canonical_manifest_sha = write(canonical_manifest, b'{"canonical":true}\n')
    split_manifest_sha = write(split_manifest, b'{"split":true}\n')
    source_manifest_sha = write(source_manifest, b'{"source":true}\n')
    canonical_h5ad_sha = write(canonical_h5ad, b"canonical-h5ad")
    source_h5ad_sha = write(source_h5ad, b"source-h5ad")

    gene_ids = ["gene-a", "gene-b"]
    gene_order_sha = worker._sha256_json(gene_ids)
    ranking_entries = [{"gene_id": value, "rank": index} for index, value in enumerate(gene_ids)]
    ranking_sha = worker._sha256_json(ranking_entries)
    go_path = graph_root / "go.npz"
    string_path = graph_root / "string.npz"
    source_hashes = {
        "go": write(go_path, b"sealed-go"),
        "string": write(string_path, b"sealed-string"),
    }
    topology_sha = worker._sha256_json(
        {"graph_gene_order_sha256": gene_order_sha, "sources": source_hashes}
    )
    manifest_payload = {
        "requested_hvg_count": 512,
        "source_artifact_sha256": source_hashes,
        "graph_gene_order_sha256": gene_order_sha,
        "topology_content_sha256": topology_sha,
        "normalized_dispersion_ranked_hvg_gene_ids": gene_ids,
        "normalized_dispersion_ranking_sha256": ranking_sha,
    }
    graph_manifest = graph_root / "manifest.json"
    graph_manifest.write_text(worker.json.dumps(manifest_payload), encoding="utf-8")
    graph_manifest_sha = worker._sha256_file(graph_manifest)
    (graph_root / "graph_gene_ids.txt").write_text("gene-a\ngene-b\n", encoding="utf-8")
    (graph_root / "hvg512_dispersion_ranking.json").write_text(
        worker.json.dumps({"ordered_entries": ranking_entries}), encoding="utf-8"
    )
    graph_artifacts = {
        "manifest": {
            "path": str(graph_manifest),
            "sha256": graph_manifest_sha,
            "size_bytes": graph_manifest.stat().st_size,
            "role": "runtime_graph_manifest",
        },
        "graph_gene_ids": {
            "path": str(graph_root / "graph_gene_ids.txt"),
            "sha256": worker._sha256_file(graph_root / "graph_gene_ids.txt"),
            "size_bytes": (graph_root / "graph_gene_ids.txt").stat().st_size,
            "role": "ordered_graph_gene_axis",
        },
        "hvg_dispersion_ranking": {
            "path": str(graph_root / "hvg512_dispersion_ranking.json"),
            "sha256": worker._sha256_file(graph_root / "hvg512_dispersion_ranking.json"),
            "size_bytes": (graph_root / "hvg512_dispersion_ranking.json").stat().st_size,
            "role": "hvg_dispersion_ranking_receipt",
        },
        "go": {
            "path": str(go_path),
            "sha256": source_hashes["go"],
            "size_bytes": go_path.stat().st_size,
            "role": "pruned_go_graph",
        },
        "string": {
            "path": str(string_path),
            "sha256": source_hashes["string"],
            "size_bytes": string_path.stat().st_size,
            "role": "pruned_string_graph",
        },
    }

    matrix_path = tmp_path / "matrix.json"
    config_path = tmp_path / "config.yaml"
    p0_path = tmp_path / "p0.json"
    batches_path = tmp_path / "batches.json"
    source_publication_receipt = tmp_path / "source-publication-receipt.json"
    matrix_sha = write(matrix_path, b'{"matrix":true}\n')
    config_sha = write(config_path, b"model: sealed\n")
    p0_sha = write(p0_path, b'{"p0":true}\n')
    batches_sha = write(batches_path, b'{"batches":true}\n')
    source_publication_receipt_sha = write(
        source_publication_receipt,
        b'{"schema_version":"source-publication-receipt-v1"}\n',
    )
    binding = _binding(tmp_path)
    binding.matrix_path = str(matrix_path)
    binding.matrix_sha256 = matrix_sha
    binding.config_path = str(config_path)
    binding.config_sha256 = config_sha
    manifest = replace(
        _batch_manifest(worker, tmp_path),
        path=str(batches_path),
        sha256=batches_sha,
        matrix_path=str(matrix_path),
        matrix_sha256=matrix_sha,
        config_path=str(config_path),
        config_sha256=config_sha,
        canonical_data_sha256=canonical_h5ad_sha,
        runtime_graph_root="graphs/hvg512",
        runtime_graph_manifest_path=str(graph_manifest),
        runtime_graph_manifest_sha256=graph_manifest_sha,
        runtime_graph_gene_order_sha256=gene_order_sha,
    )
    graph = {
        "root_path": str(graph_root),
        "manifest_path": str(graph_manifest),
        "manifest_file_sha256": graph_manifest_sha,
        "artifacts": graph_artifacts,
        "requested_hvg_count": 512,
        "graph_gene_order_sha256": gene_order_sha,
        "topology_content_sha256": topology_sha,
        "source_artifact_sha256": source_hashes,
    }
    p0 = {
        "receipt_path": str(p0_path),
        "receipt_sha256": p0_sha,
        "source": {
            "publication_receipt_path": str(source_publication_receipt),
            "publication_receipt_sha256": source_publication_receipt_sha,
            "publication_receipt_size_bytes": source_publication_receipt.stat().st_size,
        },
        "data": {
            "canonical_manifest_path": str(canonical_manifest),
            "canonical_manifest_sha256": canonical_manifest_sha,
            "canonical_manifest_size_bytes": canonical_manifest.stat().st_size,
            "canonical_data_path": str(canonical_h5ad),
            "canonical_data_sha256": canonical_h5ad_sha,
            "canonical_data_size_bytes": canonical_h5ad.stat().st_size,
            "split_manifest_path": str(split_manifest),
            "split_manifest_sha256": split_manifest_sha,
            "split_manifest_size_bytes": split_manifest.stat().st_size,
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_size_bytes": source_manifest.stat().st_size,
            "source_h5ad_path": str(source_h5ad),
            "source_h5ad_sha256": source_h5ad_sha,
            "source_h5ad_size_bytes": source_h5ad.stat().st_size,
            "artifacts": {
                "canonical_manifest": {
                    "path": str(canonical_manifest),
                    "sha256": canonical_manifest_sha,
                    "size_bytes": canonical_manifest.stat().st_size,
                    "role": "canonical_data_manifest",
                },
                "canonical_h5ad": {
                    "path": str(canonical_h5ad),
                    "sha256": canonical_h5ad_sha,
                    "size_bytes": canonical_h5ad.stat().st_size,
                    "role": "canonical_expression_and_metadata",
                },
                "split_manifest": {
                    "path": str(split_manifest),
                    "sha256": split_manifest_sha,
                    "size_bytes": split_manifest.stat().st_size,
                    "role": "canonical_condition_split",
                },
                "source_manifest": {
                    "path": str(source_manifest),
                    "sha256": source_manifest_sha,
                    "size_bytes": source_manifest.stat().st_size,
                    "role": "source_data_manifest",
                },
                "source_h5ad": {
                    "path": str(source_h5ad),
                    "sha256": source_h5ad_sha,
                    "size_bytes": source_h5ad.stat().st_size,
                    "role": "source_expression_and_metadata",
                },
            },
        },
        "row": {"graph": graph, "genept": {"status": "not_required"}},
        "a0_graph": graph,
    }
    return SimpleNamespace(
        data_root=data_root,
        binding=binding,
        manifest=manifest,
        p0=p0,
        go_path=go_path,
        source_h5ad=source_h5ad,
        source_publication_receipt=source_publication_receipt,
    )


def test_worker_cli_excludes_p3(worker: ModuleType) -> None:
    actions = {action.dest: action for action in worker._parser()._actions}
    choices = actions["stage_id"].choices
    assert set(choices) == {"p1_capacity", "p2_timing", "diagnostic_profile"}
    assert "p3_timing" not in choices
    assert actions["source_publication_receipt"].required is True
    assert actions["source_publication_receipt_sha256"].required is True


def test_p2_and_profile_require_paired_p1_prerequisite(worker: ModuleType, tmp_path: Path) -> None:
    p1_args = _args(tmp_path, stage_id="p1_capacity")
    worker._validate_args(p1_args)
    p2_args = _args(tmp_path, stage_id="p2_timing")
    with pytest.raises(worker.WorkerGateError, match="requires a P1"):
        worker._validate_args(p2_args)
    p2_args.p1_receipt = tmp_path / "p1.json"
    with pytest.raises(worker.WorkerGateError, match="supplied together"):
        worker._validate_args(p2_args)
    p2_args.p1_receipt_sha256 = "a" * 64
    worker._validate_args(p2_args)
    p1_args.p1_receipt = tmp_path / "p1.json"
    p1_args.p1_receipt_sha256 = "a" * 64
    with pytest.raises(worker.WorkerGateError, match="cannot accept"):
        worker._validate_args(p1_args)

    missing_publication_root = tmp_path / "missing-publication"
    missing_publication_root.mkdir()
    missing_publication = _args(missing_publication_root, stage_id="p1_capacity")
    missing_publication.source_publication_receipt = None
    with pytest.raises(worker.WorkerGateError, match="source publication receipt"):
        worker._validate_args(missing_publication)


def test_p2_prerequisite_is_hash_pinned_and_gpu_bound(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, stage_id="p2_timing")
    binding = _binding(tmp_path)
    manifest = _batch_manifest(worker, tmp_path)
    p0 = _p0_binding()
    prefix_sha = worker.census.batch_sequence_sha256(manifest.batches[:1])
    repository_identity = _repository_identity(worker, args)
    immutable_files = [{"label": "sealed", "path": "/sealed", "sha256": "a" * 64}]
    payload = {
        "schema_version": "nadig-vnext-performance-stage-v1",
        "status": "complete",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "stage_id": "p1_capacity",
        "protocol": worker.census.STAGE_PROTOCOLS["p1_capacity"].payload(),
        "variant_id": binding.variant_id,
        "config_sha256": binding.config_sha256,
        "matrix_sha256": binding.matrix_sha256,
        "binding": binding.payload(),
        "development_commit": args.development_commit,
        "attempted_batch_count": 1,
        "completed_step_count": 1,
        "observed_step_count": 1,
        "batch_sequence_sha256": prefix_sha,
        "frozen_batch_manifest": {
            "receipt_path": manifest.path,
            "receipt_sha256": manifest.sha256,
            "expected_batch_count": manifest.frozen_prefix_count,
            "expected_sequence_sha256": manifest.batch_sequence_sha256,
            "observed_prefix_count": 1,
            "observed_prefix_sha256": prefix_sha,
            "expected_prefix_sha256": prefix_sha,
            "prefix_matches": True,
        },
        "p0_preflight": {
            "receipt_sha256": p0["receipt_sha256"],
            "row_payload_sha256": p0["row_payload_sha256"],
        },
        "repository_identity": repository_identity,
        "final_repository_identity": repository_identity,
        "resource_preflight": {
            "schema_version": "nadig-vnext-performance-resource-preflight-v1",
            "selected_physical_gpu": {"uuid": "GPU-sealed"},
            "predicates": {
                "no_competing_compute_processes": True,
                "gpu_utilization_at_most_limit": True,
                "gpu_memory_used_at_most_limit": True,
                "disk_free_at_least_limit": True,
                "host_available_at_least_limit": True,
            },
        },
        "capacity_evidence": {
            "predicates": {
                "exact_observed_step_count": True,
                "zero_cuda_allocation_retries_or_ooms": True,
                "gpu_free_bytes_at_least_required_headroom": True,
            }
        },
        "persistent_pkl_scan": {
            "schema_version": "nadig-vnext-performance-zero-pkl-scan-v1",
            "passed": True,
            "persistent_pkl_count": 0,
            "ordered_relative_paths": [],
        },
        "final_immutable_input_evidence": {
            "schema_version": "nadig-vnext-performance-immutable-input-audit-v1",
            "file_count": 1,
            "files": immutable_files,
            "ordered_file_bindings_sha256": worker._sha256_json(immutable_files),
        },
        "instrumentation": {
            "timing_acceptance": False,
            "heavy_capacity_instrumentation": True,
            "torch_profiler_enabled": False,
            "step_timer": "native_train_step_cuda_synchronized_step_wall_ms",
            "stage_observer": "atomic_per_native_phase",
        },
        "torch_profiler_trace_sha256": None,
        "torch_profiler_table_sha256": None,
        "primary_failure": None,
        "teardown_failures": [],
        "batch_gate_failure": None,
        "batches": [manifest.batches[0].payload()],
        "steps": [
            {
                "global_step": 0,
                "phase": "measured",
                "batch_identity_sha256": manifest.batches[0].sha256,
            }
        ],
        "stage_evidence": {
            "stage_observer_failures": [],
            "terminal_stage_progress": {"sha256": "9" * 64},
        },
        "native_identity_receipts": _native_identity_bindings(worker),
        "training_only_evidence": _training_only(),
    }
    receipt_path = tmp_path / "p1-receipt.json"
    receipt_path.write_text(worker.json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.p1_receipt = receipt_path
    args.p1_receipt_sha256 = worker._sha256_file(receipt_path)
    resolved = worker._resolve_stage_prerequisite(
        args,
        binding=binding,
        batch_manifest=manifest,
        p0_preflight=p0,
    )
    assert resolved is not None
    assert resolved["physical_gpu_uuid"] == "GPU-sealed"
    worker._require_prerequisite_gpu(
        resolved,
        {"selected_physical_gpu": {"uuid": "GPU-sealed"}},
    )
    with pytest.raises(worker.WorkerGateError, match="differs from the P1"):
        worker._require_prerequisite_gpu(
            resolved,
            {"selected_physical_gpu": {"uuid": "GPU-other"}},
        )

    for _label, mutate in (
        (
            "protocol",
            lambda forged: forged["protocol"].__setitem__("measured_steps", 2),
        ),
        (
            "final source",
            lambda forged: forged["final_repository_identity"]["predicates"].__setitem__(
                "worktree_clean", False
            ),
        ),
        (
            "capacity",
            lambda forged: forged["capacity_evidence"]["predicates"].__setitem__(
                "zero_cuda_allocation_retries_or_ooms", False
            ),
        ),
        (
            "persistent PKL",
            lambda forged: forged["persistent_pkl_scan"].__setitem__("persistent_pkl_count", 1),
        ),
        (
            "native receipt",
            lambda forged: forged["native_identity_receipts"]["files"].pop(),
        ),
        (
            "immutable inputs",
            lambda forged: forged["final_immutable_input_evidence"].__setitem__("file_count", 0),
        ),
    ):
        forged = worker.json.loads(worker.json.dumps(payload))
        mutate(forged)
        receipt_path.write_text(worker.json.dumps(forged, sort_keys=True), encoding="utf-8")
        args.p1_receipt_sha256 = worker._sha256_file(receipt_path)
        with pytest.raises(worker.WorkerGateError):
            worker._resolve_stage_prerequisite(
                args,
                binding=binding,
                batch_manifest=manifest,
                p0_preflight=p0,
            )

    payload["status"] = "failed"
    receipt_path.write_text(worker.json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.p1_receipt_sha256 = worker._sha256_file(receipt_path)
    with pytest.raises(worker.WorkerGateError, match="completion evidence"):
        worker._resolve_stage_prerequisite(
            args,
            binding=binding,
            batch_manifest=manifest,
            p0_preflight=p0,
        )


def test_p0_preflight_binds_source_data_row_and_a0_graph(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    binding = _binding(tmp_path)
    manifest = _batch_manifest(worker, tmp_path)
    rows: list[dict[str, object]] = []
    for index in range(worker.census.MATRIX_ROW_COUNT):
        rows.append(
            {
                "matrix_row_index": index,
                "variant_id": f"placeholder-{index}",
                "status": "passed",
                "binding": {},
                "graph": {},
                "local_view_contract": {},
                "genept": {"status": "not_required"},
            }
        )
    rows[0] = {
        "matrix_row_index": 0,
        "variant_id": binding.variant_id,
        "status": "passed",
        "binding": binding.payload(),
        "graph": {
            "manifest_file_sha256": manifest.runtime_graph_manifest_sha256,
            "graph_gene_order_sha256": manifest.runtime_graph_gene_order_sha256,
        },
        "local_view_contract": {"effective_node_budget": 1404},
        "genept": {"status": "not_required"},
    }
    payload = {
        "schema_version": "nadig-vnext-performance-p0-preflight-v1",
        "status": "passed",
        "evidence_class": "performance_preflight_only",
        "scientific_completion": False,
        "matrix_sha256": binding.matrix_sha256,
        "matrix_row_count": worker.census.MATRIX_ROW_COUNT,
        "row_status_counts": {"blocked": 0, "passed": worker.census.MATRIX_ROW_COUNT},
        "cross_h_audit": {"status": "passed"},
        "source": {
            "repository_root": str(args.repository_root.resolve()),
            "expected_commit": args.development_commit,
            "observed_commit": args.development_commit,
            "source_tree_sha256": "a" * 64,
            "remote_url": "https://github.com/elan6666/GraD-Pert.git",
            "remote_ref": "refs/heads/main",
            "published_commit": args.development_commit,
            "formal_eligible": True,
            "publication_receipt_path": str(args.source_publication_receipt.resolve()),
            "publication_receipt_sha256": args.source_publication_receipt_sha256,
            "publication_receipt_size_bytes": args.source_publication_receipt.stat().st_size,
            "source_dirty": False,
        },
        "data": {
            "dataset_id": manifest.dataset_id,
            "protocol_id": manifest.protocol_id,
            "canonical_data_sha256": manifest.canonical_data_sha256,
            "observation_order_sha256": manifest.observation_order_sha256,
            "split_content_sha256": manifest.split_content_sha256,
        },
        "rows": rows,
        "forbidden_runtime": {
            "cuda_initialized": False,
            "model_constructed": False,
            "canonical_training_data_constructed": False,
            "canonical_validation_data_constructed": False,
            "canonical_test_data_constructed": False,
        },
    }
    receipt_path = tmp_path / "p0.json"
    receipt_path.write_text(worker.json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.p0_preflight_receipt = receipt_path
    args.p0_preflight_receipt_sha256 = worker._sha256_file(receipt_path)
    resolved = worker._resolve_p0_preflight(
        args,
        binding=binding,
        batch_manifest=manifest,
    )
    assert resolved["receipt_sha256"] == args.p0_preflight_receipt_sha256
    assert resolved["row_payload_sha256"] == worker._sha256_json(rows[0])

    payload["source"]["publication_receipt_size_bytes"] += 1
    receipt_path.write_text(worker.json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.p0_preflight_receipt_sha256 = worker._sha256_file(receipt_path)
    with pytest.raises(worker.WorkerGateError, match="source publication receipt size changed"):
        worker._resolve_p0_preflight(
            args,
            binding=binding,
            batch_manifest=manifest,
        )
    payload["source"]["publication_receipt_size_bytes"] -= 1

    payload["cross_h_audit"] = {"status": "blocked"}
    receipt_path.write_text(worker.json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.p0_preflight_receipt_sha256 = worker._sha256_file(receipt_path)
    with pytest.raises(worker.WorkerGateError, match="did not close"):
        worker._resolve_p0_preflight(
            args,
            binding=binding,
            batch_manifest=manifest,
        )

    genept_receipt_path = tmp_path / "genept-preflight.json"
    genept_receipt_path.write_bytes(b"sealed-receipt")
    genept_receipt_sha256 = worker._sha256_file(genept_receipt_path)
    binding.genept_preflight_required = True
    args.genept_preflight_receipt = genept_receipt_path
    args.genept_preflight_receipt_sha256 = genept_receipt_sha256
    payload["cross_h_audit"] = {"status": "passed"}
    rows[0]["genept"] = {
        "status": "passed",
        "receipt": {
            "path": str(genept_receipt_path),
            "sha256": genept_receipt_sha256,
        },
    }
    receipt_path.write_text(worker.json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.p0_preflight_receipt_sha256 = worker._sha256_file(receipt_path)
    with pytest.raises(worker.WorkerGateError, match="P0 GenePT receipt differs"):
        worker._resolve_p0_preflight(
            args,
            binding=binding,
            batch_manifest=manifest,
        )
    rows[0]["genept"]["receipt"]["size_bytes"] = genept_receipt_path.stat().st_size
    receipt_path.write_text(worker.json.dumps(payload, sort_keys=True), encoding="utf-8")
    args.p0_preflight_receipt_sha256 = worker._sha256_file(receipt_path)
    resolved = worker._resolve_p0_preflight(
        args,
        binding=binding,
        batch_manifest=manifest,
    )
    assert resolved["row"]["genept"]["receipt"]["size_bytes"] == len(b"sealed-receipt")


def test_preclaim_failure_is_append_only_and_does_not_claim_attempt_root(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    first = worker._write_preclaim_failure(args, worker.WorkerGateError("blocked P0"))
    second = worker._write_preclaim_failure(args, worker.WorkerGateError("blocked batch"))
    assert first != second
    assert first.is_file() and second.is_file()
    assert not any((args.census_root / args.variant_id).glob("p1_capacity/attempt-*"))
    payload = worker.json.loads(first.read_text(encoding="utf-8"))
    assert payload["attempt_root_claimed"] is False
    assert payload["cuda_runtime_loaded"] is False
    assert payload["primary_failure"]["message"] == "blocked P0"


def test_ordered_batch_identity_preserves_rows_conditions_controls_and_anchors(
    worker: ModuleType,
) -> None:
    topology = SimpleNamespace(gene_ids=tuple(f"gene-{index}" for index in range(256)))
    identity = worker.ordered_batch_identity(_Batch(3), global_step=3, topology=topology)
    assert identity.row_ids[:2] == ("perturbed-3-0", "perturbed-3-1")
    assert identity.condition_ids[:2] == ("condition-0", "condition-1")
    assert identity.control_row_ids[:2] == ("control-3-0", "control-3-1")
    assert identity.active_anchor_ids[:2] == (
        ("gene-0", "gene-100"),
        ("gene-1", "gene-101"),
    )
    assert identity.actual_batch_size == 256
    assert identity.unique_condition_count == 5


def test_ordered_batch_identity_is_stable_across_different_numeric_gene_axes(
    worker: ModuleType,
) -> None:
    first = _Batch(3)
    first_axis = tuple(f"gene-{index}" for index in range(256))
    first_identity = worker.ordered_batch_identity(
        first,
        global_step=3,
        topology=SimpleNamespace(gene_ids=first_axis),
    )

    second = _Batch(3)
    second_axis = tuple(reversed(first_axis))
    second.anchors_by_condition = {
        condition: tuple(second_axis.index(first_axis[index]) for index in indices)
        for condition, indices in first.anchors_by_condition.items()
    }
    second_identity = worker.ordered_batch_identity(
        second,
        global_step=3,
        topology=SimpleNamespace(gene_ids=second_axis),
    )
    assert second_identity.active_anchor_ids == first_identity.active_anchor_ids
    assert second_identity.sha256 == first_identity.sha256


def test_ordered_batch_identity_fails_on_anchor_axis_or_condition_mismatch(
    worker: ModuleType,
) -> None:
    topology = SimpleNamespace(gene_ids=("gene-a", "gene-b"))
    out_of_range = _Batch(0)
    with pytest.raises(worker.WorkerGateError, match="outside the topology"):
        worker.ordered_batch_identity(out_of_range, global_step=0, topology=topology)

    mismatched = _Batch(0)
    mismatched.anchors_by_condition.pop("condition-4")
    with pytest.raises(worker.WorkerGateError, match="conditions differ"):
        worker.ordered_batch_identity(mismatched, global_step=0, topology=topology)


@pytest.mark.parametrize(
    ("stage_id", "expected_steps", "expected_timings"),
    [("p1_capacity", 1, 0), ("p2_timing", 25, 20)],
)
def test_bounded_worker_reuses_native_path_and_never_reaches_validation(
    worker: ModuleType,
    tmp_path: Path,
    stage_id: str,
    expected_steps: int,
    expected_timings: int,
) -> None:
    native = _Native()
    runtime = worker.RuntimeModules(
        torch=_Torch(),
        native_execution=native,
        engine_class=_Engine,
    )
    attempt_root = tmp_path / stage_id / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        (args := _args(tmp_path, stage_id=stage_id)),
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        runtime=runtime,
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    assert state.primary_failure is None
    assert len(state.steps) == expected_steps
    assert len(state.batches) == expected_steps
    assert len(state.timing_samples_ms) == expected_timings
    assert native.requested_steps == expected_steps
    assert native.validation_calls == 0
    assert state.evaluation["validation_callback_count"] == 0
    assert state.evaluation["truth_access_attempts"] == []
    assert _Engine.train_step.__name__ == "train_step"
    if stage_id == "p1_capacity":
        progress = attempt_root / "stage-progress.json"
        assert progress.is_file()
        assert '"status": "complete"' in progress.read_text(encoding="utf-8")
    else:
        assert not (attempt_root / "stage-progress.json").exists()


def test_frozen_batch_mismatch_fails_before_optimizer_step(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    class CountingEngine(_Engine):
        original_train_step_calls = 0

        def train_step(self, batch, *, global_step: int):
            type(self).original_train_step_calls += 1
            return super().train_step(batch, global_step=global_step)

    native = _Native(engine_class=CountingEngine)
    manifest = _batch_manifest(worker, tmp_path)
    expected = manifest.batches[0]
    wrong_first = worker.census.OrderedBatchIdentity.create(
        global_step=0,
        row_ids=expected.row_ids,
        condition_ids=expected.condition_ids,
        control_row_ids=("wrong-control", *expected.control_row_ids[1:]),
        active_anchor_ids=expected.active_anchor_ids,
        actual_batch_size=expected.actual_batch_size,
        unique_condition_count=expected.unique_condition_count,
    )
    wrong_batches = (wrong_first, *manifest.batches[1:])
    manifest = replace(
        manifest,
        batches=wrong_batches,
        batch_sequence_sha256=worker.census.batch_sequence_sha256(wrong_batches),
    )
    args = _args(tmp_path, stage_id="p1_capacity")
    attempt_root = tmp_path / "mismatched-batch" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        args,
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=_Torch(), native_execution=native, engine_class=CountingEngine
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=manifest,
        p0_preflight=_p0_binding(),
    )
    assert isinstance(state.primary_failure, worker.WorkerGateError)
    assert "frozen semantic batch prefix" in str(state.primary_failure)
    assert native.requested_steps == 1
    assert CountingEngine.original_train_step_calls == 0
    assert state.steps == []
    assert state.batches == []
    assert state.batch_gate_failure is not None
    assert state.batch_gate_failure["batch_index"] == 0
    assert state.batch_gate_failure["optimizer_step_executed"] is False


def test_receipt_has_protocol_identity_batch_digest_and_capacity_evidence(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    native = _Native()
    args = _args(tmp_path, stage_id="p2_timing")
    binding = _binding(tmp_path)
    attempt_root = tmp_path / "receipt" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        args,
        binding=binding,
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=_Torch(), native_execution=native, engine_class=_Engine
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    state.final_repository_identity = state.repository_identity
    state.final_immutable_input_evidence = {
        "schema_version": "nadig-vnext-performance-immutable-input-audit-v1"
    }
    receipt = worker._build_stage_receipt(
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=_preflight(),
        state=state,
        p0_preflight=_p0_binding(),
        batch_manifest=_batch_manifest(worker, tmp_path),
        stage_prerequisite=None,
    )
    assert receipt["status"] == "complete"
    assert receipt["evidence_class"] == "performance_training_only"
    assert receipt["scientific_completion"] is False
    assert receipt["protocol"] == worker.census.STAGE_PROTOCOLS["p2_timing"].payload()
    assert receipt["observed_step_count"] == 25
    assert len(receipt["timing_samples_ms"]) == 20
    assert receipt["torch_profiler_trace_sha256"] is None
    assert receipt["batch_sequence_sha256"] == worker.census.batch_sequence_sha256(state.batches)
    assert all(receipt["capacity_evidence"]["predicates"].values())
    assert all(receipt["repository_identity"]["predicates"].values())
    assert receipt["persistent_pkl_scan"]["persistent_pkl_count"] == 0
    assert len(receipt["native_identity_receipts"]["files"]) == 6
    worker.census.require_training_only_evidence(receipt["training_only_evidence"])


def test_first_step_failure_preserves_attempted_batch_and_last_entered_stage(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    class FailingEngine(_Engine):
        def train_step(self, batch, *, global_step: int):
            assert self.stage_observer is not None
            self.stage_observer(_Event(global_step=global_step, status="entered"), self)
            self.stage_observer(_Event(global_step=global_step, status="failure"), self)
            raise RuntimeError("synthetic first-step OOM")

    native = _Native(engine_class=FailingEngine)
    args = _args(tmp_path, stage_id="p1_capacity")
    attempt_root = tmp_path / "failed-p1" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        args,
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=_Torch(), native_execution=native, engine_class=FailingEngine
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    assert isinstance(state.primary_failure, RuntimeError)
    assert str(state.primary_failure) == "synthetic first-step OOM"
    assert len(state.batches) == 1
    assert state.batches[0].global_step == 0
    assert state.steps == []
    progress = worker.json.loads((attempt_root / "stage-progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["last_entered_stage"] == "views"
    assert progress["last_failed_stage"] == "views"
    failure_event = progress["stage_events"][-1]
    assert failure_event["event"] == "failure"
    assert failure_event["telemetry"]["cpu_rss_bytes"] > 0
    assert failure_event["telemetry"]["gpu_total_bytes"] == 20 * 1024**3
    receipt = worker._build_stage_receipt(
        args,
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        resource_preflight=_preflight(),
        state=state,
        p0_preflight=_p0_binding(),
        batch_manifest=_batch_manifest(worker, tmp_path),
        stage_prerequisite=None,
    )
    assert receipt["attempted_batch_count"] == 1
    assert receipt["completed_step_count"] == 0
    assert len(receipt["batches"]) == 1
    progress_binding = receipt["stage_evidence"]["terminal_stage_progress"]
    assert progress_binding["sha256"] == worker._sha256_file(attempt_root / "stage-progress.json")


def test_diagnostic_uses_exact_profiler_schedule_without_timing_samples(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    native = _Native()
    torch = _Torch()
    args = _args(tmp_path, stage_id="diagnostic_profile")
    binding = _binding(tmp_path)
    attempt_root = tmp_path / "diagnostic" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        args,
        binding=binding,
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=torch,
            native_execution=native,
            engine_class=_Engine,
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    assert state.primary_failure is None
    assert len(state.steps) == 5
    assert state.timing_samples_ms == []
    assert torch.profiler.schedule_payload == {"wait": 1, "warmup": 1, "active": 3, "repeat": 1}
    assert torch.profiler.instance is not None
    assert torch.profiler.instance.started is True
    assert torch.profiler.instance.stopped is True
    assert torch.profiler.instance.steps == 5
    assert state.profiler_trace is not None and state.profiler_trace.is_file()
    assert state.profiler_table is not None and state.profiler_table.is_file()
    receipt = worker._build_stage_receipt(
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=_preflight(),
        state=state,
        p0_preflight=_p0_binding(),
        batch_manifest=_batch_manifest(worker, tmp_path),
        stage_prerequisite=None,
    )
    assert receipt["torch_profiler_trace_sha256"] is not None
    assert receipt["torch_profiler_table_sha256"] is not None


def test_non_256_batch_fails_closed(worker: ModuleType) -> None:
    batch = _Batch(0)
    batch.condition_ids = batch.condition_ids[:-1]
    batch.perturbed_row_ids = batch.perturbed_row_ids[:-1]
    batch.control_row_ids = batch.control_row_ids[:-1]
    with pytest.raises(worker.WorkerGateError, match="non-256"):
        worker.ordered_batch_identity(
            batch,
            global_step=0,
            topology=SimpleNamespace(gene_ids=tuple(f"gene-{index}" for index in range(256))),
        )


def test_genept_row_requires_hash_pinned_preflight_and_passes_it_to_native(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    binding.genept_preflight_required = True
    args = _args(tmp_path, stage_id="p1_capacity")
    with pytest.raises(worker.WorkerGateError, match="requires a hash-pinned"):
        worker._resolve_genept_preflight(args, binding=binding)

    receipt_path = tmp_path / "genept-preflight.json"
    receipt_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    receipt_sha256 = worker._sha256_file(receipt_path)
    args.genept_preflight_receipt = receipt_path
    args.genept_preflight_receipt_sha256 = receipt_sha256
    resolved = worker._resolve_genept_preflight(args, binding=binding)
    assert resolved == (receipt_path.resolve(), receipt_sha256)

    native = _Native()
    observed_kwargs: dict[str, object] = {}
    original_run = native.run_native_experiment

    def capture_run(**kwargs) -> None:
        observed_kwargs.update(kwargs)
        original_run(**kwargs)

    native.run_native_experiment = capture_run
    attempt_root = tmp_path / "genept" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        args,
        binding=binding,
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=_Torch(), native_execution=native, engine_class=_Engine
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=resolved,
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    assert state.primary_failure is None
    assert observed_kwargs["genept_preflight_receipt"] == receipt_path.resolve()
    assert observed_kwargs["genept_preflight_receipt_sha256"] == receipt_sha256


def test_repository_identity_records_tree_and_fails_dirty_or_mismatched_commit(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    identity = SimpleNamespace(
        commit=args.development_commit,
        dirty=False,
        tree_sha256="a" * 64,
        remote_url="https://github.com/elan6666/GraD-Pert.git",
        remote_ref="refs/heads/main",
        published_commit=args.development_commit,
        formal_eligible=True,
        publication_receipt_sha256=args.source_publication_receipt_sha256,
    )
    observed_kwargs: dict[str, object] = {}

    def inspect(*_args, **kwargs):
        observed_kwargs.update(kwargs)
        return identity

    monkeypatch.setattr(worker, "inspect_source_identity", inspect)
    p0_source = {
        "source_tree_sha256": identity.tree_sha256,
        "remote_url": identity.remote_url,
        "remote_ref": identity.remote_ref,
        "published_commit": identity.published_commit,
        **_source_publication_binding(args),
    }
    evidence = worker._repository_identity_evidence(args, expected_p0_source=p0_source)
    assert evidence["head_commit"] == args.development_commit
    assert len(evidence["head_tree"]) == 40
    assert evidence["status_porcelain"] == ""
    assert all(evidence["predicates"].values())
    assert observed_kwargs["publication_receipt"] == args.source_publication_receipt.resolve()
    assert (
        observed_kwargs["expected_publication_receipt_sha256"]
        == args.source_publication_receipt_sha256
    )

    args.development_commit = "f" * 40
    mismatched = worker._repository_identity_evidence(args, expected_p0_source=p0_source)
    assert mismatched["predicates"]["head_equals_development_commit"] is False
    (Path(args.repository_root) / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = worker._repository_identity_evidence(args, expected_p0_source=p0_source)
    assert dirty["predicates"]["worktree_clean"] is False
    assert "?? untracked.txt" in dirty["status_porcelain"]


@pytest.mark.parametrize(
    ("remote_url", "published_commit", "failed_predicate"),
    [
        (
            "https://github.com/example/wrong.git",
            None,
            "remote_url_equals_p0",
        ),
        (
            "https://github.com/elan6666/GraD-Pert.git",
            "f" * 40,
            "published_commit_equals_development_commit",
        ),
    ],
)
def test_repository_identity_rejects_wrong_remote_or_unpublished_commit(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_url: str,
    published_commit: str | None,
    failed_predicate: str,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    identity = SimpleNamespace(
        commit=args.development_commit,
        dirty=False,
        tree_sha256="a" * 64,
        remote_url=remote_url,
        remote_ref="refs/heads/main",
        published_commit=(
            args.development_commit if published_commit is None else published_commit
        ),
        formal_eligible=True,
        publication_receipt_sha256=args.source_publication_receipt_sha256,
    )
    monkeypatch.setattr(worker, "inspect_source_identity", lambda *_args, **_kwargs: identity)
    evidence = worker._repository_identity_evidence(
        args,
        expected_p0_source={
            "source_tree_sha256": "a" * 64,
            "remote_url": "https://github.com/elan6666/GraD-Pert.git",
            "remote_ref": "refs/heads/main",
            "published_commit": args.development_commit,
            **_source_publication_binding(args),
        },
    )
    assert evidence["predicates"][failed_predicate] is False


def test_source_publication_receipt_missing_wrong_or_mutated_fails_closed(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing_args = _args(missing_root, stage_id="p1_capacity")
    missing_args.source_publication_receipt.unlink()
    with pytest.raises(worker.WorkerGateError, match="source publication receipt is missing"):
        worker._repository_identity_evidence(missing_args)

    wrong_hash_root = tmp_path / "wrong-hash"
    wrong_hash_root.mkdir()
    wrong_hash_args = _args(wrong_hash_root, stage_id="p1_capacity")
    wrong_hash_args.source_publication_receipt_sha256 = "f" * 64
    with pytest.raises(worker.WorkerGateError, match="source publication receipt changed"):
        worker._repository_identity_evidence(wrong_hash_args)

    mutated_root = tmp_path / "mutated"
    mutated_root.mkdir()
    mutated_args = _args(mutated_root, stage_id="p1_capacity")
    original = mutated_args.source_publication_receipt.read_bytes()
    mutated_args.source_publication_receipt.write_bytes(b"x" + original[1:])
    with pytest.raises(worker.WorkerGateError, match="source publication receipt changed"):
        worker._repository_identity_evidence(mutated_args)


def test_persistent_pkl_scan_fails_without_deleting_and_hash_pins_paths(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    class PklNative(_Native):
        def run_native_experiment(self, **kwargs) -> None:
            pkl_path = Path(kwargs["run_root"]) / "temporary" / "cache.pkl"
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            pkl_path.write_bytes(b"persistent")
            super().run_native_experiment(**kwargs)

    args = _args(tmp_path, stage_id="p1_capacity")
    attempt_root = tmp_path / "pkl" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        args,
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=_Torch(), native_execution=PklNative(), engine_class=_Engine
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    assert isinstance(state.primary_failure, worker.WorkerGateError)
    assert state.persistent_pkl_scan["ordered_relative_paths"] == ["native-run/temporary/cache.pkl"]
    assert state.persistent_pkl_scan["ordered_relative_paths_sha256"] == worker._sha256_json(
        ["native-run/temporary/cache.pkl"]
    )
    assert (attempt_root / "native-run/temporary/cache.pkl").is_file()


def test_observer_telemetry_failure_does_not_replace_primary_training_failure(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingEngine(_Engine):
        def train_step(self, batch, *, global_step: int):
            assert self.stage_observer is not None
            self.stage_observer(_Event(global_step=global_step, status="entered"), self)
            raise RuntimeError("primary failure")

    def fail_telemetry(*_args, **_kwargs):
        raise RuntimeError("secondary telemetry failure")

    monkeypatch.setattr(worker, "_resource_telemetry", fail_telemetry)
    args = _args(tmp_path, stage_id="p1_capacity")
    attempt_root = tmp_path / "observer" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker._execute_bounded_native(
        args,
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=_Torch(),
            native_execution=_Native(engine_class=FailingEngine),
            engine_class=FailingEngine,
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    assert isinstance(state.primary_failure, RuntimeError)
    assert str(state.primary_failure) == "primary failure"
    assert state.stage_observer_failures
    assert state.stage_observer_failures[0]["message"] == "secondary telemetry failure"


def test_immutable_input_audit_detects_live_graph_and_data_tampering(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    frozen = _immutable_inputs_fixture(worker, tmp_path)
    evidence = worker._require_immutable_inputs(
        binding=frozen.binding,
        p0_preflight=frozen.p0,
        batch_manifest=frozen.manifest,
        stage_prerequisite=None,
        data_root=frozen.data_root,
    )
    assert evidence["schema_version"] == "nadig-vnext-performance-immutable-input-audit-v1"
    assert evidence["file_count"] == len(evidence["files"])
    assert evidence["file_count"] >= 12

    frozen.go_path.write_bytes(b"tampered-go")
    with pytest.raises(worker.WorkerGateError, match=r"go artifact (size )?changed"):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )

    frozen = _immutable_inputs_fixture(worker, tmp_path / "fresh")
    frozen.source_h5ad.write_bytes(b"tampered-source")
    with pytest.raises(worker.WorkerGateError, match=r"source_h5ad artifact (size )?changed"):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )

    frozen = _immutable_inputs_fixture(worker, tmp_path / "crlf")
    graph_axis = Path(frozen.p0["row"]["graph"]["artifacts"]["graph_gene_ids"]["path"])
    graph_axis.write_bytes(b"gene-a\r\ngene-b\r\n")
    with pytest.raises(
        worker.WorkerGateError,
        match=r"graph_gene_ids artifact (size )?changed",
    ):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )

    frozen = _immutable_inputs_fixture(worker, tmp_path / "reformat")
    ranking_path = Path(frozen.p0["row"]["graph"]["artifacts"]["hvg_dispersion_ranking"]["path"])
    ranking_payload = worker.json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking_path.write_text(worker.json.dumps(ranking_payload, indent=2), encoding="utf-8")
    with pytest.raises(
        worker.WorkerGateError,
        match=r"hvg_dispersion_ranking artifact (size )?changed",
    ):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )

    frozen = _immutable_inputs_fixture(worker, tmp_path / "wrong-size")
    source_artifact = frozen.p0["data"]["artifacts"]["source_h5ad"]
    source_artifact["size_bytes"] += 1
    frozen.p0["data"]["source_h5ad_size_bytes"] += 1
    with pytest.raises(worker.WorkerGateError, match="source_h5ad artifact size changed"):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )

    frozen = _immutable_inputs_fixture(worker, tmp_path / "wrong-graph-size")
    frozen.p0["row"]["graph"]["artifacts"]["go"]["size_bytes"] += 1
    with pytest.raises(worker.WorkerGateError, match="go artifact size changed"):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )


def test_candidate_graph_artifact_audit_has_no_hvg_ranking_dependency(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_root = tmp_path / "data" / "graphs" / "txpert-candidate"
    graph_root.mkdir(parents=True)
    candidate_path = tmp_path / "gears_gene_set.csv"
    candidate_path.write_bytes(b"sealed candidate source")
    gene_ids = [f"gene-{index:04d}" for index in range(9853)]
    gene_order_sha = worker._sha256_json(gene_ids)
    targets = gene_ids[:3]
    source_hashes = {}
    for source_name in ("go", "string"):
        path = graph_root / f"{source_name}.npz"
        path.write_bytes(f"sealed-{source_name}".encode())
        source_hashes[source_name] = worker._sha256_file(path)
    topology_sha = worker._sha256_json(
        {"graph_gene_order_sha256": gene_order_sha, "sources": source_hashes}
    )
    manifest = {
        "schema_version": "txpert-candidate-gene-graph-v1",
        "dataset_id": "nadig_jurkat",
        "protocol_id": "within_cell_unseen_single",
        "canonical_data_sha256": "1" * 64,
        "split_content_sha256": "2" * 64,
        "source_h5ad_sha256": "3" * 64,
        "source_registry_sha256": "4" * 64,
        "graph_axis_policy": "txpert_candidate_gene_universe",
        "selection_method": "frozen_txpert_gears_gene_set_order",
        "txpert_public_commit": worker.TXPERT_PUBLIC_COMMIT,
        "candidate_gene_set_path": str(candidate_path),
        "candidate_gene_set_sha256": worker.TXPERT_CANDIDATE_GENE_SET_SHA256,
        "requested_gene_count": 9853,
        "expression_gene_count": 5000,
        "candidate_gene_ids": gene_ids,
        "candidate_gene_order_sha256": gene_order_sha,
        "candidate_target_ids": targets,
        "candidate_target_order_sha256": worker._sha256_json(targets),
        "graph_gene_ids": gene_ids,
        "graph_gene_order_sha256": gene_order_sha,
        "graph_gene_count": 9853,
        "source_artifact_sha256": source_hashes,
        "source_pruned_nonself_edge_count": {"go": 1, "string": 1},
        "topology_content_sha256": topology_sha,
        "top_k_incoming_per_source": 20,
        "control_graph_node_included": False,
        "gene_feature_policy": "learned_id",
        "materialization_wall_ms": 1.0,
    }
    manifest_path = graph_root / "manifest.json"
    manifest_path.write_text(worker.json.dumps(manifest), encoding="utf-8")
    gene_axis_path = graph_root / "graph_gene_ids.txt"
    gene_axis_path.write_text("\n".join(gene_ids) + "\n", encoding="utf-8")

    roles = {
        "manifest": "runtime_graph_manifest",
        "graph_gene_ids": "ordered_graph_gene_axis",
        "go": "pruned_go_graph",
        "string": "pruned_string_graph",
    }
    paths = {
        "manifest": manifest_path,
        "graph_gene_ids": gene_axis_path,
        "go": graph_root / "go.npz",
        "string": graph_root / "string.npz",
    }
    artifacts = {
        artifact_id: {
            "path": str(path),
            "sha256": worker._sha256_file(path),
            "size_bytes": path.stat().st_size,
            "role": roles[artifact_id],
        }
        for artifact_id, path in paths.items()
    }
    graph = {
        "root_path": str(graph_root),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": artifacts["manifest"]["sha256"],
        "artifacts": artifacts,
        "requested_graph_gene_count": 9853,
        "graph_axis_policy": "txpert_candidate_gene_universe",
        "graph_axis_source_sha256": worker.TXPERT_CANDIDATE_GENE_SET_SHA256,
        "graph_gene_order_sha256": gene_order_sha,
        "topology_content_sha256": topology_sha,
        "candidate_target_order_sha256": worker._sha256_json(targets),
        "source_artifact_sha256": source_hashes,
    }
    real_sha256_file = worker._sha256_file
    monkeypatch.setattr(
        worker,
        "_sha256_file",
        lambda path, **kwargs: (
            worker.TXPERT_CANDIDATE_GENE_SET_SHA256
            if Path(path) == candidate_path
            else real_sha256_file(path, **kwargs)
        ),
    )

    evidence = worker._require_graph_artifacts(
        graph, label="selected row", allowed_root=(tmp_path / "data").resolve()
    )
    assert len(evidence) == 5
    assert not any("ranking" in str(item["label"]) for item in evidence)

    graph["graph_axis_source_sha256"] = "f" * 64
    with pytest.raises(worker.WorkerGateError, match="candidate graph identity"):
        worker._require_graph_artifacts(
            graph, label="selected row", allowed_root=(tmp_path / "data").resolve()
        )


def test_source_publication_receipt_is_rehashed_by_immutable_audit(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    frozen = _immutable_inputs_fixture(worker, tmp_path)
    evidence = worker._require_immutable_inputs(
        binding=frozen.binding,
        p0_preflight=frozen.p0,
        batch_manifest=frozen.manifest,
        stage_prerequisite=None,
        data_root=frozen.data_root,
    )
    labels = [binding["label"] for binding in evidence["files"]]
    assert labels.count("source publication receipt") == 1

    original = frozen.source_publication_receipt.read_bytes()
    frozen.source_publication_receipt.write_bytes(b"x" + original[1:])
    with pytest.raises(worker.WorkerGateError, match="source publication receipt changed"):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )


def test_genept_artifact_size_is_bound_by_p0(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    frozen = _immutable_inputs_fixture(worker, tmp_path)
    genept_receipt_path = tmp_path / "genept-preflight.json"
    genept_receipt_path.write_bytes(b"sealed-receipt")
    genept_path = tmp_path / "genept.npz"
    genept_path.write_bytes(b"genept")
    frozen.p0["row"]["genept"] = {
        "status": "passed",
        "receipt": {
            "path": str(genept_receipt_path),
            "sha256": worker._sha256_file(genept_receipt_path),
            "size_bytes": genept_receipt_path.stat().st_size,
        },
        "artifact": {
            "path": str(genept_path),
            "sha256": worker._sha256_file(genept_path),
            "size_bytes": genept_path.stat().st_size + 1,
        },
    }
    with pytest.raises(worker.WorkerGateError, match="GenePT selected artifact size changed"):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )


def test_genept_preflight_receipt_is_rehashed_by_immutable_audit(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    frozen = _immutable_inputs_fixture(worker, tmp_path)
    genept_receipt_path = tmp_path / "genept-preflight.json"
    genept_receipt_path.write_bytes(b"sealed-receipt")
    genept_path = tmp_path / "genept.npz"
    genept_path.write_bytes(b"genept")
    frozen.p0["row"]["genept"] = {
        "status": "passed",
        "receipt": {
            "path": str(genept_receipt_path),
            "sha256": worker._sha256_file(genept_receipt_path),
            "size_bytes": genept_receipt_path.stat().st_size,
        },
        "artifact": {
            "path": str(genept_path),
            "sha256": worker._sha256_file(genept_path),
            "size_bytes": genept_path.stat().st_size,
        },
    }
    evidence = worker._require_immutable_inputs(
        binding=frozen.binding,
        p0_preflight=frozen.p0,
        batch_manifest=frozen.manifest,
        stage_prerequisite=None,
        data_root=frozen.data_root,
    )
    labels = [binding["label"] for binding in evidence["files"]]
    assert labels.count("GenePT preflight receipt") == 1
    assert labels.count("GenePT selected artifact") == 1

    genept_receipt_path.write_bytes(b"edited-receipt")
    with pytest.raises(
        worker.WorkerGateError,
        match=r"GenePT preflight receipt (size )?changed",
    ):
        worker._require_immutable_inputs(
            binding=frozen.binding,
            p0_preflight=frozen.p0,
            batch_manifest=frozen.manifest,
            stage_prerequisite=None,
            data_root=frozen.data_root,
        )


def test_final_source_drift_fails_without_replacing_primary_failure(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    initial = _repository_identity(worker, args)
    state = worker.WorkerState(
        repository_identity=initial,
        primary_failure=RuntimeError("primary training failure"),
    )
    identity = SimpleNamespace(
        commit=args.development_commit,
        dirty=False,
        tree_sha256=initial["source_tree_sha256"],
        remote_url=initial["remote_url"],
        remote_ref=initial["remote_ref"],
        published_commit=args.development_commit,
        formal_eligible=True,
        publication_receipt_sha256=args.source_publication_receipt_sha256,
    )
    monkeypatch.setattr(worker, "inspect_source_identity", lambda *_args, **_kwargs: identity)
    (Path(args.repository_root) / "drift.txt").write_text("dirty\n", encoding="utf-8")
    worker._record_final_repository_identity(args, state=state)
    assert isinstance(state.primary_failure, RuntimeError)
    assert str(state.primary_failure) == "primary training failure"
    assert state.final_repository_identity is not None
    assert state.final_repository_identity["predicates"]["worktree_clean"] is False
    assert state.teardown_failures[-1]["stage"] == "final_repository_identity"
    clean_failure = worker.WorkerState(repository_identity=initial)
    worker._record_final_repository_identity(args, state=clean_failure)
    assert isinstance(clean_failure.primary_failure, worker.WorkerGateError)


def test_final_source_content_tree_detects_ignored_file_drift(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    initial = _repository_identity(worker, args)
    tree_sha256 = "b" * 64
    identity = SimpleNamespace(
        commit=args.development_commit,
        dirty=False,
        tree_sha256=tree_sha256,
        remote_url=initial["remote_url"],
        remote_ref=initial["remote_ref"],
        published_commit=args.development_commit,
        formal_eligible=True,
        publication_receipt_sha256=args.source_publication_receipt_sha256,
    )
    monkeypatch.setattr(worker, "inspect_source_identity", lambda *_args, **_kwargs: identity)
    state = worker.WorkerState(repository_identity=initial)
    worker._record_final_repository_identity(
        args,
        state=state,
        expected_p0_source={
            "source_tree_sha256": initial["source_tree_sha256"],
            "remote_url": initial["remote_url"],
            "remote_ref": initial["remote_ref"],
            "published_commit": args.development_commit,
            **_source_publication_binding(args),
        },
    )
    assert isinstance(state.primary_failure, worker.WorkerGateError)
    assert state.final_repository_identity is not None
    assert state.final_repository_identity["status_porcelain"] == ""
    assert state.final_repository_identity["predicates"]["source_content_tree_equals_p0"] is False


def test_nonfinite_metrics_fail_and_running_receipt_becomes_terminal(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    class NonFiniteEngine(_Engine):
        def train_step(self, batch, *, global_step: int):
            return _Metrics(step_wall_ms=float("nan"))

    args = _args(tmp_path, stage_id="p1_capacity")
    binding = _binding(tmp_path)
    attempt_root = tmp_path / "nonfinite" / "attempt-001"
    attempt_root.mkdir(parents=True)
    receipt_path = attempt_root / "stage-receipt.json"
    receipt_path.write_text('{"status":"running"}\n', encoding="utf-8")
    state = worker._execute_bounded_native(
        args,
        binding=binding,
        attempt_root=attempt_root,
        runtime=worker.RuntimeModules(
            torch=_Torch(),
            native_execution=_Native(engine_class=NonFiniteEngine),
            engine_class=NonFiniteEngine,
        ),
        resource_preflight=_preflight(),
        repository_identity=_repository_identity(worker, args),
        genept_preflight=(None, None),
        batch_manifest=_batch_manifest(worker, tmp_path),
        p0_preflight=_p0_binding(),
    )
    assert isinstance(state.primary_failure, worker.WorkerGateError)
    assert "non-finite float" in str(state.primary_failure)
    assert state.steps == []
    state.final_repository_identity = state.repository_identity
    receipt = worker._write_terminal_stage_receipt(
        receipt_path,
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=_preflight(),
        state=state,
        p0_preflight=_p0_binding(),
        batch_manifest=_batch_manifest(worker, tmp_path),
        stage_prerequisite=None,
    )
    assert receipt["status"] == "failed"
    on_disk = worker.json.loads(receipt_path.read_text(encoding="utf-8"))
    assert on_disk["status"] == "failed"
    assert "NaN" not in receipt_path.read_text(encoding="utf-8")


def test_receipt_construction_failure_writes_minimal_terminal_fallback(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    binding = _binding(tmp_path)
    attempt_root = tmp_path / "fallback" / "attempt-001"
    attempt_root.mkdir(parents=True)
    receipt_path = attempt_root / "stage-receipt.json"
    receipt_path.write_text('{"status":"running"}\n', encoding="utf-8")
    state = worker.WorkerState(primary_failure=RuntimeError("primary failure"))

    def fail_receipt_construction(*_args, **_kwargs):
        raise ValueError("receipt construction failed")

    monkeypatch.setattr(worker, "_build_stage_receipt", fail_receipt_construction)
    receipt = worker._write_terminal_stage_receipt(
        receipt_path,
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=_preflight(),
        state=state,
        p0_preflight=_p0_binding(),
        batch_manifest=_batch_manifest(worker, tmp_path),
        stage_prerequisite=None,
    )
    assert receipt["status"] == "failed"
    assert receipt["running_receipt_replaced"] is True
    assert receipt["primary_failure"] == {
        "type": "RuntimeError",
        "message": "primary failure",
    }
    assert receipt["receipt_construction_failure"] == {
        "type": "ValueError",
        "message": "receipt construction failed",
    }
    forbidden_measurement_fields = {
        "batch_sequence_sha256",
        "batches",
        "steps",
        "timing_samples_ms",
        "timing_summary_ms",
        "torch_profiler_trace_sha256",
        "torch_profiler_table_sha256",
    }
    assert forbidden_measurement_fields.isdisjoint(receipt)
    assert worker.json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_resource_failure_receipt_declares_that_native_runtime_never_started(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    binding = _binding(tmp_path)
    attempt_root = tmp_path / "resource-failure" / "attempt-001"
    attempt_root.mkdir(parents=True)
    state = worker.WorkerState(
        primary_failure=worker.WorkerGateError("physical GPU/host/disk preflight failed"),
        repository_identity=_repository_identity(worker, args),
    )
    state.final_repository_identity = state.repository_identity
    state.final_immutable_input_evidence = {
        "schema_version": "nadig-vnext-performance-immutable-input-audit-v1",
        "file_count": 1,
        "files": [
            {
                "label": "sealed matrix",
                "path": str((tmp_path / "matrix.json").resolve()),
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
        ],
        "ordered_file_bindings_sha256": "b" * 64,
    }
    state.native_identity_receipts = {
        "schema_version": "nadig-vnext-native-small-identity-bindings-v1",
        "candidate_names": [],
        "files": [],
        "ordered_bindings_sha256": worker._sha256_json([]),
    }
    state.persistent_pkl_scan = {
        "schema_version": "nadig-vnext-performance-zero-pkl-scan-v1",
        "attempt_root": str(attempt_root),
        "persistent_pkl_count": 0,
        "ordered_relative_paths": [],
        "ordered_relative_paths_sha256": worker._sha256_json([]),
        "passed": True,
    }
    resource = {
        "schema_version": "nadig-vnext-performance-resource-preflight-v1",
        "selected_physical_gpu": {"uuid": "GPU-test"},
        "predicates": {"host_available_at_least_limit": False},
    }

    receipt = worker._build_stage_receipt(
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=resource,
        state=state,
        p0_preflight=_p0_binding(),
        batch_manifest=_batch_manifest(worker, tmp_path),
        stage_prerequisite=None,
    )

    assert receipt["status"] == "failed"
    assert receipt["resource_preflight_failure"] is True
    assert receipt["native_runtime_started"] is False
    assert receipt["native_identity_receipts"]["files"] == []


def test_preclaim_rejects_non_sentinel_variant_before_other_prerequisites(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, stage_id="p1_capacity")
    args.variant_id = "e1_frozen_genept"
    binding = SimpleNamespace(variant_id=args.variant_id)
    monkeypatch.setattr(worker, "_validate_args", lambda _args: None)
    monkeypatch.setattr(worker.census, "bind_matrix_variant", lambda *_args, **_kwargs: binding)

    with pytest.raises(worker.WorkerGateError, match="sentinel or capacity-only"):
        worker._resolve_preclaim_inputs(args)


@pytest.mark.parametrize(
    "variant_id",
    [
        "h4_txpert_candidate_ratio_half",
        "m1_single_string_gat",
        "m2_single_string_transformer",
        "w2_string_fixed_prior",
        "w3_string_prior_residual",
        "ws_string_weight_shuffle",
        "o1_no_condition",
        "o2_no_masked_node",
        "o3_no_spread",
    ],
)
def test_preclaim_allows_explicit_capacity_only_rows_only_for_capacity(
    worker: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant_id: str,
) -> None:
    binding = SimpleNamespace(
        variant_id=variant_id,
        matrix_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    monkeypatch.setattr(worker, "_validate_args", lambda _args: None)
    monkeypatch.setattr(worker.census, "bind_matrix_variant", lambda *_args, **_kwargs: binding)

    args = _args(tmp_path, stage_id="p2_timing")
    args.variant_id = binding.variant_id
    with pytest.raises(worker.WorkerGateError, match="capacity-only"):
        worker._resolve_preclaim_inputs(args)

    args.stage_id = "p1_capacity"
    monkeypatch.setattr(
        worker.census,
        "load_frozen_batch_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("past allowlist")),
    )
    with pytest.raises(RuntimeError, match="past allowlist"):
        worker._resolve_preclaim_inputs(args)
