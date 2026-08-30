from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/performance/ablation_performance_census.py"
MATRIX = PROJECT_ROOT / "configs/ablations/nadig_jurkat/matrix.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ablation_performance_census_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load ablation performance census script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def census() -> ModuleType:
    return _load_script()


@pytest.fixture(scope="module")
def matrix_sha256() -> str:
    return hashlib.sha256(MATRIX.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bindings(census: ModuleType, matrix_sha256: str):
    return census.bind_matrix_variants(
        MATRIX,
        repository_root=PROJECT_ROOT,
        expected_matrix_sha256=matrix_sha256,
    )


def _batch(census: ModuleType, global_step: int, *, suffix: str = ""):
    batch_size = census.EXACT_TRAIN_BATCH_SIZE
    condition_ids = [f"condition-{index % 5}{suffix}" for index in range(batch_size)]
    return census.OrderedBatchIdentity.create(
        global_step=global_step,
        row_ids=[f"row-{global_step}-{index}{suffix}" for index in range(batch_size)],
        condition_ids=condition_ids,
        control_row_ids=[f"control-{global_step}-{index}{suffix}" for index in range(batch_size)],
        active_anchor_ids=[[f"anchor-{index % 7}{suffix}"] for index in range(batch_size)],
        actual_batch_size=batch_size,
        unique_condition_count=len(set(condition_ids)),
    )


def _batches(census: ModuleType, count: int):
    return tuple(_batch(census, index) for index in range(count))


def _batch_manifest_payload(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> dict[str, object]:
    batches = _batches(census, census.EXACT_FROZEN_BATCH_COUNT)
    runtime_graph_root = "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"
    graph_manifest_path = tmp_path / runtime_graph_root / "manifest.json"
    graph_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    graph_manifest_path.write_text('{"sealed": true}\n', encoding="utf-8")
    return {
        "schema_version": "nadig-vnext-performance-batch-manifest-v2",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "matrix_id": census.SUCCESSOR_MATRIX_ID,
        "matrix_path": bindings[0].matrix_path,
        "matrix_sha256": bindings[0].matrix_sha256,
        "a0_config_path": bindings[0].config_path,
        "a0_config_sha256": bindings[0].config_sha256,
        "dataset_id": "nadig_jurkat",
        "protocol_id": "within_cell_unseen_single",
        "run_seed": 1,
        "epoch": 0,
        "batch_size": 256,
        "max_unique_conditions": 8,
        "epoch_step_count": 582,
        "frozen_prefix_count": 110,
        "batch_order_policy": census.EXACT_BATCH_ORDER_POLICY,
        "control_pairing_policy": census.EXACT_CONTROL_PAIRING_POLICY,
        "canonical_data_sha256": "d" * 64,
        "observation_order_sha256": "f" * 64,
        "split_content_sha256": "e" * 64,
        "ordered_training_row_ids_sha256": "1" * 64,
        "ordered_control_pools_sha256": "2" * 64,
        "runtime_graph_root": runtime_graph_root,
        "runtime_graph_manifest_path": str(graph_manifest_path.resolve()),
        "runtime_graph_manifest_sha256": hashlib.sha256(
            graph_manifest_path.read_bytes()
        ).hexdigest(),
        "runtime_graph_gene_order_sha256": "4" * 64,
        "forbidden_runtime": {
            "cuda_imported_or_initialized": False,
            "expression_array_reads": 0,
            "model_constructed": False,
            "optimizer_constructed": False,
            "test_object_constructed": False,
            "validation_object_constructed": False,
        },
        "batch_sequence_sha256": census.batch_sequence_sha256(batches),
        "batches": [
            {**batch.payload(), "batch_identity_sha256": batch.sha256} for batch in batches
        ],
    }


def _frozen_batch_manifest(census: ModuleType, bindings, tmp_path: Path):
    payload = _batch_manifest_payload(census, bindings, tmp_path)
    path = tmp_path / "frozen-batches.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return census.load_frozen_batch_manifest(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_matrix_sha256=bindings[0].matrix_sha256,
        expected_config_sha256=bindings[0].config_sha256,
    )


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


def _native_identity_receipts(*, genept: bool) -> dict[str, object]:
    names = set(
        {
            "config.resolved.yaml",
            "source_identity.json",
            "environment.json",
            "resolved_local_view_contract.json",
            "training_data.json",
            "run_meta.json",
        }
    )
    if genept:
        names.update({"genept_preflight.json", "genept_feature.json"})
    return {
        "files": [{"relative_path": f"native-run/small_results/{name}"} for name in sorted(names)]
    }


def _repository_identity_payload() -> dict[str, object]:
    return {
        "schema_version": "nadig-vnext-performance-repository-identity-v1",
        "repository_root": "/sealed/source",
        "declared_development_commit": "a" * 40,
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "source_tree_sha256": "c" * 64,
        "remote_url": "https://github.com/elan6666/GraD-Pert.git",
        "remote_ref": "refs/heads/codex/vnext-performance",
        "published_commit": "a" * 40,
        "formal_eligible": True,
        "status_porcelain": "",
        "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "predicates": {
            "head_equals_development_commit": True,
            "worktree_clean": True,
            "formal_source_eligible": True,
            "published_commit_equals_development_commit": True,
            "remote_ref_equals_p0": True,
            "source_content_tree_equals_p0": True,
            "remote_url_equals_p0": True,
            "publication_receipt_equals_p0": True,
        },
    }


def _immutable_input_evidence(census: ModuleType) -> dict[str, object]:
    files = [
        {
            "label": "sealed matrix",
            "path": "/sealed/matrix.json",
            "sha256": "d" * 64,
            "size_bytes": 128,
        }
    ]
    return {
        "schema_version": "nadig-vnext-performance-immutable-input-audit-v1",
        "file_count": len(files),
        "files": files,
        "ordered_file_bindings_sha256": census._sha256_json(files),
    }


def _stage_payload(
    census: ModuleType,
    binding,
    stage_id: str,
    batch_manifest,
    *,
    p0_preflight_sha256: str,
    stage_prerequisite: dict[str, object] | None = None,
    status: str = "complete",
) -> dict[str, object]:
    protocol = census.STAGE_PROTOCOLS[stage_id]
    expected_batches = batch_manifest.batches
    observed_count = protocol.total_steps if status == "complete" else 0
    timing_samples = (
        [float(100 + index) for index in range(protocol.measured_steps)]
        if protocol.timing_acceptance and status == "complete"
        else []
    )
    repository_identity = _repository_identity_payload()
    immutable_input_evidence = _immutable_input_evidence(census)
    return {
        "schema_version": "nadig-vnext-performance-stage-v1",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "variant_id": binding.variant_id,
        "config_sha256": binding.config_sha256,
        "matrix_sha256": binding.matrix_sha256,
        "binding": binding.payload(),
        "stage_id": stage_id,
        "protocol": protocol.payload(),
        "status": status,
        "training_only_evidence": _training_only(),
        "instrumentation": {
            "timing_acceptance": protocol.timing_acceptance,
            "heavy_capacity_instrumentation": protocol.heavy_capacity_instrumentation,
            "torch_profiler_enabled": protocol.torch_profiler_enabled,
        },
        "attempted_batch_count": observed_count,
        "completed_step_count": observed_count,
        "observed_step_count": observed_count,
        "batches": [batch.payload() for batch in expected_batches[:observed_count]],
        "batch_sequence_sha256": census.batch_sequence_sha256(expected_batches[:observed_count]),
        "p0_preflight": {
            "receipt_sha256": p0_preflight_sha256,
            "preclaim_immutable_input_evidence": immutable_input_evidence,
        },
        "frozen_batch_manifest": {
            "receipt_path": batch_manifest.path,
            "receipt_sha256": batch_manifest.sha256,
            "expected_batch_count": batch_manifest.frozen_prefix_count,
            "expected_sequence_sha256": batch_manifest.batch_sequence_sha256,
            "observed_prefix_count": observed_count,
            "observed_prefix_sha256": census.batch_sequence_sha256(
                expected_batches[:observed_count]
            ),
            "expected_prefix_sha256": census.batch_sequence_sha256(
                expected_batches[:observed_count]
            ),
            "prefix_matches": True,
        },
        "stage_prerequisite": stage_prerequisite,
        "batch_gate_failure": None,
        "steps": [
            {
                "global_step": index,
                "phase": "warmup" if index < protocol.warmup_steps else "measured",
                "batch_identity_sha256": expected_batches[index].sha256,
            }
            for index in range(observed_count)
        ],
        "repository_identity": repository_identity,
        "final_repository_identity": repository_identity,
        "final_immutable_input_evidence": immutable_input_evidence,
        "resource_preflight": {
            "selected_physical_gpu": {"uuid": "GPU-test"},
            "predicates": {"idle": True},
        },
        "capacity_evidence": {"predicates": {"capacity": True}},
        "persistent_pkl_scan": {
            "passed": True,
            "persistent_pkl_count": 0,
        },
        "native_identity_receipts": _native_identity_receipts(
            genept=binding.genept_preflight_required
        ),
        "timing_samples_ms": timing_samples,
        "torch_profiler_trace_sha256": ("a" * 64 if stage_id == "diagnostic_profile" else None),
        "torch_profiler_table_sha256": ("b" * 64 if stage_id == "diagnostic_profile" else None),
        "primary_failure": None if status == "complete" else {"type": "RuntimeError"},
        "teardown_failures": [],
    }


def _fallback_stage_payload(
    census: ModuleType,
    binding,
    stage_id: str = "p1_capacity",
) -> dict[str, object]:
    return {
        "schema_version": "nadig-vnext-performance-stage-v1",
        "evidence_class": "performance_training_only",
        "scientific_completion": False,
        "variant_id": binding.variant_id,
        "config_sha256": binding.config_sha256,
        "matrix_sha256": binding.matrix_sha256,
        "binding": binding.payload(),
        "stage_id": stage_id,
        "protocol": census.STAGE_PROTOCOLS[stage_id].payload(),
        "attempt_root": "/sealed/attempt-001",
        "development_commit": "a" * 40,
        "status": "failed",
        "running_receipt_replaced": True,
        "completed_step_count": 0,
        "attempted_batch_count": 0,
        "primary_failure": {"type": "WorkerGateError", "message": "receipt failed"},
        "teardown_failures": [],
        "receipt_construction_failure": {
            "type": "ValueError",
            "message": "non-finite payload",
        },
    }


def _resource_preflight_failure_payload(
    census: ModuleType,
    binding,
    batch_manifest,
) -> dict[str, object]:
    payload = _stage_payload(
        census,
        binding,
        "p1_capacity",
        batch_manifest,
        p0_preflight_sha256="9" * 64,
        status="failed",
    )
    payload.update(
        {
            "resource_preflight_failure": True,
            "native_runtime_started": False,
            "resource_preflight": {
                "schema_version": "nadig-vnext-performance-resource-preflight-v1",
                "selected_physical_gpu": {"uuid": "GPU-test"},
                "predicates": {
                    "no_competing_compute_processes": True,
                    "gpu_utilization_at_most_limit": True,
                    "gpu_memory_used_at_most_limit": True,
                    "disk_free_at_least_limit": True,
                    "host_available_at_least_limit": False,
                },
            },
            "persistent_pkl_scan": {
                "schema_version": "nadig-vnext-performance-zero-pkl-scan-v1",
                "passed": True,
                "persistent_pkl_count": 0,
                "ordered_relative_paths": [],
            },
            "capacity_evidence": {
                "minimum_gpu_free_bytes": 0,
                "gpu_total_bytes": 0,
                "required_gpu_free_bytes": 4 * 1024**3,
                "cuda_retry_or_oom_counter_max": 0,
                "predicates": {
                    "exact_observed_step_count": False,
                    "zero_cuda_allocation_retries_or_ooms": True,
                    "gpu_free_bytes_at_least_required_headroom": False,
                },
            },
            "native_identity_receipts": {
                "schema_version": "nadig-vnext-native-small-identity-bindings-v1",
                "files": [],
            },
            "primary_failure": {
                "type": "WorkerGateError",
                "message": "physical GPU/host/disk preflight failed",
            },
        }
    )
    return payload


def _write_receipt(path: Path, payload: dict[str, object]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_stage_protocols_are_frozen_and_separate_timing_from_profile(census: ModuleType) -> None:
    protocols = census.STAGE_PROTOCOLS
    assert (protocols["p1_capacity"].warmup_steps, protocols["p1_capacity"].measured_steps) == (
        0,
        1,
    )
    assert protocols["p1_capacity"].heavy_capacity_instrumentation is True
    assert protocols["p1_capacity"].timing_acceptance is False
    assert (protocols["p2_timing"].warmup_steps, protocols["p2_timing"].measured_steps) == (
        5,
        20,
    )
    assert (protocols["p3_timing"].warmup_steps, protocols["p3_timing"].measured_steps) == (
        10,
        100,
    )
    assert protocols["diagnostic_profile"].profiler_schedule == {
        "wait": 1,
        "warmup": 1,
        "active": 3,
    }
    assert protocols["diagnostic_profile"].total_steps == 5
    assert protocols["diagnostic_profile"].timing_acceptance is False


def test_performance_sentinel_is_exact_ordered_and_keeps_scientific_matrix_intact(
    census: ModuleType,
    bindings,
) -> None:
    selected = census.bind_performance_sentinel(bindings)

    assert len(bindings) == 25
    assert tuple(binding.variant_id for binding in selected) == (
        "a0_ratio_ring_half",
        "h3_hvg5000_ratio_half",
        "l1_fanout_ratio_half",
        "l2_ring_half_count8",
        "m4_adaptive_source_gat",
        "w1_string_edge_feature",
        "d2_control_transformer",
        "e2_genept_id_residual",
    )
    assert len(selected) == 8
    assert sum(binding.genept_preflight_required for binding in selected) == 1
    assert selected[-1].genept_preflight_required is True
    assert not any(binding.variant_id.startswith("o") for binding in selected)
    assert set(census.PERFORMANCE_SENTINEL_ROLES) == {binding.variant_id for binding in selected}


def test_performance_worker_rejects_every_non_sentinel_matrix_row(
    census: ModuleType,
    bindings,
) -> None:
    selected_ids = set(census.PERFORMANCE_SENTINEL_VARIANT_IDS)
    for binding in bindings:
        if binding.variant_id in selected_ids:
            census.require_performance_sentinel_variant(binding.variant_id)
        else:
            with pytest.raises(ValueError, match="eight-row sentinel"):
                census.require_performance_sentinel_variant(binding.variant_id)


def test_sentinel_plan_cli_emits_only_hash_bound_representative_rows(
    census: ModuleType,
    matrix_sha256: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        census.main(
            [
                "--matrix",
                str(MATRIX),
                "--repository-root",
                str(PROJECT_ROOT),
                "--expected-matrix-sha256",
                matrix_sha256,
                "sentinel-plan",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "nadig-vnext-performance-sentinel-plan-v1"
    assert payload["scientific_matrix_row_count"] == 25
    assert payload["selected_row_count"] == 8
    assert payload["selected_variant_ids"] == list(census.PERFORMANCE_SENTINEL_VARIANT_IDS)
    assert [row["variant_id"] for row in payload["rows"]] == payload["selected_variant_ids"]
    assert all(row["performance_role"] for row in payload["rows"])
    assert payload["scientific_completion"] is False


def test_aggregate_distinguishes_unmeasured_sentinel_from_unselected_matrix_rows(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    batch_manifest = _frozen_batch_manifest(census, bindings, tmp_path)
    selected_ids = set(census.PERFORMANCE_SENTINEL_VARIANT_IDS)
    records = [
        {
            "variant_id": binding.variant_id,
            "state": (
                "unavailable_preflight"
                if binding.variant_id in selected_ids
                else "not_selected_performance_sentinel"
            ),
            "disposition_reason": "synthetic selection boundary",
            "stages": {},
        }
        for binding in bindings
    ]

    report = census.aggregate_census_report(
        bindings=bindings,
        row_records=records,
        batch_manifest=batch_manifest,
        p0_preflight_sha256="9" * 64,
    )

    assert report["performance_sentinel_id"] == census.PERFORMANCE_SENTINEL_ID
    assert report["selected_row_count"] == 8
    assert report["selected_variant_ids"] == list(census.PERFORMANCE_SENTINEL_VARIANT_IDS)
    assert len(report["disposition_sections"]["not_selected_performance_sentinel"]) == 17


def test_exact_matrix_binding_covers_25_rows_and_rejects_tampering(
    census: ModuleType,
    bindings,
    matrix_sha256: str,
) -> None:
    assert len(bindings) == 25
    assert bindings[0].variant_id == census.A0_VARIANT_ID
    assert [binding.matrix_row_index for binding in bindings] == list(range(25))
    assert all(binding.run_seed == 1 for binding in bindings)
    with pytest.raises(ValueError, match="matrix SHA-256 differs"):
        census.bind_matrix_variants(
            MATRIX,
            repository_root=PROJECT_ROOT,
            expected_matrix_sha256="0" * 64,
        )
    selected = census.bind_matrix_variant(
        MATRIX,
        repository_root=PROJECT_ROOT,
        expected_matrix_sha256=matrix_sha256,
        variant_id="d2_control_transformer",
    )
    assert selected.semantic_factor == "decoder_mode"


def test_batch_identity_is_order_sensitive_and_prefix_bound(census: ModuleType) -> None:
    expected = _batches(census, 3)
    assert census.batch_sequence_sha256(expected) == census.batch_sequence_sha256(expected)
    census.require_batch_prefix(expected[:2], expected)
    changed = list(expected[:2])
    changed[1] = _batch(census, 1, suffix="-changed")
    with pytest.raises(ValueError, match="frozen prefix"):
        census.require_batch_prefix(changed, expected)
    with pytest.raises(ValueError, match="zero-based contiguous"):
        census.batch_sequence_sha256((expected[1],))


@pytest.mark.parametrize(
    "field",
    ["global_step", "actual_batch_size", "unique_condition_count"],
)
def test_batch_identity_rejects_boolean_numeric_fields(
    census: ModuleType,
    field: str,
) -> None:
    payload = _batch(census, 0).payload()
    payload[field] = True
    with pytest.raises(ValueError, match="numeric identity fields"):
        census.OrderedBatchIdentity.from_payload(payload)
    with pytest.raises(ValueError, match="plain integers"):
        census.OrderedBatchIdentity.create(**payload)


def test_batch_identity_rejects_empty_active_anchor_group(census: ModuleType) -> None:
    payload = _batch(census, 0).payload()
    active_anchor_ids = payload["active_anchor_ids"]
    assert isinstance(active_anchor_ids, list)
    active_anchor_ids[0] = []
    with pytest.raises(ValueError, match="anchor groups must be nonempty"):
        census.OrderedBatchIdentity.from_payload(payload)


def test_batch_manifest_is_exact_count_hash_pinned_and_identity_bound(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    payload = _batch_manifest_payload(census, bindings, tmp_path)
    path = tmp_path / "batch-manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = census.load_frozen_batch_manifest(
        path,
        expected_sha256=expected_sha256,
        expected_matrix_sha256=bindings[0].matrix_sha256,
        expected_config_sha256=bindings[0].config_sha256,
    )
    assert len(manifest.batches) == 110
    assert manifest.batch_sequence_sha256 == payload["batch_sequence_sha256"]
    assert manifest.matrix_path == str(MATRIX.resolve())
    assert manifest.config_path == str(Path(bindings[0].config_path).resolve())
    assert manifest.runtime_graph_root == payload["runtime_graph_root"]
    assert manifest.runtime_graph_manifest_path == payload["runtime_graph_manifest_path"]

    payload["batches"] = payload["batches"][:-1]
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="rows are malformed"):
        census.load_frozen_batch_manifest(path)


def test_freeze_batches_uses_metadata_only_api_and_semantic_gene_anchors(
    census: ModuleType,
    matrix_sha256: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Other server tests legitimately import Torch during collection.  This
    # functional unit isolates the fresh-process precondition; the fail-closed
    # production guard is exercised separately below.
    monkeypatch.delitem(census.sys.modules, "torch", raising=False)
    graph_root = tmp_path / "vnext/graph_axes/nadig_jurkat/hvg512_plus_targets"
    graph_root.mkdir(parents=True)
    (graph_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    topology = SimpleNamespace(gene_ids=("G0", "G1", "PERT"))
    graph_manifest = SimpleNamespace(
        requested_hvg_count=512,
        graph_gene_count=3,
        graph_gene_order_sha256=census.sha256_json(["G0", "G1", "PERT"]),
    )
    monkeypatch.setattr(
        census,
        "load_vnext_graph_topology",
        lambda _root: (topology, graph_manifest),
    )

    conditions = tuple("PERT" if index % 2 == 0 else "G0+PERT" for index in range(256))
    specs = tuple(
        SimpleNamespace(
            perturbed_row_ids=tuple(f"row-{step}-{index}" for index in range(256)),
            control_row_ids=tuple(f"control-{step}-{index}" for index in range(256)),
            condition_ids=conditions,
            anchor_gene_ids_by_condition={
                "PERT": ("PERT",),
                "G0+PERT": ("G0", "PERT"),
            },
        )
        for step in range(582)
    )

    class FakeTrainingData:
        attempt_expression_read = False

        def __init__(self, **kwargs) -> None:
            assert kwargs["graph_gene_ids_override"] == topology.gene_ids
            self.manifest = SimpleNamespace(
                canonical_adata_sha256="d" * 64,
                observation_order_sha256="f" * 64,
            )
            self.split = SimpleNamespace(split_content_sha256="e" * 64)
            self.row_ids = ("train-0", "train-1")
            self.train_row_indices = (0, 1)
            self.control_pools = {"ctx": ("control-0", "control-1")}

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def require_experiment_data_contract(self, **_kwargs: object) -> None:
            return None

        def training_batch_identity_specs(self, **kwargs):
            assert kwargs == {"epoch": 0, "batch_size": 256, "max_unique_conditions": 8}
            if self.attempt_expression_read:
                self._read_expression_indices((0,))
            return specs

        def _read_expression_indices(self, _indices: object) -> object:
            raise AssertionError("freeze guard did not intercept an expression read")

    monkeypatch.setattr(census, "CanonicalTrainingData", FakeTrainingData)
    payload = census.freeze_batch_manifest(
        matrix_path=MATRIX,
        repository_root=PROJECT_ROOT,
        expected_matrix_sha256=matrix_sha256,
        data_root=tmp_path,
    )
    assert payload["frozen_prefix_count"] == 110
    assert len(payload["batches"]) == 110
    first = payload["batches"][0]
    assert first["active_anchor_ids"][:2] == [["PERT"], ["G0", "PERT"]]
    assert payload["forbidden_runtime"]["expression_array_reads"] == 0
    assert payload["forbidden_runtime"]["cuda_imported_or_initialized"] is False
    assert payload["runtime_graph_manifest_path"] == str((graph_root / "manifest.json").resolve())

    FakeTrainingData.attempt_expression_read = True
    with pytest.raises(RuntimeError, match="attempted to read expression arrays"):
        census.freeze_batch_manifest(
            matrix_path=MATRIX,
            repository_root=PROJECT_ROOT,
            expected_matrix_sha256=matrix_sha256,
            data_root=tmp_path,
        )


def test_freeze_batches_rejects_a_loaded_torch_runtime(
    census: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(census.sys.modules, "torch", object())
    with pytest.raises(ValueError, match="before importing the CUDA runtime surface"):
        census.freeze_batch_manifest(
            matrix_path=MATRIX,
            repository_root=PROJECT_ROOT,
            expected_matrix_sha256="0" * 64,
            data_root=tmp_path,
        )


def test_evidence_output_claim_is_exclusive_and_preserves_first_bytes(
    census: ModuleType,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "batch-manifest.json"
    census._claim_json_output(destination, {"status": "claimed", "owner": "first"})
    sealed = destination.read_bytes()
    with pytest.raises(FileExistsError):
        census._claim_json_output(destination, {"status": "claimed", "owner": "second"})
    assert destination.read_bytes() == sealed


def test_freeze_cli_preserves_structured_failure_after_claim(
    census: ModuleType,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "failed-batch-manifest.json"
    with pytest.raises(ValueError):
        census.main(
            [
                "--matrix",
                str(MATRIX),
                "--repository-root",
                str(PROJECT_ROOT),
                "--expected-matrix-sha256",
                "0" * 64,
                "freeze-batches",
                "--data-root",
                str(tmp_path),
                "--output",
                str(destination),
            ]
        )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["scientific_completion"] is False
    assert payload["primary_failure"]["type"] == "ValueError"


def test_training_only_evidence_fails_closed(census: ModuleType) -> None:
    census.require_training_only_evidence(_training_only())
    accessed = _training_only()
    accessed["validation_accessed"] = True
    with pytest.raises(ValueError, match="validation_accessed"):
        census.require_training_only_evidence(accessed)


def _stable_promotion(census: ModuleType, *, selected: bool = False):
    return census.decide_p3_promotion(
        variant_id="m1_single_string_gat",
        step_wall_ms=[100.0] * 20,
        reserved_gpu_bytes=[10 * census.GIB] * 20,
        free_gpu_bytes=[50 * census.GIB] * 20,
        total_gpu_bytes=80 * census.GIB,
        selected_implementation_target=selected,
    )


def test_promotion_is_stable_when_no_preregistered_trigger_fires(census: ModuleType) -> None:
    decision = _stable_promotion(census)
    assert decision.promoted is False
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("step_wall", "reserved", "free", "selected", "expected_reason"),
    [
        (
            [100.0, 130.0] * 10,
            [10] * 20,
            [50] * 20,
            False,
            "relative_mad_above_limit",
        ),
        (
            [100.0] * 8 + [130.0] * 2 + [100.0] * 8 + [130.0] * 2,
            [10] * 20,
            [50] * 20,
            False,
            "p95_over_p50_above_limit",
        ),
        (
            [100.0] * 10 + [115.0] * 10,
            [10] * 20,
            [50] * 20,
            False,
            "half_drift_above_limit",
        ),
        (
            [100.0] * 20,
            [10] * 15 + [11] * 5,
            [50] * 20,
            False,
            "reserved_memory_growth_above_limit",
        ),
        (
            [100.0] * 20,
            [10] * 20,
            [13] * 20,
            False,
            "near_absolute_gpu_headroom",
        ),
        (
            [100.0] * 20,
            [10] * 20,
            [50] * 20,
            True,
            "selected_implementation_target",
        ),
    ],
)
def test_every_preregistered_p3_promotion_trigger(
    census: ModuleType,
    step_wall: list[float],
    reserved: list[int],
    free: list[int],
    selected: bool,
    expected_reason: str,
) -> None:
    decision = census.decide_p3_promotion(
        variant_id="m1_single_string_gat",
        step_wall_ms=step_wall,
        reserved_gpu_bytes=[value * census.GIB for value in reserved],
        free_gpu_bytes=[value * census.GIB for value in free],
        total_gpu_bytes=80 * census.GIB,
        selected_implementation_target=selected,
    )
    assert decision.promoted is True
    assert expected_reason in decision.reasons


def test_non_a0_promotion_always_pairs_the_reference(census: ModuleType) -> None:
    a0 = census.PromotionDecision(
        variant_id=census.A0_VARIANT_ID,
        promoted=False,
        reasons=(),
        statistics={},
    )
    candidate = _stable_promotion(census, selected=True)
    paired = census.pair_a0_promotion((a0, candidate))
    assert paired[0].promoted is True
    assert "paired_reference_a0" in paired[0].reasons
    assert paired[1].promoted is True


def test_attempt_roots_are_fresh_numbered_and_fail_on_unexpected_entries(
    census: ModuleType,
    tmp_path: Path,
) -> None:
    first = census.claim_fresh_attempt_root(
        tmp_path,
        variant_id=census.A0_VARIANT_ID,
        stage_id="p1_capacity",
    )
    second = census.claim_fresh_attempt_root(
        tmp_path,
        variant_id=census.A0_VARIANT_ID,
        stage_id="p1_capacity",
    )
    assert first.name == "attempt-001"
    assert second.name == "attempt-002"
    (second.parent / "manual-output.txt").write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected entry"):
        census.claim_fresh_attempt_root(
            tmp_path,
            variant_id=census.A0_VARIANT_ID,
            stage_id="p1_capacity",
        )


def test_atomic_failure_receipt_preserves_primary_and_teardown_errors(
    census: ModuleType,
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "failure.json"

    def operation(observer):
        assert observer is not None
        observer.entered("student_local", {"reserved_gpu_bytes": 123})
        raise MemoryError("synthetic oom")

    def teardown() -> None:
        raise RuntimeError("synthetic teardown")

    with pytest.raises(MemoryError, match="synthetic oom"):
        census.execute_with_atomic_stage_receipt(
            receipt_path=receipt_path,
            base_receipt={"scope": "performance_training_only"},
            operation=operation,
            optional_step_observer_available=True,
            teardown=teardown,
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["last_entered_stage"] == "student_local"
    assert receipt["last_completed_stage"] is None
    assert receipt["primary_failure"]["type"] == "MemoryError"
    assert receipt["teardown_failures"][0]["type"] == "RuntimeError"


def test_atomic_stage_observer_accepts_nested_native_phases(
    census: ModuleType,
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "nested.json"
    observer = census.AtomicStageObserver(receipt_path, {"scope": "performance_training_only"})
    observer.entered("student_local_index")
    observer.entered("student_local_view")
    observer.completed("student_local_view")
    observer.completed("student_local_index")
    observer.finalize(result={"steps": 1}, primary_failure=None, teardown_failures=[])

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["last_entered_stage"] == "student_local_view"
    assert receipt["last_completed_stage"] == "student_local_index"
    assert [(event["event"], event["stage"]) for event in receipt["stage_events"]] == [
        ("entered", "student_local_index"),
        ("entered", "student_local_view"),
        ("completed", "student_local_view"),
        ("completed", "student_local_index"),
    ]


def _records_with_a0_timing(
    census: ModuleType,
    bindings,
    batch_manifest,
    tmp_path: Path,
    *,
    include_p3: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for binding in bindings:
        if binding.variant_id != census.A0_VARIANT_ID:
            records.append(
                {
                    "variant_id": binding.variant_id,
                    "state": (
                        "unavailable_preflight"
                        if binding.variant_id in census.PERFORMANCE_SENTINEL_VARIANT_IDS
                        else "not_selected_performance_sentinel"
                    ),
                    "disposition_reason": "synthetic unavailable",
                    "stages": {},
                }
            )
            continue
        stages: dict[str, object] = {}
        p0_sha256 = "9" * 64
        p1_pointer = _write_receipt(
            tmp_path / binding.variant_id / "p1_capacity.json",
            _stage_payload(
                census,
                binding,
                "p1_capacity",
                batch_manifest,
                p0_preflight_sha256=p0_sha256,
            ),
        )
        stages["p1_capacity"] = p1_pointer
        prerequisite = {**p1_pointer, "physical_gpu_uuid": "GPU-test"}
        stages["p2_timing"] = _write_receipt(
            tmp_path / binding.variant_id / "p2_timing.json",
            _stage_payload(
                census,
                binding,
                "p2_timing",
                batch_manifest,
                p0_preflight_sha256=p0_sha256,
                stage_prerequisite=prerequisite,
            ),
        )
        if include_p3:
            stages["p3_timing"] = _write_receipt(
                tmp_path / binding.variant_id / "p3_timing.json",
                _stage_payload(
                    census,
                    binding,
                    "p3_timing",
                    batch_manifest,
                    p0_preflight_sha256=p0_sha256,
                    stage_prerequisite=prerequisite,
                ),
            )
        records.append(
            {
                "variant_id": binding.variant_id,
                "state": "p3_complete" if include_p3 else "p2_complete",
                "stages": stages,
            }
        )
    return records


def test_aggregate_requires_exact_25_order_and_separates_20_and_100_panels(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    batch_manifest = _frozen_batch_manifest(census, bindings, tmp_path)
    records = _records_with_a0_timing(
        census,
        bindings,
        batch_manifest,
        tmp_path,
        include_p3=True,
    )
    report = census.aggregate_census_report(
        bindings=bindings,
        row_records=records,
        batch_manifest=batch_manifest,
        p0_preflight_sha256="9" * 64,
    )
    assert report["status"] == "complete_with_preregistered_unavailable_or_capacity_failures"
    assert report["measured_20_row_count"] == 1
    assert report["measured_100_row_count"] == 1
    assert [row["variant_id"] for row in report["timing_panels"]["p2_20_measured_steps"]] == [
        census.A0_VARIANT_ID
    ]
    assert [row["variant_id"] for row in report["timing_panels"]["p3_100_measured_steps"]] == [
        census.A0_VARIANT_ID
    ]
    assert report["timing_panels_must_not_be_ranked_together"] is True
    with pytest.raises(ValueError, match="exact matrix order"):
        census.aggregate_census_report(
            bindings=bindings,
            row_records=list(reversed(records)),
            batch_manifest=batch_manifest,
            p0_preflight_sha256="9" * 64,
        )


def test_complete_stage_rejects_terminal_source_and_immutable_input_forgeries(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    batch_manifest = _frozen_batch_manifest(census, bindings, tmp_path)
    binding = bindings[0]
    payload = _stage_payload(
        census,
        binding,
        "p1_capacity",
        batch_manifest,
        p0_preflight_sha256="9" * 64,
    )
    census._validate_stage_receipt(
        payload,
        binding=binding,
        stage_id="p1_capacity",
        batch_manifest=batch_manifest,
        p0_preflight_sha256="9" * 64,
    )

    for expected_message, mutate in (
        (
            "final repository identity is not clean",
            lambda forged: forged["final_repository_identity"]["predicates"].__setitem__(
                "worktree_clean", False
            ),
        ),
        (
            "final repository identity is not clean",
            lambda forged: forged["final_repository_identity"]["predicates"].__setitem__(
                "publication_receipt_equals_p0", False
            ),
        ),
        (
            "repository identity changed",
            lambda forged: forged["final_repository_identity"].__setitem__("head_tree", "f" * 40),
        ),
        (
            "final immutable-input digest differs",
            lambda forged: forged["final_immutable_input_evidence"].__setitem__(
                "ordered_file_bindings_sha256", "0" * 64
            ),
        ),
    ):
        forged = json.loads(json.dumps(payload))
        mutate(forged)
        with pytest.raises(ValueError, match=expected_message):
            census._validate_stage_receipt(
                forged,
                binding=binding,
                stage_id="p1_capacity",
                batch_manifest=batch_manifest,
                p0_preflight_sha256="9" * 64,
            )

    changed = json.loads(json.dumps(payload))
    final_inputs = changed["final_immutable_input_evidence"]
    final_inputs["files"][0]["size_bytes"] = 129
    final_inputs["ordered_file_bindings_sha256"] = census._sha256_json(final_inputs["files"])
    with pytest.raises(ValueError, match="immutable inputs changed"):
        census._validate_stage_receipt(
            changed,
            binding=binding,
            stage_id="p1_capacity",
            batch_manifest=batch_manifest,
            p0_preflight_sha256="9" * 64,
        )


def test_terminal_fallback_is_preserved_as_execution_failed_without_metrics(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    batch_manifest = _frozen_batch_manifest(census, bindings, tmp_path)
    fallback = _fallback_stage_payload(census, bindings[0])
    pointer = _write_receipt(tmp_path / "fallback.json", fallback)
    records = [
        {
            "variant_id": binding.variant_id,
            "state": (
                "execution_failed"
                if index == 0
                else (
                    "unavailable_preflight"
                    if binding.variant_id in census.PERFORMANCE_SENTINEL_VARIANT_IDS
                    else "not_selected_performance_sentinel"
                )
            ),
            "disposition_reason": "synthetic terminal fallback",
            "stages": {"p1_capacity": pointer} if index == 0 else {},
        }
        for index, binding in enumerate(bindings)
    ]
    report = census.aggregate_census_report(
        bindings=bindings,
        row_records=records,
        batch_manifest=batch_manifest,
        p0_preflight_sha256="9" * 64,
    )
    assert report["status"] == "partial_blocked"
    assert report["measured_20_row_count"] == 0
    first_stage = report["rows"][0]["stages"]["p1_capacity"]
    assert first_stage["status"] == "failed"
    assert first_stage["terminal_receipt_kind"] == "construction_fallback"
    assert first_stage["timing_summary_ms"] is None

    fallback["timing_samples_ms"] = []
    with pytest.raises(ValueError, match="cannot expose measurement evidence"):
        census._validate_stage_receipt(
            fallback,
            binding=bindings[0],
            stage_id="p1_capacity",
            batch_manifest=batch_manifest,
            p0_preflight_sha256="9" * 64,
        )


def test_resource_preflight_failure_preserves_primary_failure_without_native_identity(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    batch_manifest = _frozen_batch_manifest(census, bindings, tmp_path)
    payload = _resource_preflight_failure_payload(census, bindings[0], batch_manifest)

    summary = census._validate_stage_receipt(
        payload,
        binding=bindings[0],
        stage_id="p1_capacity",
        batch_manifest=batch_manifest,
        p0_preflight_sha256="9" * 64,
    )

    assert summary["status"] == "failed"
    assert summary["terminal_receipt_kind"] == "resource_preflight_failure"
    assert summary["attempted_batch_count"] == 0
    assert summary["physical_gpu_uuid"] == "GPU-test"
    assert summary["receipt_primary_failure"] == payload["primary_failure"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["resource_preflight"]["predicates"].update(
                host_available_at_least_limit=True
            ),
            "predicates are malformed",
        ),
        (
            lambda payload: payload["native_identity_receipts"].update(
                files=[{"relative_path": "native-run/small_results/run_meta.json"}]
            ),
            "PKL/native evidence differs",
        ),
        (
            lambda payload: payload["final_repository_identity"]["predicates"].update(
                worktree_clean=False
            ),
            "repository identity is not clean",
        ),
        (
            lambda payload: payload.update(primary_failure=None),
            "receipt is malformed",
        ),
    ),
)
def test_resource_preflight_failure_rejects_forged_terminal_evidence(
    census: ModuleType,
    bindings,
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    batch_manifest = _frozen_batch_manifest(census, bindings, tmp_path)
    payload = _resource_preflight_failure_payload(census, bindings[0], batch_manifest)
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        census._validate_stage_receipt(
            payload,
            binding=bindings[0],
            stage_id="p1_capacity",
            batch_manifest=batch_manifest,
            p0_preflight_sha256="9" * 64,
        )


def test_aggregate_rejects_tampered_receipt_and_profiler_timing_mixing(
    census: ModuleType,
    bindings,
    tmp_path: Path,
) -> None:
    batch_manifest = _frozen_batch_manifest(census, bindings, tmp_path)
    records = _records_with_a0_timing(
        census,
        bindings,
        batch_manifest,
        tmp_path,
        include_p3=False,
    )
    a0 = records[0]
    p2_pointer = a0["stages"]["p2_timing"]
    assert isinstance(p2_pointer, dict)
    receipt_path = Path(p2_pointer["receipt_path"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["torch_profiler_trace_sha256"] = "c" * 64
    receipt_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    p2_pointer["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cannot enter a timing-acceptance receipt"):
        census.aggregate_census_report(
            bindings=bindings,
            row_records=records,
            batch_manifest=batch_manifest,
            p0_preflight_sha256="9" * 64,
        )
    p2_pointer["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        census.aggregate_census_report(
            bindings=bindings,
            row_records=records,
            batch_manifest=batch_manifest,
            p0_preflight_sha256="9" * 64,
        )
