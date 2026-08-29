from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from dataclasses import dataclass
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
        }


class _Engine:
    def __init__(self) -> None:
        self.stage_observer = None
        self.stage_observer_failures: list[dict[str, object]] = []
        self.last_view_stats = {"effective_node_budget": 1405}

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
        with self.CanonicalEvaluationData(split_name="val") as validation:
            validation.configure_expression_cache(enabled=True)
            engine = self.engine_class()
            for global_step in range(100):
                self.requested_steps += 1
                engine.train_step(_Batch(global_step), global_step=global_step)
            self.validation_calls += 1
            self.evaluate_validation_macro_delta()


def _args(tmp_path: Path, *, stage_id: str) -> Namespace:
    return Namespace(
        stage_id=stage_id,
        device="cuda:0",
        data_root=tmp_path / "data",
        repository_root=PROJECT_ROOT,
        development_commit="a" * 40,
        genept_preflight_receipt=None,
        genept_preflight_receipt_sha256=None,
        minimum_gpu_headroom_fraction=0.15,
        minimum_gpu_free_bytes=4 * 1024**3,
    )


def _binding(tmp_path: Path):
    return SimpleNamespace(
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


def test_worker_cli_excludes_p3(worker: ModuleType) -> None:
    choices = next(
        action.choices for action in worker._parser()._actions if action.dest == "stage_id"
    )
    assert set(choices) == {"p1_capacity", "p2_timing", "diagnostic_profile"}
    assert "p3_timing" not in choices


def test_ordered_batch_identity_preserves_rows_conditions_controls_and_anchors(
    worker: ModuleType,
) -> None:
    identity = worker.ordered_batch_identity(_Batch(3), global_step=3)
    assert identity.row_ids[:2] == ("perturbed-3-0", "perturbed-3-1")
    assert identity.condition_ids[:2] == ("condition-0", "condition-1")
    assert identity.control_row_ids[:2] == ("control-3-0", "control-3-1")
    assert identity.active_anchor_ids[:2] == (("0", "100"), ("1", "101"))
    assert identity.actual_batch_size == 256
    assert identity.unique_condition_count == 5


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
        _args(tmp_path, stage_id=stage_id),
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        runtime=runtime,
        resource_preflight=_preflight(),
        genept_preflight=(None, None),
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
        genept_preflight=(None, None),
    )
    receipt = worker._build_stage_receipt(
        args,
        binding=binding,
        attempt_root=attempt_root,
        resource_preflight=_preflight(),
        state=state,
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
    worker.census.require_training_only_evidence(receipt["training_only_evidence"])


def test_first_step_failure_preserves_attempted_batch_and_last_entered_stage(
    worker: ModuleType,
    tmp_path: Path,
) -> None:
    class FailingEngine(_Engine):
        def train_step(self, batch, *, global_step: int):
            assert self.stage_observer is not None
            self.stage_observer(_Event(global_step=global_step, status="entered"), self)
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
        genept_preflight=(None, None),
    )
    assert isinstance(state.primary_failure, RuntimeError)
    assert str(state.primary_failure) == "synthetic first-step OOM"
    assert len(state.batches) == 1
    assert state.batches[0].global_step == 0
    assert state.steps == []
    progress = worker.json.loads((attempt_root / "stage-progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["last_entered_stage"] == "views"
    receipt = worker._build_stage_receipt(
        args,
        binding=_binding(tmp_path),
        attempt_root=attempt_root,
        resource_preflight=_preflight(),
        state=state,
    )
    assert receipt["attempted_batch_count"] == 1
    assert receipt["completed_step_count"] == 0
    assert len(receipt["batches"]) == 1


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
        genept_preflight=(None, None),
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
    )
    assert receipt["torch_profiler_trace_sha256"] is not None
    assert receipt["torch_profiler_table_sha256"] is not None


def test_non_256_batch_fails_closed(worker: ModuleType) -> None:
    batch = _Batch(0)
    batch.condition_ids = batch.condition_ids[:-1]
    batch.perturbed_row_ids = batch.perturbed_row_ids[:-1]
    batch.control_row_ids = batch.control_row_ids[:-1]
    with pytest.raises(worker.WorkerGateError, match="non-256"):
        worker.ordered_batch_identity(batch, global_step=0)


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
        genept_preflight=resolved,
    )
    assert state.primary_failure is None
    assert observed_kwargs["genept_preflight_receipt"] == receipt_path.resolve()
    assert observed_kwargs["genept_preflight_receipt_sha256"] == receipt_sha256
