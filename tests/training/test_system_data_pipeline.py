from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

import gradpert.training.data as data_module
from gradpert.evaluation.data import CanonicalEvaluationData, EvaluationCacheStats
from gradpert.training.data import (
    CanonicalTrainingData,
    TrainingPipelineStats,
    _TrainingBatchSpec,
)
from gradpert.training.systems import NativeSystemOptions


def _enabled_options() -> NativeSystemOptions:
    return NativeSystemOptions(
        merged_hdf5_reads=True,
        control_expression_cache=True,
        background_prefetch=True,
        resident_graph_tensors=True,
        validation_expression_cache=True,
        buffered_training_logs=True,
        single_checkpoint_serialization=True,
        pin_memory=True,
        nonblocking_transfer=True,
        prefetch_depth=2,
        log_buffer_steps=64,
    )


def _fake_data(matrix: np.ndarray) -> CanonicalTrainingData:
    data = object.__new__(CanonicalTrainingData)
    data.system_options = NativeSystemOptions()
    data.pipeline_stats = TrainingPipelineStats()
    data._control_expression_cache = None
    data._control_cache_position = {}

    def read(self: CanonicalTrainingData, indices):  # type: ignore[no-untyped-def]
        return np.ascontiguousarray(matrix[list(indices)], dtype=np.float32)

    data._read_expression_indices = MethodType(read, data)  # type: ignore[method-assign]
    return data


def _spec() -> _TrainingBatchSpec:
    return _TrainingBatchSpec(
        perturbed_indices=(2, 3, 1),
        perturbed_row_ids=("p2", "p3", "p1"),
        control_indices=(4, 0, 4),
        control_row_ids=("c4", "c0", "c4"),
        condition_ids=("A+ctrl", "B+ctrl", "A+ctrl"),
        anchors_by_condition={"A+ctrl": (0,), "B+ctrl": (1,)},
    )


def test_merged_read_and_control_cache_preserve_exact_values_and_order() -> None:
    matrix = np.arange(18, dtype=np.float32).reshape(6, 3)
    spec = _spec()
    baseline = _fake_data(matrix)._materialize_cpu_batch(spec)

    merged_data = _fake_data(matrix)
    merged_data.system_options = _enabled_options()
    merged = merged_data._materialize_cpu_batch(spec)
    assert merged_data.pipeline_stats.merged_read_batches == 1

    cached_data = _fake_data(matrix)
    cached_data.system_options = _enabled_options()
    cached_data._control_expression_cache = np.ascontiguousarray(matrix[[0, 4]])
    cached_data._control_cache_position = {0: 0, 4: 1}
    cached = cached_data._materialize_cpu_batch(spec)
    assert cached_data.pipeline_stats.cached_control_batches == 1

    for observed in (merged, cached):
        assert np.array_equal(observed.control_expression, baseline.control_expression)
        assert np.array_equal(observed.target_expression, baseline.target_expression)
        assert observed.spec == baseline.spec


def test_prefetch_startup_failure_falls_back_without_row_loss(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    matrix = np.arange(18, dtype=np.float32).reshape(6, 3)
    data = _fake_data(matrix)
    data.system_options = _enabled_options()

    def unavailable(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("thread pool unavailable")

    monkeypatch.setattr(data_module, "ThreadPoolExecutor", unavailable)
    batches = list(data._iter_cpu_batches((_spec(), _spec())))
    assert len(batches) == 2
    assert all(batch.spec == _spec() for batch in batches)
    assert data.pipeline_stats.prefetch_active is False
    assert data.pipeline_stats.prefetch_fallback_reason == "RuntimeError"


def test_system_options_reject_partial_enablement() -> None:
    with pytest.raises(ValueError, match="all seven"):
        NativeSystemOptions(merged_hdf5_reads=True)
    with pytest.raises(ValueError, match="pinned nonblocking"):
        NativeSystemOptions(
            merged_hdf5_reads=True,
            control_expression_cache=True,
            background_prefetch=True,
            resident_graph_tensors=True,
            validation_expression_cache=True,
            buffered_training_logs=True,
            single_checkpoint_serialization=True,
        )


def test_validation_cache_restores_duplicates_and_fails_closed_on_miss() -> None:
    data = object.__new__(CanonicalEvaluationData)
    data._expression_cache = np.arange(12, dtype=np.float32).reshape(4, 3)
    data._cache_position = {2: 0, 5: 1, 7: 2, 9: 3}
    data.cache_stats = EvaluationCacheStats(requested=True, active=True, cached_rows=4)
    observed = data._read_expression_indices((7, 2, 7))
    assert np.array_equal(observed, data._expression_cache[[2, 0, 2]])
    assert data.cache_stats.cache_hits == 1
    with pytest.raises(RuntimeError, match="lacks required canonical row 8"):
        data._read_expression_indices((8,))
