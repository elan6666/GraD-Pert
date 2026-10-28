"""Training-only numerical column exposure, without canonical files or model calls."""

from types import SimpleNamespace

import numpy as np
import pytest

from gradpert.training.data import CanonicalTrainingData, TrainingPipelineStats
from gradpert.training.systems import DISABLED_NATIVE_SYSTEM_OPTIONS, NativeSystemOptions


class FakeH5ad:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def __getitem__(self, key):
        rows, columns = key
        parent = self

        class Selection:
            @property
            def X(self):
                parent.reads.append((rows, columns))
                # Slice column axis first, before exposing numerical values.
                return parent.values[:, columns][rows]

        return Selection()


def data_fixture():
    data = object.__new__(CanonicalTrainingData)
    values = np.arange(12 * 10, dtype=np.float32).reshape(12, 10)
    data._adata = FakeH5ad(values)
    data.row_ids = tuple(str(i) for i in range(12))
    data.manifest = SimpleNamespace(n_expression_genes=10)
    data.pipeline_stats = TrainingPipelineStats()
    data._control_expression_cache = None
    data._control_cache_position = {}
    data.system_options = DISABLED_NATIVE_SYSTEM_OPTIONS
    data.control_row_indices = (0, 1, 2)
    return data, values


def test_slice_before_X_and_duplicate_row_global_axis_parity():
    data, values = data_fixture()
    allowed = [0, 2, 5, 9]
    values[:, [1, 3, 4, 6, 7, 8]] = np.nan  # Never read, including finite checks.
    data.configure_training_expression_columns(allowed)
    result = data._read_expression_indices([8, 2, 8, 3])
    np.testing.assert_array_equal(result[:, allowed], values[[8, 2, 8, 3]][:, allowed])
    assert result.shape == (4, 10) and result.dtype == np.float32
    assert np.all(result[:, [1, 3, 4, 6, 7, 8]] == 0)
    assert len(data._adata.reads) == 2
    for rows, columns in data._adata.reads:
        assert isinstance(rows, slice)
        np.testing.assert_array_equal(columns, allowed)
    assert [(r.start, r.stop) for r, _ in data._adata.reads] == [(2, 4), (8, 9)]


def test_default_off_full_axis_and_evaluation_control_loader():
    data, values = data_fixture()
    data._row_index = {r: i for i, r in enumerate(data.row_ids)}
    data.condition_ids = ("ctrl",) * 12
    data.split = SimpleNamespace(control_condition_id="ctrl")
    result = data.load_control_rows(["4", "1", "4"])
    np.testing.assert_array_equal(result, values[[4, 1, 4]])
    assert isinstance(data._adata.reads[0][1], slice)
    with pytest.raises(RuntimeError, match="before"):
        data.configure_training_expression_columns([0, 2])


@pytest.mark.parametrize("columns", [[], [1, 1], [2, 1], [-1], [10], [1.0], [True], [[1]]])
def test_invalid_axis(columns):
    data, _ = data_fixture()
    with pytest.raises(ValueError):
        data.configure_training_expression_columns(columns)


def test_setup_rejects_cache_requested_built_or_iteration():
    for state in ("requested", "built", "iterated"):
        data, _ = data_fixture()
        if state == "requested":
            data.pipeline_stats.control_cache_requested = True
        elif state == "built":
            data._control_expression_cache = np.ones((1, 10))
        else:
            data.pipeline_stats.yielded_batches = 1
        with pytest.raises(RuntimeError):
            data.configure_training_expression_columns([0, 2])
    data, _ = data_fixture()
    data.configure_training_expression_columns([0, 2])
    with pytest.raises(RuntimeError):
        data.configure_training_expression_columns([0, 2])


@pytest.mark.parametrize("merged", [False, True])
def test_materialize_paths_preserve_allowed_values(merged):
    data, values = data_fixture()
    data.configure_training_expression_columns([0, 2, 5, 9])
    data.system_options = enabled_options() if merged else DISABLED_NATIVE_SYSTEM_OPTIONS
    spec = SimpleNamespace(control_indices=(2, 0, 2), perturbed_indices=(7, 6, 7))
    batch = data._materialize_cpu_batch(spec)
    np.testing.assert_array_equal(
        batch.control_expression[:, [0, 2, 5, 9]], values[[2, 0, 2]][:, [0, 2, 5, 9]]
    )
    np.testing.assert_array_equal(
        batch.target_expression[:, [0, 2, 5, 9]], values[[7, 6, 7]][:, [0, 2, 5, 9]]
    )
    for _, columns in data._adata.reads:
        np.testing.assert_array_equal(columns, [0, 2, 5, 9])


def test_control_cache_and_cached_target_share_read_boundary():
    data, values = data_fixture()
    data.configure_training_expression_columns([0, 2])
    data.configure_system_optimizations(enabled_options())
    spec = SimpleNamespace(control_indices=(2, 0, 2), perturbed_indices=(7, 6, 7))
    batch = data._materialize_cpu_batch(spec)
    np.testing.assert_array_equal(batch.control_expression[:, [0, 2]], values[[2, 0, 2]][:, [0, 2]])
    np.testing.assert_array_equal(batch.target_expression[:, [0, 2]], values[[7, 6, 7]][:, [0, 2]])
    assert data.pipeline_stats.cached_control_batches == 1
    for _, columns in data._adata.reads:
        np.testing.assert_array_equal(columns, [0, 2])


def enabled_options():
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
    )


def test_control_only_and_allowed_nonfinite_fail_closed():
    data, values = data_fixture()
    data.configure_training_expression_columns([0, 2])
    data.control_only_training = True
    spec = SimpleNamespace(control_indices=(2, 0, 2), perturbed_indices=(7, 6, 7))
    batch = data._materialize_cpu_batch(spec)
    assert np.isnan(batch.target_expression).all()
    np.testing.assert_array_equal(batch.control_expression[:, [0, 2]], values[[2, 0, 2]][:, [0, 2]])
    assert all(r.stop <= 3 for r, _ in data._adata.reads)
    values[0, 2] = np.nan
    with pytest.raises(ValueError, match="invalid shape/values"):
        data._read_expression_indices([0])
