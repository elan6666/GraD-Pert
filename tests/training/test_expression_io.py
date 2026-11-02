"""Synthetic coordinate guards, independent of AnnData internals or real data."""

import numpy as np
import pytest

from gradpert.training.expression_io import _read_offsets, _runs, read_expression_selection


class GuardedArray:
    def __init__(self, array, allowed=None):
        self.array = np.asarray(array)
        self.shape, self.dtype = self.array.shape, self.array.dtype
        self.attrs = {}
        self.allowed = allowed
        self.selections = []

    def __getitem__(self, key):
        mask = np.zeros(self.shape, dtype=bool)
        mask[key] = True
        if self.allowed is not None:
            assert not np.any(mask & ~self.allowed), "forbidden numerical payload read"
        self.selections.append(key)
        return self.array[key]


class Group(dict):
    pass


def backing(matrix, encoding, rows, cols):
    allowed = np.zeros(matrix.shape, dtype=bool)
    allowed[np.ix_(rows, cols)] = True
    if encoding == "dense":
        return GuardedArray(matrix, allowed)
    csr = encoding == "csr_matrix"
    payload, index, ptr, permitted = [], [], [0], []
    major = matrix if csr else matrix.T
    selected = allowed if csr else allowed.T
    for i in range(len(major)):
        for j in range(major.shape[1]):
            if major[i, j] != 0:
                payload.append(major[i, j])
                index.append(j)
                permitted.append(selected[i, j])
        ptr.append(len(payload))
    result = Group(
        data=GuardedArray(np.asarray(payload, dtype=matrix.dtype), np.array(permitted)),
        indices=GuardedArray(np.array(index, dtype=np.int64)),
        indptr=GuardedArray(np.array(ptr, dtype=np.int64)),
    )
    result.attrs = {"encoding-type": encoding, "shape": matrix.shape}
    return result


@pytest.mark.parametrize("encoding", ["dense", "csr_matrix", "csc_matrix"])
def test_exact_authorized_reads_parity_and_rng(encoding):
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(17, 19)).astype(np.float64)
    matrix[rng.random(matrix.shape) < 0.6] = 0
    matrix[4] = 0
    requested = np.array([12, 1, 4, 12, 16, 1])
    rows, inverse = np.unique(requested, return_inverse=True)
    cols = np.array([0, 2, 5, 8, 9, 17])
    obj = backing(matrix, encoding, rows, cols)
    before = rng.bit_generator.state
    actual = read_expression_selection(obj, rows, cols)
    np.testing.assert_array_equal(actual, matrix[np.ix_(rows, cols)].astype(np.float32))
    np.testing.assert_array_equal(
        actual[inverse], matrix[np.ix_(requested, cols)].astype(np.float32)
    )
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert before == rng.bit_generator.state


@pytest.mark.parametrize("encoding", ["dense", "csr_matrix", "csc_matrix"])
def test_excluded_poison_and_empty_sparse_selected_rows(encoding):
    matrix = np.zeros((5, 7), dtype=np.float32)
    matrix[0, 1] = 4
    matrix[1, 2] = np.nan  # Forbidden column.
    matrix[3, 1] = np.inf  # Forbidden row.
    rows, cols = np.array([0, 1, 2]), np.array([1, 4, 6])
    actual = read_expression_selection(backing(matrix, encoding, rows, cols), rows, cols)
    np.testing.assert_array_equal(actual, np.array([[4, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32))


@pytest.mark.parametrize("axis", [[], [1, 1], [2, 1], [-1, 2], [0, 7], [True], [1.0]])
@pytest.mark.parametrize("which", ["rows", "columns"])
def test_invalid_axis_rejected_before_payload(axis, which):
    obj = GuardedArray(np.zeros((7, 7)))
    with pytest.raises(ValueError):
        read_expression_selection(
            obj, axis if which == "rows" else [0], axis if which == "columns" else [0]
        )
    assert not obj.selections


def test_unknown_sparse_encoding_fails_closed():
    obj = Group()
    obj.attrs = {"shape": (5, 7), "encoding-type": "unknown"}
    with pytest.raises(ValueError, match="unsupported"):
        read_expression_selection(obj, [0], [0])


@pytest.mark.parametrize("encoding", ["dense", "csr_matrix", "csc_matrix"])
def test_selected_nonfinite_rejected(encoding):
    matrix = np.zeros((3, 3), np.float32)
    matrix[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        read_expression_selection(backing(matrix, encoding, [0], [1]), [0], [1])


def test_sparse_duplicate_coordinates_sum_in_source_dtype():
    obj = Group(
        data=GuardedArray(np.array([1e8, 1, -1e8], np.float64), np.ones(3, bool)),
        indices=GuardedArray(np.array([1, 1, 1])),
        indptr=GuardedArray(np.array([0, 3, 3])),
    )
    obj.attrs = {"shape": (2, 3), "encoding-type": "csr_matrix"}
    np.testing.assert_array_equal(read_expression_selection(obj, [0, 1], [1]), [[1], [0]])


def test_offset_runs_do_not_bridge_forbidden_gaps():
    offsets = np.array([0, 1, 4, 7, 8, 9, 100])
    assert _runs(offsets) == [(0, 2), (4, 5), (7, 10), (100, 101)]
    permitted = np.zeros(101, bool)
    permitted[offsets] = True
    obj = GuardedArray(np.arange(101), permitted)
    np.testing.assert_array_equal(_read_offsets(obj, offsets), offsets)
    assert len(obj.selections) == 4


@pytest.mark.parametrize("encoding", ["dense", "csr_matrix", "csc_matrix"])
def test_real_h5py_file_exact_selection(tmp_path, encoding):
    h5py = pytest.importorskip("h5py", reason="server synthetic CPU validation required")
    rows, cols = np.array([0, 3, 5, 8]), np.array([1, 3, 6])
    matrix = np.arange(63, dtype=np.float32).reshape(9, 7)
    matrix[3] = 0
    obj = backing(matrix, encoding, rows, cols)
    with h5py.File(tmp_path / "synthetic.h5ad", "w") as file:
        if encoding == "dense":
            x = file.create_dataset("X", data=matrix, chunks=(2, 3), compression="gzip")
        else:
            x = file.create_group("X")
            x.attrs.update(obj.attrs)
            for name, dataset in obj.items():
                x.create_dataset(name, data=dataset.array, chunks=True)
        np.testing.assert_array_equal(
            read_expression_selection(x, rows, cols), matrix[np.ix_(rows, cols)]
        )


def test_real_h5py_bounded_union_gather(tmp_path):
    h5py = pytest.importorskip("h5py", reason="server synthetic CPU validation required")
    offsets = np.arange(0, 2000, 3)
    with h5py.File(tmp_path / "offsets.h5", "w") as file:
        data = file.create_dataset("data", data=np.arange(2000, dtype=np.float64))
        np.testing.assert_array_equal(_read_offsets(data, offsets), offsets)


@pytest.mark.parametrize(
    "key,value",
    [
        ("indptr", [0, 2, 1, 2]),
        ("indptr", [1, 1, 1, 2]),
        ("indices", [1, 9]),
    ],
)
def test_malformed_sparse_metadata_rejected_before_values(key, value):
    obj = Group(
        data=GuardedArray(np.array([3.0, 4.0]), np.zeros(2, bool)),
        indices=GuardedArray(np.array([1, 2])),
        indptr=GuardedArray(np.array([0, 1, 1, 2])),
    )
    obj.attrs = {"shape": (3, 3), "encoding-type": "csr_matrix"}
    obj[key] = GuardedArray(np.array(value))
    with pytest.raises(ValueError, match="invalid sparse"):
        read_expression_selection(obj, [0, 2], [1, 2])
    assert not obj["data"].selections


def test_float32_conversion_overflow_fails_closed():
    obj = GuardedArray(np.array([[1e300]], dtype=np.float64), np.ones((1, 1), bool))
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="float32"):
        read_expression_selection(obj, [0], [0])


@pytest.mark.parametrize("encoding", ["unknown", b"csr_matrix"])
def test_unknown_dense_encoding_fails_before_payload(encoding):
    obj = GuardedArray(np.zeros((5, 7)))
    obj.attrs["encoding-type"] = encoding
    with pytest.raises(ValueError, match="unsupported"):
        read_expression_selection(obj, [0, 2], [1, 4])
    assert not obj.selections
