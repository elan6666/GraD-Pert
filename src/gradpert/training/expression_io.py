"""Read exact authorized expression coordinates without AnnData sparse fallbacks.

Sparse index/pointer arrays are structural metadata. Numerical payload is fetched
only after both axes have been filtered. HDF5 selections are bounded unions of
hyperslabs, never whole-array materialization followed by masking.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_MAX_RUNS = 128
_MAJOR_BATCH = 128


def _axis(values: Sequence[int], size: int, name: str) -> np.ndarray[Any, Any]:
    axis = np.asarray(values)
    if (
        axis.ndim != 1
        or axis.dtype.kind not in "iu"
        or not len(axis)
        or np.any(axis < 0)
        or np.any(axis >= size)
        or np.any(axis[1:] <= axis[:-1])
    ):
        raise ValueError(f"{name} must be nonempty sorted unique in-axis integers")
    return axis.astype(np.int64, copy=False)


def _runs(offsets: np.ndarray[Any, Any]) -> list[tuple[int, int]]:
    """Exact contiguous spans; never bridge an unauthorized gap."""
    if not len(offsets):
        return []
    cuts = np.flatnonzero(np.diff(offsets) != 1) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(offsets)]
    return [(int(offsets[a]), int(offsets[b - 1]) + 1) for a, b in zip(starts, ends, strict=True)]


def _hdf5_dataset(value: Any) -> bool:
    try:
        import h5py
    except ImportError:
        return False
    return isinstance(value, h5py.Dataset)


def _read_offsets(dataset: Any, offsets: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Gather sorted unique offsets with O(runs/128) real HDF5 reads."""
    result = np.empty(len(offsets), dtype=dataset.dtype)
    runs = _runs(offsets)
    cursor = 0
    for start in range(0, len(runs), _MAX_RUNS):
        group = runs[start : start + _MAX_RUNS]
        length = sum(b - a for a, b in group)
        if _hdf5_dataset(dataset):
            import h5py

            space = dataset.id.get_space()
            memory = h5py.h5s.create_simple((length,))
            try:
                space.select_none()
                for a, b in group:
                    space.select_hyperslab((a,), (1,), block=(b - a,), op=h5py.h5s.SELECT_OR)
                dataset.id.read(memory, space, result[cursor : cursor + length])
            finally:
                memory.close()
                space.close()
        else:
            # Proxies use exactly the same authorized spans, with no low-level bypass.
            position = cursor
            for a, b in group:
                result[position : position + b - a] = dataset[a:b]
                position += b - a
        cursor += length
    return result


def _dense(backing: Any, rows: np.ndarray[Any, Any], cols: np.ndarray[Any, Any]) -> np.ndarray:
    output = np.empty((len(rows), len(cols)), dtype=np.float32)
    spans = _runs(cols)
    for row_position, row in enumerate(rows):
        column_position = 0
        for offset in range(0, len(spans), _MAX_RUNS):
            group = spans[offset : offset + _MAX_RUNS]
            width = sum(b - a for a, b in group)
            if _hdf5_dataset(backing):
                import h5py

                block = np.empty((1, width), dtype=backing.dtype)
                space = backing.id.get_space()
                memory = h5py.h5s.create_simple(block.shape)
                try:
                    space.select_none()
                    for a, b in group:
                        space.select_hyperslab(
                            (int(row), a), (1, 1), block=(1, b - a), op=h5py.h5s.SELECT_OR
                        )
                    backing.id.read(memory, space, block)
                finally:
                    memory.close()
                    space.close()
            else:
                block = np.concatenate(
                    [np.asarray(backing[int(row) : int(row) + 1, a:b]) for a, b in group],
                    axis=1,
                )
            if not np.isfinite(block).all():
                raise ValueError("authorized expression values must be finite")
            output[row_position, column_position : column_position + width] = block[0]
            column_position += width
    return output


def _sparse(
    backing: Any,
    rows: np.ndarray[Any, Any],
    cols: np.ndarray[Any, Any],
    csr: bool,
    shape: tuple[int, int],
) -> np.ndarray:
    pointer, indices, data = backing["indptr"], backing["indices"], backing["data"]
    major_size, minor_size = shape if csr else shape[::-1]
    if (
        len(pointer.shape) != 1
        or pointer.shape[0] != major_size + 1
        or len(indices.shape) != 1
        or len(data.shape) != 1
        or indices.shape != data.shape
        or pointer.dtype.kind not in "iu"
        or indices.dtype.kind not in "iu"
        or data.dtype.kind not in "fiu"
    ):
        raise ValueError("invalid sparse expression backing")
    ptr = np.asarray(pointer[:])  # Structural metadata only.
    if ptr[0] != 0 or ptr[-1] != data.shape[0] or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("invalid sparse expression pointers")
    output = np.zeros((len(rows), len(cols)), dtype=np.float32)
    major, minor = (rows, cols) if csr else (cols, rows)
    for start in range(0, len(major), _MAJOR_BATCH):
        batch = major[start : start + _MAJOR_BATCH]
        lengths = ptr[batch + 1] - ptr[batch]
        offsets = np.concatenate(
            [np.arange(int(ptr[i]), int(ptr[i + 1]), dtype=np.int64) for i in batch]
        )
        index = _read_offsets(indices, offsets)  # Gene/row identity metadata, not expression.
        if np.any(index < 0) or np.any(index >= minor_size):
            raise ValueError("invalid sparse expression indices")
        locations = np.searchsorted(minor, index)
        allowed = (locations < len(minor)) & (minor[np.minimum(locations, len(minor) - 1)] == index)
        # Crucially filter before touching any numerical payload dataset.
        values = _read_offsets(data, offsets[allowed])
        if not np.isfinite(values).all():
            raise ValueError("authorized expression values must be finite")
        major_positions = np.repeat(np.arange(len(batch)), lengths.astype(np.int64))[allowed]
        block = np.zeros((len(batch), len(minor)), dtype=data.dtype)
        # Duplicate sparse coordinates sum in the original payload dtype before float32 conversion.
        np.add.at(block, (major_positions, locations[allowed]), values)
        if csr:
            output[start : start + len(batch)] = block
        else:
            output[:, start : start + len(batch)] = block.T
    return output


def read_expression_selection(
    backing: Any, rows: Sequence[int], columns: Sequence[int]
) -> np.ndarray[Any, Any]:
    """Return [requested rows, allowed columns] float32; both axes sorted unique.

    ``backing`` is the H5AD X Dataset (dense) or CSR/CSC Group. No AnnData API
    is invoked. Callers own repeated-row restoration and full-axis zero filling.
    Unknown storage encodings fail closed. No sampling or RNG is involved.
    """
    if hasattr(backing, "shape"):
        shape = tuple(backing.shape)
        declared = getattr(backing, "attrs", {}).get("encoding-type")
        if isinstance(declared, bytes):
            declared = declared.decode("ascii")
        if declared not in (None, "array"):
            raise ValueError("unsupported expression storage encoding")
        encoding = "dense"
    else:
        shape = tuple(backing.attrs.get("shape", ()))
        encoding = backing.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("ascii")
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape):
        raise ValueError("expression backing requires a positive two-dimensional shape")
    row_axis, col_axis = _axis(rows, shape[0], "rows"), _axis(columns, shape[1], "columns")
    if encoding == "dense":
        if backing.dtype.kind not in "fiu":
            raise ValueError("dense expression backing must have a real numerical dtype")
        result = _dense(backing, row_axis, col_axis)
    elif encoding in ("csr_matrix", "csc_matrix"):
        result = _sparse(backing, row_axis, col_axis, encoding == "csr_matrix", shape)
    else:
        raise ValueError("unsupported expression storage encoding")
    if not np.isfinite(result).all():
        raise ValueError("authorized expression float32 output must be finite")
    return np.ascontiguousarray(result)
