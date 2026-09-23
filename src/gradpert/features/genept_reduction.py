"""Deterministic, GenePT-only PCA for a sealed graph-gene axis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pca_genept(values: NDArray[np.float32], width: int) -> tuple[NDArray[np.float32], float]:
    """Return centered PCA scores with canonical component signs.

    No expression, condition labels, graph edges or test outcomes enter this fit.
    The caller pins the GenePT source and exact ordered graph-gene axis.
    """
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or type(width) is not int
        or not 0 < width < min(values.shape)
    ):
        raise ValueError("PCA requires a finite matrix and a smaller positive width")
    centered = values.astype(np.float64) - values.mean(axis=0, dtype=np.float64)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[-width:][::-1]
    components = eigenvectors[:, order]
    pivots = np.abs(components).argmax(axis=0)
    signs = np.sign(components[pivots, np.arange(width)])
    components *= signs
    reduced = np.ascontiguousarray(centered @ components, dtype=np.float32)
    total_variance = np.clip(eigenvalues, 0, None).sum()
    if total_variance <= 0:
        raise ValueError("GenePT vectors have no PCA variance")
    explained = float(np.clip(eigenvalues[order], 0, None).sum() / total_variance)
    if not np.isfinite(reduced).all() or not np.isfinite(explained):
        raise ValueError("nonfinite PCA output")
    return reduced, explained
