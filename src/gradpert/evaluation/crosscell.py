"""Model-independent expression row reads and frozen Systema reference."""

import numpy as np
from scipy import sparse


def source_reference(adata, conditions, rows):
    """Equal condition weighting over original source train+validation rows."""
    ids, codes = np.unique(conditions[rows], return_inverse=True)
    sums = np.zeros((len(ids), adata.n_vars), dtype=np.float64)
    for start in range(0, len(rows), 4096):
        selected = rows[start : start + 4096]
        active, inverse = np.unique(codes[start : start + len(selected)], return_inverse=True)
        indicator = sparse.csr_matrix(
            (np.ones(len(selected), dtype=np.float64), (inverse, np.arange(len(selected)))),
            shape=(len(active), len(selected)),
        )
        sums[active] += (indicator @ adata.X[selected]).toarray()
    return (sums / np.bincount(codes)[:, None]).mean(axis=0)


def read_rows(adata, rows):
    ordered, reverse = np.unique(rows, return_inverse=True)
    return np.asarray(adata.X[ordered].toarray(), dtype=np.float32)[reverse]
