from types import SimpleNamespace

import numpy as np
import pytest

from gradpert.evaluation.state import _mean_rows
from scripts.analysis.split_half_control_reference import compatible_control_indices


def test_complete_pool_matches_contexts_and_is_cell_weighted():
    contexts = np.array(["Jurkat::a", "Jurkat::a", "Jurkat::b", "Jurkat::c", "HepG2::a"])
    matrix = np.array([[0, 0], [0, 0], [9, 6], [99, 99], [99, 99]], dtype=np.float32)
    rows = compatible_control_indices(contexts, ("Jurkat::a", "Jurkat::b", "Jurkat::a"))
    assert rows.tolist() == [0, 1, 2]
    result = _mean_rows(SimpleNamespace(X=matrix), rows)
    assert np.allclose(result, [3, 2])
    # A batch-equal mean would be [4.5,3]; complete-pool mean does not use it.
    assert not np.allclose(result, [4.5, 3])
    assert result.dtype == np.float32


def test_missing_control_context_fails_closed():
    with pytest.raises(ValueError, match="lacks controls"):
        compatible_control_indices(np.array(["Jurkat::a"]), ("Jurkat::a", "Jurkat::b"))
    with pytest.raises(ValueError, match="empty"):
        compatible_control_indices(np.array(["Jurkat::a"]), ())
