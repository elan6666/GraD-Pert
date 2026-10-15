"""Pure identity/grouping contracts; real model parity is gated on the server."""

import sys
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/v2"))
from cap40_posthoc import recipe, records_from_data


def data(rows, controls):
    ids = ["t0", "t1", "c0", "c1"]

    class Obs:
        def __init__(self):
            self.iloc = self

        def __getitem__(self, i):
            return {"cell_type": "T"}

    spec = N(perturbed_row_ids=rows, control_row_ids=controls, condition_ids=("p",) * len(rows))
    return N(
        row_ids=ids,
        _row_index={x: i for i, x in enumerate(ids)},
        context_ids=np.array(["T::A", "T::B", "T::A", "T::B"]),
        batch_ids=np.array(["A", "B", "A", "B"]),
        _adata=N(obs=Obs()),
        train_row_indices=[0, 1],
        training_batch_identity_specs=lambda **kw: [spec],
    )


def test_exact_rows_with_context_matched_controls():
    rows = records_from_data(data(("t1", "t0"), ("c1", "c0")))
    assert [r["truth_row_id"] for r in rows] == ["t1", "t0"]
    assert rows[0]["batch"] == "B"
    assert rows[0]["control_row_id"] == "c1"


@pytest.mark.parametrize(
    "rows,controls",
    [(("t0", "t0"), ("c0", "c0")), (("t0",), ("c0",)), (("t0", "t1"), ("c1", "c0"))],
)
def test_reject_duplicate_missing_and_mismatched_context(rows, controls):
    with pytest.raises(ValueError):
        records_from_data(data(rows, controls))


def test_recipe_binds_actual_mask_edges_gene_and_view_identity():
    g = N(
        ids=np.array([1, 2]),
        masked_positions=np.array([0]),
        neighbors=np.array([[1], [0]]),
        valid=np.array([[True], [True]]),
        sources=np.zeros((2, 1, 4), bool),
        context=None,
    )
    b = N(
        graph=g,
        query_positions=np.array([0, 1]),
        graph_views=(g,),
        cell_views=(N(positions=np.array([0]), mask=np.array([[True]])),),
    )
    a = recipe(b)
    b.cell_views[0].mask[0, 0] = False
    assert a != recipe(b)
    assert a["queries"] == [1, 2]
