"""Hash-pinned external annotation and condition-wise essential local controls."""

import csv
import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from gradpert.graphs.views import build_ring_induced_view, stable_view_seed


def load_essential_ids(path: Path, expected: str, genes: tuple[str, ...]) -> tuple[int, ...]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("essential annotation hash mismatch")
    with path.open() as stream:
        rows = list(csv.reader(stream))
    # Frozen DepMap20Q1 one-column SYMBOL (ENTREZ) table; no fuzzy aliases.
    if (
        not rows
        or rows[0] != ["gene"]
        or any(
            len(row) != 1 or re.fullmatch(r"[^\s()]+ \([0-9]+\)", row[0]) is None
            for row in rows[1:]
        )
    ):
        raise ValueError("invalid frozen essential annotation schema")
    labels = [row[0].split(" (")[0] for row in rows[1:]]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("empty or ambiguous essential labels")
    label_set = set(labels)
    ids = tuple(i for i, gene in enumerate(genes) if gene in label_set)
    if not ids:
        raise ValueError("essential annotation has no runtime coverage")
    return ids


def essential_local_views(
    views: Any, topology: Any, *, ids: tuple[int, ...], policy: str, run_seed: int, global_step: int
) -> Any:
    """Replace locals only; globals/prediction/masks/order are untouched.

    Matched RingInduced must reach exactly the mandatory set size. No random
    disconnected filler and no silent under-sized control is accepted.
    """
    if policy not in {"essential_only", "essential_size_ring"}:
        raise ValueError("unknown essential local policy")
    if not ids or any(i < 0 or i >= topology.n_nodes for i in ids):
        raise ValueError("invalid essential node IDs")
    result = {}
    for condition, anchors in views.anchors_by_condition.items():
        mandatory = tuple(sorted(set(ids) | set(anchors)))
        locals_ = []
        for index, old in enumerate(views.locals_by_condition[condition]):
            if old.masked_anchor_ids or views.masked_local_indices_by_condition[condition]:
                raise ValueError("essential R50 requires zero local anchor masking")
            seed = stable_view_seed(
                run_seed=run_seed,
                global_step=global_step,
                condition_id=condition,
                namespace="local_boundary",
                view_index=index,
            )
            view = build_ring_induced_view(
                topology,
                anchors=mandatory if policy == "essential_only" else anchors,
                node_budget=len(mandatory),
                seed=seed,
                view_id=old.view_id,
                mask_anchors=False,
            )
            if len(view.node_ids) != len(mandatory):
                raise ValueError("matched ring cannot reach essential node count")
            locals_.append(view)
        result[condition] = tuple(locals_)
    return replace(views, locals_by_condition=result)
