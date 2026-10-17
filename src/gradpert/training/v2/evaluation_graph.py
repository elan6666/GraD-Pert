"""Evaluation-scoped reuse of frozen, control-independent graph representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from gradpert.modeling.v2 import GraDPertV2

from .views import NeighborhoodIndex


def _versions(model: GraDPertV2) -> tuple[tuple[int, torch.dtype, torch.device], ...]:
    # Tensor versions invalidate the cache after load_state_dict/in-place updates.
    return tuple((p._version, p.dtype, p.device) for p in model.graph.parameters())


@dataclass(frozen=True)
class EvaluationGraphMemory:
    model: GraDPertV2
    index: NeighborhoodIndex
    values: Tensor
    parameter_versions: tuple[tuple[int, torch.dtype, torch.device], ...]

    def select(self, model: GraDPertV2, index: NeighborhoodIndex, ids: Tensor) -> Tensor:
        if (
            model is not self.model
            or index is not self.index
            or model.training
            or ids.device != self.values.device
            or model.randomize_relay_order
            or self.parameter_versions != _versions(model)
        ):
            raise ValueError("evaluation graph reuse requires the same frozen model/index")
        return self.values[ids]


@torch.no_grad()
def prepare_evaluation_graph(
    model: GraDPertV2, index: NeighborhoodIndex, *, device: torch.device
) -> EvaluationGraphMemory | None:
    """Build once per evaluation; never persist across checkpoints or training.

    Relay nonconditional graph layers already update the full graph before query
    selection. Control-conditioned graphs cannot reuse the final representation.
    Other graph architectures retain their existing execution path.
    """
    model.eval()
    if (
        not index.relay
        or model.graph.read_mode != "relay"
        or model.options.control_conditioned_graph
    ):
        return None
    view = index.view(
        np.arange(index.n_nodes, dtype=np.int64),
        [(0,)],
        rng=np.random.default_rng(0),
        device=device,
        induced=False,
    )
    values = model.graph(view.ids, view.neighbors, view.valid, view.sources, context=view.context)
    return EvaluationGraphMemory(model, index, values, _versions(model))
