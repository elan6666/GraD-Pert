"""Partial fixed-probe objectives never stand in for global-population KoLeo."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .objective import JointObjective, TrainingBatch


def diagnostic_objective(
    objective: JointObjective, batch: TrainingBatch, **kwargs: Any
) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
    """Call inside the caller's existing read-only/RNG scope.

    Fixed small probes do not contain the complete effective training population.
    Defer KoLeo neighborhood construction, discard the temporary CLS references,
    and label the remaining objective explicitly. No weights or modes are changed.
    Disabled KoLeo takes the exact original forward path and adds no metadata.
    """
    if not (objective.lambda2 and objective.weights[1][2]):
        loss, terms = objective(batch, **kwargs)
        return loss, terms, {}
    if objective.loss_reduction is None:
        raise ValueError("fixed-probe KoLeo exclusion requires unified loss_reduction")
    if "deferred_koleo" in kwargs:
        raise ValueError("diagnostic objective owns its unavailable KoLeo population")
    deferred: list[tuple[Tensor, Tensor, Tensor]] = []
    try:
        loss, terms = objective(batch, deferred_koleo=deferred, **kwargs)
    finally:
        deferred.clear()
    terms = {key: value for key, value in terms.items() if key != "ssl2_koleo"}
    return (
        loss,
        terms,
        {
            "diagnostic_objective_scope": {
                "unavailable_terms": {"ssl2_koleo": "requires_complete_global_population"},
                "joint_scope": "prediction_and_ssl_excluding_unavailable_koleo",
                "ssl2_gradient_scope": "dino_and_ibot_excluding_unavailable_koleo",
            }
        },
    )
