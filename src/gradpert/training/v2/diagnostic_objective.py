"""Partial probes explicitly omit unavailable effective-population objectives."""

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
    koleo = bool(objective.lambda2 and objective.weights[1][2])
    au = bool(getattr(objective, "au_enabled", False))
    cr = bool(getattr(objective, "cr_enabled", False))
    spectral = bool(getattr(objective, "spectral_enabled", False))
    if not (koleo or au or cr or spectral):
        loss, terms = objective(batch, **kwargs)
        return loss, terms, {}
    if objective.loss_reduction is None:
        raise ValueError("fixed-probe KoLeo exclusion requires unified loss_reduction")
    if "deferred_koleo" in kwargs or "deferred_au" in kwargs:
        raise ValueError("diagnostic objective owns its unavailable representation population")
    deferred: list[tuple[Tensor, Tensor, Tensor]] = []
    collection: dict[str, Any] = {}
    if koleo:
        collection["deferred_koleo"] = deferred
    if au or cr or spectral:
        collection["omit_au_for_diagnostic"] = True
    try:
        loss, terms = objective(batch, **collection, **kwargs)
    finally:
        deferred.clear()
    unavailable = (["ssl2_koleo"] if koleo else []) + (
        ["au_alignment", "au_uniformity"] if au else []
    )
    if cr and not objective.au_uniformity_weight:
        unavailable = [key for key in unavailable if key != "au_uniformity"]
    unavailable += ["cr_negative"] if cr else []
    terms = {key: value for key, value in terms.items() if key not in unavailable}
    unavailable += ["spectral_rank", "spectral_cosine"] if spectral else []
    if getattr(objective, "spectral_amplitude_weight", 0):
        unavailable.append("spectral_amplitude")
    suffix = "representation_terms" if au or cr or spectral else "koleo"
    return (
        loss,
        terms,
        {
            "diagnostic_objective_scope": {
                "unavailable_terms": {
                    key: "requires_complete_global_population" for key in unavailable
                },
                "joint_scope": f"prediction_and_ssl_excluding_unavailable_{suffix}",
                "ssl2_gradient_scope": f"dino_and_ibot_excluding_unavailable_{suffix}",
            }
        },
    )
