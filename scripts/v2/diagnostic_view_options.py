"""Validated options for native diagnostic sampling, separate from training."""

from dataclasses import replace


def diagnostic_view_options(options, *, deep):
    """Enable the historical full view recipe without enabling training losses.

    Native assemble_batch uses lambda1/2 only as sampling switches. A frozen
    readout configuration correctly rejects nonzero SSL losses, so its deep
    sampling clone declares an ordinary joint schema and removes readout-only
    parent fields. This validated clone is solely an assemble_batch argument:
    never install it on the runtime, model, objective, optimizer or saved config.
    The actual frozen objective retains zero SSL and frozen Teacher/centers.
    """
    changes = dict(lambda1=float(deep), lambda2=float(deep))
    if deep and getattr(options, "training_mode", "joint") == "frozen_readout":
        changes.update(
            training_mode="joint",
            readout_parent_run_root="",
            readout_parent_checkpoint_sha256="",
            readout_parent_source_commit="",
            readout_parent_run_id="",
            readout_parent_epoch=0,
        )
    if not deep:
        if getattr(options, "au_alignment_weight", 0.0) or getattr(
            options, "au_uniformity_weight", 0.0
        ):
            changes.update(au_alignment_weight=0.0, au_uniformity_weight=0.0)
        if getattr(options, "cr_weight", 0.0):
            changes.update(cr_weight=0.0)
        for name in ("spectral_rank_weight", "spectral_cosine_weight"):
            if hasattr(options, name):
                changes[name] = 0.0
    return replace(options, **changes)
