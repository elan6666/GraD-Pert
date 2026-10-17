"""Explicit self-distillation pretraining followed by prediction-only finetuning."""

from __future__ import annotations

from typing import Any

import torch

from .objective import JointObjective
from .optimizer import V2Optimizer


class StagedObjective(JointObjective):
    """One identity, a checkpointed phase, and a single optimizer-moment reset.

    Global successful-update counts remain continuous; Adam/Muon moments do not.
    Default JointObjective checkpoints never acquire these additional state keys.
    """

    def __init__(self, *args: Any, pretrain_steps: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if type(pretrain_steps) is not int or pretrain_steps < 1:
            raise ValueError("pretraining requires a positive successful-update budget")
        if not self.lambda1 or not self.lambda2:
            raise ValueError("staged pretraining requires both self-distillation branches")
        if self.population_response or self.auxiliary_mask_ratio or self.masked_response_ratio:
            raise ValueError("staged mode cannot silently add other supervised objectives")
        self.decode_ssl_views = False
        self.pretrain_steps = pretrain_steps
        self.pretrain_lambdas = (self.lambda1, self.lambda2)
        self.register_buffer("stage_phase", torch.zeros((), dtype=torch.long))
        self.register_load_state_dict_post_hook(self._restore_phase)
        self._apply_phase(0)

    def _apply_phase(self, phase: int) -> None:
        if phase not in (0, 1):
            raise ValueError("unknown staged-training phase")
        self.prediction_enabled = bool(phase)
        self.teacher_updates_enabled = not bool(phase)
        self.lambda1, self.lambda2 = (0.0, 0.0) if phase else self.pretrain_lambdas

    def _restore_phase(self, module: torch.nn.Module, incompatible_keys: object) -> None:
        self._apply_phase(int(self.stage_phase.item()))

    def prepare_update(self, optimizer: V2Optimizer) -> None:
        phase = int(optimizer.steps >= self.pretrain_steps)
        stage = "mse_finetune" if phase else "ssl_pretrain"
        previous = getattr(optimizer, "training_stage", None)
        if previous not in (None, "ssl_pretrain", "mse_finetune"):
            raise ValueError("unknown saved optimizer training stage")
        if previous == "mse_finetune" and not phase:
            raise ValueError("cannot rewind a finetuning optimizer into pretraining")
        if phase and previous != "mse_finetune":
            if optimizer.steps != self.pretrain_steps:
                raise ValueError("optimizer reset must occur at the exact stage boundary")
            optimizer.zero_grad()
            for opt in (optimizer.muon, optimizer.adamw):
                if opt is not None:
                    opt.state.clear()
            optimizer.stage_reset_count += 1
        optimizer.training_stage = stage
        self.stage_phase.fill_(phase)
        self._apply_phase(phase)

    @property
    def training_stage(self) -> str:
        return "mse_finetune" if self.prediction_enabled else "ssl_pretrain"
