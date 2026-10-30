"""Opt-in complete-condition response anchoring on the native update clock."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from gradpert.hashing import sha256_file

from .objective import TrainingBatch
from .response_anchor import (
    PAIR_MODE,
    FixedResponseBank,
    group_loss_coefficient,
    group_mean_loss,
    group_schedule,
    isolated_rng,
    pair_schedule,
    prediction_contrast_loss,
)

if TYPE_CHECKING:
    from .runtime import Runtime

SERVER_ROOT = Path("/data/yilangliu")


def load_response_bank(path: str, digest: str, allowed: np.ndarray) -> FixedResponseBank:
    """Enforce the server boundary before either scientific array file is read."""
    manifest = Path(path)
    if (
        manifest.is_symlink()
        or manifest.resolve(strict=True) != manifest.absolute()
        or not manifest.resolve().is_relative_to(SERVER_ROOT)
        or sha256_file(manifest) != digest
    ):
        raise ValueError("response anchor manifest must be a sealed server file")
    metadata = json.loads(manifest.read_text())
    relative = metadata["arrays"]["path"]
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("response anchor arrays must be adjacent to the manifest")
    arrays = manifest.parent / relative
    if arrays.is_symlink() or arrays.resolve(strict=True).parent != manifest.resolve().parent:
        raise ValueError("response anchor arrays escaped the sealed bank directory")
    return FixedResponseBank.from_files(manifest, digest, allowed)


def population_prediction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Compare the complete native prediction population mean in FP32."""
    if (
        prediction.ndim != 2
        or not prediction.shape[0]
        or target.shape != prediction.shape[1:]
        or target.requires_grad
    ):
        raise ValueError("population loss requires complete cells and a fixed query target")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("nonfinite population prediction or target")
    return (prediction.float().mean(dim=0) - target.detach().float()).square().mean()


class ResponseAnchorController:
    """External bank/cursor-free schedule; no target arrays enter state_dict."""

    def __init__(self, runtime: Runtime, bank: FixedResponseBank) -> None:
        self.runtime, self.bank = runtime, bank
        identity = bank.identity
        if (
            identity.get("split_content_sha256") != runtime.data.split.split_content_sha256
            or identity.get("selection") != runtime.identity.get("training_row_selection")
            or identity.get("allowed_policy") != runtime.identity.get("training_expression_policy")
            or not set(bank.condition_ids) <= set(runtime.data.split.train_conditions)
        ):
            raise ValueError(
                "response anchor bank differs from native TRAIN/split/holdout identity"
            )
        if runtime.steps_per_epoch != 150 or len(bank.condition_ids) != 1322:
            raise ValueError("response anchor requires the frozen cap40 population/update budget")
        if any(
            g >= len(runtime.data.graph_gene_ids)
            or {runtime.data.graph_gene_ids[int(g)]}
            != (set(condition.split("+")) - {runtime.data.split.control_condition_id})
            for g, condition in zip(
                bank.condition_anchor_gene_indices, bank.condition_ids, strict=True
            )
        ):
            raise ValueError("response anchor single-target identity is not the native graph gene")
        mode = getattr(runtime.options, "response_anchor_loss", "mse")
        if mode not in ("mse", "cosine_direction", "population_prediction", PAIR_MODE):
            raise ValueError("unknown response anchor loss")
        if mode == PAIR_MODE and not getattr(
            runtime.options, "response_anchor_prediction_head_grad", True
        ):
            raise ValueError("pair contrast requires normal native prediction head gradients")
        if mode in ("population_prediction", PAIR_MODE):
            from .population_anchor import load_population_targets

            self.population_targets = load_population_targets(
                runtime.options.response_anchor_population_path,
                runtime.options.response_anchor_population_sha256,
                runtime.options.response_anchor_population_arrays_sha256,
                bank,
            )
        if mode == "cosine_direction":
            norms = torch.linalg.vector_norm(
                torch.from_numpy(bank.condition_targets.copy()).float(), dim=1
            )
            if not torch.isfinite(norms).all() or bool((norms <= 1e-6).any()):
                raise ValueError("cosine direction bank target norms must be finite and >1e-6")
            self._direction_targets = {
                "response_anchor_loss": mode,
                "eps": 1e-6,
                "target_count": len(norms),
                "target_norm_min": float(norms.min()),
                "target_norm_max": float(norms.max()),
                "target_clamped_count": int((norms <= 1e-6).sum()),
            }
            self._direction_diagnostics: dict[str, float] = {}
        self.world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def direction_summary(self) -> dict[str, str | float | int | None]:
        """Return local scalar observations only; they never control loss or sampling."""
        if not hasattr(self, "_direction_targets"):
            return {}
        return {
            **self._direction_targets,
            "head_observation_count": 0,
            "head_norm_min": None,
            "head_norm_max": None,
            "head_clamped_count": 0,
            **self._direction_diagnostics,
        }

    def groups(self, step: int) -> tuple[int, ...] | tuple[tuple[int, int], ...]:
        if type(step) is not int or step < 0:
            raise ValueError("response anchor requires the successful native update cursor")
        epoch, within = divmod(step, self.runtime.steps_per_epoch)
        schedule = pair_schedule if self.pair_mode else group_schedule
        return schedule(
            len(self.bank.condition_ids),
            self.runtime.steps_per_epoch,
            epoch,
            self.runtime.options.response_anchor_seed,
            self.world,
            self.rank,
        )[within]

    @property
    def pair_mode(self) -> bool:
        return getattr(self.runtime.options, "response_anchor_loss", "mse") == PAIR_MODE

    @property
    def loss_units(self) -> int:
        return len(self.bank.condition_ids) // (2 if self.pair_mode else 1)

    def unit_rng(self, unit: int | tuple[int, int], epoch: int) -> AbstractContextManager[None]:
        if not self.pair_mode:
            if not isinstance(unit, int):
                raise ValueError("group mode requires a condition index")
            return self.group_rng(unit, epoch)
        if not isinstance(unit, tuple) or len(unit) != 2:
            raise ValueError("pair mode requires two condition indices")
        digest = hashlib.sha256(
            f"population-pair:{self.runtime.options.response_anchor_seed}:{epoch}:{unit[0]}:{unit[1]}".encode()
        ).digest()
        device = self.runtime.device
        return isolated_rng(
            int.from_bytes(digest[:8], "big") % 2**63,
            (device.index or 0,) if device.type == "cuda" else (),
        )

    def unit_loss(
        self,
        batch: TrainingBatch,
        unit: int | tuple[int, int],
        epoch: int,
        *,
        bf16: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> Tensor:
        if not self.pair_mode:
            if not isinstance(unit, int):
                raise ValueError("group mode requires a condition index")
            return self.group_loss(batch, unit, epoch, bf16=bf16)
        if not isinstance(unit, tuple) or len(unit) != 2:
            raise ValueError("pair mode requires two condition indices")
        if not getattr(self.runtime.options, "response_anchor_prediction_head_grad", True):
            raise ValueError("pair contrast requires normal native prediction head gradients")
        p, q = unit
        if any(type(c) is not int or not 0 <= c < len(self.bank.condition_ids) for c in (p, q)):
            raise ValueError("pair condition outside frozen bank")
        if p == q:
            raise ValueError("self pair forbidden")
        # The caller keeps unit_rng alive through backward/checkpoint recomputation.
        left, tp = self.native_group_forward(batch, p, epoch, bf16=bf16)
        right, tq = self.native_group_forward(batch, q, epoch, bf16=bf16)
        loss = prediction_contrast_loss(left["prediction"], right["prediction"], tp, tq)
        if diagnostics is not None:
            mp, mq = left["prediction"].float().mean(0), right["prediction"].float().mean(0)
            delta, truth = mp - mq, tp.float() - tq.float()
            diagnostics.update(
                target_contrast_rms=float(truth.square().mean().sqrt()),
                prediction_contrast_rms=float(delta.detach().square().mean().sqrt()),
                contrast_error_rms=float(loss.detach().sqrt()),
                common_error_energy=float(
                    ((mp.detach() - tp + mq.detach() - tq) / 2).square().mean()
                ),
                difference_error_energy=float(loss.detach()),
                group_rows=[len(left["prediction"]), len(right["prediction"])],
            )
            diagnostics["absolute_loss"] = (
                population_prediction_loss(left["prediction"], tp)
                + population_prediction_loss(right["prediction"], tq)
            ) / 2
        return loss

    def group_rng(self, condition: int, epoch: int) -> AbstractContextManager[None]:
        digest = hashlib.sha256(
            f"anchor:{self.runtime.options.response_anchor_seed}:{epoch}:{condition}".encode()
        ).digest()
        device = self.runtime.device
        devices = (device.index or 0,) if device.type == "cuda" else ()
        return isolated_rng(int.from_bytes(digest[:8], "big") % 2**63, devices)

    def native_group_forward(
        self, batch: TrainingBatch, condition: int, epoch: int, *, bf16: bool = False
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Shared native group assembly/forward, with no Teacher or readout duplication."""
        runtime, bank = self.runtime, self.bank
        if not runtime.objective.student.training:
            raise ValueError("response anchor is a TRAIN-only auxiliary path")
        queries = batch.graph.ids[batch.query_positions].detach().cpu().numpy()
        positions = np.searchsorted(bank.allowed_gene_indices, queries)
        if (positions >= len(bank.allowed_gene_indices)).any() or not np.array_equal(
            bank.allowed_gene_indices[positions], queries
        ):
            raise ValueError("anchor query axis includes excluded expression")
        control = torch.from_numpy(bank.control_for_condition(condition)[:, positions].copy()).to(
            runtime.device
        )
        population = getattr(runtime.options, "response_anchor_loss", "mse") in (
            "population_prediction",
            PAIR_MODE,
        )
        target_values = (
            self.population_targets[condition, positions]
            if population
            else bank.condition_targets[condition]
        )
        target = torch.from_numpy(target_values.copy()).to(runtime.device)
        anchor = int(bank.condition_anchor_gene_indices[condition])
        ids = np.union1d(queries, [anchor])
        # Private streams include graph construction and stochastic forward. Main
        # view seeds, pending statistics and optimizer/EMA clocks are untouched.
        with self.group_rng(condition, epoch):
            graph = runtime.index.view(
                ids,
                [[anchor]],
                rng=np.random.default_rng(
                    runtime.options.response_anchor_seed + epoch * 1322 + condition
                ),
                device=runtime.device,
                induced=False,
            )
            auxiliary = TrainingBatch(
                graph,
                torch.tensor(np.searchsorted(ids, queries), device=runtime.device),
                control,
                torch.zeros_like(control),
                torch.zeros(len(control), device=runtime.device, dtype=torch.long),
                (),
                (),
            )
            if self.pair_mode and not torch.equal(
                graph.ids[auxiliary.query_positions], batch.graph.ids[batch.query_positions]
            ):
                raise ValueError("pair native graph changed the ordered query gene IDs")
            with torch.autocast(runtime.device.type, dtype=torch.bfloat16, enabled=bf16):
                gene, perturbation = runtime.objective._batch_graph(
                    runtime.objective.student, auxiliary
                )
                decoder_options: dict[str, Any] = {}
                if population and not getattr(
                    runtime.options, "response_anchor_prediction_head_grad", True
                ):
                    decoder_options["prediction_head_grad"] = False
                outputs = runtime.objective.student.encode_response(
                    runtime.objective._select_graph(gene, auxiliary.query_positions),
                    control,
                    perturbation,
                    decode=population,
                    **decoder_options,
                    **runtime.objective.response_metadata(auxiliary),
                )
        return outputs, target

    def group_loss(
        self, batch: TrainingBatch, condition: int, epoch: int, *, bf16: bool = False
    ) -> Tensor:
        outputs, target = self.native_group_forward(batch, condition, epoch, bf16=bf16)
        if getattr(self.runtime.options, "response_anchor_loss", "mse") in (
            "population_prediction",
            PAIR_MODE,
        ):
            loss = population_prediction_loss(outputs["prediction"], target)
        else:
            # Keep the historical readout autocast precision after native forward.
            with torch.autocast(self.runtime.device.type, dtype=torch.bfloat16, enabled=bf16):
                loss = group_mean_loss(
                    outputs["response_cls"],
                    target,
                    self.runtime.objective.student.response_anchor_readout,
                    getattr(self.runtime.options, "response_anchor_loss", "mse"),
                    diagnostics=getattr(self, "_direction_diagnostics", None),
                )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite response anchor group loss")
        return loss

    def backward(
        self,
        batch: TrainingBatch,
        step: int,
        *,
        bf16: bool = False,
        weight_override: float | None = None,
    ) -> dict[str, float]:
        weight = (
            self.runtime.options.response_anchor_weight
            if weight_override is None
            else weight_override
        )
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("invalid anchor gradient weight")
        coefficient = group_loss_coefficient(
            self.loss_units, self.runtime.steps_per_epoch, self.world, weight
        )
        total, rows, complete40 = 0.0, 0, 0
        groups = self.groups(step)
        epoch = step // self.runtime.steps_per_epoch
        for unit in groups:
            # Keep the private RNG active through recomputation/backward too.
            with self.unit_rng(unit, epoch):
                loss = self.unit_loss(batch, unit, epoch, bf16=bf16)
                (coefficient * loss).backward()
            total += float(loss.detach())
            conditions = unit if isinstance(unit, tuple) else (unit,)
            for condition in conditions:
                count = int(np.diff(self.bank.control_slot_offsets)[condition])
                rows += count
                complete40 += int(count == 40)
        local = total * self.runtime.steps_per_epoch / self.loss_units
        metrics = {
            "response_anchor_loss": local,
            "response_anchor_weighted": weight * local,
            "response_anchor_groups": float(len(groups) * (2 if self.pair_mode else 1)),
            "response_anchor_rows": float(rows),
            "response_anchor_complete40_groups": float(complete40),
        }

        if self.pair_mode:
            metrics.update(
                response_anchor_pairs=float(len(groups)),
                response_anchor_pair_live_cells_max=float(
                    max(
                        (
                            sum(
                                int(np.diff(self.bank.control_slot_offsets)[p])
                                for p in pair
                                if isinstance(pair, tuple)
                            )
                            for pair in groups
                        ),
                        default=0,
                    )
                ),
            )
        return metrics
