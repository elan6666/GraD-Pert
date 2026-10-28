"""Canonical server data and model construction for the independent v2 runner."""

from __future__ import annotations

import random
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from gradpert.config.schema import ExperimentConfig
from gradpert.config.v2 import V2Options
from gradpert.features.text_prior import verify_text_prior_npz
from gradpert.graphs.materialization import DatasetGraphLayout, load_dataset_graph_topology
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.data import CanonicalTrainingData
from gradpert.training.expression_policy import expression_policy

from .objective import JointObjective, TrainingBatch
from .optimizer import V2Optimizer
from .views import NeighborhoodIndex, assemble_batch


@dataclass
class Runtime:
    data: CanonicalTrainingData
    options: V2Options
    index: NeighborhoodIndex
    objective: JointObjective
    optimizer: V2Optimizer
    generator: np.random.Generator
    device: torch.device
    batch_size: int
    identity: dict[str, Any]
    allowed_expression_ids: np.ndarray[Any, Any] | None = None
    purpose: Literal["training", "evaluation"] = "training"

    @cached_property
    def _au_context_codes(self) -> tuple[int, ...]:
        codes = {value: index for index, value in enumerate(sorted(set(self.data.context_ids)))}
        return tuple(codes[value] for value in self.data.context_ids)

    def au_context_indices(self, raw: Any) -> torch.Tensor | None:
        """Map controls to stable data-wide context codes without changing sampling."""
        if not (
            getattr(self.options, "au_alignment_weight", 0)
            or getattr(self.options, "au_uniformity_weight", 0)
            or getattr(self.options, "cr_weight", 0)
        ):
            return None
        return torch.tensor(
            [self._au_context_codes[self.data._row_index[row]] for row in raw.control_row_ids],
            device=raw.control_expression.device,
            dtype=torch.long,
        )

    @cached_property
    def steps_per_epoch(self) -> int:
        if self.purpose != "training":
            raise RuntimeError("evaluation runtime cannot enter training")
        return self.data.steps_per_epoch(
            batch_size=self.batch_size,
            max_unique_conditions=(
                0
                if self.options.max_conditions == 0
                else min(self.options.max_conditions, self.batch_size)
            ),
        )

    def batches(self, epoch: int, *, cpu_prefetch: bool = False) -> Iterator[TrainingBatch]:
        if not cpu_prefetch:
            yield from self._batches(epoch, self.device, self.generator)
            return
        from .prefetch import prefetch_cpu

        stream = prefetch_cpu(
            lambda rng: self._batches(epoch, torch.device("cpu"), rng), self.generator
        )
        try:
            for batch in stream:
                yield cast(TrainingBatch, _move_batch_tree(batch, self.device))
        finally:
            stream.close()

    def _batches(
        self,
        epoch: int,
        device: torch.device,
        generator: np.random.Generator,
        *,
        diagnostic: bool = False,
    ) -> Generator[TrainingBatch, None, None]:
        if self.purpose != "training":
            raise RuntimeError("evaluation runtime cannot enter training")
        staged = self.options.training_mode in ("ssl_then_mse", "ssl_then_joint")
        pretraining = staged and (
            not self.objective.prediction_enabled
            if diagnostic
            else self.optimizer.steps < self.objective.pretrain_steps
        )
        self.data.control_only_training = pretraining
        options = (
            replace(self.options, lambda1=0.0, lambda2=0.0)
            if self.options.training_mode == "ssl_then_mse" and not pretraining
            else self.options
        )
        sampler_epoch = (
            max(0, epoch - self.options.ssl_pretrain_epochs)
            if self.options.training_mode == "ssl_then_joint" and not pretraining
            else epoch
        )
        for raw in self.data.iter_train_epoch(
            epoch=sampler_epoch,
            device=device,
            batch_size=self.batch_size,
            max_unique_conditions=(
                0
                if self.options.max_conditions == 0
                else min(self.options.max_conditions, self.batch_size)
            ),
        ):
            yield assemble_batch(
                raw,
                self.index,
                options,
                generator,
                allowed_expression_ids=self.allowed_expression_ids,
                context_index=self.au_context_indices(raw),
            )

    def loss_diagnostics(self) -> dict[str, Any]:
        """Use four fixed training rows and private views; never validation/test rows."""
        from .engine import slice_cells
        from .loss_diagnostics import evaluate_loss_diagnostics

        generator = np.random.default_rng(int(self.identity["run_seed"]) + 0x71A1)
        stream = self._batches(0, self.device, generator, diagnostic=True)
        try:
            batch = next(stream)
        finally:
            stream.close()
        fixed = slice_cells(batch, 0, min(4, len(batch.control)))
        result = evaluate_loss_diagnostics(self.objective, fixed, bf16=True)
        if self.options.process_diagnostics:
            from .process_diagnostics import (
                evaluate_graph_gradient_diagnostics,
                evaluate_process_diagnostics,
            )

            result["process"] = evaluate_process_diagnostics(
                self.objective, fixed, bf16=True, seed=0x71A1
            )
            result["process"]["graph_gradients"] = evaluate_graph_gradient_diagnostics(
                self.objective, fixed, bf16=True
            )
        return result

    def validation_batches(self, *, batch_size: int) -> Iterator[tuple[TrainingBatch, str]]:
        """Replay one fixed validation view recipe without consuming training RNG."""
        if self.purpose != "training" or batch_size < 2:
            raise ValueError("joint validation requires a training runtime and batch >= 2")
        generator = np.random.default_rng(int(self.identity["run_seed"]) + 0x5A11D)
        for raw in self.data.iter_validation_epoch(
            device=self.device,
            batch_size=batch_size,
            max_unique_conditions=(
                0
                if self.options.max_conditions == 0
                else min(self.options.max_conditions, batch_size)
            ),
        ):
            identity = sha256_json(
                {
                    "perturbed_row_ids": list(raw.perturbed_row_ids),
                    "control_row_ids": list(raw.control_row_ids),
                    "condition_ids": list(raw.condition_ids),
                }
            )
            yield (
                assemble_batch(
                    raw,
                    self.index,
                    self.options,
                    generator,
                    allowed_expression_ids=self.allowed_expression_ids,
                    context_index=self.au_context_indices(raw),
                ),
                identity,
            )


def _move_batch_tree(value: Any, device: torch.device) -> Any:
    """Transfer immutable batch structure on the consumer thread, without RNG draws."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _move_batch_tree(getattr(value, field.name), device)
                for field in fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(_move_batch_tree(item, device) for item in value)
    return value


def validate_world(configured: int, observed: int, purpose: str) -> None:
    if purpose == "training":
        if configured != observed:
            raise ValueError("configured world size differs from the process group")
    elif purpose == "evaluation":
        if observed != 1:
            raise ValueError("standalone evaluation requires one process")
    else:
        raise ValueError("unknown runtime purpose")


@contextmanager
def prepare_runtime(
    config: ExperimentConfig,
    *,
    data_root: Path,
    run_seed: int,
    device: torch.device,
    purpose: Literal["training", "evaluation"] = "training",
) -> Iterator[Runtime]:
    """Verify sealed inputs before model allocation; close backed H5AD on all exits.

    Source publication, allocator and compute admission are execution-level gates.
    This constructor is shared by full training and the capacity runner.
    """
    if config.model_id != "gradpert_v2" or config.model.version != "v2":
        raise ValueError("v2 runtime cannot execute a v1 configuration")
    arch, options = V2Options.parse_parameters(config.model.parameters)
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    validate_world(options.world_size, world, purpose)
    if world > 1 and device.type == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("each distributed worker must expose exactly one physical GPU")
    if run_seed not in config.training.run_seeds:
        raise ValueError("seed differs from sealed configuration")
    if not data_root.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("scientific data stays on the server")
    graph_manifest = DatasetGraphLayout(
        data_root, config.dataset_id, config.data.protocol_id
    ).manifest
    if Path(options.graph_manifest_path).resolve() != graph_manifest.resolve():
        raise ValueError("configured graph manifest differs from runtime dataset")
    if sha256_file(graph_manifest) != options.graph_manifest_sha256:
        raise ValueError("graph manifest checksum mismatch")
    topology = load_dataset_graph_topology(
        dataset_id=config.dataset_id, protocol_id=config.data.protocol_id, data_root=data_root
    )
    prior_path = Path(options.genept_artifact_path).resolve()
    if not prior_path.is_relative_to("/data/yilangliu"):
        raise ValueError("GenePT artifacts stay on the server")
    prior = verify_text_prior_npz(
        prior_path,
        expected_sha256=options.genept_sha256,
        expected_gene_ids=topology.gene_ids,
        perturbation_target_gene_ids=topology.gene_ids,
    )
    if arch.learned_genept_projection and prior.embedding_width != 2048:
        raise ValueError("U4 requires the original 2048-wide GenePT prior, not PCA")
    if prior.gene_ids != topology.gene_ids:
        raise ValueError("v2 requires complete GenePT coverage of the frozen graph axis")
    with ExitStack() as stack:
        data = stack.enter_context(
            CanonicalTrainingData(
                dataset_id=config.dataset_id,
                protocol_id=config.data.protocol_id,
                data_root=data_root,
                run_seed=run_seed,
            )
        )
        data.require_experiment_data_contract(
            registry_version=config.data.registry_version, split_policy=config.data.split_policy
        )
        if tuple(data.graph_gene_ids) != topology.gene_ids:
            raise ValueError("training and graph axes differ")
        if tuple(data.expression_gene_ids) != topology.gene_ids[: len(data.expression_gene_ids)]:
            raise ValueError("v2 expression axis must be the graph prefix")
        train_selection_receipt = None
        if options.train_selection_path:
            from .train_selection import apply_train_selection

            selection_path = Path(options.train_selection_path)
            if not selection_path.resolve().is_relative_to("/data/yilangliu"):
                raise ValueError("training row selection must be sealed on the server")
            train_selection_receipt = apply_train_selection(
                data, selection_path, options.train_selection_sha256
            )
        data.population_batching = options.population_response
        allowed_expression_ids, expression_policy_receipt = expression_policy(
            tuple(data.expression_gene_ids),
            tuple(data.split.test_conditions),
            data.split.control_condition_id,
            enabled=config.model.excludes_test_target_expression,
        )
        if options.expression_holdout_path:
            from .holdout import load_partition

            holdout_path = Path(options.expression_holdout_path).resolve()
            if not holdout_path.is_relative_to("/data/yilangliu"):
                raise ValueError("expression partition must be sealed on the server")
            g1_allowed = load_partition(
                holdout_path, options.expression_holdout_sha256, tuple(data.expression_gene_ids)
            )
            if allowed_expression_ids is not None:
                axis = np.arange(len(data.expression_gene_ids))
                if len(
                    np.intersect1d(
                        np.setdiff1d(axis, allowed_expression_ids),
                        np.setdiff1d(axis, g1_allowed),
                    )
                ):
                    raise ValueError("G1 must sample additional genes outside default exclusions")
            allowed_expression_ids = (
                g1_allowed
                if allowed_expression_ids is None
                else np.intersect1d(allowed_expression_ids, g1_allowed)
            )
        if allowed_expression_ids is not None and len(allowed_expression_ids) < options.query_count:
            raise ValueError("training expression partition smaller than query budget")
        if options.response_anchor_weight and purpose == "training":
            if allowed_expression_ids is None:
                raise ValueError("response anchor requires expression-column holdout")
            data.configure_training_expression_columns(allowed_expression_ids)
        if options.training_mode == "frozen_readout" and purpose == "training":
            steps = data.steps_per_epoch(
                batch_size=int(config.training.train_batch_size.value),
                max_unique_conditions=(
                    0
                    if options.max_conditions == 0
                    else min(options.max_conditions, int(config.training.train_batch_size.value))
                ),
            )
            if int(config.training.max_epochs.value) != 2 or steps != 150:
                raise ValueError("frozen readout requires exactly two epochs and 300 updates")
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)
        seed = torch.from_numpy(prior.values.copy())
        if options.gene_initialization == "random":
            seed = torch.randn_like(seed) * 0.02
        student = GraDPertV2(seed, arch).to(device)
        if options.response_anchor_weight:
            from .response_anchor import attach_anchor_readout

            if arch.width != 256 or config.continuation is not None:
                raise ValueError("response anchor requires a fresh native width256 model")
            attach_anchor_readout(student, arch.width, options.response_anchor_seed)
        if config.continuation is not None and config.continuation.mode == "lora":
            from gradpert.modeling.v2.lora import insert_lora

            assert config.continuation.lora_rank is not None
            assert config.continuation.lora_alpha is not None
            insert_lora(
                student,
                rank=config.continuation.lora_rank,
                alpha=config.continuation.lora_alpha,
            )
        from .staged import SSLThenJointObjective, StagedObjective

        staged = options.training_mode in ("ssl_then_mse", "ssl_then_joint")
        if staged and (
            config.continuation is not None
            or options.validation_mode != "disabled"
            or not 0 < options.ssl_pretrain_epochs < int(config.training.max_epochs.value)
        ):
            raise ValueError("staged training needs a fresh train-only budget with both stages")
        objective_type = (
            SSLThenJointObjective
            if options.training_mode == "ssl_then_joint"
            else StagedObjective
            if staged
            else JointObjective
        )
        if options.training_mode == "frozen_readout":
            from .readout import FrozenReadoutObjective

            if config.continuation is not None:
                raise ValueError("frozen readout is not generic continuation")
            objective_type = FrozenReadoutObjective
        objective = objective_type(
            student,
            lambda1=options.lambda1,
            lambda2=options.lambda2,
            ssl1_weights=(options.ssl1_condition, options.ssl1_node, options.ssl1_spread),
            ssl2_weights=(options.ssl2_dino, options.ssl2_ibot, options.ssl2_koleo),
            loss_reduction=options.loss_reduction,
            ssl1_reduction=options.ssl1_reduction,
            prediction_reduction=options.prediction_loss,
            koleo_exclude_same_condition=options.koleo_exclude_same_condition,
            prediction_error_power=options.prediction_error_power,
            prediction_reduction_override=options.prediction_reduction_override,
            auxiliary_mask_ratio=options.auxiliary_mask_ratio,
            lambda_gene_mask=options.lambda_gene_mask,
            lambda_cls_mask=options.lambda_cls_mask,
            auxiliary_seed=run_seed + 0x4A51,
            masked_response_ratio=options.masked_response_ratio,
            lambda_masked_response=options.lambda_masked_response,
            population_response=options.population_response,
            lambda_mmd=options.lambda_mmd,
            au_alignment_weight=options.au_alignment_weight,
            au_uniformity_weight=options.au_uniformity_weight,
            au_temperature=options.au_temperature,
            au_warmup_steps=options.au_warmup_steps,
            cr_weight=options.cr_weight,
            cr_epsilon_squared=options.cr_epsilon_squared,
            cr_warmup_steps=options.cr_warmup_steps,
            population_aux_space=options.population_aux_space,
            spectral_rank_target=options.spectral_rank_target,
            spectral_rank_weight=options.spectral_rank_weight,
            spectral_cosine_weight=options.spectral_cosine_weight,
            spectral_softness=options.spectral_softness,
            spectral_warmup_steps=options.spectral_warmup_steps,
            spectral_amplitude_target=options.spectral_amplitude_target,
            spectral_amplitude_weight=options.spectral_amplitude_weight,
            teacher_temperature_start=options.teacher_temperature_start,
            teacher_temperature_end=options.teacher_temperature_end,
            teacher_temperature_warmup_steps=(
                options.teacher_temperature_warmup_epochs
                * data.steps_per_epoch(
                    batch_size=int(config.training.train_batch_size.value),
                    max_unique_conditions=(
                        0
                        if options.max_conditions == 0
                        else min(
                            options.max_conditions, int(config.training.train_batch_size.value)
                        )
                    ),
                )
            ),
            **(
                {
                    "pretrain_steps": options.ssl_pretrain_epochs
                    * data.steps_per_epoch(
                        batch_size=int(config.training.train_batch_size.value),
                        max_unique_conditions=(
                            0
                            if options.max_conditions == 0
                            else min(
                                options.max_conditions, int(config.training.train_batch_size.value)
                            )
                        ),
                    )
                }
                if staged
                else {}
            ),
        ).to(device)
        optimizer = V2Optimizer(
            student,
            lr=float(config.training.learning_rate.value),
            weight_decay=float(config.training.weight_decay.value),
            prototype_freeze_steps=(
                arch.prototype_freeze_epochs
                * data.steps_per_epoch(
                    batch_size=int(config.training.train_batch_size.value),
                    max_unique_conditions=(
                        0
                        if options.max_conditions == 0
                        else min(
                            options.max_conditions, int(config.training.train_batch_size.value)
                        )
                    ),
                )
                if arch.prototype_freeze_epochs
                else 0
            ),
        )
        if world > 1:
            # Model initialization is shared; dropout streams are rank-specific.
            rank_seed = run_seed + torch.distributed.get_rank()
            torch.manual_seed(rank_seed)
            random.seed(rank_seed)
        identity: dict[str, Any] = {
            "canonical_sha256": data.manifest.canonical_adata_sha256,
            "split_sha256": data.split.split_content_sha256,
            "expression_gene_order_sha256": data.manifest.expression_gene_order_sha256,
            "graph_gene_order_sha256": sha256_json(list(topology.gene_ids)),
            "graph_manifest_sha256": options.graph_manifest_sha256,
            "genept_sha256": prior.source_sha256,
            "run_seed": run_seed,
        }
        identity["loss_protocol"] = {
            "version": (
                "unified-global-population-v2-relay"
                if arch.attention == "relay_full"
                else "unified-global-population-v1"
            ),
            "reduction": options.loss_reduction,
            "koleo_population": "complete_global_effective_batch",
            "koleo_exclusion": (
                "same_perturbation_condition"
                if options.koleo_exclude_same_condition
                else "self_only"
            ),
            "batch_order": (
                "random_mixed_seeded_v2"
                if options.max_conditions == 0
                else "condition_limited_seeded_v1"
            ),
            "ibot": "masked_tokens_per_cell_then_valid_cell_population",
            "exceptions": ["ssl1_node", "ssl1_spread", "teacher_centers"],
        }
        if options.training_mode == "frozen_readout":
            identity["loss_protocol"]["frozen_readout"] = {
                "mode": "existing_prediction_mlp_only",
                "prediction": "native_row_mean_mse_softplus",
                "inactive": ["ssl1", "ssl2", "alignment", "uniformity", "coding_rate"],
                "teacher_and_centers": "frozen_no_updates",
                "representation_mode": "eval_no_relay_randomization",
                "optimizer": "native_head_only_fresh_moments",
            }
        if staged:
            identity["loss_protocol"]["staged_training"] = {
                "mode": options.training_mode,
                "pretrain_epochs": options.ssl_pretrain_epochs,
                "pretrain_steps": objective.pretrain_steps,
                "finetune_epochs": int(config.training.max_epochs.value)
                - options.ssl_pretrain_epochs,
                "finetune_learning_rate": options.finetune_learning_rate,
                "pretrain_truth": "not_loaded_nan_sentinel",
                "transition": "student_retained_optimizer_moments_reset",
                "finetune_teacher_centers": "frozen_unused",
            }
        if isinstance(objective, SSLThenJointObjective):
            joint_epochs = int(config.training.max_epochs.value) - options.ssl_pretrain_epochs
            identity["loss_protocol"]["staged_training"] = {
                "mode": options.training_mode,
                "pretrain_epochs": options.ssl_pretrain_epochs,
                "pretrain_steps": objective.pretrain_steps,
                "joint_epochs": joint_epochs,
                "joint_steps": objective.pretrain_steps
                * joint_epochs
                // options.ssl_pretrain_epochs,
                "total_steps": objective.pretrain_steps
                * int(config.training.max_epochs.value)
                // options.ssl_pretrain_epochs,
                "pretrain_truth": "not_loaded_nan_sentinel",
                "transition": "student_teacher_centers_optimizer_moments_retained",
                "joint_teacher_centers": "continuously_updated",
                "learning_rate_clocks": "separate_prefix_and_joint_endpoint_cosines",
                "auxiliary_warmup_clock": "global_successful_updates",
                "sampler_epochs": "prefix_local_then_joint_local",
                "view_rng": "continuous_checkpointed_not_baseline_replay",
            }
        if objective.spectral_enabled:
            identity["loss_protocol"]["spectral_pair"] = {
                "rank_target": options.spectral_rank_target,
                "rank_weight": options.spectral_rank_weight,
                "cosine_weight": options.spectral_cosine_weight,
                "softness": options.spectral_softness,
                "warmup_completed_optimizer_updates": options.spectral_warmup_steps,
                "representation": "student_response_cls_after_final_rmsnorm_before_projector",
                "population_scope": (
                    "complete_effective_batch_all_microbatches_and_ranks_all_contexts"
                ),
                "deduplication": "control_row_id_and_condition_id_before_rank_partition",
                "rank_formula": (
                    "mean_globals_softness_softplus_log_target_minus_log_rho_over_softness"
                ),
                "rho": "exp_spectral_entropy_divided_by_min_D_Nminus1",
                "covariance": (
                    "separate_globals_raw_centered_population_fp64_eigvalsh_no_jitter_no_l2"
                ),
                "spectrum_roundoff": (
                    "negative_tolerance_64_D_eps64_max_abs_eigenvalue_then_clamp_zero"
                ),
                "near_zero_trace_rejection": 1e-24,
                "cosine": "same_input_two_globals_both_student_gradients",
                "partial_diagnostics": "unavailable_requires_complete_global_population",
            }
            if options.spectral_amplitude_weight:
                identity["loss_protocol"]["spectral_pair"]["amplitude"] = {
                    "target_q": options.spectral_amplitude_target,
                    "weight": options.spectral_amplitude_weight,
                    "formula": "mean_globals_relu_sqrt_target_minus_sqrt_q_plus_epsilon",
                    "q": "centered_population_covariance_trace_over_mean_raw_squared_norm",
                    "epsilon_q": 1e-8,
                    "denominator_gradients": True,
                    "statistics_dtype": "float64",
                    "near_zero_total_energy_rejection": 1e-24,
                    "q_roundoff": "64_D_eps64_tolerance_then_clamp_zero_one",
                }
        if objective.cr_enabled:
            identity["loss_protocol"]["coding_rate"] = {
                "weight": options.cr_weight,
                "epsilon_squared": options.cr_epsilon_squared,
                "warmup_completed_optimizer_updates": options.cr_warmup_steps,
                "representation": (
                    "student_ssl2_cls_features_before_l2_and_prototypes"
                    if options.population_aux_space == "ssl2_cls_bottleneck"
                    else "student_response_cls_after_final_rmsnorm_before_projector"
                ),
                "population_scope": (
                    "complete_effective_batch_all_microbatches_and_ranks_all_contexts"
                ),
                "deduplication": "control_row_id_and_condition_id_before_rank_partition",
                "formula": (
                    "mean_two_globals_negative_half_logdet_I_plus_d_over_N_epsilon_squared_UtU"
                ),
                "normalization": "separate_global_unit_rows_uncentered_fp64_cholesky",
                "partial_diagnostics": "unavailable_requires_complete_global_population",
            }
        if objective.au_enabled:
            identity["loss_protocol"]["alignment_uniformity"] = {
                "alignment_weight": options.au_alignment_weight,
                "uniformity_weight": options.au_uniformity_weight,
                "temperature": options.au_temperature,
                "warmup_completed_optimizer_updates": options.au_warmup_steps,
                "representation": (
                    "student_ssl2_cls_features_before_l2_and_prototypes"
                    if options.population_aux_space == "ssl2_cls_bottleneck"
                    else "student_response_cls_after_final_rmsnorm_before_projector"
                ),
                "population_scope": "complete_effective_batch_all_microbatches_and_ranks",
                "deduplication": "control_row_id_and_condition_id_before_rank_partition",
                "alignment": "same_control_and_condition_two_global_views_both_student_gradients",
                "uniformity_pairs": "unordered_same_cell_type_batch_context_different_condition",
                "reduction": "log_mean_exp_over_all_eligible_pairs_then_mean_two_globals",
                "partial_diagnostics": "unavailable_requires_complete_global_population",
            }
        if (
            options.prediction_error_power != 2
            or options.prediction_reduction_override != "inherit"
            or options.auxiliary_mask_ratio
        ):
            identity["loss_protocol"]["prediction_task"] = {
                "error_power": options.prediction_error_power,
                "reduction_override": options.prediction_reduction_override,
                "ssl_reduction_unchanged": options.loss_reduction,
            }
            identity["loss_protocol"]["control_reconstruction"] = {
                "mask_ratio": options.auxiliary_mask_ratio,
                "lambda_gene": options.lambda_gene_mask,
                "lambda_cls": options.lambda_cls_mask,
                "auxiliary_seed": run_seed + 0x4A51,
                "target": "training_control_expression",
                "inference_use": False,
            }
        if options.population_response or options.masked_response_ratio:
            identity["loss_protocol"]["new_response_supervision"] = {
                "population_response": options.population_response,
                "lambda_mmd": options.lambda_mmd,
                "mmd_bandwidth": "detached_training_truth_median_offdiagonal",
                "masked_response_ratio": options.masked_response_ratio,
                "lambda_masked_response": options.lambda_masked_response,
                "masked_raw_residual": "blocked",
                "epoch_budget": "each_selected_training_truth_once",
            }
        if options.population_response:
            identity["loss_protocol"]["batch_order"] = "same_condition_population_train_rows_once"
            identity["loss_protocol"]["population_scope"] = (
                "complete_effective_batch_all_microbatches_and_ranks"
            )
        if arch.control_conditioned_graph:
            identity["loss_protocol"]["conditional_ssl1_scope"] = (
                "local_control_rows_global_condition_weights_masked_node_mean"
            )
        if any(
            (
                arch.prior_shared_adapter,
                arch.gene_conditioned_readout,
                arch.direct_target_flag,
                arch.learned_genept_projection,
            )
        ):
            identity["unseen_gene_mechanisms"] = {
                "prior_shared_adapter": arch.prior_shared_adapter,
                "gene_conditioned_readout": arch.gene_conditioned_readout,
                "direct_target_flag": arch.direct_target_flag,
                "prior_source": "initial_reduced_genept_only",
                "adapter_bottleneck": max(1, arch.width // 4),
                "initialization": "zero_final_layer_separate_rng_stream",
            }
        if arch.learned_genept_projection:
            identity["unseen_gene_mechanisms"].update(
                learned_genept_projection=True,
                prior_source="original_2048_genept_frozen",
                prior_embedding_width=prior.embedding_width,
                adapter_bottleneck=None,
                initialization="xavier_uniform_zero_bias_separate_rng_stream",
                independent_gene_table=False,
                pca=False,
            )
            if arch.genept_projection_activation != "none":
                identity["unseen_gene_mechanisms"]["genept_projection_activation"] = (
                    arch.genept_projection_activation
                )
        if arch.mlp_profile == "unified" and arch.learned_genept_projection:
            identity["unseen_gene_mechanisms"]["shared_projection"] = {
                "linear": [prior.embedding_width, arch.width],
                "nonlinear": [arch.width, 4 * arch.width, arch.width],
                "activation": "clipped_swiglu",
                "rmsnorm_epsilon": 1e-6,
                "bias": True,
            }
            if arch.gene_conditioned_readout:
                identity["unseen_gene_mechanisms"]["readout_correction"] = {
                    "input": "same_frozen_raw_genept_no_duplicate_buffer",
                    "dimensions": [prior.embedding_width, max(1, arch.width // 4), arch.width],
                    "initialization": "zero_down_weight_and_bias_separate_rng_stream",
                    "activation": "clipped_swiglu",
                }
        identity["training_expression_policy"] = expression_policy_receipt
        if train_selection_receipt is not None:
            identity["training_row_selection"] = train_selection_receipt
        if allowed_expression_ids is not None:
            identity["effective_training_expression_ids_sha256"] = sha256_json(
                [data.expression_gene_ids[int(i)] for i in allowed_expression_ids]
            )
        if options.expression_holdout_path:
            identity["expression_holdout_sha256"] = options.expression_holdout_sha256
        runtime = Runtime(
            data,
            options,
            NeighborhoodIndex(
                topology,
                options.graph_expander_degree,
                options.graph_expander_seed,
                expander_type=options.graph_expander_type,
                propagated=arch.graph_read_mode == "propagated",
                relay=arch.graph_read_mode == "relay",
            ),
            objective,
            optimizer,
            np.random.default_rng(run_seed),
            device,
            int(config.training.train_batch_size.value),
            identity,
            allowed_expression_ids,
            purpose,
        )
        if options.response_anchor_weight:
            from .response_anchor_runtime import ResponseAnchorController, load_response_bank

            objective.response_anchor_enabled = True
            identity["response_anchor"] = {
                "manifest_sha256": options.response_anchor_sha256,
                "arrays_sha256": options.response_anchor_arrays_sha256,
                "weight": options.response_anchor_weight,
                "seed": options.response_anchor_seed,
                "head": "Linear256_256_default_random_bias_true_private_rng",
                "teacher_readout": "unused_ema_mirror_no_forward_or_target_values",
                "schedule": "hash_permutation_step_i_mod_T_rank_floor_i_div_T_mod_world",
            }
            if purpose == "training":
                if allowed_expression_ids is None:
                    raise ValueError("response anchor requires the frozen expression holdout")
                bank = load_response_bank(
                    options.response_anchor_path,
                    options.response_anchor_sha256,
                    allowed_expression_ids,
                )
                manifest_digest = (
                    sha256_json(data.manifest.payload())
                    if hasattr(data.manifest, "payload")
                    else sha256_file(data.layout.canonical_manifest)
                )
                if bank.identity.get("data_manifest") != manifest_digest:
                    raise ValueError("response anchor canonical manifest identity differs")
                if bank.arrays_sha256 != options.response_anchor_arrays_sha256:
                    raise ValueError("response anchor array identity differs from configuration")
                controller = ResponseAnchorController(runtime, bank)
                objective.response_anchor_backward = controller.backward
        yield runtime
