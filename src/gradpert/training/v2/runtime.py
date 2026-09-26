"""Canonical server data and model construction for the independent v2 runner."""

from __future__ import annotations

import random
from collections.abc import Iterator
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
        self, epoch: int, device: torch.device, generator: np.random.Generator
    ) -> Iterator[TrainingBatch]:
        if self.purpose != "training":
            raise RuntimeError("evaluation runtime cannot enter training")
        for raw in self.data.iter_train_epoch(
            epoch=epoch,
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
                self.options,
                generator,
                allowed_expression_ids=self.allowed_expression_ids,
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
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)
        seed = torch.from_numpy(prior.values.copy())
        if options.gene_initialization == "random":
            seed = torch.randn_like(seed) * 0.02
        student = GraDPertV2(seed, arch).to(device)
        objective = JointObjective(
            student,
            lambda1=options.lambda1,
            lambda2=options.lambda2,
            ssl1_weights=(options.ssl1_condition, options.ssl1_node, options.ssl1_spread),
            ssl2_weights=(options.ssl2_dino, options.ssl2_ibot, options.ssl2_koleo),
            loss_reduction=options.loss_reduction,
            ssl1_reduction=options.ssl1_reduction,
            prediction_reduction=options.prediction_loss,
            koleo_exclude_same_condition=options.koleo_exclude_same_condition,
        ).to(device)
        optimizer = V2Optimizer(
            student,
            lr=float(config.training.learning_rate.value),
            weight_decay=float(config.training.weight_decay.value),
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
        identity["training_expression_policy"] = expression_policy_receipt
        if allowed_expression_ids is not None:
            identity["effective_training_expression_ids_sha256"] = sha256_json(
                [data.expression_gene_ids[int(i)] for i in allowed_expression_ids]
            )
        if options.expression_holdout_path:
            identity["expression_holdout_sha256"] = options.expression_holdout_sha256
        yield Runtime(
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
