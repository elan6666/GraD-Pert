"""Canonical server data and model construction for the independent v2 runner."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradpert.config.schema import ExperimentConfig
from gradpert.config.v2 import V2Options
from gradpert.features.text_prior import verify_text_prior_npz
from gradpert.graphs.materialization import DatasetGraphLayout, load_dataset_graph_topology
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling.v2 import GraDPertV2
from gradpert.training.data import CanonicalTrainingData

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

    @cached_property
    def steps_per_epoch(self) -> int:
        return self.data.steps_per_epoch(
            batch_size=self.batch_size,
            max_unique_conditions=min(self.options.max_conditions, self.batch_size),
        )

    def batches(self, epoch: int) -> Iterator[TrainingBatch]:
        for raw in self.data.iter_train_epoch(
            epoch=epoch,
            device=self.device,
            batch_size=self.batch_size,
            max_unique_conditions=min(self.options.max_conditions, self.batch_size),
        ):
            yield assemble_batch(raw, self.index, self.options, self.generator)


@contextmanager
def prepare_runtime(
    config: ExperimentConfig, *, data_root: Path, run_seed: int, device: torch.device
) -> Iterator[Runtime]:
    """Verify sealed inputs before model allocation; close backed H5AD on all exits.

    Source publication, allocator and compute admission are execution-level gates.
    This constructor is shared by full training and the capacity runner.
    """
    if config.model_id != "gradpert_v2" or config.model.version != "v2":
        raise ValueError("v2 runtime cannot execute a v1 configuration")
    arch, options = V2Options.parse_parameters(config.model.parameters)
    if options.world_size != 1:
        raise ValueError("distributed v2 execution is not yet verified")
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
            ssl1_reduction=options.ssl1_reduction,
            prediction_reduction=options.prediction_loss,
        ).to(device)
        optimizer = V2Optimizer(
            student,
            lr=float(config.training.learning_rate.value),
            weight_decay=float(config.training.weight_decay.value),
        )
        identity = {
            "canonical_sha256": data.manifest.canonical_adata_sha256,
            "split_sha256": data.split.split_content_sha256,
            "expression_gene_order_sha256": data.manifest.expression_gene_order_sha256,
            "graph_gene_order_sha256": sha256_json(list(topology.gene_ids)),
            "graph_manifest_sha256": options.graph_manifest_sha256,
            "genept_sha256": prior.source_sha256,
            "run_seed": run_seed,
        }
        yield Runtime(
            data,
            options,
            NeighborhoodIndex(topology, options.graph_expander_degree, options.graph_expander_seed),
            objective,
            optimizer,
            np.random.default_rng(run_seed),
            device,
            int(config.training.train_batch_size.value),
            identity,
        )
