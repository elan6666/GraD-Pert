"""Server-only sustained-step fit gate for the global prototype-head width."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path

from gradpert.data import load_dataset_registry
from gradpert.data._io import atomic_json
from gradpert.data.registry import DATASET_IDS

PROTOTYPE_CANDIDATES = (65536, 32768, 16384, 8192)
CAPACITY_PROBE_STEPS = 128


@dataclass(frozen=True)
class DatasetCapacityObservation:
    dataset_id: str
    protocol_id: str
    graph_gene_count: int
    expression_gene_count: int
    train_cell_count: int
    steps_per_epoch: int
    requested_probe_steps: int
    observed_probe_steps: int
    maximum_unique_conditions: int
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    peak_reserved_step: int | None
    reserved_bytes_by_step: tuple[int, ...]
    accepted: bool
    failure: str | None


@dataclass(frozen=True)
class PrototypeCandidateObservation:
    prototype_count: int
    accepted: bool
    datasets: tuple[DatasetCapacityObservation, ...]


def fit_global_prototype_head(
    *,
    data_root: str | Path,
    dataset_registry_root: str | Path,
    output_path: str | Path,
    device_name: str = "cuda:0",
    run_seed: int = 1,
    batch_size: int = 256,
    max_unique_conditions: int = 8,
    usable_memory_fraction: float = 0.85,
    probe_steps: int = CAPACITY_PROBE_STEPS,
) -> dict[str, object]:
    """Choose the largest candidate passing sustained real steps on all datasets."""

    import torch

    from gradpert.graphs import load_dataset_graph_topology
    from gradpert.modeling import CenterState, GraDPertJointModel
    from gradpert.training.data import CanonicalTrainingData
    from gradpert.training.step import GraDPertStepEngine, build_native_optimizer

    if not torch.cuda.is_available() or not device_name.startswith("cuda:"):
        raise RuntimeError("prototype-head fit requires an explicit CUDA device")
    if (batch_size, max_unique_conditions) != (256, 8):
        raise ValueError("v1 prototype fit requires batch 256 and condition cap 8")
    if usable_memory_fraction != 0.85:
        raise ValueError("v1 prototype fit threshold is frozen at 85% of usable memory")
    if probe_steps != CAPACITY_PROBE_STEPS:
        raise ValueError(f"v1 prototype fit requires exactly {CAPACITY_PROBE_STEPS} steps")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    threshold_bytes = int(free_bytes * usable_memory_fraction)
    properties = torch.cuda.get_device_properties(device)

    candidates: list[PrototypeCandidateObservation] = []
    selected: int | None = None
    for prototype_count in PROTOTYPE_CANDIDATES:
        dataset_observations: list[DatasetCapacityObservation] = []
        candidate_accepted = True
        for dataset_id in DATASET_IDS:
            entry = load_dataset_registry(Path(dataset_registry_root) / f"{dataset_id}.yaml")
            peak_allocated: int | None = None
            peak_reserved: int | None = None
            failure: str | None = None
            accepted = False
            graph_gene_count = 0
            expression_gene_count = 0
            train_cell_count = 0
            steps_per_epoch = 0
            observed_probe_steps = 0
            maximum_unique_conditions = 0
            peak_reserved_step: int | None = None
            reserved_bytes_by_step: list[int] = []
            data: CanonicalTrainingData | None = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                data = CanonicalTrainingData(
                    dataset_id=entry.dataset_id,
                    protocol_id=entry.protocol_id,
                    data_root=data_root,
                    run_seed=run_seed,
                )
                topology = load_dataset_graph_topology(
                    dataset_id=entry.dataset_id,
                    protocol_id=entry.protocol_id,
                    data_root=data_root,
                )
                graph_gene_count = data.manifest.n_graph_genes
                expression_gene_count = data.manifest.n_expression_genes
                train_cell_count = len(data.train_row_indices)
                steps_per_epoch = data.steps_per_epoch(
                    batch_size=batch_size,
                    max_unique_conditions=max_unique_conditions,
                )
                model = GraDPertJointModel(
                    graph_gene_count=graph_gene_count,
                    expression_gene_count=expression_gene_count,
                    prototype_count=prototype_count,
                ).to(device)
                optimizer = build_native_optimizer(model)
                centers = CenterState.zeros(
                    prototype_count=prototype_count,
                    device=device,
                )
                heldout_ids = tuple(
                    sorted(
                        {
                            anchor
                            for condition in (
                                *data.split.val_conditions,
                                *data.split.test_conditions,
                            )
                            for anchor in data.anchors_by_condition[condition]
                        }
                    )
                )
                engine = GraDPertStepEngine(
                    model=model,
                    topology=topology,
                    optimizer=optimizer,
                    centers=centers,
                    run_seed=run_seed,
                    total_schedule_steps=100 * steps_per_epoch,
                    heldout_target_ids=heldout_ids,
                )
                expected_probe_steps = min(probe_steps, steps_per_epoch)
                for step_index, batch in enumerate(
                    data.iter_train_epoch(
                        epoch=0,
                        device=device,
                        batch_size=batch_size,
                        max_unique_conditions=max_unique_conditions,
                    )
                ):
                    if step_index >= expected_probe_steps:
                        break
                    maximum_unique_conditions = max(
                        maximum_unique_conditions,
                        len(set(batch.condition_ids)),
                    )
                    metrics = engine.train_step(batch, global_step=step_index)
                    torch.cuda.synchronize(device)
                    observed_probe_steps += 1
                    current_peak_reserved = int(torch.cuda.max_memory_reserved(device))
                    reserved_bytes_by_step.append(current_peak_reserved)
                    if peak_reserved_step is None or current_peak_reserved > max(
                        reserved_bytes_by_step[:-1], default=-1
                    ):
                        peak_reserved_step = step_index
                    del metrics, batch
                    if current_peak_reserved > threshold_bytes:
                        failure = "peak_reserved_exceeds_85_percent_of_initial_free_memory"
                        break
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                accepted = (
                    observed_probe_steps == expected_probe_steps
                    and peak_reserved <= threshold_bytes
                )
                if not accepted and failure is None:
                    failure = "capacity_probe_ended_before_required_steps"
                del engine, centers, optimizer, model, topology
            except torch.cuda.OutOfMemoryError:
                failure = "cuda_out_of_memory"
                accepted = False
            finally:
                if data is not None:
                    data.close()
                gc.collect()
                torch.cuda.empty_cache()
            candidate_accepted = candidate_accepted and accepted
            dataset_observations.append(
                DatasetCapacityObservation(
                    dataset_id=entry.dataset_id,
                    protocol_id=entry.protocol_id,
                    graph_gene_count=graph_gene_count,
                    expression_gene_count=expression_gene_count,
                    train_cell_count=train_cell_count,
                    steps_per_epoch=steps_per_epoch,
                    requested_probe_steps=probe_steps,
                    observed_probe_steps=observed_probe_steps,
                    maximum_unique_conditions=maximum_unique_conditions,
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                    peak_reserved_step=peak_reserved_step,
                    reserved_bytes_by_step=tuple(reserved_bytes_by_step),
                    accepted=accepted,
                    failure=failure,
                )
            )
            if not accepted:
                break
        observation = PrototypeCandidateObservation(
            prototype_count=prototype_count,
            accepted=candidate_accepted,
            datasets=tuple(dataset_observations),
        )
        candidates.append(observation)
        if candidate_accepted:
            selected = prototype_count
            break
    if selected is None:
        raise RuntimeError("no frozen prototype-head candidate passed the server fit gate")
    receipt: dict[str, object] = {
        "schema_version": "prototype-head-capacity-v2",
        "status": "development_capacity_passed",
        "formal_eligible": False,
        "formal_eligibility_reason": "requires_identical_clean_local_github_server_commit",
        "device": device_name,
        "device_name": properties.name,
        "device_total_bytes": int(total_bytes),
        "initial_free_bytes": int(free_bytes),
        "usable_memory_fraction": usable_memory_fraction,
        "acceptance_threshold_bytes": threshold_bytes,
        "batch_size": batch_size,
        "max_unique_conditions_per_batch": max_unique_conditions,
        "capacity_probe_steps": probe_steps,
        "run_seed": run_seed,
        "selected_prototype_count": selected,
        "candidates": [
            {
                "prototype_count": item.prototype_count,
                "accepted": item.accepted,
                "datasets": [asdict(dataset) for dataset in item.datasets],
            }
            for item in candidates
        ],
    }
    atomic_json(output_path, receipt)
    return receipt
