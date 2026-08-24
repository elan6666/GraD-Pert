"""End-to-end nonlearned baseline prediction and common evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gradpert.artifacts import PredictionConditionArrays
from gradpert.baselines import (
    GeneralTrainDeltaBaseline,
    GlobalTrainDeltaBaseline,
    MatchedControlBaseline,
    fit_training_deltas,
)
from gradpert.config import load_experiment_config
from gradpert.contracts import RunManifest
from gradpert.data._io import atomic_json, atomic_text
from gradpert.evaluation import CanonicalEvaluationData
from gradpert.execution.artifact_run import seal_evaluated_run
from gradpert.execution.identity import (
    EnvironmentIdentity,
    SourceIdentity,
    inspect_environment,
    inspect_source_identity,
)
from gradpert.hashing import sha256_file
from gradpert.training.data import CanonicalTrainingData, write_training_data_receipt


@dataclass(frozen=True)
class NonlearnedRunResult:
    run_id: str
    run_root: Path
    run_manifest: RunManifest
    source: SourceIdentity
    environment: EnvironmentIdentity


def run_nonlearned_experiment(
    *,
    config_path: str | Path,
    data_root: str | Path,
    run_root: str | Path,
    run_id: str,
    repository_root: str | Path,
    formal: bool,
    development_commit: str | None = None,
) -> NonlearnedRunResult:
    """Fit from canonical train rows and evaluate once on frozen test controls."""

    config_file = Path(config_path).resolve(strict=True)
    config = load_experiment_config(config_file)
    if config.model.family != "nonlearned" or config.training.run_seeds != [1]:
        raise ValueError(
            "nonlearned runner requires one inference-only baseline config with shared seed 1"
        )
    run_seed = config.training.run_seeds[0]
    destination = Path(run_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("run root must be new and empty")
    destination.mkdir(parents=True, exist_ok=True)
    source = inspect_source_identity(
        repository_root,
        formal=formal,
        expected_repository=config.source_code.repository,
        development_commit=development_commit,
    )
    environment = inspect_environment(
        repository_root,
        device_name="cpu",
        require_cuda=False,
    )
    config_sha256 = sha256_file(config_file)
    small_root = destination / "small_results"
    atomic_text(small_root / "config.resolved.yaml", config_file.read_text(encoding="utf-8"))
    atomic_json(small_root / "source_identity.json", source.payload())
    atomic_json(small_root / "environment.json", environment.payload())

    with (
        CanonicalTrainingData(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            data_root=data_root,
            run_seed=run_seed,
        ) as training_data,
        CanonicalEvaluationData(
            dataset_id=config.dataset_id,
            protocol_id=config.data.protocol_id,
            split_name="test",
            data_root=data_root,
        ) as test_data,
    ):
        training_data.require_experiment_data_contract(
            registry_version=config.data.registry_version,
            split_policy=config.data.split_policy,
        )
        write_training_data_receipt(training_data, small_root / "training_data.json")
        delta_baseline: GlobalTrainDeltaBaseline | GeneralTrainDeltaBaseline | None = None
        if config.model_id == "matched_control_mean":
            fit_receipt = {"strategy": "return_input_control_rows", "reads_train": False}
        else:
            fit = training_data.load_baseline_fit_data()
            registry = fit_training_deltas(
                train_perturbed_expression=fit.perturbed_expression,
                train_condition_ids=fit.condition_ids,
                train_context_ids=fit.context_ids,
                train_batch_ids=fit.batch_ids,
                train_control_expression=fit.control_expression,
                control_context_ids=fit.control_context_ids,
                control_batch_ids=fit.control_batch_ids,
            )
            delta_baseline = (
                GlobalTrainDeltaBaseline(registry)
                if config.model_id == "global_train_delta"
                else GeneralTrainDeltaBaseline(registry)
            )
            fit_receipt = {
                "strategy": config.model_id,
                "reads_train": True,
                "train_perturbed_rows": int(fit.perturbed_expression.shape[0]),
                "train_control_rows": int(fit.control_expression.shape[0]),
                "training_condition_ids": list(registry.training_condition_ids),
            }
        atomic_json(
            small_root / "baseline_fit.json",
            {"schema_version": "nonlearned-fit-v1", **fit_receipt},
        )
        predictions: list[PredictionConditionArrays] = []
        for draw in test_data.control_manifest.draws:
            loaded = test_data.load_control_rows(tuple(draw.ordered_row_ids))
            if delta_baseline is None:
                predicted = MatchedControlBaseline.predict(loaded.expression)
            else:
                predicted = delta_baseline.predict(
                    condition_id=draw.condition_id,
                    input_controls=loaded.expression,
                )
            predictions.append(
                PredictionConditionArrays(
                    condition_id=draw.condition_id,
                    prediction=predicted,
                    input_control=loaded.expression,
                    input_control_row_ids=loaded.ordered_row_ids,
                )
            )
        sealed = seal_evaluated_run(
            destination=destination,
            config=config,
            config_sha256=config_sha256,
            run_id=run_id,
            run_seed=run_seed,
            source=source,
            environment=environment,
            training_data=training_data,
            test_data=test_data,
            predictions=predictions,
            checkpoint_sha256=None,
        )
    return NonlearnedRunResult(
        run_id=run_id,
        run_root=destination,
        run_manifest=sealed.run_manifest,
        source=source,
        environment=environment,
    )
