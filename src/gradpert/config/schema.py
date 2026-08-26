"""Typed, self-contained experiment configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DatasetId = Literal[
    "replogle_k562_essential",
    "replogle_rpe1_essential",
    "nadig_jurkat",
    "nadig_hepg2",
    "norman",
]
ModelId = Literal[
    "gradpert_b2",
    "gears",
    "txpert_public",
    "matched_control_mean",
    "global_train_delta",
    "general_train_delta",
]
ProvenanceKind = Literal[
    "official",
    "paper",
    "project_preregistered",
    "server_fit",
    "user_locked",
]
HeadlineMetricId = Literal[
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
]
Scalar = str | int | float | bool


class StrictModel(BaseModel):
    """Base schema that rejects hidden or misspelled fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourcedValue(StrictModel):
    """A scalar experiment value with explicit provenance."""

    value: Scalar
    source: ProvenanceKind
    reference: str = Field(min_length=1)


class SourceCode(StrictModel):
    repository: str
    commit: str
    license_boundary: str
    execution: Literal["native", "isolated", "none"]


class DataConfig(StrictModel):
    dataset_id: DatasetId
    protocol_id: str
    registry_version: str
    canonical_data_root: str
    preprocessing_id: str
    split_policy: str
    split_seed: int
    expected_split_hash: Literal["from_canonical_manifest"]
    expected_expression_gene_order_hash: Literal["from_canonical_manifest"]
    expected_graph_gene_order_hash: Literal["from_canonical_manifest"]
    expected_control_manifest_hash: Literal["from_canonical_manifest"]


class ModelConfig(StrictModel):
    model_id: ModelId
    family: Literal["native_learned", "external_learned", "nonlearned"]
    implementation: str
    parameters: dict[str, SourcedValue]

    @model_validator(mode="after")
    def require_parameters(self) -> ModelConfig:
        if not self.parameters:
            raise ValueError("model.parameters must be explicit and non-empty")
        return self


class TrainingConfig(StrictModel):
    learned: bool
    smoke_epochs: SourcedValue
    formal_run_policy: Literal[
        "smoke_then_full", "smoke_only", "fixed_epoch_pilot", "inference_only"
    ]
    max_epochs: SourcedValue
    early_stopping: bool
    early_stopping_patience: SourcedValue
    monitor: str
    monitor_mode: Literal["max", "none"]
    min_delta: float
    train_batch_size: SourcedValue
    eval_batch_size: SourcedValue
    optimizer: SourcedValue
    learning_rate: SourcedValue
    weight_decay: SourcedValue
    scheduler: SourcedValue
    run_seeds: list[int]

    @model_validator(mode="after")
    def enforce_budget(self) -> TrainingConfig:
        if self.learned:
            if self.smoke_epochs.value != 1:
                raise ValueError("learned models require a one-epoch integration smoke")
            if self.formal_run_policy == "smoke_then_full":
                if self.max_epochs.value not in {100, 200}:
                    raise ValueError(
                        "full native runs require max_epochs=200; sealed legacy pilot "
                        "configs may retain max_epochs=100"
                    )
                if not self.early_stopping or self.early_stopping_patience.value != 10:
                    raise ValueError("full native runs require early-stopping patience=10")
                if self.monitor != "val/txpert_macro_pearson_delta" or self.monitor_mode != "max":
                    raise ValueError("full native runs require the common validation monitor")
                if self.run_seeds != [1, 2, 3, 4]:
                    raise ValueError("full native runs require run seeds 1,2,3,4")
            elif self.formal_run_policy == "smoke_only":
                if self.max_epochs.value != 1:
                    raise ValueError("external smoke-only runs require max_epochs=1")
                if self.early_stopping or self.early_stopping_patience.value != 0:
                    raise ValueError(
                        "one-epoch external smoke does not use orchestrator early stopping"
                    )
                if self.monitor != "none" or self.monitor_mode != "none":
                    raise ValueError("one-epoch external smoke uses monitor=none")
                if self.run_seeds != [1]:
                    raise ValueError("external smoke-only runs use the shared seed 1")
            elif self.formal_run_policy == "fixed_epoch_pilot":
                if self.max_epochs.value != 10:
                    raise ValueError("fixed-epoch native pilots require max_epochs=10")
                if self.early_stopping or self.early_stopping_patience.value != 10:
                    raise ValueError(
                        "fixed-epoch native pilots disable stopping but retain patience=10 state"
                    )
                if self.monitor != "val/txpert_macro_pearson_delta" or self.monitor_mode != "max":
                    raise ValueError("fixed-epoch native pilots require the validation monitor")
                if self.run_seeds != [1]:
                    raise ValueError("fixed-epoch native pilots use only seed 1")
            else:
                raise ValueError("learned models require a learned formal_run_policy")
            if self.min_delta != 0.0:
                raise ValueError("learned models require min_delta=0")
        else:
            if self.smoke_epochs.value != 0 or self.max_epochs.value != 0:
                raise ValueError("nonlearned models must not declare fitting epochs")
            if self.formal_run_policy != "inference_only" or self.early_stopping:
                raise ValueError("nonlearned models must not declare training epochs")
            if self.monitor_mode != "none" or self.run_seeds != [1]:
                raise ValueError("nonlearned models use monitor_mode=none and shared seed 1")
        return self


class EvaluationConfig(StrictModel):
    evaluator_version: str
    prediction_schema_version: str
    bundle_schema_version: str
    evaluation_seed: int
    n_controls_per_condition: int
    sample_controls_with_replacement: bool
    control_context_policy: Literal["truth_cell_context_resampling"]
    truth_access: Literal["evaluator_only"]
    headline_metrics: list[HeadlineMetricId]

    @model_validator(mode="after")
    def enforce_common_protocol(self) -> EvaluationConfig:
        if self.evaluation_seed != 20260824:
            raise ValueError("evaluation_seed must be frozen at 20260824")
        if self.n_controls_per_condition != 300 or not self.sample_controls_with_replacement:
            raise ValueError("evaluation requires 300 controls sampled with replacement")
        expected = [
            "txpert_macro_pearson_delta",
            "trishift_pearson_delta",
            "systema_pearson",
        ]
        if self.headline_metrics != expected:
            raise ValueError("headline metric IDs/order are frozen")
        return self


class ArtifactConfig(StrictModel):
    root: str
    large_artifacts_server_only: bool
    result_mode: Literal["metrics_only", "single_pkl"]
    result_pkl_name: Literal["result.pkl"]
    inference_recipe_schema_version: Literal["inference-recipe-v1"]
    small_sync_extensions: list[str]
    sync_requires_dry_run: bool

    @model_validator(mode="after")
    def enforce_server_only(self) -> ArtifactConfig:
        if not self.large_artifacts_server_only or not self.sync_requires_dry_run:
            raise ValueError("large artifacts must be server-only and sync must dry-run")
        allowed = [".txt", ".json", ".jsonl", ".csv", ".md"]
        if self.small_sync_extensions != allowed:
            raise ValueError(f"small_sync_extensions must equal {allowed}")
        return self


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    experiment_id: str
    model_id: ModelId
    dataset_id: DatasetId
    source_code: SourceCode
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactConfig

    @model_validator(mode="after")
    def enforce_identity(self) -> ExperimentConfig:
        if self.model_id != self.model.model_id:
            raise ValueError("top-level and nested model_id differ")
        if self.dataset_id != self.data.dataset_id:
            raise ValueError("top-level and nested dataset_id differ")
        expected_id = f"{self.model_id}__{self.dataset_id}"
        if self.experiment_id != expected_id:
            raise ValueError(f"experiment_id must equal {expected_id}")
        expected_learned = self.model.family != "nonlearned"
        if self.training.learned != expected_learned:
            raise ValueError("training.learned does not match model family")
        allowed_policies = {
            "native_learned": {"smoke_then_full", "fixed_epoch_pilot"},
            "external_learned": {"smoke_only"},
            "nonlearned": {"inference_only"},
        }[self.model.family]
        if self.training.formal_run_policy not in allowed_policies:
            expected = ",".join(sorted(allowed_policies))
            raise ValueError(f"{self.model.family} requires formal_run_policy in {{{expected}}}")
        is_legacy_performance_pilot = "performance_pilot_variant" in self.model.parameters
        if (
            self.model_id == "gradpert_b2"
            and self.training.formal_run_policy == "smoke_then_full"
            and not is_legacy_performance_pilot
        ):
            if self.training.max_epochs.value != 200:
                raise ValueError("formal gradpert_b2 requires max_epochs=200")
            required_b2_parameters: dict[str, Scalar] = {
                "graph_axis_policy": "canonical_full",
                "systems_optimizations": "all_seven_semantics_preserving_v1",
                "systems_merged_hdf5_reads": True,
                "systems_control_expression_cache": True,
                "systems_background_prefetch": True,
                "systems_pin_memory": True,
                "systems_nonblocking_transfer": True,
                "systems_prefetch_depth": 2,
                "systems_resident_graph_tensors": True,
                "systems_validation_expression_cache": True,
                "systems_buffered_training_logs": True,
                "systems_log_buffer_steps": 64,
                "systems_single_checkpoint_serialization": True,
            }
            observed = {
                name: self.model.parameters[name].value
                for name in required_b2_parameters
                if name in self.model.parameters
            }
            if observed != required_b2_parameters:
                raise ValueError(
                    "formal gradpert_b2 requires the canonical full graph and all seven "
                    "semantics-preserving systems optimizations"
                )
        external_contracts = {
            "gears": (
                "https://github.com/snap-stanford/GEARS.git",
                "f374e43e197b295016d80395d7a54ddb81cc6769",
                "benchmarks.gears.runner",
            ),
            "txpert_public": (
                "https://github.com/valence-labs/TxPert.git",
                "08d82eea86746b044cf7531f4ec8c5f60e1cb73f",
                "benchmarks.txpert.runner",
            ),
        }
        if self.model_id in external_contracts:
            repository, commit, implementation = external_contracts[self.model_id]
            if (
                self.source_code.repository != repository
                or self.source_code.commit != commit
                or self.source_code.execution != "isolated"
                or self.model.implementation != implementation
            ):
                raise ValueError(
                    f"{self.model_id} must invoke the frozen official package through its "
                    "isolated benchmark runner"
                )
        elif self.source_code.execution == "isolated":
            raise ValueError("only external learned benchmarks use isolated execution")
        return self
