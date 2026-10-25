"""Explicit scientific-parent warm start and immutable frozen-feature readout."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from gradpert.config.schema import ExperimentConfig
from gradpert.config.v2 import V2Options
from gradpert.data._io import read_json
from gradpert.execution.v2_training_stage import training_stage_evidence
from gradpert.hashing import sha256_file, sha256_json
from gradpert.modeling.v2.operators import RelayDeltaAttention

from .objective import JointObjective
from .optimizer import V2Optimizer

HEAD_SHAPES = {
    "prediction.0.weight": (256, 512),
    "prediction.0.bias": (256,),
    "prediction.2.weight": (1, 256),
    "prediction.2.bias": (1,),
}


def frozen_digest(objective: JointObjective) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(objective.state_dict().items()):
        if name.startswith("student.prediction."):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prediction_head_digest(objective: JointObjective) -> str:
    """Byte identity for the existing four prediction tensors, independent of mode."""
    digest = hashlib.sha256()
    for name, value in sorted(objective.student.prediction.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenReadoutObjective(JointObjective):
    """Retain native prediction and reductions; disable every representation update."""

    parent_frozen_sha256: str
    parent_prediction_head_sha256: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.lambda1
            or self.lambda2
            or self.au_enabled
            or self.cr_enabled
            or self.population_response
            or self.auxiliary_mask_ratio
            or self.masked_response_ratio
        ):
            raise ValueError("frozen readout supports prediction MSE only")
        if self.student.options.prediction_parameterization != "softplus":
            raise ValueError("frozen readout preserves native nonnegative Softplus")
        if self.student.prior_readout is not None:
            raise ValueError("frozen readout requires the existing prediction MLP alone")
        actual = {
            n: tuple(p.shape)
            for n, p in self.student.named_parameters()
            if n.startswith("prediction.")
        }
        if actual != HEAD_SHAPES:
            raise ValueError("frozen readout prediction architecture differs")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("student.prediction."))
        self.register_buffer("stage_phase", torch.ones((), dtype=torch.long))
        self.teacher_updates_enabled = False
        self.decode_ssl_views = False
        self.train()

    def train(self, mode: bool = True) -> FrozenReadoutObjective:
        super().train(mode)
        self.student.eval()
        self.teacher.eval()
        if self.student.options.attention == "relay_full":
            self.student.set_relay_order_randomization(False)
            self.teacher.set_relay_order_randomization(False)
        self.student.prediction.train(mode)
        return self

    @property
    def training_stage(self) -> str:
        return "frozen_readout"

    def prepare_update(self, optimizer: V2Optimizer) -> None:
        if not hasattr(self, "parent_frozen_sha256"):
            raise ValueError("readout parent must be verified before training")
        if optimizer.training_stage not in (None, "frozen_readout"):
            raise ValueError("readout optimizer stage differs")
        optimizer.training_stage = "frozen_readout"


def frozen_state_proof(objective: JointObjective, optimizer: V2Optimizer) -> dict[str, Any]:
    actual = frozen_digest(objective)
    if actual != getattr(objective, "parent_frozen_sha256", None):
        raise ValueError("frozen parent state changed")
    for prefix, model in (("student", objective.student), ("teacher", objective.teacher)):
        for name, module in model.named_modules():
            if prefix == "student" and (name == "prediction" or name.startswith("prediction.")):
                continue
            if module.training:
                raise ValueError(f"frozen module must remain in eval mode: {prefix}.{name}")
            if isinstance(module, RelayDeltaAttention) and module.random_order_enabled():
                raise ValueError(f"frozen relay order must remain deterministic: {prefix}.{name}")
        if model.randomize_relay_order:
            raise ValueError(f"frozen relay order must remain deterministic: {prefix}")
    head_sha256 = prediction_head_digest(objective)
    parent_head_sha256 = getattr(objective, "parent_prediction_head_sha256", None)
    if parent_head_sha256 is None:
        raise ValueError("readout parent prediction head identity is missing")
    trainable = {n: p for n, p in objective.student.named_parameters() if p.requires_grad}
    if set(trainable) != set(HEAD_SHAPES) or {r["name"] for r in optimizer.routes} != set(
        trainable
    ):
        raise ValueError("optimizer must own only the four existing prediction tensors")
    if any(
        p.grad is not None
        for n, p in objective.named_parameters()
        if not n.startswith("student.prediction.")
    ):
        raise ValueError("frozen parameter acquired a gradient")
    return {
        "frozen_sha256": actual,
        "parent_frozen_sha256": actual,
        "frozen_equal": True,
        "frozen_gradients_none": True,
        "frozen_modules_eval": True,
        "relay_order_randomization": False,
        "parent_prediction_head_sha256": parent_head_sha256,
        "prediction_head_sha256": head_sha256,
        "prediction_head_changed": head_sha256 != parent_head_sha256,
        "teacher_updates": 0,
        "center_updates": 0,
        "inactive_objectives": ["ssl1", "ssl2", "alignment", "uniformity", "coding_rate"],
        "trainable_tensors": [{"name": n, "shape": list(p.shape)} for n, p in trainable.items()],
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "optimizer_routes": optimizer.routes,
    }


def restore_frozen_parent(
    config: ExperimentConfig,
    *,
    data_identity: dict[str, Any],
    objective: JointObjective,
    optimizer: V2Optimizer,
    server_root: Path = Path("/data/yilangliu"),
) -> dict[str, Any]:
    """Load complete parent state only, leaving fresh child moments, counters and RNG."""
    _, options = V2Options.parse_parameters(config.model.parameters)
    if options.training_mode != "frozen_readout" or not isinstance(
        objective, FrozenReadoutObjective
    ):
        raise ValueError("explicit frozen readout mode required")
    root = Path(options.readout_parent_run_root).resolve(strict=True)
    if not root.is_relative_to(server_root.resolve()):
        raise ValueError("parent must stay on the server")
    manifest = read_json(root / "run_manifest.json")
    prior_config = read_json(root / "resolved_config.json")
    journal = read_json(root / "fit/epoch_state.json")
    terminal = read_json(root / "TRAIN_COMPLETE.json")
    if terminal != training_stage_evidence(root, manifest, options.readout_parent_epoch):
        raise ValueError("scientific parent TRAIN_COMPLETE differs")
    source = manifest["source"]
    if (
        manifest["run_id"] != options.readout_parent_run_id
        or source["commit"] != options.readout_parent_source_commit
        or source["dirty"]
        or source["published_commit"] != source["commit"]
        or manifest["resolved_config_sha256"] != sha256_json(prior_config)
        or {k: v for k, v in manifest["data"].items() if k != "loss_protocol"}
        != {k: v for k, v in data_identity.items() if k != "loss_protocol"}
    ):
        raise ValueError("parent source/config/data binding differs")
    parent_options = {n: v["value"] for n, v in prior_config["model"]["parameters"].items()}
    if parent_options.get("training_mode") != "ssl_then_joint":
        raise ValueError("readout parent must be the staged joint scientific run")
    current = config.model_dump(mode="json")
    for section in ("data", "evaluation", "artifacts"):
        if current[section] != prior_config[section]:
            raise ValueError(f"readout parent {section} protocol differs")
    for name in (
        "train_batch_size",
        "eval_batch_size",
        "weight_decay",
        "learning_rate",
        "scheduler",
    ):
        if current["training"][name] != prior_config["training"][name]:
            raise ValueError(f"readout parent training {name} differs")
    for name in (
        "query_count",
        "eval_query_count",
        "microbatch",
        "world_size",
        "accumulation",
        "max_conditions",
        "graph_view_mode",
        "cell_view_mode",
        "train_selection_path",
        "train_selection_sha256",
        "expression_holdout_path",
        "expression_holdout_sha256",
        "gene_initialization",
    ):
        if name in parent_options and getattr(options, name) != parent_options[name]:
            raise ValueError(f"readout parent sampling {name} differs")
    if int(config.training.max_epochs.value) != 2:
        raise ValueError("readout child requires exactly two fresh epochs")
    checkpoint = (root / "fit" / journal["last"]["file"]).resolve(strict=True)
    if (
        not checkpoint.is_relative_to((root / "fit").resolve())
        or journal["last"]["epoch"] != options.readout_parent_epoch
        or journal["last"]["sha256"] != options.readout_parent_checkpoint_sha256
        or sha256_file(checkpoint) != options.readout_parent_checkpoint_sha256
    ):
        raise ValueError("parent last checkpoint differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "gradpert-v2-checkpoint-1"
        or payload.get("model_version") != "v2"
        or payload["identity"] != manifest
        or payload["architecture"] != objective.student.options.payload()
        or payload["progress"].get("history") != read_json(root / "fit/history.json")
        or journal["budget"] != [3, 150]
        or payload["optimizer"].get("steps") != 450
        or payload["optimizer"].get("training_stage") != "joint"
        or payload["optimizer"].get("stage_reset_count") != 0
        or payload["objective"].get("stage_phase") is None
        or int(payload["objective"]["stage_phase"].item()) != 1
    ):
        raise ValueError("parent checkpoint identity/architecture/staged phase differs")
    if optimizer.steps or any(
        opt is not None and opt.state for opt in (optimizer.muon, optimizer.adamw)
    ):
        raise ValueError("parent warm start requires fresh child optimizer")
    objective.load_state_dict(payload["objective"], strict=True)
    objective.pending.clear()
    objective.parent_frozen_sha256 = frozen_digest(objective)
    objective.parent_prediction_head_sha256 = prediction_head_digest(objective)
    proof = frozen_state_proof(objective, optimizer)
    return {
        "mode": "frozen_readout",
        "parent_run_id": manifest["run_id"],
        "parent_training_source": source,
        "parent_epoch": options.readout_parent_epoch,
        "parent_run_root": str(root),
        "parent_checkpoint": str(checkpoint),
        "parent_config_sha256": manifest["config_sha256"],
        "parent_resolved_config_sha256": manifest["resolved_config_sha256"],
        "parent_checkpoint_sha256": options.readout_parent_checkpoint_sha256,
        "parent_scientific_training_complete": True,
        "parent_full_test_complete": False,
        "optimizer_restored": False,
        "child_initial_steps": 0,
        "parent_evidence_sha256": {
            n: sha256_file(root / n)
            for n in (
                "TRAIN_COMPLETE.json",
                "run_manifest.json",
                "resolved_config.json",
                "fit/epoch_state.json",
            )
        },
        **proof,
    }
