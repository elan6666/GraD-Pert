from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.schema import ExperimentConfig
from gradpert.config.v2 import V2Options

PROBE = Path(__file__).resolve().parents[2] / "configs/v2/capacity/gradpert_v2/nadig_jurkat.yaml"
GLM53 = Path(__file__).resolve().parents[2] / "configs/v2/glm53_flash_jurkat"
ONE_EPOCH = (
    Path(__file__).resolve().parents[2]
    / "configs/v2/optimized_single_pass_jurkat/one_epoch_m66_a2/gradpert_v2/nadig_jurkat.yaml"
)
CHUNK_SWEEP = Path(__file__).resolve().parents[2] / "configs/v2/chunk_sweep_jurkat"
SOURCE_GATE_CHUNK = Path(__file__).resolve().parents[2] / "configs/v2/source_key_gate_chunk_jurkat"
SOURCE_GATE_DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "configs/v2/source_key_gate_jurkat/one_epoch_m66_a2/gradpert_v2/nadig_jurkat.yaml"
)
SOURCE_GATE_CAPACITY = (
    Path(__file__).resolve().parents[2] / "configs/v2/source_key_gate_capacity_jurkat"
)


@pytest.mark.parametrize("variant,topk,chunk", [("default", 500, 32), ("top100", 100, 8)])
def test_glm53_configs_keep_b1_seed_and_explicit_sparse_settings(variant, topk, chunk):
    path = GLM53 / variant / "gradpert_v2/nadig_jurkat.yaml"
    config = load_experiment_config(path)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert (arch.attention, arch.ffn_type, arch.sparse_topk) == ("hybrid_sparse", "swiglu", topk)
    assert arch.sparse_index_dim == 64 and arch.sparse_query_chunk == chunk
    assert (arch.kda_layers, arch.prototypes) == ((2, 8192) if variant == "default" else (3, 16384))
    assert options.genept_artifact_path.endswith("v2-genept-pca256-b4e3a08.npz")
    assert (options.microbatch, options.accumulation, options.world_size) == (32, 2, 2)
    assert config.training.train_batch_size.value == 128
    assert config.training.max_epochs.value == 5


@pytest.mark.parametrize("microbatch", [36, 38, 40, 48, 56, 64])
def test_glm53_capacity_profiles_only_change_physical_and_global_batch(microbatch):
    baseline = yaml.safe_load((GLM53 / "capacity_m36_a2/gradpert_v2/nadig_jurkat.yaml").read_text())
    baseline["model"]["parameters"]["microbatch"]["value"] = 32
    baseline["training"]["train_batch_size"]["value"] = 128
    path = GLM53 / f"capacity_m{microbatch}_a2/gradpert_v2/nadig_jurkat.yaml"
    profile = yaml.safe_load(path.read_text())
    assert load_experiment_config(path).training.train_batch_size.value == 4 * microbatch
    assert profile["model"]["parameters"]["microbatch"]["value"] == microbatch
    profile["model"]["parameters"]["microbatch"]["value"] = 32
    profile["training"]["train_batch_size"]["value"] = 128
    assert profile == baseline


def test_no_outer_checkpoint_profile_is_a_compute_only_candidate():
    baseline = yaml.safe_load((GLM53 / "capacity_m36_a2/gradpert_v2/nadig_jurkat.yaml").read_text())
    baseline["model"]["parameters"]["microbatch"]["value"] = 32
    baseline["training"]["train_batch_size"]["value"] = 128
    path = GLM53 / "performance_no_outer_checkpoint/gradpert_v2/nadig_jurkat.yaml"
    candidate = yaml.safe_load(path.read_text())
    assert load_experiment_config(path).training.train_batch_size.value == 128
    assert candidate["model"]["parameters"]["checkpoint_layers"]["value"] is False
    candidate["model"]["parameters"]["checkpoint_layers"]["value"] = True
    assert candidate == baseline


@pytest.mark.parametrize("microbatch", [32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72])
def test_compact_capacity_profiles_only_change_batch(microbatch):
    baseline = yaml.safe_load((GLM53 / "default/gradpert_v2/nadig_jurkat.yaml").read_text())
    path = GLM53 / f"compact_m{microbatch}_a2/gradpert_v2/nadig_jurkat.yaml"
    profile = yaml.safe_load(path.read_text())
    assert load_experiment_config(path).training.train_batch_size.value == 4 * microbatch
    profile["model"]["parameters"]["microbatch"]["value"] = 32
    profile["training"]["train_batch_size"]["value"] = 128
    assert profile == baseline


def test_capacity_probe_keeps_complete_method_and_fixed_protocol():
    config = load_experiment_config(PROBE)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert config.model.version == "v2"
    assert arch.width == 256 and arch.prototypes == 16384
    assert (options.lambda1, options.lambda2) == (1, 0.1)
    assert config.training.max_epochs.value == 50
    assert options.local_views == 4
    assert config.training.train_batch_size.value == options.microbatch * options.accumulation


def test_v2_five_epoch_protocol_is_explicit_and_does_not_reinterpret_old_runs():
    payload = load_experiment_config(PROBE).model_dump(mode="json")
    payload["training"]["formal_run_policy"] = "v2_fixed_5"
    payload["training"]["max_epochs"]["value"] = 5
    five = ExperimentConfig.model_validate(payload)
    assert five.training.max_epochs.value == 5
    payload["training"]["max_epochs"]["value"] = 50
    with pytest.raises(ValueError, match="exactly 5 epochs"):
        ExperimentConfig.model_validate(payload)


def test_v2_one_epoch_is_separate_policy_and_keeps_the_current_method():
    config = load_experiment_config(ONE_EPOCH)
    assert config.training.formal_run_policy == "v2_fixed_1"
    assert config.training.max_epochs.value == 1
    assert config.training.train_batch_size.value == 264
    assert config.training.monitor == "val/prediction_loss"
    assert config.training.early_stopping is False
    _, options = V2Options.parse_parameters(config.model.parameters)
    assert options.lambda2 == 1.0
    assert (
        options.lambda2 * options.ssl2_dino,
        options.lambda2 * options.ssl2_ibot,
        options.lambda2 * options.ssl2_koleo,
    ) == (0.8, 0.4, 0.1)
    previous = yaml.safe_load(
        (ONE_EPOCH.parents[2] / "capacity_m66_a2/gradpert_v2/nadig_jurkat.yaml").read_text()
    )
    updated = yaml.safe_load(ONE_EPOCH.read_text())
    previous["training"]["formal_run_policy"] = "v2_fixed_1"
    previous["training"]["max_epochs"] = updated["training"]["max_epochs"]
    previous["model"]["parameters"]["lambda2"] = updated["model"]["parameters"]["lambda2"]
    assert updated == previous
    payload = config.model_dump(mode="json")
    payload["training"]["max_epochs"]["value"] = 5
    with pytest.raises(ValueError, match="exactly 1 epoch"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    "variant,field,value",
    [
        ("scan48_m32_a2", "relay_scan_chunk_size", 48),
        ("scan64_m32_a2", "relay_scan_chunk_size", 64),
        ("rows96_m32_a2", "relay_graph_chunk_rows", 96),
        ("rows128_m32_a2", "relay_graph_chunk_rows", 128),
        ("sequence48_m32_a2", "relay_sequence_chunk_size", 48),
        ("sequence64_m32_a2", "relay_sequence_chunk_size", 64),
    ],
)
def test_chunk_sweep_changes_one_execution_parameter(variant, field, value):
    reference_path = CHUNK_SWEEP / "reference_m32_a2/gradpert_v2/nadig_jurkat.yaml"
    candidate_path = CHUNK_SWEEP / variant / "gradpert_v2/nadig_jurkat.yaml"
    reference = yaml.safe_load(reference_path.read_text())
    candidate = yaml.safe_load(candidate_path.read_text())
    assert candidate["model"]["parameters"].pop(field)["value"] == value
    assert candidate == reference
    config = load_experiment_config(candidate_path)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert getattr(arch, field) == value
    assert (options.lambda2, options.ssl2_dino, options.ssl2_ibot, options.ssl2_koleo) == (
        1.0,
        0.8,
        0.4,
        0.1,
    )
    assert config.training.max_epochs.value == 1
    assert config.training.train_batch_size.value == 128


@pytest.mark.parametrize(
    "variant,fields",
    [
        ("gate_only_m32_a2", {"graph_source_key_gate": True}),
        (
            "chunks_only_m32_a2",
            {"relay_scan_chunk_size": 64, "relay_sequence_chunk_size": 256},
        ),
        (
            "combined_m32_a2",
            {
                "graph_source_key_gate": True,
                "relay_scan_chunk_size": 64,
                "relay_sequence_chunk_size": 256,
            },
        ),
        (
            "combined_rows96_m32_a2",
            {
                "graph_source_key_gate": True,
                "relay_scan_chunk_size": 64,
                "relay_sequence_chunk_size": 256,
                "relay_graph_chunk_rows": 96,
            },
        ),
        (
            "combined_rows128_m32_a2",
            {
                "graph_source_key_gate": True,
                "relay_scan_chunk_size": 64,
                "relay_sequence_chunk_size": 256,
                "relay_graph_chunk_rows": 128,
            },
        ),
    ],
)
@pytest.mark.parametrize("microbatch", [16, 32])
def test_source_gate_chunk_profiles_are_self_contained_and_factorial(variant, fields, microbatch):
    reference_path = SOURCE_GATE_CHUNK / f"reference_m{microbatch}_a2/gradpert_v2/nadig_jurkat.yaml"
    candidate_path = (
        SOURCE_GATE_CHUNK
        / variant.replace("_m32_a2", f"_m{microbatch}_a2")
        / "gradpert_v2/nadig_jurkat.yaml"
    )
    reference = yaml.safe_load(reference_path.read_text())
    current_one_epoch = yaml.safe_load(ONE_EPOCH.read_text())
    current_one_epoch["model"]["parameters"]["microbatch"]["value"] = microbatch
    current_one_epoch["training"]["train_batch_size"]["value"] = microbatch * 4
    current_one_epoch["training"]["eval_batch_size"]["value"] = microbatch * 4
    assert reference == current_one_epoch
    candidate = yaml.safe_load(candidate_path.read_text())
    for name, value in fields.items():
        assert candidate["model"]["parameters"].pop(name)["value"] == value
    assert candidate == reference
    config = load_experiment_config(candidate_path)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert all(getattr(arch, name) == value for name, value in fields.items())
    assert (options.microbatch, options.accumulation, options.world_size) == (
        microbatch,
        2,
        2,
    )
    assert (options.lambda1, options.lambda2) == (1.0, 1.0)
    assert (config.training.max_epochs.value, config.training.train_batch_size.value) == (
        1,
        microbatch * 4,
    )


def test_source_gate_default_retains_previous_chunks_and_batch():
    previous = yaml.safe_load(ONE_EPOCH.read_text())
    current = yaml.safe_load(SOURCE_GATE_DEFAULT.read_text())
    gate = current["model"]["parameters"].pop("graph_source_key_gate")
    assert gate["value"] is True
    assert current == previous
    config = load_experiment_config(SOURCE_GATE_DEFAULT)
    architecture, options = V2Options.parse_parameters(config.model.parameters)
    assert architecture.graph_source_key_gate
    assert architecture.relay_scan_chunk_size == 32
    assert architecture.relay_sequence_chunk_size is None
    assert architecture.relay_graph_chunk_rows == 64
    assert (options.microbatch, options.accumulation, options.world_size) == (66, 2, 2)
    assert config.training.train_batch_size.value == 264


@pytest.mark.parametrize(
    "microbatch,rows,sequence",
    [
        (67, 64, 32),
        (68, 64, 32),
        (70, 64, 32),
        (72, 64, 32),
        (68, 32, 32),
        (70, 32, 32),
        (72, 32, 32),
        (68, 64, 16),
        (70, 64, 16),
        (72, 64, 16),
        (74, 64, 16),
        (76, 64, 16),
        (77, 64, 16),
        (78, 64, 16),
        (80, 64, 16),
        (66, 96, 32),
        (66, 128, 32),
    ],
)
def test_source_gate_capacity_profiles_only_change_batch_and_chunks(microbatch, rows, sequence):
    base = yaml.safe_load(SOURCE_GATE_DEFAULT.read_text())
    path = (
        SOURCE_GATE_CAPACITY
        / f"m{microbatch}_rows{rows}_seq{sequence}/gradpert_v2/nadig_jurkat.yaml"
    )
    payload = yaml.safe_load(path.read_text())
    payload["model"]["parameters"]["microbatch"]["value"] = 66
    payload["training"]["train_batch_size"]["value"] = 264
    if rows != 64:
        assert payload["model"]["parameters"].pop("relay_graph_chunk_rows")["value"] == rows
    if sequence != 32:
        assert payload["model"]["parameters"].pop("relay_sequence_chunk_size")["value"] == sequence
    assert payload == base
    config = load_experiment_config(path)
    architecture, options = V2Options.parse_parameters(config.model.parameters)
    assert architecture.graph_source_key_gate
    assert architecture.relay_scan_chunk_size == 32
    assert architecture.relay_graph_chunk_rows == rows
    assert (architecture.relay_sequence_chunk_size or 32) == sequence
    assert (options.microbatch, options.accumulation, options.world_size) == (microbatch, 2, 2)
    assert config.training.train_batch_size.value == 4 * microbatch


@pytest.mark.parametrize("change", ["version", "batch", "policy", "unknown"])
def test_v2_config_rejects_ambiguous_or_inconsistent_execution(change):
    payload = load_experiment_config(PROBE).model_dump(mode="json")
    if change == "version":
        payload["model"]["version"] = "v1"
    elif change == "batch":
        payload["training"]["train_batch_size"]["value"] = 100
    elif change == "policy":
        payload["training"]["formal_run_policy"] = "r50_selection"
    else:
        payload["model"]["parameters"]["unregistered_switch"] = {
            "value": 1,
            "source": "project_preregistered",
            "reference": "test",
        }
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(payload)


def test_expression_visibility_defaults_and_legacy_serialization():
    from gradpert.config.schema import ModelConfig

    v2 = load_experiment_config(PROBE).model
    assert v2.excludes_test_target_expression
    assert v2.model_dump()["exclude_test_target_expression"] is True
    for model_id, version in (("gradpert_b2", None), ("gradpert_v2", "v2")):
        payload = v2.model_dump()
        payload.update(model_id=model_id, version=version)
        payload.pop("exclude_test_target_expression")
        model = ModelConfig.model_validate(payload)
        assert model.excludes_test_target_expression == (model_id == "gradpert_v2")
        if model_id == "gradpert_b2":
            assert "exclude_test_target_expression" not in model.model_dump()
        for value in (False, True):
            changed = ModelConfig.model_validate(
                {**payload, "exclude_test_target_expression": value}
            )
            assert changed.excludes_test_target_expression is value
        with pytest.raises(ValueError):
            ModelConfig.model_validate({**payload, "exclude_test_target_expression": "false"})
