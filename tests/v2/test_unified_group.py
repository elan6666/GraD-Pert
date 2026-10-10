"""The authorized first batch keeps one protocol and exactly declared arm changes."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options
from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "unified_matrix", ROOT / "scripts/v2/generate_unified_group.py"
)
assert spec and spec.loader
matrix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)


def generated(tmp_path: Path, batch: int = 8) -> Path:
    parent = tmp_path / matrix.PARENT
    parent.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / matrix.PARENT, parent)
    output = tmp_path / matrix.DEFAULT_OUTPUT
    matrix.generate(tmp_path, output, batch)
    return output / "manifest.json"


def test_checked_in_matrix_has_only_ten_authorized_arms():
    m = matrix.verify_manifest(ROOT, ROOT / matrix.DEFAULT_OUTPUT / "manifest.json")
    assert [row["name"] for row in m["rows"]] == [
        "N0",
        "U24",
        "MR1",
        "P1",
        "C1",
        "O1",
        "VH",
        "S1-L4",
        "CG1",
        "S12-L4",
    ]
    assert m["baseline_prechange_sha"] == "40fb032365a7d61b8de6caae7e4fb15b78c2ee24"
    assert (m["epochs"], m["validation"], m["test_roles"]) == (6, "disabled", ["last"])
    assert len({row["experiment_id"] for row in m["rows"]}) == 10
    batch = m["common_microbatch"]
    assert batch >= 2 and m["global_batch"] == batch
    for row in m["rows"]:
        cfg = load_experiment_config(ROOT / row["config"])
        architecture, options = V2Options.parse_parameters(cfg.model.parameters)
        assert architecture.mlp_profile == "unified"
        assert architecture.genept_projection_activation == "none"
        assert architecture.learned_genept_projection
        assert architecture.graph_source_key_gate and architecture.relay_passes == 1
        assert architecture.relay_scan_chunk_size == 32
        assert architecture.relay_graph_chunk_rows == 64
        assert architecture.relay_sequence_chunk_size == 16
        assert (options.world_size, options.accumulation, options.microbatch) == (1, 1, batch)
        assert cfg.training.train_batch_size.value == cfg.training.eval_batch_size.value == batch
        assert options.independent_view_rng and options.validation_mode == "disabled"
        assert cfg.model.exclude_test_target_expression
        assert cfg.training.max_epochs.value == 6
        assert cfg.training.run_seeds == [1] and not cfg.training.early_stopping
        assert options.ssl1_spread == options.ssl2_koleo == 0
        assert options.lambda1 == options.lambda2 == 1
        assert options.ssl1_condition == options.ssl1_node == 1
        assert options.ssl2_dino == options.ssl2_ibot == 1
        assert cfg.evaluation.n_controls_per_condition == 300


def test_local_recipes_and_population_changes_are_exact(tmp_path):
    m = matrix.verify_manifest(tmp_path, generated(tmp_path))
    options = {
        row["name"]: V2Options.parse_parameters(
            load_experiment_config(tmp_path / row["config"]).model.parameters
        )[1]
        for row in m["rows"]
    }
    assert (options["N0"].ssl1_local_views, options["N0"].ssl2_local_views) == (2, 2)
    assert options["S1-L4"].ssl1_local_layout == "go_string_random"
    assert options["S1-L4"].ssl2_local_views == 2
    assert options["S12-L4"].ssl1_local_layout == "go_go_string_string"
    assert options["S12-L4"].ssl2_local_views == 4
    assert options["P1"].population_response and options["P1"].max_conditions == 1
    assert options["P1"].lambda_mmd == 1
    assert options["MR1"].masked_response_ratio == 0.25
    assert options["MR1"].lambda_masked_response == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 20),
        ("world_size", 2),
        ("accumulation", 2),
        ("global_batch", 272),
        ("validation", "joint_only"),
        ("test_roles", ["best", "last"]),
        ("seed", 2),
        ("baseline_prechange_sha", "0" * 40),
    ],
)
def test_changed_experiment_scope_is_rejected(tmp_path, field, value):
    path = generated(tmp_path)
    m = json.loads(path.read_text())
    m[field] = value
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="authorized first batch"):
        matrix.verify_manifest(tmp_path, path)


@pytest.mark.parametrize(
    "arm,field,value",
    [
        ("N0", "lambda2", 0.8),
        ("CG1", "graph_source_key_gate", False),
        ("U24", "lambda_mmd", 2),
        ("S12-L4", "ssl1_local_layout", "go_string_random"),
        ("MR1", "masked_response_ratio", 0.5),
    ],
)
def test_undeclared_arm_changes_fail_even_with_rehashed_yaml(tmp_path, arm, field, value):
    path = generated(tmp_path)
    m = json.loads(path.read_text())
    row = next(row for row in m["rows"] if row["name"] == arm)
    cfg = tmp_path / row["config"]
    v = yaml.safe_load(cfg.read_text())
    v["model"]["parameters"][field]["value"] = value
    cfg.write_text(yaml.safe_dump(v))
    row["sha256"] = sha256_file(cfg)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        matrix.verify_manifest(tmp_path, path)


def test_arm_provenance_cannot_be_relabelled(tmp_path):
    path = generated(tmp_path)
    m = json.loads(path.read_text())
    row = m["rows"][1]
    cfg = tmp_path / row["config"]
    v = yaml.safe_load(cfg.read_text())
    v["model"]["parameters"]["width"]["reference"] = "unverified-paper"
    cfg.write_text(yaml.safe_dump(v))
    row["sha256"] = sha256_file(cfg)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="unrelated"):
        matrix.verify_manifest(tmp_path, path)


@pytest.mark.parametrize("action", ["reorder", "append_paused", "omit", "declaration"])
def test_queue_scope_and_declared_deltas_cannot_change(tmp_path, action):
    path = generated(tmp_path)
    m = json.loads(path.read_text())
    if action == "reorder":
        m["rows"][1:3] = list(reversed(m["rows"][1:3]))
    elif action == "append_paused":
        m["rows"].append({**m["rows"][0], "name": "D1"})
    elif action == "omit":
        m["rows"].pop()
    else:
        m["rows"][1]["changes"]["lambda2"] = 0
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="authorized first batch"):
        matrix.verify_manifest(tmp_path, path)


def test_config_path_cannot_escape_source(tmp_path):
    path = generated(tmp_path)
    m = json.loads(path.read_text())
    m["rows"][0]["config"] = "../nadig_jurkat.yaml"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="escaped"):
        matrix.verify_manifest(tmp_path, path)


def test_rebatching_replaces_all_arms_and_cannot_overwrite_tampered_matrix(tmp_path):
    path = generated(tmp_path)
    with pytest.raises(FileExistsError):
        matrix.generate(tmp_path, path.parent, 16)
    m = matrix.generate(tmp_path, path.parent, 16, replace=True)
    assert m["common_microbatch"] == m["global_batch"] == 16
    for row in m["rows"]:
        cfg = load_experiment_config(tmp_path / row["config"])
        assert cfg.model.parameters["microbatch"].value == 16
        assert cfg.training.train_batch_size.value == cfg.training.eval_batch_size.value == 16
    m["epochs"] = 7
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="authorized first batch"):
        matrix.generate(tmp_path, path.parent, 8, replace=True)


@pytest.mark.parametrize("batch", [0, 1, -1, True, 8.0])
def test_invalid_batch_is_rejected(tmp_path, batch):
    path = tmp_path / matrix.PARENT
    path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / matrix.PARENT, path)
    with pytest.raises(ValueError, match="at least two"):
        matrix.generate(tmp_path, tmp_path / matrix.DEFAULT_OUTPUT, batch)


def test_parent_checksum_is_frozen(tmp_path):
    path = generated(tmp_path)
    parent = tmp_path / matrix.PARENT
    parent.write_text(parent.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="parent checksum"):
        matrix.verify_manifest(tmp_path, path)
