"""Frozen DDP matrix and resource admission retain the scientific protocol."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from generate_unified_ddp import PRIORITY, configs, verify_manifest  # noqa: E402
from generate_unified_group import baseline  # noqa: E402
from run_unified_ddp import fit_budget, probe  # noqa: E402


def test_all_ten_configs_restore_dual_accumulation_and_only_change_execution():
    manifest = verify_manifest(ROOT, ROOT / "configs/v2/unified_ddp_20261010/manifest.json")
    assert tuple(manifest["priority"]) == ("N0", "CG1", "U24", "MR1", "P1")
    reference = baseline(ROOT, 32)
    for row in manifest["rows"]:
        cfg = load_experiment_config(ROOT / row["config"])
        _, options = V2Options.parse_parameters(cfg.model.parameters)
        assert (options.microbatch, options.world_size, options.accumulation) == (68, 2, 2)
        assert cfg.training.train_batch_size.value == 272
        assert cfg.training.eval_batch_size.value == 32
        value = yaml.safe_load((ROOT / row["config"]).read_text())
        for key in ("world_size", "accumulation", "microbatch", *row["changes"]):
            value["model"]["parameters"][key] = reference["model"]["parameters"][key]
        value["training"]["train_batch_size"] = reference["training"]["train_batch_size"]
        assert value == reference
    assert len(manifest["rows"]) == 10 and len(PRIORITY) == 5


def test_cg_and_population_config_are_loadable_with_two_rank_accumulation():
    values = configs(ROOT, 48)
    for name in ("CG1", "P1"):
        from gradpert.config.schema import ExperimentConfig

        cfg = ExperimentConfig.model_validate(values[name])
        _, options = V2Options.parse_parameters(cfg.model.parameters)
        assert options.world_size == options.accumulation == 2


def test_resource_admission_keeps_margin_and_no_unmeasured_overlap():
    assert fit_budget(50, 100) == 0.65
    assert fit_budget(60, 100) == 1.0
    with pytest.raises(ValueError):
        fit_budget(0, 100)


def test_probe_uses_two_ranks_and_exactly_ten_steps(tmp_path):
    plan = {
        "repository_root": "/data/yilangliu/source",
        "config": "cfg",
        "data_root": "data",
        "publication": "pub",
        "publication_sha256": "hash",
    }
    command = probe({"name": "CG1", "plan": plan}, tmp_path)
    assert command[1:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        "/data/yilangliu/source/scripts/v2/capacity_probe.py",
    ]
    assert command[command.index("--steps") + 1] == "10"
    assert command[command.index("--gpu") + 1] == "0,1"


def test_manifest_does_not_accept_order_or_unrelated_method_change(tmp_path):
    import shutil

    from generate_unified_ddp import generate
    from generate_unified_group import PARENT

    parent = tmp_path / PARENT
    parent.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / PARENT, parent)
    directory = tmp_path / "configs/v2/ddp"
    generate(tmp_path, directory, 48)
    path = directory / "manifest.json"
    value = json.loads(path.read_text())
    changed = copy.deepcopy(value)
    changed["priority"][:2] = list(reversed(changed["priority"][:2]))
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="priority"):
        verify_manifest(tmp_path, path)
