import json
import runpy
from pathlib import Path

import pytest

from gradpert.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/v2"
PARENT = ROOT / "configs/v2/capacity/m16/gradpert_v2/nadig_jurkat.yaml"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return runpy.run_path(str(SCRIPTS / "run_group.py"))


def preflight(tmp_path, config_hash="c" * 64):
    receipt = {
        "kind": "integration_only",
        "status": "passed",
        "steps_completed": 1,
        "resume_checkpoint_sha256": "b" * 64,
        "source": {"dirty": False, "commit": "a" * 40, "published_commit": "a" * 40},
        "config_sha256": config_hash,
        "data_root": "/data/yilangliu/data",
        "data": {"run_seed": 1},
        "world_size": 1,
        "peak_allocated_bytes": 1024,
        "last_terms": {"gradient_norm": 0.5},
    }
    path = tmp_path / f"{config_hash}.json"
    path.write_text(json.dumps(receipt))
    entry = {
        "receipt": str(path),
        "sha256": sha256_file(path),
        "config_sha256": config_hash,
        "seed": 1,
    }
    plan = {
        "config_sha256": config_hash,
        "source_commit": "a" * 40,
        "data_root": "/data/yilangliu/data",
        "seed": 1,
        "gpu": "0",
    }
    return receipt, entry, plan


def test_prepare_pins_verified_rows_to_exact_preflight_without_creating_runs(tmp_path, api):
    generate = runpy.run_path(str(SCRIPTS / "generate_group.py"))["generate"]
    group = generate(PARENT, "H1", tmp_path / "group")
    entries = []
    for row in group["rows"]:
        _, entry, _ = preflight(tmp_path, row["sha256"])
        entries.append(entry)
    index = tmp_path / "preflights.json"
    index.write_text(json.dumps({"rows": entries}))

    def resolver(args):
        return {
            "config_sha256": sha256_file(args.config),
            "source_commit": "a" * 40,
            "data_root": "/data/yilangliu/data",
            "seed": 1,
            "gpu": args.gpu,
        }

    queue = api["prepare_queue"](
        tmp_path / "group/manifest.json",
        PARENT,
        tmp_path / "runtime.json",
        "0",
        index,
        None,
        resolver=resolver,
    )
    assert len(queue["items"]) == 2
    assert queue["preflight_index_sha256"] == sha256_file(index)
    assert not (tmp_path / "runtime.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("source_commit", "x" * 40), ("seed", 2), ("gpu", "0,1"), ("config_sha256", "wrong")],
)
def test_preflight_cannot_admit_another_source_seed_topology_or_config(tmp_path, api, field, value):
    _, entry, plan = preflight(tmp_path)
    api["validate_preflight"](plan, entry)
    plan[field] = value
    with pytest.raises(ValueError):
        api["validate_preflight"](plan, entry)


def test_resume_preserves_owned_run_and_requires_committed_epoch(tmp_path, api):
    root = tmp_path / "run"
    plan = {"run_root": str(root)}
    assert api["next_action"](plan) == "launch"
    root.mkdir()
    with pytest.raises(ValueError, match="owned"):
        api["next_action"](plan)
    (root / "launch.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="epoch commit"):
        api["next_action"](plan)
    (root / "fit").mkdir()
    (root / "fit/epoch_state.json").write_text("{}")
    assert api["next_action"](plan) == "resume"


def test_queue_lease_prevents_simultaneous_owner(tmp_path, api):
    with (
        api["lock"](tmp_path / "queue.lock"),
        pytest.raises(RuntimeError, match="owns lease"),
        api["lock"](tmp_path / "queue.lock"),
    ):
        pass
