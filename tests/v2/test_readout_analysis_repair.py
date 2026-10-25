"""R11 analysis recovery preserves training identity and native numerical gates."""

import contextlib
import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import run_fgpo_medium as medium  # noqa: E402
import run_readout_analysis_repair as repair  # noqa: E402
import run_representation_postfit as postfit  # noqa: E402
from cap40_posthoc import read, sha, write  # noqa: E402

FROZEN_ANALYSIS_SHA = "1ad4edf5a486865438e3d11e37427a4b132004d5"


@pytest.fixture
def frozen_repair_source(tmp_path):
    """Restore the minimal published R11 tree consumed by its exact source gate."""
    source = tmp_path / "frozen-reviewed-r11-source"
    for path in (
        "scripts/v2/diagnostic_view_options.py",
        "scripts/v2/cap40_posthoc.py",
        "scripts/v2/representation_panel.py",
    ):
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "show", FROZEN_ANALYSIS_SHA + ":" + path]
            )
        )
    return source


def queue(tmp_path):
    return {
        "family": repair.SCHEMA,
        "source_commit": repair.TRAINING_SHA,
        "source": str(tmp_path / "old-source"),
        "output": str(tmp_path / "old-queue"),
        "rows": [
            {
                "name": "R11-READOUT",
                "plan": {
                    "run_root": str(tmp_path / "old-run"),
                    "run_id": "unchanged-readout",
                    "config": str(tmp_path / "old-source/config.yaml"),
                    "data_root": "/data/yilangliu/data",
                    "config_sha256": "config-seal",
                    "resolved_config": {
                        "source_code": {"repository": "https://github.com/elan6666/GraD-Pert"}
                    },
                },
            }
        ],
        "medium_analysis": {
            "parent_manifest": str(tmp_path / "parent.json"),
            "parent_manifest_sha256": "parent-seal",
        },
        "shared_gpu_policy": {},
    }


def original(path):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", repair.TRAINING_SHA + ":" + path], text=True
    )


def fake_git(monkeypatch, *, changes=None, dirty="", commit="b" * 40):
    def execute(source, *args):
        if args[0] == "status":
            return dirty
        if args[0] == "rev-parse":
            return commit
        if args[0] == "diff":
            assert args == ("diff", "--name-status", "--no-renames", repair.TRAINING_SHA, "HEAD")
            return "\n".join(
                status + "\t" + path
                for path, status in (repair.ALLOWED_CHANGES if changes is None else changes).items()
            )
        assert args[0] == "show"
        assert args[1].startswith(repair.TRAINING_SHA + ":")
        return original(args[1].split(":", 1)[1])

    monkeypatch.setattr(repair, "git", execute)


def test_exact_reviewed_ast_and_helper_guard(tmp_path, monkeypatch, frozen_repair_source):
    fake_git(monkeypatch, commit=FROZEN_ANALYSIS_SHA)
    result = repair.validate_repair_source(queue(tmp_path), frozen_repair_source)
    assert result["commit"] == FROZEN_ANALYSIS_SHA
    assert result["diff"] == repair.ALLOWED_CHANGES
    assert sha(frozen_repair_source / "scripts/v2/diagnostic_view_options.py") == repair.HELPER_SHA


@pytest.mark.parametrize("kind", ["dirty", "unchanged", "source", "config", "removed", "renamed"])
def test_source_guard_rejects_changed_or_unpublished_method(tmp_path, monkeypatch, kind):
    changes = dict(repair.ALLOWED_CHANGES)
    dirty, commit = "", "b" * 40
    if kind == "dirty":
        dirty = "?? unexpected.py"
    elif kind == "unchanged":
        commit = repair.TRAINING_SHA
    elif kind in ("source", "config"):
        changes["src/gradpert/modeling/v2.py" if kind == "source" else "configs/changed.yaml"] = "M"
    elif kind == "removed":
        changes.pop("scripts/v2/representation_panel.py")
    else:
        changes["scripts/v2/representation_panel.py"] = "R100"
    fake_git(monkeypatch, changes=changes, dirty=dirty, commit=commit)
    with pytest.raises(ValueError):
        repair.validate_repair_source(queue(tmp_path), ROOT)


@pytest.mark.parametrize("panel", [False, True])
def test_guard_rejects_numerical_change_inside_declared_script(panel):
    path = "scripts/v2/representation_panel.py" if panel else "scripts/v2/cap40_posthoc.py"
    text = (ROOT / path).read_text()
    assert repair.normalized_sampling(original(path), current=False, panel=panel) == (
        repair.normalized_sampling(text, current=True, panel=panel)
    )
    changed = text.replace("np.random.default_rng(seed)", "np.random.default_rng(seed + 1)", 1)
    assert repair.normalized_sampling(original(path), current=False, panel=panel) != (
        repair.normalized_sampling(changed, current=True, panel=panel)
    )


@pytest.mark.parametrize("target", ["old-queue", "old-run", "old-source", ".", "outside"])
def test_output_never_overlaps_original(tmp_path, monkeypatch, target):
    monkeypatch.setattr(repair, "SERVER_ROOT", tmp_path)
    output = tmp_path.parent / "outside" if target == "outside" else tmp_path / target
    with pytest.raises(ValueError, match="independent"):
        repair.validate_output(queue(tmp_path), output)


def test_panel_command_only_rewrites_script_and_output(tmp_path):
    q = queue(tmp_path)
    write(Path(q["output"]) / "queue.json", q)
    row = q["rows"][0]
    old = postfit.panel_command(q, row)
    command = repair.repair_panel_command(q, row, ROOT, tmp_path / "new-panel")
    expected = old.copy()
    expected[1] = str(ROOT / "scripts/v2/run_representation_panel.py")
    expected[expected.index("--output") + 1] = str(tmp_path / "new-panel")
    assert command == expected
    assert command[command.index("--method-sha") + 1] == repair.TRAINING_SHA
    assert command[command.index("--resource-queue-sha256") + 1] == sha(
        Path(q["output"]) / "queue.json"
    )
    assert command[command.index("--checkpoint-role") + 1] == "last"


def test_panel_root_override_cannot_expand_other_families(tmp_path):
    with pytest.raises(ValueError, match="R11"):
        postfit.validate_panel({"family": "other"}, {}, root=tmp_path)


@pytest.mark.parametrize("failure", [None, "medium", "mutate-original", "pkl", "analysis-source"])
def test_analysis_only_lifecycle(tmp_path, monkeypatch, failure):
    import run_group

    import gradpert
    import gradpert.execution.identity as identity

    q = queue(tmp_path)
    queue_path = Path(q["output"]) / "queue.json"
    write(queue_path, q)
    write(
        Path(q["rows"][0]["plan"]["run_root"]) / "fit/frozen_readout_proof.json",
        {"frozen_equal": True},
    )
    monkeypatch.setattr(gradpert, "__file__", str(Path(q["source"]) / "src/gradpert/__init__.py"))
    monkeypatch.setattr(repair, "SERVER_ROOT", tmp_path)
    proof = {"commit": "b" * 40, "diff": repair.ALLOWED_CHANGES}
    monkeypatch.setattr(repair, "validate_repair_source", lambda *a: proof)
    published = SimpleNamespace(commit="b" * 40, dirty=False, payload=lambda: {"commit": "b" * 40})
    monkeypatch.setattr(identity, "inspect_source_identity", lambda *a, **kw: published)
    monkeypatch.setattr(
        postfit, "context", lambda *a: {"last": {"sha256": "checkpoint", "epoch": 2}}
    )
    monkeypatch.setattr(
        postfit,
        "validate_validation",
        lambda *a: {"receipt": "original-val", "sha256": "validation"},
    )
    monkeypatch.setattr(repair, "validate_controller", lambda *a: {})
    monkeypatch.setattr(run_group, "lock", lambda *a: contextlib.nullcontext())
    calls = []

    def execute_medium(path, arm, *, repair_output, repair_publication, repair_publication_sha256):
        calls.append("medium")
        assert repair_publication == tmp_path / "publication.json"
        assert repair_publication_sha256 == "publication"
        assert read(path) == q and arm == "R11-READOUT"
        if failure == "medium":
            raise RuntimeError("original seal failure remains preserved")
        write(
            repair_output / "MEDIUM_COMPLETE.json",
            {
                "updates": 0,
                "report_controller_commit": "b" * 40,
            },
        )
        write(
            repair_output / "analysis/summary.json",
            {
                "analysis": {
                    "commit": "a" * 40 if failure == "analysis-source" else "b" * 40,
                    "dirty": False,
                },
            },
        )
        if failure == "mutate-original":
            write(queue_path, {**q, "changed": True})
        if failure == "pkl":
            (repair_output / "unexpected.pkl").write_bytes(b"bad")

    monkeypatch.setattr(medium, "execute", execute_medium)
    monkeypatch.setattr(
        medium,
        "validate_medium_complete",
        lambda *a, root: {"receipt": str(root / "MEDIUM_COMPLETE.json")},
    )

    def run_panel(command, *, env, **kwargs):
        calls.append("panel")
        assert command[command.index("--method-sha") + 1] == repair.TRAINING_SHA
        assert env["PYTHONPATH"] == str(Path(q["source"]) / "src")
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        write(
            Path(command[command.index("--output") + 1]) / "COMPLETE.json",
            {
                "updates": 0,
                "analysis_source": {"commit": "b" * 40, "dirty": False},
            },
        )

    monkeypatch.setattr(repair.subprocess, "run", run_panel)
    monkeypatch.setattr(
        postfit, "validate_panel", lambda *a, root: {"receipt": str(root / "COMPLETE.json")}
    )
    output = tmp_path / "repair"
    args = dict(
        analysis_sha="b" * 40,
        publication=tmp_path / "publication.json",
        publication_sha256="publication",
    )
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            repair.execute(queue_path, output, **args)
        assert not (output / "ANALYSIS_REPAIR_COMPLETE.json").exists()
        assert read(output / "FAILURE.json")["updates"] == 0
    else:
        receipt = read(repair.execute(queue_path, output, **args))
        assert calls == ["medium", "panel"]
        assert receipt["updates"] == 0 and receipt["zero_pkl"]
        assert receipt["retrained"] is receipt["validation_rerun"] is False
        assert receipt["inputs"]["training_source_commit"] == repair.TRAINING_SHA
        assert receipt["inputs"]["analysis_source"]["commit"] != repair.TRAINING_SHA
        assert read(queue_path) == q


@pytest.mark.parametrize("bad", [None, "tolerance", "error", "population", "state", "endpoint"])
def test_native_r11_panel_override_preserves_strict_parity_and_population(
    tmp_path, monkeypatch, bad
):
    from representation_panel import panel_schedule
    from test_representation_panel import parent

    q = queue(tmp_path)
    row = q["rows"][0]
    frozen = parent()
    example = frozen["schedule"][0]
    frozen["schedule"] = []
    for index in range(32):
        group = copy.deepcopy(example)
        group["condition_id"] = f"p{index}"
        for control, record in enumerate(group["rows"]):
            record.update(condition_id=f"p{index}", truth_row_id=f"t{index}-{control}")
        frozen["schedule"].append(group)
    write(q["medium_analysis"]["parent_manifest"], frozen)
    q["medium_analysis"]["parent_manifest_sha256"] = sha(q["medium_analysis"]["parent_manifest"])
    monkeypatch.setattr(repair, "SERVER_ROOT", tmp_path)
    monkeypatch.setattr(postfit, "context", lambda *a: {"last": {"sha256": "checkpoint"}})
    panel = panel_schedule(frozen)
    assert panel["conditions"] == 32 and panel["rows"] == 128
    root = tmp_path / "new-panel"
    write(root / "panel.json", panel)
    record = dict(
        status="completed",
        updates=0,
        method_sha=q["source_commit"],
        run_id=row["plan"]["run_id"],
        checkpoint_sha256="checkpoint",
        config_sha256=row["plan"]["config_sha256"],
        state_unchanged=True,
        parent_manifest=q["medium_analysis"]["parent_manifest"],
        parent_manifest_sha256=q["medium_analysis"]["parent_manifest_sha256"],
        state_before={"model": "same"},
        state_after={"model": "same"},
        panel_sha256=panel["panel_sha256"],
        panel_rows=128,
        checkpoint_endpoint=dict(
            role="last", epoch=2, training_stage="frozen_readout", full_training_epochs=2
        ),
        parity=dict(
            used_path_passed=True,
            tolerance=2e-6,
            cell_chunk_used=False,
            graph_max_abs=2e-6,
            response_max_abs=2e-6,
        ),
        outputs=postfit.output_seal(root),
    )
    if bad == "tolerance":
        record["parity"]["tolerance"] = 3e-6
    elif bad == "error":
        record["parity"]["response_max_abs"] = 2.00001e-6
    elif bad == "population":
        record["panel_rows"] = 127
    elif bad == "state":
        record["state_after"] = {"model": "changed"}
    elif bad == "endpoint":
        record["checkpoint_endpoint"]["training_stage"] = "joint"
    write(root / "COMPLETE.json", record)
    if bad:
        with pytest.raises(ValueError):
            postfit.validate_panel(q, row, root=root)
    else:
        assert postfit.validate_panel(q, row, root=root)["panel_sha256"] == panel["panel_sha256"]


@pytest.mark.parametrize("bad", [None, "count", "population", "parity"])
def test_native_medium_override_keeps_original_counts_and_parity(tmp_path, monkeypatch, bad):
    q = queue(tmp_path)
    row = q["rows"][0]
    root = tmp_path / "new-medium"
    write(
        Path(row["plan"]["run_root"]) / "fit/epoch_state.json", {"last": {"sha256": "checkpoint"}}
    )
    write(Path(q["output"]) / "queue.json", q)
    parent = {
        "schedule": [
            dict(
                index=0,
                condition_id="original",
                rows=[],
                deep_rows=[],
                gradient_rows=[],
                queries=[],
                view_seed=3,
            )
        ]
    }
    monkeypatch.setattr(medium, "validate_analysis_spec", lambda *a: parent)
    pop = medium.population_identity(parent["schedule"])
    record = dict(
        status="completed",
        arm="R11-READOUT",
        source_commit=q["source_commit"],
        queue_sha256=sha(Path(q["output"]) / "queue.json"),
        checkpoint_sha256="checkpoint",
        updates=0,
        diagnostic_population_sha256=pop,
        **medium.COUNTS,
        parity=dict(passed=True, atol=1e-8, max_abs=1e-8),
        outputs={},
    )
    summary = dict(
        updates=0,
        state_before="same",
        state_after="same",
        method_sha=q["source_commit"],
        checkpoint_sha256="checkpoint",
        config_sha256=row["plan"]["config_sha256"],
        diagnostic_population_sha256=pop,
        **medium.COUNTS,
    )
    if bad == "count":
        summary["rows"] -= 1
    elif bad == "population":
        summary["diagnostic_population_sha256"] = "changed"
    elif bad == "parity":
        record["parity"]["atol"] = 2e-8
    write(root / "MEDIUM_COMPLETE.json", record)
    write(root / "analysis/summary.json", summary)
    write(root / "analysis/COMPLETE.json", {"state_unchanged": True})
    if bad:
        with pytest.raises(ValueError):
            medium.validate_medium_complete(q, row, root=root)
    else:
        assert medium.validate_medium_complete(q, row, root=root)["rows"] == 5317


@pytest.mark.parametrize("bad", [None, "visible", "allocator", "cublas", "own-process"])
def test_cpu_controller_checks_original_admission_policy(tmp_path, monkeypatch, bad):
    import shared_gpu

    q = queue(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if bad == "visible" else "")
    monkeypatch.setenv(
        "PYTORCH_ALLOC_CONF", "wrong" if bad == "allocator" else "expandable_segments:True"
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "wrong" if bad == "cublas" else ":4096:8")
    observed = {"processes": [{"identity": {"uid": os.getuid()}}] if bad == "own-process" else []}
    monkeypatch.setattr(shared_gpu, "snapshot", lambda pid: observed)
    admissions = []
    monkeypatch.setattr(shared_gpu, "admit", lambda *a: admissions.append(a))
    if bad:
        with pytest.raises(ValueError):
            repair.validate_controller(q)
        assert not admissions
    else:
        assert repair.validate_controller(q) == observed
        assert admissions == [(q["shared_gpu_policy"], observed)]


def test_direct_native_r11_repair_requires_new_publication(tmp_path, monkeypatch):
    q = queue(tmp_path)
    monkeypatch.setattr(repair, "validate_repair_source", lambda *a: {})
    with pytest.raises(ValueError, match="published analysis receipt"):
        medium.validate_report_repair(q, "R11-READOUT", tmp_path / "new", ROOT)
