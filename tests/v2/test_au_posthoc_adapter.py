"""Observer-only AU adaptation preserves the original scientific options."""

import copy
import dataclasses
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Options

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import cap40_posthoc as posthoc  # noqa: E402
import run_fgpo_medium as medium  # noqa: E402


@pytest.mark.parametrize("arm", ["R3-AUB", "R3-AUS"])
@pytest.mark.parametrize("deep", [False, True])
def test_real_au_options_only_disable_shallow_assembly(tmp_path, monkeypatch, arm, deep):
    config = ROOT / f"configs/v2/representation_round3_20261023/{arm}/gradpert_v2/nadig_jurkat.yaml"
    _, options = V2Options.parse_parameters(load_experiment_config(config).model.parameters)
    before = copy.deepcopy(options)
    runtime = SimpleNamespace(options=options, index=object())
    views = ModuleType("gradpert.training.v2.views")
    views.assemble_batch = lambda raw, index, assembly_options, rng, **kwargs: assembly_options
    monkeypatch.setitem(sys.modules, views.__name__, views)
    monkeypatch.setattr(posthoc, "make_raw", lambda *a, **kw: object())
    actual = posthoc.assemble(runtime, [], [1], 3, deep=deep)
    expected = (
        dict(lambda1=1.0, lambda2=1.0)
        if deep
        else dict(lambda1=0.0, lambda2=0.0, au_alignment_weight=0.0, au_uniformity_weight=0.0)
    )
    assert actual == dataclasses.replace(options, **expected)
    assert runtime.options is options
    assert options == before
    assert options.au_alignment_weight > 0 and options.au_uniformity_weight > 0


@pytest.mark.parametrize("arm", ["R3-AUB", "R3-AUS", "R3-RI"])
@pytest.mark.parametrize("drift", [None, "cap40", "src", "dirty"])
def test_report_guard_allows_only_exact_au_shallow_patch(tmp_path, monkeypatch, arm, drift):
    monkeypatch.setattr(medium, "SERVER_ROOT", tmp_path)
    source = tmp_path / "repair-source"
    target = source / "scripts/v2/cap40_posthoc.py"
    target.parent.mkdir(parents=True)
    corrected = (ROOT / "scripts/v2/cap40_posthoc.py").read_text()
    before = "else dataclasses.replace(runtime.options, lambda1=0.0, lambda2=0.0)"
    after = """else dataclasses.replace(
            runtime.options,
            lambda1=0.0,
            lambda2=0.0,
            **(
                {"au_alignment_weight": 0.0, "au_uniformity_weight": 0.0}
                if getattr(runtime.options, "au_alignment_weight", 0.0)
                or getattr(runtime.options, "au_uniformity_weight", 0.0)
                else {}
            ),
        )"""
    assert corrected.count(after) == 1
    original = corrected.replace(after, before)
    target.write_text(corrected + ("\n# unrelated edit\n" if drift == "cap40" else ""))
    queue = dict(
        family=medium.RS3_SCHEMA,
        source_commit="a" * 40,
        output=str(tmp_path / "queue"),
        rows=[dict(plan=dict(run_root=str(tmp_path / "run")))],
    )

    def git(command, **kwargs):
        if "status" in command:
            return " M src/changed.py" if drift == "dirty" else ""
        if "show" in command:
            assert command[-1] == queue["source_commit"] + ":scripts/v2/cap40_posthoc.py"
            return original
        assert "--name-only" in command
        return "scripts/v2/cap40_posthoc.py\n" + ("src/changed.py\n" if drift == "src" else "")

    monkeypatch.setattr(medium.subprocess, "check_output", git)
    output = tmp_path / "fresh-repair"
    if drift is None and arm != "R3-RI":
        assert medium.validate_report_repair(queue, arm, output, source) == output
    else:
        with pytest.raises(ValueError):
            medium.validate_report_repair(queue, arm, output, source)


@pytest.mark.parametrize("deep", [False, True])
def test_historical_options_without_au_fields_still_assemble(monkeypatch, deep):
    @dataclasses.dataclass(frozen=True)
    class HistoricalOptions:
        lambda1: float = 0.5
        lambda2: float = 0.5

    options = HistoricalOptions()
    runtime = SimpleNamespace(options=options, index=object())
    views = ModuleType("gradpert.training.v2.views")
    views.assemble_batch = lambda raw, index, assembly_options, rng, **kwargs: assembly_options
    monkeypatch.setitem(sys.modules, views.__name__, views)
    monkeypatch.setattr(posthoc, "make_raw", lambda *a, **kw: object())
    actual = posthoc.assemble(runtime, [], [1], 3, deep=deep)
    value = 1.0 if deep else 0.0
    assert actual == HistoricalOptions(lambda1=value, lambda2=value)
    assert runtime.options is options
    assert options == HistoricalOptions()
