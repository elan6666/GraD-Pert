"""CPU preparation cannot allocate a GPU context during an active guarded fit."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
import run_cpu_tests as launcher  # noqa: E402


def test_cpu_environment_overrides_inherited_cuda_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("PYTHONPATH", "unrelated-source")
    env = launcher.cpu_test_environment(ROOT)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONPATH"] == str(ROOT / "src")
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_bootstrap_rejects_visible_cuda_before_importing_pytest(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    with pytest.raises(RuntimeError, match="no visible CUDA"):
        exec(launcher.CPU_TEST_BOOTSTRAP, {})


def test_cpu_child_preserves_args_exit_code_and_source(monkeypatch, tmp_path):
    calls = []

    def subprocess_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(launcher.subprocess, "run", subprocess_run)
    assert launcher.run(tmp_path, ["-q", "tests/v2/test_graph_mhc.py"], "python") == 7
    command, options = calls[0]
    assert command == [
        "python",
        "-c",
        launcher.CPU_TEST_BOOTSTRAP,
        "-q",
        "tests/v2/test_graph_mhc.py",
    ]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert options["cwd"] == tmp_path.resolve()
    assert options["check"] is False
