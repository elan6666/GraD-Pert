import runpy
from pathlib import Path

import pytest

POLICY = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/v2/capacity_probe.py"))[
    "probe_policy"
]


def test_short_integration_evidence_is_a_separate_kind():
    assert POLICY(True, None) == (1, 0, "integration_only")
    assert POLICY(False, None) == (128, 8, "capacity_only")
    with pytest.raises(ValueError, match="128"):
        POLICY(False, 1)
    with pytest.raises(ValueError, match="exactly one"):
        POLICY(True, 128)


def test_bounded_benchmark_cannot_be_mistaken_for_capacity():
    assert POLICY(False, None, True) == (5, 1, "benchmark_only")
    assert POLICY(False, 7, True) == (7, 1, "benchmark_only")
    with pytest.raises(ValueError, match="at least three"):
        POLICY(False, 2, True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        POLICY(True, None, True)


@pytest.mark.parametrize(
    "flags,expected",
    [
        (["--profile-last-update"], "profile requires"),
        (["--no-sequence-checkpoint"], "benchmark-only diagnostic"),
        (["--benchmark-only", "--steps", "3", "--warmup-steps", "3"], "custom warmup"),
        (["--benchmark-only", "--profile-last-update", "--sync-phase-timing"], "profile requires"),
        (["--benchmark-only", "--profile-memory"], "profile-memory requires"),
        (["--integration-only"], "launch requires PYTORCH_ALLOC_CONF"),
    ],
)
def test_probe_rejects_invalid_diagnostics_before_importing_cuda(flags, expected):
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/v2/capacity_probe.py"
    env = os.environ.copy()
    env.pop("PYTORCH_ALLOC_CONF", None)
    env["WORLD_SIZE"], env["LOCAL_RANK"] = "1", "0"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "/not-read.yaml",
            "--data-root",
            "/data/yilangliu/not-read",
            "--output",
            "/data/yilangliu/not-created",
            "--gpu",
            "0",
            "--publication",
            "/not-read.json",
            "--publication-sha256",
            "unused",
            *flags,
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert expected in result.stderr
