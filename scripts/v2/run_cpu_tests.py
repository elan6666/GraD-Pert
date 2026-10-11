"""Run synthetic tests with GPUs hidden, including lazy checkpoint RNG capture."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CPU_TEST_BOOTSTRAP = """
import sys
import torch
if torch.cuda.is_available() or torch.cuda.is_initialized():
    raise RuntimeError('CPU test child must have no visible CUDA devices')
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""


def cpu_test_environment(source: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(source.resolve() / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    return env


def run(source: Path, pytest_args: list[str], python: str = sys.executable) -> int:
    child = subprocess.run(
        [python, "-c", CPU_TEST_BOOTSTRAP, *pytest_args],
        cwd=source.resolve(),
        env=cpu_test_environment(source),
        check=False,
    )
    return child.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if not pytest_args:
        parser.error("supply explicit CPU test paths after --")
    return run(args.source, pytest_args, args.python)


if __name__ == "__main__":
    raise SystemExit(main())
