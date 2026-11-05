"""Validate device binding without importing torch or initializing CUDA."""

import ast
from pathlib import Path

import pytest


def assignment():
    path = Path(__file__).resolve().parents[2] / "scripts/v2/update_parity.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_gpu_assignment"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("gpu", ["0", "1"])
def test_single_rank_only_binds_the_selected_card(gpu):
    assert assignment()(gpu, 1, 0) == [gpu]


@pytest.mark.parametrize("gpu", ["0,1", "1,0"])
@pytest.mark.parametrize("rank", [0, 1])
def test_historical_dual_rank_order_is_preserved(gpu, rank):
    devices = assignment()(gpu, 2, rank)
    assert devices == gpu.split(",")
    assert devices[rank] == gpu.split(",")[rank]


@pytest.mark.parametrize(
    "gpu,world,rank",
    [
        ("", 1, 0),
        ("2", 1, 0),
        (" 1", 1, 0),
        ("0,0", 2, 0),
        ("0,1", 1, 0),
        ("1", 2, 0),
        ("0,1,2", 3, 0),
        ("1", 0, 0),
        ("1", 1, -1),
        ("1", 1, 1),
        ("0,1", 2, 2),
        ("1", True, 0),
        ("1", 1, False),
    ],
)
def test_invalid_assignments_fail_before_cuda(gpu, world, rank):
    with pytest.raises(ValueError, match="one worker per selected physical GPU"):
        assignment()(gpu, world, rank)
