"""Numeric gates reject empty/failed evidence; no tolerance relaxation."""

import copy
import importlib
import sys
from pathlib import Path

import pytest
import torch

from benchmarks.expression_holdout.scdfm_parity import compare

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/v2"))
queue = importlib.import_module("run_official_expression_group")


def test_comparator_records_parameter_optimizer_and_rng_failures():
    value = {
        "weight": torch.ones(3),
        "optimizer": {"step": torch.tensor(3)},
        "rng": torch.arange(3),
    }
    assert not compare(value, copy.deepcopy(value))["failures"]
    other = copy.deepcopy(value)
    other["weight"][1] += 0.001
    other["optimizer"]["step"] += 1
    other["rng"][0] += 1
    result = compare(value, other)
    assert len(result["failures"]) == 3 and result["tensors"] == 3


@pytest.mark.parametrize("change", ["pairs", "updates", "tolerance", "failure", "config"])
def test_strict_offload_gate_rejects_missing_or_relaxed_receipts(change):
    row = {"config_sha256": "c", "runtime_sha256": "r"}
    receipt = {
        "schema": "scdfm-layout-offload-parity-1",
        "status": "passed",
        "source_clean": True,
        "rtol": 1e-6,
        "atol": 1e-7,
        **row,
        "pairs": {name: {"failures": [], "tensors": 10} for name in ("none-b", "cpu")},
        "modes": {name: {"updates": 3} for name in ("none-a", "none-b", "cpu")},
    }
    queue.validate_numerics(receipt, row)
    if change == "pairs":
        receipt["pairs"] = {}
    elif change == "updates":
        receipt["modes"]["cpu"]["updates"] = 2
    elif change == "tolerance":
        receipt["atol"] = 1e-4
    elif change == "failure":
        receipt["pairs"]["cpu"]["failures"] = ["weight"]
    else:
        receipt["config_sha256"] = "other"
    with pytest.raises(ValueError, match="numeric gate"):
        queue.validate_numerics(receipt, row)
