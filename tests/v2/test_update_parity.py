"""Evidence helpers must notice gradient and ordered-input changes."""

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gradpert.config.v2 import V2Architecture

SPEC = importlib.util.spec_from_file_location(
    "update_parity", Path(__file__).resolve().parents[2] / "scripts/v2/update_parity.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_input_digest_preserves_gene_order_dtype_and_rng_bits():
    @dataclass
    class Batch:
        genes: torch.Tensor
        rng: np.ndarray

    batch = Batch(torch.tensor([1, 2]), np.array([3, 4], dtype=np.uint32))
    original = MODULE.tree_digest(batch)
    assert original == MODULE.tree_digest(Batch(batch.genes.clone(), batch.rng.copy()))
    assert original != MODULE.tree_digest(Batch(batch.genes.flip(0), batch.rng))
    assert original != MODULE.tree_digest(Batch(batch.genes.float(), batch.rng))
    batch.rng[0] += 1
    assert original != MODULE.tree_digest(batch)


def test_parity_checks_small_gradients_and_optimizer_structure():
    reference = {"gradient": torch.zeros(4), "optimizer": {"step": 2, "state": torch.ones(2)}}
    candidate = MODULE.cpu_copy(reference)
    assert MODULE.compare_trees(reference, candidate)["passed"]
    candidate["gradient"][0] = 1e-3
    failed = MODULE.compare_trees(reference, candidate)
    assert not failed["passed"]
    assert failed["failures"][0]["path"] == "root/gradient"
    assert torch.equal(reference["gradient"], torch.zeros(4))
    del candidate["optimizer"]["step"]
    assert not MODULE.compare_trees(reference, candidate)["passed"]


def test_nonfinite_or_dtype_changes_do_not_pass_parity():
    assert not MODULE.compare_trees(torch.ones(2), torch.tensor([1.0, float("nan")]))["passed"]
    assert not MODULE.compare_trees(torch.ones(2), torch.ones(2, dtype=torch.float64))["passed"]


def test_digest_accepts_zero_stride_singletons_broadcast_and_bfloat16():
    for dtype in (torch.int64, torch.bfloat16):
        seed = torch.tensor([3], dtype=dtype)
        singleton = torch.as_strided(seed, (1,), (0,))
        assert MODULE.tree_digest(singleton) == MODULE.tree_digest(seed)
        broadcast = singleton.expand(5)
        assert MODULE.tree_digest(broadcast) == MODULE.tree_digest(torch.full((5,), 3, dtype=dtype))


def test_reference_repeat_cannot_hide_an_architecture_change():
    original = V2Architecture(
        attention="relay_full", graph_read_mode="relay", graph_layers=4, kda_layers=2
    )
    assert MODULE.execution_changes(original, original, True) == []
    changed = replace(original, relay_validate_once=True)
    assert MODULE.execution_changes(original, changed, False) == ["relay_validate_once"]
    with pytest.raises(AssertionError, match="identical architecture"):
        MODULE.execution_changes(original, changed, True)
    with pytest.raises(AssertionError, match="one execution factor"):
        MODULE.execution_changes(original, original, False)
    with pytest.raises(AssertionError, match="one execution factor"):
        MODULE.execution_changes(original, replace(changed, relay_kernel="inductor"), False)


def test_sequence_checkpoint_diagnostic_changes_both_sides_only():
    from types import SimpleNamespace

    def model():
        return SimpleNamespace(
            cell=SimpleNamespace(checkpoint_layers=True),
            response=SimpleNamespace(checkpoint_layers=True),
            graph=SimpleNamespace(checkpoint_chunks=True),
        )

    objective = SimpleNamespace(student=model(), teacher=model())
    MODULE.disable_sequence_checkpoint(objective)
    for network in (objective.student, objective.teacher):
        assert not network.cell.checkpoint_layers
        assert not network.response.checkpoint_layers
        assert network.graph.checkpoint_chunks


@pytest.mark.parametrize("conflict", ["--reference-repeat", "--no-sequence-checkpoint"])
def test_candidate_checkpoint_diagnostic_rejects_ambiguous_overrides(conflict):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/v2/update_parity.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            "unused",
            "--candidate-config",
            "unused",
            "--data-root",
            "/data/yilangliu",
            "--output",
            "/data/yilangliu/unused",
            "--gpu",
            "0,1",
            "--publication",
            "unused",
            "--publication-sha256",
            "unused",
            "--candidate-no-sequence-checkpoint",
            conflict,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "candidate-only checkpoint diagnostic" in result.stderr
