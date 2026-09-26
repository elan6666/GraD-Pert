"""Evidence helpers must notice gradient and ordered-input changes."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

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
