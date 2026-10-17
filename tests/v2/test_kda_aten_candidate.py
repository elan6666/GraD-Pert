"""Optional compiled candidate; ordinary native tests never require a compiler."""

import copy
import importlib.util
import os
import runpy
import sys
from pathlib import Path

import pytest
import torch

from gradpert.modeling.v2 import operators
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.optimizer import V2Optimizer

pytestmark = pytest.mark.skipif(
    os.environ.get("GRADPERT_TEST_ATEN_CANDIDATE") != "1", reason="opt-in compiled probe"
)
ROOT = Path(__file__).resolve().parents[2]


def helper(path):
    spec = importlib.util.spec_from_file_location("kda_candidate_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANDIDATE = helper("scripts/v2/kda_aten_candidate.py")
FIXTURE = helper("tests/v2/test_relay_method.py")


def exact(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            exact(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("length", [1, 17, 32, 47, 97])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_same_block_arithmetic_gradients_and_rng(length, bf16, chunk_size):
    torch.manual_seed(34 + length)
    shape = (2, length, 4, 64)
    inputs = (
        torch.nn.functional.normalize(torch.randn(shape), dim=-1),
        torch.randn(shape, dtype=torch.bfloat16 if bf16 else torch.float32),
        -torch.rand(shape) * 0.1,
        torch.rand(shape[:-1]),
        torch.randn(2, 4, 64, 64) * 0.1,
    )
    upstream = torch.randn(2, 4, 64, 64)
    results = []
    CANDIDATE.load_candidate()  # compilation must not contaminate RNG comparison
    rng = torch.get_rng_state().clone()
    for f in (operators.chunk_delta_final_state, CANDIDATE.final_state):
        args = [t.clone().requires_grad_() for t in inputs]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            output = f(*args[:4], state=args[4], chunk_size=chunk_size)
        gradients = torch.autograd.grad((output * upstream).sum(), args)
        results.append((output, gradients))
    exact(*results)
    exact(torch.get_rng_state(), rng)


def test_checkpointed_nonzero_lr_multistep_complete_update_and_cleanup():
    objective, batch = FIXTURE.relay_training_fixture(checkpointed=True)
    CANDIDATE.load_candidate()
    results = []
    original = operators.chunk_delta_final_state
    for enabled in (False, True):
        current = copy.deepcopy(objective)
        optimizer = V2Optimizer(current.student, 0.001, 0)
        torch.manual_seed(712)

        def updates(current, optimizer):
            terms = []
            for _ in range(3):
                loss = optimizer_step(
                    current,
                    optimizer,
                    batch,
                    microbatch=2,
                    lr=0.001,
                    momentum=0.99,
                    bf16=False,
                )
                gradients = {
                    n: p.grad.clone()
                    for n, p in current.student.named_parameters()
                    if p.grad is not None
                }
                terms.append((loss, gradients))
            return terms

        if enabled:
            with CANDIDATE.candidate_scope():
                terms = updates(current, optimizer)
        else:
            terms = updates(current, optimizer)
        assert operators.chunk_delta_final_state is original
        results.append((terms, current.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
    exact(*results)
    with pytest.raises(RuntimeError, match="injected"), CANDIDATE.candidate_scope():
        raise RuntimeError("injected")
    assert operators.chunk_delta_final_state is original


def test_mixed_candidates_fail_closed():
    with pytest.raises(ValueError, match="other candidates"):
        CANDIDATE.final_state(None, None, None, None, fused_gram=True)


@pytest.mark.parametrize(
    "script,flags,message",
    [
        ("capacity_probe.py", ["--aten-kda"], "benchmark-only"),
        (
            "capacity_probe.py",
            ["--aten-kda", "--benchmark-only", "--profile-last-update"],
            "unprofiled",
        ),
        ("update_parity.py", ["--candidate-aten-kda", "--reference-repeat"], "one execution"),
    ],
)
def test_invalid_diagnostic_modes_reject_before_cuda(monkeypatch, capsys, script, flags, message):
    path = ROOT / "scripts/v2" / script
    args = [
        str(path),
        "--config",
        "unused.yaml",
        "--data-root",
        "/data/yilangliu",
        "--output",
        "/data/yilangliu/unused",
        "--gpu",
        "0,1",
        "--publication",
        "unused.json",
        "--publication-sha256",
        "0" * 64,
    ]
    if script == "update_parity.py":
        args += ["--candidate-config", "unused.yaml"]
    monkeypatch.setattr(sys, "argv", [*args, *flags])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(path), run_name="__main__")
    assert error.value.code == 2
    assert message in capsys.readouterr().err
