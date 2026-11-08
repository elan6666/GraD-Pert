"""Layout inventory, terminal-only replacement, real updates and state restore."""

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_restored_ffn import restored
from torch import nn

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.hashing import sha256_file
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.operators import LatentAttention, RelayDeltaAttention
from gradpert.modeling.v2.standard_attention import StandardAttention
from gradpert.training.v2.checkpoint import load_checkpoint, save_checkpoint
from gradpert.training.v2.engine import optimizer_step
from gradpert.training.v2.objective import JointObjective
from gradpert.training.v2.optimizer import V2Optimizer, routes

ROOT = Path(__file__).resolve().parents[2]


def build(terminal="latent", layout="llm_pre"):
    parent, batch = restored("BR")
    arch = replace(
        parent.student.options,
        ffn_variant="gelu",
        response_injection_residual=True,
        normalization_layout=layout,
        terminal_attention=terminal,
        relay_validate_once=True,
        cache_kda_constants=True,
    )
    torch.manual_seed(131)
    model = GraDPertV2(parent.student.graph.embedding.weight.detach().clone(), arch)
    rng = torch.get_rng_state().clone()
    return JointObjective(model, ssl1_weights=(1, 1, 0), ssl2_weights=(1, 1, 0)), batch, rng


@pytest.mark.parametrize("terminal", ["latent", "standard"])
def test_layout_inventory_and_graph_final_norm_exactly_once(terminal):
    objective, batch, _ = build(terminal)
    m = objective.student
    assert not any(isinstance(x, nn.LayerNorm) for x in m.modules())
    assert isinstance(m.graph.norm, nn.Identity) and isinstance(m.expression[-1], nn.Identity)
    assert all(
        isinstance(getattr(x, n), nn.RMSNorm) for x in m.graph.layers for n in ("norm1", "norm2")
    )
    assert all(isinstance(x[0], nn.RMSNorm) for x in m.response.injections)
    assert all(x[0].normalized_shape == (2 * m.options.width,) for x in m.response.injections)
    counts = []
    handle = m.graph.final_norm.register_forward_hook(lambda *args: counts.append(1))
    assert batch.graph.context is not None
    m.graph.relay_prefix(batch.graph.context)
    assert not counts
    m.graph(
        batch.graph.ids,
        batch.graph.neighbors,
        batch.graph.valid,
        batch.graph.sources,
        context=batch.graph.context,
    )
    handle.remove()
    assert counts == [1]
    assert isinstance(objective.teacher.graph.final_norm, nn.RMSNorm)


def test_common_initialization_rng_kda_and_all_four_terminal_paths():
    a, _, arng = build()
    b, _, brng = build("standard")
    old, _, oldrng = build(layout="historical")
    assert torch.equal(arng, brng) and torch.equal(arng, oldrng)
    for model in (a.student, b.student):
        assert all(isinstance(x.read, RelayDeltaAttention) for x in model.graph.layers[:3])
        assert all(isinstance(x.sublayer, RelayDeltaAttention) for x in model.cell.layers[:4:2])
        assert all(
            isinstance(x.sublayer, RelayDeltaAttention) for x in model.response.self_layers[:2]
        )
        assert all(
            isinstance(x.sublayer, RelayDeltaAttention) for x in model.response.cross_layers[:2]
        )
    assert sum(isinstance(x, StandardAttention) for x in b.student.modules()) == 3
    assert not any(isinstance(x, LatentAttention) for x in b.student.modules())
    assert b.student.graph.layers[-1].compress is None
    assert b.student.graph.layers[-1].latent_norm is None
    left, right = a.student.state_dict(), b.student.state_dict()
    changed = {"graph.layers.3.key.weight", "graph.layers.3.value.weight"}
    for path in (
        "cell.layers.4.sublayer",
        "response.self_layers.2.sublayer",
        "response.cross_layers.2.sublayer",
    ):
        changed.update({path + ".key.weight", path + ".value.weight"})
    for name in left.keys() & right.keys():
        if name not in changed:
            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
    for name, value in old.student.state_dict().items():
        if name in left and value.shape == left[name].shape:
            torch.testing.assert_close(value, left[name], atol=0, rtol=0)
    for route in routes(b.student):
        if route["name"].endswith(("query.weight", "key.weight", "value.weight")):
            assert route["heads"] == b.student.options.heads
        if "norm" in route["name"]:
            assert route["optimizer"] == "adamw"


@pytest.mark.parametrize("cross", [False, True])
def test_standard_attention_matches_explicit_softmax_outputs_and_gradients(cross):
    torch.manual_seed(11)
    m = StandardAttention(8, 2, 0).double()
    x = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)
    memory = torch.randn(2, 7, 8, dtype=torch.double, requires_grad=True) if cross else x
    q = m.query(x).unflatten(-1, (2, 4)).transpose(1, 2)
    k = m.key(memory).unflatten(-1, (2, 4)).transpose(1, 2)
    v = m.value(memory).unflatten(-1, (2, 4)).transpose(1, 2)
    expected = m.output(((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2).flatten(-2))
    actual = m(x, memory if cross else None)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    inputs = (x, memory, *m.parameters()) if cross else (x, *m.parameters())
    ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), inputs)
    for aa, ee in zip(ga, ge, strict=True):
        torch.testing.assert_close(aa, ee, atol=1e-11, rtol=1e-11)


def test_standard_cls_intervention_preserves_cls_read_but_blocks_gene_influence():
    m = StandardAttention(8, 2, 0).eval()
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, -1] += torch.randn(2, 8) * 10
    out = m(x, block_cls_to_gene=True)
    other = m(changed, block_cls_to_gene=True)
    torch.testing.assert_close(out[:, :-1], other[:, :-1], atol=0, rtol=0)
    torch.testing.assert_close(out[:, -1], m(x)[:, -1])
    assert not torch.allclose(out[:, -1], other[:, -1])


def test_uncompressed_graph_read_cannot_see_non_neighbors():
    objective, _, _ = build("standard")
    read = objective.student.graph.layers[-1].eval()
    width = objective.student.options.width
    query = torch.randn(2, width)
    memory = torch.randn(5, width, requires_grad=True)
    neighbors = torch.tensor([[0, 1, -1], [1, 2, -1]])
    valid = neighbors >= 0
    sources = torch.zeros(2, 3, 4)
    actual = read(query, memory, neighbors, valid, sources)
    changed = memory.detach().clone()
    changed[3:] += 100
    torch.testing.assert_close(
        actual, read(query, changed, neighbors, valid, sources), atol=0, rtol=0
    )
    gradient = torch.autograd.grad(actual.square().sum(), memory)[0]
    assert torch.count_nonzero(gradient[3:]) == 0
    assert gradient[:3].abs().sum() > 0


@pytest.mark.parametrize("terminal", ["latent", "standard"])
def test_full_update_and_exact_resume(terminal, tmp_path):
    o, batch, _ = build(terminal)
    optimizer = V2Optimizer(o.student, 1e-3, 0)
    generator = np.random.default_rng(1)
    before = {k: v.clone() for k, v in o.student.state_dict().items()}
    for _ in range(2):
        loss = optimizer_step(o, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False)
        assert all(math.isfinite(v) for v in loss.values()) and not o.pending
    assert any(not torch.equal(before[k], v) for k, v in o.student.state_dict().items())
    assert all(p.grad is None for p in o.teacher.parameters())
    path = tmp_path / "state.pt"
    identity = {"architecture": o.student.options.payload()}
    save_checkpoint(
        path, o, optimizer, identity=identity, progress={"step": 2}, generator=generator
    )
    expected = optimizer_step(o, optimizer, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False)
    state, rng = copy.deepcopy(o.state_dict()), torch.get_rng_state().clone()
    resumed, _, _ = build(terminal)
    opt = V2Optimizer(resumed.student, 1e-3, 0)
    load_checkpoint(path, resumed, opt, identity=identity, generator=generator)
    actual = optimizer_step(resumed, opt, batch, microbatch=4, lr=1e-3, momentum=0.9, bf16=False)
    assert actual == expected and torch.equal(rng, torch.get_rng_state())
    assert opt.state_dict()["routes"] == optimizer.state_dict()["routes"]
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)


def test_config_manifest_and_historical_identity():
    assert "normalization_layout" not in V2Architecture().payload()
    assert "terminal_attention" not in V2Architecture().payload()
    manifest = json.loads(
        (ROOT / "configs/v2/llm_norm_attention_20261109/manifest.json").read_text()
    )
    configs = []
    parent = load_experiment_config(ROOT / manifest["parent_config"]).model_dump()
    for row in manifest["rows"]:
        p = ROOT / row["config"]
        assert sha256_file(p) == row["sha256"]
        cfg = load_experiment_config(p)
        arch, opt = V2Options.parse_parameters(cfg.model.parameters)
        assert cfg.training.max_epochs.value == 2 and arch.normalization_layout == "llm_pre"
        assert arch.relay_validate_once and arch.cache_kda_constants
        assert (opt.microbatch, opt.accumulation, opt.world_size) == (80, 4, 1)
        value = cfg.model_dump()
        configs.append(value)
        value = copy.deepcopy(value)
        value["model"]["parameters"].pop("normalization_layout")
        value["model"]["parameters"].pop("terminal_attention")
        assert value == parent
    configs[0]["model"]["parameters"]["terminal_attention"] = configs[1]["model"]["parameters"][
        "terminal_attention"
    ]
    assert configs[0] == configs[1]


@pytest.mark.parametrize(
    "changes",
    [
        {"normalization_layout": "unknown"},
        {"terminal_attention": "unknown"},
        {"mla_query_rank": 4},
        {"graph_input_norm": "rmsnorm"},
        {"ffn_input_norm": "rmsnorm"},
        {"graph_mhc": True},
        {"attention_replacement": "softmax"},
        {"response_injection_residual": False},
    ],
)
def test_invalid_combinations_fail_closed(changes):
    o, _, _ = build()
    with pytest.raises(ValueError):
        replace(o.student.options, **changes)
