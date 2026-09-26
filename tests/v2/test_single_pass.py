"""One writing pass changes no final-state readout or historical configuration."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.operators import RelayDeltaAttention, delta_final_state


def test_single_pass_cross_reads_final_control_state() -> None:
    torch.manual_seed(17)
    layer = RelayDeltaAttention(8, 2)
    layer.write_passes = 1
    control = torch.randn(2, 4, 8, requires_grad=True)
    query = torch.randn(2, 5, 8, requires_grad=True)
    order = torch.tensor([[3, 0, 2, 1], [1, 3, 0, 2]])
    writes = [layer._ordered(t, order) for t in layer._project_writes(control)]
    expected = layer._read(query, delta_final_state(*writes))
    actual = layer.cross(query, control, order=order)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (query, control), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (query, control)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)


def test_current_config_only_changes_scan_passes_and_preserves_teacher() -> None:
    root = Path(__file__).resolve().parents[2] / "configs/v2"
    old = load_experiment_config(root / "relay_jurkat/gradpert_v2/nadig_jurkat.yaml")
    new = load_experiment_config(root / "single_pass_jurkat/gradpert_v2/nadig_jurkat.yaml")
    old_arch, old_options = V2Options.parse_parameters(old.model.parameters)
    new_arch, new_options = V2Options.parse_parameters(new.model.parameters)
    assert replace(new_arch, relay_passes=2) == old_arch
    assert new_options == old_options
    assert "relay_passes" not in old_arch.payload()
    assert new_arch.payload()["relay_passes"] == 1
    arch = replace(
        new_arch,
        width=8,
        heads=2,
        latent_rank=4,
        streams=1,
        projector_hidden=12,
        projector_bottleneck=4,
        prototypes=11,
    )
    model = GraDPertV2(torch.randn(9, 8), arch)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    old_model = GraDPertV2(torch.randn(9, 8), replace(arch, relay_passes=2))
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in old_model.parameters()
    )
    for network in (model, teacher):
        layers = [m for m in network.modules() if isinstance(m, RelayDeltaAttention)]
        assert len(layers) == 9  # graph3 + cell2 + response(self2,cross2)
        assert all(m.write_passes == 1 for m in layers)


@pytest.mark.parametrize("value", [0, 3, True, 1.0, "1"])
def test_invalid_scan_passes_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="relay_passes"):
        V2Architecture.parse({"relay_passes": value})
