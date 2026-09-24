"""The new graph topology and selected-node propagation are independently checked."""

from pathlib import Path

import numpy as np
import torch

from gradpert.config import load_experiment_config
from gradpert.config.v2 import V2Architecture, V2Options
from gradpert.graphs import GraphTopology
from gradpert.graphs.pruning import PrunedSourceGraph
from gradpert.modeling.v2 import GraDPertV2
from gradpert.modeling.v2.operators import DeltaAttention, IndexedLatentAttention, LatentAttention
from gradpert.training.v2.views import NeighborhoodIndex


def topology(n: int = 9) -> GraphTopology:
    genes = tuple(f"g{i}" for i in range(n))
    return GraphTopology(
        genes,
        {name: PrunedSourceGraph(name, n, genes, (), 20) for name in ("go", "string")},
    )


def test_hamiltonian_cycles_are_fixed_bidirectional_and_deduplicated():
    first = NeighborhoodIndex(topology(), 3, 7, expander_type="hamiltonian")
    second = NeighborhoodIndex(topology(), 3, 7, expander_type="hamiltonian")
    assert first.rows == second.rows
    for query, row in enumerate(first.rows):
        expander = {memory for memory, sources in row.items() if sources[2]}
        assert 2 <= len(expander) <= 6
        assert query not in expander
        assert row[query][3]
        for memory in expander:
            assert first.rows[memory][query][2]


def test_two_layer_selected_read_matches_full_synchronous_propagation():
    torch.manual_seed(9)
    options = V2Architecture(
        width=8,
        heads=2,
        latent_rank=4,
        streams=2,
        graph_read_mode="propagated",
        dropout=0,
        projector_hidden=12,
        projector_bottleneck=4,
        prototypes=11,
        checkpoint_layers=False,
    )
    model = GraDPertV2(torch.randn(9, 8), options).eval()
    index = NeighborhoodIndex(topology(), 1, 7, expander_type="hamiltonian", propagated=True)
    selected = np.array([0, 2], dtype=np.int64)
    partial = index.view(
        selected, [(0,)], rng=np.random.default_rng(3), device=torch.device("cpu"), induced=False
    )
    full = index.view(
        np.arange(9),
        [(0,)],
        rng=np.random.default_rng(3),
        device=torch.device("cpu"),
        induced=False,
    )
    assert partial.context is not None
    actual = model.graph(
        partial.ids,
        partial.neighbors,
        partial.valid,
        partial.sources,
        context=partial.context,
    )
    memory = model.graph.norm(model.graph.adapter(model.graph.embedding.weight))
    first = model.graph.layers[0](
        memory[full.ids], memory, full.neighbors, full.valid, full.sources
    )
    full_neighbor_positions = full.neighbors
    expected = model.graph.layers[1](
        first, first, full_neighbor_positions, full.valid, full.sources
    )[selected]
    torch.testing.assert_close(actual, expected)
    masked = torch.tensor([0])
    actual_masked = model.graph(
        partial.ids,
        partial.neighbors,
        partial.valid,
        partial.sources,
        masked,
        partial.context,
    )
    masked_memory = memory.index_copy(0, masked, model.graph.mask_token[None])
    first_masked = model.graph.layers[0](
        masked_memory[full.ids], masked_memory, full.neighbors, full.valid, full.sources
    )
    expected_masked = model.graph.layers[1](
        first_masked, first_masked, full_neighbor_positions, full.valid, full.sources
    )[selected]
    torch.testing.assert_close(actual_masked, expected_masked)


def test_new_profile_has_two_kda_one_full_mla_no_dsa():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/v2/hamiltonian_mla_jurkat/gradpert_v2/nadig_jurkat.yaml"
    )
    config = load_experiment_config(path)
    arch, options = V2Options.parse_parameters(config.model.parameters)
    assert (arch.graph_layers, arch.graph_read_mode) == (2, "propagated")
    assert (options.graph_expander_type, options.graph_expander_degree) == ("hamiltonian", 3)
    assert (arch.kda_layers, arch.attention) == (2, "hybrid")
    model = GraDPertV2(torch.randn(9, 256), arch)
    for encoder in (model.cell, model.response):
        kinds = [type(layer.sublayer) for layer in encoder.layers[::2]]
        assert kinds == [DeltaAttention, DeltaAttention, LatentAttention]
        assert IndexedLatentAttention not in kinds
