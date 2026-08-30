"""Config-selectable native graph encoders for the B2-vNext experiment surface.

The implementations in this module are native GraD-Pert components.  They use
public graph-model behavior as alignment evidence, but do not import or call an
upstream model package.  Graph construction remains outside the model: callers
provide ordered, already-pruned source tensors and receive one state per node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StringWeightMode(str, Enum):
    """Receipt-stable STRING numerical-weight routes."""

    SELECTION_ONLY = "selection_only"
    NORMALIZED_EDGE_FEATURE = "normalized_edge_feature"
    FIXED_NORMALIZED_PRIOR = "fixed_normalized_prior"
    PRIOR_LOGIT_RESIDUAL = "prior_logit_residual"
    SHUFFLED_NORMALIZED_EDGE_FEATURE = "shuffled_normalized_edge_feature"


@dataclass(frozen=True)
class GraphSourceTensors:
    """One ordered graph source over a shared node axis."""

    name: str
    edge_index: Tensor
    edge_weight: Tensor | None = None


@dataclass(frozen=True)
class SparseUnionTensors:
    """Deterministic sparse-attention union and its source-membership channels."""

    edge_index: Tensor
    edge_membership: Tensor
    local_edge_index: Tensor
    channel_names: tuple[str, ...]


OrderedGraphSourcePairs = tuple[tuple[str, tuple[tuple[int, int], ...]], ...]


def _validate_sources(
    node_count: int,
    sources: tuple[GraphSourceTensors, ...],
    expected_names: tuple[str, ...],
) -> None:
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    names = tuple(source.name for source in sources)
    if names != expected_names:
        raise ValueError(f"ordered graph sources must be {expected_names}, received {names}")
    if len(set(names)) != len(names):
        raise ValueError("graph source names must be unique")
    source_devices = {source.edge_index.device for source in sources}
    if len(source_devices) != 1:
        raise ValueError("all graph source edge tensors must share one device")
    for source in sources:
        if source.edge_index.dtype != torch.long:
            raise ValueError(f"{source.name} edge_index must use torch.long")
        if source.edge_index.ndim != 2 or source.edge_index.shape[0] != 2:
            raise ValueError(f"{source.name} edge_index must have shape [2, E]")
        if source.edge_index.numel() > 0:
            minimum = int(source.edge_index.min().item())
            maximum = int(source.edge_index.max().item())
            if minimum < 0 or maximum >= node_count:
                raise ValueError(f"{source.name} edge index is outside the shared node axis")
        if source.edge_weight is not None:
            if source.edge_weight.ndim not in {1, 2}:
                raise ValueError(f"{source.name} edge_weight must have shape [E] or [E, 1]")
            if source.edge_weight.ndim == 2 and source.edge_weight.shape[1] != 1:
                raise ValueError(f"{source.name} edge_weight second dimension must be 1")
            if source.edge_weight.shape[0] != source.edge_index.shape[1]:
                raise ValueError(f"{source.name} edge count and weight count differ")
            if not bool(torch.isfinite(source.edge_weight).all().item()):
                raise ValueError(f"{source.name} edge weights must be finite")


def _stable_segment_softmax(logits: Tensor, target: Tensor, node_count: int) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("attention logits must have shape [E, H]")
    if logits.shape[0] == 0:
        return logits
    index = target.view(-1, 1).expand_as(logits)
    maxima = torch.full(
        (node_count, logits.shape[1]),
        -torch.inf,
        device=logits.device,
        dtype=logits.dtype,
    )
    maxima.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
    stabilized = logits - maxima.index_select(0, target)
    numerators = stabilized.exp()
    denominators = torch.zeros_like(maxima)
    denominators.index_add_(0, target, numerators)
    return numerators / denominators.index_select(0, target).clamp_min(1e-12)


def _add_explicit_self_loops(
    edge_index: Tensor,
    edge_weight: Tensor | None,
    node_count: int,
) -> tuple[Tensor, Tensor | None]:
    nonself = edge_index[0] != edge_index[1]
    filtered_edges = edge_index[:, nonself]
    loops = torch.arange(node_count, device=edge_index.device, dtype=torch.long).repeat(2, 1)
    result_edges = torch.cat((filtered_edges, loops), dim=1)
    if edge_weight is None:
        return result_edges, None
    flattened = edge_weight.reshape(-1)
    loop_weight = torch.ones(node_count, device=edge_weight.device, dtype=edge_weight.dtype)
    return result_edges, torch.cat((flattened[nonself], loop_weight), dim=0)


def _prepare_normalized_string_weights(
    edge_index: Tensor,
    edge_weight: Tensor,
    *,
    shuffle_nonself: bool,
) -> Tensor:
    """Mirror the frozen official global STRING normalization before GAT use."""

    flattened = edge_weight.reshape(-1)
    nonself = edge_index[0] != edge_index[1]
    if not bool(nonself.any().item()):
        raise ValueError("normalized STRING routes require at least one non-self edge")
    maximum = flattened[nonself].max()
    if not bool(torch.isfinite(maximum).item()) or float(maximum.item()) <= 0:
        raise ValueError("normalized STRING routes require a positive finite maximum weight")
    normalized = (flattened / maximum).abs()
    if not shuffle_nonself:
        return normalized
    indices = torch.nonzero(nonself, as_tuple=False).reshape(-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260828)
    permutation = torch.randperm(indices.shape[0], generator=generator).to(indices.device)
    result = normalized.clone()
    result.index_copy_(
        0,
        indices,
        normalized.index_select(0, indices.index_select(0, permutation)),
    )
    return result


class NativeGraphEncoder(nn.Module, ABC):
    """Common node-state interface used by every native graph backend."""

    source_names: tuple[str, ...]
    output_dim: int

    @abstractmethod
    def forward(
        self,
        node_inputs: Tensor,
        sources: tuple[GraphSourceTensors, ...],
    ) -> Tensor:
        """Encode the shared node axis as a dense ``[N, output_dim]`` tensor."""


class _NativeGATv2Layer(nn.Module):
    """GATv2 message passing with explicit, mutually exclusive prior routes."""

    def __init__(
        self,
        *,
        input_dim: int,
        head_dim: int,
        head_count: int,
        concat: bool,
        dropout: float,
        add_self_loops: bool,
        weight_mode: StringWeightMode,
    ) -> None:
        super().__init__()
        if min(input_dim, head_dim, head_count) <= 0:
            raise ValueError("GAT dimensions and head count must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.head_dim = head_dim
        self.head_count = head_count
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.weight_mode = weight_mode
        projected_dim = head_dim * head_count
        self.source_projection = nn.Linear(input_dim, projected_dim, bias=False)
        self.target_projection = nn.Linear(input_dim, projected_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(1, head_count, head_dim))
        self.edge_projection = (
            nn.Linear(1, projected_dim, bias=False)
            if weight_mode
            in {
                StringWeightMode.NORMALIZED_EDGE_FEATURE,
                StringWeightMode.SHUFFLED_NORMALIZED_EDGE_FEATURE,
            }
            else None
        )
        output_width = projected_dim if concat else head_dim
        self.bias = nn.Parameter(torch.empty(output_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.source_projection.weight)
        nn.init.xavier_uniform_(self.target_projection.weight)
        nn.init.xavier_uniform_(self.attention)
        if self.edge_projection is not None:
            nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        node_inputs: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        node_count = node_inputs.shape[0]
        if edge_weight is not None:
            edge_weight = edge_weight.reshape(-1).to(
                device=node_inputs.device, dtype=node_inputs.dtype
            )
        if self.add_self_loops:
            edge_index, edge_weight = _add_explicit_self_loops(edge_index, edge_weight, node_count)
        mode = self.weight_mode
        if mode is not StringWeightMode.SELECTION_ONLY and edge_weight is None:
            raise ValueError(f"STRING weight mode {mode.value} requires edge weights")
        if (
            mode
            in {
                StringWeightMode.FIXED_NORMALIZED_PRIOR,
                StringWeightMode.PRIOR_LOGIT_RESIDUAL,
            }
            and edge_weight is not None
            and bool((edge_weight < 0).any().item())
        ):
            raise ValueError(f"STRING weight mode {mode.value} requires nonnegative weights")

        source = edge_index[0]
        target = edge_index[1]
        source_state = self.source_projection(node_inputs).view(
            node_count, self.head_count, self.head_dim
        )
        target_state = self.target_projection(node_inputs).view(
            node_count, self.head_count, self.head_dim
        )
        pair_state = source_state.index_select(0, source) + target_state.index_select(0, target)
        if self.edge_projection is not None:
            if edge_weight is None:
                raise RuntimeError("validated numerical edge weights are unavailable")
            edge_state = self.edge_projection(edge_weight.view(-1, 1)).view(
                -1, self.head_count, self.head_dim
            )
            pair_state = pair_state + edge_state
        learned_logits = (F.leaky_relu(pair_state, negative_slope=0.2) * self.attention).sum(dim=-1)

        if mode is StringWeightMode.FIXED_NORMALIZED_PRIOR:
            if edge_weight is None:
                raise RuntimeError("validated normalized prior is unavailable")
            logits = edge_weight.clamp_min(1e-12).log().view(-1, 1).expand_as(learned_logits)
        elif mode is StringWeightMode.PRIOR_LOGIT_RESIDUAL:
            if edge_weight is None:
                raise RuntimeError("validated normalized prior is unavailable")
            logits = learned_logits + edge_weight.clamp_min(1e-12).log().view(-1, 1)
        else:
            logits = learned_logits

        coefficients = _stable_segment_softmax(logits, target, node_count)
        coefficients = F.dropout(coefficients, p=self.dropout, training=self.training)
        messages = source_state.index_select(0, source) * coefficients.unsqueeze(-1)
        aggregated = torch.zeros(
            (node_count, self.head_count, self.head_dim),
            device=node_inputs.device,
            dtype=node_inputs.dtype,
        )
        aggregated.index_add_(0, target, messages)
        output = aggregated.reshape(node_count, -1) if self.concat else aggregated.mean(dim=1)
        return output + self.bias


class SingleSourceGATEncoder(NativeGraphEncoder):
    """Native single-source GATv2 tower with explicit STRING prior semantics."""

    def __init__(
        self,
        *,
        source_name: str = "string",
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
        layer_count: int = 4,
        head_count: int = 2,
        concat_heads: bool = True,
        skip_concat: bool = True,
        dropout: float = 0.2,
        add_self_loops: bool = True,
        string_weight_mode: StringWeightMode | str = StringWeightMode.SELECTION_ONLY,
        prepare_string_weights: bool = True,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim, layer_count, head_count) <= 0:
            raise ValueError("encoder dimensions, layers, and heads must be positive")
        mode = StringWeightMode(string_weight_mode)
        if source_name != "string" and mode is not StringWeightMode.SELECTION_ONLY:
            raise ValueError("numerical weight routes are supported only for STRING")
        self.source_names = (source_name,)
        self.output_dim = output_dim
        self.dropout = dropout
        self.skip_concat = skip_concat
        self.string_weight_mode = mode
        self.prepare_string_weights = prepare_string_weights
        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        width = input_dim
        for _ in range(layer_count):
            self.layers.append(
                _NativeGATv2Layer(
                    input_dim=width,
                    head_dim=hidden_dim,
                    head_count=head_count,
                    concat=concat_heads,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    weight_mode=mode,
                )
            )
            main_width = hidden_dim * head_count if concat_heads else hidden_dim
            if skip_concat:
                self.skips.append(nn.Linear(width, main_width, bias=True))
                width = main_width * 2
            else:
                self.skips.append(nn.Identity())
                width = main_width
        self.output = nn.Linear(width, output_dim)

    def forward(
        self,
        node_inputs: Tensor,
        sources: tuple[GraphSourceTensors, ...],
    ) -> Tensor:
        if node_inputs.ndim != 2:
            raise ValueError("node_inputs must have shape [N, D]")
        _validate_sources(node_inputs.shape[0], sources, self.source_names)
        source = sources[0]
        edge_index = source.edge_index.to(device=node_inputs.device)
        edge_weight = source.edge_weight
        if (
            self.prepare_string_weights
            and self.string_weight_mode is not StringWeightMode.SELECTION_ONLY
        ):
            if edge_weight is None:
                raise ValueError(
                    f"STRING weight mode {self.string_weight_mode.value} requires edge weights"
                )
            edge_weight = _prepare_normalized_string_weights(
                edge_index,
                edge_weight,
                shuffle_nonself=(
                    self.string_weight_mode is StringWeightMode.SHUFFLED_NORMALIZED_EDGE_FEATURE
                ),
            )
        state = node_inputs
        for layer, skip in zip(self.layers, self.skips, strict=True):
            residual = state
            state = layer(state, edge_index, edge_weight)
            if self.skip_concat:
                state = torch.cat((state, skip(residual)), dim=-1)
            state = F.leaky_relu(state, negative_slope=0.2)
            state = F.dropout(state, p=self.dropout, training=self.training)
        state = self.output(state)
        state = F.leaky_relu(state, negative_slope=0.2)
        return F.dropout(state, p=self.dropout, training=self.training)


def _edge_pairs(edge_index: Tensor) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(source), int(target))
        for source, target in zip(
            edge_index[0].detach().cpu().tolist(),
            edge_index[1].detach().cpu().tolist(),
            strict=True,
        )
    )


def _is_undirected(pairs: tuple[tuple[int, int], ...]) -> bool:
    pair_set = set(pairs)
    return all((target, source) in pair_set for source, target in pair_set)


def _expander_pairs(node_count: int, degree: int) -> tuple[tuple[int, int], ...]:
    if degree < 0:
        raise ValueError("expander_degree must be nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    result: list[tuple[int, int]] = []
    for _ in range(degree):
        permutation = torch.randperm(node_count, generator=generator).tolist()
        for index, node in enumerate(permutation):
            previous = permutation[(index - 1) % node_count]
            result.extend(((node, previous), (previous, node)))
    return tuple(result)


def build_sparse_union(
    *,
    node_count: int,
    sources: tuple[GraphSourceTensors, ...],
    expected_names: tuple[str, ...],
    add_reverse_edges: bool = True,
    add_self_loops: bool = True,
    expander_degree: int = 3,
) -> SparseUnionTensors:
    """Build a deterministic union with explicit source/reverse/global channels."""

    _validate_sources(node_count, sources, expected_names)
    channels: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for source in sources:
        pairs = _edge_pairs(source.edge_index)
        channels.append((source.name, pairs))
        if add_reverse_edges and not _is_undirected(pairs):
            channels.append((f"{source.name}:reverse", tuple((b, a) for a, b in pairs)))
    if add_self_loops:
        channels.append(("self", tuple((node, node) for node in range(node_count))))
    if expander_degree > 0:
        channels.append(("expander", _expander_pairs(node_count, expander_degree)))
    if not channels:
        raise ValueError("sparse graph Transformer requires at least one edge channel")

    edge_to_membership: dict[tuple[int, int], list[float]] = {}
    for channel_index, (_, pairs) in enumerate(channels):
        for pair in pairs:
            edge_channels = edge_to_membership.setdefault(pair, [0.0] * len(channels))
            edge_channels[channel_index] = 1.0
    ordered_edges = sorted(edge_to_membership)
    if not ordered_edges:
        raise ValueError("sparse graph Transformer requires at least one materialized edge")
    device = sources[0].edge_index.device
    edge_index = torch.tensor(ordered_edges, device=device, dtype=torch.long).t().contiguous()
    membership_tensor = torch.tensor(
        [edge_to_membership[pair] for pair in ordered_edges],
        device=device,
        dtype=torch.float32,
    )
    return SparseUnionTensors(
        edge_index=edge_index,
        edge_membership=membership_tensor,
        local_edge_index=sources[0].edge_index,
        channel_names=tuple(name for name, _ in channels),
    )


def build_sparse_union_from_ordered_pairs(
    *,
    node_count: int,
    sources: OrderedGraphSourcePairs,
    expected_names: tuple[str, ...],
    device: torch.device,
    add_reverse_edges: bool = True,
    add_self_loops: bool = True,
    expander_degree: int = 3,
) -> SparseUnionTensors:
    """Build the exact sparse union without a device-to-host edge round trip."""

    if node_count <= 0:
        raise ValueError("node_count must be positive")
    names = tuple(name for name, _ in sources)
    if names != expected_names:
        raise ValueError(f"ordered graph sources must be {expected_names}, received {names}")
    if len(set(names)) != len(names):
        raise ValueError("graph source names must be unique")
    if not sources:
        raise ValueError("sparse graph Transformer requires at least one source")
    for source_name, pairs in sources:
        if any(
            source < 0 or source >= node_count or target < 0 or target >= node_count
            for source, target in pairs
        ):
            raise ValueError(f"{source_name} edge index is outside the shared node axis")

    channels: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for source_name, pairs in sources:
        channels.append((source_name, pairs))
        if add_reverse_edges and not _is_undirected(pairs):
            channels.append((f"{source_name}:reverse", tuple((b, a) for a, b in pairs)))
    if add_self_loops:
        channels.append(("self", tuple((node, node) for node in range(node_count))))
    if expander_degree > 0:
        channels.append(("expander", _expander_pairs(node_count, expander_degree)))
    if not channels:
        raise ValueError("sparse graph Transformer requires at least one edge channel")

    pair_arrays: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    channel_arrays: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    for channel_index, (_, pairs) in enumerate(channels):
        pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        pair_arrays.append(pair_array)
        channel_arrays.append(np.full(pair_array.shape[0], channel_index, dtype=np.int64))
    all_pairs = np.concatenate(pair_arrays, axis=0)
    all_channels = np.concatenate(channel_arrays, axis=0)
    if all_pairs.shape[0] == 0:
        raise ValueError("sparse graph Transformer requires at least one materialized edge")
    order = np.lexsort((all_pairs[:, 1], all_pairs[:, 0]))
    sorted_pairs = all_pairs[order]
    sorted_channels = all_channels[order]
    unique_start = np.ones(sorted_pairs.shape[0], dtype=np.bool_)
    unique_start[1:] = np.any(sorted_pairs[1:] != sorted_pairs[:-1], axis=1)
    group_ids = np.cumsum(unique_start, dtype=np.int64) - 1
    ordered_edges = np.ascontiguousarray(sorted_pairs[unique_start])
    membership = np.zeros((ordered_edges.shape[0], len(channels)), dtype=np.float32)
    membership[group_ids, sorted_channels] = 1.0
    first_source_pairs = np.asarray(sources[0][1], dtype=np.int64).reshape(-1, 2)

    edge_index = torch.as_tensor(ordered_edges, device=device, dtype=torch.long).t().contiguous()
    membership_tensor = torch.as_tensor(membership, device=device, dtype=torch.float32)
    local_edge_index = (
        torch.as_tensor(first_source_pairs, device=device, dtype=torch.long).t().contiguous()
    )
    return SparseUnionTensors(
        edge_index=edge_index,
        edge_membership=membership_tensor,
        local_edge_index=local_edge_index,
        channel_names=tuple(name for name, _ in channels),
    )


class _SparseGraphTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        head_count: int,
        dropout: float,
        add_local_message_passing: bool,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or head_count <= 0 or hidden_dim % head_count != 0:
            raise ValueError("hidden_dim must be positive and divisible by head_count")
        self.hidden_dim = hidden_dim
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count
        self.dropout = dropout
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(hidden_dim, head_count, bias=True)
        self.attention_norm = nn.BatchNorm1d(hidden_dim)
        self.add_local_message_passing = add_local_message_passing
        self.local_layer = (
            _NativeGATv2Layer(
                input_dim=hidden_dim,
                head_dim=hidden_dim,
                head_count=2,
                concat=False,
                dropout=dropout,
                add_self_loops=True,
                weight_mode=StringWeightMode.SELECTION_ONLY,
            )
            if add_local_message_passing
            else None
        )
        self.feed_forward_1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.feed_forward_2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.feed_forward_norm = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        node_states: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        local_edge_index: Tensor,
    ) -> Tensor:
        node_count = node_states.shape[0]
        source = edge_index[0]
        target = edge_index[1]
        query = self.query(node_states).view(node_count, self.head_count, self.head_dim)
        key = self.key(node_states).view(node_count, self.head_count, self.head_dim)
        value = self.value(node_states).view(node_count, self.head_count, self.head_dim)
        edge = self.edge(edge_features).view(-1, self.head_count, self.head_dim)
        edge_bias = self.edge_bias(edge_features)
        score = (key.index_select(0, source) * query.index_select(0, target) * edge).sum(
            dim=-1
        ) / sqrt(self.head_dim)
        score = (score + edge_bias).clamp(-5.0, 5.0).exp()
        messages = value.index_select(0, source) * score.unsqueeze(-1)
        weighted_value = torch.zeros_like(value)
        weighted_value.index_add_(0, target, messages)
        normalizer = torch.zeros(
            (node_count, self.head_count),
            device=node_states.device,
            dtype=node_states.dtype,
        )
        normalizer.index_add_(0, target, score)
        attention = weighted_value / normalizer.clamp_min(1e-6).unsqueeze(-1)
        attention = attention.reshape(node_count, self.hidden_dim)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        state = self.attention_norm(node_states + attention)
        if self.local_layer is not None:
            local = self.local_layer(node_states, local_edge_index, None)
            state = state + local
        feed_forward = self.feed_forward_2(
            F.dropout(
                F.relu(self.feed_forward_1(state)),
                p=self.dropout,
                training=self.training,
            )
        )
        feed_forward = F.dropout(feed_forward, p=self.dropout, training=self.training)
        return cast(Tensor, self.feed_forward_norm(state + feed_forward))


class SparseGraphTransformerEncoder(NativeGraphEncoder):
    """Native single- or multi-source sparse graph Transformer encoder."""

    def __init__(
        self,
        *,
        source_names: tuple[str, ...],
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
        layer_count: int = 4,
        head_count: int = 2,
        dropout: float = 0.1,
        add_reverse_edges: bool = True,
        add_self_loops: bool = True,
        expander_degree: int = 3,
        add_local_message_passing: bool = True,
        string_weight_mode: StringWeightMode | str = StringWeightMode.SELECTION_ONLY,
    ) -> None:
        super().__init__()
        if not source_names or len(set(source_names)) != len(source_names):
            raise ValueError("source_names must be a nonempty ordered unique tuple")
        if min(input_dim, hidden_dim, output_dim, layer_count, head_count) <= 0:
            raise ValueError("encoder dimensions, layers, and heads must be positive")
        mode = StringWeightMode(string_weight_mode)
        if mode is not StringWeightMode.SELECTION_ONLY:
            raise ValueError("sparse graph Transformer supports only selection_only STRING weights")
        self.source_names = source_names
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.add_reverse_edges = add_reverse_edges
        self.add_self_loops = add_self_loops
        self.expander_degree = expander_degree
        self.input_projection = (
            nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        )
        expected_channels: list[str] = []
        for source_name in source_names:
            expected_channels.append(source_name)
            if add_reverse_edges:
                expected_channels.append(f"{source_name}:reverse")
        if add_self_loops:
            expected_channels.append("self")
        if expander_degree > 0:
            expected_channels.append("expander")
        self.edge_channel_names = tuple(expected_channels)
        maximum_channel_count = len(self.edge_channel_names)
        self.edge_source_projection = nn.Linear(maximum_channel_count, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                _SparseGraphTransformerLayer(
                    hidden_dim=hidden_dim,
                    head_count=head_count,
                    dropout=dropout,
                    add_local_message_passing=add_local_message_passing,
                )
                for _ in range(layer_count)
            ]
        )
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def _encode_union(
        self,
        node_inputs: Tensor,
        union: SparseUnionTensors,
    ) -> Tensor:
        if node_inputs.ndim != 2:
            raise ValueError("node_inputs must have shape [N, D]")
        membership = torch.zeros(
            (union.edge_membership.shape[0], len(self.edge_channel_names)),
            device=node_inputs.device,
            dtype=node_inputs.dtype,
        )
        fixed_channel = {name: index for index, name in enumerate(self.edge_channel_names)}
        for union_index, channel_name in enumerate(union.channel_names):
            try:
                destination = fixed_channel[channel_name]
            except KeyError as error:
                raise RuntimeError(
                    f"sparse union channel {channel_name!r} is outside the encoder contract"
                ) from error
            membership[:, destination] = union.edge_membership[:, union_index].to(
                device=node_inputs.device, dtype=node_inputs.dtype
            )
        edge_features = self.edge_source_projection(membership)
        state = self.input_projection(node_inputs)
        edge_index = union.edge_index.to(device=node_inputs.device)
        local_edge_index = union.local_edge_index.to(device=node_inputs.device)
        for layer in self.layers:
            state = layer(state, edge_index, edge_features, local_edge_index)
        state = F.leaky_relu(self.output(state), negative_slope=0.2)
        return F.dropout(state, p=self.dropout, training=self.training)

    def forward_union(self, node_inputs: Tensor, union: SparseUnionTensors) -> Tensor:
        """Encode a prebuilt disconnected-view union with fixed channel names."""

        return self._encode_union(node_inputs, union)

    def forward(
        self,
        node_inputs: Tensor,
        sources: tuple[GraphSourceTensors, ...],
    ) -> Tensor:
        union = build_sparse_union(
            node_count=node_inputs.shape[0],
            sources=sources,
            expected_names=self.source_names,
            add_reverse_edges=self.add_reverse_edges,
            add_self_loops=self.add_self_loops,
            expander_degree=self.expander_degree,
        )
        return self._encode_union(node_inputs, union)


class AdaptiveSourceGATEncoder(NativeGraphEncoder):
    """GraD-Pert-native per-source GAT towers with node-adaptive fusion.

    This route is deliberately identified as a project-preregistered native
    architecture.  It does not claim parity with an external hybrid model.
    """

    def __init__(
        self,
        *,
        source_names: tuple[str, ...],
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
        layer_count: int = 4,
        head_count: int = 2,
        dropout: float = 0.2,
        string_weight_mode: StringWeightMode | str = StringWeightMode.SELECTION_ONLY,
    ) -> None:
        super().__init__()
        if len(source_names) < 2 or len(set(source_names)) != len(source_names):
            raise ValueError("adaptive source fusion requires at least two unique sources")
        self.source_names = source_names
        self.output_dim = output_dim
        mode = StringWeightMode(string_weight_mode)
        self.towers = nn.ModuleDict(
            {
                source_name: SingleSourceGATEncoder(
                    source_name=source_name,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    layer_count=layer_count,
                    head_count=head_count,
                    dropout=dropout,
                    string_weight_mode=(
                        mode if source_name == "string" else StringWeightMode.SELECTION_ONLY
                    ),
                )
                for source_name in source_names
            }
        )
        self.source_queries = nn.ParameterDict(
            {source_name: nn.Parameter(torch.empty(1, output_dim)) for source_name in source_names}
        )
        for query in self.source_queries.values():
            nn.init.trunc_normal_(query, std=0.02)

    def forward(
        self,
        node_inputs: Tensor,
        sources: tuple[GraphSourceTensors, ...],
    ) -> Tensor:
        if node_inputs.ndim != 2:
            raise ValueError("node_inputs must have shape [N, D]")
        _validate_sources(node_inputs.shape[0], sources, self.source_names)
        states: list[Tensor] = []
        scores: list[Tensor] = []
        for source in sources:
            state = self.towers[source.name](node_inputs, (source,))
            states.append(state)
            scores.append((state * self.source_queries[source.name]).sum(dim=-1))
        stacked_states = torch.stack(states, dim=1)
        weights = torch.softmax(torch.stack(scores, dim=1), dim=1).unsqueeze(-1)
        return (stacked_states * weights).sum(dim=1)
