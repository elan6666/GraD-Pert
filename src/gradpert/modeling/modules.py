"""Native graph encoder, consistency head, and additive expression predictor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv  # type: ignore[import-untyped]

from gradpert.config.native import NativeArchitectureOptions
from gradpert.graphs import GraphView
from gradpert.modeling.encoders import (
    AdaptiveSourceGATEncoder,
    GraphSourceTensors,
    NativeGraphEncoder,
    SingleSourceGATEncoder,
    SparseGraphTransformerEncoder,
    SparseUnionTensors,
    StringWeightMode,
    _prepare_normalized_string_weights,
    build_sparse_union,
)


@dataclass(frozen=True)
class EncodedGraphView:
    node_ids: tuple[int, ...]
    node_states: Tensor

    def node_states_for(self, node_ids: tuple[int, ...]) -> Tensor:
        local_by_global = {node_id: index for index, node_id in enumerate(self.node_ids)}
        try:
            local_ids = [local_by_global[node_id] for node_id in node_ids]
        except KeyError as error:
            raise ValueError(
                f"requested node is absent from graph view: {error.args[0]}"
            ) from error
        index = torch.as_tensor(local_ids, device=self.node_states.device, dtype=torch.long)
        return self.node_states.index_select(0, index)

    def condition_state(self, anchor_ids: tuple[int, ...]) -> Tensor:
        if not anchor_ids:
            raise ValueError("condition requires at least one active anchor")
        return self.node_states_for(anchor_ids).sum(dim=0)


class _GraphAttentionTower(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 128,
        layer_count: int = 4,
        head_count: int = 2,
        head_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if (input_dim, layer_count, head_count, head_dim, output_dim, dropout) != (
            128,
            4,
            2,
            128,
            64,
            0.1,
        ):
            raise ValueError("v1 graph tower dimensions/dropout are frozen")
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        width = input_dim
        for _ in range(layer_count):
            main_width = head_count * head_dim
            self.layers.append(
                GATv2Conv(
                    in_channels=width,
                    out_channels=head_dim,
                    heads=head_count,
                    concat=True,
                    dropout=dropout,
                    negative_slope=0.2,
                    add_self_loops=False,
                    bias=True,
                )
            )
            self.skips.append(nn.Linear(width, main_width, bias=True))
            width = main_width * 2
        self.output = nn.Linear(width, output_dim)

    def forward(self, node_inputs: Tensor, edge_index: Tensor) -> Tensor:
        state = node_inputs
        for layer, skip in zip(self.layers, self.skips, strict=True):
            main = layer(state, edge_index)
            residual = skip(state)
            state = torch.cat((main, residual), dim=-1)
            state = F.leaky_relu(state, negative_slope=0.2)
            state = F.dropout(state, p=self.dropout, training=self.training)
        return cast(Tensor, self.output(state))


class AdaptiveGeneGraphEncoder(nn.Module):
    """Two source-specific towers with node-adaptive one-head fusion."""

    def __init__(self, n_genes: int) -> None:
        super().__init__()
        if n_genes <= 0:
            raise ValueError("n_genes must be positive")
        self.n_genes = n_genes
        self.gene_embeddings = nn.Parameter(torch.empty(n_genes, 128))
        self.mask_token = nn.Parameter(torch.empty(1, 128))
        self.towers = nn.ModuleDict(
            {
                "go": _GraphAttentionTower(),
                "string": _GraphAttentionTower(),
            }
        )
        self.relation_queries = nn.ParameterDict(
            {
                "go": nn.Parameter(torch.empty(1, 64)),
                "string": nn.Parameter(torch.empty(1, 64)),
            }
        )
        self.register_buffer("_resident_full_node_ids", None, persistent=False)
        self.register_buffer("_resident_prediction_go", None, persistent=False)
        self.register_buffer("_resident_prediction_string", None, persistent=False)
        self._resident_full_node_ids_tuple: tuple[int, ...] | None = None
        self._resident_prediction_view: GraphView | None = None
        self.reset_parameters()

    def configure_resident_graph_tensors(self, prediction_view: GraphView) -> None:
        """Materialize only immutable full-node and prediction-edge tensors once."""

        expected_nodes = tuple(range(self.n_genes))
        if (
            prediction_view.view_id != "prediction"
            or prediction_view.node_ids != expected_nodes
            or prediction_view.masked_node_ids
            or prediction_view.masked_anchor_ids
        ):
            raise ValueError("resident graph tensors require the clean full prediction view")
        device = self.gene_embeddings.device
        self._resident_full_node_ids = torch.as_tensor(
            expected_nodes, device=device, dtype=torch.long
        )
        self._resident_prediction_go = torch.as_tensor(
            prediction_view.local_edge_index("go"), device=device, dtype=torch.long
        )
        self._resident_prediction_string = torch.as_tensor(
            prediction_view.local_edge_index("string"), device=device, dtype=torch.long
        )
        self._resident_full_node_ids_tuple = expected_nodes
        self._resident_prediction_view = prediction_view

    def configure_string_weight_contract(self, prediction_view: GraphView) -> None:
        """Legacy selection-only encoder has no numerical STRING route."""

        del prediction_view

    def resident_graph_tensor_payload(self) -> dict[str, object]:
        return {
            "active": self._resident_prediction_view is not None,
            "node_count": (
                int(self._resident_full_node_ids.numel())
                if self._resident_full_node_ids is not None
                else 0
            ),
            "go_edge_count": (
                int(self._resident_prediction_go.shape[1])
                if self._resident_prediction_go is not None
                else 0
            ),
            "string_edge_count": (
                int(self._resident_prediction_string.shape[1])
                if self._resident_prediction_string is not None
                else 0
            ),
        }

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.gene_embeddings, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for query in self.relation_queries.values():
            nn.init.trunc_normal_(query, std=0.02)

    def _node_inputs(self, view: GraphView) -> Tensor:
        node_ids = (
            self._resident_full_node_ids
            if self._resident_full_node_ids is not None
            and view.node_ids == self._resident_full_node_ids_tuple
            else torch.as_tensor(
                view.node_ids,
                device=self.gene_embeddings.device,
                dtype=torch.long,
            )
        )
        inputs = self.gene_embeddings.index_select(0, node_ids)
        masked = set(view.masked_node_ids) | set(view.masked_anchor_ids)
        if masked:
            local_by_global = {node_id: index for index, node_id in enumerate(view.node_ids)}
            local_mask_ids = [local_by_global[node_id] for node_id in sorted(masked)]
            inputs = inputs.clone()
            index = torch.as_tensor(
                local_mask_ids,
                device=inputs.device,
                dtype=torch.long,
            )
            inputs.index_copy_(0, index, self.mask_token.expand(len(local_mask_ids), -1))
        return inputs

    def forward_many(self, views: Sequence[GraphView]) -> tuple[EncodedGraphView, ...]:
        """Encode disjoint graph views in one accelerator launch group.

        Offsetting each view's local edge indices creates one disconnected graph,
        so message passing remains identical to independent encoder calls while
        avoiding Python and kernel-launch overhead for every condition-local view.
        """

        if not views:
            raise ValueError("at least one graph view is required")
        if any(
            node_id < 0 or node_id >= self.n_genes for view in views for node_id in view.node_ids
        ):
            raise ValueError("graph view node ID is outside embedding table")

        inputs_by_view = tuple(self._node_inputs(view) for view in views)
        node_counts = tuple(inputs.shape[0] for inputs in inputs_by_view)
        inputs = torch.cat(inputs_by_view, dim=0)
        source_states = []
        source_scores = []
        for source_name in ("go", "string"):
            edges = []
            node_offset = 0
            for view, node_count in zip(views, node_counts, strict=True):
                resident_edges = (
                    self._resident_prediction_go
                    if source_name == "go"
                    else self._resident_prediction_string
                )
                local_edges = (
                    resident_edges
                    if resident_edges is not None and view is self._resident_prediction_view
                    else torch.as_tensor(
                        view.local_edge_index(source_name),
                        device=inputs.device,
                        dtype=torch.long,
                    )
                )
                edges.append(local_edges + node_offset)
                node_offset += node_count
            edge_index = torch.cat(edges, dim=1)
            state = F.leaky_relu(
                self.towers[source_name](inputs, edge_index),
                negative_slope=0.2,
            )
            source_states.append(state)
            source_scores.append((state * self.relation_queries[source_name]).sum(dim=-1))
        stacked_states = torch.stack(source_states, dim=1)
        weights = torch.softmax(torch.stack(source_scores, dim=1), dim=1).unsqueeze(-1)
        fused = (stacked_states * weights).sum(dim=1)
        chunks = []
        start = 0
        for node_count in node_counts:
            chunks.append(fused[start : start + node_count])
            start += node_count
        return tuple(
            EncodedGraphView(node_ids=view.node_ids, node_states=states)
            for view, states in zip(views, chunks, strict=True)
        )

    def forward(self, view: GraphView) -> EncodedGraphView:
        if any(node_id < 0 or node_id >= self.n_genes for node_id in view.node_ids):
            raise ValueError("graph view node ID is outside embedding table")
        return self.forward_many((view,))[0]


def _string_weight_mode(label: str) -> StringWeightMode:
    labels = {
        "selection_only": StringWeightMode.SELECTION_ONLY,
        "edge_feature": StringWeightMode.NORMALIZED_EDGE_FEATURE,
        "fixed_prior": StringWeightMode.FIXED_NORMALIZED_PRIOR,
        "prior_residual": StringWeightMode.PRIOR_LOGIT_RESIDUAL,
        "shuffled_edge_feature": StringWeightMode.SHUFFLED_NORMALIZED_EDGE_FEATURE,
    }
    try:
        return labels[label]
    except KeyError as error:
        raise ValueError(f"unsupported STRING weight mode: {label}") from error


def _native_graph_backend(options: NativeArchitectureOptions) -> NativeGraphEncoder:
    string_weight_mode = _string_weight_mode(options.string_weight_mode)
    if options.graph_encoder_family == "single_source_gat":
        return SingleSourceGATEncoder(
            source_name="string",
            input_dim=options.graph_input_dim,
            hidden_dim=options.graph_hidden_dim,
            output_dim=options.graph_output_dim,
            layer_count=options.graph_layer_count,
            head_count=options.graph_head_count,
            dropout=options.graph_dropout,
            string_weight_mode=string_weight_mode,
            prepare_string_weights=False,
        )
    if options.graph_encoder_family in {
        "single_source_sparse_transformer",
        "multi_source_sparse_transformer",
    }:
        return SparseGraphTransformerEncoder(
            source_names=options.graph_sources,
            add_reverse_edges=options.graph_add_reverse_edges,
            add_self_loops=options.graph_add_self_loops,
            expander_degree=options.graph_expander_degree,
            add_local_message_passing=options.graph_first_source_local_branch,
            input_dim=options.graph_input_dim,
            hidden_dim=options.graph_hidden_dim,
            output_dim=options.graph_output_dim,
            layer_count=options.graph_layer_count,
            head_count=options.graph_head_count,
            dropout=options.graph_dropout,
            string_weight_mode=string_weight_mode,
        )
    if options.graph_encoder_family == "adaptive_source_gat_fusion":
        return AdaptiveSourceGATEncoder(
            source_names=options.graph_sources,
            input_dim=options.graph_input_dim,
            hidden_dim=options.graph_hidden_dim,
            output_dim=options.graph_output_dim,
            layer_count=options.graph_layer_count,
            head_count=options.graph_head_count,
            dropout=options.graph_dropout,
            string_weight_mode=string_weight_mode,
        )
    raise ValueError(f"unsupported configurable encoder: {options.graph_encoder_family}")


class ConfigurableGeneGraphEncoder(nn.Module):
    """Config-selected node features plus one native graph backend."""

    def __init__(
        self,
        n_genes: int,
        options: NativeArchitectureOptions,
        *,
        genept_matrix: Tensor | None = None,
    ) -> None:
        super().__init__()
        if n_genes <= 0:
            raise ValueError("n_genes must be positive")
        self.n_genes = n_genes
        self.options = options
        self.gene_embeddings = nn.Parameter(torch.empty(n_genes, options.graph_input_dim))
        self.mask_token = nn.Parameter(torch.empty(1, options.graph_input_dim))
        self.genept_projection: nn.Linear | None = None
        self.register_buffer("_genept_matrix", None, persistent=False)
        if options.gene_feature_mode != "learned_id":
            if (
                genept_matrix is None
                or genept_matrix.ndim != 2
                or genept_matrix.shape[0] != n_genes
                or genept_matrix.shape[1] < 1
            ):
                raise ValueError("GenePT feature modes require an ordered finite [N,D] matrix")
            matrix = genept_matrix.detach().to(dtype=torch.float32).contiguous()
            if not bool(torch.isfinite(matrix).all().item()):
                raise ValueError("GenePT matrix must be finite")
            if options.gene_feature_mode == "genept_shuffled":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(20260828)
                order = torch.randperm(n_genes, generator=generator).to(matrix.device)
                matrix = matrix.index_select(0, order)
            self._genept_matrix = matrix
            if options.gene_feature_mode in {
                "frozen_genept_projection",
                "genept_id_residual",
                "genept_shuffled",
            }:
                self.genept_projection = nn.Linear(
                    int(matrix.shape[1]), options.graph_input_dim, bias=True
                )
            if options.gene_feature_mode in {
                "frozen_genept_projection",
                "genept_shuffled",
            }:
                self.gene_embeddings.requires_grad_(False)
        self.backend = _native_graph_backend(options)
        self._resident_prediction_view: GraphView | None = None
        self._resident_source_tensors: tuple[GraphSourceTensors, ...] | None = None
        self._resident_sparse_union: SparseUnionTensors | None = None
        self._string_weight_by_global_edge: dict[tuple[int, int], float] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.gene_embeddings, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if self.options.gene_feature_mode == "genept_initialized":
            if self._genept_matrix is None:
                raise RuntimeError("GenePT initialization matrix is unavailable")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(20260828)
            projection = torch.randn(
                int(self._genept_matrix.shape[1]),
                self.options.graph_input_dim,
                generator=generator,
                dtype=torch.float32,
            ) / (float(self._genept_matrix.shape[1]) ** 0.5)
            initialized = self._genept_matrix.cpu().matmul(projection)
            with torch.no_grad():
                self.gene_embeddings.copy_(initialized.to(self.gene_embeddings.device))

    def _full_node_inputs(self) -> Tensor:
        if self.options.gene_feature_mode in {
            "frozen_genept_projection",
            "genept_shuffled",
        }:
            if self._genept_matrix is None or self.genept_projection is None:
                raise RuntimeError("GenePT projected input is unavailable")
            return cast(Tensor, self.genept_projection(self._genept_matrix))
        if self.options.gene_feature_mode == "genept_id_residual":
            if self._genept_matrix is None or self.genept_projection is None:
                raise RuntimeError("GenePT residual input is unavailable")
            projected = cast(Tensor, self.genept_projection(self._genept_matrix))
            return self.gene_embeddings + projected
        return self.gene_embeddings

    def _node_inputs(self, view: GraphView) -> Tensor:
        node_ids = torch.as_tensor(
            view.node_ids,
            device=self.gene_embeddings.device,
            dtype=torch.long,
        )
        inputs = self._full_node_inputs().index_select(0, node_ids)
        masked = set(view.masked_node_ids) | set(view.masked_anchor_ids)
        if masked:
            local_by_global = {node_id: index for index, node_id in enumerate(view.node_ids)}
            local_ids = [local_by_global[node_id] for node_id in sorted(masked)]
            inputs = inputs.clone()
            index = torch.as_tensor(local_ids, device=inputs.device, dtype=torch.long)
            inputs.index_copy_(0, index, self.mask_token.expand(len(local_ids), -1))
        return inputs

    def _source_tensors(self, view: GraphView) -> tuple[GraphSourceTensors, ...]:
        if view is self._resident_prediction_view and self._resident_source_tensors is not None:
            return self._resident_source_tensors
        result = []
        for source_name in self.options.graph_sources:
            edges = view.edges_by_source[source_name]
            edge_index = torch.as_tensor(
                view.local_edge_index(source_name),
                device=self.gene_embeddings.device,
                dtype=torch.long,
            )
            weight_values = [edge.weight for edge in edges]
            if source_name == "string" and self.options.string_weight_mode != "selection_only":
                if self._string_weight_by_global_edge is None:
                    raise RuntimeError(
                        "numerical STRING routes require the frozen full-topology weight contract"
                    )
                try:
                    weight_values = [
                        (
                            1.0
                            if edge.source == edge.target
                            else self._string_weight_by_global_edge[(edge.source, edge.target)]
                        )
                        for edge in edges
                    ]
                except KeyError as error:
                    raise ValueError(
                        f"STRING view edge is absent from the frozen full topology: {error.args[0]}"
                    ) from error
            edge_weight = torch.as_tensor(
                weight_values,
                device=self.gene_embeddings.device,
                dtype=self.gene_embeddings.dtype,
            )
            result.append(
                GraphSourceTensors(
                    name=source_name,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            )
        return tuple(result)

    def configure_string_weight_contract(self, prediction_view: GraphView) -> None:
        """Freeze topology-global official normalization and the WS permutation once."""

        if self.options.string_weight_mode == "selection_only":
            self._string_weight_by_global_edge = None
            return
        if (
            prediction_view.view_id != "prediction"
            or prediction_view.node_ids != tuple(range(self.n_genes))
            or prediction_view.masked_node_ids
            or prediction_view.masked_anchor_ids
        ):
            raise ValueError("STRING weight contract requires the clean full prediction view")
        edges = tuple(
            edge for edge in prediction_view.edges_by_source["string"] if edge.source != edge.target
        )
        if not edges:
            raise ValueError("STRING weight contract requires non-self full-topology edges")
        edge_index = torch.as_tensor(
            [[edge.source for edge in edges], [edge.target for edge in edges]],
            device=self.gene_embeddings.device,
            dtype=torch.long,
        )
        raw_weights = torch.as_tensor(
            [edge.weight for edge in edges],
            device=self.gene_embeddings.device,
            dtype=self.gene_embeddings.dtype,
        )
        prepared = _prepare_normalized_string_weights(
            edge_index,
            raw_weights,
            shuffle_nonself=(self.options.string_weight_mode == "shuffled_edge_feature"),
        )
        self._string_weight_by_global_edge = {
            (edge.source, edge.target): float(weight)
            for edge, weight in zip(edges, prepared.detach().cpu().tolist(), strict=True)
        }

    def configure_resident_graph_tensors(self, prediction_view: GraphView) -> None:
        if (
            prediction_view.view_id != "prediction"
            or prediction_view.node_ids != tuple(range(self.n_genes))
            or prediction_view.masked_node_ids
            or prediction_view.masked_anchor_ids
        ):
            raise ValueError("resident graph tensors require the clean full prediction view")
        self._resident_prediction_view = prediction_view
        self._resident_source_tensors = self._source_tensors(prediction_view)
        if isinstance(self.backend, SparseGraphTransformerEncoder):
            self._resident_sparse_union = build_sparse_union(
                node_count=self.n_genes,
                sources=self._resident_source_tensors,
                expected_names=self.backend.source_names,
                add_reverse_edges=self.backend.add_reverse_edges,
                add_self_loops=self.backend.add_self_loops,
                expander_degree=self.backend.expander_degree,
            )
        else:
            self._resident_sparse_union = None

    def resident_graph_tensor_payload(self) -> dict[str, object]:
        sources = self._resident_source_tensors or ()
        return {
            "active": self._resident_prediction_view is not None,
            "node_count": self.n_genes if sources else 0,
            "edge_count_by_source": {
                source.name: int(source.edge_index.shape[1]) for source in sources
            },
            "sparse_union_active": self._resident_sparse_union is not None,
            "sparse_union_edge_count": (
                int(self._resident_sparse_union.edge_index.shape[1])
                if self._resident_sparse_union is not None
                else 0
            ),
            "sparse_union_channel_names": (
                list(self._resident_sparse_union.channel_names)
                if self._resident_sparse_union is not None
                else []
            ),
        }

    def _batched_sources(
        self,
        views: Sequence[GraphView],
        node_counts: tuple[int, ...],
    ) -> tuple[GraphSourceTensors, ...]:
        result = []
        for source_name in self.options.graph_sources:
            indices = []
            weights = []
            offset = 0
            for view, count in zip(views, node_counts, strict=True):
                source = next(
                    item for item in self._source_tensors(view) if item.name == source_name
                )
                indices.append(source.edge_index + offset)
                if source.edge_weight is None:
                    raise RuntimeError("native view source weights are unavailable")
                weights.append(source.edge_weight)
                offset += count
            result.append(
                GraphSourceTensors(
                    name=source_name,
                    edge_index=torch.cat(indices, dim=1),
                    edge_weight=torch.cat(weights, dim=0),
                )
            )
        return tuple(result)

    def _batched_sparse_union(
        self,
        views: Sequence[GraphView],
        node_counts: tuple[int, ...],
        backend: SparseGraphTransformerEncoder,
    ) -> SparseUnionTensors:
        if (
            len(views) == 1
            and views[0] is self._resident_prediction_view
            and self._resident_sparse_union is not None
        ):
            return self._resident_sparse_union
        edge_indices = []
        memberships = []
        local_indices = []
        fixed = {name: index for index, name in enumerate(backend.edge_channel_names)}
        offset = 0
        for view, count in zip(views, node_counts, strict=True):
            union = build_sparse_union(
                node_count=count,
                sources=self._source_tensors(view),
                expected_names=backend.source_names,
                add_reverse_edges=backend.add_reverse_edges,
                add_self_loops=backend.add_self_loops,
                expander_degree=backend.expander_degree,
            )
            aligned = union.edge_membership.new_zeros((union.edge_membership.shape[0], len(fixed)))
            for index, name in enumerate(union.channel_names):
                aligned[:, fixed[name]] = union.edge_membership[:, index]
            edge_indices.append(union.edge_index + offset)
            memberships.append(aligned)
            local_indices.append(union.local_edge_index + offset)
            offset += count
        return SparseUnionTensors(
            edge_index=torch.cat(edge_indices, dim=1),
            edge_membership=torch.cat(memberships, dim=0),
            local_edge_index=torch.cat(local_indices, dim=1),
            channel_names=backend.edge_channel_names,
        )

    def forward_many(self, views: Sequence[GraphView]) -> tuple[EncodedGraphView, ...]:
        if not views:
            raise ValueError("at least one graph view is required")
        # Official Exphormer processes one fixed graph per forward.  Its
        # BatchNorm layers therefore compute statistics per graph/view during
        # training.  Concatenating disconnected crops would couple unrelated
        # global/local views through shared batch statistics, so the native
        # sparse Transformer preserves the independent-view contract here.
        if self.training and isinstance(self.backend, SparseGraphTransformerEncoder):
            encoded = []
            for view in views:
                inputs = self._node_inputs(view)
                union = self._batched_sparse_union((view,), (int(inputs.shape[0]),), self.backend)
                states = self.backend.forward_union(inputs, union)
                encoded.append(EncodedGraphView(node_ids=view.node_ids, node_states=states))
            return tuple(encoded)
        inputs_by_view = tuple(self._node_inputs(view) for view in views)
        node_counts = tuple(int(inputs.shape[0]) for inputs in inputs_by_view)
        inputs = torch.cat(inputs_by_view, dim=0)
        if isinstance(self.backend, SparseGraphTransformerEncoder):
            union = self._batched_sparse_union(views, node_counts, self.backend)
            states = self.backend.forward_union(inputs, union)
        else:
            states = self.backend(inputs, self._batched_sources(views, node_counts))
        splits = torch.split(states, list(node_counts), dim=0)
        return tuple(
            EncodedGraphView(node_ids=view.node_ids, node_states=state)
            for view, state in zip(views, splits, strict=True)
        )

    def forward(self, view: GraphView) -> EncodedGraphView:
        return self.forward_many((view,))[0]


class ControlConditionTransformer(nn.Module):
    """Two-token pre-norm control-conditioned shift block."""

    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(128, 64)

    def forward(self, basal: Tensor, condition: Tensor) -> Tensor:
        tokens = torch.stack((basal, condition), dim=1)
        encoded = self.encoder(tokens)
        return cast(Tensor, self.output(encoded.reshape(encoded.shape[0], 128)))


class ControlConditionMLP(nn.Module):
    """Concat MLP sized within one percent of the Transformer control block."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 301),
            nn.GELU(),
            nn.Linear(301, 64),
        )

    def forward(self, basal: Tensor, condition: Tensor) -> Tensor:
        return cast(Tensor, self.layers(torch.cat((basal, condition), dim=-1)))


class ConsistencyProjector(nn.Module):
    def __init__(self, prototype_count: int) -> None:
        super().__init__()
        if prototype_count not in {65536, 32768, 16384, 8192}:
            raise ValueError("prototype_count must come from the frozen server-fit candidates")
        self.prototype_count = prototype_count
        self.mlp = nn.Sequential(
            nn.Linear(64, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
        )
        self.prototype_layer = nn.utils.weight_norm(nn.Linear(256, prototype_count, bias=False))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        weight_v = cast(Tensor, self.prototype_layer.weight_v)
        weight_g = cast(Tensor, self.prototype_layer.weight_g)
        nn.init.trunc_normal_(weight_v, std=0.02)
        weight_g.data.fill_(1.0)

    def forward(self, states: Tensor) -> Tensor:
        bottleneck = F.normalize(self.mlp(states), dim=-1, p=2)
        return cast(Tensor, self.prototype_layer(bottleneck))


class BasalStateEncoder(nn.Module):
    def __init__(self, gene_count: int) -> None:
        super().__init__()
        if gene_count <= 0:
            raise ValueError("gene_count must be positive")
        self.network = nn.Sequential(
            nn.Linear(gene_count, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 64),
        )

    def forward(self, control_expression: Tensor) -> Tensor:
        return cast(Tensor, self.network(control_expression))


class ExpressionDecoder(nn.Module):
    def __init__(self, gene_count: int) -> None:
        super().__init__()
        if gene_count <= 0:
            raise ValueError("gene_count must be positive")
        self.network = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, gene_count),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return cast(Tensor, self.network(latent))


class GraDPertJointModel(nn.Module):
    """Joint prediction and graph self-distillation model for the B2 route."""

    def __init__(
        self,
        *,
        graph_gene_count: int,
        expression_gene_count: int,
        prototype_count: int,
        architecture: NativeArchitectureOptions | None = None,
        genept_matrix: Tensor | None = None,
    ) -> None:
        super().__init__()
        if expression_gene_count <= 0 or graph_gene_count <= 0:
            raise ValueError("graph and expression gene counts must both be positive")
        self.graph_gene_count = graph_gene_count
        self.expression_gene_count = expression_gene_count
        self.architecture = architecture or NativeArchitectureOptions.from_parameters({})
        configurable = self.architecture.graph_encoder_family != "adaptive_relation_gat"
        self.student_encoder = (
            ConfigurableGeneGraphEncoder(
                graph_gene_count,
                self.architecture,
                genept_matrix=genept_matrix,
            )
            if configurable
            else AdaptiveGeneGraphEncoder(graph_gene_count)
        )
        self.student_projector = ConsistencyProjector(prototype_count)
        self.basal_encoder = BasalStateEncoder(expression_gene_count)
        self.expression_decoder = ExpressionDecoder(expression_gene_count)
        # Construct teacher modules independently, then copy the exact student state.
        # Weight-normalized prototype layers expose computed tensors that cannot
        # safely participate in ``deepcopy``.
        self.teacher_encoder = (
            ConfigurableGeneGraphEncoder(
                graph_gene_count,
                self.architecture,
                genept_matrix=genept_matrix,
            )
            if configurable
            else AdaptiveGeneGraphEncoder(graph_gene_count)
        )
        self.control_condition_fusion: nn.Module | None
        if self.architecture.decoder_mode == "additive":
            self.control_condition_fusion = None
        elif self.architecture.decoder_mode == "parameter_matched_mlp":
            self.control_condition_fusion = ControlConditionMLP()
        else:
            self.control_condition_fusion = ControlConditionTransformer()
        self.teacher_projector = ConsistencyProjector(prototype_count)
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        self.teacher_projector.load_state_dict(self.student_projector.state_dict())
        for parameter in self.teacher_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.teacher_projector.parameters():
            parameter.requires_grad_(False)

    def prediction_condition_state(
        self,
        prediction_view: GraphView,
        anchor_ids: tuple[int, ...],
    ) -> Tensor:
        encoded = self.student_encoder(prediction_view)
        return cast(Tensor, encoded.condition_state(anchor_ids))

    def prediction_condition_states(
        self,
        prediction_view: GraphView,
        anchors_by_condition: Mapping[str, tuple[int, ...]],
    ) -> tuple[tuple[str, ...], Tensor]:
        """Encode the deterministic graph once, then gather every unique condition."""

        if not anchors_by_condition:
            raise ValueError("prediction requires at least one condition")
        condition_ids = tuple(sorted(anchors_by_condition))
        encoded = self.student_encoder(prediction_view)
        states = torch.stack(
            [encoded.condition_state(anchors_by_condition[item]) for item in condition_ids]
        )
        return condition_ids, states

    def decode_expression(self, control_expression: Tensor, perturbation: Tensor) -> Tensor:
        if (
            control_expression.ndim != 2
            or control_expression.shape[1] != self.expression_gene_count
        ):
            raise ValueError("control expression must be [cells, expression_gene_count]")
        if perturbation.shape != (control_expression.shape[0], 64):
            raise ValueError("perturbation state must be [cells, 64]")
        basal = self.basal_encoder(control_expression)
        fused = (
            basal + perturbation
            if self.control_condition_fusion is None
            else self.control_condition_fusion(basal, perturbation)
        )
        return cast(Tensor, self.expression_decoder(fused))

    def predict_expression_batch(
        self,
        control_expression: Tensor,
        prediction_view: GraphView,
        condition_ids: Sequence[str],
        anchors_by_condition: Mapping[str, tuple[int, ...]],
    ) -> Tensor:
        """Predict cells while sharing one full-graph forward across conditions."""

        if len(condition_ids) != control_expression.shape[0]:
            raise ValueError("condition IDs must align with control-expression rows")
        unique_ids, unique_states = self.prediction_condition_states(
            prediction_view,
            anchors_by_condition,
        )
        index_by_condition = {condition_id: index for index, condition_id in enumerate(unique_ids)}
        try:
            row_indices = [index_by_condition[condition_id] for condition_id in condition_ids]
        except KeyError as error:
            raise ValueError(
                f"batch condition has no active-anchor mapping: {error.args[0]}"
            ) from error
        indices = torch.as_tensor(
            row_indices,
            device=unique_states.device,
            dtype=torch.long,
        )
        perturbation = unique_states.index_select(0, indices)
        return self.decode_expression(control_expression, perturbation)

    def predict_expression(
        self,
        control_expression: Tensor,
        prediction_view: GraphView,
        anchor_ids: tuple[int, ...],
    ) -> Tensor:
        condition_id = "single_condition"
        return self.predict_expression_batch(
            control_expression,
            prediction_view,
            [condition_id] * control_expression.shape[0],
            {condition_id: anchor_ids},
        )
