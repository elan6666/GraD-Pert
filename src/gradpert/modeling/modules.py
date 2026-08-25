"""Native graph encoder, consistency head, and additive expression predictor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv  # type: ignore[import-untyped]

from gradpert.graphs import GraphView


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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.gene_embeddings, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for query in self.relation_queries.values():
            nn.init.trunc_normal_(query, std=0.02)

    def _node_inputs(self, view: GraphView) -> Tensor:
        node_ids = torch.as_tensor(
            view.node_ids,
            device=self.gene_embeddings.device,
            dtype=torch.long,
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
                local_edges = torch.as_tensor(
                    view.local_edge_index(source_name),
                    device=inputs.device,
                    dtype=torch.long,
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
    ) -> None:
        super().__init__()
        if expression_gene_count <= 0 or graph_gene_count <= 0:
            raise ValueError("graph and expression gene counts must both be positive")
        self.graph_gene_count = graph_gene_count
        self.expression_gene_count = expression_gene_count
        self.student_encoder = AdaptiveGeneGraphEncoder(graph_gene_count)
        self.student_projector = ConsistencyProjector(prototype_count)
        self.basal_encoder = BasalStateEncoder(expression_gene_count)
        self.expression_decoder = ExpressionDecoder(expression_gene_count)
        # Construct teacher modules independently, then copy the exact student state.
        # Weight-normalized prototype layers expose computed tensors that cannot
        # safely participate in ``deepcopy``.
        self.teacher_encoder = AdaptiveGeneGraphEncoder(graph_gene_count)
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
        return cast(Tensor, self.expression_decoder(basal + perturbation))

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
