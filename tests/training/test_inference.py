from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.contracts import (  # noqa: E402
    ControlDraw,
    EvaluationControlManifest,
)
from gradpert.graphs import (  # noqa: E402
    GraphTopology,
    build_prediction_graph_view,
    prune_incoming_edges,
)
from gradpert.hashing import sha256_json  # noqa: E402
from gradpert.modeling import GraDPertJointModel  # noqa: E402
from gradpert.training.inference import (  # noqa: E402
    LoadedControlRows,
    iter_frozen_control_predictions,
    predict_frozen_controls,
)


def _topology() -> GraphTopology:
    genes = ("A", "B", "C")
    edges = (("B", "A", 1.0), ("C", "B", 1.0))
    return GraphTopology(
        gene_ids=genes,
        sources={
            source: prune_incoming_edges(
                source_name=source,
                gene_ids=genes,
                weighted_edges=edges,
            )
            for source in ("string", "go")
        },
    )


def _manifest() -> EvaluationControlManifest:
    row_ids = [f"c{index % 5}" for index in range(300)]
    context_ids = ["K562::b1"] * 300
    return EvaluationControlManifest(
        schema_version="evaluation-controls-v1",
        dataset_id="replogle_k562_essential",
        protocol_id="within_cell_unseen_single",
        split_name="val",
        split_content_sha256="a" * 64,
        evaluation_seed=20260824,
        rng="numpy_pcg64",
        sample_with_replacement=True,
        context_policy="truth_cell_context_resampling",
        n_controls_per_condition=300,
        draws=[
            ControlDraw(
                condition_id="A+ctrl",
                context_policy="truth_cell_context_resampling",
                source_pool_sha256="b" * 64,
                ordered_context_ids=context_ids,
                ordered_context_ids_sha256=sha256_json(context_ids),
                ordered_row_ids=row_ids,
                ordered_row_ids_sha256=sha256_json(row_ids),
            )
        ],
    )


def test_inference_preserves_exact_control_rows_and_never_collapses_population() -> None:
    torch.manual_seed(1)
    model = GraDPertJointModel(
        graph_gene_count=3,
        expression_gene_count=2,
        prototype_count=8192,
    )
    manifest = _manifest()
    values = np.arange(600, dtype=np.float32).reshape(300, 2)

    def load(row_ids: tuple[str, ...]) -> LoadedControlRows:
        return LoadedControlRows(ordered_row_ids=row_ids, expression=values)

    result = predict_frozen_controls(
        model=model,
        prediction_view=build_prediction_graph_view(_topology()),
        control_manifest=manifest,
        anchors_by_condition={"A+ctrl": (0,)},
        load_control_rows=load,
        device=torch.device("cpu"),
        decode_batch_size=64,
    )
    assert len(result) == 1
    assert result[0].prediction.shape == (300, 2)
    assert np.array_equal(result[0].input_control, values)
    assert result[0].input_control_row_ids == tuple(manifest.draws[0].ordered_row_ids)


def test_streaming_inference_preserves_manifest_order() -> None:
    torch.manual_seed(1)
    model = GraDPertJointModel(
        graph_gene_count=3,
        expression_gene_count=2,
        prototype_count=8192,
    )
    manifest = _manifest()
    values = np.arange(600, dtype=np.float32).reshape(300, 2)

    streamed = list(
        iter_frozen_control_predictions(
            model=model,
            prediction_view=build_prediction_graph_view(_topology()),
            control_manifest=manifest,
            anchors_by_condition={"A+ctrl": (0,)},
            load_control_rows=lambda row_ids: LoadedControlRows(
                ordered_row_ids=row_ids,
                expression=values,
            ),
            device=torch.device("cpu"),
            decode_batch_size=64,
        )
    )
    assert len(streamed) == 1
    assert streamed[0].condition_id == "A+ctrl"
    assert streamed[0].input_control_row_ids == tuple(manifest.draws[0].ordered_row_ids)


def test_inference_rejects_loader_reordering() -> None:
    model = GraDPertJointModel(
        graph_gene_count=3,
        expression_gene_count=2,
        prototype_count=8192,
    )
    with pytest.raises(ValueError, match="changed row order"):
        predict_frozen_controls(
            model=model,
            prediction_view=build_prediction_graph_view(_topology()),
            control_manifest=_manifest(),
            anchors_by_condition={"A+ctrl": (0,)},
            load_control_rows=lambda row_ids: LoadedControlRows(
                ordered_row_ids=tuple(reversed(row_ids)),
                expression=np.ones((300, 2), dtype=np.float32),
            ),
            device=torch.device("cpu"),
            decode_batch_size=64,
        )
