from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.graphs import (  # noqa: E402
    GraphTopology,
    build_graph_view_batch,
    prune_incoming_edges,
)
from gradpert.modeling import GraDPertJointModel  # noqa: E402


def _views():  # type: ignore[no-untyped-def]
    genes = ("A", "B", "C", "D", "E", "F", "G")
    string = prune_incoming_edges(
        source_name="string",
        gene_ids=genes,
        weighted_edges=(("B", "A", 1.0), ("C", "A", 1.0), ("D", "B", 1.0)),
        top_k=20,
    )
    go = prune_incoming_edges(
        source_name="go",
        gene_ids=genes,
        weighted_edges=(("C", "A", 1.0), ("E", "C", 1.0), ("F", "B", 1.0)),
        top_k=20,
    )
    topology = GraphTopology(gene_ids=genes, sources={"string": string, "go": go})
    return build_graph_view_batch(
        topology,
        anchors=(0,),
        heldout_target_ids=(),
        run_seed=1,
        global_step=0,
        condition_id="A+ctrl",
        local_node_budget=5,
    )


def test_joint_model_shapes_teacher_identity_and_prediction_determinism() -> None:
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
    )
    student = model.student_encoder.state_dict()
    teacher = model.teacher_encoder.state_dict()
    assert student.keys() == teacher.keys()
    assert all(torch.equal(student[key], teacher[key]) for key in student)
    student_projector = model.student_projector.state_dict()
    teacher_projector = model.teacher_projector.state_dict()
    assert student_projector.keys() == teacher_projector.keys()
    assert all(
        torch.equal(student_projector[key], teacher_projector[key]) for key in student_projector
    )
    assert not any(parameter.requires_grad for parameter in model.teacher_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.teacher_projector.parameters())

    views = _views()
    model.eval()
    controls = torch.randn(4, 5)
    first = model.predict_expression(controls, views.prediction, (0,))
    second = model.predict_expression(controls, views.prediction, (0,))
    assert first.shape == (4, 5)
    assert torch.equal(first, second)

    encoded = model.student_encoder(views.prediction)
    assert encoded.node_states.shape == (7, 64)
    assert torch.equal(
        encoded.condition_state((0, 1)),
        encoded.node_states[0] + encoded.node_states[1],
    )


def test_masked_view_routes_gradient_to_shared_mask_token() -> None:
    model = GraDPertJointModel(
        graph_gene_count=7,
        expression_gene_count=5,
        prototype_count=8192,
    )
    views = _views()
    masked = views.globals[views.masked_global_index]
    assert masked.masked_node_ids
    encoded = model.student_encoder(masked)
    loss = encoded.node_states.square().mean()
    loss.backward()
    assert model.student_encoder.mask_token.grad is not None
    assert torch.isfinite(model.student_encoder.mask_token.grad).all()
