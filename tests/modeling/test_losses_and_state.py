from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gradpert.modeling import (  # noqa: E402
    CenterState,
    condition_consistency_loss,
    cosine_teacher_momentum,
    embedding_spread_loss,
    masked_node_consistency_loss,
    update_center,
    update_teacher_ema,
)


@pytest.mark.parametrize(("local_count", "term_count"), [(4, 10), (8, 18)])
def test_condition_consistency_has_expected_terms_and_gradients(
    local_count: int,
    term_count: int,
) -> None:
    center = torch.zeros(1, 8)
    students = tuple(torch.randn(3, 8, requires_grad=True) for _ in range(local_count + 2))
    teachers = (torch.randn(3, 8), torch.randn(3, 8))
    loss = condition_consistency_loss(
        student_view_logits=students,
        teacher_global_logits=teachers,
        center=center,
    )
    assert loss.ndim == 0
    loss.backward()
    assert all(item.grad is not None for item in students)
    assert term_count == 2 * (local_count + 1)


def test_condition_consistency_preserves_legacy_eight_local_arithmetic() -> None:
    torch.manual_seed(31)
    center = torch.randn(1, 8)
    students = tuple(torch.randn(3, 8) for _ in range(10))
    teachers = (torch.randn(3, 8), torch.randn(3, 8))
    observed = condition_consistency_loss(
        student_view_logits=students,
        teacher_global_logits=teachers,
        center=center,
    )

    legacy_terms = []
    for teacher_index, teacher_logits in enumerate(teachers):
        targets = torch.softmax((teacher_logits - center) / 0.04, dim=-1).detach()
        for student_index, student_logits in enumerate(students):
            if student_index == teacher_index:
                continue
            log_probabilities = torch.log_softmax(student_logits / 0.1, dim=-1)
            legacy_terms.append(-(targets * log_probabilities).sum(dim=-1).mean())
    expected = torch.stack(legacy_terms).mean()
    assert torch.equal(observed, expected)


def test_condition_consistency_rejects_unregistered_local_count() -> None:
    with pytest.raises(ValueError, match="four or eight locals"):
        condition_consistency_loss(
            student_view_logits=tuple(torch.randn(3, 8) for _ in range(7)),
            teacher_global_logits=(torch.randn(3, 8), torch.randn(3, 8)),
            center=torch.zeros(1, 8),
        )


def test_masked_node_normalizes_over_one_shared_global_and_handles_empty() -> None:
    center = torch.zeros(1, 4)
    students = torch.randn(2, 4, requires_grad=True)
    teachers = torch.randn(2, 4)
    loss = masked_node_consistency_loss(
        student_logits=students,
        teacher_logits=teachers,
        center=center,
    )
    loss.backward()
    assert students.grad is not None

    empty = torch.empty(0, 4, requires_grad=True)
    empty_loss = masked_node_consistency_loss(
        student_logits=empty,
        teacher_logits=torch.empty(0, 4),
        center=center,
    )
    assert empty_loss.item() == 0


def test_spread_unavailable_for_single_condition_and_finite_otherwise() -> None:
    unavailable, flag = embedding_spread_loss(torch.randn(1, 64, requires_grad=True))
    assert not flag
    assert unavailable.item() == 0
    value, flag = embedding_spread_loss(torch.randn(4, 64, requires_grad=True))
    assert flag
    assert torch.isfinite(value)


def test_post_step_ema_center_and_cosine_schedule() -> None:
    student = torch.nn.Linear(2, 2, bias=False)
    teacher = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        student.weight.fill_(2)
        teacher.weight.fill_(0)
    update_teacher_ema(student, teacher, momentum=0.75)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5))

    state = CenterState.zeros(prototype_count=4, device=torch.device("cpu"))
    update_center(state.condition, torch.full((2, 4), 2.0))
    assert torch.allclose(state.condition, torch.full((1, 4), 0.2))
    assert cosine_teacher_momentum(global_step=0, total_steps=100) == pytest.approx(0.996)
    assert cosine_teacher_momentum(global_step=100, total_steps=100) == pytest.approx(1.0)
