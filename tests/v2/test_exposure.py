import pytest

from gradpert.training.v2.exposure import checkpoint_expression_groups


def test_checkpoint_groups_use_actual_seen_queries_not_allowed_pool():
    history = [
        {"epoch": 1, "seen_expression_gene_indices": [0, 2]},
        {"epoch": 2, "seen_expression_gene_indices": [0, 1, 2]},
    ]
    axis = ("a", "b", "c", "heldout")
    groups, receipt = checkpoint_expression_groups(
        history, checkpoint_epoch=1, expression_gene_ids=axis, allowed_gene_indices=(0, 1, 2)
    )
    assert groups == {"seen_expression": (0, 2), "unseen_expression": (1, 3)}
    assert receipt["checkpoint_epoch"] == 1
    groups, _ = checkpoint_expression_groups(
        history, checkpoint_epoch=2, expression_gene_ids=axis, allowed_gene_indices=(0, 1, 2)
    )
    assert groups["unseen_expression"] == (3,)


def test_checkpoint_groups_reject_unrecorded_and_invalid_exposure():
    with pytest.raises(ValueError, match="lacks actual"):
        checkpoint_expression_groups(
            [{"epoch": 1}],
            checkpoint_epoch=1,
            expression_gene_ids=("a", "b"),
            allowed_gene_indices=None,
        )
    with pytest.raises(ValueError, match="frozen gene axis"):
        checkpoint_expression_groups(
            [{"epoch": 1, "seen_expression_gene_indices": [1]}],
            checkpoint_epoch=1,
            expression_gene_ids=("a", "b"),
            allowed_gene_indices=(0,),
        )


def test_all_seen_reports_an_explicit_empty_unseen_group():
    groups, receipt = checkpoint_expression_groups(
        [{"epoch": 1, "seen_expression_gene_indices": [0, 1]}],
        checkpoint_epoch=1,
        expression_gene_ids=("a", "b"),
        allowed_gene_indices=None,
    )
    assert groups["unseen_expression"] == ()
    assert receipt["groups"]["unseen_expression"]["gene_count"] == 0
