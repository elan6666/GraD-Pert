import copy

import pytest

from scripts.performance.compare_bounded_epochs import compare_segments


def fixture_receipts():
    state = {k: "a" * 64 for k in ("model", "teacher", "gradients", "optimizer", "centers", "rng")}
    boundaries = []
    for epoch in range(4):
        row = dict(completed_epochs=epoch, global_step=epoch * 2, state=dict(state))
        if epoch:
            row.update(
                checkpoint_identity={"config": "a"},
                validation_rows=list(range(epoch)),
                checkpoints={
                    role: {"content_sha256": "b" * 64, "progress": epoch}
                    for role in ("best", "last")
                },
            )
        boundaries.append(row)
    base = dict(
        status="complete",
        error=None,
        terminal_predicates={"source_unchanged": True},
        scientific_completion=False,
        source=dict(dirty=False, commit="a" * 40, tree_sha256="c" * 64),
        start_epoch=0,
        stop_before_epoch=3,
        schedule_epochs=50,
        persistent_pkl_count=0,
        guard=dict(validation_constructor_count=1, test_access_attempts=0),
        preflight_predicates={"idle": True},
        boundaries=boundaries,
        config_sha256="d" * 64,
        frozen_identity={"graph": "e"},
        implementation="cpu_vectorized",
    )
    candidate = copy.deepcopy(base)
    candidate["implementation"] = "cpu_array"
    parent, resumed = copy.deepcopy(candidate), copy.deepcopy(candidate)
    parent["stop_before_epoch"], parent["boundaries"] = 1, parent["boundaries"][:2]
    resumed["start_epoch"], resumed["boundaries"] = 1, resumed["boundaries"][1:]
    resumed["boundaries"][0]["state"]["gradients"] = "f" * 64
    return base, candidate, parent, resumed


def test_exact_and_fresh_resume_gradient_boundary() -> None:
    compare_segments(*fixture_receipts())


@pytest.mark.parametrize("fault", ["state", "checkpoint", "schedule", "source", "resume_rng"])
def test_comparator_rejects_drift(fault) -> None:
    reference, candidate, parent, resumed = fixture_receipts()
    if fault == "state":
        candidate["boundaries"][2]["state"]["model"] = "wrong"
    if fault == "checkpoint":
        candidate["boundaries"][3]["checkpoints"]["last"]["content_sha256"] = "wrong"
    if fault == "schedule":
        candidate["schedule_epochs"] = 3
    if fault == "source":
        candidate["source"]["commit"] = "wrong"
    if fault == "resume_rng":
        resumed["boundaries"][0]["state"]["rng"] = "wrong"
    with pytest.raises(ValueError):
        compare_segments(reference, candidate, parent, resumed)
