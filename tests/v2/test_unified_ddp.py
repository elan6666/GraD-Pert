"""Global P1/CG1 reductions across accumulation, unequal ranks and singleton tails."""

import copy
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_unified_integration import fixture

from gradpert.training.v2.engine import optimizer_step, slice_cells
from gradpert.training.v2.optimizer import V2Optimizer


class ReductionReferenceSGD:
    """Isolate reduction from Muon's quantized Newton-Schulz trajectory.

    Real Muon/AdamW updates are separately checked for bitwise cross-rank
    synchronization below; this oracle preserves the tight gradient/state
    tolerances and has no BF16 orthogonalization boundary.
    """

    def __init__(self, model, lr, weight_decay):
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        self.steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.steps += 1

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "steps": self.steps}


def deterministic_fixture(arm, count):
    objective, batch, _, _ = fixture(arm, checkpointed=False)
    for model in (objective.student, objective.teacher):
        model.options = replace(model.options, dropout=0, random_gene_order=False)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
            if isinstance(getattr(module, "dropout", None), (float, int)):
                module.dropout = 0.0
            if hasattr(module, "randomize_order"):
                module.randomize_order = False
            if hasattr(module, "options") and hasattr(module.options, "random_gene_order"):
                module.options = replace(module.options, dropout=0, random_gene_order=False)
    if arm == "CG1":
        with torch.no_grad():
            objective.student.graph_modulation.projection.weight.fill_(0.02)
        objective.teacher.load_state_dict(objective.student.state_dict())
    # High precision and identical micro shapes isolate reduction semantics
    # from float32 RMSNorm/Adam amplification and shape-dependent GEMM order.
    # Prototype logits/centers deliberately keep their native float32 path.
    objective = objective.double()
    for model in (objective.student, objective.teacher):
        for head in (model.ssl1_cls, model.ssl1_node, model.ssl2_cls, model.ssl2_node):
            head.prototypes.float()
    for name in ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node"):
        setattr(objective, name + "_center", getattr(objective, name + "_center").float())
    batch = replace(batch, control=batch.control.double(), truth=batch.truth.double())
    return objective, slice_cells(batch, 0, count)


def assert_optimizer_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=1e-4)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_optimizer_state(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_optimizer_state(left, right)
    else:
        assert actual == expected


def assert_optimizer_rank_sync(state):
    if isinstance(state, torch.Tensor):
        shared = state.clone()
        dist.broadcast(shared, src=0)
        torch.testing.assert_close(state, shared, atol=0, rtol=0)
    elif isinstance(state, dict):
        for value in state.values():
            assert_optimizer_rank_sync(value)
    elif isinstance(state, (tuple, list)):
        for value in state:
            assert_optimizer_rank_sync(value)


def _distributed_worker(rank, rendezvous, reference, arm, count, strategy):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        objective, batch = deterministic_fixture(arm, count)
        objective.loss_reduction = strategy
        local = slice_cells(batch, count * rank // 2, count * (rank + 1) // 2)
        optimizer = ReductionReferenceSGD(objective.student, 1e-3, 0)
        expected = torch.load(reference, weights_only=True)
        for step in range(2):
            metrics = optimizer_step(
                objective,
                optimizer,
                local,
                microbatch=1,
                lr=1e-3,
                momentum=0.9,
                bf16=False,
                global_condition_index=batch.condition_index,
            )
            for name, parameter in objective.student.named_parameters():
                if parameter.grad is not None:
                    torch.testing.assert_close(
                        parameter.grad,
                        expected[step]["gradients"][name],
                        atol=2e-7,
                        rtol=1e-4,
                        msg=lambda m, name=name: "gradient " + name + ": " + m,
                    )
            for name, parameter in objective.state_dict().items():
                torch.testing.assert_close(
                    parameter,
                    expected[step]["objective"][name],
                    atol=5e-6,
                    rtol=1e-4,
                    msg=lambda m, name=name: "state " + name + ": " + m,
                )
            for name in ("prediction", "ssl1_condition", "ssl1_node", "ssl2_dino", "ssl2_ibot"):
                assert metrics[name] == pytest.approx(
                    expected[step]["metrics"][name], abs=4e-6, rel=1e-5
                )
            assert optimizer.steps == step + 1
            assert_optimizer_state(optimizer.state_dict(), expected[step]["optimizer"])
            assert not objective.pending
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "arm,count,strategy",
    [
        ("P1", 3, "row_mean"),
        ("P1", 1, "row_mean"),
        ("CG1", 3, "row_mean"),
        ("CG1", 3, "condition_mean"),
    ],
)
def test_two_rank_accumulated_complete_updates_match_global_batch(tmp_path, arm, count, strategy):
    objective, batch = deterministic_fixture(arm, count)
    objective.loss_reduction = strategy
    optimizer = ReductionReferenceSGD(objective.student, 1e-3, 0)
    records = []
    for _ in range(2):
        metrics = optimizer_step(
            objective, optimizer, batch, microbatch=1, lr=1e-3, momentum=0.9, bf16=False
        )
        records.append(
            {
                "metrics": metrics,
                "objective": {k: v.clone() for k, v in objective.state_dict().items()},
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "gradients": {
                    k: p.grad.clone()
                    for k, p in objective.student.named_parameters()
                    if p.grad is not None
                },
            }
        )
    reference = str(tmp_path / "reference.pt")
    torch.save(records, reference)
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "rendezvous"), reference, arm, count, strategy),
        nprocs=2,
        join=True,
    )


def test_p1_accumulation_matches_full_population_instead_of_local_mmd():
    reference, batch = deterministic_fixture("P1", 4)
    accumulated, _ = deterministic_fixture("P1", 4)
    full = optimizer_step(
        reference,
        V2Optimizer(reference.student, 1e-3, 0),
        batch,
        microbatch=4,
        lr=1e-3,
        momentum=0.9,
        bf16=False,
    )
    split = optimizer_step(
        accumulated,
        V2Optimizer(accumulated.student, 1e-3, 0),
        batch,
        microbatch=2,
        lr=1e-3,
        momentum=0.9,
        bf16=False,
    )
    assert split["population_mmd"] == pytest.approx(full["population_mmd"], abs=1e-7)
    for name, value in accumulated.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=5e-6, rtol=1e-4)


def _native_worker(rank, rendezvous, arm):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        objective, batch, _, _ = fixture(arm, checkpointed=True)
        local = slice_cells(batch, 3 * rank // 2, 3 * (rank + 1) // 2)
        optimizer = V2Optimizer(objective.student, 1e-3, 0)
        for _ in range(2):
            metrics = optimizer_step(
                objective,
                optimizer,
                local,
                microbatch=1,
                lr=1e-3,
                momentum=0.9,
                bf16=False,
                global_condition_index=batch.condition_index[:3],
            )
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            for parameter in objective.state_dict().values():
                shared = parameter.clone()
                dist.broadcast(shared, src=0)
                torch.testing.assert_close(parameter, shared, atol=0, rtol=0)
            assert_optimizer_rank_sync(optimizer.state_dict())
            assert not objective.pending
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("arm", ["N0", "CG1", "P1"])
def test_native_float32_randomized_checkpointed_ranks_remain_synchronized(tmp_path, arm):
    mp.spawn(_native_worker, args=(str(tmp_path / "native-rendezvous"), arm), nprocs=2, join=True)
