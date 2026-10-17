"""Scoped graph reuse preserves identities, randomness and frozen predictions."""

import copy
from dataclasses import fields

import numpy as np
import pytest
import torch
from test_fgpo_family import new_fixture
from test_relay_method import small_index

from gradpert.training.v2.evaluation import predict_controls
from gradpert.training.v2.evaluation_graph import prepare_evaluation_graph


@pytest.mark.parametrize("bitgen", [np.random.PCG64, np.random.Philox, np.random.MT19937])
def test_context_cache_keeps_every_tensor_and_rng_draw_and_has_no_writable_alias(bitgen):
    fast = small_index(16)
    reference = copy.deepcopy(fast)
    reference.cache_full_context = False
    a, b = np.random.Generator(bitgen(51)), np.random.Generator(bitgen(51))
    for induced, dropout, ids in (
        (False, 0, [1, 3, 7]),
        (False, 0, [0, 4, 9]),
        (True, 0, [2, 3, 5]),
        (False, 0.2, [0, 4, 9]),
        (False, 0, [1, 3, 7]),
    ):
        kw = dict(
            ids=np.array(ids),
            targets=[(ids[0],)],
            device=torch.device("cpu"),
            induced=induced,
            edge_dropout=dropout,
            mask_ratio=0.4,
        )
        left, right = reference.view(rng=a, **kw), fast.view(rng=b, **kw)
        for f in fields(left):
            x, y = getattr(left, f.name), getattr(right, f.name)
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y), f.name
            elif x is not None:
                for sub in fields(x):
                    assert torch.equal(getattr(x, sub.name), getattr(y, sub.name)), sub.name
        np.testing.assert_equal(a.bit_generator.state, b.bit_generator.state)
        # Returned CPU tensors can be modified without corrupting future views.
        right.context.neighbors.zero_()
    assert fast.context_cache_builds == 1 and fast.context_cache_hits == 2


@pytest.mark.parametrize("query_count", [3, 12])
def test_graph_memory_reuses_all_layers_across_target_sets_with_exact_outputs(query_count):
    obj, _ = new_fixture("B0")
    model, index = obj.student.eval(), small_index(16)
    controls = np.random.default_rng(1).random((5, 12)).astype(np.float32)
    kwargs = dict(device=torch.device("cpu"), cell_batch=2, query_count=query_count)
    memory = prepare_evaluation_graph(model, index, device=torch.device("cpu"))
    calls = []
    handle = model.graph.register_forward_hook(lambda *args: calls.append(1))
    rng = torch.get_rng_state().clone()
    for targets in ((0,), (15,), (2, 14)):
        reference = predict_controls(model, index, controls, targets, **kwargs)
        count = len(calls)
        actual = predict_controls(model, index, controls, targets, graph_memory=memory, **kwargs)
        assert len(calls) == count
        np.testing.assert_array_equal(actual, reference)
        assert torch.equal(rng, torch.get_rng_state())
    handle.remove()
    with torch.no_grad():
        next(model.graph.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="same frozen"):
        predict_controls(model, index, controls, (0,), graph_memory=memory, **kwargs)


def test_conditional_graph_never_caches_control_dependent_final_features():
    obj, _ = new_fixture("CG1")
    assert (
        prepare_evaluation_graph(obj.student, small_index(16), device=torch.device("cpu")) is None
    )
