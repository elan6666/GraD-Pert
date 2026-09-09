from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gradpert.execution.native import _final_test_for_policy


def test_r50_does_not_claim_or_access_test():
    trainer = SimpleNamespace(progress=SimpleNamespace(test_evaluations=0), test_best_once=Mock())
    callback = Mock(side_effect=AssertionError("test access forbidden"))
    _final_test_for_policy("r50_selection", trainer, callback)
    callback.assert_not_called()
    trainer.test_best_once.assert_not_called()


def test_r50_rejects_test_consumed_state():
    trainer = SimpleNamespace(progress=SimpleNamespace(test_evaluations=1), test_best_once=Mock())
    with pytest.raises(RuntimeError, match="test-consumed"):
        _final_test_for_policy("r50_selection", trainer, Mock())
    trainer.test_best_once.assert_not_called()


@pytest.mark.parametrize(
    "policy", ["fixed_epoch_pilot", "vnext_combination_100", "smoke_then_full"]
)
def test_historical_policies_keep_exact_test_callback(policy):
    trainer = SimpleNamespace(test_best_once=Mock())
    callback = Mock()
    _final_test_for_policy(policy, trainer, callback)
    trainer.test_best_once.assert_called_once_with(callback)
