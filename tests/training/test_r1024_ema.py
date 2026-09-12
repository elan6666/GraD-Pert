import pytest

from gradpert.modeling.state import cosine_teacher_momentum
from tests.training.test_step_and_resume import _batch, _components


@pytest.mark.parametrize("start", [0.990, 0.994, 0.996])
def test_configurable_ema_reaches_real_update(start):
    _, _, _, engine = _components()
    engine.teacher_ema_start = start
    result = engine.train_step(_batch(), global_step=0)
    assert result.teacher_momentum == start
    assert cosine_teacher_momentum(global_step=399, total_steps=399, start=start) == 1.0
