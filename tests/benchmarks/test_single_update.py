import pytest
import torch

from benchmarks.common.single_update import capture_single_update


def test_real_adam_update_stops_before_second_batch_or_validation(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.weight.detach().clone()

    def fit():
        model(torch.ones(3, 2)).square().mean().backward()
        optimizer.step()
        pytest.fail("must stop before next batch/validation")

    checkpoint = tmp_path / "step.pt"
    result = capture_single_update(fit=fit, model_supplier=lambda: model, checkpoint=checkpoint)
    assert result["completed_steps"] == 1
    assert result["completed_epochs"] == 0
    assert not torch.equal(before, model.weight)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["role"] == "diagnostic_after_step_not_best"
    optimizer.step()  # Hook was removed; this must not raise the internal stop.


def test_real_failure_propagates_and_removes_hook(tmp_path):
    model = torch.nn.Linear(1, 1)

    def fail():
        raise ValueError("actual failure")

    with pytest.raises(ValueError, match="actual failure"):
        capture_single_update(
            fit=fail, model_supplier=lambda: model, checkpoint=tmp_path / "step.pt"
        )
    assert not (tmp_path / "step.pt").exists()
    model(torch.ones(1, 1)).sum().backward()
    torch.optim.Adam(model.parameters()).step()
