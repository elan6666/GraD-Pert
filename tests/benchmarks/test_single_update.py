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
    assert len(result["backward_root_losses"]) == 1
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


def test_nonfinite_loss_with_finite_derivative_is_rejected_before_update(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    before = model.weight.detach().clone()
    original_backward = torch.autograd.backward

    def fit():
        loss = model(torch.ones(1, 1)).sum() + float("nan")
        loss.backward()
        optimizer.step()

    with pytest.raises(RuntimeError, match="nonfinite backward loss"):
        capture_single_update(
            fit=fit, model_supplier=lambda: model, checkpoint=tmp_path / "step.pt"
        )
    assert torch.equal(before, model.weight)
    assert torch.autograd.backward is original_backward
    assert not (tmp_path / "step.pt").exists()


def test_preexisting_gradients_do_not_substitute_for_a_real_backward(tmp_path):
    model = torch.nn.Linear(1, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(RuntimeError, match="without observed backward loss"):
        capture_single_update(
            fit=optimizer.step, model_supplier=lambda: model, checkpoint=tmp_path / "step.pt"
        )
    assert not (tmp_path / "step.pt").exists()
