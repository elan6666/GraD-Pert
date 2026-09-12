"""Diagnostic boundary after one official optimizer update, before validation."""

from pathlib import Path

from gradpert.execution.step_resources import capture_step_resources


class SingleUpdateComplete(RuntimeError):
    """Internal control flow; never interpret an arbitrary training error as success."""


def capture_single_update(*, fit, model_supplier, checkpoint):
    """Run one continuous official fit until its first model optimizer update.

    Separate diagnostic process only. The global hook is removed on all exits.
    Callers must still clean their adapter files and seal source/data receipts.
    The saved state is diagnostic, never a validation-selected best checkpoint.
    """
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise FileExistsError("diagnostic checkpoint must be fresh")
    captured = []
    backward_losses = []
    original_backward = torch.autograd.backward

    def observed_backward(tensors, *args, **kwargs):
        # Observe the actual autograd roots; Lightning may normalize its loss.
        # Do not claim these are unscaled model.loss outputs or change backward.
        roots = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
        if not roots:
            raise RuntimeError("single-update diagnostic observed no backward loss")
        values = []
        for root in roots:
            if not isinstance(root, torch.Tensor) or root.numel() != 1:
                raise RuntimeError("single-update diagnostic requires scalar backward roots")
            if not torch.isfinite(root).all():
                raise RuntimeError("nonfinite backward loss before optimizer update")
            values.append(float(root.detach().cpu().item()))
        backward_losses.extend(values)
        return original_backward(tensors, *args, **kwargs)

    def after_update(optimizer, args, kwargs):
        model = model_supplier()
        model_parameters = {id(p) for p in model.parameters()}
        parameters = [p for g in optimizer.param_groups for p in g["params"]]
        if not parameters or any(id(p) not in model_parameters for p in parameters):
            raise RuntimeError("unexpected optimizer in single-update diagnostic")
        if captured:
            raise RuntimeError("more than one optimizer update")
        if not backward_losses:
            raise RuntimeError("optimizer update without observed backward loss")
        if not any(p.grad is not None for p in parameters):
            raise RuntimeError("optimizer update had no gradients")
        for p in model.parameters():
            if not torch.isfinite(p).all() or (
                p.grad is not None and not torch.isfinite(p.grad).all()
            ):
                raise RuntimeError("nonfinite model or gradient after update")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        resources = capture_step_resources(parameters[0].device, checkpoint.parent)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_steps": 1,
                "completed_epochs": 0,
                "role": "diagnostic_after_step_not_best",
            },
            checkpoint,
        )
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)

        def compare(expected, actual):
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(expected.detach().cpu(), actual, rtol=0, atol=0)
                if not torch.isfinite(expected).all():
                    raise RuntimeError("nonfinite saved state")
            elif isinstance(expected, dict):
                if expected.keys() != actual.keys():
                    raise RuntimeError("checkpoint keys changed")
                for key in expected:
                    compare(expected[key], actual[key])
            elif isinstance(expected, (tuple, list)):
                if len(expected) != len(actual):
                    raise RuntimeError("checkpoint sequence changed")
                for left, right in zip(expected, actual, strict=True):
                    compare(left, right)
            elif expected != actual:
                raise RuntimeError("checkpoint metadata changed")

        compare(model.state_dict(), loaded["model"])
        compare(optimizer.state_dict(), loaded["optimizer"])
        captured.append(
            {
                "completed_steps": 1,
                "completed_epochs": 0,
                "checkpoint_serialization_exact": True,
                "resources": resources,
                "backward_root_losses": list(backward_losses),
                "backward_loss_scope": "actual_autograd_roots_may_be_framework_normalized",
            }
        )
        raise SingleUpdateComplete("official optimizer completed exactly one update")

    hook = register_optimizer_step_post_hook(after_update)
    torch.autograd.backward = observed_backward
    try:
        try:
            fit()
        except SingleUpdateComplete:
            if len(captured) != 1:
                raise RuntimeError("single-update signal without evidence") from None
        else:
            raise RuntimeError("official fit returned without its single-update boundary")
    finally:
        torch.autograd.backward = original_backward
        hook.remove()
    return captured[0]
