"""Validation-only stop boundary around the frozen GEARS train function."""

from __future__ import annotations

import math
from types import FunctionType, MethodType


def train_with_patience(model, *, epochs, lr, weight_decay, patience, on_epoch):
    """Execute unchanged official bytecode with a bounded epoch iterator.

    The official loop computes train metrics then val metrics, and snapshots
    its best model after those calls. Stop at the NEXT epoch boundary so its
    best-model assignment and normal post-loop cleanup still execute. The
    namespace is private to this call; upstream module globals are untouched.
    """
    original = model.train.__func__
    namespace = dict(original.__globals__)
    compute = namespace["compute_metrics"]
    history = []
    calls = 0
    best = math.inf
    bad = 0

    def metrics(results):
        nonlocal calls, best, bad
        result = compute(results)
        calls += 1
        if calls % 2 == 0:
            value = float(result[0]["mse_de"])
            if not math.isfinite(value):
                raise RuntimeError("nonfinite official GEARS validation MSE")
            improved = value < best
            best, bad = (value, 0) if improved else (best, bad + 1)
            history.append(
                {
                    "epoch": len(history) + 1,
                    "val_mse_de": value,
                    "best": improved,
                    "non_improving_epochs": bad,
                }
            )
            on_epoch(list(history))
        return result

    def epoch_range(count):
        if count != epochs:
            raise RuntimeError("frozen GEARS epoch range contract changed")
        for index in range(count):
            if bad >= patience:
                return
            yield index
            if calls != (index + 1) * 2:
                raise RuntimeError("expected one train and one validation metric call per epoch")

    if "test_loader" in model.dataloader:
        raise ValueError("test loader forbidden during official GEARS fit")
    namespace.update(range=epoch_range, compute_metrics=metrics)
    function = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    MethodType(function, model)(epochs=epochs, lr=lr, weight_decay=weight_decay)
    if not history:
        raise RuntimeError("GEARS produced no validation history")
    return history
