"""One-batch CPU lookahead with consumption-bound view RNG state."""

from __future__ import annotations

import copy
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def prefetch_cpu(
    factory: Callable[[np.random.Generator], Iterator[T]],
    generator: np.random.Generator,
) -> Generator[T, None, None]:
    """Factory must use only CPU operations and the supplied private generator.

    Unconsumed lookahead never advances the checkpoint-visible generator.
    Worker exceptions propagate; closing waits for the sole outstanding CPU task.
    """
    private = copy.deepcopy(generator)
    iterator = factory(private)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradpert-v2-views")

    def prepare() -> tuple[T, dict[str, Any]]:
        item = next(iterator)
        return item, copy.deepcopy(dict(private.bit_generator.state))

    future = executor.submit(prepare)
    try:
        while True:
            try:
                item, state = future.result()
            except StopIteration:
                return
            generator.bit_generator.state = state
            future = executor.submit(prepare)
            yield item
    finally:
        future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
