"""Explicit, all-or-none system optimizations for performance pilots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NativeSystemOptions:
    """Seven semantics-preserving optimizations and their bounded tunables."""

    merged_hdf5_reads: bool = False
    control_expression_cache: bool = False
    background_prefetch: bool = False
    resident_graph_tensors: bool = False
    validation_expression_cache: bool = False
    buffered_training_logs: bool = False
    single_checkpoint_serialization: bool = False
    pin_memory: bool = False
    nonblocking_transfer: bool = False
    prefetch_depth: int = 1
    log_buffer_steps: int = 64

    def __post_init__(self) -> None:
        primary = (
            self.merged_hdf5_reads,
            self.control_expression_cache,
            self.background_prefetch,
            self.resident_graph_tensors,
            self.validation_expression_cache,
            self.buffered_training_logs,
            self.single_checkpoint_serialization,
        )
        if any(primary) and not all(primary):
            raise ValueError("performance pilots require all seven system optimizations together")
        if self.background_prefetch:
            if not self.pin_memory or not self.nonblocking_transfer:
                raise ValueError("background prefetch requires pinned nonblocking transfer")
            if self.prefetch_depth not in {1, 2}:
                raise ValueError("prefetch depth must be one or two")
        elif self.pin_memory or self.nonblocking_transfer:
            raise ValueError("pinned/nonblocking transfer requires background prefetch")
        if self.buffered_training_logs:
            if not 1 <= self.log_buffer_steps <= 1024:
                raise ValueError("log buffer steps must be between 1 and 1024")
        elif self.log_buffer_steps != 64:
            raise ValueError("disabled systems use the inert default log buffer size")

    @property
    def enabled(self) -> bool:
        return self.merged_hdf5_reads

    def payload(self) -> dict[str, Any]:
        return asdict(self)


DISABLED_NATIVE_SYSTEM_OPTIONS = NativeSystemOptions()
