"""Optional, out-of-process experiment telemetry."""

from gradpert.tracking.trackio_sidecar import (
    TrackingGateError,
    TrackioSidecarConfig,
    run_trackio_sidecar,
)

__all__ = [
    "TrackingGateError",
    "TrackioSidecarConfig",
    "run_trackio_sidecar",
]
