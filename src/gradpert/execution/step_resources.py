"""Post-update diagnostic resources; never a sustained-capacity guarantee."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from gradpert.execution.system_resources import host_available_memory_bytes


def capture_step_resources(device: Any, output_directory: Path) -> dict[str, Any]:
    """Read counters without resetting peaks or altering the allocator/model.

    CUDA peaks cover process initialization and the first update. A passing
    snapshot does not prove later-step capacity or exclude transient peer load.
    CPU fixtures are explicitly not CUDA acceptance evidence.
    """
    import torch

    device = torch.device(device)
    result: dict[str, Any] = {
        "device": str(device),
        "host_available_bytes": host_available_memory_bytes(),
        "disk_free_bytes": shutil.disk_usage(output_directory).free,
        "cuda_acceptance": False,
        "scope": "process_peak_and_post_update_snapshot_not_sustained_capacity",
    }
    if device.type != "cuda":
        return result
    if os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError("one-step CUDA requires expandable_segments:True")
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    required = {"reserved_bytes.all.peak", "num_alloc_retries", "num_ooms"}
    if not required.issubset(stats):
        raise RuntimeError("CUDA resource counters unavailable")
    peak = int(stats["reserved_bytes.all.peak"])
    minimum = max(4 * 1024**3, (int(total) * 15 + 99) // 100)
    headroom = min(int(free), int(total) - peak)
    result.update(
        cuda_free_bytes=int(free),
        cuda_total_bytes=int(total),
        peak_reserved_bytes=peak,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        allocator_retries=int(stats["num_alloc_retries"]),
        allocator_ooms=int(stats["num_ooms"]),
        conservative_headroom_bytes=headroom,
        required_headroom_bytes=minimum,
    )
    result["cuda_acceptance"] = (
        headroom >= minimum and result["allocator_retries"] == 0 and result["allocator_ooms"] == 0
    )
    if not result["cuda_acceptance"]:
        raise RuntimeError("one-step resource gate failed: " + json.dumps(result, sort_keys=True))
    return result
