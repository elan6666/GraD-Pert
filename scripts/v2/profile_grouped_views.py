"""CPU ABBA benchmark of pair-layout reuse on the medium probe shape.

Synthetic features isolate observer scheduling; this is not biological evidence,
GPU throughput, or an end-to-end training/medium-analysis speedup claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from grouped_view_diagnostics import GroupedViewBank


def execute(output: Path) -> dict:
    if output.exists():
        raise ValueError("preserve previous performance evidence")
    bank = GroupedViewBank()
    rng = np.random.default_rng(42)
    records = [
        dict(
            cell_line="Jurkat",
            cell_type="T",
            context_id="one_context",
            condition_id=f"p{i // 4}",
            control_row_id=f"c{i % 479}",
        )
        for i in range(506)
    ]
    for view in ("global0", "global1", "local0", "local1"):
        bank.add("ssl2_response_cls", "student", view, records, rng.normal(size=(506, 256)))
    measurements = []
    expected = None
    for reuse in (False, True, True, False):
        started = time.perf_counter()
        result = bank.result(reuse_pair_layout=reuse)
        seconds = time.perf_counter() - started
        digest = hashlib.sha256(
            json.dumps(result, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("reuse changed a diagnostic statistic or sample identity")
        expected = digest
        measurements.append(dict(reuse_pair_layout=reuse, seconds=seconds, output_sha256=digest))
    reference = np.median([m["seconds"] for m in measurements if not m["reuse_pair_layout"]])
    candidate = np.median([m["seconds"] for m in measurements if m["reuse_pair_layout"]])
    receipt = dict(
        schema="medium-view-layout-abba-1",
        passed=True,
        synthetic_features=True,
        shape=[506, 256],
        views=4,
        numpy=np.__version__,
        platform=platform.platform(),
        exact_aggregate_parity=True,
        measurements=measurements,
        reference_median_seconds=reference,
        candidate_median_seconds=candidate,
        speedup=reference / candidate,
        limitation="CPU observer component only; no end-to-end speedup or model-quality claim",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(execute(parser.parse_args().output), indent=2))
