from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_stage_launcher_is_dry_run_by_default(tmp_path: Path) -> None:
    source = tmp_path / "runs"
    small = source / "full" / "gradpert_b2" / "norman" / "seed-1" / "small_results"
    small.mkdir(parents=True)
    (small / "metrics.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "stage"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "server" / "stage_small_results.py"),
            "--source-root",
            str(source),
            "--destination-root",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    payload = json.loads(completed.stdout)
    assert payload["file_count"] == 1
    assert not destination.exists()
