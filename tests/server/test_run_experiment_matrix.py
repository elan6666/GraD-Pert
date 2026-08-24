from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_matrix_launcher_is_dry_run_and_writes_nothing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    gears_data = tmp_path / "gears-data"
    data_root.mkdir()
    gears_data.mkdir()
    runs_root = tmp_path / "runs"
    receipt_root = tmp_path / "receipts"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "server" / "run_experiment_matrix.py"),
            "--phase",
            "smoke",
            "--project-root",
            str(ROOT),
            "--data-root",
            str(data_root),
            "--runs-root",
            str(runs_root),
            "--native-python",
            sys.executable,
            "--gears-python",
            sys.executable,
            "--gears-checkout",
            str(ROOT),
            "--gears-data-root",
            str(gears_data),
            "--txpert-python",
            sys.executable,
            "--txpert-checkout",
            str(ROOT),
            "--device",
            "cuda:0",
            "--expected-commit",
            "a" * 40,
            "--development",
            "--receipt-root",
            str(receipt_root),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    payload = json.loads(completed.stdout)
    assert payload["task_count"] == 15
    assert payload["selected_for_execution"] == []
    assert not runs_root.exists()
    assert not receipt_root.exists()
