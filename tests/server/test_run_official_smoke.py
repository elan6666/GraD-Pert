from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_official_smoke_launcher_is_dry_run_by_default(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "server" / "run_official_smoke.py"),
            "--model",
            "txpert_public",
            "--dataset",
            "norman",
            "--project-root",
            str(ROOT),
            "--official-python",
            sys.executable,
            "--official-checkout",
            str(ROOT),
            "--data-root",
            str(ROOT),
            "--run-root",
            str(tmp_path / "runs"),
            "--device",
            "cuda:1",
            "--development-commit",
            "development-snapshot",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt = json.loads(completed.stdout)
    assert receipt["execute"] is False
    assert receipt["cwd"] == str(ROOT)
    assert receipt["pythonpath"] == f"{ROOT / 'src'}:{ROOT}"
    assert receipt["command"][:3] == [
        str(Path(sys.executable).resolve()),
        "-m",
        "benchmarks.txpert.runner",
    ]
    assert "--formal" not in receipt["command"]
    assert receipt["command"][-2:] == ["--development-commit", "development-snapshot"]
    assert not (tmp_path / "runs").exists()
