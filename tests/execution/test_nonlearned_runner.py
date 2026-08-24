from __future__ import annotations

from pathlib import Path

import pytest

from gradpert.execution.nonlearned import run_nonlearned_experiment

ROOT = Path(__file__).resolve().parents[2]


def test_nonlearned_runner_accepts_the_shared_seed_one_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "occupied"
    run_root.mkdir()
    (run_root / "sentinel").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new and empty"):
        run_nonlearned_experiment(
            config_path=(
                ROOT
                / "configs"
                / "experiments"
                / "matched_control_mean"
                / "replogle_k562_essential.yaml"
            ),
            data_root=tmp_path / "data",
            run_root=run_root,
            run_id="shared-seed-one-contract",
            repository_root=ROOT,
            formal=False,
            development_commit="development-snapshot",
        )
