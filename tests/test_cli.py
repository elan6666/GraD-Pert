from __future__ import annotations

import json
from pathlib import Path

from gradpert.cli import main


def test_doctor_json_is_explicitly_local(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["formal_compute_allowed"] is False
    assert payload["gradpert_version"]


def test_help_is_success(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main([]) == 0
    assert "GraD-Pert research CLI" in capsys.readouterr().out


def test_config_matrix_cli(capsys) -> None:  # type: ignore[no-untyped-def]
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    assert main(["config", "verify", "--all", "--json", "--root", str(root)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["count"] == 30


def test_data_status_reports_missing_without_mutation(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    registry = Path(__file__).resolve().parents[1] / "registry" / "datasets"
    assert (
        main(
            [
                "data",
                "status",
                "--dataset",
                "norman",
                "--json",
                "--data-root",
                str(tmp_path / "data"),
                "--registry-root",
                str(registry),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["entries"][0]["source_state"] == "missing"
    assert report["entries"][0]["canonical_state"] == "missing"
    assert not (tmp_path / "data").exists()
