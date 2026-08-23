from __future__ import annotations

import json

from gradpert.cli import main


def test_doctor_json_is_explicitly_local(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["formal_compute_allowed"] is False
    assert payload["gradpert_version"]


def test_help_is_success(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main([]) == 0
    assert "GraD-Pert research CLI" in capsys.readouterr().out
