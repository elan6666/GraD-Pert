import runpy
from pathlib import Path

import pytest

POLICY = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/v2/capacity_probe.py"))[
    "probe_policy"
]


def test_short_integration_evidence_is_a_separate_kind():
    assert POLICY(True, None) == (1, 0, "integration_only")
    assert POLICY(False, None) == (128, 8, "capacity_only")
    with pytest.raises(ValueError, match="128"):
        POLICY(False, 1)
    with pytest.raises(ValueError, match="exactly one"):
        POLICY(True, 128)
