from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path,fit_marker",
    [
        ("benchmarks/gears/runner.py", "model = api.fit_one_epoch("),
        ("benchmarks/txpert/runner.py", "trainer = api.fit_one_epoch("),
    ],
)
def test_canonical_test_reader_is_constructed_only_after_official_fit(
    relative_path: str,
    fit_marker: str,
) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    fit_index = source.index(fit_marker)
    checkpoint_index = source.index("checkpoint_sha256 = sha256_file(", fit_index)
    training_receipt_index = source.index('small_root / "training_receipt.json"', fit_index)
    test_reader_index = source.index("with CanonicalEvaluationData(", fit_index)

    assert source.find("CanonicalEvaluationData(", 0, fit_index) == -1
    assert fit_index < checkpoint_index < training_receipt_index < test_reader_index
