from __future__ import annotations

import pytest

from gradpert.hashing import canonical_json_bytes, sha256_json


def test_canonical_json_is_key_order_independent() -> None:
    left = {"b": [2, 1], "a": {"x": "基因"}}
    right = {"a": {"x": "基因"}, "b": [2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        canonical_json_bytes({"value": value})
