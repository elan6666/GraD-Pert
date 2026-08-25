from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmarks.txpert.runner import _write_official_cache


class FakeAdata:
    def __init__(self, *, base: object) -> None:
        self.X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.uns: dict[str, Any] = {"log1p": {"base": base}, "retained": {"value": 7}}

    def write_h5ad(self, path: str | Path) -> None:
        Path(path).write_bytes(b"adapted-h5ad-fixture")


def test_txpert_cache_drops_only_null_log1p_base(tmp_path: Path) -> None:
    adata = FakeAdata(base=None)
    before = adata.X.copy()
    adapted = SimpleNamespace(
        adata=adata,
        train_conditions=("A+ctrl",),
        val_conditions=("B+ctrl",),
    )

    receipt = _write_official_cache(cache_root=tmp_path / "cache", adapted=adapted)

    assert adata.uns == {"log1p": {}, "retained": {"value": 7}}
    np.testing.assert_array_equal(adata.X, before)
    assert receipt["h5ad_compatibility_policy"] == "drop_uns_log1p_base_only_when_null"
    assert receipt["removed_null_metadata_paths"] == ["/uns/log1p/base"]


def test_txpert_cache_preserves_non_null_log1p_base(tmp_path: Path) -> None:
    adata = FakeAdata(base=2.0)
    adapted = SimpleNamespace(
        adata=adata,
        train_conditions=("A+ctrl",),
        val_conditions=("B+ctrl",),
    )

    receipt = _write_official_cache(cache_root=tmp_path / "cache", adapted=adapted)

    assert adata.uns["log1p"] == {"base": 2.0}
    assert receipt["removed_null_metadata_paths"] == []
