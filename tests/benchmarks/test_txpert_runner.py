from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmarks.txpert.runner import _register_anndata_null_reader, _write_official_cache


class FakeAdata:
    def __init__(self, *, base: object) -> None:
        self.X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.uns: dict[str, Any] = {"log1p": {"base": base}, "retained": {"value": 7}}

    def write_h5ad(self, path: str | Path) -> None:
        Path(path).write_bytes(b"adapted-h5ad-fixture")


class FakeIOSpec:
    def __init__(self, encoding_type: str, encoding_version: str) -> None:
        self.encoding_type = encoding_type
        self.encoding_version = encoding_version


class FakeRegistry:
    def __init__(self, *, already_registered: bool) -> None:
        self.already_registered = already_registered
        self.registered_reader: object | None = None

    def has_read(self, source_type: type, spec: FakeIOSpec) -> bool:
        assert source_type.__name__ == "FakeDataset"
        assert (spec.encoding_type, spec.encoding_version) == ("null", "0.1.0")
        return self.already_registered

    def register_read(self, source_type: type, spec: FakeIOSpec):
        assert source_type.__name__ == "FakeDataset"
        assert (spec.encoding_type, spec.encoding_version) == ("null", "0.1.0")

        def register(reader: object) -> object:
            self.registered_reader = reader
            return reader

        return register


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


def test_frozen_anndata_null_reader_is_registered_with_exact_semantics(monkeypatch) -> None:
    class FakeDataset:
        pass

    registry = FakeRegistry(already_registered=False)
    modules = {
        "h5py": SimpleNamespace(Dataset=FakeDataset),
        "anndata._io.specs.registry": SimpleNamespace(
            _REGISTRY=registry,
            IOSpec=FakeIOSpec,
        ),
    }
    monkeypatch.setattr(
        "benchmarks.txpert.runner.importlib.import_module",
        lambda name: modules[name],
    )

    receipt = _register_anndata_null_reader()

    assert receipt == {
        "schema_version": "anndata-null-reader-compatibility-v1",
        "encoding_type": "null",
        "encoding_version": "0.1.0",
        "registered_at_runtime": True,
        "read_semantics": "return_none",
    }
    assert callable(registry.registered_reader)
    assert registry.registered_reader(object(), object()) is None
