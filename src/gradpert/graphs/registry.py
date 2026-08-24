"""Strict registry for the two frozen public graph source files."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictGraphRegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class GraphSourceFile(_StrictGraphRegistryModel):
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["csv", "parquet"]
    source_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    weight_column: str = Field(min_length=1)
    expected_empty_endpoint_rows: int = Field(ge=0)

    @model_validator(mode="after")
    def enforce_path_and_columns(self) -> GraphSourceFile:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("graph source relative_path must be safe and relative")
        if len({self.source_column, self.target_column, self.weight_column}) != 3:
            raise ValueError("graph source, target and weight columns must differ")
        expected_suffix = ".csv" if self.format == "csv" else ".parquet"
        if path.suffix != expected_suffix:
            raise ValueError("graph source path suffix differs from declared format")
        return self


class GraphSourceRegistry(_StrictGraphRegistryModel):
    schema_version: Literal["graph-source-registry-v1"]
    registry_id: Literal["public_string_go_v1"]
    repository: Literal["https://github.com/valence-labs/TxPert.git"]
    commit: Literal["08d82eea86746b044cf7531f4ec8c5f60e1cb73f"]
    license_notice_path: Literal["license.pdf"]
    use_scope: str = Field(min_length=1)
    sources: dict[Literal["go", "string"], GraphSourceFile]

    @model_validator(mode="after")
    def require_exact_sources(self) -> GraphSourceRegistry:
        if set(self.sources) != {"go", "string"}:
            raise ValueError("v1 graph registry requires exactly GO and STRING")
        if self.sources["go"].format != "csv" or self.sources["string"].format != ("parquet"):
            raise ValueError("v1 GO/STRING source formats are frozen")
        return self


def load_graph_source_registry(path: str | Path) -> GraphSourceRegistry:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("graph registry must be a regular non-symlink file")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("graph registry root must be a mapping")
    return GraphSourceRegistry.model_validate(payload)
