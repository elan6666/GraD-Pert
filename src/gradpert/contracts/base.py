"""Shared strict types and validators for manifest models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
GitCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class StrictManifest(BaseModel):
    """Manifest base that rejects unknown fields and in-place mutation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
