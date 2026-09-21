"""GraD-Pert v2. Importing this module does not alter the v1 model lifecycle."""

from gradpert.config.v2 import V2Architecture

from .model import GraDPertV2

__all__ = ["GraDPertV2", "V2Architecture"]
