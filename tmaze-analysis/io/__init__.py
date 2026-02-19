"""Data loaders for T-maze analysis."""

from .loaders import (
    TMazeEEGLoader,
    TMazefMRILoader,
    TMazeSubjectLoader,
    load_hcp_atlas
)

__all__ = [
    "TMazeEEGLoader",
    "TMazefMRILoader",
    "TMazeSubjectLoader",
    "load_hcp_atlas"
]
