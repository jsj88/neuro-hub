"""Configuration settings for T-maze analysis."""

from .settings import (
    TMazeConfig,
    EEGConfig,
    FMRIConfig,
    ClassificationConfig,
    get_default_config
)

__all__ = [
    "TMazeConfig",
    "EEGConfig",
    "FMRIConfig",
    "ClassificationConfig",
    "get_default_config"
]
