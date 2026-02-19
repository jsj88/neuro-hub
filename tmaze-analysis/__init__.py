"""
T-Maze EEG-fMRI Classification Analysis Toolkit

Consolidated analysis framework for T-maze reward learning paradigm.
Supports EEG temporal decoding, fMRI ROI classification, multimodal fusion,
group-level statistics, deep learning, connectivity analysis, and pipeline automation.
"""

__version__ = "0.2.0"
__author__ = "Jaleesa Stringfellow"

# Core modules
from . import core
from . import io
from . import classification
from . import rsa
from . import visualization

# New modules (v0.2.0)
from . import statistics
from . import connectivity
from . import deeplearning
from . import pipelines

__all__ = [
    'core',
    'io',
    'classification',
    'rsa',
    'visualization',
    'statistics',
    'connectivity',
    'deeplearning',
    'pipelines'
]
