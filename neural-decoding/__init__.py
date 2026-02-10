"""
Neural Decoding Toolkit for Human Neuroimaging

A Python toolkit for classification-based neural decoding of
pre-processed fMRI, EEG, and behavioral data.

Example:
    >>> from neural_decoding.io import FMRILoader
    >>> from neural_decoding.models import SVMDecoder
    >>>
    >>> loader = FMRILoader()
    >>> dataset = loader.load("data.nii.gz", "mask.nii.gz", "events.csv", "condition")
    >>>
    >>> decoder = SVMDecoder()
    >>> results = decoder.cross_validate(dataset)
    >>> print(f"Accuracy: {results.accuracy:.1%}")
"""

__version__ = "0.1.0"
__author__ = "Neuro-Hub"

# Core data structures
from .core import DecodingDataset, DecodingResults, DecodingConfig

# Data loaders
from .io import FMRILoader, EEGLoader, BehaviorLoader, MultimodalLoader

# Models
from .models import (
    BaseDecoder,
    SVMDecoder,
    RandomForestDecoder,
    LogisticDecoder,
    EnsembleDecoder,
    LDADecoder,
    SearchlightDecoder,
    TemporalDecoder,
    TemporalGeneralizationDecoder
)

# Feature extraction and selection
from .features import (
    ROIExtractor,
    VoxelExtractor,
    TimeWindowExtractor,
    TrialAverager,
    ANOVASelector,
    RFESelector,
    StabilitySelector
)

# Validation
from .validation import (
    LeaveOneRunOut,
    LeaveOneSubjectOut,
    StratifiedGroupKFold,
    PermutationTest
)

__all__ = [
    # Core
    "DecodingDataset",
    "DecodingResults",
    "DecodingConfig",
    # IO
    "FMRILoader",
    "EEGLoader",
    "BehaviorLoader",
    "MultimodalLoader",
    # Models
    "BaseDecoder",
    "SVMDecoder",
    "RandomForestDecoder",
    "LogisticDecoder",
    "EnsembleDecoder",
    "LDADecoder",
    "SearchlightDecoder",
    "TemporalDecoder",
    "TemporalGeneralizationDecoder",
    # Features
    "ROIExtractor",
    "VoxelExtractor",
    "TimeWindowExtractor",
    "TrialAverager",
    "ANOVASelector",
    "RFESelector",
    "StabilitySelector",
    # Validation
    "LeaveOneRunOut",
    "LeaveOneSubjectOut",
    "StratifiedGroupKFold",
    "PermutationTest",
]
