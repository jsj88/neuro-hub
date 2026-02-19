"""Classification modules for T-maze analysis."""

from .classifiers import (
    TMazeClassifier,
    classify_roi,
    classify_all_rois,
    run_cross_validation
)
from .temporal import (
    temporal_decoding,
    temporal_generalization,
    find_significant_times
)
from .multimodal import (
    multimodal_fusion,
    early_fusion,
    late_fusion
)

__all__ = [
    # Core classifiers
    "TMazeClassifier",
    "classify_roi",
    "classify_all_rois",
    "run_cross_validation",
    # Temporal
    "temporal_decoding",
    "temporal_generalization",
    "find_significant_times",
    # Multimodal
    "multimodal_fusion",
    "early_fusion",
    "late_fusion"
]
