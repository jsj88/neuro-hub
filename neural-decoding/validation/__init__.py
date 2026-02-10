"""Cross-validation, metrics, and permutation testing for neural decoding."""

from .cross_validation import (
    LeaveOneRunOut,
    LeaveOneSubjectOut,
    StratifiedGroupKFold
)
from .metrics import (
    compute_metrics,
    balanced_accuracy,
    sensitivity,
    specificity,
    cohens_kappa
)
from .permutation import PermutationTest

__all__ = [
    "LeaveOneRunOut",
    "LeaveOneSubjectOut",
    "StratifiedGroupKFold",
    "compute_metrics",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "cohens_kappa",
    "PermutationTest"
]
