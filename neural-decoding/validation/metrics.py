"""
Performance metrics for neural decoding.

Provides classification metrics commonly used in neuroimaging studies.
"""

from typing import Dict, Optional, List
import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        cohen_kappa_score,
        confusion_matrix
    )

    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "cohens_kappa": cohen_kappa_score(y_true, y_pred),
        "n_classes": n_classes,
        "n_samples": len(y_true)
    }

    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)

    for i, cls in enumerate(classes):
        name = class_names[i] if class_names else str(cls)

        # Sensitivity (recall) for this class
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        metrics[f"sensitivity_{name}"] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity for this class
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - tp
        metrics[f"specificity_{name}"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced accuracy (average recall across classes).

    Useful for imbalanced datasets.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Balanced accuracy score
    """
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)


def sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Optional[int] = 1
) -> float:
    """
    Sensitivity (true positive rate / recall).

    For binary classification: TP / (TP + FN)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label

    Returns:
        Sensitivity score
    """
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)


def specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Optional[int] = 1
) -> float:
    """
    Specificity (true negative rate).

    For binary classification: TN / (TN + FP)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label

    Returns:
        Specificity score
    """
    # Swap positive and negative
    neg_label = [l for l in np.unique(y_true) if l != pos_label][0]

    # Specificity = sensitivity of negative class
    return sensitivity(y_true, y_pred, pos_label=neg_label)


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cohen's kappa (agreement corrected for chance).

    Values:
    - 1: Perfect agreement
    - 0: Agreement expected by chance
    - <0: Less than chance agreement

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Kappa score
    """
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred)


def auc_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    multi_class: str = "ovr"
) -> float:
    """
    Area under ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        multi_class: Strategy for multiclass ("ovr" or "ovo")

    Returns:
        AUC-ROC score
    """
    from sklearn.metrics import roc_auc_score

    n_classes = len(np.unique(y_true))

    if n_classes == 2:
        # Binary: use probability of positive class
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        return roc_auc_score(y_true, y_proba)
    else:
        # Multiclass
        return roc_auc_score(y_true, y_proba, multi_class=multi_class)


def chance_level(n_classes: int) -> float:
    """
    Get chance level accuracy for given number of classes.

    Args:
        n_classes: Number of classes

    Returns:
        Chance level (1 / n_classes)
    """
    return 1.0 / n_classes


def d_prime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Optional[int] = 1
) -> float:
    """
    d' (d-prime) sensitivity index from signal detection theory.

    Measures discriminability between classes.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label

    Returns:
        d' value
    """
    from scipy.stats import norm

    # Get hit rate (sensitivity) and false alarm rate (1 - specificity)
    hit_rate = sensitivity(y_true, y_pred, pos_label=pos_label)
    fa_rate = 1 - specificity(y_true, y_pred, pos_label=pos_label)

    # Avoid infinite values
    hit_rate = np.clip(hit_rate, 0.001, 0.999)
    fa_rate = np.clip(fa_rate, 0.001, 0.999)

    # d' = z(hit_rate) - z(false_alarm_rate)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)


def information_transfer_rate(
    accuracy: float,
    n_classes: int,
    trial_duration: float
) -> float:
    """
    Information transfer rate (ITR) for BCI applications.

    Measures bits of information transmitted per unit time.

    Args:
        accuracy: Classification accuracy (0-1)
        n_classes: Number of classes
        trial_duration: Duration of one trial in seconds

    Returns:
        ITR in bits per minute
    """
    if accuracy <= 0 or accuracy >= 1:
        return 0.0

    # Bits per trial (mutual information)
    p = accuracy
    q = (1 - p) / (n_classes - 1) if n_classes > 1 else 0

    bits_per_trial = (
        np.log2(n_classes) +
        p * np.log2(p) +
        (1 - p) * np.log2(q) if q > 0 else 0
    )

    # Convert to bits per minute
    trials_per_minute = 60.0 / trial_duration
    return max(0, bits_per_trial * trials_per_minute)
