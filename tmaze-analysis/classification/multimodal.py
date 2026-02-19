"""
Multimodal EEG-fMRI fusion for T-maze classification.

Implements early fusion (feature concatenation) and late fusion (classifier averaging).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..core.containers import TMazeEEGData, TMAzefMRIData, TMazeSubject
from .classifiers import TMazeClassifier, ClassificationResult


@dataclass
class MultimodalResult:
    """Container for multimodal classification results."""
    accuracy: float
    accuracy_std: float
    eeg_only_accuracy: float
    fmri_only_accuracy: float
    fusion_improvement: float
    cv_scores: np.ndarray
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Compute improvement over best unimodal
        best_unimodal = max(self.eeg_only_accuracy, self.fmri_only_accuracy)
        self.fusion_improvement = self.accuracy - best_unimodal


def early_fusion(
    eeg_features: np.ndarray,
    fmri_features: np.ndarray,
    y: np.ndarray,
    classifier_type: str = 'svm',
    cv: int = 5,
    normalize_modalities: bool = True
) -> MultimodalResult:
    """
    Early fusion: concatenate EEG and fMRI features.

    Parameters
    ----------
    eeg_features : np.ndarray
        EEG features (n_samples, n_eeg_features)
    fmri_features : np.ndarray
        fMRI features (n_samples, n_fmri_features)
    y : np.ndarray
        Labels
    classifier_type : str
        Classifier type
    cv : int
        CV folds
    normalize_modalities : bool
        Z-score each modality separately before concatenation

    Returns
    -------
    MultimodalResult
    """
    # Ensure same number of samples
    n_samples = min(len(eeg_features), len(fmri_features), len(y))
    eeg_features = eeg_features[:n_samples]
    fmri_features = fmri_features[:n_samples]
    y = y[:n_samples]

    # Normalize each modality
    if normalize_modalities:
        scaler_eeg = StandardScaler()
        scaler_fmri = StandardScaler()
        eeg_norm = scaler_eeg.fit_transform(eeg_features)
        fmri_norm = scaler_fmri.fit_transform(fmri_features)
    else:
        eeg_norm = eeg_features
        fmri_norm = fmri_features

    # Concatenate features
    X_fused = np.hstack([eeg_norm, fmri_norm])

    # Classify
    clf = TMazeClassifier(classifier_type=classifier_type, cv=cv)

    # Fused results
    fused_result = clf.cross_validate(X_fused, y)

    # Compare to unimodal
    eeg_result = clf.cross_validate(eeg_features, y)
    fmri_result = clf.cross_validate(fmri_features, y)

    return MultimodalResult(
        accuracy=fused_result.accuracy,
        accuracy_std=fused_result.accuracy_std,
        eeg_only_accuracy=eeg_result.accuracy,
        fmri_only_accuracy=fmri_result.accuracy,
        fusion_improvement=fused_result.accuracy - max(
            eeg_result.accuracy, fmri_result.accuracy
        ),
        cv_scores=fused_result.cv_scores,
        metadata={
            'fusion_type': 'early',
            'classifier': classifier_type,
            'n_eeg_features': eeg_features.shape[1],
            'n_fmri_features': fmri_features.shape[1],
            'n_fused_features': X_fused.shape[1]
        }
    )


def late_fusion(
    eeg_features: np.ndarray,
    fmri_features: np.ndarray,
    y: np.ndarray,
    classifier_type: str = 'svm',
    cv: int = 5,
    weights: Tuple[float, float] = (0.5, 0.5)
) -> MultimodalResult:
    """
    Late fusion: average classifier predictions.

    Parameters
    ----------
    eeg_features : np.ndarray
        EEG features
    fmri_features : np.ndarray
        fMRI features
    y : np.ndarray
        Labels
    classifier_type : str
        Classifier type
    cv : int
        CV folds
    weights : Tuple[float, float]
        Weights for EEG and fMRI predictions

    Returns
    -------
    MultimodalResult
    """
    # Ensure same number of samples
    n_samples = min(len(eeg_features), len(fmri_features), len(y))
    eeg_features = eeg_features[:n_samples]
    fmri_features = fmri_features[:n_samples]
    y = y[:n_samples]

    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fused_scores = []
    eeg_scores = []
    fmri_scores = []

    for train_idx, test_idx in cv_obj.split(eeg_features, y):
        # Train EEG classifier
        clf_eeg = TMazeClassifier(classifier_type=classifier_type, cv=2)
        clf_eeg.fit(eeg_features[train_idx], y[train_idx])

        # Train fMRI classifier
        clf_fmri = TMazeClassifier(classifier_type=classifier_type, cv=2)
        clf_fmri.fit(fmri_features[train_idx], y[train_idx])

        # Get predictions
        eeg_proba = clf_eeg.predict_proba(eeg_features[test_idx])
        fmri_proba = clf_fmri.predict_proba(fmri_features[test_idx])

        # Weighted average of probabilities
        fused_proba = weights[0] * eeg_proba + weights[1] * fmri_proba
        fused_pred = np.argmax(fused_proba, axis=1)

        # Calculate accuracies
        fused_acc = np.mean(fused_pred == y[test_idx])
        eeg_acc = np.mean(clf_eeg.predict(eeg_features[test_idx]) == y[test_idx])
        fmri_acc = np.mean(clf_fmri.predict(fmri_features[test_idx]) == y[test_idx])

        fused_scores.append(fused_acc)
        eeg_scores.append(eeg_acc)
        fmri_scores.append(fmri_acc)

    return MultimodalResult(
        accuracy=np.mean(fused_scores),
        accuracy_std=np.std(fused_scores),
        eeg_only_accuracy=np.mean(eeg_scores),
        fmri_only_accuracy=np.mean(fmri_scores),
        fusion_improvement=np.mean(fused_scores) - max(
            np.mean(eeg_scores), np.mean(fmri_scores)
        ),
        cv_scores=np.array(fused_scores),
        metadata={
            'fusion_type': 'late',
            'classifier': classifier_type,
            'weights': weights
        }
    )


def multimodal_fusion(
    subject: TMazeSubject,
    fusion_type: str = 'early',
    eeg_feature_type: str = 'rewp_mean',
    classifier_type: str = 'svm',
    cv: int = 5,
    **kwargs
) -> MultimodalResult:
    """
    Multimodal EEG-fMRI fusion for a single subject.

    Parameters
    ----------
    subject : TMazeSubject
        Subject with both EEG and fMRI data
    fusion_type : str
        'early' or 'late'
    eeg_feature_type : str
        'rewp_mean': Mean REWP amplitude
        'all_times': All time points
        'peak': Peak amplitude only
    classifier_type : str
        Classifier type
    cv : int
        CV folds
    **kwargs
        Additional parameters for fusion functions

    Returns
    -------
    MultimodalResult
    """
    if not subject.is_multimodal:
        raise ValueError("Subject must have both EEG and fMRI data")

    # Extract EEG features based on type
    eeg_data = subject.eeg_data

    if eeg_feature_type == 'rewp_mean':
        # Mean amplitude in REWP window (240-340ms)
        rewp_data, _ = eeg_data.get_rewp_window()
        eeg_features = rewp_data.mean(axis=2)  # Mean over time
        eeg_features = eeg_features.mean(axis=1, keepdims=True)  # Mean over channels

    elif eeg_feature_type == 'all_times':
        # Flatten all time points
        eeg_features = eeg_data.data.reshape(eeg_data.n_epochs, -1)

    elif eeg_feature_type == 'peak':
        # Peak amplitude in REWP window
        rewp_data, _ = eeg_data.get_rewp_window()
        eeg_features = rewp_data.max(axis=2).mean(axis=1, keepdims=True)

    elif eeg_feature_type == 'fcz_rewp':
        # FCz channels in REWP window
        fcz_data = eeg_data.get_fcz_channels()
        time_mask = (eeg_data.times >= 0.24) & (eeg_data.times <= 0.34)
        eeg_features = fcz_data[:, :, time_mask].mean(axis=(1, 2), keepdims=True)

    else:
        raise ValueError(f"Unknown eeg_feature_type: {eeg_feature_type}")

    # Get fMRI features
    fmri_features = subject.fmri_data.data

    # Get labels (from fMRI, assuming trial alignment)
    y = subject.fmri_data.labels

    # Align samples
    n_samples = min(len(eeg_features), len(fmri_features))
    eeg_features = eeg_features[:n_samples]
    fmri_features = fmri_features[:n_samples]
    y = y[:n_samples]

    # Run fusion
    if fusion_type == 'early':
        return early_fusion(
            eeg_features, fmri_features, y,
            classifier_type=classifier_type,
            cv=cv,
            **kwargs
        )
    elif fusion_type == 'late':
        return late_fusion(
            eeg_features, fmri_features, y,
            classifier_type=classifier_type,
            cv=cv,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")


def compare_fusion_methods(
    subject: TMazeSubject,
    classifier_types: List[str] = ['lda', 'svm'],
    cv: int = 5
) -> Dict[str, MultimodalResult]:
    """
    Compare different fusion methods and classifiers.

    Parameters
    ----------
    subject : TMazeSubject
        Multimodal subject data
    classifier_types : List[str]
        Classifiers to try
    cv : int
        CV folds

    Returns
    -------
    Dict[str, MultimodalResult]
        Results for each method/classifier combination
    """
    results = {}

    for clf_type in classifier_types:
        # Early fusion
        key = f"early_{clf_type}"
        results[key] = multimodal_fusion(
            subject,
            fusion_type='early',
            classifier_type=clf_type,
            cv=cv
        )

        # Late fusion
        key = f"late_{clf_type}"
        results[key] = multimodal_fusion(
            subject,
            fusion_type='late',
            classifier_type=clf_type,
            cv=cv
        )

    return results


def group_multimodal_analysis(
    subjects: List[TMazeSubject],
    fusion_type: str = 'early',
    classifier_type: str = 'svm',
    cv: int = 5
) -> Dict[str, Union[float, List[MultimodalResult]]]:
    """
    Group-level multimodal analysis.

    Parameters
    ----------
    subjects : List[TMazeSubject]
        List of multimodal subjects
    fusion_type : str
        Fusion type
    classifier_type : str
        Classifier type
    cv : int
        CV folds

    Returns
    -------
    Dict
        Group statistics and per-subject results
    """
    # Filter to multimodal subjects
    multimodal_subjects = [s for s in subjects if s.is_multimodal]

    if not multimodal_subjects:
        raise ValueError("No multimodal subjects found")

    # Run per-subject analysis
    subject_results = []
    for subj in multimodal_subjects:
        result = multimodal_fusion(
            subj,
            fusion_type=fusion_type,
            classifier_type=classifier_type,
            cv=cv
        )
        subject_results.append(result)

    # Compute group statistics
    fused_accuracies = [r.accuracy for r in subject_results]
    eeg_accuracies = [r.eeg_only_accuracy for r in subject_results]
    fmri_accuracies = [r.fmri_only_accuracy for r in subject_results]
    improvements = [r.fusion_improvement for r in subject_results]

    return {
        'n_subjects': len(multimodal_subjects),
        'subject_ids': [s.subject_id for s in multimodal_subjects],
        'fused_mean': np.mean(fused_accuracies),
        'fused_std': np.std(fused_accuracies),
        'eeg_mean': np.mean(eeg_accuracies),
        'eeg_std': np.std(eeg_accuracies),
        'fmri_mean': np.mean(fmri_accuracies),
        'fmri_std': np.std(fmri_accuracies),
        'improvement_mean': np.mean(improvements),
        'improvement_std': np.std(improvements),
        'subject_results': subject_results
    }
