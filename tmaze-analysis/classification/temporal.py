"""
Temporal decoding for T-maze EEG analysis.

Time-resolved classification to identify when reward information emerges.
Based on MNE SlidingEstimator patterns from T-maze notebooks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import warnings

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

try:
    from mne.decoding import (
        SlidingEstimator,
        GeneralizingEstimator,
        cross_val_multiscore
    )
    HAS_MNE_DECODING = True
except ImportError:
    HAS_MNE_DECODING = False
    warnings.warn("MNE decoding not available. Install mne for temporal decoding.")

from scipy import stats
from scipy.ndimage import label as scipy_label

from ..core.containers import TMazeEEGData


@dataclass
class TemporalDecodingResult:
    """Container for temporal decoding results."""
    times: np.ndarray
    scores: np.ndarray
    scores_std: np.ndarray
    cv_scores: np.ndarray  # (n_folds, n_times)
    significant_times: Optional[np.ndarray] = None
    significant_clusters: Optional[List[Tuple[float, float]]] = None
    p_values: Optional[np.ndarray] = None
    peak_time: Optional[float] = None
    peak_score: Optional[float] = None
    onset_time: Optional[float] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Compute peak
        self.peak_time = self.times[np.argmax(self.scores)]
        self.peak_score = np.max(self.scores)


def temporal_decoding(
    eeg_data: TMazeEEGData,
    classifier_type: str = 'svm',
    scoring: str = 'roc_auc',
    cv: int = 5,
    n_jobs: int = -1,
    verbose: bool = True
) -> TemporalDecodingResult:
    """
    Time-resolved decoding using MNE SlidingEstimator.

    Parameters
    ----------
    eeg_data : TMazeEEGData
        EEG data container
    classifier_type : str
        'svm' or 'lda'
    scoring : str
        Scoring metric ('roc_auc', 'accuracy')
    cv : int
        Number of CV folds
    n_jobs : int
        Parallel jobs
    verbose : bool
        Print progress

    Returns
    -------
    TemporalDecodingResult
        Time-resolved classification results
    """
    if not HAS_MNE_DECODING:
        raise ImportError("MNE required for temporal decoding")

    X = eeg_data.data  # (n_epochs, n_channels, n_times)
    y = eeg_data.labels
    times = eeg_data.times

    # Create classifier pipeline
    if classifier_type == 'svm':
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', probability=True)
        )
    elif classifier_type == 'lda':
        clf = make_pipeline(
            StandardScaler(),
            LDA(solver='lsqr', shrinkage='auto')
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    # Create sliding estimator
    time_decod = SlidingEstimator(
        clf,
        n_jobs=n_jobs,
        scoring=scoring,
        verbose=verbose
    )

    # Cross-validate
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_multiscore(
        time_decod, X, y,
        cv=cv_obj,
        n_jobs=n_jobs
    )  # (n_folds, n_times)

    scores = cv_scores.mean(axis=0)
    scores_std = cv_scores.std(axis=0)

    result = TemporalDecodingResult(
        times=times,
        scores=scores,
        scores_std=scores_std,
        cv_scores=cv_scores,
        metadata={
            'classifier': classifier_type,
            'scoring': scoring,
            'cv_folds': cv,
            'n_epochs': eeg_data.n_epochs,
            'n_channels': eeg_data.n_channels
        }
    )

    return result


def temporal_generalization(
    eeg_data: TMazeEEGData,
    classifier_type: str = 'svm',
    scoring: str = 'roc_auc',
    cv: int = 5,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Temporal generalization analysis (train on time t, test on time t').

    Parameters
    ----------
    eeg_data : TMazeEEGData
        EEG data container
    classifier_type : str
        'svm' or 'lda'
    scoring : str
        Scoring metric
    cv : int
        Number of CV folds
    n_jobs : int
        Parallel jobs
    verbose : bool
        Print progress

    Returns
    -------
    gen_matrix : np.ndarray
        Generalization matrix (n_times, n_times)
    times : np.ndarray
        Time vector
    """
    if not HAS_MNE_DECODING:
        raise ImportError("MNE required for temporal generalization")

    X = eeg_data.data
    y = eeg_data.labels
    times = eeg_data.times

    # Create classifier
    if classifier_type == 'svm':
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='linear', probability=True)
        )
    else:
        clf = make_pipeline(
            StandardScaler(),
            LDA(solver='lsqr', shrinkage='auto')
        )

    # Generalizing estimator
    gen = GeneralizingEstimator(
        clf,
        n_jobs=n_jobs,
        scoring=scoring,
        verbose=verbose
    )

    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_multiscore(gen, X, y, cv=cv_obj, n_jobs=n_jobs)

    gen_matrix = cv_scores.mean(axis=0)

    return gen_matrix, times


def find_significant_times(
    result: TemporalDecodingResult,
    chance_level: float = 0.5,
    alpha: float = 0.05,
    method: str = 'ttest'
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Find time points with significant decoding.

    Parameters
    ----------
    result : TemporalDecodingResult
        Temporal decoding results
    chance_level : float
        Chance level for comparison (0.5 for AUC/accuracy)
    alpha : float
        Significance threshold
    method : str
        'ttest' or 'cluster'

    Returns
    -------
    significant_mask : np.ndarray
        Boolean mask of significant time points
    clusters : List[Tuple[float, float]]
        List of (start_time, end_time) for significant clusters
    """
    scores = result.cv_scores  # (n_folds, n_times)
    times = result.times

    if method == 'ttest':
        # One-sample t-test at each time point
        t_stats, p_values = stats.ttest_1samp(scores, chance_level, axis=0)
        significant_mask = p_values < alpha

    elif method == 'cluster':
        # Cluster-based permutation test (simplified)
        t_stats, p_values = stats.ttest_1samp(scores, chance_level, axis=0)

        # Find clusters of significant points
        threshold = stats.t.ppf(1 - alpha / 2, df=scores.shape[0] - 1)
        significant_mask = np.abs(t_stats) > threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    # Find contiguous clusters
    labeled_array, num_features = scipy_label(significant_mask)
    clusters = []

    for i in range(1, num_features + 1):
        cluster_mask = labeled_array == i
        cluster_times = times[cluster_mask]
        if len(cluster_times) > 0:
            clusters.append((cluster_times[0], cluster_times[-1]))

    # Update result
    result.significant_times = significant_mask
    result.significant_clusters = clusters
    result.p_values = p_values

    # Find onset (first significant time)
    if any(significant_mask):
        result.onset_time = times[np.argmax(significant_mask)]

    return significant_mask, clusters


def rewp_temporal_analysis(
    eeg_data: TMazeEEGData,
    fcz_only: bool = True,
    classifier_type: str = 'svm',
    cv: int = 5,
    n_jobs: int = -1
) -> TemporalDecodingResult:
    """
    Specialized temporal analysis for REWP (Reward Positivity).

    Focuses on frontocentral electrodes and 0-500ms window.

    Parameters
    ----------
    eeg_data : TMazeEEGData
        EEG data
    fcz_only : bool
        Use only FCz-region channels
    classifier_type : str
        Classifier type
    cv : int
        CV folds
    n_jobs : int
        Parallel jobs

    Returns
    -------
    TemporalDecodingResult
    """
    # Focus on FCz channels for REWP
    if fcz_only:
        fcz_channels = ['FCz', 'Fz', 'Cz', 'FC1', 'FC2']
        available = [ch for ch in fcz_channels if ch in eeg_data.channels]

        if available:
            channel_indices = [eeg_data.channels.index(ch) for ch in available]
            X = eeg_data.data[:, channel_indices, :]
        else:
            X = eeg_data.data
            warnings.warn("FCz channels not found, using all channels")
    else:
        X = eeg_data.data

    # Crop to REWP window (0-500ms)
    time_mask = (eeg_data.times >= 0) & (eeg_data.times <= 0.5)
    X = X[:, :, time_mask]
    times = eeg_data.times[time_mask]

    # Create temporary data container
    temp_data = TMazeEEGData(
        data=X,
        times=times,
        labels=eeg_data.labels,
        condition_names=eeg_data.condition_names,
        channels=available if fcz_only and available else eeg_data.channels,
        sfreq=eeg_data.sfreq,
        subject_id=eeg_data.subject_id
    )

    # Run temporal decoding
    result = temporal_decoding(
        temp_data,
        classifier_type=classifier_type,
        cv=cv,
        n_jobs=n_jobs
    )

    # Find significant periods
    find_significant_times(result, chance_level=0.5, alpha=0.05)

    result.metadata['analysis'] = 'REWP'
    result.metadata['fcz_only'] = fcz_only

    return result


def sliding_window_classification(
    eeg_data: TMazeEEGData,
    window_size: float = 0.050,  # 50ms
    step_size: float = 0.010,   # 10ms
    classifier_type: str = 'lda',
    cv: int = 5
) -> TemporalDecodingResult:
    """
    Sliding window classification (without MNE).

    Manual implementation for when MNE is not available.

    Parameters
    ----------
    eeg_data : TMazeEEGData
        EEG data
    window_size : float
        Window size in seconds
    step_size : float
        Step size in seconds
    classifier_type : str
        Classifier type
    cv : int
        CV folds

    Returns
    -------
    TemporalDecodingResult
    """
    from .classifiers import TMazeClassifier

    X = eeg_data.data  # (n_epochs, n_channels, n_times)
    y = eeg_data.labels
    times = eeg_data.times
    sfreq = eeg_data.sfreq

    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)

    center_times = []
    all_scores = []

    # Slide window across time
    for start in range(0, X.shape[2] - window_samples, step_samples):
        end = start + window_samples
        center_time = times[start + window_samples // 2]

        # Extract window and flatten
        X_window = X[:, :, start:end].reshape(X.shape[0], -1)

        # Cross-validate
        clf = TMazeClassifier(classifier_type=classifier_type, cv=cv)
        result = clf.cross_validate(X_window, y)

        center_times.append(center_time)
        all_scores.append(result.cv_scores)

    center_times = np.array(center_times)
    all_scores = np.array(all_scores).T  # (n_folds, n_windows)

    return TemporalDecodingResult(
        times=center_times,
        scores=all_scores.mean(axis=0),
        scores_std=all_scores.std(axis=0),
        cv_scores=all_scores,
        metadata={
            'window_size': window_size,
            'step_size': step_size,
            'classifier': classifier_type
        }
    )
