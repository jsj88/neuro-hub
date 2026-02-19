"""
Classification functions for T-maze analysis.

Extracted and cleaned from 5 years of T-maze analysis notebooks.
Core patterns: LDA, SVM with cross-validation, per-ROI classification.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import warnings

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    StratifiedKFold,
    LeaveOneGroupOut
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from ..core.containers import TMazeEEGData, TMAzefMRIData, TMazeSubject


@dataclass
class ClassificationResult:
    """Container for classification results."""
    accuracy: float
    accuracy_std: float
    auc: Optional[float] = None
    auc_std: Optional[float] = None
    cv_scores: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    feature_importances: Optional[np.ndarray] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self):
        return (f"ClassificationResult(accuracy={self.accuracy:.1%} ± {self.accuracy_std:.1%}, "
                f"auc={self.auc:.3f})" if self.auc else
                f"ClassificationResult(accuracy={self.accuracy:.1%} ± {self.accuracy_std:.1%})")


class TMazeClassifier:
    """
    Main classifier for T-maze decoding analysis.

    Implements the CLASSIFICATION_CV_v3 pattern from T-maze notebooks.

    Parameters
    ----------
    classifier_type : str
        'lda', 'svm', 'logistic', or 'rf'
    cv : int or sklearn CV object
        Cross-validation strategy (default: 5-fold stratified)
    standardize : bool
        Whether to z-score features (default: True)
    return_probs : bool
        Whether to return probability estimates (default: True)
    n_jobs : int
        Parallel jobs for CV (default: -1)
    random_state : int
        Random seed (default: 42)
    **kwargs
        Additional parameters for the classifier
    """

    def __init__(
        self,
        classifier_type: str = 'lda',
        cv: Union[int, Any] = 5,
        standardize: bool = True,
        return_probs: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs
    ):
        self.classifier_type = classifier_type.lower()
        self.cv = cv if not isinstance(cv, int) else StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        self.standardize = standardize
        self.return_probs = return_probs
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs

        self.pipeline_ = None
        self.is_fitted_ = False

    def _create_pipeline(self):
        """Create sklearn pipeline based on classifier type."""
        if self.classifier_type == 'lda':
            clf = LDA(
                n_components=1,
                solver='lsqr',
                shrinkage='auto',
                **self.kwargs
            )
        elif self.classifier_type == 'svm':
            clf = SVC(
                kernel=self.kwargs.get('kernel', 'linear'),
                C=self.kwargs.get('C', 1.0),
                probability=True,
                random_state=self.random_state,
                **{k: v for k, v in self.kwargs.items()
                   if k not in ['kernel', 'C']}
            )
        elif self.classifier_type == 'logistic':
            clf = LogisticRegression(
                penalty=self.kwargs.get('penalty', 'l2'),
                C=self.kwargs.get('C', 1.0),
                max_iter=1000,
                random_state=self.random_state,
                **{k: v for k, v in self.kwargs.items()
                   if k not in ['penalty', 'C']}
            )
        elif self.classifier_type == 'rf':
            clf = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', None),
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **{k: v for k, v in self.kwargs.items()
                   if k not in ['n_estimators', 'max_depth']}
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        if self.standardize:
            return make_pipeline(StandardScaler(), clf)
        return clf

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TMazeClassifier':
        """Fit the classifier."""
        self.pipeline_ = self._create_pipeline()
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted. Call fit() first.")
        return self.pipeline_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Classifier not fitted. Call fit() first.")
        return self.pipeline_.predict_proba(X)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> ClassificationResult:
        """
        Run cross-validated classification.

        This is the core CLASSIFICATION_CV_v3 pattern.

        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Labels (n_samples,)
        groups : np.ndarray, optional
            Group labels for leave-one-group-out CV

        Returns
        -------
        ClassificationResult
            Classification results with accuracy, AUC, predictions
        """
        pipeline = self._create_pipeline()

        # Use appropriate CV
        cv = self.cv
        if groups is not None:
            cv = LeaveOneGroupOut()

        # Cross-validation scores
        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=cv, groups=groups,
            scoring='accuracy',
            n_jobs=self.n_jobs
        )

        # Get predictions
        predictions = cross_val_predict(
            pipeline, X, y,
            cv=cv, groups=groups,
            n_jobs=self.n_jobs
        )

        # Get probabilities if available
        probabilities = None
        auc = None
        auc_std = None

        if self.return_probs:
            try:
                probabilities = cross_val_predict(
                    pipeline, X, y,
                    cv=cv, groups=groups,
                    method='predict_proba',
                    n_jobs=self.n_jobs
                )

                # Calculate AUC
                auc_scores = []
                for train_idx, test_idx in cv.split(X, y, groups):
                    p = pipeline.fit(X[train_idx], y[train_idx])
                    proba = p.predict_proba(X[test_idx])
                    if proba.shape[1] == 2:
                        auc_scores.append(roc_auc_score(y[test_idx], proba[:, 1]))

                if auc_scores:
                    auc = np.mean(auc_scores)
                    auc_std = np.std(auc_scores)
            except Exception:
                pass

        # Confusion matrix
        cm = confusion_matrix(y, predictions)

        # Fit final model for feature importances
        pipeline.fit(X, y)
        importances = self._get_feature_importances(pipeline)

        return ClassificationResult(
            accuracy=np.mean(cv_scores),
            accuracy_std=np.std(cv_scores),
            auc=auc,
            auc_std=auc_std,
            cv_scores=cv_scores,
            predictions=predictions,
            probabilities=probabilities,
            confusion_matrix=cm,
            feature_importances=importances,
            metadata={
                'classifier': self.classifier_type,
                'n_samples': len(y),
                'n_features': X.shape[1],
                'cv_folds': len(cv_scores)
            }
        )

    def _get_feature_importances(self, pipeline) -> Optional[np.ndarray]:
        """Extract feature importances from fitted pipeline."""
        try:
            clf = pipeline[-1] if hasattr(pipeline, '__getitem__') else pipeline

            if hasattr(clf, 'coef_'):
                return np.abs(clf.coef_).ravel()
            elif hasattr(clf, 'feature_importances_'):
                return clf.feature_importances_
            elif hasattr(clf, 'scalings_'):  # LDA
                return np.abs(clf.scalings_).ravel()
        except Exception:
            pass
        return None


def classify_roi(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str = 'lda',
    cv: int = 5,
    return_probs: bool = True,
    **kwargs
) -> ClassificationResult:
    """
    Classify single ROI or feature set.

    Convenience function wrapping TMazeClassifier.

    Parameters
    ----------
    X : np.ndarray
        Features (n_samples, n_features) or (n_samples,) for single feature
    y : np.ndarray
        Labels
    classifier_type : str
        'lda', 'svm', 'logistic', 'rf'
    cv : int
        Number of CV folds
    return_probs : bool
        Return probability estimates
    **kwargs
        Additional classifier parameters

    Returns
    -------
    ClassificationResult
    """
    # Ensure 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    clf = TMazeClassifier(
        classifier_type=classifier_type,
        cv=cv,
        return_probs=return_probs,
        **kwargs
    )

    return clf.cross_validate(X, y)


def classify_all_rois(
    fmri_data: TMAzefMRIData,
    classifier_type: str = 'lda',
    cv: int = 5,
    n_jobs: int = -1,
    verbose: bool = True
) -> Dict[str, ClassificationResult]:
    """
    Classify each ROI independently.

    This is the per-region analysis pattern from T-maze notebooks.

    Parameters
    ----------
    fmri_data : TMAzefMRIData
        fMRI data container with ROI values
    classifier_type : str
        Classifier type
    cv : int
        Number of CV folds
    n_jobs : int
        Parallel jobs (not yet implemented for ROI loop)
    verbose : bool
        Print progress

    Returns
    -------
    Dict[str, ClassificationResult]
        Results for each ROI
    """
    results = {}

    for i, roi_name in enumerate(fmri_data.roi_names):
        if verbose and i % 50 == 0:
            print(f"Processing ROI {i+1}/{fmri_data.n_rois}: {roi_name}")

        X = fmri_data.data[:, i].reshape(-1, 1)
        y = fmri_data.labels

        result = classify_roi(X, y, classifier_type=classifier_type, cv=cv)
        result.metadata['roi_name'] = roi_name
        result.metadata['roi_index'] = i

        results[roi_name] = result

    return results


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    classifier_type: str = 'lda',
    cv_type: str = 'stratified',
    n_splits: int = 5,
    **kwargs
) -> ClassificationResult:
    """
    Run cross-validation with specified CV strategy.

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    groups : np.ndarray, optional
        Group labels for group-aware CV
    classifier_type : str
        Classifier type
    cv_type : str
        'stratified', 'logo' (leave-one-group-out), or 'loo'
    n_splits : int
        Number of folds for stratified CV
    **kwargs
        Additional classifier parameters

    Returns
    -------
    ClassificationResult
    """
    if cv_type == 'stratified':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_type == 'logo':
        cv = LeaveOneGroupOut()
    elif cv_type == 'loo':
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")

    clf = TMazeClassifier(
        classifier_type=classifier_type,
        cv=cv,
        **kwargs
    )

    return clf.cross_validate(X, y, groups=groups)


def get_top_rois(
    roi_results: Dict[str, ClassificationResult],
    metric: str = 'accuracy',
    n_top: int = 20,
    min_accuracy: float = 0.55
) -> List[Tuple[str, ClassificationResult]]:
    """
    Get top performing ROIs.

    Parameters
    ----------
    roi_results : Dict[str, ClassificationResult]
        Results from classify_all_rois
    metric : str
        'accuracy' or 'auc'
    n_top : int
        Number of top ROIs to return
    min_accuracy : float
        Minimum accuracy threshold

    Returns
    -------
    List[Tuple[str, ClassificationResult]]
        Sorted list of (roi_name, result) tuples
    """
    filtered = [
        (name, res) for name, res in roi_results.items()
        if res.accuracy >= min_accuracy
    ]

    if metric == 'auc':
        sorted_results = sorted(
            filtered,
            key=lambda x: x[1].auc if x[1].auc else 0,
            reverse=True
        )
    else:
        sorted_results = sorted(
            filtered,
            key=lambda x: x[1].accuracy,
            reverse=True
        )

    return sorted_results[:n_top]


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str = 'lda',
    n_permutations: int = 1000,
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Permutation test for classification significance.

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    classifier_type : str
        Classifier type
    n_permutations : int
        Number of permutations
    cv : int
        CV folds
    n_jobs : int
        Parallel jobs
    random_state : int
        Random seed

    Returns
    -------
    observed : float
        Observed accuracy
    p_value : float
        P-value from permutation test
    null_distribution : np.ndarray
        Null distribution of accuracies
    """
    from sklearn.model_selection import permutation_test_score

    clf = TMazeClassifier(classifier_type=classifier_type, cv=cv)
    pipeline = clf._create_pipeline()

    observed, null_dist, p_value = permutation_test_score(
        pipeline, X, y,
        cv=clf.cv,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        random_state=random_state,
        scoring='accuracy'
    )

    return observed, p_value, null_dist
