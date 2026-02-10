"""
Feature selection methods for neural decoding.

Provides univariate and multivariate feature selection to reduce
dimensionality and improve classifier performance.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset


class BaseSelector(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for feature selectors."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSelector":
        """Fit selector to data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features from data."""
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    @abstractmethod
    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        pass

    def transform_dataset(self, dataset: DecodingDataset) -> DecodingDataset:
        """Transform a DecodingDataset with feature selection."""
        X_selected = self.transform(dataset.X)

        # Get selected feature names
        mask = self.get_support()
        feature_names = None
        if dataset.feature_names is not None:
            feature_names = [
                name for name, selected in zip(dataset.feature_names, mask)
                if selected
            ]

        return DecodingDataset(
            X=X_selected,
            y=dataset.y,
            groups=dataset.groups,
            feature_names=feature_names,
            class_names=dataset.class_names,
            metadata={
                **dataset.metadata,
                "feature_selection": self.__class__.__name__,
                "n_features_original": dataset.n_features,
                "n_features_selected": X_selected.shape[1]
            },
            modality=dataset.modality
        )


class ANOVASelector(BaseSelector):
    """
    ANOVA F-test feature selection.

    Selects features with the highest F-scores from one-way ANOVA.
    Fast univariate method suitable for initial feature reduction.

    Example:
        >>> selector = ANOVASelector(k=500)
        >>> selector.fit(X_train, y_train)
        >>> X_selected = selector.transform(X_test)
    """

    def __init__(
        self,
        k: int = 500,
        percentile: Optional[int] = None
    ):
        """
        Initialize ANOVA selector.

        Args:
            k: Number of features to select (used if percentile is None)
            percentile: Select top percentile of features (overrides k)
        """
        self.k = k
        self.percentile = percentile

        self.scores_ = None
        self.pvalues_ = None
        self.support_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANOVASelector":
        """Fit ANOVA selector."""
        from sklearn.feature_selection import f_classif

        # Compute F-scores
        self.scores_, self.pvalues_ = f_classif(X, y)

        # Handle NaN scores
        self.scores_ = np.nan_to_num(self.scores_, nan=0)

        # Determine number of features
        n_features = X.shape[1]
        if self.percentile is not None:
            k = int(n_features * self.percentile / 100)
        else:
            k = min(self.k, n_features)

        # Get top k indices
        top_indices = np.argsort(self.scores_)[::-1][:k]

        # Create support mask
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[top_indices] = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select top features."""
        if self.support_ is None:
            raise ValueError("Must call fit() first")
        return X[:, self.support_]

    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        return self.support_

    def get_scores(self) -> np.ndarray:
        """Get F-scores for all features."""
        return self.scores_

    def __repr__(self) -> str:
        n_selected = np.sum(self.support_) if self.support_ is not None else None
        return f"ANOVASelector(k={self.k}, selected={n_selected})"


class RFESelector(BaseSelector):
    """
    Recursive Feature Elimination.

    Iteratively removes the least important features based on
    classifier weights. More computationally expensive but often
    better than univariate methods.

    Example:
        >>> selector = RFESelector(n_features=100, step=0.1)
        >>> selector.fit(X_train, y_train)
        >>> X_selected = selector.transform(X_test)
    """

    def __init__(
        self,
        n_features: int = 100,
        step: float = 0.1,
        estimator=None,
        n_jobs: int = -1,
        verbose: int = 0
    ):
        """
        Initialize RFE selector.

        Args:
            n_features: Number of features to select
            step: Fraction of features to remove at each step
            estimator: Classifier with coef_ or feature_importances_
            n_jobs: Parallel jobs
            verbose: Verbosity level
        """
        self.n_features = n_features
        self.step = step
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.rfe_ = None
        self.support_ = None
        self.ranking_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFESelector":
        """Fit RFE selector."""
        from sklearn.feature_selection import RFE
        from sklearn.svm import LinearSVC

        # Default estimator
        estimator = self.estimator
        if estimator is None:
            estimator = LinearSVC(random_state=42, max_iter=10000)

        # Convert step to integer if needed
        step = self.step
        if isinstance(step, float) and step < 1:
            step = max(1, int(X.shape[1] * step))

        # Run RFE
        self.rfe_ = RFE(
            estimator=estimator,
            n_features_to_select=min(self.n_features, X.shape[1]),
            step=step,
            verbose=self.verbose
        )
        self.rfe_.fit(X, y)

        self.support_ = self.rfe_.support_
        self.ranking_ = self.rfe_.ranking_

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features."""
        if self.rfe_ is None:
            raise ValueError("Must call fit() first")
        return self.rfe_.transform(X)

    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        return self.support_

    def get_ranking(self) -> np.ndarray:
        """Get feature rankings (1 = selected, higher = eliminated earlier)."""
        return self.ranking_

    def __repr__(self) -> str:
        n_selected = np.sum(self.support_) if self.support_ is not None else None
        return f"RFESelector(n_features={self.n_features}, selected={n_selected})"


class StabilitySelector(BaseSelector):
    """
    Stability selection via subsampling.

    Selects features that are consistently selected across
    multiple bootstrap samples. More robust than single-pass methods.

    Example:
        >>> selector = StabilitySelector(n_bootstrap=100, threshold=0.6)
        >>> selector.fit(X_train, y_train)
        >>> X_selected = selector.transform(X_test)
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.75,
        threshold: float = 0.6,
        base_selector=None,
        n_jobs: int = -1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize stability selector.

        Args:
            n_bootstrap: Number of bootstrap samples
            sample_fraction: Fraction of samples per bootstrap
            threshold: Selection frequency threshold (0-1)
            base_selector: Selector to use on each bootstrap
            n_jobs: Parallel jobs
            random_state: Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.base_selector = base_selector
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.stability_scores_ = None
        self.support_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StabilitySelector":
        """Fit stability selector."""
        from joblib import Parallel, delayed
        import copy

        np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        subsample_size = int(n_samples * self.sample_fraction)

        # Default base selector
        if self.base_selector is None:
            base_selector = ANOVASelector(percentile=20)
        else:
            base_selector = self.base_selector

        def run_bootstrap(seed):
            """Run one bootstrap iteration."""
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_samples, subsample_size, replace=False)

            selector = copy.deepcopy(base_selector)
            selector.fit(X[indices], y[indices])
            return selector.get_support()

        # Run bootstraps in parallel
        seeds = np.random.randint(0, 2**31, size=self.n_bootstrap)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_bootstrap)(seed) for seed in seeds
        )

        # Calculate selection frequencies
        selection_counts = np.sum(results, axis=0)
        self.stability_scores_ = selection_counts / self.n_bootstrap

        # Apply threshold
        self.support_ = self.stability_scores_ >= self.threshold

        # Ensure at least one feature is selected
        if not np.any(self.support_):
            # Select top feature by stability
            self.support_[np.argmax(self.stability_scores_)] = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select stable features."""
        if self.support_ is None:
            raise ValueError("Must call fit() first")
        return X[:, self.support_]

    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        return self.support_

    def get_stability_scores(self) -> np.ndarray:
        """Get selection frequency for all features."""
        return self.stability_scores_

    def __repr__(self) -> str:
        n_selected = np.sum(self.support_) if self.support_ is not None else None
        return f"StabilitySelector(threshold={self.threshold}, selected={n_selected})"


class VarianceSelector(BaseSelector):
    """
    Remove features with low variance.

    Fast preprocessing step to remove constant or near-constant features.

    Example:
        >>> selector = VarianceSelector(threshold=0.01)
        >>> X_filtered = selector.fit_transform(X)
    """

    def __init__(self, threshold: float = 0.0):
        """
        Initialize variance selector.

        Args:
            threshold: Minimum variance to keep feature
        """
        self.threshold = threshold
        self.variances_ = None
        self.support_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "VarianceSelector":
        """Fit selector."""
        self.variances_ = np.var(X, axis=0)
        self.support_ = self.variances_ > self.threshold
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Remove low-variance features."""
        if self.support_ is None:
            raise ValueError("Must call fit() first")
        return X[:, self.support_]

    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        return self.support_

    def __repr__(self) -> str:
        n_selected = np.sum(self.support_) if self.support_ is not None else None
        return f"VarianceSelector(threshold={self.threshold}, selected={n_selected})"
