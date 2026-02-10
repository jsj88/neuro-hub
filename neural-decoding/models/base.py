"""
Base decoder class for neural decoding.

Provides the abstract interface that all decoder implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, List
import numpy as np

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset
from core.results import DecodingResults


class BaseDecoder(ABC):
    """
    Abstract base class for all neural decoders.

    All decoders must implement fit(), predict(), and score() methods.
    Provides cross_validate() and permutation_test() methods that work
    with any decoder implementation.

    Example:
        >>> class MyDecoder(BaseDecoder):
        ...     def fit(self, X, y):
        ...         # Training logic
        ...         return self
        ...     def predict(self, X):
        ...         # Prediction logic
        ...         return predictions
        ...     def score(self, X, y):
        ...         return accuracy
    """

    def __init__(self, random_state: Optional[int] = 42):
        """
        Initialize decoder.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.is_fitted_ = False
        self.classes_ = None
        self.n_features_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDecoder":
        """
        Fit the decoder on training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: Fitted decoder instance
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy on test data.

        Args:
            X: Test features (n_samples, n_features)
            y: True labels (n_samples,)

        Returns:
            Classification accuracy
        """
        pass

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (if supported).

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes) or None
        """
        return None

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances (if available).

        Returns:
            Feature importances (n_features,) or None
        """
        return None

    def cross_validate(
        self,
        dataset: DecodingDataset,
        cv=None,
        return_predictions: bool = True
    ) -> DecodingResults:
        """
        Perform cross-validation on dataset.

        Args:
            dataset: DecodingDataset with features and labels
            cv: Cross-validation splitter (default: 5-fold stratified)
            return_predictions: Whether to store all predictions

        Returns:
            DecodingResults with CV scores and metrics
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import confusion_matrix, accuracy_score
        import copy

        X, y = dataset.X, dataset.y

        # Default CV strategy
        if cv is None:
            if dataset.groups is not None:
                from sklearn.model_selection import LeaveOneGroupOut
                cv = LeaveOneGroupOut()
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True,
                                    random_state=self.random_state)

        # Get splits
        if hasattr(cv, 'get_n_splits'):
            if dataset.groups is not None:
                splits = list(cv.split(X, y, groups=dataset.groups))
            else:
                splits = list(cv.split(X, y))
        else:
            splits = list(cv.split(X, y))

        # Cross-validation
        cv_scores = []
        all_predictions = np.zeros(len(y), dtype=y.dtype)
        all_probabilities = None
        feature_importances = []

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone decoder for this fold
            fold_decoder = copy.deepcopy(self)
            fold_decoder.fit(X_train, y_train)

            # Score
            score = fold_decoder.score(X_test, y_test)
            cv_scores.append(score)

            # Predictions
            if return_predictions:
                all_predictions[test_idx] = fold_decoder.predict(X_test)

                # Probabilities
                proba = fold_decoder.predict_proba(X_test)
                if proba is not None:
                    if all_probabilities is None:
                        all_probabilities = np.zeros((len(y), proba.shape[1]))
                    all_probabilities[test_idx] = proba

            # Feature importances
            imp = fold_decoder.get_feature_importances()
            if imp is not None:
                feature_importances.append(imp)

        # Compute metrics
        accuracy = np.mean(cv_scores)
        cm = confusion_matrix(y, all_predictions)

        # Per-class accuracy
        class_names = dataset.class_names or [str(c) for c in np.unique(y)]
        accuracy_per_class = {}
        for i, name in enumerate(class_names):
            class_mask = y == np.unique(y)[i]
            if np.sum(class_mask) > 0:
                accuracy_per_class[name] = np.mean(
                    all_predictions[class_mask] == y[class_mask]
                )

        # Average feature importances
        avg_importances = None
        if feature_importances:
            avg_importances = np.mean(feature_importances, axis=0)

        return DecodingResults(
            accuracy=accuracy,
            accuracy_per_class=accuracy_per_class,
            confusion_matrix=cm,
            predictions=all_predictions if return_predictions else None,
            true_labels=y,
            probabilities=all_probabilities,
            feature_importances=avg_importances,
            cv_scores=cv_scores,
            class_names=class_names,
            metadata={
                "n_folds": len(cv_scores),
                "cv_strategy": cv.__class__.__name__
            }
        )

    def permutation_test(
        self,
        dataset: DecodingDataset,
        cv=None,
        n_permutations: int = 1000,
        n_jobs: int = -1
    ) -> DecodingResults:
        """
        Perform permutation test for statistical significance.

        Args:
            dataset: DecodingDataset
            cv: Cross-validation splitter
            n_permutations: Number of permutations
            n_jobs: Number of parallel jobs

        Returns:
            DecodingResults with permutation p-value
        """
        from sklearn.model_selection import permutation_test_score, StratifiedKFold

        X, y = dataset.X, dataset.y

        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=self.random_state)

        # Run permutation test
        score, perm_scores, pvalue = permutation_test_score(
            self._get_sklearn_estimator(),
            X, y,
            cv=cv,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            random_state=self.random_state,
            scoring='accuracy'
        )

        # Also get full CV results
        results = self.cross_validate(dataset, cv=cv)
        results.permutation_pvalue = pvalue
        results.permutation_scores = perm_scores
        results.metadata["n_permutations"] = n_permutations

        return results

    def _get_sklearn_estimator(self):
        """Get underlying sklearn estimator for permutation test."""
        raise NotImplementedError(
            "Subclass must implement _get_sklearn_estimator()"
        )

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"
