"""
Classifier implementations for neural decoding.

Provides SVM, Random Forest, Logistic Regression, and Ensemble decoders
optimized for neuroimaging data.
"""

from typing import Optional, List, Union, Any
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .base import BaseDecoder


class SVMDecoder(BaseDecoder):
    """
    Support Vector Machine decoder.

    Linear SVM is the default choice for high-dimensional neuroimaging data.
    The linear kernel provides interpretable weights and works well when
    n_features >> n_samples (typical for voxel-wise fMRI decoding).

    Example:
        >>> decoder = SVMDecoder(kernel="linear", C=1.0)
        >>> results = decoder.cross_validate(dataset)
        >>> print(f"Accuracy: {results.accuracy:.1%}")
    """

    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        gamma: str = "scale",
        standardize: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize SVM decoder.

        Args:
            kernel: Kernel type ("linear", "rbf", "poly")
            C: Regularization parameter (smaller = more regularization)
            gamma: Kernel coefficient for "rbf" and "poly"
            standardize: Whether to z-score features
            random_state: Random seed
        """
        super().__init__(random_state=random_state)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.standardize = standardize

        # Build pipeline
        self._build_pipeline()

    def _build_pipeline(self):
        """Build sklearn pipeline."""
        steps = []

        if self.standardize:
            steps.append(("scaler", StandardScaler()))

        svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state
        )
        steps.append(("svm", svm))

        self.pipeline_ = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMDecoder":
        """Fit SVM on training data."""
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        self.classes_ = self.pipeline_.classes_
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.pipeline_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return self.pipeline_.score(X, y)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances (SVM weights for linear kernel).

        Returns:
            For linear kernel: coefficient magnitudes
            For other kernels: None
        """
        if not self.is_fitted_:
            return None

        svm = self.pipeline_.named_steps["svm"]

        if self.kernel == "linear":
            # For binary: coef_ is (1, n_features)
            # For multiclass: coef_ is (n_classes, n_features)
            coef = svm.coef_
            if coef.ndim == 1:
                return np.abs(coef)
            else:
                # Average absolute weights across classes
                return np.mean(np.abs(coef), axis=0)

        return None

    def _get_sklearn_estimator(self):
        """Get sklearn estimator for permutation test."""
        return self.pipeline_

    def __repr__(self) -> str:
        return f"SVMDecoder(kernel={self.kernel}, C={self.C})"


class RandomForestDecoder(BaseDecoder):
    """
    Random Forest decoder.

    Good for capturing non-linear patterns and providing feature importances.
    Works well for ROI-based decoding where features are interpretable.

    Example:
        >>> decoder = RandomForestDecoder(n_estimators=100)
        >>> results = decoder.cross_validate(dataset)
        >>> top_features = results.get_top_features(n=20)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        standardize: bool = False,
        n_jobs: int = -1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Random Forest decoder.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_leaf: Minimum samples per leaf
            max_features: Features to consider per split
            standardize: Whether to z-score features
            n_jobs: Parallel jobs for fitting
            random_state: Random seed
        """
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.standardize = standardize
        self.n_jobs = n_jobs

        self._build_pipeline()

    def _build_pipeline(self):
        """Build sklearn pipeline."""
        steps = []

        if self.standardize:
            steps.append(("scaler", StandardScaler()))

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        steps.append(("rf", rf))

        self.pipeline_ = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestDecoder":
        """Fit Random Forest on training data."""
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        self.classes_ = self.pipeline_.classes_
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.pipeline_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return self.pipeline_.score(X, y)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Get Gini importances from fitted forest."""
        if not self.is_fitted_:
            return None

        rf = self.pipeline_.named_steps["rf"]
        return rf.feature_importances_

    def _get_sklearn_estimator(self):
        """Get sklearn estimator for permutation test."""
        return self.pipeline_

    def __repr__(self) -> str:
        return f"RandomForestDecoder(n_estimators={self.n_estimators})"


class LogisticDecoder(BaseDecoder):
    """
    Logistic Regression decoder.

    Fast, interpretable decoder with probabilistic outputs.
    L2 regularization (default) works well for high-dimensional data.

    Example:
        >>> decoder = LogisticDecoder(C=1.0)
        >>> results = decoder.cross_validate(dataset)
        >>> proba = decoder.predict_proba(X_test)
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        standardize: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Logistic Regression decoder.

        Args:
            C: Inverse regularization strength
            penalty: Regularization type ("l1", "l2", "elasticnet")
            solver: Optimization algorithm
            max_iter: Maximum iterations
            standardize: Whether to z-score features
            random_state: Random seed
        """
        super().__init__(random_state=random_state)
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.standardize = standardize

        self._build_pipeline()

    def _build_pipeline(self):
        """Build sklearn pipeline."""
        steps = []

        if self.standardize:
            steps.append(("scaler", StandardScaler()))

        # Adjust solver for penalty type
        solver = self.solver
        if self.penalty == "l1" and solver == "lbfgs":
            solver = "saga"
        elif self.penalty == "elasticnet":
            solver = "saga"

        lr = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        steps.append(("lr", lr))

        self.pipeline_ = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticDecoder":
        """Fit Logistic Regression on training data."""
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        self.classes_ = self.pipeline_.classes_
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.pipeline_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return self.pipeline_.score(X, y)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Get coefficient magnitudes as importances."""
        if not self.is_fitted_:
            return None

        lr = self.pipeline_.named_steps["lr"]
        coef = lr.coef_

        if coef.ndim == 1:
            return np.abs(coef)
        else:
            return np.mean(np.abs(coef), axis=0)

    def _get_sklearn_estimator(self):
        """Get sklearn estimator for permutation test."""
        return self.pipeline_

    def __repr__(self) -> str:
        return f"LogisticDecoder(C={self.C}, penalty={self.penalty})"


class EnsembleDecoder(BaseDecoder):
    """
    Ensemble decoder combining multiple classifiers.

    Combines predictions from multiple decoders using voting.
    Soft voting (probability averaging) typically works better than
    hard voting (majority vote).

    Example:
        >>> ensemble = EnsembleDecoder([
        ...     SVMDecoder(kernel="linear"),
        ...     RandomForestDecoder(),
        ...     LogisticDecoder()
        ... ], voting="soft")
        >>> results = ensemble.cross_validate(dataset)
    """

    def __init__(
        self,
        decoders: List[BaseDecoder],
        voting: str = "soft",
        weights: Optional[List[float]] = None,
        random_state: Optional[int] = 42
    ):
        """
        Initialize Ensemble decoder.

        Args:
            decoders: List of decoder instances to combine
            voting: Voting strategy ("soft" or "hard")
            weights: Optional weights for each decoder
            random_state: Random seed
        """
        super().__init__(random_state=random_state)
        self.decoders = decoders
        self.voting = voting
        self.weights = weights

        self._build_ensemble()

    def _build_ensemble(self):
        """Build sklearn VotingClassifier."""
        estimators = [
            (f"decoder_{i}", d._get_sklearn_estimator())
            for i, d in enumerate(self.decoders)
        ]

        self.ensemble_ = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleDecoder":
        """Fit all decoders on training data."""
        self.ensemble_.fit(X, y)
        self.is_fitted_ = True
        self.classes_ = self.ensemble_.classes_
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble voting."""
        return self.ensemble_.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (soft voting only)."""
        if self.voting == "soft":
            return self.ensemble_.predict_proba(X)
        return None

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return self.ensemble_.score(X, y)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Average feature importances across decoders.

        Only includes decoders that provide importances.
        """
        if not self.is_fitted_:
            return None

        importances = []
        for decoder in self.decoders:
            imp = decoder.get_feature_importances()
            if imp is not None:
                importances.append(imp)

        if not importances:
            return None

        return np.mean(importances, axis=0)

    def _get_sklearn_estimator(self):
        """Get sklearn estimator for permutation test."""
        return self.ensemble_

    def __repr__(self) -> str:
        decoder_names = [d.__class__.__name__ for d in self.decoders]
        return f"EnsembleDecoder({decoder_names}, voting={self.voting})"


class LDADecoder(BaseDecoder):
    """
    Linear Discriminant Analysis decoder.

    Finds linear combinations that best separate classes.
    Works well when class distributions are approximately Gaussian.

    Example:
        >>> decoder = LDADecoder()
        >>> results = decoder.cross_validate(dataset)
    """

    def __init__(
        self,
        solver: str = "svd",
        shrinkage: Optional[Union[str, float]] = None,
        standardize: bool = False,
        random_state: Optional[int] = 42
    ):
        """
        Initialize LDA decoder.

        Args:
            solver: Solver type ("svd", "lsqr", "eigen")
            shrinkage: Regularization ("auto" or float 0-1)
            standardize: Whether to z-score features
            random_state: Random seed
        """
        super().__init__(random_state=random_state)
        self.solver_type = solver
        self.shrinkage = shrinkage
        self.standardize = standardize

        self._build_pipeline()

    def _build_pipeline(self):
        """Build sklearn pipeline."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        steps = []

        if self.standardize:
            steps.append(("scaler", StandardScaler()))

        lda = LinearDiscriminantAnalysis(
            solver=self.solver_type,
            shrinkage=self.shrinkage if self.solver_type != "svd" else None
        )
        steps.append(("lda", lda))

        self.pipeline_ = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDADecoder":
        """Fit LDA on training data."""
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        self.classes_ = self.pipeline_.classes_
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.pipeline_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        return self.pipeline_.score(X, y)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Get LDA coefficients as importances."""
        if not self.is_fitted_:
            return None

        lda = self.pipeline_.named_steps["lda"]
        coef = lda.coef_

        if coef.ndim == 1:
            return np.abs(coef)
        else:
            return np.mean(np.abs(coef), axis=0)

    def _get_sklearn_estimator(self):
        """Get sklearn estimator for permutation test."""
        return self.pipeline_

    def __repr__(self) -> str:
        return f"LDADecoder(solver={self.solver_type})"
