"""
Cross-validation strategies for neural decoding.

Provides neuroimaging-specific CV strategies that respect the structure
of the data (runs, subjects, sessions).
"""

from typing import Iterator, Tuple, Optional
import numpy as np
from sklearn.model_selection import BaseCrossValidator

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset


class LeaveOneRunOut(BaseCrossValidator):
    """
    Leave-one-run-out cross-validation.

    Standard CV strategy for fMRI where data is collected in runs.
    Prevents temporal autocorrelation leakage between train and test.

    Example:
        >>> cv = LeaveOneRunOut()
        >>> for train_idx, test_idx in cv.split(X, y, groups=runs):
        ...     X_train, X_test = X[train_idx], X[test_idx]
    """

    def __init__(self):
        """Initialize leave-one-run-out CV."""
        pass

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature matrix
            y: Labels (unused, for sklearn compatibility)
            groups: Run labels for each sample

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            raise ValueError("LeaveOneRunOut requires groups (run labels)")

        unique_runs = np.unique(groups)

        for test_run in unique_runs:
            test_idx = np.where(groups == test_run)[0]
            train_idx = np.where(groups != test_run)[0]
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> int:
        """Get number of folds (one per run)."""
        if groups is None:
            raise ValueError("LeaveOneRunOut requires groups")
        return len(np.unique(groups))

    def split_dataset(
        self,
        dataset: DecodingDataset
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Split DecodingDataset."""
        if dataset.groups is None:
            raise ValueError("Dataset must have groups for LeaveOneRunOut")

        return self.split(dataset.X, dataset.y, groups=dataset.groups)


class LeaveOneSubjectOut(BaseCrossValidator):
    """
    Leave-one-subject-out cross-validation.

    For multi-subject analyses, tests generalization across subjects.
    Each fold leaves out all data from one subject.

    Example:
        >>> cv = LeaveOneSubjectOut()
        >>> results = decoder.cross_validate(group_dataset, cv=cv)
    """

    def __init__(self):
        """Initialize leave-one-subject-out CV."""
        pass

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each subject.

        Args:
            X: Feature matrix
            y: Labels (unused)
            groups: Subject labels for each sample

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            raise ValueError("LeaveOneSubjectOut requires groups (subject labels)")

        unique_subjects = np.unique(groups)

        for test_subject in unique_subjects:
            test_idx = np.where(groups == test_subject)[0]
            train_idx = np.where(groups != test_subject)[0]
            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> int:
        """Get number of folds (one per subject)."""
        if groups is None:
            raise ValueError("LeaveOneSubjectOut requires groups")
        return len(np.unique(groups))

    def split_dataset(
        self,
        dataset: DecodingDataset
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Split DecodingDataset."""
        if dataset.groups is None:
            raise ValueError("Dataset must have groups for LeaveOneSubjectOut")

        return self.split(dataset.X, dataset.y, groups=dataset.groups)


class StratifiedGroupKFold(BaseCrossValidator):
    """
    Stratified K-Fold that respects group structure.

    Ensures that:
    1. All samples from a group are in the same fold
    2. Class proportions are approximately maintained

    Useful when you have multiple runs per subject and want
    to do k-fold while keeping runs together.

    Example:
        >>> cv = StratifiedGroupKFold(n_splits=5)
        >>> for train_idx, test_idx in cv.split(X, y, groups=runs):
        ...     # Each run stays together
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42
    ):
        """
        Initialize stratified group k-fold.

        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle groups
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified group-aware splits.

        Args:
            X: Feature matrix
            y: Labels for stratification
            groups: Group labels

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            raise ValueError("StratifiedGroupKFold requires groups")

        if y is None:
            raise ValueError("StratifiedGroupKFold requires y for stratification")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits:
            raise ValueError(
                f"Number of groups ({n_groups}) must be >= n_splits ({self.n_splits})"
            )

        # Get majority class per group for stratification
        group_labels = []
        for g in unique_groups:
            group_mask = groups == g
            # Use most common label in group
            labels, counts = np.unique(y[group_mask], return_counts=True)
            group_labels.append(labels[np.argmax(counts)])
        group_labels = np.array(group_labels)

        # Stratified assignment of groups to folds
        rng = np.random.RandomState(self.random_state)
        fold_indices = np.zeros(n_groups, dtype=int)

        for label in np.unique(group_labels):
            label_mask = group_labels == label
            label_groups = np.where(label_mask)[0]

            if self.shuffle:
                rng.shuffle(label_groups)

            # Distribute groups of this class across folds
            for i, g_idx in enumerate(label_groups):
                fold_indices[g_idx] = i % self.n_splits

        # Generate splits
        for fold in range(self.n_splits):
            test_groups = unique_groups[fold_indices == fold]
            train_groups = unique_groups[fold_indices != fold]

            test_idx = np.where(np.isin(groups, test_groups))[0]
            train_idx = np.where(np.isin(groups, train_groups))[0]

            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> int:
        """Get number of folds."""
        return self.n_splits


class RepeatedStratifiedGroupKFold(BaseCrossValidator):
    """
    Repeated stratified group k-fold cross-validation.

    Runs StratifiedGroupKFold multiple times with different random states
    for more stable estimates.

    Example:
        >>> cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=10)
        >>> results = decoder.cross_validate(dataset, cv=cv)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = 42
    ):
        """
        Initialize repeated stratified group k-fold.

        Args:
            n_splits: Number of folds per repeat
            n_repeats: Number of times to repeat
            random_state: Base random seed
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate repeated stratified group splits."""
        for repeat in range(self.n_repeats):
            seed = None if self.random_state is None else self.random_state + repeat

            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=seed
            )

            for train_idx, test_idx in cv.split(X, y, groups):
                yield train_idx, test_idx

    def get_n_splits(
        self,
        X: np.ndarray = None,
        y: np.ndarray = None,
        groups: np.ndarray = None
    ) -> int:
        """Get total number of folds."""
        return self.n_splits * self.n_repeats
