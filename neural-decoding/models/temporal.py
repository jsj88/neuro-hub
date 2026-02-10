"""
Temporal decoding for time-resolved analysis.

Performs classification at each time point to track when information
becomes available in the neural signal (typically used with EEG/MEG).
"""

from typing import Optional, Tuple, List
import numpy as np

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset
from core.results import DecodingResults
from .base import BaseDecoder
from .classifiers import SVMDecoder


class TemporalDecoder:
    """
    Time-resolved decoding for EEG/MEG.

    Performs classification at each time point (or sliding window)
    to identify when task-relevant information emerges.

    Example:
        >>> decoder = TemporalDecoder(time_window=0.05, step=0.01)
        >>> decoder.fit(epochs_dataset)
        >>> times, scores = decoder.get_temporal_scores()
        >>> decoder.plot()
    """

    def __init__(
        self,
        decoder: Optional[BaseDecoder] = None,
        time_window: float = 0.05,
        step: float = 0.01,
        cv=None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize temporal decoder.

        Args:
            decoder: Decoder to use (default: linear SVM)
            time_window: Width of sliding window in seconds
            step: Step size between windows in seconds
            cv: Cross-validation splitter
            n_jobs: Parallel jobs
            verbose: Verbosity level
            random_state: Random seed
        """
        self.decoder = decoder or SVMDecoder(kernel="linear")
        self.time_window = time_window
        self.step = step
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.times_ = None
        self.scores_ = None
        self.scores_std_ = None
        self.sfreq_ = None

    def fit(
        self,
        dataset: DecodingDataset,
        times: Optional[np.ndarray] = None,
        sfreq: Optional[float] = None
    ) -> "TemporalDecoder":
        """
        Run time-resolved decoding.

        Args:
            dataset: DecodingDataset with EEG/MEG epochs
                    X should be (n_epochs, n_channels * n_times) or
                    passed with separate epochs data
            times: Time points in seconds (if not in metadata)
            sfreq: Sampling frequency (if not in metadata)

        Returns:
            self: Fitted instance with temporal scores
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        import copy

        # Get time info from metadata or parameters
        if times is not None:
            self.times_ = times
        elif "tmin" in dataset.metadata and "tmax" in dataset.metadata:
            tmin = dataset.metadata["tmin"]
            tmax = dataset.metadata["tmax"]
            sfreq = sfreq or dataset.metadata.get("sfreq", 100)
            n_times = int((tmax - tmin) * sfreq) + 1
            self.times_ = np.linspace(tmin, tmax, n_times)
        else:
            raise ValueError("Need time info: pass times or include in metadata")

        self.sfreq_ = sfreq or dataset.metadata.get("sfreq", 100)

        # Get data dimensions
        n_channels = dataset.metadata.get("n_channels", 1)
        n_times = len(self.times_)

        # Reshape data to (n_epochs, n_channels, n_times)
        X = dataset.X
        if X.ndim == 2:
            n_epochs = X.shape[0]
            X = X.reshape(n_epochs, n_channels, n_times)

        y = dataset.y

        # Set up CV
        if self.cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=self.random_state)
        else:
            cv = self.cv

        # Calculate time windows
        half_win = self.time_window / 2
        window_starts = np.arange(
            self.times_[0] + half_win,
            self.times_[-1] - half_win,
            self.step
        )

        self.scores_ = []
        self.scores_std_ = []
        center_times = []

        for t_center in window_starts:
            # Get time indices for this window
            t_start = t_center - half_win
            t_end = t_center + half_win

            time_mask = (self.times_ >= t_start) & (self.times_ <= t_end)

            if not np.any(time_mask):
                continue

            # Extract features for this window
            # Average over time or flatten
            X_window = X[:, :, time_mask].mean(axis=2)  # (n_epochs, n_channels)

            # Cross-validate
            fold_decoder = copy.deepcopy(self.decoder)
            estimator = fold_decoder._get_sklearn_estimator()

            scores = cross_val_score(
                estimator, X_window, y,
                cv=cv,
                scoring="accuracy",
                n_jobs=self.n_jobs
            )

            self.scores_.append(np.mean(scores))
            self.scores_std_.append(np.std(scores))
            center_times.append(t_center)

            if self.verbose:
                print(f"Time {t_center:.3f}s: {np.mean(scores):.1%}")

        self.times_ = np.array(center_times)
        self.scores_ = np.array(self.scores_)
        self.scores_std_ = np.array(self.scores_std_)

        return self

    def fit_mne(self, epochs, time_window: Optional[float] = None) -> "TemporalDecoder":
        """
        Fit directly from MNE Epochs object.

        Args:
            epochs: MNE Epochs object
            time_window: Override time window (seconds)

        Returns:
            self: Fitted instance
        """
        from mne.decoding import SlidingEstimator, cross_val_multiscore
        from sklearn.model_selection import StratifiedKFold
        import copy

        if time_window:
            self.time_window = time_window

        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y = epochs.events[:, 2]

        self.times_ = epochs.times
        self.sfreq_ = epochs.info["sfreq"]

        # Create sliding estimator
        estimator = copy.deepcopy(self.decoder)._get_sklearn_estimator()
        slider = SlidingEstimator(estimator, scoring="accuracy", n_jobs=self.n_jobs)

        # Cross-validate
        if self.cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=self.random_state)
        else:
            cv = self.cv

        scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=self.n_jobs)

        self.scores_ = np.mean(scores, axis=0)
        self.scores_std_ = np.std(scores, axis=0)

        return self

    def get_temporal_scores(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time points and accuracy scores.

        Returns:
            Tuple of (times, scores) arrays
        """
        if self.scores_ is None:
            raise ValueError("Must call fit() first")

        return self.times_, self.scores_

    def get_significant_periods(
        self,
        chance_level: float = 0.5,
        alpha: float = 0.05,
        n_permutations: int = 1000
    ) -> List[Tuple[float, float]]:
        """
        Find time periods with above-chance accuracy.

        Args:
            chance_level: Expected chance accuracy
            alpha: Significance level
            n_permutations: For cluster-based correction

        Returns:
            List of (start_time, end_time) tuples for significant periods
        """
        if self.scores_ is None:
            raise ValueError("Must call fit() first")

        # Simple threshold (no cluster correction)
        # For proper statistics, use MNE's cluster permutation
        significant = self.scores_ > (chance_level + 2 * self.scores_std_)

        # Find contiguous periods
        periods = []
        in_period = False
        start_idx = 0

        for i, sig in enumerate(significant):
            if sig and not in_period:
                start_idx = i
                in_period = True
            elif not sig and in_period:
                periods.append((self.times_[start_idx], self.times_[i-1]))
                in_period = False

        if in_period:
            periods.append((self.times_[start_idx], self.times_[-1]))

        return periods

    def plot(
        self,
        chance_level: float = 0.5,
        show_std: bool = True,
        title: str = "Temporal Decoding",
        xlabel: str = "Time (s)",
        ylabel: str = "Accuracy",
        figsize: Tuple[float, float] = (10, 5),
        output_path: Optional[str] = None
    ):
        """
        Plot temporal decoding results.

        Args:
            chance_level: Horizontal line for chance
            show_std: Show standard deviation shading
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            output_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self.scores_ is None:
            raise ValueError("Must call fit() first")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot scores
        ax.plot(self.times_, self.scores_, 'b-', linewidth=2, label='Accuracy')

        # Plot std shading
        if show_std and self.scores_std_ is not None:
            ax.fill_between(
                self.times_,
                self.scores_ - self.scores_std_,
                self.scores_ + self.scores_std_,
                alpha=0.3, color='blue'
            )

        # Chance level
        ax.axhline(y=chance_level, color='gray', linestyle='--',
                  label=f'Chance ({chance_level:.0%})')

        # Stimulus onset
        if self.times_[0] < 0 < self.times_[-1]:
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x:.0%}')
        )

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def __repr__(self) -> str:
        fitted = "fitted" if self.scores_ is not None else "not fitted"
        return f"TemporalDecoder(window={self.time_window}s, step={self.step}s, {fitted})"


class TemporalGeneralizationDecoder:
    """
    Temporal generalization matrix (train time x test time).

    Tests whether patterns learned at one time generalize to other times.
    Diagonal = standard temporal decoding.
    Off-diagonal = cross-temporal generalization.

    Example:
        >>> decoder = TemporalGeneralizationDecoder()
        >>> decoder.fit(epochs_dataset)
        >>> decoder.plot()
    """

    def __init__(
        self,
        decoder: Optional[BaseDecoder] = None,
        cv=None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize temporal generalization decoder.

        Args:
            decoder: Decoder to use (default: linear SVM)
            cv: Cross-validation splitter
            n_jobs: Parallel jobs
            verbose: Verbosity level
            random_state: Random seed
        """
        self.decoder = decoder or SVMDecoder(kernel="linear")
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.times_ = None
        self.scores_ = None  # (n_train_times, n_test_times)

    def fit(self, epochs) -> "TemporalGeneralizationDecoder":
        """
        Compute temporal generalization matrix from MNE Epochs.

        Args:
            epochs: MNE Epochs object

        Returns:
            self: Fitted instance
        """
        from mne.decoding import GeneralizingEstimator, cross_val_multiscore
        from sklearn.model_selection import StratifiedKFold
        import copy

        X = epochs.get_data()
        y = epochs.events[:, 2]

        self.times_ = epochs.times

        # Create generalizing estimator
        estimator = copy.deepcopy(self.decoder)._get_sklearn_estimator()
        gen = GeneralizingEstimator(estimator, scoring="accuracy", n_jobs=self.n_jobs)

        # Cross-validate
        if self.cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=self.random_state)
        else:
            cv = self.cv

        scores = cross_val_multiscore(gen, X, y, cv=cv, n_jobs=self.n_jobs)

        self.scores_ = np.mean(scores, axis=0)

        return self

    def plot(
        self,
        chance_level: float = 0.5,
        title: str = "Temporal Generalization",
        cmap: str = "RdBu_r",
        figsize: Tuple[float, float] = (8, 8),
        output_path: Optional[str] = None
    ):
        """
        Plot temporal generalization matrix.

        Args:
            chance_level: Value to center colormap
            title: Plot title
            cmap: Colormap
            figsize: Figure size
            output_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self.scores_ is None:
            raise ValueError("Must call fit() first")

        fig, ax = plt.subplots(figsize=figsize)

        # Center colormap on chance
        vmax = np.max(np.abs(self.scores_ - chance_level)) + chance_level
        vmin = chance_level - (vmax - chance_level)

        im = ax.imshow(
            self.scores_,
            origin='lower',
            extent=[self.times_[0], self.times_[-1],
                   self.times_[0], self.times_[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )

        ax.set_xlabel("Test Time (s)")
        ax.set_ylabel("Train Time (s)")
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Accuracy")

        # Add diagonal and axes
        ax.plot([self.times_[0], self.times_[-1]],
               [self.times_[0], self.times_[-1]],
               'k--', linewidth=1)

        if self.times_[0] < 0 < self.times_[-1]:
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def __repr__(self) -> str:
        fitted = "fitted" if self.scores_ is not None else "not fitted"
        return f"TemporalGeneralizationDecoder({fitted})"
