"""
Permutation testing for statistical significance.

Tests whether decoding accuracy is significantly above chance by
comparing to a null distribution from shuffled labels.
"""

from typing import Optional, Tuple
import numpy as np
from joblib import Parallel, delayed
import copy

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset
from core.results import DecodingResults


class PermutationTest:
    """
    Permutation test for classification significance.

    Computes a null distribution by repeatedly shuffling labels
    and re-running classification. P-value is the proportion of
    permuted scores >= observed score.

    Example:
        >>> perm_test = PermutationTest(n_permutations=1000)
        >>> results = perm_test.test(decoder, dataset)
        >>> print(f"p = {results.permutation_pvalue:.4f}")
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        n_jobs: int = -1,
        random_state: Optional[int] = 42,
        verbose: int = 1
    ):
        """
        Initialize permutation test.

        Args:
            n_permutations: Number of permutations
            n_jobs: Parallel jobs (-1 = all CPUs)
            random_state: Random seed
            verbose: Verbosity level
        """
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.null_distribution_ = None
        self.observed_score_ = None
        self.pvalue_ = None

    def test(
        self,
        decoder,
        dataset: DecodingDataset,
        cv=None,
        scoring: str = "accuracy"
    ) -> DecodingResults:
        """
        Run permutation test.

        Args:
            decoder: Decoder instance
            dataset: DecodingDataset
            cv: Cross-validation splitter
            scoring: Metric to use

        Returns:
            DecodingResults with permutation p-value
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X, y = dataset.X, dataset.y
        groups = dataset.groups

        # Default CV
        if cv is None:
            if groups is not None:
                from sklearn.model_selection import LeaveOneGroupOut
                cv = LeaveOneGroupOut()
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True,
                                    random_state=self.random_state)

        # Get sklearn estimator
        estimator = decoder._get_sklearn_estimator()

        # Observed score
        if groups is not None:
            observed_scores = cross_val_score(
                estimator, X, y, cv=cv, groups=groups, scoring=scoring
            )
        else:
            observed_scores = cross_val_score(
                estimator, X, y, cv=cv, scoring=scoring
            )
        self.observed_score_ = np.mean(observed_scores)

        if self.verbose:
            print(f"Observed score: {self.observed_score_:.1%}")
            print(f"Running {self.n_permutations} permutations...")

        # Generate permutation seeds
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**31, size=self.n_permutations)

        # Run permutations
        def run_permutation(seed):
            """Run one permutation."""
            rng_perm = np.random.RandomState(seed)
            y_perm = rng_perm.permutation(y)

            if groups is not None:
                scores = cross_val_score(
                    copy.deepcopy(estimator), X, y_perm,
                    cv=cv, groups=groups, scoring=scoring
                )
            else:
                scores = cross_val_score(
                    copy.deepcopy(estimator), X, y_perm,
                    cv=cv, scoring=scoring
                )
            return np.mean(scores)

        self.null_distribution_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(run_permutation)(seed) for seed in seeds
        )
        self.null_distribution_ = np.array(self.null_distribution_)

        # Compute p-value
        self.pvalue_ = (
            np.sum(self.null_distribution_ >= self.observed_score_) + 1
        ) / (self.n_permutations + 1)

        if self.verbose:
            print(f"P-value: {self.pvalue_:.4f}")

        # Get full CV results
        results = decoder.cross_validate(dataset, cv=cv)
        results.permutation_pvalue = self.pvalue_
        results.permutation_scores = self.null_distribution_
        results.metadata["n_permutations"] = self.n_permutations

        return results

    def get_null_distribution(self) -> np.ndarray:
        """Get null distribution of permuted scores."""
        if self.null_distribution_ is None:
            raise ValueError("Must call test() first")
        return self.null_distribution_

    def plot(
        self,
        figsize: Tuple[float, float] = (8, 5),
        output_path: Optional[str] = None
    ):
        """
        Plot null distribution with observed score.

        Args:
            figsize: Figure size
            output_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self.null_distribution_ is None:
            raise ValueError("Must call test() first")

        fig, ax = plt.subplots(figsize=figsize)

        # Histogram of null distribution
        ax.hist(
            self.null_distribution_,
            bins=50,
            color='gray',
            alpha=0.7,
            label='Null distribution'
        )

        # Observed score
        ax.axvline(
            x=self.observed_score_,
            color='red',
            linewidth=2,
            label=f'Observed: {self.observed_score_:.1%}'
        )

        # Significance threshold (95th percentile)
        threshold = np.percentile(self.null_distribution_, 95)
        ax.axvline(
            x=threshold,
            color='orange',
            linestyle='--',
            label=f'95th percentile: {threshold:.1%}'
        )

        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count')
        ax.set_title(f'Permutation Test (p = {self.pvalue_:.4f})')
        ax.legend()

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def __repr__(self) -> str:
        if self.pvalue_ is not None:
            return f"PermutationTest(p={self.pvalue_:.4f}, n={self.n_permutations})"
        return f"PermutationTest(n_permutations={self.n_permutations})"


class ClusterPermutationTest:
    """
    Cluster-based permutation test for time-resolved decoding.

    Corrects for multiple comparisons across time points using
    cluster-based statistics.

    Example:
        >>> cluster_test = ClusterPermutationTest(n_permutations=1000)
        >>> significant_clusters = cluster_test.test(times, scores, chance=0.5)
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        threshold: float = 0.05,
        n_jobs: int = -1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize cluster permutation test.

        Args:
            n_permutations: Number of permutations
            threshold: Cluster-forming threshold (p-value)
            n_jobs: Parallel jobs
            random_state: Random seed
        """
        self.n_permutations = n_permutations
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

    def test(
        self,
        times: np.ndarray,
        scores: np.ndarray,
        scores_std: np.ndarray,
        chance_level: float = 0.5,
        n_folds: int = 5
    ) -> dict:
        """
        Test for significant clusters in temporal decoding.

        Args:
            times: Time points
            scores: Mean accuracy at each time point
            scores_std: Std of accuracy at each time point
            chance_level: Expected chance accuracy
            n_folds: Number of CV folds (for SE calculation)

        Returns:
            Dictionary with cluster info
        """
        from scipy import stats

        # Convert accuracy to t-statistics
        se = scores_std / np.sqrt(n_folds)
        t_stats = (scores - chance_level) / se

        # Cluster-forming threshold
        t_threshold = stats.t.ppf(1 - self.threshold, df=n_folds - 1)

        # Find clusters above threshold
        clusters = self._find_clusters(t_stats, t_threshold)

        # Get cluster statistics (sum of t-values)
        cluster_stats = [np.sum(t_stats[c]) for c in clusters]

        # Permutation testing
        # (Simplified - full implementation would permute across subjects)
        rng = np.random.RandomState(self.random_state)

        null_cluster_stats = []
        for _ in range(self.n_permutations):
            # Random sign flipping
            signs = rng.choice([-1, 1], size=len(scores))
            t_perm = t_stats * signs

            perm_clusters = self._find_clusters(t_perm, t_threshold)
            if perm_clusters:
                null_cluster_stats.append(max(np.sum(t_perm[c]) for c in perm_clusters))
            else:
                null_cluster_stats.append(0)

        null_cluster_stats = np.array(null_cluster_stats)

        # Compute p-values for each cluster
        cluster_pvalues = []
        for stat in cluster_stats:
            pval = np.mean(null_cluster_stats >= stat)
            cluster_pvalues.append(pval)

        # Format results
        significant_clusters = []
        for i, (cluster, stat, pval) in enumerate(zip(clusters, cluster_stats, cluster_pvalues)):
            if pval < 0.05:
                significant_clusters.append({
                    "cluster_id": i,
                    "time_start": times[cluster[0]],
                    "time_end": times[cluster[-1]],
                    "indices": cluster,
                    "cluster_stat": stat,
                    "p_value": pval
                })

        return {
            "significant_clusters": significant_clusters,
            "all_clusters": clusters,
            "cluster_stats": cluster_stats,
            "cluster_pvalues": cluster_pvalues,
            "t_stats": t_stats,
            "threshold": t_threshold
        }

    def _find_clusters(
        self,
        values: np.ndarray,
        threshold: float
    ) -> list:
        """Find contiguous clusters above threshold."""
        above = values > threshold

        clusters = []
        in_cluster = False
        start_idx = 0

        for i, val in enumerate(above):
            if val and not in_cluster:
                start_idx = i
                in_cluster = True
            elif not val and in_cluster:
                clusters.append(list(range(start_idx, i)))
                in_cluster = False

        if in_cluster:
            clusters.append(list(range(start_idx, len(above))))

        return clusters
