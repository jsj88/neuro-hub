"""
Permutation testing for T-maze group analysis.

Provides non-parametric statistical tests that don't assume
normal distribution of accuracies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy import stats
from scipy.ndimage import label


@dataclass
class PermutationResult:
    """Container for permutation test results."""
    observed: float
    p_value: float
    null_distribution: np.ndarray
    n_permutations: int
    significant: bool
    alpha: float
    test_statistic: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        sig = "*" if self.significant else ""
        return (f"PermutationResult(observed={self.observed:.4f}, "
                f"p={self.p_value:.4f}{sig}, n_perm={self.n_permutations})")


@dataclass
class ClusterResult:
    """Container for cluster permutation test results."""
    significant_clusters: List[Tuple[int, int, float, float]]  # (start, end, stat, p)
    cluster_stats: np.ndarray
    p_values: np.ndarray
    threshold: float
    n_permutations: int
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        n_sig = len(self.significant_clusters)
        return f"ClusterResult({n_sig} significant clusters)"


class GroupPermutationTest:
    """
    Permutation testing for group-level classification accuracy.

    Tests whether mean accuracy across subjects is significantly
    above chance using permutation of subject labels.

    Parameters
    ----------
    n_permutations : int
        Number of permutations (default: 10000)
    test_statistic : str
        'mean', 't', or custom callable
    tail : str
        'greater', 'less', or 'two-sided'
    random_state : int, optional
        Random seed

    Examples
    --------
    >>> perm = GroupPermutationTest(n_permutations=10000)
    >>> result = perm.test(accuracies, chance=0.5)
    >>> print(f"p = {result.p_value:.4f}")
    """

    def __init__(
        self,
        n_permutations: int = 10000,
        test_statistic: str = 'mean',
        tail: str = 'greater',
        random_state: Optional[int] = None
    ):
        self.n_permutations = n_permutations
        self.test_statistic = test_statistic
        self.tail = tail
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def _get_statistic_func(self) -> Callable:
        """Get the test statistic function."""
        if callable(self.test_statistic):
            return self.test_statistic
        elif self.test_statistic == 'mean':
            return np.mean
        elif self.test_statistic == 't':
            return lambda x: np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x)))
        else:
            raise ValueError(f"Unknown test_statistic: {self.test_statistic}")

    def test(
        self,
        accuracies: np.ndarray,
        chance: float = 0.5,
        alpha: float = 0.05
    ) -> PermutationResult:
        """
        Run permutation test.

        Parameters
        ----------
        accuracies : np.ndarray
            Per-subject accuracies
        chance : float
            Chance level (null hypothesis value)
        alpha : float
            Significance threshold

        Returns
        -------
        PermutationResult
        """
        accuracies = np.asarray(accuracies)
        n = len(accuracies)
        stat_func = self._get_statistic_func()

        # Observed statistic (relative to chance)
        centered = accuracies - chance
        observed = stat_func(centered)

        # Null distribution via sign flipping
        null_dist = np.zeros(self.n_permutations)

        for i in range(self.n_permutations):
            # Random sign flip (equivalent to permuting under H0)
            signs = np.random.choice([-1, 1], size=n)
            permuted = centered * signs
            null_dist[i] = stat_func(permuted)

        # Calculate p-value
        if self.tail == 'greater':
            p_value = np.mean(null_dist >= observed)
        elif self.tail == 'less':
            p_value = np.mean(null_dist <= observed)
        else:  # two-sided
            p_value = np.mean(np.abs(null_dist) >= np.abs(observed))

        # Ensure p-value is not exactly 0 (add continuity correction)
        p_value = max(p_value, 1 / (self.n_permutations + 1))

        return PermutationResult(
            observed=observed + chance,  # Report in original scale
            p_value=p_value,
            null_distribution=null_dist + chance,
            n_permutations=self.n_permutations,
            significant=p_value < alpha,
            alpha=alpha,
            test_statistic=str(self.test_statistic),
            metadata={'chance': chance, 'tail': self.tail}
        )

    def test_paired(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05
    ) -> PermutationResult:
        """
        Paired permutation test comparing two conditions.

        Parameters
        ----------
        group1 : np.ndarray
            Accuracies for condition 1 (per subject)
        group2 : np.ndarray
            Accuracies for condition 2 (per subject)
        alpha : float
            Significance threshold

        Returns
        -------
        PermutationResult
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        if len(group1) != len(group2):
            raise ValueError("Groups must have same length for paired test")

        diff = group1 - group2
        n = len(diff)
        stat_func = self._get_statistic_func()

        observed = stat_func(diff)

        # Null distribution via sign flipping differences
        null_dist = np.zeros(self.n_permutations)

        for i in range(self.n_permutations):
            signs = np.random.choice([-1, 1], size=n)
            permuted = diff * signs
            null_dist[i] = stat_func(permuted)

        # P-value
        if self.tail == 'greater':
            p_value = np.mean(null_dist >= observed)
        elif self.tail == 'less':
            p_value = np.mean(null_dist <= observed)
        else:
            p_value = np.mean(np.abs(null_dist) >= np.abs(observed))

        p_value = max(p_value, 1 / (self.n_permutations + 1))

        return PermutationResult(
            observed=observed,
            p_value=p_value,
            null_distribution=null_dist,
            n_permutations=self.n_permutations,
            significant=p_value < alpha,
            alpha=alpha,
            test_statistic=str(self.test_statistic),
            metadata={'tail': self.tail, 'paired': True}
        )


def cluster_permutation_group(
    subject_timeseries: np.ndarray,
    times: np.ndarray,
    chance: float = 0.5,
    n_permutations: int = 1000,
    threshold: float = 2.0,
    alpha: float = 0.05,
    tail: str = 'greater'
) -> ClusterResult:
    """
    Cluster-based permutation test for temporal data across subjects.

    Corrects for multiple comparisons in time using cluster statistics.

    Parameters
    ----------
    subject_timeseries : np.ndarray
        Accuracy timeseries (n_subjects, n_times)
    times : np.ndarray
        Time vector
    chance : float
        Chance level
    n_permutations : int
        Number of permutations
    threshold : float
        T-statistic threshold for cluster formation
    alpha : float
        Cluster-level alpha
    tail : str
        'greater', 'less', or 'two-sided'

    Returns
    -------
    ClusterResult
    """
    n_subjects, n_times = subject_timeseries.shape

    # Center data
    centered = subject_timeseries - chance

    # Observed t-statistics at each time point
    t_obs = np.zeros(n_times)
    for t in range(n_times):
        t_obs[t] = np.mean(centered[:, t]) / (np.std(centered[:, t], ddof=1) / np.sqrt(n_subjects))

    # Find observed clusters
    obs_clusters, obs_cluster_stats = _find_clusters(t_obs, threshold, tail)

    # Permutation distribution of max cluster stats
    max_cluster_stats = np.zeros(n_permutations)

    for perm in range(n_permutations):
        # Sign-flip permutation
        signs = np.random.choice([-1, 1], size=n_subjects)
        permuted = centered * signs[:, np.newaxis]

        # T-statistics for permuted data
        t_perm = np.zeros(n_times)
        for t in range(n_times):
            t_perm[t] = np.mean(permuted[:, t]) / (np.std(permuted[:, t], ddof=1) / np.sqrt(n_subjects))

        # Find max cluster stat
        _, perm_cluster_stats = _find_clusters(t_perm, threshold, tail)
        if len(perm_cluster_stats) > 0:
            max_cluster_stats[perm] = np.max(np.abs(perm_cluster_stats))

    # Compute cluster p-values
    significant_clusters = []
    p_values = np.ones(len(obs_cluster_stats))

    for i, (cluster, stat) in enumerate(zip(obs_clusters, obs_cluster_stats)):
        p_val = np.mean(max_cluster_stats >= np.abs(stat))
        p_val = max(p_val, 1 / (n_permutations + 1))
        p_values[i] = p_val

        if p_val < alpha:
            start_idx, end_idx = cluster[0], cluster[-1]
            significant_clusters.append((
                int(start_idx),
                int(end_idx),
                float(stat),
                float(p_val)
            ))

    return ClusterResult(
        significant_clusters=significant_clusters,
        cluster_stats=np.array(obs_cluster_stats),
        p_values=p_values,
        threshold=threshold,
        n_permutations=n_permutations,
        metadata={
            'times': times,
            'tail': tail,
            'chance': chance,
            't_observed': t_obs
        }
    )


def _find_clusters(
    stat_map: np.ndarray,
    threshold: float,
    tail: str
) -> Tuple[List[np.ndarray], List[float]]:
    """Find clusters of contiguous supra-threshold values."""
    if tail == 'greater':
        mask = stat_map > threshold
    elif tail == 'less':
        mask = stat_map < -threshold
    else:  # two-sided
        mask = np.abs(stat_map) > threshold

    # Label connected components
    labeled, n_clusters = label(mask)

    clusters = []
    cluster_stats = []

    for i in range(1, n_clusters + 1):
        cluster_mask = labeled == i
        cluster_indices = np.where(cluster_mask)[0]
        cluster_stat = np.sum(stat_map[cluster_mask])

        clusters.append(cluster_indices)
        cluster_stats.append(cluster_stat)

    return clusters, cluster_stats


def tfce_correction(
    stat_map: np.ndarray,
    n_permutations: int = 1000,
    e_value: float = 0.5,
    h_value: float = 2.0,
    dh: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Threshold-Free Cluster Enhancement for 1D data.

    Enhances the statistical map using TFCE, then computes p-values
    via permutation testing.

    Parameters
    ----------
    stat_map : np.ndarray
        T-statistic map (n_times,)
    n_permutations : int
        Number of permutations
    e_value : float
        Extent exponent (default: 0.5)
    h_value : float
        Height exponent (default: 2.0)
    dh : float
        Height increment

    Returns
    -------
    tfce_map : np.ndarray
        TFCE-enhanced statistics
    p_values : np.ndarray
        Permutation p-values
    """
    # Compute TFCE for observed data
    tfce_obs = _compute_tfce_1d(stat_map, e_value, h_value, dh)

    # Note: Full TFCE implementation requires the original data
    # for permutation. This is a simplified version.

    # For now, return enhanced map and placeholder p-values
    # In practice, you'd need subject-level data to permute

    return tfce_obs, np.ones_like(tfce_obs)  # Placeholder


def _compute_tfce_1d(
    stat_map: np.ndarray,
    e: float = 0.5,
    h: float = 2.0,
    dh: float = 0.1
) -> np.ndarray:
    """Compute TFCE for 1D statistical map."""
    tfce = np.zeros_like(stat_map)
    h_values = np.arange(dh, np.max(np.abs(stat_map)) + dh, dh)

    for h_thresh in h_values:
        # Positive tail
        pos_mask = stat_map >= h_thresh
        if np.any(pos_mask):
            labeled, n_clusters = label(pos_mask)
            for c in range(1, n_clusters + 1):
                cluster_mask = labeled == c
                extent = np.sum(cluster_mask)
                tfce[cluster_mask] += (extent ** e) * (h_thresh ** h) * dh

        # Negative tail
        neg_mask = stat_map <= -h_thresh
        if np.any(neg_mask):
            labeled, n_clusters = label(neg_mask)
            for c in range(1, n_clusters + 1):
                cluster_mask = labeled == c
                extent = np.sum(cluster_mask)
                tfce[cluster_mask] -= (extent ** e) * (h_thresh ** h) * dh

    return tfce


def permutation_cluster_1d(
    data: np.ndarray,
    times: np.ndarray,
    n_permutations: int = 1000,
    threshold: Optional[float] = None,
    alpha: float = 0.05
) -> Dict:
    """
    Simplified cluster permutation test for single-subject data.

    Parameters
    ----------
    data : np.ndarray
        Data array (n_trials, n_times)
    times : np.ndarray
        Time vector
    n_permutations : int
        Number of permutations
    threshold : float, optional
        T-statistic threshold (auto-computed if None)
    alpha : float
        Significance level

    Returns
    -------
    Dict
        Results including clusters, p-values, and t-statistics
    """
    n_trials, n_times = data.shape

    # Compute observed t-stats (vs zero)
    t_obs = np.zeros(n_times)
    for t in range(n_times):
        t_obs[t], _ = stats.ttest_1samp(data[:, t], 0)

    # Auto threshold at 95th percentile of t-distribution
    if threshold is None:
        threshold = stats.t.ppf(0.95, df=n_trials - 1)

    # Find observed clusters
    obs_clusters, obs_stats = _find_clusters(t_obs, threshold, 'two-sided')

    # Permutation
    max_stats = np.zeros(n_permutations)

    for perm in range(n_permutations):
        # Sign flip
        signs = np.random.choice([-1, 1], size=n_trials)
        perm_data = data * signs[:, np.newaxis]

        # T-stats
        t_perm = np.zeros(n_times)
        for t in range(n_times):
            t_perm[t], _ = stats.ttest_1samp(perm_data[:, t], 0)

        _, perm_stats = _find_clusters(t_perm, threshold, 'two-sided')
        if len(perm_stats) > 0:
            max_stats[perm] = np.max(np.abs(perm_stats))

    # P-values
    cluster_p_values = []
    significant_clusters = []

    for cluster, stat in zip(obs_clusters, obs_stats):
        p_val = np.mean(max_stats >= np.abs(stat))
        p_val = max(p_val, 1 / (n_permutations + 1))
        cluster_p_values.append(p_val)

        if p_val < alpha:
            start_time = times[cluster[0]]
            end_time = times[cluster[-1]]
            significant_clusters.append({
                'start': start_time,
                'end': end_time,
                'start_idx': int(cluster[0]),
                'end_idx': int(cluster[-1]),
                'stat': stat,
                'p_value': p_val
            })

    return {
        't_observed': t_obs,
        'threshold': threshold,
        'clusters': obs_clusters,
        'cluster_stats': obs_stats,
        'cluster_p_values': cluster_p_values,
        'significant_clusters': significant_clusters,
        'n_permutations': n_permutations,
        'times': times
    }


def max_stat_permutation(
    observed_stats: np.ndarray,
    n_permutations: int = 10000,
    stat_func: Callable = np.mean,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Max-statistic permutation for family-wise error control.

    Parameters
    ----------
    observed_stats : np.ndarray
        Observed statistics across tests (e.g., ROIs)
    n_permutations : int
        Number of permutations
    stat_func : Callable
        Function to compute statistic from permuted data
    alpha : float
        Family-wise error rate

    Returns
    -------
    p_values : np.ndarray
        Corrected p-values for each test
    critical_value : float
        Critical value for significance
    null_max_dist : np.ndarray
        Null distribution of max statistics
    """
    n_tests = len(observed_stats)

    # This is a placeholder - actual implementation needs raw data
    # to perform permutations properly

    # Simple approximation using normal theory
    max_null = np.max(np.random.randn(n_permutations, n_tests), axis=1)
    critical_value = np.percentile(max_null, 100 * (1 - alpha))

    p_values = np.array([np.mean(max_null >= obs) for obs in observed_stats])

    return p_values, critical_value, max_null
