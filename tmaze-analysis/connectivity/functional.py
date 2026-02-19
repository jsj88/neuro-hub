"""
Functional connectivity analysis for T-maze fMRI data.

Computes static functional connectivity matrices using various
correlation methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.linalg import inv


@dataclass
class FCResult:
    """Container for functional connectivity results."""
    matrix: np.ndarray
    roi_names: Optional[List[str]]
    method: str
    n_timepoints: int
    metadata: Dict = field(default_factory=dict)

    @property
    def n_rois(self) -> int:
        return self.matrix.shape[0]

    def get_edge(self, roi1: str, roi2: str) -> float:
        """Get connectivity between two ROIs."""
        if self.roi_names is None:
            raise ValueError("ROI names not provided")
        i = self.roi_names.index(roi1)
        j = self.roi_names.index(roi2)
        return self.matrix[i, j]

    def get_node_strength(self, roi: Optional[str] = None) -> Union[float, np.ndarray]:
        """Get node strength (sum of connections)."""
        strengths = np.sum(np.abs(self.matrix), axis=1) - np.diag(self.matrix)
        if roi is None:
            return strengths
        if self.roi_names is None:
            raise ValueError("ROI names not provided")
        return strengths[self.roi_names.index(roi)]


def compute_fc_matrix(
    timeseries: np.ndarray,
    method: str = 'pearson',
    roi_names: Optional[List[str]] = None,
    regularize: bool = False,
    shrinkage: float = 0.1
) -> FCResult:
    """
    Compute functional connectivity matrix.

    Parameters
    ----------
    timeseries : np.ndarray
        ROI timeseries (n_timepoints, n_rois) or (n_trials, n_timepoints, n_rois)
    method : str
        'pearson', 'spearman', or 'covariance'
    roi_names : List[str], optional
        Names of ROIs
    regularize : bool
        Apply shrinkage regularization
    shrinkage : float
        Shrinkage intensity (0-1)

    Returns
    -------
    FCResult
        Functional connectivity matrix and metadata
    """
    # Handle trial-averaged case
    if timeseries.ndim == 3:
        n_trials, n_times, n_rois = timeseries.shape
        timeseries = timeseries.reshape(-1, n_rois)

    n_timepoints, n_rois = timeseries.shape

    if method == 'pearson':
        fc = np.corrcoef(timeseries.T)
    elif method == 'spearman':
        fc, _ = stats.spearmanr(timeseries)
        if n_rois == 2:
            fc = np.array([[1, fc], [fc, 1]])
    elif method == 'covariance':
        fc = np.cov(timeseries.T)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Regularization
    if regularize and method != 'covariance':
        # Ledoit-Wolf shrinkage toward identity
        fc = (1 - shrinkage) * fc + shrinkage * np.eye(n_rois)

    # Ensure symmetric and no NaN
    fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 1.0 if method != 'covariance' else np.diag(fc))
    fc = np.nan_to_num(fc, nan=0.0)

    return FCResult(
        matrix=fc,
        roi_names=roi_names,
        method=method,
        n_timepoints=n_timepoints,
        metadata={'regularized': regularize, 'shrinkage': shrinkage}
    )


def partial_correlation(
    timeseries: np.ndarray,
    roi_names: Optional[List[str]] = None,
    regularize: bool = True,
    shrinkage: float = 0.1
) -> FCResult:
    """
    Compute partial correlation (precision) matrix.

    Removes indirect connections by conditioning on all other nodes.

    Parameters
    ----------
    timeseries : np.ndarray
        ROI timeseries (n_timepoints, n_rois)
    roi_names : List[str], optional
        ROI names
    regularize : bool
        Apply regularization (recommended for n_rois > n_timepoints)
    shrinkage : float
        Shrinkage parameter

    Returns
    -------
    FCResult
        Partial correlation matrix
    """
    if timeseries.ndim == 3:
        timeseries = timeseries.reshape(-1, timeseries.shape[-1])

    n_timepoints, n_rois = timeseries.shape

    # Compute covariance
    cov = np.cov(timeseries.T)

    # Regularize if needed
    if regularize:
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(n_rois) * np.trace(cov) / n_rois

    # Compute precision matrix (inverse covariance)
    try:
        precision = inv(cov)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        precision = np.linalg.pinv(cov)

    # Convert to partial correlation
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)

    return FCResult(
        matrix=partial_corr,
        roi_names=roi_names,
        method='partial_correlation',
        n_timepoints=n_timepoints,
        metadata={'regularized': regularize, 'shrinkage': shrinkage}
    )


def fc_condition_contrast(
    timeseries_cond1: np.ndarray,
    timeseries_cond2: np.ndarray,
    method: str = 'pearson',
    test: str = 'permutation',
    n_permutations: int = 1000,
    roi_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare FC between two conditions.

    Parameters
    ----------
    timeseries_cond1 : np.ndarray
        Timeseries for condition 1 (n_trials, n_times, n_rois) or (n_times, n_rois)
    timeseries_cond2 : np.ndarray
        Timeseries for condition 2
    method : str
        Correlation method
    test : str
        'permutation' or 'parametric'
    n_permutations : int
        Number of permutations
    roi_names : List[str], optional
        ROI names

    Returns
    -------
    diff_matrix : np.ndarray
        FC difference (cond1 - cond2)
    p_matrix : np.ndarray
        P-values for each edge
    """
    # Compute FC for each condition
    fc1 = compute_fc_matrix(timeseries_cond1, method=method, roi_names=roi_names)
    fc2 = compute_fc_matrix(timeseries_cond2, method=method, roi_names=roi_names)

    # Fisher z-transform before subtraction
    z1 = fisher_z_transform(fc1.matrix)
    z2 = fisher_z_transform(fc2.matrix)

    diff_matrix = z1 - z2
    n_rois = fc1.n_rois

    if test == 'parametric':
        # Z-test for difference in correlations
        n1 = fc1.n_timepoints
        n2 = fc2.n_timepoints
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z_stat = diff_matrix / se
        p_matrix = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    elif test == 'permutation':
        # Combine data and permute labels
        if timeseries_cond1.ndim == 2:
            combined = np.vstack([timeseries_cond1, timeseries_cond2])
            split = timeseries_cond1.shape[0]
        else:
            combined = np.concatenate([timeseries_cond1, timeseries_cond2], axis=0)
            split = timeseries_cond1.shape[0]

        # Permutation distribution
        null_diffs = np.zeros((n_permutations, n_rois, n_rois))

        for perm in range(n_permutations):
            perm_idx = np.random.permutation(combined.shape[0])
            perm1 = combined[perm_idx[:split]]
            perm2 = combined[perm_idx[split:]]

            fc_perm1 = compute_fc_matrix(perm1, method=method)
            fc_perm2 = compute_fc_matrix(perm2, method=method)

            null_diffs[perm] = fisher_z_transform(fc_perm1.matrix) - fisher_z_transform(fc_perm2.matrix)

        # Two-sided p-values
        p_matrix = np.mean(np.abs(null_diffs) >= np.abs(diff_matrix), axis=0)
        p_matrix = np.maximum(p_matrix, 1 / (n_permutations + 1))

    else:
        raise ValueError(f"Unknown test: {test}")

    return diff_matrix, p_matrix


def fisher_z_transform(fc_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Fisher z-transformation to correlation matrix.

    Transforms correlations to normally distributed z-scores.

    Parameters
    ----------
    fc_matrix : np.ndarray
        Correlation matrix (-1 to 1)

    Returns
    -------
    np.ndarray
        Z-transformed matrix
    """
    # Clip to avoid infinity
    fc_clipped = np.clip(fc_matrix, -0.9999, 0.9999)
    z = np.arctanh(fc_clipped)
    return z


def inverse_fisher_z(z_matrix: np.ndarray) -> np.ndarray:
    """Inverse Fisher z-transformation."""
    return np.tanh(z_matrix)


def fc_to_adjacency(
    fc_matrix: np.ndarray,
    threshold: Optional[float] = None,
    density: Optional[float] = None,
    binarize: bool = True,
    absolute: bool = True
) -> np.ndarray:
    """
    Convert FC matrix to adjacency matrix for graph analysis.

    Parameters
    ----------
    fc_matrix : np.ndarray
        Functional connectivity matrix
    threshold : float, optional
        Absolute threshold for edges
    density : float, optional
        Target density (0-1). Overrides threshold.
    binarize : bool
        Return binary (vs weighted) adjacency
    absolute : bool
        Use absolute values of correlations

    Returns
    -------
    np.ndarray
        Adjacency matrix
    """
    fc = np.abs(fc_matrix) if absolute else fc_matrix.copy()
    np.fill_diagonal(fc, 0)  # Remove self-connections

    if density is not None:
        # Threshold to achieve target density
        n = fc.shape[0]
        n_edges = int(density * n * (n - 1) / 2)
        # Get upper triangle values
        triu_idx = np.triu_indices(n, k=1)
        values = fc[triu_idx]
        sorted_vals = np.sort(values)[::-1]
        threshold = sorted_vals[min(n_edges, len(sorted_vals) - 1)]

    if threshold is not None:
        fc[fc < threshold] = 0

    if binarize:
        fc = (fc > 0).astype(float)

    return fc


def compute_fc_reliability(
    timeseries: np.ndarray,
    method: str = 'split_half',
    n_splits: int = 100
) -> Tuple[float, np.ndarray]:
    """
    Assess reliability of FC estimates.

    Parameters
    ----------
    timeseries : np.ndarray
        ROI timeseries (n_timepoints, n_rois)
    method : str
        'split_half' or 'jackknife'
    n_splits : int
        Number of split-half iterations

    Returns
    -------
    reliability : float
        Mean reliability (correlation between split-halves)
    reliability_matrix : np.ndarray
        Reliability for each edge
    """
    n_times, n_rois = timeseries.shape

    if method == 'split_half':
        correlations = []

        for _ in range(n_splits):
            # Random split
            idx = np.random.permutation(n_times)
            half = n_times // 2
            fc1 = compute_fc_matrix(timeseries[idx[:half]]).matrix
            fc2 = compute_fc_matrix(timeseries[idx[half:2*half]]).matrix

            # Correlation between upper triangles
            triu_idx = np.triu_indices(n_rois, k=1)
            r, _ = stats.pearsonr(fc1[triu_idx], fc2[triu_idx])
            correlations.append(r)

        # Spearman-Brown correction
        mean_r = np.mean(correlations)
        reliability = 2 * mean_r / (1 + mean_r)

        # Per-edge reliability (not Spearman-Brown corrected)
        reliability_matrix = np.zeros((n_rois, n_rois))
        # Would need to store per-edge correlations across splits

        return reliability, reliability_matrix

    elif method == 'jackknife':
        # Leave-one-timepoint-out
        fc_full = compute_fc_matrix(timeseries).matrix
        fc_jackknife = np.zeros((n_times, n_rois, n_rois))

        for t in range(n_times):
            mask = np.ones(n_times, dtype=bool)
            mask[t] = False
            fc_jackknife[t] = compute_fc_matrix(timeseries[mask]).matrix

        # Variance across jackknife samples
        var_fc = np.var(fc_jackknife, axis=0)

        # Standard error
        se_fc = np.sqrt((n_times - 1) * var_fc)

        # Could compute z-score for reliability: fc_full / se_fc

        return np.mean(1 - var_fc), se_fc

    else:
        raise ValueError(f"Unknown method: {method}")


def network_connectivity(
    fc_matrix: np.ndarray,
    roi_names: List[str],
    network_assignments: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute within- and between-network connectivity.

    Parameters
    ----------
    fc_matrix : np.ndarray
        FC matrix
    roi_names : List[str]
        ROI names
    network_assignments : Dict[str, List[str]]
        {network_name: [roi_names]}

    Returns
    -------
    Dict
        Within/between network connectivity values
    """
    networks = list(network_assignments.keys())
    n_networks = len(networks)

    # Create ROI to network mapping
    roi_to_network = {}
    for net, rois in network_assignments.items():
        for roi in rois:
            if roi in roi_names:
                roi_to_network[roi] = net

    results = {}

    for i, net1 in enumerate(networks):
        results[net1] = {}

        # Get indices for this network
        idx1 = [roi_names.index(r) for r in network_assignments[net1] if r in roi_names]

        for j, net2 in enumerate(networks):
            idx2 = [roi_names.index(r) for r in network_assignments[net2] if r in roi_names]

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            # Get submatrix
            submat = fc_matrix[np.ix_(idx1, idx2)]

            if i == j:
                # Within-network: upper triangle mean
                triu_vals = submat[np.triu_indices(len(idx1), k=1)]
                if len(triu_vals) > 0:
                    results[net1][net2] = np.mean(triu_vals)
            else:
                # Between-network: all values
                results[net1][net2] = np.mean(submat)

    return results
