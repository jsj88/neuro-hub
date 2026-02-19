"""
Dynamic connectivity analysis for T-maze data.

Captures time-varying patterns in functional connectivity using
sliding window and other dynamic methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.signal import windows

from .functional import compute_fc_matrix, fisher_z_transform


@dataclass
class DynamicConnectivityResult:
    """Container for dynamic connectivity results."""
    fc_timeseries: np.ndarray  # (n_windows, n_rois, n_rois)
    window_centers: np.ndarray
    window_size: int
    step_size: int
    method: str
    roi_names: Optional[List[str]] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return self.fc_timeseries.shape[0]

    @property
    def n_rois(self) -> int:
        return self.fc_timeseries.shape[1]

    def get_edge_timeseries(self, roi1: int, roi2: int) -> np.ndarray:
        """Get connectivity timeseries for a specific edge."""
        return self.fc_timeseries[:, roi1, roi2]

    def mean_fc(self) -> np.ndarray:
        """Get time-averaged FC matrix."""
        return np.mean(self.fc_timeseries, axis=0)

    def std_fc(self) -> np.ndarray:
        """Get FC variability (std across windows)."""
        return np.std(self.fc_timeseries, axis=0)


def sliding_window_fc(
    timeseries: np.ndarray,
    window_size: int,
    step_size: int = 1,
    method: str = 'pearson',
    window_type: str = 'rectangular',
    roi_names: Optional[List[str]] = None,
    tapered: bool = True
) -> DynamicConnectivityResult:
    """
    Compute dynamic FC using sliding window approach.

    Parameters
    ----------
    timeseries : np.ndarray
        ROI timeseries (n_timepoints, n_rois)
    window_size : int
        Window size in TRs
    step_size : int
        Step size between windows
    method : str
        'pearson', 'spearman', or 'partial'
    window_type : str
        'rectangular', 'hamming', 'hanning', or 'gaussian'
    roi_names : List[str], optional
        ROI names
    tapered : bool
        Apply tapering window (vs rectangular)

    Returns
    -------
    DynamicConnectivityResult
    """
    n_times, n_rois = timeseries.shape

    # Create window function
    if tapered and window_type != 'rectangular':
        if window_type == 'hamming':
            win = windows.hamming(window_size)
        elif window_type == 'hanning':
            win = windows.hann(window_size)
        elif window_type == 'gaussian':
            win = windows.gaussian(window_size, std=window_size/6)
        else:
            win = np.ones(window_size)
    else:
        win = np.ones(window_size)

    # Normalize window
    win = win / np.sum(win) * window_size

    # Calculate number of windows
    n_windows = (n_times - window_size) // step_size + 1
    window_centers = np.arange(n_windows) * step_size + window_size // 2

    # Compute FC for each window
    fc_timeseries = np.zeros((n_windows, n_rois, n_rois))

    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        window_data = timeseries[start:end] * win[:, np.newaxis]

        if method == 'partial':
            from .functional import partial_correlation
            fc_result = partial_correlation(window_data, roi_names=roi_names)
        else:
            fc_result = compute_fc_matrix(window_data, method=method, roi_names=roi_names)

        fc_timeseries[w] = fc_result.matrix

    return DynamicConnectivityResult(
        fc_timeseries=fc_timeseries,
        window_centers=window_centers,
        window_size=window_size,
        step_size=step_size,
        method=method,
        roi_names=roi_names,
        metadata={'window_type': window_type, 'tapered': tapered}
    )


def dcc_connectivity(
    timeseries: np.ndarray,
    roi_names: Optional[List[str]] = None
) -> DynamicConnectivityResult:
    """
    Dynamic Conditional Correlation (DCC-GARCH) connectivity.

    More principled approach to dynamic connectivity that models
    time-varying correlations using GARCH.

    Parameters
    ----------
    timeseries : np.ndarray
        ROI timeseries (n_timepoints, n_rois)
    roi_names : List[str], optional
        ROI names

    Returns
    -------
    DynamicConnectivityResult

    Note
    ----
    This is a simplified implementation. For full DCC-GARCH,
    use the arch package or R's rmgarch.
    """
    n_times, n_rois = timeseries.shape

    # Standardize timeseries
    ts_std = (timeseries - np.mean(timeseries, axis=0)) / np.std(timeseries, axis=0)

    # Simplified DCC using exponential smoothing
    # Q_t = (1-alpha-beta)*Qbar + alpha*epsilon_{t-1}*epsilon_{t-1}' + beta*Q_{t-1}
    alpha = 0.05
    beta = 0.90

    # Unconditional correlation
    Q_bar = np.corrcoef(ts_std.T)

    # Initialize
    Q = np.zeros((n_times, n_rois, n_rois))
    R = np.zeros((n_times, n_rois, n_rois))  # Dynamic correlations

    Q[0] = Q_bar.copy()
    R[0] = Q_bar.copy()

    for t in range(1, n_times):
        epsilon = ts_std[t-1:t].T  # Column vector
        outer_prod = epsilon @ epsilon.T

        Q[t] = (1 - alpha - beta) * Q_bar + alpha * outer_prod + beta * Q[t-1]

        # Normalize to get correlation
        D = np.sqrt(np.diag(Q[t]))
        R[t] = Q[t] / np.outer(D, D)

    return DynamicConnectivityResult(
        fc_timeseries=R,
        window_centers=np.arange(n_times),
        window_size=1,
        step_size=1,
        method='dcc',
        roi_names=roi_names,
        metadata={'alpha': alpha, 'beta': beta}
    )


def detect_fc_states(
    dfc_result: DynamicConnectivityResult,
    n_states: int = 4,
    method: str = 'kmeans'
) -> Dict:
    """
    Detect discrete FC states from dynamic connectivity.

    Parameters
    ----------
    dfc_result : DynamicConnectivityResult
        Dynamic connectivity result
    n_states : int
        Number of states to find
    method : str
        'kmeans' or 'hmm' (Hidden Markov Model)

    Returns
    -------
    Dict
        State assignments, centroids, and metrics
    """
    from sklearn.cluster import KMeans

    n_windows = dfc_result.n_windows
    n_rois = dfc_result.n_rois

    # Vectorize FC matrices (upper triangle)
    triu_idx = np.triu_indices(n_rois, k=1)
    n_edges = len(triu_idx[0])

    fc_vectors = np.zeros((n_windows, n_edges))
    for w in range(n_windows):
        fc_vectors[w] = dfc_result.fc_timeseries[w][triu_idx]

    if method == 'kmeans':
        # K-means clustering
        kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
        state_labels = kmeans.fit_predict(fc_vectors)
        centroids_vec = kmeans.cluster_centers_
        inertia = kmeans.inertia_

        # Reconstruct centroid matrices
        centroids = np.zeros((n_states, n_rois, n_rois))
        for s in range(n_states):
            centroids[s][triu_idx] = centroids_vec[s]
            centroids[s] = centroids[s] + centroids[s].T
            np.fill_diagonal(centroids[s], 1)

        # State metrics
        state_counts = np.bincount(state_labels, minlength=n_states)
        dwell_times = _compute_dwell_times(state_labels)
        transition_matrix = _compute_transition_matrix(state_labels, n_states)

        return {
            'state_labels': state_labels,
            'centroids': centroids,
            'n_states': n_states,
            'state_counts': state_counts,
            'state_fractions': state_counts / n_windows,
            'dwell_times': dwell_times,
            'transition_matrix': transition_matrix,
            'inertia': inertia,
            'method': method
        }

    elif method == 'hmm':
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError("hmmlearn required for HMM state detection")

        model = hmm.GaussianHMM(n_components=n_states, n_iter=100, random_state=42)
        model.fit(fc_vectors)
        state_labels = model.predict(fc_vectors)

        # Get state means as centroids
        centroids = np.zeros((n_states, n_rois, n_rois))
        for s in range(n_states):
            centroids[s][triu_idx] = model.means_[s]
            centroids[s] = centroids[s] + centroids[s].T
            np.fill_diagonal(centroids[s], 1)

        state_counts = np.bincount(state_labels, minlength=n_states)
        dwell_times = _compute_dwell_times(state_labels)

        return {
            'state_labels': state_labels,
            'centroids': centroids,
            'n_states': n_states,
            'state_counts': state_counts,
            'state_fractions': state_counts / n_windows,
            'dwell_times': dwell_times,
            'transition_matrix': model.transmat_,
            'log_likelihood': model.score(fc_vectors),
            'method': method
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_dwell_times(state_labels: np.ndarray) -> Dict[int, List[int]]:
    """Compute dwell times for each state."""
    dwell_times = {}
    current_state = state_labels[0]
    current_dwell = 1

    for label in state_labels[1:]:
        if label == current_state:
            current_dwell += 1
        else:
            if current_state not in dwell_times:
                dwell_times[current_state] = []
            dwell_times[current_state].append(current_dwell)
            current_state = label
            current_dwell = 1

    # Add last dwell
    if current_state not in dwell_times:
        dwell_times[current_state] = []
    dwell_times[current_state].append(current_dwell)

    return dwell_times


def _compute_transition_matrix(state_labels: np.ndarray, n_states: int) -> np.ndarray:
    """Compute state transition probability matrix."""
    trans = np.zeros((n_states, n_states))

    for i in range(len(state_labels) - 1):
        trans[state_labels[i], state_labels[i+1]] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    trans = trans / row_sums

    return trans


def fc_variability(
    dfc_result: DynamicConnectivityResult,
    metric: str = 'std'
) -> np.ndarray:
    """
    Compute FC variability across time.

    Parameters
    ----------
    dfc_result : DynamicConnectivityResult
        Dynamic connectivity result
    metric : str
        'std', 'range', 'cv' (coefficient of variation), or 'iqr'

    Returns
    -------
    np.ndarray
        Variability matrix (n_rois, n_rois)
    """
    fc_ts = dfc_result.fc_timeseries

    if metric == 'std':
        return np.std(fc_ts, axis=0)
    elif metric == 'range':
        return np.max(fc_ts, axis=0) - np.min(fc_ts, axis=0)
    elif metric == 'cv':
        mean_fc = np.mean(fc_ts, axis=0)
        std_fc = np.std(fc_ts, axis=0)
        # Avoid division by zero
        mean_fc[mean_fc == 0] = 1
        return std_fc / np.abs(mean_fc)
    elif metric == 'iqr':
        q75 = np.percentile(fc_ts, 75, axis=0)
        q25 = np.percentile(fc_ts, 25, axis=0)
        return q75 - q25
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compare_dfc_conditions(
    dfc1: DynamicConnectivityResult,
    dfc2: DynamicConnectivityResult,
    metric: str = 'variability',
    n_permutations: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare dynamic connectivity between conditions.

    Parameters
    ----------
    dfc1, dfc2 : DynamicConnectivityResult
        Dynamic connectivity results for two conditions
    metric : str
        'variability' or 'mean'
    n_permutations : int
        Number of permutations for significance testing

    Returns
    -------
    diff_matrix : np.ndarray
        Difference in metric between conditions
    p_matrix : np.ndarray
        P-values
    """
    if metric == 'variability':
        var1 = fc_variability(dfc1)
        var2 = fc_variability(dfc2)
    elif metric == 'mean':
        var1 = dfc1.mean_fc()
        var2 = dfc2.mean_fc()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    diff_matrix = var1 - var2

    # Permutation test
    combined = np.concatenate([dfc1.fc_timeseries, dfc2.fc_timeseries], axis=0)
    n1 = dfc1.n_windows

    null_diffs = np.zeros((n_permutations,) + diff_matrix.shape)

    for perm in range(n_permutations):
        perm_idx = np.random.permutation(combined.shape[0])
        perm1 = combined[perm_idx[:n1]]
        perm2 = combined[perm_idx[n1:]]

        if metric == 'variability':
            null_diffs[perm] = np.std(perm1, axis=0) - np.std(perm2, axis=0)
        else:
            null_diffs[perm] = np.mean(perm1, axis=0) - np.mean(perm2, axis=0)

    p_matrix = np.mean(np.abs(null_diffs) >= np.abs(diff_matrix), axis=0)
    p_matrix = np.maximum(p_matrix, 1 / (n_permutations + 1))

    return diff_matrix, p_matrix


def temporal_variability_index(
    timeseries: np.ndarray,
    window_sizes: List[int] = [20, 40, 60, 80]
) -> Dict:
    """
    Compute multi-scale temporal variability index.

    Measures how FC variability changes with window size.

    Parameters
    ----------
    timeseries : np.ndarray
        ROI timeseries (n_timepoints, n_rois)
    window_sizes : List[int]
        Window sizes to evaluate

    Returns
    -------
    Dict
        Variability at each scale and overall index
    """
    results = {'window_sizes': window_sizes, 'variability': []}

    for ws in window_sizes:
        if ws >= timeseries.shape[0]:
            continue

        dfc = sliding_window_fc(timeseries, window_size=ws, step_size=ws//2)
        var = fc_variability(dfc, metric='std')

        # Mean variability across edges
        triu_idx = np.triu_indices(dfc.n_rois, k=1)
        mean_var = np.mean(var[triu_idx])
        results['variability'].append(mean_var)

    # Temporal variability index: slope of variability vs window size
    if len(results['variability']) >= 2:
        x = np.log(window_sizes[:len(results['variability'])])
        y = np.log(np.array(results['variability']) + 1e-10)
        slope, _, _, _, _ = stats.linregress(x, y)
        results['tvi'] = slope
    else:
        results['tvi'] = np.nan

    return results
