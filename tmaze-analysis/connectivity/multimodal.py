"""
Multimodal connectivity analysis for T-maze EEG-fMRI data.

Combines EEG and fMRI connectivity measures for cross-modal analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats, signal


@dataclass
class MultimodalConnectivityResult:
    """Container for multimodal connectivity results."""
    coupling_matrix: np.ndarray
    eeg_features: List[str]
    fmri_rois: List[str]
    method: str
    metadata: Dict = field(default_factory=dict)

    @property
    def n_eeg_features(self) -> int:
        return self.coupling_matrix.shape[0]

    @property
    def n_rois(self) -> int:
        return self.coupling_matrix.shape[1]


def eeg_fmri_coupling(
    eeg_features: np.ndarray,
    fmri_timeseries: np.ndarray,
    method: str = 'correlation',
    eeg_feature_names: Optional[List[str]] = None,
    roi_names: Optional[List[str]] = None,
    lag: int = 0
) -> MultimodalConnectivityResult:
    """
    Compute EEG-fMRI coupling.

    Parameters
    ----------
    eeg_features : np.ndarray
        EEG features (n_timepoints, n_features)
        e.g., band power in different frequency bands
    fmri_timeseries : np.ndarray
        fMRI ROI timeseries (n_timepoints, n_rois)
    method : str
        'correlation', 'partial', 'regression', or 'canonical'
    eeg_feature_names : List[str], optional
        Names of EEG features
    roi_names : List[str], optional
        Names of fMRI ROIs
    lag : int
        Lag in TRs (positive = EEG leads fMRI)

    Returns
    -------
    MultimodalConnectivityResult
    """
    n_times_eeg, n_eeg = eeg_features.shape
    n_times_fmri, n_rois = fmri_timeseries.shape

    # Handle different temporal resolutions
    if n_times_eeg != n_times_fmri:
        # Resample EEG to match fMRI
        from scipy.ndimage import zoom
        ratio = n_times_fmri / n_times_eeg
        eeg_resampled = zoom(eeg_features, (ratio, 1), order=1)
        eeg_features = eeg_resampled[:n_times_fmri]

    # Apply lag
    if lag != 0:
        if lag > 0:
            eeg_features = eeg_features[:-lag]
            fmri_timeseries = fmri_timeseries[lag:]
        else:
            eeg_features = eeg_features[-lag:]
            fmri_timeseries = fmri_timeseries[:lag]

    n_times = min(eeg_features.shape[0], fmri_timeseries.shape[0])
    eeg_features = eeg_features[:n_times]
    fmri_timeseries = fmri_timeseries[:n_times]

    if method == 'correlation':
        # Simple correlation matrix
        coupling = np.zeros((n_eeg, n_rois))
        for i in range(n_eeg):
            for j in range(n_rois):
                r, _ = stats.pearsonr(eeg_features[:, i], fmri_timeseries[:, j])
                coupling[i, j] = r

    elif method == 'partial':
        # Partial correlation controlling for other variables
        coupling = _partial_correlation_matrix(eeg_features, fmri_timeseries)

    elif method == 'regression':
        # Multiple regression: predict each ROI from all EEG features
        coupling = np.zeros((n_eeg, n_rois))

        for j in range(n_rois):
            # Add intercept
            X = np.column_stack([np.ones(n_times), eeg_features])
            y = fmri_timeseries[:, j]

            # OLS
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            coupling[:, j] = beta[1:]  # Exclude intercept

    elif method == 'canonical':
        # Canonical correlation analysis
        from sklearn.cross_decomposition import CCA

        cca = CCA(n_components=min(n_eeg, n_rois, 5))
        cca.fit(eeg_features, fmri_timeseries)

        # Use loadings as coupling
        coupling = cca.x_loadings_ @ cca.y_loadings_.T

    else:
        raise ValueError(f"Unknown method: {method}")

    return MultimodalConnectivityResult(
        coupling_matrix=coupling,
        eeg_features=eeg_feature_names or [f'EEG_{i}' for i in range(n_eeg)],
        fmri_rois=roi_names or [f'ROI_{i}' for i in range(n_rois)],
        method=method,
        metadata={'lag': lag, 'n_timepoints': n_times}
    )


def _partial_correlation_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute partial correlation between X and Y features."""
    n_x = X.shape[1]
    n_y = Y.shape[1]

    # Combined data
    combined = np.column_stack([X, Y])

    # Correlation matrix
    corr = np.corrcoef(combined.T)

    # Precision matrix
    try:
        precision = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(corr)

    # Partial correlation
    d = np.sqrt(np.diag(precision))
    partial = -precision / np.outer(d, d)

    # Extract X-Y block
    return partial[:n_x, n_x:]


def information_flow(
    eeg_features: np.ndarray,
    fmri_timeseries: np.ndarray,
    method: str = 'granger',
    max_lag: int = 5,
    eeg_feature_names: Optional[List[str]] = None,
    roi_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute directional information flow between EEG and fMRI.

    Parameters
    ----------
    eeg_features : np.ndarray
        EEG features (n_timepoints, n_features)
    fmri_timeseries : np.ndarray
        fMRI timeseries (n_timepoints, n_rois)
    method : str
        'granger', 'transfer_entropy', or 'pdc' (partial directed coherence)
    max_lag : int
        Maximum lag for Granger/TE
    eeg_feature_names : List[str], optional
        Names of EEG features
    roi_names : List[str], optional
        Names of ROIs

    Returns
    -------
    Dict
        Information flow matrices (EEG->fMRI and fMRI->EEG)
    """
    n_times = min(eeg_features.shape[0], fmri_timeseries.shape[0])
    eeg_features = eeg_features[:n_times]
    fmri_timeseries = fmri_timeseries[:n_times]

    n_eeg = eeg_features.shape[1]
    n_rois = fmri_timeseries.shape[1]

    if method == 'granger':
        # Granger causality
        eeg_to_fmri = np.zeros((n_eeg, n_rois))
        fmri_to_eeg = np.zeros((n_rois, n_eeg))

        for i in range(n_eeg):
            for j in range(n_rois):
                # EEG -> fMRI
                gc_ef = _granger_causality(eeg_features[:, i], fmri_timeseries[:, j], max_lag)
                eeg_to_fmri[i, j] = gc_ef

                # fMRI -> EEG
                gc_fe = _granger_causality(fmri_timeseries[:, j], eeg_features[:, i], max_lag)
                fmri_to_eeg[j, i] = gc_fe

        return {
            'eeg_to_fmri': eeg_to_fmri,
            'fmri_to_eeg': fmri_to_eeg,
            'net_flow': eeg_to_fmri - fmri_to_eeg.T,  # Positive = EEG drives fMRI
            'method': method,
            'max_lag': max_lag
        }

    elif method == 'transfer_entropy':
        eeg_to_fmri = np.zeros((n_eeg, n_rois))
        fmri_to_eeg = np.zeros((n_rois, n_eeg))

        for i in range(n_eeg):
            for j in range(n_rois):
                te_ef = _transfer_entropy(eeg_features[:, i], fmri_timeseries[:, j], max_lag)
                eeg_to_fmri[i, j] = te_ef

                te_fe = _transfer_entropy(fmri_timeseries[:, j], eeg_features[:, i], max_lag)
                fmri_to_eeg[j, i] = te_fe

        return {
            'eeg_to_fmri': eeg_to_fmri,
            'fmri_to_eeg': fmri_to_eeg,
            'net_flow': eeg_to_fmri - fmri_to_eeg.T,
            'method': method,
            'max_lag': max_lag
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def _granger_causality(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    """Simple bivariate Granger causality test."""
    n = len(y) - max_lag

    # Restricted model: y from its own past
    Y_restricted = np.zeros((n, max_lag))
    for lag in range(max_lag):
        Y_restricted[:, lag] = y[max_lag - lag - 1:n + max_lag - lag - 1]

    y_target = y[max_lag:]

    # OLS for restricted model
    beta_r = np.linalg.lstsq(Y_restricted, y_target, rcond=None)[0]
    resid_r = y_target - Y_restricted @ beta_r
    rss_r = np.sum(resid_r ** 2)

    # Unrestricted model: y from both y and x past
    X_unrestricted = np.zeros((n, 2 * max_lag))
    for lag in range(max_lag):
        X_unrestricted[:, lag] = y[max_lag - lag - 1:n + max_lag - lag - 1]
        X_unrestricted[:, max_lag + lag] = x[max_lag - lag - 1:n + max_lag - lag - 1]

    beta_u = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)[0]
    resid_u = y_target - X_unrestricted @ beta_u
    rss_u = np.sum(resid_u ** 2)

    # F-statistic (Granger causality index)
    gc = np.log(rss_r / rss_u) if rss_u > 0 else 0

    return max(0, gc)  # Clip at 0


def _transfer_entropy(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Simplified transfer entropy using binning."""
    n = len(y) - k

    # Discretize
    n_bins = 4
    x_binned = np.digitize(x, np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1]))
    y_binned = np.digitize(y, np.percentile(y, np.linspace(0, 100, n_bins + 1)[1:-1]))

    # Build joint histograms
    # TE(X->Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)

    y_future = y_binned[k:]
    y_past = y_binned[k-1:-1]
    x_past = x_binned[k-1:-1]

    # Conditional entropy using histogram
    def entropy(x):
        _, counts = np.unique(x, return_counts=True)
        p = counts / len(x)
        return -np.sum(p * np.log2(p + 1e-10))

    def cond_entropy(y, x):
        # H(Y|X) = H(Y,X) - H(X)
        joint = list(zip(y, x))
        return entropy(joint) - entropy(x)

    h_y_given_ypast = cond_entropy(y_future, y_past)

    # For conditioning on both y_past and x_past
    combined_past = list(zip(y_past, x_past))
    h_y_given_both = cond_entropy(y_future, combined_past)

    te = h_y_given_ypast - h_y_given_both
    return max(0, te)


def joint_connectivity_graph(
    eeg_connectivity: np.ndarray,
    fmri_connectivity: np.ndarray,
    coupling: np.ndarray,
    eeg_names: Optional[List[str]] = None,
    fmri_names: Optional[List[str]] = None,
    threshold: float = 0.0
) -> Dict:
    """
    Create joint EEG-fMRI connectivity graph.

    Parameters
    ----------
    eeg_connectivity : np.ndarray
        EEG connectivity matrix (n_eeg, n_eeg)
    fmri_connectivity : np.ndarray
        fMRI connectivity matrix (n_rois, n_rois)
    coupling : np.ndarray
        EEG-fMRI coupling matrix (n_eeg, n_rois)
    eeg_names : List[str], optional
        EEG node names
    fmri_names : List[str], optional
        fMRI node names
    threshold : float
        Minimum connection strength

    Returns
    -------
    Dict
        Joint graph representation
    """
    n_eeg = eeg_connectivity.shape[0]
    n_rois = fmri_connectivity.shape[0]
    n_total = n_eeg + n_rois

    # Create combined adjacency matrix
    combined = np.zeros((n_total, n_total))

    # EEG-EEG block
    combined[:n_eeg, :n_eeg] = eeg_connectivity

    # fMRI-fMRI block
    combined[n_eeg:, n_eeg:] = fmri_connectivity

    # EEG-fMRI coupling blocks
    combined[:n_eeg, n_eeg:] = coupling
    combined[n_eeg:, :n_eeg] = coupling.T

    # Apply threshold
    combined[np.abs(combined) < threshold] = 0

    # Node labels and types
    node_names = []
    node_types = []

    if eeg_names:
        node_names.extend(eeg_names)
    else:
        node_names.extend([f'EEG_{i}' for i in range(n_eeg)])
    node_types.extend(['eeg'] * n_eeg)

    if fmri_names:
        node_names.extend(fmri_names)
    else:
        node_names.extend([f'ROI_{i}' for i in range(n_rois)])
    node_types.extend(['fmri'] * n_rois)

    return {
        'adjacency': combined,
        'node_names': node_names,
        'node_types': node_types,
        'n_eeg': n_eeg,
        'n_rois': n_rois,
        'threshold': threshold
    }


def cross_modal_correlation(
    eeg_fc: np.ndarray,
    fmri_fc: np.ndarray,
    method: str = 'mantel'
) -> Tuple[float, float]:
    """
    Test correlation between EEG and fMRI connectivity patterns.

    Parameters
    ----------
    eeg_fc : np.ndarray
        EEG functional connectivity matrix
    fmri_fc : np.ndarray
        fMRI functional connectivity matrix (must match EEG parcellation)
    method : str
        'mantel' or 'pearson'

    Returns
    -------
    r : float
        Correlation coefficient
    p : float
        P-value
    """
    # Get upper triangle
    n = eeg_fc.shape[0]
    triu_idx = np.triu_indices(n, k=1)

    eeg_vec = eeg_fc[triu_idx]
    fmri_vec = fmri_fc[triu_idx]

    if method == 'mantel':
        # Mantel test with permutation
        observed_r, _ = stats.spearmanr(eeg_vec, fmri_vec)

        n_perm = 1000
        perm_r = np.zeros(n_perm)

        for i in range(n_perm):
            # Permute rows/columns together (matrix permutation)
            perm_idx = np.random.permutation(n)
            eeg_perm = eeg_fc[np.ix_(perm_idx, perm_idx)]
            perm_r[i], _ = stats.spearmanr(eeg_perm[triu_idx], fmri_vec)

        p_value = np.mean(np.abs(perm_r) >= np.abs(observed_r))
        return observed_r, p_value

    elif method == 'pearson':
        r, p = stats.pearsonr(eeg_vec, fmri_vec)
        return r, p

    else:
        raise ValueError(f"Unknown method: {method}")


def optimal_lag_coupling(
    eeg_features: np.ndarray,
    fmri_timeseries: np.ndarray,
    max_lag: int = 10,
    roi_names: Optional[List[str]] = None
) -> Dict:
    """
    Find optimal lag for EEG-fMRI coupling.

    Parameters
    ----------
    eeg_features : np.ndarray
        EEG features (n_timepoints, n_features)
    fmri_timeseries : np.ndarray
        fMRI timeseries (n_timepoints, n_rois)
    max_lag : int
        Maximum lag to test (in TRs)
    roi_names : List[str], optional
        ROI names

    Returns
    -------
    Dict
        Optimal lags and correlations for each EEG-ROI pair
    """
    n_eeg = eeg_features.shape[1]
    n_rois = fmri_timeseries.shape[1]

    optimal_lags = np.zeros((n_eeg, n_rois), dtype=int)
    max_correlations = np.zeros((n_eeg, n_rois))

    lags = np.arange(-max_lag, max_lag + 1)

    for i in range(n_eeg):
        for j in range(n_rois):
            correlations = []

            for lag in lags:
                if lag > 0:
                    eeg = eeg_features[:-lag, i]
                    fmri = fmri_timeseries[lag:, j]
                elif lag < 0:
                    eeg = eeg_features[-lag:, i]
                    fmri = fmri_timeseries[:lag, j]
                else:
                    eeg = eeg_features[:, i]
                    fmri = fmri_timeseries[:, j]

                r, _ = stats.pearsonr(eeg, fmri)
                correlations.append(r)

            correlations = np.array(correlations)
            best_idx = np.argmax(np.abs(correlations))
            optimal_lags[i, j] = lags[best_idx]
            max_correlations[i, j] = correlations[best_idx]

    return {
        'optimal_lags': optimal_lags,
        'max_correlations': max_correlations,
        'lags_tested': lags,
        'roi_names': roi_names
    }
