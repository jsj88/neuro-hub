"""
Representational Similarity Analysis for T-maze data.

RSA compares neural representations to theoretical models.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from ..core.containers import TMAzefMRIData, TMazeConditions


def compute_rdm(
    patterns: np.ndarray,
    metric: str = 'correlation'
) -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix.

    Parameters
    ----------
    patterns : np.ndarray
        Pattern data (n_conditions, n_features)
    metric : str
        Distance metric ('correlation', 'euclidean', 'cosine')

    Returns
    -------
    np.ndarray
        RDM (n_conditions, n_conditions)
    """
    if metric == 'correlation':
        # 1 - correlation
        rdm = 1 - np.corrcoef(patterns)
    else:
        # Use scipy distance
        distances = pdist(patterns, metric=metric)
        rdm = squareform(distances)

    return rdm


def model_rdm_tmaze(
    model_type: str = 'reward'
) -> np.ndarray:
    """
    Create theoretical model RDM for T-maze conditions.

    Parameters
    ----------
    model_type : str
        'reward': Reward vs No-Reward
        'maze': Maze vs No-Maze
        'full': All 4 conditions distinct

    Returns
    -------
    np.ndarray
        Model RDM (4, 4) for T-maze conditions
    """
    # Condition order: MazeReward, MazeNoReward, NoMazeReward, NoMazeNoReward

    if model_type == 'reward':
        # Reward conditions similar, no-reward conditions similar
        model = np.array([
            [0, 1, 0, 1],  # MazeReward
            [1, 0, 1, 0],  # MazeNoReward
            [0, 1, 0, 1],  # NoMazeReward
            [1, 0, 1, 0]   # NoMazeNoReward
        ], dtype=float)

    elif model_type == 'maze':
        # Maze conditions similar, no-maze conditions similar
        model = np.array([
            [0, 0, 1, 1],  # MazeReward
            [0, 0, 1, 1],  # MazeNoReward
            [1, 1, 0, 0],  # NoMazeReward
            [1, 1, 0, 0]   # NoMazeNoReward
        ], dtype=float)

    elif model_type == 'interaction':
        # Maze x Reward interaction
        model = np.array([
            [0, 1, 1, 0],  # MazeReward
            [1, 0, 0, 1],  # MazeNoReward
            [1, 0, 0, 1],  # NoMazeReward
            [0, 1, 1, 0]   # NoMazeNoReward
        ], dtype=float)

    elif model_type == 'full':
        # All conditions distinct
        model = 1 - np.eye(4)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def compare_rdms(
    neural_rdm: np.ndarray,
    model_rdm: np.ndarray,
    method: str = 'spearman'
) -> Tuple[float, float]:
    """
    Compare neural RDM to model RDM.

    Parameters
    ----------
    neural_rdm : np.ndarray
        Neural RDM
    model_rdm : np.ndarray
        Model RDM
    method : str
        'spearman' or 'pearson'

    Returns
    -------
    r : float
        Correlation coefficient
    p : float
        P-value
    """
    # Get lower triangle (excluding diagonal)
    mask = np.tril(np.ones_like(neural_rdm, dtype=bool), k=-1)
    neural_vec = neural_rdm[mask]
    model_vec = model_rdm[mask]

    if method == 'spearman':
        r, p = stats.spearmanr(neural_vec, model_vec)
    elif method == 'pearson':
        r, p = stats.pearsonr(neural_vec, model_vec)
    else:
        raise ValueError(f"Unknown method: {method}")

    return r, p


def condition_average_patterns(
    fmri_data: TMAzefMRIData,
    conditions: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Average patterns by condition for RSA.

    Parameters
    ----------
    fmri_data : TMAzefMRIData
        fMRI data
    conditions : List[str], optional
        Condition names to include

    Returns
    -------
    patterns : np.ndarray
        Averaged patterns (n_conditions, n_features)
    condition_names : List[str]
        Condition labels in order
    """
    if conditions is None:
        conditions = fmri_data.condition_names

    patterns = []
    for i, cond in enumerate(conditions):
        mask = fmri_data.labels == i
        if np.any(mask):
            mean_pattern = fmri_data.data[mask].mean(axis=0)
            patterns.append(mean_pattern)

    return np.array(patterns), conditions


def roi_rsa(
    fmri_data: TMAzefMRIData,
    model_type: str = 'reward',
    method: str = 'spearman'
) -> Tuple[float, float, np.ndarray]:
    """
    RSA for a single ROI or region.

    Parameters
    ----------
    fmri_data : TMAzefMRIData
        fMRI data
    model_type : str
        Model type for comparison
    method : str
        Correlation method

    Returns
    -------
    r : float
        Correlation with model
    p : float
        P-value
    neural_rdm : np.ndarray
        Neural RDM
    """
    # Get condition-averaged patterns
    patterns, _ = condition_average_patterns(fmri_data)

    # Compute neural RDM
    neural_rdm = compute_rdm(patterns, metric='correlation')

    # Get model RDM
    model_rdm = model_rdm_tmaze(model_type)

    # Compare
    r, p = compare_rdms(neural_rdm, model_rdm, method=method)

    return r, p, neural_rdm


def searchlight_rsa(
    bold_img_path: str,
    mask_path: str,
    labels: np.ndarray,
    model_type: str = 'reward',
    radius: float = 5.0,
    n_jobs: int = -1
) -> 'nibabel.Nifti1Image':
    """
    Searchlight RSA analysis.

    Parameters
    ----------
    bold_img_path : str
        Path to 4D BOLD image
    mask_path : str
        Path to brain mask
    labels : np.ndarray
        Condition labels for each volume
    model_type : str
        Model type for comparison
    radius : float
        Searchlight radius in mm
    n_jobs : int
        Parallel jobs

    Returns
    -------
    Nifti1Image
        RSA correlation map
    """
    try:
        import nibabel as nib
        from nilearn.image import load_img
        from nilearn.maskers import NiftiSpheresMasker
        from nilearn.decoding import SearchLight
    except ImportError:
        raise ImportError("nibabel and nilearn required for searchlight RSA")

    # This is a placeholder - full implementation would use
    # custom searchlight with RSA metric

    raise NotImplementedError(
        "Full searchlight RSA requires custom implementation. "
        "Consider using RSA toolbox or pyrsa package."
    )


def rsa_multiple_models(
    fmri_data: TMAzefMRIData,
    model_types: List[str] = ['reward', 'maze', 'interaction']
) -> Dict[str, Tuple[float, float]]:
    """
    Compare neural RDM to multiple model RDMs.

    Parameters
    ----------
    fmri_data : TMAzefMRIData
        fMRI data
    model_types : List[str]
        Model types to compare

    Returns
    -------
    Dict[str, Tuple[float, float]]
        {model_type: (r, p)} for each model
    """
    # Get condition-averaged patterns
    patterns, _ = condition_average_patterns(fmri_data)
    neural_rdm = compute_rdm(patterns)

    results = {}
    for model_type in model_types:
        model_rdm = model_rdm_tmaze(model_type)
        r, p = compare_rdms(neural_rdm, model_rdm)
        results[model_type] = (r, p)

    return results


def roi_rsa_all(
    fmri_data: TMAzefMRIData,
    model_type: str = 'reward',
    verbose: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    RSA for each ROI independently.

    Parameters
    ----------
    fmri_data : TMAzefMRIData
        fMRI data with ROI values
    model_type : str
        Model type
    verbose : bool
        Print progress

    Returns
    -------
    Dict[str, Tuple[float, float]]
        {roi_name: (r, p)} for each ROI
    """
    results = {}
    model_rdm = model_rdm_tmaze(model_type)

    # Get condition labels
    unique_labels = np.unique(fmri_data.labels)

    for i, roi_name in enumerate(fmri_data.roi_names):
        if verbose and i % 50 == 0:
            print(f"Processing ROI {i+1}/{fmri_data.n_rois}")

        # Get patterns for this ROI
        roi_data = fmri_data.data[:, i:i+1]  # Keep 2D

        # Average by condition
        patterns = []
        for label in unique_labels:
            mask = fmri_data.labels == label
            patterns.append(roi_data[mask].mean())

        patterns = np.array(patterns).reshape(-1, 1)

        # For single feature, use simple distance
        neural_rdm = compute_rdm(patterns, metric='euclidean')

        # Compare to model
        r, p = compare_rdms(neural_rdm, model_rdm)
        results[roi_name] = (r, p)

    return results
