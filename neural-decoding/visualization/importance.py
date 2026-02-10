"""
Feature importance visualization for neural decoding.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    title: str = "Feature Importance",
    color: str = "steelblue",
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[str] = None
):
    """
    Plot top N feature importances as horizontal bar chart.

    Args:
        importances: Feature importance values
        feature_names: Names for features
        top_n: Number of features to show
        title: Plot title
        color: Bar color
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Get top features
    top_indices = np.argsort(np.abs(importances))[::-1][:top_n]
    top_importances = importances[top_indices]

    if feature_names is not None:
        top_names = [feature_names[i] for i in top_indices]
    else:
        top_names = [f"Feature {i}" for i in top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal bars
    y_pos = np.arange(len(top_names))
    colors = [color if imp >= 0 else 'coral' for imp in top_importances]

    ax.barh(y_pos, top_importances, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # Top feature at top

    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_weight_map(
    weights: np.ndarray,
    mask_path: str,
    threshold: Optional[float] = None,
    percentile: float = 95,
    cmap: str = "RdBu_r",
    title: str = "Classifier Weights",
    output_path: Optional[str] = None
):
    """
    Plot classifier weights on brain.

    Args:
        weights: Weight vector (n_voxels,)
        mask_path: Path to brain mask
        threshold: Absolute threshold for display (overrides percentile)
        percentile: Display top percentile of weights
        cmap: Colormap
        title: Plot title
        output_path: Path to save figure

    Returns:
        Nilearn display object
    """
    from nilearn import plotting
    import nibabel as nib

    # Load mask
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # Create weight map
    weight_data = np.zeros_like(mask_data)
    weight_data[mask_data > 0] = weights

    weight_img = nib.Nifti1Image(weight_data, mask_img.affine)

    # Threshold
    if threshold is None:
        threshold = np.percentile(np.abs(weights), percentile)

    # Plot
    display = plotting.plot_stat_map(
        weight_img,
        threshold=threshold,
        cmap=cmap,
        title=title,
        colorbar=True
    )

    if output_path:
        display.savefig(output_path, dpi=300)

    return display


def plot_channel_importance(
    importances: np.ndarray,
    channel_names: List[str],
    info=None,
    title: str = "Channel Importance",
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[str] = None
):
    """
    Plot EEG channel importances on scalp topography.

    Args:
        importances: Importance per channel
        channel_names: Channel names
        info: MNE info object with channel positions
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    import mne

    if info is not None:
        # Use MNE topomap
        fig, ax = plt.subplots(figsize=figsize)

        mne.viz.plot_topomap(
            importances,
            info,
            axes=ax,
            cmap=cmap,
            show=False
        )
        ax.set_title(title)

    else:
        # Simple bar chart
        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(channel_names))
        colors = [cmap if imp >= 0 else 'coral' for imp in importances]

        ax.barh(y_pos, importances, color='steelblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(channel_names)
        ax.invert_yaxis()

        ax.set_xlabel('Importance')
        ax.set_title(title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roi_importance_bar(
    importances: np.ndarray,
    roi_names: List[str],
    top_n: int = 20,
    title: str = "ROI Importance",
    color: str = "steelblue",
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[str] = None
):
    """
    Plot ROI importances as bar chart.

    Args:
        importances: Importance per ROI
        roi_names: ROI names
        top_n: Number of ROIs to show
        title: Plot title
        color: Bar color
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Get top ROIs
    top_indices = np.argsort(np.abs(importances))[::-1][:top_n]
    top_importances = importances[top_indices]
    top_names = [roi_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_names))
    colors = [color if imp >= 0 else 'coral' for imp in top_importances]

    ax.barh(y_pos, top_importances, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()

    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_importance_stability(
    importances_list: List[np.ndarray],
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    title: str = "Feature Importance Stability",
    figsize: Tuple[float, float] = (10, 8),
    output_path: Optional[str] = None
):
    """
    Plot feature importance stability across CV folds.

    Args:
        importances_list: List of importance arrays (one per fold)
        feature_names: Feature names
        top_n: Number of features to show
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Stack and compute mean/std
    importances = np.stack(importances_list)
    mean_imp = np.mean(importances, axis=0)
    std_imp = np.std(importances, axis=0)

    # Get top features by mean importance
    top_indices = np.argsort(np.abs(mean_imp))[::-1][:top_n]

    if feature_names is not None:
        top_names = [feature_names[i] for i in top_indices]
    else:
        top_names = [f"Feature {i}" for i in top_indices]

    top_means = mean_imp[top_indices]
    top_stds = std_imp[top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_names))

    ax.barh(y_pos, top_means, xerr=top_stds,
            color='steelblue', edgecolor='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()

    ax.set_xlabel('Mean Importance ± Std')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
