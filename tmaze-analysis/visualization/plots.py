"""
Visualization functions for T-maze analysis.

Plotting utilities for temporal decoding, ROI results, and multimodal analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from ..classification.classifiers import ClassificationResult
from ..classification.temporal import TemporalDecodingResult
from ..classification.multimodal import MultimodalResult


def plot_temporal_decoding(
    result: TemporalDecodingResult,
    chance_level: float = 0.5,
    show_significance: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "Temporal Decoding",
    ylabel: str = "AUC / Accuracy",
    color: str = 'steelblue',
    **kwargs
) -> plt.Figure:
    """
    Plot temporal decoding results.

    Parameters
    ----------
    result : TemporalDecodingResult
        Temporal decoding results
    chance_level : float
        Chance level for reference line
    show_significance : bool
        Highlight significant time windows
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title
    ylabel : str
        Y-axis label
    color : str
        Line color

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    times = result.times
    scores = result.scores
    scores_std = result.scores_std

    # Plot mean with confidence band
    ax.plot(times, scores, color=color, linewidth=2, label='Decoding')
    ax.fill_between(
        times,
        scores - scores_std,
        scores + scores_std,
        alpha=0.3,
        color=color
    )

    # Chance level
    ax.axhline(y=chance_level, color='gray', linestyle='--',
               linewidth=1, label='Chance')

    # Zero line (stimulus onset)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Highlight significant periods
    if show_significance and result.significant_clusters:
        for start, end in result.significant_clusters:
            ax.axvspan(start, end, alpha=0.2, color='green',
                      label='Significant' if start == result.significant_clusters[0][0] else None)

    # Mark peak
    if result.peak_time is not None:
        ax.plot(result.peak_time, result.peak_score, 'r*', markersize=15,
               label=f'Peak: {result.peak_score:.2f} at {result.peak_time*1000:.0f}ms')

    # REWP window
    ax.axvspan(0.24, 0.34, alpha=0.1, color='orange', label='REWP window')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)

    # Set axis limits
    ax.set_xlim([times[0], times[-1]])

    plt.tight_layout()
    return fig


def plot_temporal_generalization(
    gen_matrix: np.ndarray,
    times: np.ndarray,
    chance_level: float = 0.5,
    ax: Optional[plt.Axes] = None,
    title: str = "Temporal Generalization",
    cmap: str = 'RdBu_r',
    **kwargs
) -> plt.Figure:
    """
    Plot temporal generalization matrix.

    Parameters
    ----------
    gen_matrix : np.ndarray
        Generalization matrix (n_times, n_times)
    times : np.ndarray
        Time vector
    chance_level : float
        Chance level for centering colormap
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title
    cmap : str
        Colormap

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Center colormap around chance
    vmax = max(abs(gen_matrix.max() - chance_level),
               abs(gen_matrix.min() - chance_level))
    vmin = chance_level - vmax
    vmax = chance_level + vmax

    im = ax.imshow(
        gen_matrix,
        extent=[times[0], times[-1], times[-1], times[0]],
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='upper'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Score')

    # Reference lines
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    # Diagonal
    ax.plot([times[0], times[-1]], [times[0], times[-1]],
            'k--', linewidth=0.5)

    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_roi_accuracies(
    roi_results: Dict[str, ClassificationResult],
    n_top: int = 20,
    chance_level: float = 0.5,
    ax: Optional[plt.Axes] = None,
    title: str = "Top ROI Classification Accuracies",
    **kwargs
) -> plt.Figure:
    """
    Plot top ROI classification accuracies.

    Parameters
    ----------
    roi_results : Dict[str, ClassificationResult]
        Results from classify_all_rois
    n_top : int
        Number of top ROIs to show
    chance_level : float
        Chance level
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # Sort by accuracy
    sorted_results = sorted(
        roi_results.items(),
        key=lambda x: x[1].accuracy,
        reverse=True
    )[:n_top]

    names = [r[0] for r in sorted_results]
    accuracies = [r[1].accuracy for r in sorted_results]
    stds = [r[1].accuracy_std for r in sorted_results]

    y_pos = np.arange(len(names))

    # Horizontal bar plot
    bars = ax.barh(y_pos, accuracies, xerr=stds,
                   color='steelblue', edgecolor='black', capsize=3)

    # Chance line
    ax.axvline(x=chance_level, color='red', linestyle='--',
               label='Chance', linewidth=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    result: ClassificationResult,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "Confusion Matrix",
    cmap: str = 'Blues',
    **kwargs
) -> plt.Figure:
    """
    Plot confusion matrix.

    Parameters
    ----------
    result : ClassificationResult
        Classification result with confusion matrix
    class_names : List[str], optional
        Class labels
    normalize : bool
        Normalize by row
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title
    cmap : str
        Colormap

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    cm = result.confusion_matrix

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    return fig


def plot_multimodal_comparison(
    result: MultimodalResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Multimodal Fusion Comparison",
    **kwargs
) -> plt.Figure:
    """
    Plot multimodal fusion results comparison.

    Parameters
    ----------
    result : MultimodalResult
        Multimodal classification result
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    methods = ['EEG Only', 'fMRI Only', 'Fused']
    accuracies = [
        result.eeg_only_accuracy,
        result.fmri_only_accuracy,
        result.accuracy
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black')

    # Add error bar for fused
    ax.errorbar(2, result.accuracy, yerr=result.accuracy_std,
               color='black', capsize=5)

    # Chance line
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{acc:.1%}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim([0.4, 1.0])

    # Add improvement annotation
    if result.fusion_improvement > 0:
        ax.annotate(
            f'+{result.fusion_improvement:.1%}',
            xy=(2, result.accuracy),
            xytext=(2.3, result.accuracy + 0.05),
            arrowprops=dict(arrowstyle='->', color='green'),
            color='green',
            fontsize=12,
            fontweight='bold'
        )

    plt.tight_layout()
    return fig


def plot_rewp_waveform(
    eeg_data,  # TMazeEEGData
    condition_colors: Optional[Dict[str, str]] = None,
    channels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "REWP Waveform",
    **kwargs
) -> plt.Figure:
    """
    Plot ERP waveforms by condition.

    Parameters
    ----------
    eeg_data : TMazeEEGData
        EEG data
    condition_colors : Dict[str, str], optional
        Color for each condition
    channels : List[str], optional
        Channels to average
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    times = eeg_data.times * 1000  # Convert to ms

    # Default colors
    if condition_colors is None:
        condition_colors = {
            'Reward': '#2ecc71',
            'No Reward': '#e74c3c'
        }

    # Get channel data
    if channels:
        channel_indices = [eeg_data.channels.index(ch)
                          for ch in channels if ch in eeg_data.channels]
        data = eeg_data.data[:, channel_indices, :].mean(axis=1)
    else:
        data = eeg_data.data.mean(axis=1)

    # Plot by condition
    for label_val in np.unique(eeg_data.labels):
        mask = eeg_data.labels == label_val
        condition_data = data[mask]

        mean = condition_data.mean(axis=0)
        sem = condition_data.std(axis=0) / np.sqrt(mask.sum())

        label = eeg_data.condition_names[label_val] if label_val < len(eeg_data.condition_names) else f'Condition {label_val}'
        color = list(condition_colors.values())[label_val % len(condition_colors)]

        ax.plot(times, mean, color=color, linewidth=2, label=label)
        ax.fill_between(times, mean - sem, mean + sem, alpha=0.3, color=color)

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # REWP window
    ax.axvspan(240, 340, alpha=0.2, color='yellow', label='REWP window')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(title)
    ax.legend(loc='upper right')

    # Invert y-axis (ERP convention)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_rdm(
    rdm: np.ndarray,
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Representational Dissimilarity Matrix",
    cmap: str = 'viridis',
    **kwargs
) -> plt.Figure:
    """
    Plot Representational Dissimilarity Matrix.

    Parameters
    ----------
    rdm : np.ndarray
        RDM (n_conditions, n_conditions)
    labels : List[str], optional
        Condition labels
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title
    cmap : str
        Colormap

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    if labels is None:
        labels = [f'Cond {i}' for i in range(rdm.shape[0])]

    im = ax.imshow(rdm, cmap=cmap, vmin=0)
    plt.colorbar(im, ax=ax, label='Dissimilarity')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)

    # Add values
    for i in range(rdm.shape[0]):
        for j in range(rdm.shape[1]):
            ax.text(j, i, f'{rdm[i, j]:.2f}',
                   ha='center', va='center',
                   color='white' if rdm[i, j] > rdm.max()/2 else 'black',
                   fontsize=10)

    plt.tight_layout()
    return fig


def plot_subject_accuracies(
    subject_results: Dict[str, ClassificationResult],
    chance_level: float = 0.5,
    ax: Optional[plt.Axes] = None,
    title: str = "Per-Subject Classification Accuracy",
    **kwargs
) -> plt.Figure:
    """
    Plot accuracy across subjects.

    Parameters
    ----------
    subject_results : Dict[str, ClassificationResult]
        {subject_id: result}
    chance_level : float
        Chance level
    ax : plt.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    subjects = list(subject_results.keys())
    accuracies = [r.accuracy for r in subject_results.values()]
    stds = [r.accuracy_std for r in subject_results.values()]

    x = np.arange(len(subjects))

    bars = ax.bar(x, accuracies, yerr=stds, capsize=3,
                  color='steelblue', edgecolor='black')

    ax.axhline(y=chance_level, color='red', linestyle='--',
               label='Chance', linewidth=2)
    ax.axhline(y=np.mean(accuracies), color='green', linestyle='-',
               label=f'Mean: {np.mean(accuracies):.1%}', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


def create_summary_figure(
    temporal_result: Optional[TemporalDecodingResult] = None,
    roi_results: Optional[Dict[str, ClassificationResult]] = None,
    multimodal_result: Optional[MultimodalResult] = None,
    rdm: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create multi-panel summary figure.

    Parameters
    ----------
    temporal_result : TemporalDecodingResult, optional
        Temporal decoding results
    roi_results : Dict[str, ClassificationResult], optional
        ROI classification results
    multimodal_result : MultimodalResult, optional
        Multimodal fusion results
    rdm : np.ndarray, optional
        RDM for RSA panel
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
    """
    n_panels = sum([
        temporal_result is not None,
        roi_results is not None,
        multimodal_result is not None,
        rdm is not None
    ])

    if n_panels == 0:
        raise ValueError("At least one result must be provided")

    # Create figure with appropriate layout
    fig = plt.figure(figsize=figsize)

    if n_panels == 1:
        axes = [fig.add_subplot(1, 1, 1)]
    elif n_panels == 2:
        axes = [fig.add_subplot(1, 2, i+1) for i in range(2)]
    elif n_panels == 3:
        axes = [fig.add_subplot(2, 2, i+1) for i in range(3)]
    else:
        axes = [fig.add_subplot(2, 2, i+1) for i in range(4)]

    ax_idx = 0

    if temporal_result is not None:
        plot_temporal_decoding(temporal_result, ax=axes[ax_idx])
        ax_idx += 1

    if roi_results is not None:
        plot_roi_accuracies(roi_results, ax=axes[ax_idx])
        ax_idx += 1

    if multimodal_result is not None:
        plot_multimodal_comparison(multimodal_result, ax=axes[ax_idx])
        ax_idx += 1

    if rdm is not None:
        plot_rdm(rdm, ax=axes[ax_idx])
        ax_idx += 1

    plt.tight_layout()
    return fig
