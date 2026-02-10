"""
Temporal visualization for EEG/MEG decoding.

Plot time-resolved decoding accuracy and temporal generalization matrices.
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt


def plot_temporal_decoding(
    times: np.ndarray,
    scores: np.ndarray,
    scores_std: Optional[np.ndarray] = None,
    chance_level: float = 0.5,
    significant_times: Optional[List[Tuple[float, float]]] = None,
    title: str = "Temporal Decoding",
    xlabel: str = "Time (s)",
    ylabel: str = "Accuracy",
    color: str = "blue",
    figsize: Tuple[float, float] = (10, 5),
    output_path: Optional[str] = None
):
    """
    Plot time-resolved decoding accuracy.

    Args:
        times: Time points in seconds
        scores: Accuracy at each time point
        scores_std: Standard deviation (for shading)
        chance_level: Horizontal line for chance
        significant_times: List of (start, end) tuples for significant periods
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Line color
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Main accuracy line
    ax.plot(times, scores, color=color, linewidth=2, label='Accuracy')

    # Standard deviation shading
    if scores_std is not None:
        ax.fill_between(
            times,
            scores - scores_std,
            scores + scores_std,
            alpha=0.3,
            color=color
        )

    # Significant periods
    if significant_times:
        for start, end in significant_times:
            ax.axvspan(start, end, alpha=0.2, color='green', label='Significant')

    # Chance level
    ax.axhline(y=chance_level, color='gray', linestyle='--',
               label=f'Chance ({chance_level:.0%})')

    # Stimulus onset
    if times[0] < 0 < times[-1]:
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_temporal_generalization(
    scores: np.ndarray,
    times: np.ndarray,
    chance_level: float = 0.5,
    cmap: str = "RdBu_r",
    title: str = "Temporal Generalization",
    figsize: Tuple[float, float] = (8, 8),
    output_path: Optional[str] = None
):
    """
    Plot temporal generalization matrix.

    Args:
        scores: Generalization matrix (train_times x test_times)
        times: Time points in seconds
        chance_level: Value to center colormap on
        cmap: Colormap
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Center colormap on chance
    vmax = np.max(np.abs(scores - chance_level)) + chance_level
    vmin = chance_level - (vmax - chance_level)

    im = ax.imshow(
        scores,
        origin='lower',
        extent=[times[0], times[-1], times[0], times[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal'
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy')

    # Diagonal (standard temporal decoding)
    ax.plot([times[0], times[-1]], [times[0], times[-1]],
            'k--', linewidth=1, alpha=0.5)

    # Axes at stimulus onset
    if times[0] < 0 < times[-1]:
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Test Time (s)')
    ax.set_ylabel('Train Time (s)')
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_temporal_comparison(
    results_list: List[dict],
    labels: List[str],
    chance_level: float = 0.5,
    colors: Optional[List[str]] = None,
    title: str = "Temporal Decoding Comparison",
    figsize: Tuple[float, float] = (10, 5),
    output_path: Optional[str] = None
):
    """
    Compare temporal decoding across conditions or subjects.

    Args:
        results_list: List of dicts with 'times', 'scores', 'scores_std'
        labels: Labels for each result
        chance_level: Horizontal line for chance
        colors: Colors for each line
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    fig, ax = plt.subplots(figsize=figsize)

    for i, (result, label) in enumerate(zip(results_list, labels)):
        times = result['times']
        scores = result['scores']
        scores_std = result.get('scores_std')

        ax.plot(times, scores, color=colors[i], linewidth=2, label=label)

        if scores_std is not None:
            ax.fill_between(
                times,
                scores - scores_std,
                scores + scores_std,
                alpha=0.2,
                color=colors[i]
            )

    ax.axhline(y=chance_level, color='gray', linestyle='--', label='Chance')

    if results_list[0]['times'][0] < 0 < results_list[0]['times'][-1]:
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
    ax.legend(loc='best')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_decoding_dynamics(
    times: np.ndarray,
    scores: np.ndarray,
    erp: Optional[np.ndarray] = None,
    erp_times: Optional[np.ndarray] = None,
    chance_level: float = 0.5,
    title: str = "Decoding Dynamics",
    figsize: Tuple[float, float] = (12, 5),
    output_path: Optional[str] = None
):
    """
    Plot decoding accuracy alongside ERP.

    Args:
        times: Decoding time points
        scores: Decoding accuracy
        erp: ERP waveform (optional)
        erp_times: ERP time points
        chance_level: Chance level
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if erp is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # Decoding accuracy
    ax1.plot(times, scores, 'b-', linewidth=2)
    ax1.axhline(y=chance_level, color='gray', linestyle='--')
    ax1.set_ylabel('Decoding Accuracy')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

    if times[0] < 0 < times[-1]:
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # ERP
    if ax2 is not None and erp is not None:
        erp_times = erp_times if erp_times is not None else times
        ax2.plot(erp_times, erp, 'k-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('ERP Amplitude (µV)')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        if erp_times[0] < 0 < erp_times[-1]:
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    else:
        ax1.set_xlabel('Time (s)')

    fig.suptitle(title)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
