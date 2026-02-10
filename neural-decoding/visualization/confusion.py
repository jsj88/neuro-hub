"""
Confusion matrix visualization for neural decoding.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[str] = None
):
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix (n_classes x n_classes)
        class_names: Class labels
        normalize: Normalize by row (true labels)
        cmap: Colormap
        title: Plot title
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    import seaborn as sns

    cm = confusion_matrix.copy()
    n_classes = cm.shape[0]

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title_suffix = " (Normalized)"
    else:
        fmt = "d"
        title_suffix = ""

    fig, ax = plt.subplots(figsize=figsize)

    labels = class_names if class_names else [str(i) for i in range(n_classes)]

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title + title_suffix)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_multi_confusion(
    confusion_matrices: List[np.ndarray],
    titles: List[str],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    cmap: str = "Blues",
    figsize_per_plot: Tuple[float, float] = (5, 4),
    n_cols: int = 3,
    output_path: Optional[str] = None
):
    """
    Plot multiple confusion matrices in a grid.

    Args:
        confusion_matrices: List of confusion matrices
        titles: Title for each matrix
        class_names: Class labels
        normalize: Normalize by row
        cmap: Colormap
        figsize_per_plot: Size per subplot
        n_cols: Number of columns
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    import seaborn as sns

    n_plots = len(confusion_matrices)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )

    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (cm, title) in enumerate(zip(confusion_matrices, titles)):
        ax = axes[i]

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            fmt = "d"

        labels = class_names if class_names else [str(j) for j in range(cm.shape[0])]

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=0,
            vmax=1 if normalize else None,
            cbar=False
        )

        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_difference(
    cm1: np.ndarray,
    cm2: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix Difference",
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[str] = None
):
    """
    Plot difference between two confusion matrices.

    Args:
        cm1: First confusion matrix
        cm2: Second confusion matrix
        class_names: Class labels
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    import seaborn as sns

    # Normalize both
    cm1_norm = cm1.astype(float) / cm1.sum(axis=1, keepdims=True)
    cm2_norm = cm2.astype(float) / cm2.sum(axis=1, keepdims=True)

    # Difference
    diff = cm1_norm - cm2_norm

    fig, ax = plt.subplots(figsize=figsize)

    labels = class_names if class_names else [str(i) for i in range(diff.shape[0])]

    # Center colormap at 0
    vmax = np.max(np.abs(diff))

    sns.heatmap(
        diff,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=-vmax,
        vmax=vmax,
        center=0
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_class_accuracy(
    accuracy_per_class: dict,
    title: str = "Per-Class Accuracy",
    color: str = "steelblue",
    figsize: Tuple[float, float] = (10, 5),
    output_path: Optional[str] = None
):
    """
    Plot bar chart of per-class accuracy.

    Args:
        accuracy_per_class: Dict of {class_name: accuracy}
        title: Plot title
        color: Bar color
        figsize: Figure size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    classes = list(accuracy_per_class.keys())
    accuracies = list(accuracy_per_class.values())

    bars = ax.bar(classes, accuracies, color=color, edgecolor='black')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{acc:.1%}',
            ha='center',
            va='bottom'
        )

    # Mean line
    mean_acc = np.mean(accuracies)
    ax.axhline(y=mean_acc, color='red', linestyle='--',
               label=f'Mean: {mean_acc:.1%}')

    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig
