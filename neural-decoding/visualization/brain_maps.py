"""
Brain visualization tools for neural decoding.

Plot accuracy maps, searchlight results, and ROI-based results on brain surfaces.
"""

from typing import Optional, Tuple, List
import numpy as np


def plot_accuracy_map(
    accuracy_map,
    mask_path: Optional[str] = None,
    threshold: float = 0.5,
    display_mode: str = "ortho",
    cut_coords: Optional[Tuple] = None,
    cmap: str = "hot",
    title: str = "Decoding Accuracy",
    colorbar: bool = True,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4)
):
    """
    Plot accuracy map on brain.

    Args:
        accuracy_map: NIfTI image or path with accuracy values
        mask_path: Optional brain mask for background
        threshold: Minimum accuracy to display
        display_mode: Nilearn display mode ("ortho", "x", "y", "z", "glass", "mosaic")
        cut_coords: Coordinates to display (None = automatic)
        cmap: Colormap
        title: Plot title
        colorbar: Show colorbar
        output_path: Path to save figure

    Returns:
        Nilearn display object
    """
    from nilearn import plotting
    import nibabel as nib

    # Load if path
    if isinstance(accuracy_map, str):
        accuracy_map = nib.load(accuracy_map)

    # Create plot
    display = plotting.plot_stat_map(
        accuracy_map,
        threshold=threshold,
        display_mode=display_mode,
        cut_coords=cut_coords,
        cmap=cmap,
        title=title,
        colorbar=colorbar
    )

    if output_path:
        display.savefig(output_path, dpi=300)

    return display


def plot_glass_brain(
    accuracy_map,
    threshold: float = 0.5,
    cmap: str = "hot",
    title: str = "Searchlight Results",
    output_path: Optional[str] = None
):
    """
    Plot accuracy map on glass brain.

    Args:
        accuracy_map: NIfTI image with accuracy values
        threshold: Minimum accuracy to display
        cmap: Colormap
        title: Plot title
        output_path: Path to save figure

    Returns:
        Nilearn display object
    """
    from nilearn import plotting
    import nibabel as nib

    if isinstance(accuracy_map, str):
        accuracy_map = nib.load(accuracy_map)

    display = plotting.plot_glass_brain(
        accuracy_map,
        threshold=threshold,
        cmap=cmap,
        title=title,
        colorbar=True,
        display_mode='lyrz'
    )

    if output_path:
        display.savefig(output_path, dpi=300)

    return display


def plot_roi_importance(
    importances: np.ndarray,
    atlas_path: str,
    roi_names: Optional[List[str]] = None,
    threshold: float = 0.0,
    cmap: str = "RdBu_r",
    title: str = "ROI Importance",
    output_path: Optional[str] = None
):
    """
    Plot feature importances on ROI atlas.

    Args:
        importances: Importance values for each ROI
        atlas_path: Path to atlas NIfTI
        roi_names: Optional ROI names
        threshold: Minimum importance to display
        cmap: Colormap
        title: Plot title
        output_path: Path to save figure

    Returns:
        Nilearn display object
    """
    from nilearn import plotting
    import nibabel as nib

    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # Create importance map
    importance_data = np.zeros_like(atlas_data)

    roi_labels = np.unique(atlas_data[atlas_data > 0]).astype(int)

    for i, label in enumerate(roi_labels):
        if i < len(importances):
            importance_data[atlas_data == label] = importances[i]

    importance_img = nib.Nifti1Image(importance_data, atlas_img.affine)

    # Plot
    display = plotting.plot_stat_map(
        importance_img,
        threshold=threshold,
        cmap=cmap,
        title=title,
        colorbar=True
    )

    if output_path:
        display.savefig(output_path, dpi=300)

    return display


def plot_surface_accuracy(
    accuracy_map,
    surf_mesh: str = "fsaverage",
    hemi: str = "both",
    threshold: float = 0.5,
    cmap: str = "hot",
    title: str = "Surface Accuracy",
    output_path: Optional[str] = None
):
    """
    Plot accuracy map on cortical surface.

    Args:
        accuracy_map: NIfTI image with accuracy values
        surf_mesh: Surface mesh ("fsaverage", "fsaverage5", etc.)
        hemi: Hemisphere to plot ("left", "right", "both")
        threshold: Minimum accuracy to display
        cmap: Colormap
        title: Plot title
        output_path: Path to save figure

    Returns:
        Nilearn display object
    """
    from nilearn import plotting, datasets, surface
    import nibabel as nib
    import matplotlib.pyplot as plt

    if isinstance(accuracy_map, str):
        accuracy_map = nib.load(accuracy_map)

    # Get surface mesh
    fsaverage = datasets.fetch_surf_fsaverage(mesh=surf_mesh)

    if hemi == "both":
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'},
                                 figsize=(12, 6))

        for ax, h in zip(axes, ["left", "right"]):
            surf_data = surface.vol_to_surf(
                accuracy_map,
                fsaverage[f"pial_{h}"]
            )

            plotting.plot_surf_stat_map(
                fsaverage[f"infl_{h}"],
                surf_data,
                hemi=h,
                threshold=threshold,
                cmap=cmap,
                axes=ax,
                colorbar=True
            )
            ax.set_title(f"{h.capitalize()} Hemisphere")

        fig.suptitle(title)

    else:
        surf_data = surface.vol_to_surf(
            accuracy_map,
            fsaverage[f"pial_{hemi}"]
        )

        fig = plotting.plot_surf_stat_map(
            fsaverage[f"infl_{hemi}"],
            surf_data,
            hemi=hemi,
            threshold=threshold,
            cmap=cmap,
            title=title,
            colorbar=True
        )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig


def plot_cluster_map(
    accuracy_map,
    threshold: float = 0.5,
    cluster_threshold: int = 10,
    output_path: Optional[str] = None
):
    """
    Plot significant clusters from searchlight analysis.

    Args:
        accuracy_map: NIfTI image with accuracy values
        threshold: Accuracy threshold
        cluster_threshold: Minimum cluster size
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    from nilearn import plotting
    from scipy import ndimage
    import nibabel as nib
    import matplotlib.pyplot as plt

    if isinstance(accuracy_map, str):
        accuracy_map = nib.load(accuracy_map)

    data = accuracy_map.get_fdata()

    # Find clusters
    thresholded = data > threshold
    labeled, n_clusters = ndimage.label(thresholded)

    # Keep only large clusters
    cluster_map = np.zeros_like(data)
    cluster_id = 1

    for i in range(1, n_clusters + 1):
        mask = labeled == i
        if np.sum(mask) >= cluster_threshold:
            cluster_map[mask] = cluster_id
            cluster_id += 1

    cluster_img = nib.Nifti1Image(cluster_map, accuracy_map.affine)

    # Plot
    display = plotting.plot_roi(
        cluster_img,
        title=f"Significant Clusters (>{threshold:.0%} accuracy)",
        display_mode="ortho"
    )

    if output_path:
        display.savefig(output_path, dpi=300)

    return display
