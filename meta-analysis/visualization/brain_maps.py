"""
Brain map visualization for coordinate-based meta-analysis.

Provides nilearn-based visualizations for ALE and other CBMA results.
"""

from typing import Optional, List, Tuple, Union
from pathlib import Path
import numpy as np


class BrainMapPlotter:
    """
    Create brain map visualizations from neuroimaging results.
    
    Wraps nilearn plotting functions with sensible defaults
    for meta-analysis visualization.
    
    Example:
        >>> plotter = BrainMapPlotter()
        >>> fig = plotter.glass_brain("ale_results.nii.gz", threshold=2.0)
        >>> fig.savefig("glass_brain.png", dpi=300)
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize plotter.
        
        Args:
            output_dir: Default directory for saving figures
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self._check_nilearn()
    
    def _check_nilearn(self):
        """Verify nilearn is installed."""
        try:
            import nilearn
            self._nilearn_version = nilearn.__version__
        except ImportError:
            raise ImportError(
                "nilearn is required for brain visualization. "
                "Install with: pip install nilearn"
            )
    
    def glass_brain(
        self,
        stat_map: Union[str, "Nifti1Image"],
        threshold: float = 0.0,
        title: Optional[str] = None,
        colorbar: bool = True,
        cmap: str = "cold_hot",
        vmax: Optional[float] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4),
        display_mode: str = "lyrz"
    ):
        """
        Create glass brain visualization.
        
        Args:
            stat_map: Path to NIfTI file or nibabel image
            threshold: Display threshold
            title: Plot title
            colorbar: Show colorbar
            cmap: Colormap name
            vmax: Maximum value for colormap
            output_path: Path to save figure
            figsize: Figure size (width, height)
            display_mode: Views to show (l=left, r=right, y=coronal, z=axial)
            
        Returns:
            matplotlib Figure
        """
        from nilearn import plotting
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=figsize)
        
        display = plotting.plot_glass_brain(
            stat_map,
            threshold=threshold,
            title=title,
            colorbar=colorbar,
            cmap=cmap,
            vmax=vmax,
            figure=fig,
            display_mode=display_mode
        )
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def stat_map(
        self,
        stat_map: Union[str, "Nifti1Image"],
        threshold: float = 0.0,
        title: Optional[str] = None,
        display_mode: str = "ortho",
        cut_coords: Optional[List[float]] = None,
        colorbar: bool = True,
        cmap: str = "cold_hot",
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Create statistical map visualization with anatomical background.
        
        Args:
            stat_map: Path to NIfTI file or nibabel image
            threshold: Display threshold
            title: Plot title
            display_mode: "ortho", "x", "y", "z", or "mosaic"
            cut_coords: Coordinates for slices
            colorbar: Show colorbar
            cmap: Colormap name
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        from nilearn import plotting
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=figsize)
        
        display = plotting.plot_stat_map(
            stat_map,
            threshold=threshold,
            title=title,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            cmap=cmap,
            figure=fig
        )
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def plot_coordinates(
        self,
        coordinates: List[Tuple[float, float, float]],
        title: Optional[str] = None,
        node_size: int = 50,
        node_color: str = "red",
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Plot coordinate locations on glass brain.
        
        Args:
            coordinates: List of (x, y, z) tuples
            title: Plot title
            node_size: Size of coordinate markers
            node_color: Color of markers
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        from nilearn import plotting
        import matplotlib.pyplot as plt
        
        coords = np.array(coordinates)
        
        fig = plt.figure(figsize=figsize)
        
        # Create adjacency matrix (no connections)
        adjacency = np.zeros((len(coords), len(coords)))
        
        display = plotting.plot_connectome(
            adjacency,
            coords,
            node_size=node_size,
            node_color=node_color,
            title=title,
            figure=fig,
            display_mode="lyrz"
        )
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def mosaic(
        self,
        stat_map: Union[str, "Nifti1Image"],
        threshold: float = 0.0,
        n_cuts: int = 7,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create mosaic of axial slices.
        
        Args:
            stat_map: Path to NIfTI file or nibabel image
            threshold: Display threshold
            n_cuts: Number of slices to show
            title: Plot title
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        from nilearn import plotting
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=figsize)
        
        display = plotting.plot_stat_map(
            stat_map,
            threshold=threshold,
            title=title,
            display_mode="z",
            cut_coords=n_cuts,
            figure=fig,
            colorbar=True
        )
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def compare_maps(
        self,
        map1: Union[str, "Nifti1Image"],
        map2: Union[str, "Nifti1Image"],
        labels: Tuple[str, str] = ("Map 1", "Map 2"),
        threshold: float = 0.0,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Compare two statistical maps side by side.
        
        Args:
            map1: First NIfTI map
            map2: Second NIfTI map
            labels: Labels for each map
            threshold: Display threshold
            output_path: Path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        from nilearn import plotting
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        plotting.plot_glass_brain(
            map1,
            threshold=threshold,
            title=labels[0],
            axes=axes[0],
            colorbar=True
        )
        
        plotting.plot_glass_brain(
            map2,
            threshold=threshold,
            title=labels[1],
            axes=axes[1],
            colorbar=True
        )
        
        plt.tight_layout()
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def _save_figure(self, fig, path: str):
        """Save figure to file."""
        output_path = Path(path)
        if self.output_dir:
            output_path = self.output_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
