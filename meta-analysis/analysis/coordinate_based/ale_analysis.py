"""
Activation Likelihood Estimation (ALE) meta-analysis.

This module provides a wrapper around NiMARE's ALE implementation
for coordinate-based meta-analysis of neuroimaging studies.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import warnings


class ALEAnalysis:
    """
    Activation Likelihood Estimation meta-analysis.

    ALE is a coordinate-based meta-analysis method that models each
    reported coordinate as a 3D Gaussian probability distribution
    and computes the union of these distributions across studies.

    Example:
        >>> from core import MetaAnalysisDataset
        >>> dataset = MetaAnalysisDataset.load("my_studies.json")
        >>> ale = ALEAnalysis(dataset)
        >>> results = ale.run()
        >>> ale.save_nifti("ale_results.nii.gz")
    """

    def __init__(self, dataset: "MetaAnalysisDataset"):
        """
        Initialize ALE analysis.

        Args:
            dataset: MetaAnalysisDataset containing studies with coordinates
        """
        self.dataset = dataset
        self._nimare_ds = None
        self.results = None
        self.corrected_results = None
        self._check_nimare()

    def _check_nimare(self):
        """Check that NiMARE is installed."""
        try:
            import nimare
            self._nimare_version = nimare.__version__
        except ImportError:
            raise ImportError(
                "NiMARE is required for coordinate-based meta-analysis. "
                "Install with: pip install nimare"
            )

    def _get_nimare_dataset(self):
        """Get or create NiMARE dataset."""
        if self._nimare_ds is None:
            self._nimare_ds = self.dataset.to_nimare_dataset()
        return self._nimare_ds

    def run(
        self,
        kernel_fwhm: Optional[float] = None,
        null_method: str = "approximate",
        n_iters: int = 10000,
        correction_method: str = "fwe",
        alpha: float = 0.05,
        cluster_threshold: float = 0.001
    ) -> Dict[str, Any]:
        """
        Run ALE meta-analysis with multiple comparisons correction.

        Args:
            kernel_fwhm: FWHM for Gaussian kernel in mm.
                None = sample-size based (recommended)
            null_method: Method for null distribution.
                "approximate" (fast) or "montecarlo" (accurate)
            n_iters: Number of permutations for Monte Carlo methods
            correction_method: Multiple comparisons correction.
                "fwe" (family-wise error) or "fdr"
            alpha: Significance threshold
            cluster_threshold: Cluster-forming threshold

        Returns:
            Dictionary containing:
                - "ale_map": Unthresholded ALE values
                - "z_map": Z-score map
                - "corrected_map": Thresholded, corrected map
                - "cluster_table": Table of significant clusters
                - "summary": Analysis summary statistics
        """
        from nimare.meta.cbma.ale import ALE
        from nimare.correct import FWECorrector, FDRCorrector

        # Get NiMARE dataset
        nimare_ds = self._get_nimare_dataset()

        # Check we have enough studies
        n_studies = len(self.dataset.studies_with_coordinates)
        if n_studies < 2:
            raise ValueError(
                f"ALE requires at least 2 studies with coordinates. "
                f"Found {n_studies}."
            )

        print(f"Running ALE on {n_studies} studies with "
              f"{self.dataset.n_coordinates} coordinates...")

        # Initialize ALE estimator
        ale = ALE(
            kernel__fwhm=kernel_fwhm,
            null_method=null_method,
            n_iters=n_iters if null_method == "montecarlo" else 1
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results = ale.fit(nimare_ds)

        # Apply correction
        if correction_method == "fwe":
            corrector = FWECorrector(
                method="montecarlo",
                n_iters=n_iters,
                voxel_thresh=cluster_threshold
            )
        else:
            corrector = FDRCorrector(
                method="indep",
                alpha=alpha
            )

        self.corrected_results = corrector.transform(self.results)

        # Build results dictionary
        return self._build_results_dict(alpha)

    def _build_results_dict(self, alpha: float) -> Dict[str, Any]:
        """Build results dictionary from NiMARE results."""
        results_dict = {
            "ale_map": self.results.get_map("stat", return_type="image"),
            "z_map": self.results.get_map("z", return_type="image"),
            "summary": {
                "n_studies": len(self.dataset.studies_with_coordinates),
                "n_coordinates": self.dataset.n_coordinates,
                "alpha": alpha
            }
        }

        # Add corrected map if available
        try:
            corrected_map = self.corrected_results.get_map(
                "z_level-cluster_corr-FWE_method-montecarlo",
                return_type="image"
            )
            results_dict["corrected_map"] = corrected_map
        except Exception:
            # Try alternative map names
            for map_name in self.corrected_results.maps.keys():
                if "corr" in map_name.lower():
                    results_dict["corrected_map"] = self.corrected_results.get_map(
                        map_name, return_type="image"
                    )
                    break

        # Get cluster table
        try:
            results_dict["cluster_table"] = self._get_cluster_table()
        except Exception:
            results_dict["cluster_table"] = None

        return results_dict

    def _get_cluster_table(self):
        """Extract cluster table from results."""
        from nimare.reports import gen_table

        try:
            # Get thresholded map
            for map_name in self.corrected_results.maps.keys():
                if "corr" in map_name.lower():
                    return gen_table(
                        self.corrected_results,
                        map_name=map_name
                    )
        except Exception:
            return None

    def save_nifti(
        self,
        output_path: str,
        map_type: str = "corrected"
    ) -> str:
        """
        Save result map as NIfTI file.

        Args:
            output_path: Path for output file
            map_type: "corrected", "z", or "ale"

        Returns:
            Path to saved file
        """
        import nibabel as nib

        if self.results is None:
            raise ValueError("Run analysis first with .run()")

        if map_type == "corrected" and self.corrected_results is not None:
            # Find corrected map
            for map_name in self.corrected_results.maps.keys():
                if "corr" in map_name.lower():
                    img = self.corrected_results.get_map(map_name, return_type="image")
                    break
            else:
                img = self.results.get_map("z", return_type="image")
        elif map_type == "z":
            img = self.results.get_map("z", return_type="image")
        else:
            img = self.results.get_map("stat", return_type="image")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(output_path))

        return str(output_path)

    def plot_results(
        self,
        output_path: Optional[str] = None,
        threshold: float = 0.0,
        display_mode: str = "ortho",
        title: Optional[str] = None
    ):
        """
        Plot ALE results as brain map.

        Args:
            output_path: Path to save figure (None = display)
            threshold: Z-score threshold for display
            display_mode: "ortho", "x", "y", "z", or "glass"
            title: Plot title

        Returns:
            matplotlib Figure
        """
        from nilearn import plotting
        import matplotlib.pyplot as plt

        if self.corrected_results is None:
            raise ValueError("Run analysis first with .run()")

        # Get map to plot
        for map_name in self.corrected_results.maps.keys():
            if "corr" in map_name.lower():
                img = self.corrected_results.get_map(map_name, return_type="image")
                break
        else:
            img = self.results.get_map("z", return_type="image")

        # Create figure
        fig = plt.figure(figsize=(12, 4))

        if display_mode == "glass":
            display = plotting.plot_glass_brain(
                img,
                threshold=threshold,
                title=title or f"ALE Results (n={len(self.dataset.studies_with_coordinates)} studies)",
                figure=fig,
                colorbar=True
            )
        else:
            display = plotting.plot_stat_map(
                img,
                threshold=threshold,
                display_mode=display_mode,
                title=title or f"ALE Results",
                figure=fig,
                colorbar=True
            )

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def summary(self) -> str:
        """Generate text summary of analysis."""
        lines = [
            "ALE Meta-Analysis Summary",
            "=" * 40,
            f"Dataset: {self.dataset.name}",
            f"Studies: {len(self.dataset.studies_with_coordinates)}",
            f"Total coordinates: {self.dataset.n_coordinates}",
            f"Total sample size: {self.dataset.total_sample_size}",
        ]

        if self.results is not None:
            lines.append("")
            lines.append("Analysis completed successfully.")

        return "\n".join(lines)
