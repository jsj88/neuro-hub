"""
Searchlight analysis for whole-brain decoding.

Performs classification in local spherical neighborhoods across the brain
to identify regions that carry information about task conditions.
"""

from typing import Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset
from .base import BaseDecoder
from .classifiers import SVMDecoder


class SearchlightDecoder:
    """
    Whole-brain searchlight analysis.

    Performs classification in local spherical neighborhoods across the brain
    to create accuracy maps showing where information is encoded.

    Example:
        >>> searchlight = SearchlightDecoder(
        ...     mask_path="brain_mask.nii.gz",
        ...     radius=5.0,
        ...     n_jobs=-1
        ... )
        >>> searchlight.fit(dataset)
        >>> searchlight.save_nifti("accuracy_map.nii.gz")
    """

    def __init__(
        self,
        mask_path: str,
        radius: float = 5.0,
        decoder: Optional[BaseDecoder] = None,
        cv=None,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = 42
    ):
        """
        Initialize searchlight decoder.

        Args:
            mask_path: Path to brain mask NIfTI
            radius: Searchlight sphere radius in mm
            decoder: Decoder to use (default: linear SVM)
            cv: Cross-validation splitter
            n_jobs: Parallel jobs (-1 = all CPUs)
            verbose: Verbosity level
            random_state: Random seed
        """
        self.mask_path = mask_path
        self.radius = radius
        self.decoder = decoder or SVMDecoder(kernel="linear")
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.accuracy_map_ = None
        self.mask_img_ = None

    def fit(self, dataset: DecodingDataset) -> "SearchlightDecoder":
        """
        Run searchlight analysis.

        Args:
            dataset: DecodingDataset with fMRI data

        Returns:
            self: Fitted instance with accuracy map
        """
        from nilearn.decoding import SearchLight
        from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
        import nibabel as nib

        X, y = dataset.X, dataset.y
        groups = dataset.groups

        # Load mask
        self.mask_img_ = nib.load(self.mask_path)

        # Set up CV
        if self.cv is None:
            if groups is not None:
                cv = LeaveOneGroupOut()
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True,
                                    random_state=self.random_state)
        else:
            cv = self.cv

        # Get sklearn estimator from decoder
        estimator = self.decoder._get_sklearn_estimator()

        # Create nilearn SearchLight
        searchlight = SearchLight(
            mask_img=self.mask_img_,
            radius=self.radius,
            estimator=estimator,
            cv=cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        # We need to reconstruct the 4D data for nilearn
        # This requires knowing the original data shape
        if "data_path" in dataset.metadata:
            # Load original 4D data
            data_img = nib.load(dataset.metadata["data_path"])
            searchlight.fit(data_img, y, groups=groups)
        else:
            # Data was already masked - need to create 4D image
            mask_data = self.mask_img_.get_fdata()
            voxel_indices = np.where(mask_data > 0)

            # Reconstruct 4D
            n_samples = X.shape[0]
            shape_3d = mask_data.shape
            data_4d = np.zeros((shape_3d[0], shape_3d[1], shape_3d[2], n_samples))

            for i in range(n_samples):
                vol = np.zeros(shape_3d)
                vol[voxel_indices] = X[i]
                data_4d[:, :, :, i] = vol

            data_img = nib.Nifti1Image(data_4d, self.mask_img_.affine)
            searchlight.fit(data_img, y, groups=groups)

        self.accuracy_map_ = searchlight.scores_

        return self

    def get_accuracy_map(self):
        """
        Get accuracy map as NIfTI image.

        Returns:
            nibabel Nifti1Image with accuracy at each voxel
        """
        import nibabel as nib

        if self.accuracy_map_ is None:
            raise ValueError("Must call fit() first")

        return nib.Nifti1Image(self.accuracy_map_, self.mask_img_.affine)

    def save_nifti(self, path: str):
        """
        Save accuracy map to NIfTI file.

        Args:
            path: Output path for NIfTI file
        """
        import nibabel as nib

        accuracy_img = self.get_accuracy_map()
        nib.save(accuracy_img, path)

        if self.verbose:
            print(f"Saved searchlight map to {path}")

    def plot(
        self,
        threshold: float = 0.5,
        display_mode: str = "ortho",
        cmap: str = "hot",
        title: str = "Searchlight Accuracy",
        output_path: Optional[str] = None
    ):
        """
        Plot searchlight results on brain.

        Args:
            threshold: Minimum accuracy to display
            display_mode: Nilearn display mode ("ortho", "x", "y", "z", "glass")
            cmap: Colormap
            title: Plot title
            output_path: Path to save figure

        Returns:
            Nilearn display object
        """
        from nilearn import plotting

        if self.accuracy_map_ is None:
            raise ValueError("Must call fit() first")

        accuracy_img = self.get_accuracy_map()

        display = plotting.plot_stat_map(
            accuracy_img,
            threshold=threshold,
            display_mode=display_mode,
            cmap=cmap,
            title=title
        )

        if output_path:
            display.savefig(output_path, dpi=300)

        return display

    def get_significant_clusters(
        self,
        threshold: float = 0.5,
        cluster_threshold: int = 10
    ) -> list:
        """
        Get significant clusters above threshold.

        Args:
            threshold: Accuracy threshold
            cluster_threshold: Minimum cluster size in voxels

        Returns:
            List of cluster info dicts
        """
        from scipy import ndimage

        if self.accuracy_map_ is None:
            raise ValueError("Must call fit() first")

        # Threshold the map
        thresholded = self.accuracy_map_ > threshold

        # Find connected components
        labeled, n_clusters = ndimage.label(thresholded)

        clusters = []
        for i in range(1, n_clusters + 1):
            cluster_mask = labeled == i
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= cluster_threshold:
                # Get cluster statistics
                cluster_values = self.accuracy_map_[cluster_mask]

                # Get peak location
                peak_idx = np.unravel_index(
                    np.argmax(self.accuracy_map_ * cluster_mask),
                    self.accuracy_map_.shape
                )

                clusters.append({
                    "cluster_id": i,
                    "size_voxels": int(cluster_size),
                    "peak_accuracy": float(np.max(cluster_values)),
                    "mean_accuracy": float(np.mean(cluster_values)),
                    "peak_voxel": peak_idx
                })

        # Sort by peak accuracy
        clusters.sort(key=lambda x: x["peak_accuracy"], reverse=True)

        return clusters

    def __repr__(self) -> str:
        fitted = "fitted" if self.accuracy_map_ is not None else "not fitted"
        return f"SearchlightDecoder(radius={self.radius}mm, {fitted})"
