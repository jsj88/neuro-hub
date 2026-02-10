"""
Feature extraction utilities for neural decoding.

Provides tools to extract and transform features from neuroimaging data:
- ROI extraction from atlases
- Voxel-wise extraction with masks
- Time window extraction for EEG/MEG
- Trial averaging for noise reduction
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset


class BaseExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def fit(self, dataset: DecodingDataset) -> "BaseExtractor":
        """Fit extractor to data."""
        pass

    @abstractmethod
    def transform(self, dataset: DecodingDataset) -> DecodingDataset:
        """Transform data to extract features."""
        pass

    def fit_transform(self, dataset: DecodingDataset) -> DecodingDataset:
        """Fit and transform in one step."""
        return self.fit(dataset).transform(dataset)


class ROIExtractor(BaseExtractor):
    """
    Extract ROI-averaged features using an atlas.

    Reduces high-dimensional voxel data to region-of-interest means,
    providing more interpretable features.

    Example:
        >>> extractor = ROIExtractor(
        ...     atlas_path="harvard_oxford.nii.gz",
        ...     aggregation="mean"
        ... )
        >>> roi_dataset = extractor.fit_transform(voxel_dataset)
    """

    def __init__(
        self,
        atlas_path: str,
        aggregation: str = "mean",
        standardize: bool = True
    ):
        """
        Initialize ROI extractor.

        Args:
            atlas_path: Path to atlas NIfTI (integer labels)
            aggregation: How to combine voxels ("mean", "median", "std")
            standardize: Z-score ROI values
        """
        self.atlas_path = atlas_path
        self.aggregation = aggregation
        self.standardize = standardize

        self.n_rois_ = None
        self.roi_labels_ = None
        self.masker_ = None

    def fit(self, dataset: DecodingDataset) -> "ROIExtractor":
        """Fit extractor (load atlas info)."""
        import nibabel as nib

        atlas_img = nib.load(self.atlas_path)
        atlas_data = atlas_img.get_fdata()

        # Get unique ROI labels (excluding 0 = background)
        self.roi_labels_ = np.unique(atlas_data[atlas_data > 0]).astype(int)
        self.n_rois_ = len(self.roi_labels_)

        return self

    def transform(self, dataset: DecodingDataset) -> DecodingDataset:
        """
        Transform voxel data to ROI features.

        Note: This requires the original NIfTI data path in metadata.
        For already-masked data, use a different approach.
        """
        from nilearn.maskers import NiftiLabelsMasker
        import pandas as pd

        if self.n_rois_ is None:
            raise ValueError("Must call fit() first")

        # Need original data path
        if "data_path" not in dataset.metadata:
            raise ValueError(
                "ROIExtractor requires original NIfTI path in metadata"
            )

        # Create masker
        masker = NiftiLabelsMasker(
            labels_img=self.atlas_path,
            standardize=self.standardize,
            strategy=self.aggregation
        )

        # Extract ROI signals
        X = masker.fit_transform(dataset.metadata["data_path"])

        # Feature names
        feature_names = [f"ROI_{i}" for i in self.roi_labels_]

        # Create new dataset
        return DecodingDataset(
            X=X,
            y=dataset.y,
            groups=dataset.groups,
            feature_names=feature_names,
            class_names=dataset.class_names,
            metadata={
                **dataset.metadata,
                "extraction": "roi",
                "atlas_path": self.atlas_path,
                "aggregation": self.aggregation,
                "n_rois": self.n_rois_
            },
            modality=dataset.modality
        )

    def __repr__(self) -> str:
        return f"ROIExtractor(n_rois={self.n_rois_}, aggregation={self.aggregation})"


class VoxelExtractor(BaseExtractor):
    """
    Extract voxel-wise features using a brain mask.

    Converts 4D NIfTI data to 2D feature matrix using a brain mask.

    Example:
        >>> extractor = VoxelExtractor(mask_path="brain_mask.nii.gz")
        >>> dataset = extractor.fit_transform(nifti_data)
    """

    def __init__(
        self,
        mask_path: str,
        standardize: bool = True,
        detrend: bool = False
    ):
        """
        Initialize voxel extractor.

        Args:
            mask_path: Path to brain mask NIfTI
            standardize: Z-score normalize voxels
            detrend: Remove linear trends
        """
        self.mask_path = mask_path
        self.standardize = standardize
        self.detrend = detrend

        self.n_voxels_ = None
        self.masker_ = None

    def fit(self, dataset: DecodingDataset = None) -> "VoxelExtractor":
        """Fit extractor (initialize masker)."""
        from nilearn.maskers import NiftiMasker

        self.masker_ = NiftiMasker(
            mask_img=self.mask_path,
            standardize=self.standardize,
            detrend=self.detrend
        )

        return self

    def transform(self, dataset: DecodingDataset) -> DecodingDataset:
        """Transform NIfTI to voxel features."""
        import nibabel as nib

        if self.masker_ is None:
            raise ValueError("Must call fit() first")

        # Need original data path
        if "data_path" not in dataset.metadata:
            raise ValueError(
                "VoxelExtractor requires original NIfTI path in metadata"
            )

        # Extract voxel data
        X = self.masker_.fit_transform(dataset.metadata["data_path"])
        self.n_voxels_ = X.shape[1]

        # Generate feature names from voxel coordinates
        mask_img = nib.load(self.mask_path)
        mask_data = mask_img.get_fdata()
        voxel_coords = np.where(mask_data > 0)
        feature_names = [
            f"voxel_{x}_{y}_{z}"
            for x, y, z in zip(*voxel_coords)
        ]

        return DecodingDataset(
            X=X,
            y=dataset.y,
            groups=dataset.groups,
            feature_names=feature_names,
            class_names=dataset.class_names,
            metadata={
                **dataset.metadata,
                "extraction": "voxel",
                "mask_path": self.mask_path,
                "n_voxels": self.n_voxels_
            },
            modality=dataset.modality
        )

    def __repr__(self) -> str:
        return f"VoxelExtractor(n_voxels={self.n_voxels_})"


class TimeWindowExtractor(BaseExtractor):
    """
    Extract features from specific time windows in EEG/MEG data.

    Useful for focusing on task-relevant time periods or creating
    multiple feature sets for different time windows.

    Example:
        >>> extractor = TimeWindowExtractor(
        ...     windows=[(0.1, 0.3), (0.3, 0.5)],
        ...     aggregation="mean"
        ... )
        >>> windowed = extractor.fit_transform(eeg_dataset)
    """

    def __init__(
        self,
        windows: List[Tuple[float, float]],
        aggregation: str = "mean",
        flatten: bool = True
    ):
        """
        Initialize time window extractor.

        Args:
            windows: List of (start, end) time windows in seconds
            aggregation: How to combine time points ("mean", "median", "max")
            flatten: Flatten channels x windows to 1D
        """
        self.windows = windows
        self.aggregation = aggregation
        self.flatten = flatten

        self.times_ = None
        self.n_channels_ = None

    def fit(self, dataset: DecodingDataset) -> "TimeWindowExtractor":
        """Fit extractor (get time info from metadata)."""
        if "tmin" in dataset.metadata and "tmax" in dataset.metadata:
            tmin = dataset.metadata["tmin"]
            tmax = dataset.metadata["tmax"]
            sfreq = dataset.metadata.get("sfreq", 100)
            n_times = dataset.metadata.get("n_times", int((tmax - tmin) * sfreq) + 1)
            self.times_ = np.linspace(tmin, tmax, n_times)
        else:
            raise ValueError("Dataset must have tmin, tmax in metadata")

        self.n_channels_ = dataset.metadata.get("n_channels", 1)

        return self

    def transform(self, dataset: DecodingDataset) -> DecodingDataset:
        """Extract time window features."""
        if self.times_ is None:
            raise ValueError("Must call fit() first")

        X = dataset.X
        n_samples = X.shape[0]
        n_times = len(self.times_)

        # Reshape if flattened
        if X.ndim == 2:
            X = X.reshape(n_samples, self.n_channels_, n_times)

        # Extract windows
        window_features = []
        feature_names = []

        for i, (t_start, t_end) in enumerate(self.windows):
            # Get time indices
            time_mask = (self.times_ >= t_start) & (self.times_ <= t_end)

            if not np.any(time_mask):
                continue

            # Extract and aggregate
            window_data = X[:, :, time_mask]

            if self.aggregation == "mean":
                aggregated = window_data.mean(axis=2)
            elif self.aggregation == "median":
                aggregated = np.median(window_data, axis=2)
            elif self.aggregation == "max":
                aggregated = window_data.max(axis=2)
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

            window_features.append(aggregated)

            # Feature names
            for ch in range(self.n_channels_):
                feature_names.append(f"ch{ch}_win{i}_{t_start:.2f}-{t_end:.2f}s")

        # Stack windows
        X_new = np.hstack(window_features) if self.flatten else np.stack(window_features, axis=2)

        return DecodingDataset(
            X=X_new,
            y=dataset.y,
            groups=dataset.groups,
            feature_names=feature_names if self.flatten else None,
            class_names=dataset.class_names,
            metadata={
                **dataset.metadata,
                "extraction": "time_window",
                "windows": self.windows,
                "aggregation": self.aggregation
            },
            modality=dataset.modality
        )

    def __repr__(self) -> str:
        return f"TimeWindowExtractor(n_windows={len(self.windows)})"


class TrialAverager(BaseExtractor):
    """
    Average trials within conditions to increase SNR.

    Useful when you have many trials per condition and want to
    reduce noise by averaging within-condition trials.

    Example:
        >>> averager = TrialAverager(n_per_average=5)
        >>> averaged = averager.fit_transform(dataset)
    """

    def __init__(
        self,
        n_per_average: int = 5,
        random_state: Optional[int] = 42
    ):
        """
        Initialize trial averager.

        Args:
            n_per_average: Number of trials to average together
            random_state: Random seed for trial selection
        """
        self.n_per_average = n_per_average
        self.random_state = random_state

    def fit(self, dataset: DecodingDataset) -> "TrialAverager":
        """Fit (no-op for averager)."""
        return self

    def transform(self, dataset: DecodingDataset) -> DecodingDataset:
        """Average trials within each class."""
        np.random.seed(self.random_state)

        X = dataset.X
        y = dataset.y
        groups = dataset.groups

        X_new = []
        y_new = []
        groups_new = [] if groups is not None else None

        for label in np.unique(y):
            # Get trials for this class
            class_mask = y == label
            class_X = X[class_mask]
            n_trials = len(class_X)

            if groups is not None:
                class_groups = groups[class_mask]

            # Create averaged pseudo-trials
            n_averages = n_trials // self.n_per_average

            indices = np.random.permutation(n_trials)

            for i in range(n_averages):
                start = i * self.n_per_average
                end = start + self.n_per_average
                avg_indices = indices[start:end]

                # Average features
                X_new.append(class_X[avg_indices].mean(axis=0))
                y_new.append(label)

                # Keep first group label
                if groups is not None:
                    groups_new.append(class_groups[avg_indices[0]])

        X_new = np.array(X_new)
        y_new = np.array(y_new)
        groups_new = np.array(groups_new) if groups_new else None

        return DecodingDataset(
            X=X_new,
            y=y_new,
            groups=groups_new,
            feature_names=dataset.feature_names,
            class_names=dataset.class_names,
            metadata={
                **dataset.metadata,
                "trial_averaging": True,
                "n_per_average": self.n_per_average,
                "original_n_samples": len(y)
            },
            modality=dataset.modality
        )

    def __repr__(self) -> str:
        return f"TrialAverager(n_per_average={self.n_per_average})"
