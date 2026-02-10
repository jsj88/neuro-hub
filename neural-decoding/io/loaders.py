"""
Data loaders for pre-processed neuroimaging data.

Provides unified loading for fMRI (NIfTI), EEG (MNE), behavioral (CSV),
and multimodal fusion datasets.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '..')
from core.dataset import DecodingDataset


class BaseLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, *args, **kwargs) -> DecodingDataset:
        """Load data and return DecodingDataset."""
        pass


class FMRILoader(BaseLoader):
    """
    Load pre-processed fMRI data.
    
    Expects:
    - NIfTI file(s) with 4D fMRI data or list of 3D volumes
    - Brain mask (NIfTI)
    - Events file (CSV) with trial labels
    
    Example:
        >>> loader = FMRILoader()
        >>> dataset = loader.load(
        ...     data_path="sub-01_task_bold.nii.gz",
        ...     mask_path="brain_mask.nii.gz",
        ...     events_path="events.csv",
        ...     label_column="condition"
        ... )
    """
    
    def load(
        self,
        data_path: Union[str, List[str]],
        mask_path: str,
        events_path: str,
        label_column: str,
        run_column: Optional[str] = None,
        subject_column: Optional[str] = None,
        standardize: bool = True,
        **kwargs
    ) -> DecodingDataset:
        """
        Load fMRI data for decoding.
        
        Args:
            data_path: Path to NIfTI file or list of paths
            mask_path: Path to brain mask NIfTI
            events_path: Path to events CSV
            label_column: Column name containing class labels
            run_column: Column name for run grouping (CV)
            subject_column: Column for subject grouping
            standardize: Z-score normalize voxels
            
        Returns:
            DecodingDataset ready for classification
        """
        import nibabel as nib
        from nilearn.maskers import NiftiMasker
        
        # Load events
        events = pd.read_csv(events_path)
        labels = events[label_column].values
        
        # Get groups for CV
        groups = None
        if run_column and run_column in events.columns:
            groups = events[run_column].values
        elif subject_column and subject_column in events.columns:
            groups = events[subject_column].values
        
        # Create masker
        masker = NiftiMasker(
            mask_img=mask_path,
            standardize=standardize,
            detrend=False  # Assume already detrended
        )
        
        # Load and mask fMRI data
        if isinstance(data_path, list):
            # Multiple files (one per trial/volume)
            X = masker.fit_transform(data_path)
        else:
            # Single 4D file
            X = masker.fit_transform(data_path)
        
        # Validate alignment
        if X.shape[0] != len(labels):
            raise ValueError(
                f"Mismatch: {X.shape[0]} volumes but {len(labels)} events. "
                "Ensure events match number of volumes."
            )
        
        # Create feature names from voxel coordinates
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        voxel_coords = np.where(mask_data > 0)
        feature_names = [
            f"voxel_{x}_{y}_{z}" 
            for x, y, z in zip(*voxel_coords)
        ]
        
        # Unique class names
        class_names = list(events[label_column].unique())
        
        # Metadata
        metadata = {
            "data_path": str(data_path),
            "mask_path": str(mask_path),
            "events_path": str(events_path),
            "n_voxels": X.shape[1],
            "standardized": standardize,
            **kwargs
        }
        
        return DecodingDataset(
            X=X,
            y=labels,
            groups=groups,
            feature_names=feature_names,
            class_names=class_names,
            metadata=metadata,
            modality="fmri"
        )
    
    def load_roi(
        self,
        data_path: str,
        atlas_path: str,
        events_path: str,
        label_column: str,
        run_column: Optional[str] = None,
        aggregation: str = "mean"
    ) -> DecodingDataset:
        """
        Load fMRI data with ROI-based features.
        
        Args:
            data_path: Path to 4D NIfTI
            atlas_path: Path to atlas NIfTI (integer labels)
            events_path: Path to events CSV
            label_column: Column for class labels
            run_column: Column for run grouping
            aggregation: How to combine voxels ("mean", "median", "std")
            
        Returns:
            DecodingDataset with ROI features
        """
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker
        
        events = pd.read_csv(events_path)
        labels = events[label_column].values
        
        groups = None
        if run_column and run_column in events.columns:
            groups = events[run_column].values
        
        # Extract ROI signals
        masker = NiftiLabelsMasker(
            labels_img=atlas_path,
            standardize=True,
            strategy=aggregation
        )
        X = masker.fit_transform(data_path)
        
        # ROI labels as feature names
        atlas_img = nib.load(atlas_path)
        n_rois = len(np.unique(atlas_img.get_fdata())) - 1  # Exclude 0
        feature_names = [f"ROI_{i}" for i in range(1, n_rois + 1)]
        
        class_names = list(events[label_column].unique())
        
        metadata = {
            "data_path": str(data_path),
            "atlas_path": str(atlas_path),
            "aggregation": aggregation,
            "n_rois": n_rois
        }
        
        return DecodingDataset(
            X=X,
            y=labels,
            groups=groups,
            feature_names=feature_names,
            class_names=class_names,
            metadata=metadata,
            modality="fmri"
        )


class EEGLoader(BaseLoader):
    """
    Load pre-processed EEG/MEG epochs.
    
    Expects MNE Epochs object saved as .fif file.
    
    Example:
        >>> loader = EEGLoader()
        >>> dataset = loader.load(
        ...     epochs_path="sub-01-epo.fif",
        ...     label_column="stimulus",
        ...     time_window=(0.1, 0.5)
        ... )
    """
    
    def load(
        self,
        epochs_path: str,
        label_column: Optional[str] = None,
        time_window: Optional[Tuple[float, float]] = None,
        channels: Optional[List[str]] = None,
        flatten: bool = True,
        **kwargs
    ) -> DecodingDataset:
        """
        Load EEG epochs for decoding.
        
        Args:
            epochs_path: Path to MNE epochs file (-epo.fif)
            label_column: Event ID to use as labels (None = all events)
            time_window: Time range (tmin, tmax) in seconds
            channels: List of channels to include (None = all)
            flatten: Flatten channels x time to 1D feature vector
            
        Returns:
            DecodingDataset ready for classification
        """
        import mne
        
        # Load epochs
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        
        # Select time window
        if time_window is not None:
            epochs = epochs.crop(tmin=time_window[0], tmax=time_window[1])
        
        # Select channels
        if channels is not None:
            epochs = epochs.pick_channels(channels)
        
        # Get data and labels
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        
        # Get labels from events
        if label_column is not None:
            event_ids = epochs.event_id
            if label_column in event_ids:
                # Filter to specific event type
                epochs = epochs[label_column]
                X = epochs.get_data()
        
        # Get labels
        labels = epochs.events[:, 2]  # Event IDs
        
        # Map event IDs to names
        event_id_inv = {v: k for k, v in epochs.event_id.items()}
        class_names = list(epochs.event_id.keys())
        
        # Flatten if requested
        if flatten:
            n_epochs, n_channels, n_times = X.shape
            X = X.reshape(n_epochs, n_channels * n_times)
            
            # Feature names: channel_time
            feature_names = [
                f"{ch}_{t:.3f}s"
                for ch in epochs.ch_names
                for t in epochs.times
            ]
        else:
            feature_names = None
        
        metadata = {
            "epochs_path": str(epochs_path),
            "n_channels": len(epochs.ch_names),
            "n_times": len(epochs.times),
            "sfreq": epochs.info["sfreq"],
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
            "time_window": time_window,
            **kwargs
        }
        
        return DecodingDataset(
            X=X,
            y=labels,
            groups=None,
            feature_names=feature_names,
            class_names=class_names,
            metadata=metadata,
            modality="eeg"
        )
    
    def load_time_resolved(
        self,
        epochs_path: str,
        time_points: Optional[List[float]] = None,
        window_size: float = 0.05
    ) -> List[DecodingDataset]:
        """
        Load epochs for time-resolved decoding.
        
        Args:
            epochs_path: Path to epochs file
            time_points: Specific time points to decode
            window_size: Width of each time window (seconds)
            
        Returns:
            List of DecodingDatasets, one per time point
        """
        import mne
        
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        
        if time_points is None:
            time_points = epochs.times[::10]  # Every 10th sample
        
        datasets = []
        half_win = window_size / 2
        
        for t in time_points:
            tmin = max(epochs.tmin, t - half_win)
            tmax = min(epochs.tmax, t + half_win)
            
            epoch_crop = epochs.copy().crop(tmin=tmin, tmax=tmax)
            X = epoch_crop.get_data().mean(axis=2)  # Average over time
            
            labels = epochs.events[:, 2]
            class_names = list(epochs.event_id.keys())
            
            ds = DecodingDataset(
                X=X,
                y=labels,
                class_names=class_names,
                metadata={"time_point": t, "window_size": window_size},
                modality="eeg"
            )
            datasets.append(ds)
        
        return datasets


class BehaviorLoader(BaseLoader):
    """
    Load behavioral data for classification.
    
    Example:
        >>> loader = BehaviorLoader()
        >>> dataset = loader.load(
        ...     csv_path="behavior.csv",
        ...     feature_columns=["reaction_time", "accuracy", "confidence"],
        ...     label_column="condition"
        ... )
    """
    
    def load(
        self,
        csv_path: str,
        feature_columns: List[str],
        label_column: str,
        group_column: Optional[str] = None,
        standardize: bool = True,
        **kwargs
    ) -> DecodingDataset:
        """
        Load behavioral data.
        
        Args:
            csv_path: Path to CSV file
            feature_columns: Columns to use as features
            label_column: Column for class labels
            group_column: Column for grouping (subject, session)
            standardize: Z-score features
            
        Returns:
            DecodingDataset
        """
        df = pd.read_csv(csv_path)
        
        X = df[feature_columns].values
        labels = df[label_column].values
        
        groups = None
        if group_column and group_column in df.columns:
            groups = df[group_column].values
        
        if standardize:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
        
        class_names = list(df[label_column].unique())
        
        metadata = {
            "csv_path": str(csv_path),
            "n_features": len(feature_columns),
            "standardized": standardize,
            **kwargs
        }
        
        return DecodingDataset(
            X=X,
            y=labels,
            groups=groups,
            feature_names=feature_columns,
            class_names=class_names,
            metadata=metadata,
            modality="behavior"
        )


class MultimodalLoader(BaseLoader):
    """
    Load and fuse multiple data modalities.
    
    Supports early fusion (concatenation) and provides
    hooks for late fusion strategies.
    
    Example:
        >>> loader = MultimodalLoader()
        >>> dataset = loader.load_early_fusion(
        ...     modalities=[fmri_dataset, eeg_dataset, behavior_dataset],
        ...     normalize=True
        ... )
    """
    
    def load(
        self,
        datasets: List[DecodingDataset],
        fusion: str = "early"
    ) -> DecodingDataset:
        """
        Fuse multiple modality datasets.
        
        Args:
            datasets: List of DecodingDatasets to fuse
            fusion: Fusion strategy ("early" = concatenation)
            
        Returns:
            Fused DecodingDataset
        """
        if fusion == "early":
            return self.early_fusion(datasets)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion}")
    
    def early_fusion(
        self,
        datasets: List[DecodingDataset],
        normalize: bool = True
    ) -> DecodingDataset:
        """
        Concatenate features from multiple modalities.
        
        Args:
            datasets: List of datasets with same samples/labels
            normalize: Normalize each modality before fusion
            
        Returns:
            Fused dataset
        """
        # Validate alignment
        n_samples = datasets[0].n_samples
        labels = datasets[0].y
        
        for i, ds in enumerate(datasets[1:], 1):
            if ds.n_samples != n_samples:
                raise ValueError(
                    f"Modality {i} has {ds.n_samples} samples, "
                    f"expected {n_samples}"
                )
            if not np.array_equal(ds.y, labels):
                raise ValueError(f"Labels don't match for modality {i}")
        
        # Normalize if requested
        if normalize:
            from sklearn.preprocessing import StandardScaler
            Xs = [StandardScaler().fit_transform(ds.X) for ds in datasets]
        else:
            Xs = [ds.X for ds in datasets]
        
        # Concatenate features
        X = np.hstack(Xs)
        
        # Combine feature names with modality prefix
        feature_names = []
        for ds in datasets:
            prefix = ds.modality
            for name in (ds.feature_names or [f"f{i}" for i in range(ds.n_features)]):
                feature_names.append(f"{prefix}_{name}")
        
        # Use groups from first dataset
        groups = datasets[0].groups
        class_names = datasets[0].class_names
        
        # Combine metadata
        modalities = [ds.modality for ds in datasets]
        metadata = {
            "fusion": "early",
            "modalities": modalities,
            "normalized": normalize,
            "n_features_per_modality": [ds.n_features for ds in datasets]
        }
        
        return DecodingDataset(
            X=X,
            y=labels,
            groups=groups,
            feature_names=feature_names,
            class_names=class_names,
            metadata=metadata,
            modality="multimodal"
        )
    
    @staticmethod
    def align_datasets(
        datasets: List[DecodingDataset],
        on: str = "trial_id"
    ) -> List[DecodingDataset]:
        """
        Align datasets by trial/sample ID.
        
        Args:
            datasets: Datasets to align
            on: Metadata key to align on
            
        Returns:
            Aligned datasets with matching samples
        """
        # Get common indices
        # This is a simplified version - full implementation would
        # handle trial ID matching across modalities
        raise NotImplementedError(
            "Trial alignment requires trial_id in metadata. "
            "Ensure datasets have matching samples before fusion."
        )
