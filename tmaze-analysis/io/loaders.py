"""
Data loaders for T-maze EEG-fMRI analysis.

Supports:
- MNE epochs (.fif files) for EEG
- AFNI/NIfTI GLM betas for fMRI
- HCP atlas parcellation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings

# Optional imports with fallbacks
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    warnings.warn("MNE not installed. EEG loading unavailable.")

try:
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn import datasets
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    warnings.warn("Nilearn/nibabel not installed. fMRI loading unavailable.")

from ..core.containers import (
    TMazeEEGData,
    TMAzefMRIData,
    TMazeSubject,
    TMazeConditions
)


def load_hcp_atlas(n_rois: int = 426) -> Tuple[str, List[str]]:
    """
    Load HCP atlas for ROI extraction.

    Parameters
    ----------
    n_rois : int
        Number of ROIs (426 for HCP MMP parcellation)

    Returns
    -------
    atlas_path : str
        Path to atlas NIfTI file
    roi_names : List[str]
        List of ROI names
    """
    if not HAS_NILEARN:
        raise ImportError("Nilearn required for atlas loading")

    # Fetch Schaefer atlas (similar parcellation to HCP)
    # You can replace with your local HCP atlas path
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=None,
        verbose=0
    )

    return atlas['maps'], list(atlas['labels'])


class TMazeEEGLoader:
    """
    Load T-maze EEG data from MNE epochs files.

    Parameters
    ----------
    condition_mapping : Dict[str, int]
        Mapping from event names to binary labels
    channels : List[str], optional
        Channels to select (default: all)
    tmin : float, optional
        Start time in seconds (default: -0.2)
    tmax : float, optional
        End time in seconds (default: 0.8)
    baseline : Tuple[float, float], optional
        Baseline correction window (default: (-0.2, 0))
    """

    def __init__(
        self,
        condition_mapping: Optional[Dict[str, int]] = None,
        channels: Optional[List[str]] = None,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: Optional[Tuple[float, float]] = (-0.2, 0)
    ):
        if not HAS_MNE:
            raise ImportError("MNE required for EEG loading")

        # Default T-maze condition mapping (Reward vs No-Reward)
        self.condition_mapping = condition_mapping or {
            'MazeReward': 1,
            'NoMazeReward': 1,
            'MazeNoReward': 0,
            'NoMazeNoReward': 0
        }
        self.channels = channels
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline

    def load(self, epochs_path: str, subject_id: Optional[str] = None) -> TMazeEEGData:
        """
        Load EEG epochs from file.

        Parameters
        ----------
        epochs_path : str
            Path to MNE epochs file (.fif)
        subject_id : str, optional
            Subject ID (extracted from filename if not provided)

        Returns
        -------
        TMazeEEGData
            Loaded EEG data container
        """
        # Load epochs
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)

        # Apply time window
        epochs = epochs.crop(tmin=self.tmin, tmax=self.tmax)

        # Apply baseline correction
        if self.baseline:
            epochs = epochs.apply_baseline(self.baseline)

        # Select channels
        if self.channels:
            available = [ch for ch in self.channels if ch in epochs.ch_names]
            if available:
                epochs = epochs.pick_channels(available)

        # Get data and labels
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        times = epochs.times

        # Map events to labels
        labels = self._map_labels(epochs.events, epochs.event_id)

        # Extract subject ID
        if subject_id is None:
            subject_id = Path(epochs_path).stem.split('_')[0]

        condition_names = list(self.condition_mapping.keys())

        return TMazeEEGData(
            data=data,
            times=times,
            labels=labels,
            condition_names=condition_names,
            channels=epochs.ch_names,
            sfreq=epochs.info['sfreq'],
            subject_id=subject_id,
            metadata={
                'epochs_path': epochs_path,
                'n_dropped': len(epochs.drop_log) - len(epochs),
                'event_id': epochs.event_id
            }
        )

    def _map_labels(self, events: np.ndarray, event_id: Dict[str, int]) -> np.ndarray:
        """Map MNE events to binary labels."""
        # Invert event_id for lookup
        id_to_name = {v: k for k, v in event_id.items()}

        labels = []
        for event in events:
            event_code = event[2]
            event_name = id_to_name.get(event_code, "unknown")

            # Find matching condition
            label = None
            for cond, lbl in self.condition_mapping.items():
                if cond.lower() in event_name.lower():
                    label = lbl
                    break

            if label is None:
                label = 0  # Default
            labels.append(label)

        return np.array(labels)

    def load_multiple(
        self,
        epochs_paths: List[str],
        subject_ids: Optional[List[str]] = None
    ) -> List[TMazeEEGData]:
        """Load multiple subjects' EEG data."""
        if subject_ids is None:
            subject_ids = [None] * len(epochs_paths)

        return [
            self.load(path, subj_id)
            for path, subj_id in zip(epochs_paths, subject_ids)
        ]


class TMazefMRILoader:
    """
    Load T-maze fMRI beta images and extract ROI values.

    Parameters
    ----------
    atlas_path : str, optional
        Path to atlas NIfTI file (default: HCP 426)
    roi_names : List[str], optional
        Names for each ROI label
    standardize : bool
        Whether to z-score ROI values per run
    condition_mapping : Dict[str, int], optional
        Mapping from condition names to labels
    """

    def __init__(
        self,
        atlas_path: Optional[str] = None,
        roi_names: Optional[List[str]] = None,
        standardize: bool = True,
        condition_mapping: Optional[Dict[str, int]] = None
    ):
        if not HAS_NILEARN:
            raise ImportError("Nilearn required for fMRI loading")

        # Load atlas
        if atlas_path is None:
            self.atlas_path, self.roi_names = load_hcp_atlas(426)
        else:
            self.atlas_path = atlas_path
            self.roi_names = roi_names or [f"ROI_{i}" for i in range(426)]

        self.standardize = standardize

        # Default condition mapping
        self.condition_mapping = condition_mapping or {
            'MazeReward': 1,
            'NoMazeReward': 1,
            'MazeNoReward': 0,
            'NoMazeNoReward': 0
        }

        # Create masker
        self.masker = NiftiLabelsMasker(
            labels_img=self.atlas_path,
            standardize=standardize,
            strategy='mean',
            verbose=0
        )

    def load(
        self,
        beta_paths: List[str],
        labels: np.ndarray,
        subject_id: str,
        runs: Optional[np.ndarray] = None,
        condition_names: Optional[List[str]] = None
    ) -> TMAzefMRIData:
        """
        Load fMRI beta images and extract ROI values.

        Parameters
        ----------
        beta_paths : List[str]
            Paths to beta NIfTI images (one per trial)
        labels : np.ndarray
            Condition labels for each trial
        subject_id : str
            Subject identifier
        runs : np.ndarray, optional
            Run indices for cross-validation
        condition_names : List[str], optional
            Names for each condition

        Returns
        -------
        TMAzefMRIData
            Loaded fMRI data container
        """
        # Load and extract ROI values
        roi_data = []
        for beta_path in beta_paths:
            img = nib.load(beta_path)
            roi_values = self.masker.fit_transform(img)
            roi_data.append(roi_values.squeeze())

        data = np.array(roi_data)  # (n_trials, n_rois)

        if condition_names is None:
            condition_names = list(self.condition_mapping.keys())

        return TMAzefMRIData(
            data=data,
            labels=labels,
            condition_names=condition_names,
            roi_names=self.roi_names,
            subject_id=subject_id,
            atlas_name="HCP_426",
            runs=runs,
            metadata={'beta_paths': beta_paths}
        )

    def load_from_4d(
        self,
        bold_4d_path: str,
        events_path: str,
        subject_id: str,
        label_column: str = 'condition',
        run_column: Optional[str] = 'run'
    ) -> TMAzefMRIData:
        """
        Load from 4D BOLD and events file.

        For pre-computed GLM betas organized as 4D NIfTI.
        """
        import pandas as pd

        # Load 4D image
        img_4d = nib.load(bold_4d_path)
        data = self.masker.fit_transform(img_4d)  # (n_volumes, n_rois)

        # Load events
        events = pd.read_csv(events_path)
        labels = events[label_column].values

        # Map string labels to integers if needed
        if labels.dtype == object:
            labels = np.array([self.condition_mapping.get(l, 0) for l in labels])

        runs = events[run_column].values if run_column else None

        return TMAzefMRIData(
            data=data,
            labels=labels,
            condition_names=list(self.condition_mapping.keys()),
            roi_names=self.roi_names,
            subject_id=subject_id,
            atlas_name="HCP_426",
            runs=runs,
            metadata={'bold_path': bold_4d_path, 'events_path': events_path}
        )

    def load_afni_betas(
        self,
        afni_dir: str,
        subject_id: str,
        conditions: List[str] = None
    ) -> TMAzefMRIData:
        """
        Load AFNI GLM beta outputs.

        Expects AFNI-style directory structure with condition-labeled betas.
        """
        conditions = conditions or ['MazeReward', 'MazeNoReward',
                                     'NoMazeReward', 'NoMazeNoReward']

        beta_data = []
        labels = []

        for i, cond in enumerate(conditions):
            # Find beta files for this condition
            beta_pattern = os.path.join(afni_dir, f"*{cond}*.nii*")
            import glob
            beta_files = sorted(glob.glob(beta_pattern))

            for bf in beta_files:
                img = nib.load(bf)
                roi_values = self.masker.fit_transform(img)
                beta_data.append(roi_values.squeeze())
                labels.append(self.condition_mapping.get(cond, i))

        data = np.array(beta_data)
        labels = np.array(labels)

        return TMAzefMRIData(
            data=data,
            labels=labels,
            condition_names=conditions,
            roi_names=self.roi_names,
            subject_id=subject_id,
            atlas_name="HCP_426",
            metadata={'afni_dir': afni_dir}
        )


class TMazeSubjectLoader:
    """
    Load complete subject data (EEG + fMRI).

    Parameters
    ----------
    eeg_loader : TMazeEEGLoader, optional
    fmri_loader : TMazefMRILoader, optional
    """

    def __init__(
        self,
        eeg_loader: Optional[TMazeEEGLoader] = None,
        fmri_loader: Optional[TMazefMRILoader] = None
    ):
        self.eeg_loader = eeg_loader or TMazeEEGLoader()
        self.fmri_loader = fmri_loader or TMazefMRILoader()

    def load(
        self,
        subject_id: str,
        eeg_path: Optional[str] = None,
        fmri_path: Optional[str] = None,
        fmri_labels: Optional[np.ndarray] = None,
        fmri_runs: Optional[np.ndarray] = None
    ) -> TMazeSubject:
        """
        Load subject data from EEG and/or fMRI.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        eeg_path : str, optional
            Path to EEG epochs file
        fmri_path : str, optional
            Path to fMRI betas (directory or 4D file)
        fmri_labels : np.ndarray, optional
            Labels for fMRI trials
        fmri_runs : np.ndarray, optional
            Run indices for fMRI

        Returns
        -------
        TMazeSubject
            Subject data container
        """
        eeg_data = None
        fmri_data = None

        if eeg_path:
            eeg_data = self.eeg_loader.load(eeg_path, subject_id)

        if fmri_path:
            if os.path.isdir(fmri_path):
                fmri_data = self.fmri_loader.load_afni_betas(fmri_path, subject_id)
            elif fmri_labels is not None:
                # List of beta files
                import glob
                beta_files = sorted(glob.glob(os.path.join(fmri_path, "*.nii*")))
                fmri_data = self.fmri_loader.load(
                    beta_files, fmri_labels, subject_id, fmri_runs
                )

        return TMazeSubject(
            subject_id=subject_id,
            eeg_data=eeg_data,
            fmri_data=fmri_data
        )

    def load_batch(
        self,
        subject_info: List[Dict]
    ) -> List[TMazeSubject]:
        """
        Load multiple subjects.

        Parameters
        ----------
        subject_info : List[Dict]
            List of dicts with keys: subject_id, eeg_path, fmri_path, etc.

        Returns
        -------
        List[TMazeSubject]
        """
        subjects = []
        for info in subject_info:
            subj = self.load(**info)
            subjects.append(subj)
        return subjects
