"""
Data containers for T-maze EEG-fMRI analysis.

Based on consolidated patterns from 5 years of T-maze analysis notebooks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class TMazeConditions(Enum):
    """T-maze experimental conditions."""
    MAZE_REWARD = "MazeReward"
    MAZE_NO_REWARD = "MazeNoReward"
    NO_MAZE_REWARD = "NoMazeReward"
    NO_MAZE_NO_REWARD = "NoMazeNoReward"

    # Condition groupings for classification
    @classmethod
    def maze_conditions(cls) -> List["TMazeConditions"]:
        return [cls.MAZE_REWARD, cls.MAZE_NO_REWARD]

    @classmethod
    def no_maze_conditions(cls) -> List["TMazeConditions"]:
        return [cls.NO_MAZE_REWARD, cls.NO_MAZE_NO_REWARD]

    @classmethod
    def reward_conditions(cls) -> List["TMazeConditions"]:
        return [cls.MAZE_REWARD, cls.NO_MAZE_REWARD]

    @classmethod
    def no_reward_conditions(cls) -> List["TMazeConditions"]:
        return [cls.MAZE_NO_REWARD, cls.NO_MAZE_NO_REWARD]


@dataclass
class TMazeEEGData:
    """
    Container for T-maze EEG epoch data.

    Attributes
    ----------
    data : np.ndarray
        Epoch data (n_epochs, n_channels, n_times)
    times : np.ndarray
        Time vector in seconds
    labels : np.ndarray
        Condition labels for each epoch (0/1 for binary classification)
    condition_names : List[str]
        Names of conditions corresponding to label values
    channels : List[str]
        Channel names
    sfreq : float
        Sampling frequency in Hz
    subject_id : str
        Subject identifier
    metadata : Dict
        Additional metadata (trial info, events, etc.)
    """
    data: np.ndarray
    times: np.ndarray
    labels: np.ndarray
    condition_names: List[str]
    channels: List[str]
    sfreq: float
    subject_id: str
    metadata: Dict = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_times(self) -> int:
        return self.data.shape[2]

    @property
    def tmin(self) -> float:
        return float(self.times[0])

    @property
    def tmax(self) -> float:
        return float(self.times[-1])

    def get_time_window(self, tmin: float, tmax: float) -> Tuple[np.ndarray, np.ndarray]:
        """Extract data within a time window."""
        mask = (self.times >= tmin) & (self.times <= tmax)
        return self.data[:, :, mask], self.times[mask]

    def get_channels(self, channel_names: List[str]) -> np.ndarray:
        """Extract data for specific channels."""
        indices = [self.channels.index(ch) for ch in channel_names if ch in self.channels]
        return self.data[:, indices, :]

    def get_rewp_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get REWP time window (240-340ms post-feedback)."""
        return self.get_time_window(0.240, 0.340)

    def get_fcz_channels(self) -> np.ndarray:
        """Get frontocentral channels for REWP analysis."""
        fcz_channels = ['FCz', 'Fz', 'Cz', 'FC1', 'FC2']
        available = [ch for ch in fcz_channels if ch in self.channels]
        return self.get_channels(available)

    def __repr__(self) -> str:
        return (f"TMazeEEGData(subject={self.subject_id}, "
                f"epochs={self.n_epochs}, channels={self.n_channels}, "
                f"times={self.n_times}, sfreq={self.sfreq}Hz)")


@dataclass
class TMAzefMRIData:
    """
    Container for T-maze fMRI beta/ROI data.

    Attributes
    ----------
    data : np.ndarray
        Beta values (n_trials, n_rois) or voxel patterns
    labels : np.ndarray
        Condition labels for each trial
    condition_names : List[str]
        Names of conditions
    roi_names : List[str]
        Names of ROIs (from HCP atlas or custom)
    subject_id : str
        Subject identifier
    atlas_name : str
        Name of atlas used (e.g., "HCP_426")
    runs : np.ndarray
        Run indices for cross-validation
    metadata : Dict
        Additional metadata (TR, task timing, etc.)
    """
    data: np.ndarray
    labels: np.ndarray
    condition_names: List[str]
    roi_names: List[str]
    subject_id: str
    atlas_name: str = "HCP_426"
    runs: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return self.data.shape[0]

    @property
    def n_rois(self) -> int:
        return self.data.shape[1]

    def get_roi(self, roi_name: str) -> np.ndarray:
        """Get data for a specific ROI."""
        if roi_name in self.roi_names:
            idx = self.roi_names.index(roi_name)
            return self.data[:, idx]
        raise ValueError(f"ROI '{roi_name}' not found")

    def get_rois(self, roi_names: List[str]) -> np.ndarray:
        """Get data for multiple ROIs."""
        indices = [self.roi_names.index(roi) for roi in roi_names if roi in self.roi_names]
        return self.data[:, indices]

    def get_network_rois(self, network: str) -> np.ndarray:
        """
        Get ROIs belonging to a specific network.

        Networks: DMN, FPN, SAL, VIS, MOT, etc.
        """
        # HCP network prefixes
        network_prefixes = {
            'DMN': ['Default', 'DMN'],
            'FPN': ['Frontoparietal', 'FPN', 'DorsAttn'],
            'SAL': ['Salience', 'VentAttn', 'Cingulo'],
            'VIS': ['Visual', 'VIS'],
            'MOT': ['Somatomotor', 'MOT'],
            'LIM': ['Limbic', 'LIM']
        }

        prefixes = network_prefixes.get(network.upper(), [network])
        matching = [roi for roi in self.roi_names
                   if any(p.lower() in roi.lower() for p in prefixes)]
        return self.get_rois(matching)

    def __repr__(self) -> str:
        return (f"TMAzefMRIData(subject={self.subject_id}, "
                f"trials={self.n_trials}, rois={self.n_rois}, "
                f"atlas={self.atlas_name})")


@dataclass
class TMazeSubject:
    """
    Container for a single subject's T-maze data.

    Combines EEG and fMRI data for multimodal analysis.
    """
    subject_id: str
    eeg_data: Optional[TMazeEEGData] = None
    fmri_data: Optional[TMAzefMRIData] = None
    behavioral: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def has_eeg(self) -> bool:
        return self.eeg_data is not None

    @property
    def has_fmri(self) -> bool:
        return self.fmri_data is not None

    @property
    def is_multimodal(self) -> bool:
        return self.has_eeg and self.has_fmri

    def get_aligned_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get trial-aligned EEG and fMRI features.

        Returns
        -------
        X_eeg : np.ndarray
            EEG features (n_trials, eeg_features)
        X_fmri : np.ndarray
            fMRI features (n_trials, n_rois)
        y : np.ndarray
            Condition labels
        """
        if not self.is_multimodal:
            raise ValueError("Subject does not have both EEG and fMRI data")

        # Get EEG features (mean across REWP window)
        rewp_data, _ = self.eeg_data.get_rewp_window()
        X_eeg = rewp_data.mean(axis=2)  # Average over time

        X_fmri = self.fmri_data.data
        y = self.fmri_data.labels

        # Ensure alignment
        min_trials = min(X_eeg.shape[0], X_fmri.shape[0])

        return X_eeg[:min_trials], X_fmri[:min_trials], y[:min_trials]

    def __repr__(self) -> str:
        modalities = []
        if self.has_eeg:
            modalities.append("EEG")
        if self.has_fmri:
            modalities.append("fMRI")
        return f"TMazeSubject({self.subject_id}, modalities={modalities})"


@dataclass
class TMazeGroup:
    """Container for group-level T-maze data."""
    subjects: List[TMazeSubject]
    metadata: Dict = field(default_factory=dict)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def subject_ids(self) -> List[str]:
        return [s.subject_id for s in self.subjects]

    def get_eeg_subjects(self) -> List[TMazeSubject]:
        return [s for s in self.subjects if s.has_eeg]

    def get_fmri_subjects(self) -> List[TMazeSubject]:
        return [s for s in self.subjects if s.has_fmri]

    def get_multimodal_subjects(self) -> List[TMazeSubject]:
        return [s for s in self.subjects if s.is_multimodal]

    def stack_eeg_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack EEG data across subjects for group analysis."""
        eeg_subjects = self.get_eeg_subjects()

        X_list = []
        y_list = []
        subj_list = []

        for i, subj in enumerate(eeg_subjects):
            X_list.append(subj.eeg_data.data)
            y_list.append(subj.eeg_data.labels)
            subj_list.append(np.full(subj.eeg_data.n_epochs, i))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list)
        subjects = np.concatenate(subj_list)

        return X, y, subjects

    def stack_fmri_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack fMRI data across subjects for group analysis."""
        fmri_subjects = self.get_fmri_subjects()

        X_list = []
        y_list = []
        subj_list = []

        for i, subj in enumerate(fmri_subjects):
            X_list.append(subj.fmri_data.data)
            y_list.append(subj.fmri_data.labels)
            subj_list.append(np.full(subj.fmri_data.n_trials, i))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list)
        subjects = np.concatenate(subj_list)

        return X, y, subjects
