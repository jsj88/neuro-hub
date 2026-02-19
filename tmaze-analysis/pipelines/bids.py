"""
BIDS format handling for T-maze data.

Provides loading from BIDS datasets and export to BIDS derivatives.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
import re

try:
    from bids import BIDSLayout
    HAS_PYBIDS = True
except ImportError:
    HAS_PYBIDS = False


@dataclass
class BIDSDataset:
    """
    BIDS dataset wrapper for T-maze data.

    Parameters
    ----------
    root : Path
        Path to BIDS dataset root
    validate : bool
        Validate BIDS structure
    derivatives : bool
        Include derivatives

    Examples
    --------
    >>> dataset = BIDSDataset('/data/tmaze_bids')
    >>> subjects = dataset.get_subjects()
    >>> eeg_files = dataset.get_eeg_files('sub-01')
    """

    root: Path
    validate: bool = True
    derivatives: bool = False
    _layout: Any = field(default=None, repr=False)

    def __post_init__(self):
        self.root = Path(self.root)
        if HAS_PYBIDS:
            self._layout = BIDSLayout(
                self.root,
                validate=self.validate,
                derivatives=self.derivatives
            )

    def get_subjects(self) -> List[str]:
        """Get list of subject IDs."""
        if HAS_PYBIDS and self._layout:
            return self._layout.get_subjects()

        # Manual fallback
        subjects = []
        for path in self.root.glob('sub-*'):
            if path.is_dir():
                subjects.append(path.name.replace('sub-', ''))
        return sorted(subjects)

    def get_sessions(self, subject: str) -> List[str]:
        """Get sessions for a subject."""
        if HAS_PYBIDS and self._layout:
            return self._layout.get_sessions(subject=subject)

        subj_dir = self.root / f'sub-{subject}'
        sessions = []
        for path in subj_dir.glob('ses-*'):
            if path.is_dir():
                sessions.append(path.name.replace('ses-', ''))
        return sorted(sessions)

    def get_eeg_files(
        self,
        subject: str,
        session: Optional[str] = None,
        task: str = 'tmaze'
    ) -> List[Path]:
        """Get EEG files for a subject."""
        if HAS_PYBIDS and self._layout:
            files = self._layout.get(
                subject=subject,
                session=session,
                task=task,
                suffix='eeg',
                extension=['.set', '.fif', '.vhdr', '.edf']
            )
            return [Path(f.path) for f in files]

        # Manual search
        pattern = f'sub-{subject}'
        if session:
            pattern += f'/ses-{session}'
        pattern += f'/eeg/*task-{task}*_eeg.*'

        return list(self.root.glob(pattern))

    def get_fmri_files(
        self,
        subject: str,
        session: Optional[str] = None,
        task: str = 'tmaze'
    ) -> Dict[str, List[Path]]:
        """Get fMRI files for a subject."""
        if HAS_PYBIDS and self._layout:
            bold_files = self._layout.get(
                subject=subject,
                session=session,
                task=task,
                suffix='bold',
                extension='.nii.gz'
            )
            events_files = self._layout.get(
                subject=subject,
                session=session,
                task=task,
                suffix='events',
                extension='.tsv'
            )
            return {
                'bold': [Path(f.path) for f in bold_files],
                'events': [Path(f.path) for f in events_files]
            }

        # Manual search
        pattern = f'sub-{subject}'
        if session:
            pattern += f'/ses-{session}'
        pattern += f'/func/*task-{task}*'

        bold = list(self.root.glob(pattern + '_bold.nii.gz'))
        events = list(self.root.glob(pattern + '_events.tsv'))

        return {'bold': bold, 'events': events}

    def get_events(
        self,
        subject: str,
        session: Optional[str] = None,
        task: str = 'tmaze'
    ) -> Optional[Path]:
        """Get events file for a subject/session/task."""
        fmri_files = self.get_fmri_files(subject, session, task)
        events = fmri_files.get('events', [])
        return events[0] if events else None

    def get_participants_info(self) -> Dict[str, Dict]:
        """Load participants.tsv info."""
        participants_file = self.root / 'participants.tsv'
        if not participants_file.exists():
            return {}

        import csv
        info = {}
        with open(participants_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                subj_id = row.get('participant_id', '').replace('sub-', '')
                info[subj_id] = dict(row)

        return info


def validate_bids(
    root: Path,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate BIDS dataset structure.

    Parameters
    ----------
    root : Path
        Path to BIDS dataset
    strict : bool
        Use strict validation

    Returns
    -------
    Dict
        Validation results
    """
    root = Path(root)
    issues = []
    warnings = []

    # Check required files
    required_files = ['dataset_description.json', 'participants.tsv']
    for req in required_files:
        if not (root / req).exists():
            issues.append(f"Missing required file: {req}")

    # Check dataset_description.json
    desc_file = root / 'dataset_description.json'
    if desc_file.exists():
        try:
            with open(desc_file, 'r') as f:
                desc = json.load(f)
            if 'Name' not in desc:
                issues.append("dataset_description.json missing 'Name' field")
            if 'BIDSVersion' not in desc:
                warnings.append("dataset_description.json missing 'BIDSVersion' field")
        except json.JSONDecodeError:
            issues.append("dataset_description.json is not valid JSON")

    # Check subject folders
    subjects = list(root.glob('sub-*'))
    if not subjects:
        issues.append("No subject folders found")

    for subj_dir in subjects:
        if not subj_dir.is_dir():
            continue

        # Check for data modalities
        has_eeg = list(subj_dir.rglob('*_eeg.*'))
        has_func = list(subj_dir.rglob('*_bold.nii*'))

        if not has_eeg and not has_func:
            warnings.append(f"{subj_dir.name}: No EEG or fMRI data found")

        # Check for events files
        if has_func:
            events = list(subj_dir.rglob('*_events.tsv'))
            if not events:
                warnings.append(f"{subj_dir.name}: fMRI data without events file")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'n_subjects': len(subjects)
    }


def bids_to_tmaze(
    bids_dataset: BIDSDataset,
    subject: str,
    session: Optional[str] = None,
    task: str = 'tmaze'
) -> Dict[str, Any]:
    """
    Load T-maze data from BIDS format.

    Parameters
    ----------
    bids_dataset : BIDSDataset
        BIDS dataset
    subject : str
        Subject ID
    session : str, optional
        Session ID
    task : str
        Task name

    Returns
    -------
    Dict
        Data dictionary compatible with T-maze loaders
    """
    data = {
        'subject_id': subject,
        'session': session,
        'task': task
    }

    # Get EEG files
    eeg_files = bids_dataset.get_eeg_files(subject, session, task)
    if eeg_files:
        data['eeg_file'] = eeg_files[0]

        # Look for associated files
        eeg_dir = eeg_files[0].parent
        channels_file = list(eeg_dir.glob('*_channels.tsv'))
        electrodes_file = list(eeg_dir.glob('*_electrodes.tsv'))

        if channels_file:
            data['channels_file'] = channels_file[0]
        if electrodes_file:
            data['electrodes_file'] = electrodes_file[0]

    # Get fMRI files
    fmri_files = bids_dataset.get_fmri_files(subject, session, task)
    if fmri_files['bold']:
        data['bold_file'] = fmri_files['bold'][0]
    if fmri_files['events']:
        data['events_file'] = fmri_files['events'][0]

        # Parse events
        data['events'] = _parse_events_tsv(fmri_files['events'][0])

    # Get anatomical
    anat_pattern = f'sub-{subject}'
    if session:
        anat_pattern += f'/ses-{session}'
    anat_pattern += '/anat/*_T1w.nii*'

    anat_files = list(bids_dataset.root.glob(anat_pattern))
    if anat_files:
        data['anat_file'] = anat_files[0]

    return data


def _parse_events_tsv(events_file: Path) -> List[Dict]:
    """Parse BIDS events.tsv file."""
    import csv

    events = []
    with open(events_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            event = {}
            for key, value in row.items():
                # Try to convert to number
                try:
                    if '.' in value:
                        event[key] = float(value)
                    else:
                        event[key] = int(value)
                except (ValueError, TypeError):
                    event[key] = value
            events.append(event)

    return events


def tmaze_to_bids(
    results: Dict[str, Any],
    output_dir: Path,
    analysis_name: str = 'tmaze-classification',
    description: Optional[str] = None
) -> Path:
    """
    Export T-maze results to BIDS derivatives format.

    Parameters
    ----------
    results : Dict
        Analysis results
    output_dir : Path
        Output directory (will create derivatives folder)
    analysis_name : str
        Name of the derivative
    description : str, optional
        Description of the analysis

    Returns
    -------
    Path
        Path to derivatives directory
    """
    output_dir = Path(output_dir)
    deriv_dir = output_dir / 'derivatives' / analysis_name
    deriv_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset_description.json
    description_data = {
        'Name': analysis_name,
        'BIDSVersion': '1.8.0',
        'PipelineDescription': {
            'Name': 'tmaze-analysis',
            'Version': '0.1.0'
        },
        'GeneratedBy': [{
            'Name': 'tmaze-analysis',
            'Description': description or 'T-maze EEG-fMRI classification analysis'
        }]
    }

    with open(deriv_dir / 'dataset_description.json', 'w') as f:
        json.dump(description_data, f, indent=2)

    # Export subject-level results
    for subject_id, subject_results in results.items():
        if not isinstance(subject_results, dict):
            continue

        # Create subject directory
        subj_dir = deriv_dir / f'sub-{subject_id}'
        subj_dir.mkdir(exist_ok=True)

        # Save results as JSON
        results_file = subj_dir / f'sub-{subject_id}_desc-classification_results.json'
        with open(results_file, 'w') as f:
            json.dump(subject_results, f, indent=2, default=str)

    # Create group-level summary
    summary = {
        'n_subjects': len(results),
        'subjects': list(results.keys())
    }

    with open(deriv_dir / 'group_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Exported results to: {deriv_dir}")
    return deriv_dir


def create_bids_sidecar(
    data_type: str,
    **kwargs
) -> Dict:
    """
    Create a BIDS sidecar JSON.

    Parameters
    ----------
    data_type : str
        'eeg', 'bold', or 'events'
    **kwargs
        Additional fields

    Returns
    -------
    Dict
        Sidecar data
    """
    sidecars = {
        'eeg': {
            'TaskName': 'tmaze',
            'TaskDescription': 'T-maze reward learning paradigm',
            'SamplingFrequency': kwargs.get('sfreq', 500),
            'EEGChannelCount': kwargs.get('n_channels', 64),
            'EEGReference': kwargs.get('reference', 'average'),
            'PowerLineFrequency': kwargs.get('powerline', 60),
            'RecordingType': 'continuous'
        },
        'bold': {
            'TaskName': 'tmaze',
            'TaskDescription': 'T-maze reward learning paradigm',
            'RepetitionTime': kwargs.get('tr', 2.0),
            'SliceTimingCorrected': kwargs.get('stc', True)
        },
        'events': {
            'onset': {
                'LongName': 'Event onset time',
                'Description': 'Onset time of event relative to acquisition start',
                'Units': 'seconds'
            },
            'duration': {
                'LongName': 'Event duration',
                'Description': 'Duration of the event',
                'Units': 'seconds'
            },
            'trial_type': {
                'LongName': 'Trial type',
                'Description': 'Type of trial',
                'Levels': {
                    'MazeReward': 'Maze condition with reward',
                    'MazeNoReward': 'Maze condition without reward',
                    'NoMazeReward': 'Control condition with reward',
                    'NoMazeNoReward': 'Control condition without reward'
                }
            }
        }
    }

    base = sidecars.get(data_type, {})
    base.update(kwargs)
    return base
