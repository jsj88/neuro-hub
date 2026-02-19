"""
Configuration settings for T-maze analysis.

Default parameters for EEG, fMRI, and classification pipelines.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EEGConfig:
    """EEG analysis configuration."""
    # Time window
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Tuple[float, float] = (-0.2, 0)

    # REWP window
    rewp_tmin: float = 0.240
    rewp_tmax: float = 0.340

    # Channels
    fcz_channels: List[str] = field(default_factory=lambda: [
        'FCz', 'Fz', 'Cz', 'FC1', 'FC2'
    ])

    # Sampling
    target_sfreq: float = 200.0

    # Filtering (if needed)
    l_freq: Optional[float] = 0.1
    h_freq: Optional[float] = 30.0


@dataclass
class FMRIConfig:
    """fMRI analysis configuration."""
    # Atlas
    atlas_name: str = "HCP_426"
    n_rois: int = 426

    # Preprocessing
    standardize: bool = True
    detrend: bool = True

    # TR (in seconds)
    tr: float = 2.0

    # Networks of interest
    networks: List[str] = field(default_factory=lambda: [
        'DMN', 'FPN', 'SAL', 'VIS', 'MOT', 'LIM'
    ])


@dataclass
class ClassificationConfig:
    """Classification configuration."""
    # Default classifier
    classifier_type: str = 'lda'

    # Cross-validation
    cv_folds: int = 5
    cv_type: str = 'stratified'

    # Permutation testing
    n_permutations: int = 1000

    # Parallel processing
    n_jobs: int = -1

    # Random seed
    random_state: int = 42

    # SVM parameters
    svm_kernel: str = 'linear'
    svm_C: float = 1.0

    # LDA parameters
    lda_solver: str = 'lsqr'
    lda_shrinkage: str = 'auto'


@dataclass
class TMazeConfig:
    """Master configuration for T-maze analysis."""
    # Sub-configurations
    eeg: EEGConfig = field(default_factory=EEGConfig)
    fmri: FMRIConfig = field(default_factory=FMRIConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    # Condition definitions
    conditions: Dict[str, int] = field(default_factory=lambda: {
        'MazeReward': 0,
        'MazeNoReward': 1,
        'NoMazeReward': 2,
        'NoMazeNoReward': 3
    })

    # Binary classification mappings
    reward_mapping: Dict[str, int] = field(default_factory=lambda: {
        'MazeReward': 1,
        'NoMazeReward': 1,
        'MazeNoReward': 0,
        'NoMazeNoReward': 0
    })

    maze_mapping: Dict[str, int] = field(default_factory=lambda: {
        'MazeReward': 1,
        'MazeNoReward': 1,
        'NoMazeReward': 0,
        'NoMazeNoReward': 0
    })

    # Output settings
    output_dir: str = './results'
    save_figures: bool = True
    figure_format: str = 'png'
    figure_dpi: int = 300

    # Logging
    verbose: bool = True


def get_default_config() -> TMazeConfig:
    """Get default T-maze configuration."""
    return TMazeConfig()


def load_config(config_path: str) -> TMazeConfig:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to YAML config file

    Returns
    -------
    TMazeConfig
    """
    try:
        import yaml

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create config objects
        eeg_config = EEGConfig(**config_dict.get('eeg', {}))
        fmri_config = FMRIConfig(**config_dict.get('fmri', {}))
        clf_config = ClassificationConfig(**config_dict.get('classification', {}))

        return TMazeConfig(
            eeg=eeg_config,
            fmri=fmri_config,
            classification=clf_config,
            **{k: v for k, v in config_dict.items()
               if k not in ['eeg', 'fmri', 'classification']}
        )

    except ImportError:
        raise ImportError("PyYAML required for config loading")


def save_config(config: TMazeConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : TMazeConfig
        Configuration to save
    config_path : str
        Output path
    """
    try:
        import yaml
        from dataclasses import asdict

        config_dict = asdict(config)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    except ImportError:
        raise ImportError("PyYAML required for config saving")
