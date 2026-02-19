"""
Connectivity analysis for T-maze EEG-fMRI data.

Includes:
- Functional connectivity matrices
- Dynamic connectivity (sliding window)
- Graph theory metrics
- EEG-specific connectivity (PLI, coherence)
- Multimodal connectivity analysis
"""

from .functional import (
    compute_fc_matrix,
    partial_correlation,
    fc_condition_contrast,
    fisher_z_transform,
    fc_to_adjacency
)

from .dynamic import (
    sliding_window_fc,
    dcc_connectivity,
    detect_fc_states,
    fc_variability,
    DynamicConnectivityResult
)

from .graph import (
    compute_graph_metrics,
    modularity_detection,
    small_world_index,
    rich_club_coefficient,
    hub_identification,
    GraphMetrics
)

from .eeg_connectivity import (
    phase_lag_index,
    coherence,
    phase_synchrony,
    imaginary_coherence,
    weighted_phase_lag_index,
    EEGConnectivityResult
)

from .multimodal import (
    eeg_fmri_coupling,
    information_flow,
    joint_connectivity_graph,
    cross_modal_correlation
)

__all__ = [
    # functional
    'compute_fc_matrix',
    'partial_correlation',
    'fc_condition_contrast',
    'fisher_z_transform',
    'fc_to_adjacency',
    # dynamic
    'sliding_window_fc',
    'dcc_connectivity',
    'detect_fc_states',
    'fc_variability',
    'DynamicConnectivityResult',
    # graph
    'compute_graph_metrics',
    'modularity_detection',
    'small_world_index',
    'rich_club_coefficient',
    'hub_identification',
    'GraphMetrics',
    # eeg_connectivity
    'phase_lag_index',
    'coherence',
    'phase_synchrony',
    'imaginary_coherence',
    'weighted_phase_lag_index',
    'EEGConnectivityResult',
    # multimodal
    'eeg_fmri_coupling',
    'information_flow',
    'joint_connectivity_graph',
    'cross_modal_correlation'
]
