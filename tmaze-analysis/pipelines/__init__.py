"""
Pipeline automation for T-maze analysis.

Includes:
- Batch processing infrastructure
- HPC job submission (Slurm)
- BIDS format handling
- Automated report generation
"""

from .batch import (
    BatchProcessor,
    parallel_map,
    checkpoint_resume,
    ProcessingResult,
    BatchConfig
)

from .hpc import (
    SlurmSubmitter,
    SlurmJob,
    monitor_jobs,
    collect_results
)

from .bids import (
    BIDSDataset,
    validate_bids,
    bids_to_tmaze,
    tmaze_to_bids
)

from .reporting import (
    generate_report,
    create_figure_gallery,
    statistics_table,
    export_for_publication
)

__all__ = [
    # batch
    'BatchProcessor',
    'parallel_map',
    'checkpoint_resume',
    'ProcessingResult',
    'BatchConfig',
    # hpc
    'SlurmSubmitter',
    'SlurmJob',
    'monitor_jobs',
    'collect_results',
    # bids
    'BIDSDataset',
    'validate_bids',
    'bids_to_tmaze',
    'tmaze_to_bids',
    # reporting
    'generate_report',
    'create_figure_gallery',
    'statistics_table',
    'export_for_publication'
]
