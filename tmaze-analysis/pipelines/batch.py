"""
Batch processing infrastructure for T-maze analysis.

Provides parallel processing, checkpointing, and error handling
for multi-subject analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
import json
import time
import traceback
from datetime import datetime
import numpy as np

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    n_jobs: int = -1
    backend: str = 'loky'
    verbose: int = 10
    checkpoint_dir: Optional[Path] = None
    checkpoint_freq: int = 10
    continue_on_error: bool = True
    max_retries: int = 2
    timeout: Optional[int] = None


@dataclass
class ProcessingResult:
    """Container for individual processing result."""
    subject_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'subject_id': self.subject_id,
            'success': self.success,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


@dataclass
class BatchResult:
    """Container for batch processing results."""
    results: List[ProcessingResult]
    total_time: float
    n_successful: int
    n_failed: int
    config: Optional[BatchConfig] = None

    @property
    def success_rate(self) -> float:
        return self.n_successful / len(self.results) if self.results else 0.0

    def get_successful(self) -> List[ProcessingResult]:
        """Get successful results."""
        return [r for r in self.results if r.success]

    def get_failed(self) -> List[ProcessingResult]:
        """Get failed results."""
        return [r for r in self.results if not r.success]

    def summary(self) -> str:
        """Get summary string."""
        return (f"BatchResult: {self.n_successful}/{len(self.results)} successful "
                f"({self.success_rate:.1%}) in {self.total_time:.1f}s")


class BatchProcessor:
    """
    Batch processor for multi-subject T-maze analysis.

    Parameters
    ----------
    process_func : Callable
        Function to apply to each subject
    config : BatchConfig
        Processing configuration

    Examples
    --------
    >>> def process_subject(subj_id, data_path):
    ...     # Load and process subject data
    ...     return {'accuracy': 0.75}
    >>>
    >>> processor = BatchProcessor(process_subject)
    >>> results = processor.run(subject_ids, data_paths)
    """

    def __init__(
        self,
        process_func: Callable,
        config: Optional[BatchConfig] = None
    ):
        self.process_func = process_func
        self.config = config or BatchConfig()

        if self.config.checkpoint_dir:
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        subject_ids: List[str],
        *args,
        **kwargs
    ) -> BatchResult:
        """
        Run batch processing.

        Parameters
        ----------
        subject_ids : List[str]
            List of subject IDs to process
        *args, **kwargs
            Additional arguments passed to process_func

        Returns
        -------
        BatchResult
        """
        start_time = time.time()

        # Load checkpoint if exists
        completed = self._load_checkpoint()

        # Filter out already completed subjects
        pending = [s for s in subject_ids if s not in completed]

        if pending:
            print(f"Processing {len(pending)} subjects ({len(completed)} already completed)")

            if HAS_JOBLIB and self.config.n_jobs != 1:
                # Parallel processing
                new_results = Parallel(
                    n_jobs=self.config.n_jobs,
                    backend=self.config.backend,
                    verbose=self.config.verbose
                )(
                    delayed(self._process_single)(subj_id, *args, **kwargs)
                    for subj_id in pending
                )
            else:
                # Sequential processing
                new_results = []
                for i, subj_id in enumerate(pending):
                    result = self._process_single(subj_id, *args, **kwargs)
                    new_results.append(result)

                    # Checkpoint
                    if (i + 1) % self.config.checkpoint_freq == 0:
                        self._save_checkpoint(completed, new_results)

            # Combine with previous results
            all_results = list(completed.values()) + new_results

            # Final checkpoint
            self._save_checkpoint(completed, new_results)
        else:
            print("All subjects already processed")
            all_results = list(completed.values())

        total_time = time.time() - start_time

        return BatchResult(
            results=all_results,
            total_time=total_time,
            n_successful=sum(1 for r in all_results if r.success),
            n_failed=sum(1 for r in all_results if not r.success),
            config=self.config
        )

    def _process_single(
        self,
        subject_id: str,
        *args,
        **kwargs
    ) -> ProcessingResult:
        """Process a single subject with error handling."""
        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                result = self.process_func(subject_id, *args, **kwargs)

                return ProcessingResult(
                    subject_id=subject_id,
                    success=True,
                    result=result,
                    execution_time=time.time() - start_time,
                    metadata={'attempt': attempt + 1}
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

                if attempt < self.config.max_retries:
                    print(f"Retry {attempt + 1} for {subject_id}: {str(e)}")
                    continue

                if not self.config.continue_on_error:
                    raise

                return ProcessingResult(
                    subject_id=subject_id,
                    success=False,
                    error=error_msg,
                    execution_time=time.time() - start_time,
                    metadata={'attempts': attempt + 1}
                )

    def _load_checkpoint(self) -> Dict[str, ProcessingResult]:
        """Load checkpoint from disk."""
        if self.config.checkpoint_dir is None:
            return {}

        checkpoint_file = self.config.checkpoint_dir / 'checkpoint.json'
        if not checkpoint_file.exists():
            return {}

        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            completed = {}
            for item in data['results']:
                result = ProcessingResult(
                    subject_id=item['subject_id'],
                    success=item['success'],
                    error=item.get('error'),
                    execution_time=item.get('execution_time', 0),
                    metadata=item.get('metadata', {})
                )
                completed[result.subject_id] = result

            print(f"Loaded checkpoint with {len(completed)} completed subjects")
            return completed

        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return {}

    def _save_checkpoint(
        self,
        completed: Dict[str, ProcessingResult],
        new_results: List[ProcessingResult]
    ):
        """Save checkpoint to disk."""
        if self.config.checkpoint_dir is None:
            return

        # Combine results
        all_results = list(completed.values()) + new_results

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'n_results': len(all_results),
            'results': [r.to_dict() for r in all_results]
        }

        checkpoint_file = self.config.checkpoint_dir / 'checkpoint.json'

        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")


def parallel_map(
    func: Callable,
    items: List[Any],
    n_jobs: int = -1,
    backend: str = 'loky',
    verbose: int = 10,
    **kwargs
) -> List[Any]:
    """
    Parallel map with joblib.

    Parameters
    ----------
    func : Callable
        Function to apply
    items : List
        Items to process
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    backend : str
        Joblib backend
    verbose : int
        Verbosity level
    **kwargs
        Additional arguments for func

    Returns
    -------
    List
        Results for each item
    """
    if not HAS_JOBLIB:
        return [func(item, **kwargs) for item in items]

    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(item, **kwargs) for item in items
    )


def checkpoint_resume(
    func: Callable,
    items: List[str],
    checkpoint_file: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Run function with checkpoint/resume capability.

    Parameters
    ----------
    func : Callable
        Function to apply (takes item as first arg)
    items : List[str]
        Items to process (must be unique strings)
    checkpoint_file : str or Path
        Path to checkpoint file
    **kwargs
        Additional arguments for func

    Returns
    -------
    Dict
        Results keyed by item
    """
    checkpoint_file = Path(checkpoint_file)

    # Load existing results
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from checkpoint ({len(results)} completed)")
    else:
        results = {}

    # Process remaining items
    pending = [item for item in items if item not in results]

    for i, item in enumerate(pending):
        try:
            result = func(item, **kwargs)
            results[item] = {'success': True, 'result': result}
        except Exception as e:
            results[item] = {'success': False, 'error': str(e)}

        # Save checkpoint after each item
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(pending)}")

    return results


def create_processing_report(batch_result: BatchResult) -> str:
    """
    Create a text report from batch processing results.

    Parameters
    ----------
    batch_result : BatchResult
        Results from batch processing

    Returns
    -------
    str
        Report text
    """
    lines = [
        "=" * 60,
        "BATCH PROCESSING REPORT",
        "=" * 60,
        "",
        f"Total subjects: {len(batch_result.results)}",
        f"Successful: {batch_result.n_successful}",
        f"Failed: {batch_result.n_failed}",
        f"Success rate: {batch_result.success_rate:.1%}",
        f"Total time: {batch_result.total_time:.1f}s",
        f"Avg time per subject: {batch_result.total_time / len(batch_result.results):.1f}s",
        ""
    ]

    # Failed subjects
    failed = batch_result.get_failed()
    if failed:
        lines.append("FAILED SUBJECTS:")
        lines.append("-" * 40)
        for r in failed:
            lines.append(f"  {r.subject_id}: {r.error[:100] if r.error else 'Unknown error'}")
        lines.append("")

    # Execution times
    times = [r.execution_time for r in batch_result.results if r.success]
    if times:
        lines.append("EXECUTION TIME STATISTICS:")
        lines.append("-" * 40)
        lines.append(f"  Min: {min(times):.1f}s")
        lines.append(f"  Max: {max(times):.1f}s")
        lines.append(f"  Mean: {np.mean(times):.1f}s")
        lines.append(f"  Median: {np.median(times):.1f}s")

    lines.append("=" * 60)

    return "\n".join(lines)
