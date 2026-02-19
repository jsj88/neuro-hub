"""
HPC job submission for Slurm clusters.

Provides job submission, monitoring, and result collection
for running T-maze analysis on HPC systems like Amarel.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import subprocess
import os
import time
import json
from datetime import datetime


@dataclass
class SlurmJob:
    """Representation of a Slurm job."""
    job_id: str
    name: str
    script_path: Path
    status: str = 'PENDING'
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    output_file: Optional[Path] = None
    error_file: Optional[Path] = None
    metadata: Dict = field(default_factory=dict)

    def is_running(self) -> bool:
        return self.status in ['PENDING', 'RUNNING', 'CONFIGURING']

    def is_completed(self) -> bool:
        return self.status == 'COMPLETED'

    def is_failed(self) -> bool:
        return self.status in ['FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL']


class SlurmSubmitter:
    """
    Submit and manage Slurm jobs for T-maze analysis.

    Parameters
    ----------
    partition : str
        Slurm partition (default: 'main')
    time : str
        Time limit (default: '02:00:00')
    mem : str
        Memory per node (default: '16G')
    cpus_per_task : int
        CPUs per task (default: 4)
    account : str, optional
        Slurm account for billing
    output_dir : Path
        Directory for job outputs

    Examples
    --------
    >>> submitter = SlurmSubmitter(partition='main', time='04:00:00')
    >>> jobs = submitter.submit_array(
    ...     script_template='analysis.py',
    ...     subject_ids=['sub-01', 'sub-02'],
    ...     conda_env='tmaze'
    ... )
    >>> submitter.monitor_jobs(jobs)
    """

    def __init__(
        self,
        partition: str = 'main',
        time: str = '02:00:00',
        mem: str = '16G',
        cpus_per_task: int = 4,
        account: Optional[str] = None,
        output_dir: Optional[Path] = None,
        conda_env: Optional[str] = None
    ):
        self.partition = partition
        self.time = time
        self.mem = mem
        self.cpus_per_task = cpus_per_task
        self.account = account
        self.conda_env = conda_env

        self.output_dir = Path(output_dir) if output_dir else Path('slurm_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def submit_job(
        self,
        script_content: str,
        job_name: str,
        additional_args: Optional[Dict] = None,
        dry_run: bool = False
    ) -> SlurmJob:
        """
        Submit a single Slurm job.

        Parameters
        ----------
        script_content : str
            Content of the job script
        job_name : str
            Name for the job
        additional_args : Dict, optional
            Additional Slurm arguments
        dry_run : bool
            If True, create script but don't submit

        Returns
        -------
        SlurmJob
        """
        # Create script file
        script_path = self.output_dir / f'{job_name}.sh'

        # Build Slurm header
        header = self._build_header(job_name, additional_args)

        # Add conda activation if specified
        if self.conda_env:
            script_content = f"""
source ~/.bashrc
conda activate {self.conda_env}

{script_content}
"""

        full_script = header + script_content

        # Write script
        with open(script_path, 'w') as f:
            f.write(full_script)

        script_path.chmod(0o755)

        if dry_run:
            print(f"Dry run: Would submit {script_path}")
            return SlurmJob(
                job_id='DRY_RUN',
                name=job_name,
                script_path=script_path,
                status='DRY_RUN'
            )

        # Submit job
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {result.stderr}")

        # Parse job ID
        job_id = result.stdout.strip().split()[-1]

        return SlurmJob(
            job_id=job_id,
            name=job_name,
            script_path=script_path,
            status='PENDING',
            submit_time=datetime.now().isoformat(),
            output_file=self.output_dir / f'{job_name}_{job_id}.out',
            error_file=self.output_dir / f'{job_name}_{job_id}.err'
        )

    def submit_array(
        self,
        python_script: str,
        subject_ids: List[str],
        job_name: str = 'tmaze_analysis',
        array_batch_size: int = 10,
        dry_run: bool = False,
        **script_kwargs
    ) -> List[SlurmJob]:
        """
        Submit job array for multiple subjects.

        Parameters
        ----------
        python_script : str
            Path to Python script to run
        subject_ids : List[str]
            Subject IDs to process
        job_name : str
            Base name for jobs
        array_batch_size : int
            Number of subjects per array task
        dry_run : bool
            If True, create scripts but don't submit
        **script_kwargs
            Additional arguments passed to script

        Returns
        -------
        List[SlurmJob]
        """
        jobs = []

        # Create subject list file
        subjects_file = self.output_dir / f'{job_name}_subjects.txt'
        with open(subjects_file, 'w') as f:
            for subj in subject_ids:
                f.write(f"{subj}\n")

        n_subjects = len(subject_ids)
        n_tasks = (n_subjects + array_batch_size - 1) // array_batch_size

        # Build script content
        kwargs_str = ' '.join(f'--{k} {v}' for k, v in script_kwargs.items())

        script_content = f'''
# Get subject from list based on array task ID
SUBJECTS_FILE="{subjects_file}"
START_IDX=$((SLURM_ARRAY_TASK_ID * {array_batch_size}))
END_IDX=$(($START_IDX + {array_batch_size}))

# Process subjects in this batch
for i in $(seq $START_IDX $(($END_IDX - 1))); do
    SUBJECT=$(sed -n "$((i + 1))p" $SUBJECTS_FILE)
    if [ -n "$SUBJECT" ]; then
        echo "Processing subject: $SUBJECT"
        python {python_script} --subject $SUBJECT {kwargs_str}
    fi
done
'''

        # Submit array job
        additional_args = {
            'array': f'0-{n_tasks - 1}'
        }

        job = self.submit_job(
            script_content,
            job_name,
            additional_args=additional_args,
            dry_run=dry_run
        )

        job.metadata['subjects'] = subject_ids
        job.metadata['n_tasks'] = n_tasks

        jobs.append(job)
        return jobs

    def _build_header(
        self,
        job_name: str,
        additional_args: Optional[Dict] = None
    ) -> str:
        """Build Slurm script header."""
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={job_name}',
            f'#SBATCH --partition={self.partition}',
            f'#SBATCH --time={self.time}',
            f'#SBATCH --mem={self.mem}',
            f'#SBATCH --cpus-per-task={self.cpus_per_task}',
            f'#SBATCH --output={self.output_dir}/{job_name}_%j.out',
            f'#SBATCH --error={self.output_dir}/{job_name}_%j.err'
        ]

        if self.account:
            lines.append(f'#SBATCH --account={self.account}')

        if additional_args:
            for key, value in additional_args.items():
                lines.append(f'#SBATCH --{key}={value}')

        lines.append('')
        lines.append('echo "Job started at $(date)"')
        lines.append('echo "Running on $(hostname)"')
        lines.append('')

        return '\n'.join(lines)

    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a single job."""
        result = subprocess.run(
            ['sacct', '-j', job_id, '--format=JobID,State,ExitCode,Start,End',
             '--parsable2', '--noheader'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return {'status': 'UNKNOWN', 'error': result.stderr}

        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return {'status': 'UNKNOWN'}

        # Parse first line (main job, not steps)
        parts = lines[0].split('|')
        if len(parts) >= 4:
            return {
                'job_id': parts[0],
                'status': parts[1],
                'exit_code': parts[2],
                'start_time': parts[3] if len(parts) > 3 else None,
                'end_time': parts[4] if len(parts) > 4 else None
            }

        return {'status': 'UNKNOWN'}


def monitor_jobs(
    jobs: List[SlurmJob],
    poll_interval: int = 30,
    timeout: Optional[int] = None,
    verbose: bool = True
) -> List[SlurmJob]:
    """
    Monitor job status until completion.

    Parameters
    ----------
    jobs : List[SlurmJob]
        Jobs to monitor
    poll_interval : int
        Seconds between status checks
    timeout : int, optional
        Maximum time to wait (seconds)
    verbose : bool
        Print status updates

    Returns
    -------
    List[SlurmJob]
        Updated jobs with final status
    """
    start_time = time.time()

    while True:
        # Update status
        running = 0
        completed = 0
        failed = 0

        for job in jobs:
            if job.status in ['DRY_RUN', 'UNKNOWN']:
                continue

            if job.is_running():
                result = subprocess.run(
                    ['sacct', '-j', job.job_id, '--format=State', '--parsable2', '--noheader'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    job.status = result.stdout.strip().split('\n')[0].split('|')[0]

            if job.is_running():
                running += 1
            elif job.is_completed():
                completed += 1
            else:
                failed += 1

        if verbose:
            print(f"Status: {running} running, {completed} completed, {failed} failed")

        # Check if all done
        if running == 0:
            break

        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            print("Timeout reached")
            break

        time.sleep(poll_interval)

    return jobs


def collect_results(
    jobs: List[SlurmJob],
    results_pattern: str = '{output_dir}/results_{subject}.json'
) -> Dict[str, Any]:
    """
    Collect results from completed jobs.

    Parameters
    ----------
    jobs : List[SlurmJob]
        Completed jobs
    results_pattern : str
        Pattern for result files

    Returns
    -------
    Dict
        Collected results keyed by subject
    """
    results = {}

    for job in jobs:
        if not job.is_completed():
            continue

        subjects = job.metadata.get('subjects', [job.name])

        for subject in subjects:
            result_file = Path(
                results_pattern.format(
                    output_dir=job.script_path.parent,
                    subject=subject
                )
            )

            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        results[subject] = json.load(f)
                except Exception as e:
                    results[subject] = {'error': str(e)}
            else:
                results[subject] = {'error': 'Result file not found'}

    return results


def generate_slurm_template(
    output_path: Path,
    python_script: str,
    **kwargs
) -> Path:
    """
    Generate a template Slurm script.

    Parameters
    ----------
    output_path : Path
        Where to save the template
    python_script : str
        Python script to run
    **kwargs
        Slurm configuration options

    Returns
    -------
    Path
        Path to generated template
    """
    template = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Configuration
SUBJECT=$1
PYTHON_SCRIPT="{python_script}"

# Load environment
source ~/.bashrc
conda activate {conda_env}

# Run analysis
echo "Processing subject: $SUBJECT"
echo "Started at: $(date)"

python $PYTHON_SCRIPT --subject $SUBJECT

echo "Finished at: $(date)"
'''

    defaults = {
        'job_name': 'tmaze_analysis',
        'partition': 'main',
        'time': '04:00:00',
        'mem': '16G',
        'cpus': 4,
        'python_script': python_script,
        'conda_env': 'tmaze'
    }
    defaults.update(kwargs)

    script = template.format(**defaults)

    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        f.write(script)

    output_path.chmod(0o755)
    print(f"Generated template at: {output_path}")

    return output_path
