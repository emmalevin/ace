"""
Submit ACE training as SLURM batch jobs (one job per model), wait for completion, check exit status.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from inference_runner import (
    _job_succeeded,
    _submit_sbatch,
    _wait_for_job,
)


class TrainingJobFailureError(Exception):
    """Raised when one or more submitted training jobs failed (e.g. OOM, non-zero exit)."""

    def __init__(self, message: str, job_ids: tuple[int, int], failed_mask: tuple[bool, bool]):
        super().__init__(message)
        self.job_ids = job_ids
        self.failed_mask = failed_mask


def submit_batched_training_jobs(
    model1_script_path: str,
    model2_script_path: str,
    wait: bool = True,
    poll_interval: int = 60,
) -> tuple[int, int]:
    """
    Submit the two training SLURM scripts (model1 and model2) in parallel,
    then optionally wait for both to complete and check exit status.

    Args:
        model1_script_path: Path to model1_batch_training.sh.
        model2_script_path: Path to model2_batch_training.sh.
        wait: If True, block until both jobs have finished (default True).
        poll_interval: Seconds between squeue checks when waiting (default 60).

    Returns:
        (job_id_model1, job_id_model2).

    Raises:
        TrainingJobFailureError: If wait is True and either job failed.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_submit_sbatch, model1_script_path)
        f2 = executor.submit(_submit_sbatch, model2_script_path)
        job_id1 = f1.result()
        job_id2 = f2.result()
    print(f"Submitted model1 training job: {job_id1}")
    print(f"Submitted model2 training job: {job_id2}")
    if wait:
        print("Waiting for both training jobs to complete...")
        _wait_for_job(job_id1, poll_interval=poll_interval)
        _wait_for_job(job_id2, poll_interval=poll_interval)
        print("Both training jobs completed.")
        time.sleep(5)
        ok1 = _job_succeeded(job_id1)
        ok2 = _job_succeeded(job_id2)
        if not ok1 or not ok2:
            failed = (not ok1, not ok2)
            names = []
            if not ok1:
                names.append("model1")
            if not ok2:
                names.append("model2")
            raise TrainingJobFailureError(
                f"Training job(s) failed: {', '.join(names)} (job IDs: {job_id1}, {job_id2}). "
                "Check slurm .err/.out files for details.",
                job_ids=(job_id1, job_id2),
                failed_mask=failed,
            )
    return job_id1, job_id2
