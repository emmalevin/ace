"""
Submit ACE training as one SLURM batch job that runs both models in parallel.
"""

import time
from inference_runner import (
    _job_succeeded,
    _paired_model_exports,
    _submit_sbatch,
    _wait_for_job,
)


class TrainingJobFailureError(Exception):
    """Raised when the submitted training job failed (e.g. OOM, non-zero exit)."""

    def __init__(self, message: str, job_ids: tuple[int, ...], failed_mask: tuple[bool, ...]):
        super().__init__(message)
        self.job_ids = job_ids
        self.failed_mask = failed_mask


def submit_batched_training_jobs(
    batch_script_path: str,
    model_yaml_paths: tuple[str, str],
    wait: bool = True,
    poll_interval: int = 60,
    profile_models: tuple[bool, bool] = (False, False),
    nproc_per_model: int = 2,
) -> int:
    """
    Submit one training SLURM job that runs model1 and model2 in parallel,
    then optionally wait for completion and check exit status.

    Args:
        batch_script_path: Path to batch_training.sh.
        model_yaml_paths: (model1_yaml, model2_yaml).
        wait: If True, block until the paired job has finished (default True).
        poll_interval: Seconds between squeue checks when waiting (default 60).
        profile_models: Enable nsys profiling per model.
        nproc_per_model: Torch processes per model; training defaults to two GPUs each.

    Returns:
        Paired training Slurm job ID.

    Raises:
        TrainingJobFailureError: If wait is True and the paired job failed.
    """
    exports = _paired_model_exports(
        model_yaml_paths,
        {
            "PROFILE_MODEL1": int(profile_models[0]),
            "PROFILE_MODEL2": int(profile_models[1]),
            "NPROC_PER_MODEL": nproc_per_model,
        },
    )
    job_id = _submit_sbatch(
        batch_script_path,
        exports=exports,
        sbatch_args=["--job-name=ace_train_pair"],
    )
    print(f"Submitted paired training job: {job_id}")
    if wait:
        print("Waiting for paired training job to complete...")
        _wait_for_job(job_id, poll_interval=poll_interval)
        print("Paired training job completed.")
        time.sleep(5)
        if not _job_succeeded(job_id):
            raise TrainingJobFailureError(
                f"Training job failed (job ID: {job_id}). Check slurm .err/.out files for details.",
                job_ids=(job_id,),
                failed_mask=(True,),
            )
    return job_id
