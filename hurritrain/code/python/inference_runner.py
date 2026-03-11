"""
Run ACE inference via torchrun or by submitting batched SLURM jobs.
"""

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_inference(
    yaml_path: str,
    nproc_per_node: int = 1,
    master_port: int = 29500,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Run inference using the specified YAML config file.

    Args:
        yaml_path: Path to the YAML inference config file.
        nproc_per_node: Number of processes per node for torchrun.
        master_port: Port for distributed inference communication (default: 29500).
        python_executable: Path to Python executable (default: sys.executable).

    Returns:
        CompletedProcess from subprocess.run()
    """
    if python_executable is None:
        python_executable = sys.executable

    env = os.environ.copy()
    env["WANDB_JOB_TYPE"] = "inference"
    env["MASTER_PORT"] = str(master_port)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "-m",
        "fme.ace.inference",
        yaml_path,
    ]

    print(f"Running inference command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n{'='*60}")
        print(f"Inference failed with exit code {result.returncode}")
        print(f"{'='*60}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"{'='*60}\n")

    return result


def _submit_sbatch(script_path: str) -> int:
    """Submit a single script via sbatch; return job ID. Raises on failure."""
    script_path = os.path.abspath(script_path)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    cwd = os.path.dirname(script_path)
    result = subprocess.run(
        ["sbatch", os.path.basename(script_path)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed: {result.stderr or result.stdout or 'unknown error'}"
        )
    # Parse "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
    return int(match.group(1))


def _wait_for_job(job_id: int, poll_interval: int = 60) -> None:
    """Poll squeue until job job_id is no longer in the queue."""
    while True:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # squeue can fail if job is gone; treat as done
            return
        if not result.stdout.strip():
            return
        time.sleep(poll_interval)


def _job_succeeded(job_id: int) -> bool:
    """
    Return True if the SLURM job completed successfully (State COMPLETED and exit code 0).
    Uses sacct; may need a short delay after job ends for sacct to be updated.
    """
    result = subprocess.run(
        ["sacct", "-j", str(job_id), "-n", "-o", "State,ExitCode", "--parsable2"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    # --parsable2 is pipe-delimited; columns are State, ExitCode (JobID omitted with -o State,ExitCode)
    lines = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
    for line in lines:
        parts = line.replace("|", " ").split()
        if len(parts) >= 2:
            state, exitcode = parts[-2], parts[-1]
            # COMPLETED or CD, exit 0:0 or 0
            if state in ("COMPLETED", "CD") and (exitcode == "0:0" or exitcode == "0"):
                return True
            if state in ("FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT", "F", "CA", "OOM", "NF", "TO"):
                return False
        elif len(parts) == 1:
            if parts[0] not in ("COMPLETED", "CD"):
                return False
    return any("COMPLETED" in line or "|CD|" in line for line in lines)


class InferenceJobFailureError(Exception):
    """Raised when one or more submitted inference jobs failed (e.g. OOM, non-zero exit)."""

    def __init__(self, message: str, job_ids: tuple[int, int], failed_mask: tuple[bool, bool]):
        super().__init__(message)
        self.job_ids = job_ids
        self.failed_mask = failed_mask


def submit_batched_inference_jobs(
    model1_script_path: str,
    model2_script_path: str,
    wait: bool = True,
    poll_interval: int = 60,
) -> tuple[int, int]:
    """
    Submit the two batched inference SLURM scripts (model1 and model2) in parallel,
    then optionally wait for both to complete.

    Args:
        model1_script_path: Path to model1_batched_array_inference.sh.
        model2_script_path: Path to model2_batched_array_inference.sh.
        wait: If True, block until both jobs have finished (default True).
        poll_interval: Seconds between squeue checks when waiting (default 60).

    Returns:
        (job_id_model1, job_id_model2).
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_submit_sbatch, model1_script_path)
        f2 = executor.submit(_submit_sbatch, model2_script_path)
        job_id1 = f1.result()
        job_id2 = f2.result()
    print(f"Submitted model1 inference job: {job_id1}")
    print(f"Submitted model2 inference job: {job_id2}")
    if wait:
        print("Waiting for both jobs to complete...")
        _wait_for_job(job_id1, poll_interval=poll_interval)
        _wait_for_job(job_id2, poll_interval=poll_interval)
        print("Both inference jobs completed.")
        # Check exit status (sacct may need a moment to update)
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
            raise InferenceJobFailureError(
                f"Inference job(s) failed: {', '.join(names)} (job IDs: {job_id1}, {job_id2}). "
                "Check slurm .err/.out files for details.",
                job_ids=(job_id1, job_id2),
                failed_mask=failed,
            )
    return job_id1, job_id2
