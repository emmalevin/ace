"""
Run ACE inference via torchrun or by submitting batched SLURM jobs.
"""

import os
import re
import subprocess
import sys
import time


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


def _submit_sbatch(
    script_path: str,
    exports: dict[str, object] | None = None,
    sbatch_args: list[str] | None = None,
) -> int:
    """Submit a single script via sbatch; return job ID. Raises on failure."""
    script_path = os.path.abspath(script_path)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = ["sbatch", "--parsable"]
    if exports:
        export_values = ["ALL"]
        export_values.extend(f"{key}={value}" for key, value in exports.items())
        cmd.append(f"--export={','.join(export_values)}")
    if sbatch_args:
        cmd.extend(sbatch_args)
    cmd.append(os.path.basename(script_path))

    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(script_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed: {result.stderr or result.stdout or 'unknown error'}"
        )

    # --parsable returns "12345" or "12345;cluster".
    match = re.search(r"\d+", result.stdout.strip())
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
    return int(match.group(0))


def _paired_model_exports(
    model_yaml_paths: tuple[str, str],
    extra_exports: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build exported environment for one batch job that runs both models."""
    exports: dict[str, object] = {
        "YAML_FILE_MODEL1": os.path.abspath(model_yaml_paths[0]),
        "YAML_FILE_MODEL2": os.path.abspath(model_yaml_paths[1]),
    }
    if extra_exports:
        exports.update(extra_exports)
    return exports


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
    """Raised when a submitted inference job failed (e.g. OOM, non-zero exit)."""

    def __init__(self, message: str, job_ids: tuple[int, ...], failed_mask: tuple[bool, ...]):
        super().__init__(message)
        self.job_ids = job_ids
        self.failed_mask = failed_mask


def submit_batched_inference_jobs(
    batch_script_path: str,
    model_yaml_paths: tuple[str, str],
    wait: bool = True,
    poll_interval: int = 60,
    profile_models: tuple[bool, bool] = (False, False),
    nproc_per_model: int = 1,
) -> int:
    """
    Submit one inference SLURM job that runs model1 and model2 in parallel,
    then optionally wait for completion.

    Args:
        batch_script_path: Path to batch_fast_inference.sh.
        model_yaml_paths: (model1_yaml, model2_yaml).
        wait: If True, block until the paired job has finished (default True).
        poll_interval: Seconds between squeue checks when waiting (default 60).
        profile_models: Enable nsys profiling per model.
        nproc_per_model: Torch processes per model; inference defaults to one GPU each.

    Returns:
        Paired inference Slurm job ID.
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
        sbatch_args=["--job-name=ace_inf_pair"],
    )
    print(f"Submitted paired inference job: {job_id}")
    if wait:
        print("Waiting for paired inference job to complete...")
        _wait_for_job(job_id, poll_interval=poll_interval)
        print("Paired inference job completed.")
        time.sleep(5)
        if not _job_succeeded(job_id):
            raise InferenceJobFailureError(
                f"Inference job failed (job ID: {job_id}). Check slurm .err/.out files for details.",
                job_ids=(job_id,),
                failed_mask=(True,),
            )
    return job_id
