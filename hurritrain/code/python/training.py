"""
Run ACE training (single or parallel) via torchrun.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence


def run_training(
    yaml_path: str,
    nproc_per_node: int = 1,
    master_port: int = 29500,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Run training using the specified YAML config file.

    Args:
        yaml_path: Path to the YAML training config file.
        nproc_per_node: Number of processes per node for torchrun.
        master_port: Port for distributed training communication (default: 29500).
        python_executable: Path to Python executable (default: sys.executable).

    Returns:
        CompletedProcess from subprocess.run()
    """
    if python_executable is None:
        python_executable = sys.executable

    env = os.environ.copy()
    env["WANDB_JOB_TYPE"] = "training"
    env["MASTER_PORT"] = str(master_port)
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    env["TORCH_LOGS"] = "+dynamo"

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "-m",
        "fme.ace.train",
        yaml_path,
    ]

    print(f"Running training command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n{'='*60}")
        print(f"Training failed with exit code {result.returncode}")
        print(f"{'='*60}")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"{'='*60}\n")

    return result


def run_training_parallel(
    training_yaml_files: Sequence[str],
    base_port: int = 29500,
) -> None:
    """
    Run training for each YAML file in parallel using ThreadPoolExecutor.
    Assigns unique ports to avoid conflicts. Prints warnings on failure.
    """
    if not training_yaml_files:
        print("Warning: No valid YAML files found for training")
        return
    port_to_file = {
        base_port + i: yaml_file
        for i, yaml_file in enumerate(training_yaml_files)
    }
    print(f"Running {len(training_yaml_files)} training jobs in parallel...")
    with ThreadPoolExecutor(max_workers=len(training_yaml_files)) as executor:
        future_to_file = {
            executor.submit(run_training, yaml_file, 1, port): yaml_file
            for port, yaml_file in port_to_file.items()
        }
        for future in as_completed(future_to_file):
            yaml_file = future_to_file[future]
            try:
                result = future.result()
                if result.returncode != 0:
                    print(f"Warning: Training with {yaml_file} exited with code {result.returncode}")
                else:
                    print(f"Training with {yaml_file} completed successfully")
            except Exception as exc:
                print(f"Training with {yaml_file} generated an exception: {exc}")
