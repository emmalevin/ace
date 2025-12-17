
'''
1. Select initial random points for first iteration.
2. train the model using these points.
3. Compute the acquisition function for each point.
4. Select the point with the highest acquisition function.
5. Begin loop of training the model using the selected points.
'''

import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
import yaml


def select_random_time_indices(
    data_path: str = "/scratch/gpfs/GVECCHI/el2358/ace/training_data",
    file_pattern: str = "*.nc",
    n: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """
    Determine the total number of time indices across all netCDF files in a directory
    and randomly select n of them.

    Args:
        data_path: Path to directory containing netCDF files.
        file_pattern: Glob pattern to match files (default: "*.nc").
        n: Number of random time indices to select (default: 10).
        seed: Random seed for reproducibility (optional).

    Returns:
        Array of n randomly selected time indices.

    Example:
        If there are 2 files with monthly 6-hourly data (~30 days * 4 per day * 2 files),
        there should be ~240 time indices total. This function will randomly select n of them.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get all file paths matching the pattern
    pattern = os.path.join(data_path, file_pattern)
    file_paths = sorted(glob.glob(pattern))

    if not file_paths:
        raise ValueError(f"No files found matching '{file_pattern}' in '{data_path}'")

    # Open all files and get time dimension length from each
    total_time_indices = 0
    for file_path in file_paths:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            if "time" not in ds.dims:
                raise ValueError(f"File {file_path} does not have a 'time' dimension")
            total_time_indices += len(ds.time)

    if total_time_indices == 0:
        raise ValueError("No time indices found in any files")

    if n > total_time_indices:
        raise ValueError(
            f"Requested {n} indices but only {total_time_indices} available"
        )

    # Randomly select n indices from [0, total_time_indices)
    selected_indices = np.random.choice(
        total_time_indices, size=n, replace=False
    )
    selected_indices = np.sort(selected_indices)  # Sort for convenience

    return selected_indices


def create_index_groups(
    selected_indices: np.ndarray, group_size: int = 3
) -> list[list[int]]:
    """
    Create groups of consecutive indices from selected indices.
    
    Args:
        selected_indices: Array of selected time indices.
        group_size: Number of consecutive indices per group (default: 3).
    
    Returns:
        List of index groups, each containing group_size consecutive indices.
    
    Example:
        If selected_indices = [77, 159], returns [[77, 78, 79], [159, 160, 161]]
    """
    groups = []
    for idx in selected_indices:
        group = [int(idx + i) for i in range(group_size)]
        groups.append(group)
    return groups


def update_yaml_indices(
    yaml_path: str,
    index_groups: list[list[int]],
    total_time_indices: int | None = None,
) -> None:
    """
    Update the indices in a YAML training config file.
    
    Args:
        yaml_path: Path to the YAML config file.
        index_groups: List of index groups to set (e.g., [[77, 78, 79], [159, 160, 161]]).
        total_time_indices: Optional total number of time indices for validation.
    """
    # Validate indices don't exceed available time
    if total_time_indices is not None:
        max_idx = max(max(group) for group in index_groups)
        if max_idx >= total_time_indices:
            raise ValueError(
                f"Index {max_idx} exceeds available time indices ({total_time_indices})"
            )
    
    # Read YAML file
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check if config was loaded successfully
    if config is None:
        raise ValueError(f"YAML file {yaml_path} is empty or could not be parsed")
    
    # Navigate to train_loader.dataset.subset.indices
    if "train_loader" not in config:
        raise ValueError(f"YAML file {yaml_path} does not have 'train_loader' section")
    if "dataset" not in config["train_loader"]:
        raise ValueError(
            f"YAML file {yaml_path} does not have 'train_loader.dataset' section"
        )
    if "subset" not in config["train_loader"]["dataset"]:
        raise ValueError(
            f"YAML file {yaml_path} does not have 'train_loader.dataset.subset' section"
        )
    
    # Update indices
    config["train_loader"]["dataset"]["subset"]["indices"] = index_groups
    
    # Write back to file
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


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
    
    # Set environment variable
    env = os.environ.copy()
    env["WANDB_JOB_TYPE"] = "training"
    env["MASTER_PORT"] = str(master_port)
    
    # Build command
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "-m",
        "fme.ace.train",
        yaml_path,
    ]
    
    print(f"Running training command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, check=False)
    return result


def main(
    n_indices: int = 2,
    group_size: int = 3,
    yaml_files: list[str] | None = None,
    data_path: str = "/scratch/gpfs/GVECCHI/el2358/ace/training_data",
    seed: int | None = None,
    run_training_flag: bool = False,
    nproc_per_node: int = 1,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Main workflow: select random indices, update YAML files, and optionally run training.
    
    Args:
        n_indices: Number of random time indices to select.
        group_size: Number of consecutive indices per group.
        yaml_files: List of YAML file paths to update. If None, uses default files.
        data_path: Path to training data directory.
        seed: Random seed for reproducibility.
        run_training_flag: If True, run training after updating YAML files.
        nproc_per_node: Number of processes per node for training.
    
    Returns:
        Tuple of (selected_indices, index_groups)
    """
    # Default YAML files
    if yaml_files is None:
        base_dir = Path(__file__).parent.parent / "yaml"
        yaml_files = [
            str(base_dir / "model1_train_initial.yaml"),
            str(base_dir / "model2_train_initial.yaml"),
        ]
    
    # Step 1: Select random time indices
    print(f"Selecting {n_indices} random time indices...")
    selected_indices = select_random_time_indices(
        data_path=data_path, n=n_indices, seed=seed
    )
    print(f"Selected indices: {selected_indices}")
    
    # Step 2: Create index groups
    print(f"Creating index groups of size {group_size}...")
    index_groups = create_index_groups(selected_indices, group_size=group_size)
    print(f"Index groups: {index_groups}")
    
    # Step 3: Get total time indices for validation
    pattern = os.path.join(data_path, "*.nc")
    file_paths = sorted(glob.glob(pattern))
    total_time_indices = 0
    for file_path in file_paths:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            total_time_indices += len(ds.time)
    
    # Step 4: Update YAML files
    for yaml_file in yaml_files:
        if not os.path.exists(yaml_file):
            print(f"Warning: YAML file {yaml_file} does not exist, skipping...")
            continue
        print(f"Updating {yaml_file}...")
        update_yaml_indices(yaml_file, index_groups, total_time_indices)
        print(f"Updated {yaml_file} with indices: {index_groups}")
    
    # Step 5: Optionally run training in parallel
    if run_training_flag:
        # Filter out non-existent files
        valid_yaml_files = [f for f in yaml_files if os.path.exists(f)]
        
        if not valid_yaml_files:
            print("No valid YAML files found for training")
        else:
            print(f"\nRunning training in parallel for {len(valid_yaml_files)} YAML file(s)...")
            
            # Assign unique ports to each training job to avoid conflicts
            # Start from port 29500 and increment by 1 for each job
            base_port = 29500
            port_to_file = {
                base_port + i: yaml_file
                for i, yaml_file in enumerate(valid_yaml_files)
            }
            
            # Run training in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(valid_yaml_files)) as executor:
                # Submit all training tasks with unique ports
                future_to_file = {
                    executor.submit(run_training, yaml_file, nproc_per_node, port): yaml_file
                    for port, yaml_file in port_to_file.items()
                }
                
                # Wait for all tasks to complete and collect results
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
    
    return selected_indices, index_groups


# Example usage:
if __name__ == "__main__":
    # Select 2 random indices and update YAML files
    selected, groups = main(
        n_indices=1,
        group_size=3,
        seed=42,
        run_training_flag=True,  # Set to True to run training
        nproc_per_node=1,
    )
    print(f"\nSelected indices: {selected}")
    print(f"Index groups: {groups}")