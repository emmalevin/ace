import glob
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence
import numpy as np
import xarray as xr
import yaml
import torch
torch.cuda.empty_cache()


# 0. load px, py
def load_probability_data(
    probability_data_dir: str = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data",
    px_filename: str = "eof_kde_h500_modes3.pkl",
    py_filename: str = "kde_pdf_PRESsfc.pkl",
) -> tuple[dict, dict]:
    """
    Load the KDE probability density function files for px and py.
    
    Args:
        probability_data_dir: Directory containing the .pkl files.
        px_filename: Filename for px (EOF KDE, default: "eof_kde_h500_modes3.pkl").
        py_filename: Filename for py (variable KDE, default: "kde_pdf_PRESsfc.pkl").
    
    Returns:
        tuple: (px, py)
            - px: Dictionary containing 'kde' (gaussian_kde object) and 'eof_results' (dict)
            - py: Dictionary containing 'kde' (gaussian_kde object) and 'mean_values' (np.ndarray)
    
    Example:
        px, py = load_probability_data()
        px_kde = px['kde']
        px_eof_results = px['eof_results']
        py_kde = py['kde']
        py_mean_values = py['mean_values']
    """
    px_path = os.path.join(probability_data_dir, px_filename)
    py_path = os.path.join(probability_data_dir, py_filename)
    
    # Check if files exist
    if not os.path.exists(px_path):
        raise FileNotFoundError(
            f"px file not found: {px_path}\n"
            f"Please run px_x.py to generate this file first."
        )
    if not os.path.exists(py_path):
        raise FileNotFoundError(
            f"py file not found: {py_path}\n"
            f"Please run py_y.py to generate this file first."
        )
    
    # Load px (EOF KDE)
    print(f"Loading px from {px_path}...")
    with open(px_path, "rb") as f:
        px = pickle.load(f)
    
    # Load py (observable KDE)
    print(f"Loading py from {py_path}...")
    with open(py_path, "rb") as f:
        py = pickle.load(f)
    
    print("Probability data loaded successfully!")
    print(f"  px keys: {list(px.keys())}")
    print(f"  py keys: {list(py.keys())}")
    
    return px, py



# 1. pick initial training points

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



# 2. begin loop




# 3. train the models using the selected points


def update_yaml_training_indices(
    yaml_path: str,
    training_indices: list[int] | np.ndarray,
    total_time_indices: int | None = None,
) -> None:
    """
    Update the indices in a YAML training config file.
    
    Args:
        yaml_path: Path to the YAML config file.
        indices: List of starting indices for each group (e.g., [0, 100]).
                 Each index will be used to create a group of 2 consecutive indices
                 (e.g., 0 -> [0, 1], 100 -> [100, 101]).
        total_time_indices: Optional total number of time indices for validation.
    """
    # Convert numpy array to list if needed
    if isinstance(training_indices, np.ndarray):
        training_indices = training_indices.tolist()
    
    # Validate indices don't exceed available time
    # if total_time_indices is not None:
    #     max_idx = max(training_indices)
    #     if max_idx >= total_time_indices:
    #         raise ValueError(
    #             f"Index {max_idx} exceeds available time indices ({total_time_indices})"
    #         )
    
    # Read YAML file
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check if config was loaded successfully
    if config is None:
        raise ValueError(f"YAML file {yaml_path} is empty or could not be parsed")
    
    # Navigate to train_loader.dataset.subset
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
    
    # Update subset with flat list of starting indices
    config["train_loader"]["dataset"]["subset"] = training_indices
    
    # Write back to file
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_yaml_inference_indices(
    yaml_path: str,
    total_time_indices: int,
) -> None:
    # Read YAML file
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Check if config was loaded successfully
    if config is None:
        raise ValueError(f"YAML file {yaml_path} is empty or could not be parsed")

    if config["initial_condition"] is None:
        raise ValueError(f"YAML file {yaml_path} does not have 'initial_condition' section")
    if "start_indices" not in config["initial_condition"]:
        raise ValueError(
            f"YAML file {yaml_path} does not have 'initial_condition.start_indices' section"
        )
    config["initial_condition"]["start_indices"]["n_initial_conditions"] = total_time_indices
    config["initial_condition"]["start_indices"]["first"] = 0
    config["initial_condition"]["start_indices"]["interval"] = 1

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
    
    # Set environment variables for better error reporting
    env = os.environ.copy()
    env["WANDB_JOB_TYPE"] = "training"
    env["MASTER_PORT"] = str(master_port)
    # Enable verbose error output for PyTorch distributed
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    env["TORCH_LOGS"] = "+dynamo"
    
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
    # Capture both stdout and stderr to see the actual error
    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    
    # Print output if there was an error
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


def run_inference(yaml_path: str,
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
    """
    if python_executable is None:
        python_executable = sys.executable
    
    # Set environment variables for better error reporting
    env = os.environ.copy()
    env["WANDB_JOB_TYPE"] = "inference"
    env["MASTER_PORT"] = str(master_port)
    
    # Build command
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        "-m",
        "fme.ace.inference",
        yaml_path,
    ]
    
    print(f"Running inference command: {' '.join(cmd)}")
    # Capture both stdout and stderr to see the actual error
    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    
    # Print output if there was an error
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


def compute_min_pressure_variance_and_top_sample_indices(
    prediction_paths: Sequence[str],
    candidate_indices_list: np.ndarray | list[int],
    variable: str = "PRESsfc",
    lat_min: float = 22.0,
    lat_max: float = 29.0,
    lon_min: float = 263.0,
    lon_max: float = 277.0,
    n_top: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Open autoregressive_predictions.nc from each model, compute min surface
    pressure in the Gulf lat/lon box per sample, then variance across models
    per sample; return the n_top sample indices with highest variance.

    Uses the same Gulf box as py_y (lat 22--29 N, lon 97--83 W in 0--360).

    Args:
        prediction_paths: Paths to autoregressive_predictions.nc (one per model).
        variable: Variable name (default PRESsfc).
        lat_min, lat_max, lon_min, lon_max: Gulf box in degrees (defaults from py_y).
        n_top: Number of sample indices to return with highest variance (default 2).

    Returns:
        sample_indices: 1D array of length n_top (sample indices with highest variance).
        variances: 1D array of variance per sample (same length as sample dimension).
    """
    if len(prediction_paths) < 2:
        raise ValueError("Need at least 2 prediction files to compute variance across models")
    min_pressure_per_model = []
    for path in prediction_paths:
        with xr.open_dataset(path, decode_times=False) as ds:
            var = ds[variable]
            # Select Gulf box (same as py_y)
            box = var.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )
            # Min over lat, lon, and time -> one value per sample (first dim is sample)
            sample_dim = box.dims[0]
            min_per_sample = box.min(dim=[d for d in box.dims if d != sample_dim])
            min_pressure_per_model.append(min_per_sample.values)
    # Stack: (n_samples, n_models)
    stacked = np.stack(min_pressure_per_model, axis=-1)
    # Variance across models for each sample (axis=-1)
    variances = np.var(stacked, axis=-1)
    # Map variances back to candidate indices
    variances = variances[candidate_indices_list]
    # Indices of n_top largest variances
    top_indices = np.argsort(variances)[-n_top:][::-1]
    return candidate_indices_list[top_indices], variances[top_indices]


def main_loop(
    n_iterations: int = 1,
    n_initial_indices: int = 10,
    data_path: str = "/scratch/gpfs/GVECCHI/el2358/ace/training_data",
    yaml_dir: str | None = None,
    seed: int | None = None,
):
    """
    Main loop for active learning algorithm.
    
    Args:
        n_iterations: Number of iterations to run (default: 50).
        n_initial_indices: Number of initial random indices to select (default: 10).
        data_path: Path to training data directory.
        yaml_dir: Directory containing YAML files. If None, uses default location.
        seed: Random seed for initial index selection.
    """
    # Determine YAML file paths
    if yaml_dir is None:
        base_dir = Path(__file__).parent.parent / "yaml"
        yaml_dir = str(base_dir)
    
    training_yaml_files = [
        os.path.join(yaml_dir, "model1_train_initial.yaml"),
        os.path.join(yaml_dir, "model2_train_initial.yaml"),
    ]
    
    inference_yaml_files = [
        os.path.join(yaml_dir, "model1_inference.yaml"),
        os.path.join(yaml_dir, "model2_inference.yaml"),
    ]

    # Before loop: Generate initial list of indices
    print(f"Generating initial list of {n_initial_indices} random time indices...")
    initial_indices = select_random_time_indices(
        data_path=data_path, n=n_initial_indices, seed=seed
    )
    print(f"Initial selected indices: {initial_indices}")
    
    # Convert to list to maintain throughout iterations
    training_indices_list = initial_indices.tolist()

    # Get total time indices for validation
    pattern = os.path.join(data_path, "*.nc")
    file_paths = sorted(glob.glob(pattern))
    total_time_indices = 0
    for file_path in file_paths:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            total_time_indices += len(ds.time)


    #TEMPORARY
    total_time_indices = 10

    # Make candidate indices list
    candidate_indices_list = np.arange(total_time_indices-1)
    candidate_indices_list = np.setdiff1d(candidate_indices_list, training_indices_list)

    # Before loop: Update YAML files with initial indices
    print("Updating YAML files with initial training indices...")
    for training_yaml_file in training_yaml_files:
        if not os.path.exists(training_yaml_file):
            print(f"Warning: YAML file {training_yaml_file} does not exist, skipping...")
            continue
        update_yaml_training_indices(training_yaml_file, training_indices_list, total_time_indices)
        print(f"Updated {training_yaml_file}")
    
    print("Updating YAML files with initial inference indices...")
    for inference_yaml_file in inference_yaml_files:
        if not os.path.exists(inference_yaml_file):
            print(f"Warning: YAML file {inference_yaml_file} does not exist, skipping...")
            continue
        update_yaml_inference_indices(inference_yaml_file, total_time_indices-1)
        print(f"Updated {inference_yaml_file}")

    # Run training with initial random points
    print("\nRunning training with initial random points...")
    valid_training_yaml_files = [f for f in training_yaml_files if os.path.exists(f)]
    valid_inference_yaml_files = [f for f in inference_yaml_files if os.path.exists(f)]
    
    if not valid_training_yaml_files:
        print("Warning: No valid YAML files found for training")
    if not valid_inference_yaml_files:
        print("Warning: No valid YAML files found for inference")
    else:
        # Assign unique ports to each training job to avoid conflicts
        base_port = 29500
        port_to_file = {
            base_port + i: yaml_file
            for i, yaml_file in enumerate(valid_training_yaml_files)
        }
        
        # Run training in parallel using ThreadPoolExecutor
        print(f"Running {len(valid_training_yaml_files)} training jobs in parallel...")
        with ThreadPoolExecutor(max_workers=len(valid_training_yaml_files)) as executor:
            # Submit all training tasks with unique ports
            future_to_file = {
                executor.submit(run_training, yaml_file, 1, port): yaml_file
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
    

    


    # Main loop
    print(f"\nStarting main loop with {n_iterations} iterations...")
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")
        print(f"Current indices list: {training_indices_list}")

        # run inference on all points
        print("Running inference on all points...")
        for inference_yaml_file in valid_inference_yaml_files:
            run_inference(inference_yaml_file, 1, base_port + len(valid_training_yaml_files))
            print(f"Inference with {inference_yaml_file} completed successfully")

        # Open inference outputs and compute acquisition (variance of min PRES in Gulf)
        inference_output_dir = Path(__file__).parent.parent.parent / "inference_output"
        prediction_paths = [
            str(inference_output_dir / "model1_inference" / "autoregressive_predictions.nc"),
            str(inference_output_dir / "model2_inference" / "autoregressive_predictions.nc"),
        ]
        if all(os.path.exists(p) for p in prediction_paths):
            top_sample_indices, variances = compute_min_pressure_variance_and_top_sample_indices(
                prediction_paths,
                candidate_indices_list,
                variable="PRESsfc",
                lat_min=22.0,
                lat_max=29.0,
                lon_min=263.0,
                lon_max=277.0,
                n_top=2,
            )
            # Keep list of the 2 sample indices with highest variance (this iteration)
            high_variance_sample_indices = top_sample_indices.tolist()
            print(f"Sample indices with 2 highest variances (min PRES Gulf): {high_variance_sample_indices}")
        else:
            high_variance_sample_indices = []
            print("Warning: one or both autoregressive_predictions.nc missing; skipping acquisition.")

        # TODO: User may add: map high_variance_sample_indices to time indices, add to
        # training_indices_list, update YAML files, and optionally run training again.
        training_indices_list.extend(high_variance_sample_indices)
        candidate_indices_list = np.setdiff1d(candidate_indices_list, high_variance_sample_indices)

        print(f"Updated training indices list: {training_indices_list}")
        for training_yaml_file in training_yaml_files:
            if not os.path.exists(training_yaml_file):
                print(f"Warning: YAML file {training_yaml_file} does not exist, skipping...")
                continue
            update_yaml_training_indices(training_yaml_file, training_indices_list, total_time_indices)
            print(f"Updated {training_yaml_file}")

        # Run training with new indices
        print("\nRunning training with new indices...")
        valid_training_yaml_files = [f for f in training_yaml_files if os.path.exists(f)]
        if not valid_training_yaml_files:
            print("Warning: No valid YAML files found for training")
        else:
            # Assign unique ports to each training job to avoid conflicts
            base_port = 29500
            port_to_file = {
                base_port + i: yaml_file
                for i, yaml_file in enumerate(valid_training_yaml_files)
            }
            
            # Run training in parallel using ThreadPoolExecutor
            print(f"Running {len(valid_training_yaml_files)} training jobs in parallel...")
            with ThreadPoolExecutor(max_workers=len(valid_training_yaml_files)) as executor:
                # Submit all training tasks with unique ports
                future_to_file = {
                    executor.submit(run_training, yaml_file, 1, port): yaml_file
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

        # Main loop end
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations} completed")
        print(f"{'='*60}")
        print(f"Current training indices list: {training_indices_list}")
        print(f"Current candidate indices list: {candidate_indices_list}")
        print(f"{'='*60}\n")

# 4. run inference on each point then compute the acquisition function for each candidate point 
# 5. select the point(s) with the highest acquisition function
# 6. add the selected point(s) to the training dataset
# 7. repeat steps 3-6 until the desired number of training points is reached

#.sh files: 1 (have), 3, 4/5

# For a given iteration of the algorithm

# list of time indices already in the training dataset 

# list of 10 highest acquisition function values
# list of time indices associated with the 10 highest acquisition function values

# loop through each possible point (EXCEPT THOSE ALREADY IN THE TRAINING DATASET) 
    # run inference on the point using the trained model
    # compute the acquisition function
    # if the acquisition function value is greater than the 10th highest value, add the time index to the list and remove the lowest value
    # making sure list always has 10 values


# then I have my list of 10 highest acquisition function values and the list of time indices associated with them
# add these points to the list of points in the training dataset 


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run active learning loop")
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=3,
        help="Number of iterations to run (default: 50)",
    )
    parser.add_argument(
        "--n_initial_indices",
        type=int,
        default=2,
        help="Number of initial random indices to select (default: 10)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/scratch/gpfs/GVECCHI/el2358/ace/training_data",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--yaml_dir",
        type=str,
        default=None,
        help="Directory containing YAML files. If not provided, uses default location.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initial index selection (optional)",
    )
    
    args = parser.parse_args()
    
    main_loop(
        n_iterations=args.n_iterations,
        n_initial_indices=args.n_initial_indices,
        data_path=args.data_path,
        yaml_dir=args.yaml_dir,
        seed=args.seed,
    )