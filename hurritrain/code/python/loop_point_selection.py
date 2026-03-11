"""
Active learning loop: train on selected indices, run inference, compute acquisition, repeat.
"""

import glob
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from acquisition import (
    compute_min_pressure_variance_and_top_sample_indices,
    compute_min_pressure_variance_and_top_sample_indices_batched,
)
from index_selection import select_random_time_indices
from inference_runner import (
    InferenceJobFailureError,
    run_inference,
    submit_batched_inference_jobs,
)
from probability_data import load_probability_data
from training_runner import (
    TrainingJobFailureError,
    submit_batched_training_jobs,
)
from yaml_utils import (
    update_fast_inference_indices_for_both,
    update_yaml_training_indices,
)

torch.cuda.empty_cache()


def _exit_on_job_failure(e: Exception, msg_prefix: str = "Stopping") -> None:
    """Print message and exit with code 1 on training or inference job failure."""
    print(f"\n{'='*60}")
    print(f"{msg_prefix}: {e}")
    print(f"{'='*60}\n")
    raise SystemExit(1)


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

    # Clear existing inference output so this run starts fresh
    inference_output_dirs = [
        "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/inference_output/model1_inference",
        "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/inference_output/model2_inference",
    ]
    for d in inference_output_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("Cleared existing inference output directories.")

    px, py = load_probability_data(py_filename="kde_pres_1940.pkl")
    py_kde = py

    # Determine YAML file paths
    if yaml_dir is None:
        base_dir = Path(__file__).parent.parent / "yaml"
        yaml_dir = str(base_dir)
    
    training_yaml_files = [
        os.path.join(yaml_dir, "model1_train_initial.yaml"),
        os.path.join(yaml_dir, "model2_train_initial.yaml"),
    ]
    
    # Fast-inference YAMLs (one per model; start_indices updated once before loop)
    fast_inference_yaml_files = [
        os.path.join(yaml_dir, "model1_fast_inference.yaml"),
        os.path.join(yaml_dir, "model2_fast_inference.yaml"),
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

    # ---- FIX LATER -----
    # change total indices to 100 as a test
    total_time_indices = 100

    # Update fast-inference YAMLs with start_indices [0, 1, ..., total_time_indices-1]
    update_fast_inference_indices_for_both(
        fast_inference_yaml_files[0],
        fast_inference_yaml_files[1],
        total_time_indices,
    )
    print(f"Updated fast-inference YAMLs with {total_time_indices} start indices.")

    # Before loop: Update YAML files with initial indices
    print("Updating YAML files with initial training indices...")
    for training_yaml_file in training_yaml_files:
        if not os.path.exists(training_yaml_file):
            print(f"Warning: YAML file {training_yaml_file} does not exist, skipping...")
            continue
        update_yaml_training_indices(training_yaml_file, training_indices_list, total_time_indices)
        print(f"Updated {training_yaml_file}")

    # Run training with initial random points (submit SLURM jobs for model1 and model2)
    print("\nRunning training with initial random points...")
    valid_training_yaml_files = [f for f in training_yaml_files if os.path.exists(f)]
    valid_inference_yaml_files = [f for f in fast_inference_yaml_files if os.path.exists(f)]

    if not valid_training_yaml_files:
        print("Warning: No valid YAML files found for training")
    if not valid_inference_yaml_files:
        print("Warning: No valid YAML files found for inference")
    else:
        shell_dir = Path(__file__).parent.parent / "shell"
        model1_train_script = shell_dir / "model1_batch_training.sh"
        model2_train_script = shell_dir / "model2_batch_training.sh"
        if not model1_train_script.exists() or not model2_train_script.exists():
            print("Warning: Training batch scripts not found; skipping training.")
        else:
            try:
                t0 = time.time()
                submit_batched_training_jobs(
                    str(model1_train_script),
                    str(model2_train_script),
                    wait=True,
                )
                elapsed = time.time() - t0
                print(f"Initial training jobs finished in {elapsed/60:.1f} min ({elapsed:.0f} s).")
            except (InferenceJobFailureError, TrainingJobFailureError) as e:
                _exit_on_job_failure(e, "Stopping")
    

    


    # Main loop
    print(f"\nStarting main loop with {n_iterations} iterations...")
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")
        print(f"Current indices list: {training_indices_list}")

        # Run inference: submit model1 and model2 fast-inference SLURM jobs in parallel
        shell_dir = Path(__file__).parent.parent / "shell"
        model1_script = shell_dir / "model1_batch_fast_inference.sh"
        model2_script = shell_dir / "model2_batch_fast_inference.sh"
        if not model1_script.exists() or not model2_script.exists():
            print("Warning: Fast inference scripts not found; skipping inference.")
        else:
            print("Submitting fast inference jobs (model1 and model2) in parallel...")
            try:
                t0 = time.time()
                submit_batched_inference_jobs(
                    str(model1_script),
                    str(model2_script),
                    wait=True,
                )
                elapsed = time.time() - t0
                print(f"Inference jobs finished in {elapsed/60:.1f} min ({elapsed:.0f} s).")
            except (InferenceJobFailureError, TrainingJobFailureError) as e:
                _exit_on_job_failure(e, "Stopping loop")

        # Compute acquisition from inference zarr outputs (one zarr per model)
        inference_output_dir = Path(__file__).parent.parent.parent / "inference_output"
        zarr1 = inference_output_dir / "model1_inference" / "autoregressive_predictions.zarr"
        zarr2 = inference_output_dir / "model2_inference" / "autoregressive_predictions.zarr"
        has_zarr1 = zarr1.is_dir()
        has_zarr2 = zarr2.is_dir()
        if has_zarr1 and has_zarr2:
            try:
                top_candidate_indices, acquisition = compute_min_pressure_variance_and_top_sample_indices_batched(
                    inference_output_dir=str(inference_output_dir),
                    py_kde=py_kde,
                    candidate_time_indices=np.arange(total_time_indices),
                    training_indices=training_indices_list,
                    variable="PRESsfc",
                    lat_min=22.0,
                    lat_max=29.0,
                    lon_min=263.0,
                    lon_max=277.0,
                    n_top=2,
                )
                high_variance_sample_indices = top_candidate_indices.tolist()
                print(f"Candidate indices with 2 highest acquisition (zarr): {high_variance_sample_indices}")
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"Stopping loop: Acquisition from zarr failed: {e}")
                print(f"{'='*60}\n")
                raise SystemExit(1)
        else:
            high_variance_sample_indices = []
            print("Warning: Inference zarr stores missing (model1_inference/ and model2_inference/autoregressive_predictions.zarr); skipping acquisition.")

        # Clear inference output only after acquisition succeeded, so next iteration gets fresh zarrs
        inference_output_dirs = [
            "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/inference_output/model1_inference",
            "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/inference_output/model2_inference",
        ]
        for d in inference_output_dirs:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        # TODO: User may add: map high_variance_sample_indices to time indices, add to
        # training_indices_list, update YAML files, and optionally run training again.
        training_indices_list.extend(high_variance_sample_indices)

        print(f"Updated training indices list: {training_indices_list}")
        for training_yaml_file in training_yaml_files:
            if not os.path.exists(training_yaml_file):
                print(f"Warning: YAML file {training_yaml_file} does not exist, skipping...")
                continue
            update_yaml_training_indices(training_yaml_file, training_indices_list, total_time_indices)
            print(f"Updated {training_yaml_file}")

        # Run training with new indices (submit SLURM jobs for model1 and model2)
        print("\nRunning training with new indices...")
        shell_dir = Path(__file__).parent.parent / "shell"
        model1_train_script = shell_dir / "model1_batch_training.sh"
        model2_train_script = shell_dir / "model2_batch_training.sh"
        if not model1_train_script.exists() or not model2_train_script.exists():
            print("Warning: Training batch scripts not found; skipping training.")
        else:
            try:
                t0 = time.time()
                submit_batched_training_jobs(
                    str(model1_train_script),
                    str(model2_train_script),
                    wait=True,
                )
                elapsed = time.time() - t0
                print(f"Training jobs finished in {elapsed/60:.1f} min ({elapsed:.0f} s).")
            except (InferenceJobFailureError, TrainingJobFailureError) as e:
                _exit_on_job_failure(e, "Stopping loop")

        # Main loop end
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations} completed")
        print(f"{'='*60}")
        print(f"Current training indices list: {training_indices_list}")
        candidate_indices_list = np.setdiff1d(
            np.arange(total_time_indices - 1), np.unique(training_indices_list)
        )
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