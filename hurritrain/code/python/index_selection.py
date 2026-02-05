"""
Select random time indices from NetCDF data directory.
"""

import glob
import os

import numpy as np
import xarray as xr


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

    pattern = os.path.join(data_path, file_pattern)
    file_paths = sorted(glob.glob(pattern))

    if not file_paths:
        raise ValueError(f"No files found matching '{file_pattern}' in '{data_path}'")

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

    selected_indices = np.random.choice(
        total_time_indices, size=n, replace=False
    )
    selected_indices = np.sort(selected_indices)

    return selected_indices
