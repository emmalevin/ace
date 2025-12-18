'''
Code to compute the KDE approximated PDF of the global atmospheric state (the EOF of the 500 hPa geopotential height).
To do so we use all the training data.
'''

import glob
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde


def compute_eof_kde_pdf(
    data_dir: str = "/scratch/gpfs/GVECCHI/el2358/ace/training_data",
    variable: str = "h500",
    n_modes: int = 3,
    output_path: Optional[str] = None,
) -> tuple[gaussian_kde, dict]:
    """
    Compute EOFs of a variable globally and create a 3D KDE PDF of the first n_modes.

    Args:
        data_dir: Directory containing NetCDF files to process.
        variable: Name of the variable to extract (default: 'h500').
        n_modes: Number of EOF modes to keep (default: 3).
        output_path: Path to save the EOF results and KDE object. If None, saves to
            'eof_kde_{variable}_modes{n_modes}.pkl' in '/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data'.

    Returns:
        tuple: (kde_object, eof_results_dict)
            - kde_object: scipy.stats.gaussian_kde object for the 3D PDF
            - eof_results_dict: Dictionary containing:
                - 'eofs': EOF spatial patterns (n_modes, lat, lon)
                - 'pcs': Principal components (time, n_modes)
                - 'explained_variance': Explained variance for each mode
                - 'explained_variance_ratio': Fraction of variance explained
                - 'mean': Temporal mean field (lat, lon)
                - 'lat': Latitude coordinates
                - 'lon': Longitude coordinates
    """
    # Find all NetCDF files
    pattern = os.path.join(data_dir, "*.nc")
    file_paths = sorted(glob.glob(pattern))
    
    if not file_paths:
        raise ValueError(f"No NetCDF files found in {data_dir}")
    
    print(f"Found {len(file_paths)} NetCDF files to process")
    
    # Load and concatenate all datasets
    datasets = []
    for file_path in file_paths:
        print(f"Loading {os.path.basename(file_path)}...")
        with xr.open_dataset(file_path, decode_times=False) as ds:
            # Check if variable exists
            if variable not in ds.data_vars:
                raise ValueError(
                    f"Variable '{variable}' not found in {file_path}. "
                    f"Available variables: {list(ds.data_vars.keys())}"
                )
            datasets.append(ds[[variable]])  # Only keep the variable we need
    
    # Concatenate along time dimension
    print("Concatenating datasets...")
    ds_combined = xr.concat(datasets, dim="time")
    
    # Extract the variable
    var_data = ds_combined[variable]
    print(f"Data shape: {var_data.shape}")
    print(f"Dimensions: {var_data.dims}")
    
    # Get coordinate information
    lat = var_data.coords["latitude"].values
    lon = var_data.coords["longitude"].values
    time = var_data.coords["time"].values
    
    # Compute temporal mean
    print("Computing temporal mean...")
    mean_field = var_data.mean(dim="time")
    
    # Remove temporal mean (anomalies)
    print("Computing anomalies (removing temporal mean)...")
    anomalies = var_data - mean_field
    
    # Reshape to (time, lat*lon) for EOF computation
    print("Reshaping data for EOF computation...")
    n_time = len(time)
    n_lat = len(lat)
    n_lon = len(lon)
    n_spatial = n_lat * n_lon
    
    # Reshape anomalies to (time, lat*lon)
    data_matrix = anomalies.values.reshape(n_time, n_spatial)
    
    # Remove any NaN values by filling with 0 (or could use interpolation)
    # Check for NaNs
    nan_mask = np.isnan(data_matrix)
    if np.any(nan_mask):
        print(f"Warning: Found {np.sum(nan_mask)} NaN values. Filling with 0.")
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)
    
    # Compute EOF using SVD
    print("Computing EOFs using SVD...")
    # Center the data (should already be centered, but ensure it)
    data_centered = data_matrix - data_matrix.mean(axis=0, keepdims=True)
    
    # SVD: data_centered = U @ S @ Vt
    # U: (n_time, n_time) - left singular vectors (time patterns)
    # S: (min(n_time, n_spatial),) - singular values
    # Vt: (min(n_time, n_spatial), n_spatial) - right singular vectors (spatial patterns)
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    
    # EOFs are the right singular vectors (spatial patterns)
    # Reshape back to (n_modes, lat, lon)
    eofs = Vt[:n_modes, :].reshape(n_modes, n_lat, n_lon)
    
    # Principal components are the left singular vectors scaled by singular values
    # PCs: (n_time, n_modes)
    pcs = U[:, :n_modes] @ np.diag(S[:n_modes])
    
    # Explained variance
    # Variance explained by each mode = S^2 / (n_time - 1)
    explained_variance = S[:n_modes] ** 2 / (n_time - 1)
    total_variance = np.sum(S ** 2) / (n_time - 1)
    explained_variance_ratio = explained_variance / total_variance
    
    print(f"\nEOF computation complete!")
    print(f"Number of modes: {n_modes}")
    print(f"Explained variance by mode:")
    for i in range(n_modes):
        print(f"  Mode {i+1}: {explained_variance[i]:.2f} ({explained_variance_ratio[i]*100:.2f}%)")
    print(f"Total variance explained by first {n_modes} modes: {np.sum(explained_variance_ratio)*100:.2f}%")
    
    # Project data onto first n_modes EOFs to get PC values
    # This is what we'll use for the KDE
    pc_data = pcs[:, :n_modes]  # Shape: (n_time, n_modes)
    
    print(f"\nPC data shape for KDE: {pc_data.shape}")
    print(f"PC ranges:")
    for i in range(n_modes):
        print(f"  PC{i+1}: [{pc_data[:, i].min():.2f}, {pc_data[:, i].max():.2f}]")
    
    # Compute 3D KDE PDF
    print("\nComputing 3D KDE PDF...")
    # Transpose to (n_modes, n_time) for gaussian_kde
    kde = gaussian_kde(pc_data.T)
    
    # Prepare results dictionary
    eof_results = {
        "eofs": eofs,
        "pcs": pcs,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "mean": mean_field.values,
        "lat": lat,
        "lon": lon,
        "time": time,
        "variable": variable,
        "n_modes": n_modes,
    }
    
    # Save the results
    if output_path is None:
        default_dir = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data"
        os.makedirs(default_dir, exist_ok=True)
        output_path = os.path.join(default_dir, f"eof_kde_{variable}_modes{n_modes}.pkl")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving EOF results and KDE to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump({"kde": kde, "eof_results": eof_results}, f)
    
    print("Done!")
    
    return kde, eof_results


def project_onto_eofs(
    data: xr.DataArray | np.ndarray,
    eof_results: dict,
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Project new data onto the computed EOFs.

    Args:
        data: Data array to project. If xarray DataArray, should have 'latitude' and
            'longitude' dimensions. If numpy array, should have shape (lat, lon) or
            (time, lat, lon). If 2D, lat and lon must be provided.
        eof_results: Dictionary from compute_eof_kde_pdf containing EOF results.
        lat: Latitude coordinates (only needed if data is numpy array).
        lon: Longitude coordinates (only needed if data is numpy array).

    Returns:
        Projected PC values with shape (n_time, n_modes) or (n_modes,) if single timestep.
    """
    # Extract EOFs and mean
    eofs = eof_results["eofs"]
    mean_field = eof_results["mean"]
    n_modes = eof_results["n_modes"]
    
    # Handle input data
    if isinstance(data, xr.DataArray):
        data_values = data.values
        if "time" in data.dims:
            # Multiple timesteps
            data_values = data_values - mean_field[np.newaxis, :, :]
            n_time = data_values.shape[0]
            n_lat, n_lon = data_values.shape[1], data_values.shape[2]
            data_flat = data_values.reshape(n_time, n_lat * n_lon)
        else:
            # Single timestep
            data_values = data_values - mean_field
            n_lat, n_lon = data_values.shape
            data_flat = data_values.reshape(1, n_lat * n_lon)
    else:
        # numpy array
        if data.ndim == 3:
            # (time, lat, lon)
            data_values = data - mean_field[np.newaxis, :, :]
            n_time = data_values.shape[0]
            n_lat, n_lon = data_values.shape[1], data_values.shape[2]
            data_flat = data_values.reshape(n_time, n_lat * n_lon)
        elif data.ndim == 2:
            # (lat, lon) - single timestep
            if lat is None or lon is None:
                raise ValueError("lat and lon must be provided for 2D numpy array input")
            data_values = data - mean_field
            n_lat, n_lon = data_values.shape
            data_flat = data_values.reshape(1, n_lat * n_lon)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    # Project onto EOFs
    # EOFs shape: (n_modes, n_lat, n_lon)
    # Reshape EOFs to (n_modes, n_lat * n_lon)
    eofs_flat = eofs.reshape(n_modes, n_lat * n_lon)
    
    # Project: (n_time, n_spatial) @ (n_spatial, n_modes) = (n_time, n_modes)
    pcs = data_flat @ eofs_flat.T
    
    # If single timestep, squeeze the time dimension
    if pcs.shape[0] == 1:
        pcs = pcs[0]
    
    return pcs


if __name__ == "__main__":
    # Run the function with default parameters
    kde, eof_results = compute_eof_kde_pdf()
    print(f"\nEOF KDE PDF computed successfully!")
    print(f"Number of timesteps: {len(eof_results['time'])}")
    print(f"Number of modes: {eof_results['n_modes']}")
    print(f"KDE object: {kde}")
