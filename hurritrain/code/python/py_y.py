'''
Code to compute the KDE approximated PDF of our target variable (surface pressure in the Gulf of Mexico).
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


def compute_kde_pdf(
    data_dir: str = "/scratch/gpfs/GVECCHI/el2358/ace/training_data",
    variable: str = "PRESsfc",
    lat_min: float = 22.0,
    lat_max: float = 29.0,
    lon_min: float = 263.0,  # 97W in 0-360 system (360 - 97 = 263)
    lon_max: float = 277.0,  # 83W in 0-360 system (360 - 83 = 277)
    output_path: Optional[str] = None,
) -> tuple[gaussian_kde, np.ndarray]:
    """
    Compute a KDE PDF of a variable in a specified lat/lon box across all timesteps.

    Args:
        data_dir: Directory containing NetCDF files to process.
        variable: Name of the variable to extract (default: 'PRESsfc').
        lat_min: Minimum latitude (degrees North, default: 22.0).
        lat_max: Maximum latitude (degrees North, default: 29.0).
        lon_min: Minimum longitude (degrees East, 0-360, default: 263.0 for 97W).
        lon_max: Maximum longitude (degrees East, 0-360, default: 277.0 for 83W).
        output_path: Path to save the KDE object and mean values. If None, saves to
            'kde_pdf_{variable}.pkl' in '/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data'.

    Returns:
        tuple: (kde_object, mean_values_array)
            - kde_object: scipy.stats.gaussian_kde object for the PDF
            - mean_values_array: numpy array of mean values for each timestep
    """
    # Find all NetCDF files
    pattern = os.path.join(data_dir, "*.nc")
    file_paths = sorted(glob.glob(pattern))
    
    if not file_paths:
        raise ValueError(f"No NetCDF files found in {data_dir}")
    
    print(f"Found {len(file_paths)} NetCDF files to process")
    
    # List to store mean values for each timestep
    mean_values = []
    
    # Process each file
    for file_path in file_paths:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # Open the dataset
        with xr.open_dataset(file_path, decode_times=False) as ds:
            # Check if variable exists
            if variable not in ds.data_vars:
                raise ValueError(
                    f"Variable '{variable}' not found in {file_path}. "
                    f"Available variables: {list(ds.data_vars.keys())}"
                )
            
            # Select the region (Gulf of Mexico)
            # Note: sel uses inclusive bounds, so we use slice
            ds_region = ds.sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max),
            )
            
            # Extract the variable
            var_data = ds_region[variable]
            
            # Compute mean over spatial dimensions (latitude, longitude) for each timestep
            # This gives one value per timestep
            mean_per_timestep = var_data.mean(dim=["latitude", "longitude"])
            
            # Convert to numpy array and append to our list
            mean_values.append(mean_per_timestep.values)
    
    # Concatenate all mean values across all files
    all_means = np.concatenate(mean_values)
    
    print(f"\nExtracted {len(all_means)} timesteps total")
    print(f"Mean value range: [{all_means.min():.2f}, {all_means.max():.2f}]")
    print(f"Mean: {all_means.mean():.2f}, Std: {all_means.std():.2f}")
    
    # Compute KDE PDF
    print("\nComputing KDE PDF...")
    kde = gaussian_kde(all_means)
    
    # Save the results
    if output_path is None:
        default_dir = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data"
        os.makedirs(default_dir, exist_ok=True)
        output_path = os.path.join(default_dir, f"kde_pdf_{variable}.pkl")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving KDE PDF and mean values to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump({"kde": kde, "mean_values": all_means}, f)
    
    print("Done!")
    
    return kde, all_means


if __name__ == "__main__":
    # Run the function with default parameters
    kde, mean_values = compute_kde_pdf()
    print(f"\nKDE PDF computed successfully!")
    print(f"Number of timesteps: {len(mean_values)}")
    print(f"KDE object: {kde}")
