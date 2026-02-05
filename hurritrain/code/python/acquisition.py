"""
Acquisition function: variance of min pressure in Gulf × (1 / py_kde).
"""

from typing import Sequence

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde


def compute_min_pressure_variance_and_top_sample_indices(
    prediction_paths: Sequence[str],
    candidate_indices_list: np.ndarray | list[int],
    py_kde: gaussian_kde,
    variable: str = "PRESsfc",
    lat_min: float = 22.0,
    lat_max: float = 29.0,
    lon_min: float = 263.0,
    lon_max: float = 277.0,
    n_top: int = 2,
    kde_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Open autoregressive_predictions.nc from each model, compute min surface
    pressure in the Gulf lat/lon box per sample, then acquisition = variance * (1/py_kde)
    per sample; return the n_top candidate indices with highest acquisition.

    Uses the same Gulf box as py_y (lat 22--29 N, lon 97--83 W in 0--360).
    py_kde is evaluated at the mean (across models) of the min pressure per sample.

    Args:
        prediction_paths: Paths to autoregressive_predictions.nc (one per model).
        candidate_indices_list: Time indices corresponding to each sample (length n_samples).
        py_kde: KDE of min surface pressure from py_y (e.g. from load_probability_data()['kde']).
        variable: Variable name (default PRESsfc).
        lat_min, lat_max, lon_min, lon_max: Gulf box in degrees (defaults from py_y).
        n_top: Number of candidate indices to return with highest acquisition (default 2).
        kde_eps: Small value added to KDE density to avoid division by zero (default 1e-10).

    Returns:
        top_candidate_indices: 1D array of length n_top (candidate time indices with highest acquisition).
        acquisition: 1D array of acquisition per sample (same length as sample dimension).
    """
    if len(prediction_paths) < 2:
        raise ValueError("Need at least 2 prediction files to compute variance across models")
    min_pressure_per_model = []
    for path in prediction_paths:
        with xr.open_dataset(path, decode_times=False) as ds:
            var = ds[variable]
            box = var.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )
            sample_dim = box.dims[0]
            min_per_sample = box.min(dim=[d for d in box.dims if d != sample_dim])
            min_pressure_per_model.append(min_per_sample.values)
    stacked = np.stack(min_pressure_per_model, axis=-1)
    variances = np.var(stacked, axis=-1)
    mean_min_pressure = np.mean(stacked, axis=-1)
    py_kde_values = np.squeeze(py_kde.evaluate(mean_min_pressure.reshape(1, -1)))
    if py_kde_values.ndim != 1:
        py_kde_values = np.atleast_1d(py_kde_values)
    inv_py_kde = 1.0 / (py_kde_values + kde_eps)
    acquisition = variances * inv_py_kde
    top_sample_indices = np.argsort(acquisition)[-n_top:][::-1]
    candidate_arr = np.asarray(candidate_indices_list)
    return candidate_arr[top_sample_indices], acquisition
