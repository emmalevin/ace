"""
Acquisition function: variance of min pressure in Gulf × (1 / py_kde) × px.
"""

import glob
import os
import re
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

# Default paths for batched inference outputs and px array
_DEFAULT_INFERENCE_FOLDERS_BASE = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/inference_output/model1_inference_folders"
_DEFAULT_PX_PATH = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data/kde_pdf_values_h500_1940.npy"


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
    px = np.load("/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/probability_data/kde_pdf_values_h500_1940.npy")
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
    acquisition = variances * inv_py_kde * px
    top_sample_indices = np.argsort(acquisition)[-n_top:][::-1]
    candidate_arr = np.asarray(candidate_indices_list)
    return candidate_arr[top_sample_indices], acquisition

def _sorted_model_folders(base_path: str, prefix: str) -> list[str]:
    """List subdirs base_path/prefix_* and sort by numeric suffix (e.g. model1_0, model1_10 -> 0, 10)."""
    pattern = os.path.join(base_path, f"{prefix}_*")
    dirs = glob.glob(pattern)
    dirs = [d for d in dirs if os.path.isdir(d)]
    def key(p: str) -> int:
        name = os.path.basename(p)
        m = re.match(rf"{re.escape(prefix)}_(\d+)", name)
        if m is None:
            return -1
        return int(m.group(1))
    return sorted(dirs, key=key)


def _min_pressure_per_time_from_folders(
    folder_paths: list[str],
    variable: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Open autoregressive_predictions.nc in each folder, extract min PRES in box per sample, concatenate."""
    all_min = []
    for folder in folder_paths:
        nc_path = os.path.join(folder, "autoregressive_predictions.nc")
        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"Missing {nc_path}")
        with xr.open_dataset(nc_path, decode_times=False) as ds:
            var = ds[variable]
            box = var.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )
            sample_dim = box.dims[0]
            min_per_sample = box.min(dim=[d for d in box.dims if d != sample_dim])
            all_min.append(min_per_sample.values)
    return np.concatenate(all_min, axis=0)


def compute_min_pressure_variance_and_top_sample_indices_batched(
    inference_folders_base: str = _DEFAULT_INFERENCE_FOLDERS_BASE,
    py_kde: gaussian_kde | None = None,
    px_path: str = _DEFAULT_PX_PATH,
    variable: str = "PRESsfc",
    lat_min: float = 22.0,
    lat_max: float = 29.0,
    lon_min: float = 263.0,
    lon_max: float = 277.0,
    n_top: int = 2,
    kde_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each model, load autoregressive_predictions.nc from model1_* and model2_* subdirs,
    concatenate min surface pressure in the Gulf box across all time indices. Compute
    acquisition = var * inv_py_kde * px at each time index; return top n_top time indices.

    Uses the same Gulf box as py_y (lat 22--29 N, lon 97--83 W in 0--360).
    """
    if py_kde is None:
        raise ValueError("py_kde is required")

    folders1 = _sorted_model_folders(inference_folders_base, "model1")
    folders2 = _sorted_model_folders(inference_folders_base, "model2")
    if not folders1 or not folders2:
        raise ValueError(
            f"No model1_* or model2_* folders found under {inference_folders_base}"
        )

    min_pressure_model1 = _min_pressure_per_time_from_folders(
        folders1, variable, lat_min, lat_max, lon_min, lon_max
    )
    min_pressure_model2 = _min_pressure_per_time_from_folders(
        folders2, variable, lat_min, lat_max, lon_min, lon_max
    )
    n_time = len(min_pressure_model1)
    if len(min_pressure_model2) != n_time:
        raise ValueError(
            f"Model1 and model2 have different lengths: {n_time} vs {len(min_pressure_model2)}"
        )

    time_indices = np.arange(n_time)
    px = np.load(px_path).squeeze()
    if px.size != n_time:
        raise ValueError(
            f"px length {px.size} does not match number of time indices {n_time}"
        )
    if px.ndim > 1:
        px = px.ravel()[:n_time]

    # Variance across the two models at each time index
    variances = np.var(
        np.stack([min_pressure_model1, min_pressure_model2], axis=-1), axis=-1
    )
    mean_min_pressure = (min_pressure_model1 + min_pressure_model2) / 2.0
    py_kde_values = np.squeeze(
        py_kde.evaluate(mean_min_pressure.reshape(1, -1))
    )
    if py_kde_values.ndim != 1:
        py_kde_values = np.atleast_1d(py_kde_values)
    inv_py_kde = 1.0 / (py_kde_values + kde_eps)
    acquisition = variances * inv_py_kde * px

    top_sample_indices = np.argsort(acquisition)[-n_top:][::-1]
    return time_indices[top_sample_indices], acquisition