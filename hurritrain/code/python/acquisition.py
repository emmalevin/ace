"""
Acquisition function: variance of min pressure in Gulf × (1 / py_kde) × px.
"""

import os
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.stats import gaussian_kde

# Default paths for inference outputs (zarr) and px array
_DEFAULT_INFERENCE_OUTPUT_DIR = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/inference_output"
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


def _min_pressure_per_sample_from_zarr(
    zarr_path: str,
    variable: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    time_index: int = 1,
) -> np.ndarray:
    """
    Open autoregressive_predictions.zarr; select Gulf box, use time step time_index (default 1),
    then min over lat and lon to get one value per sample. Returns 1D array of shape (n_sample,).
    """
    ds = xr.open_zarr(zarr_path)
    var = ds[variable]
    box = var.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    if "time" in box.dims:
        box = box.isel(time=time_index)
    # First dim is sample; min over remaining dims (e.g. lat, lon)
    sample_dim = box.dims[0]
    min_per_sample = box.min(dim=[d for d in box.dims if d != sample_dim])
    out = np.asarray(min_per_sample.values).ravel()
    ds.close()
    return out


def compute_min_pressure_variance_and_top_sample_indices_batched(
    inference_output_dir: str = _DEFAULT_INFERENCE_OUTPUT_DIR,
    py_kde: gaussian_kde | None = None,
    px_path: str = _DEFAULT_PX_PATH,
    candidate_time_indices: np.ndarray | None = None,
    variable: str = "PRESsfc",
    lat_min: float = 22.0,
    lat_max: float = 29.0,
    lon_min: float = 263.0,
    lon_max: float = 277.0,
    time_index: int = 1,
    n_top: int = 2,
    kde_eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one zarr per model (autoregressive_predictions.zarr in model1_inference and
    model2_inference). Dimensions (sample, time, lat, lon); use time step time_index (default 1).
    Compute min surface pressure in Gulf box per sample, then acquisition = var * inv_py_kde * px;
    return top n_top time indices (and acquisition per sample).

    inference_output_dir: path containing model1_inference/ and model2_inference/.
    candidate_time_indices: 1D array of length n_sample mapping zarr sample position i to the
        time index in the full dataset. Used to subsample px (full length e.g. 1462) to the
        inference samples. If None, px is required to have length n_sample.
    """
    if py_kde is None:
        raise ValueError("py_kde is required")

    zarr1 = os.path.join(inference_output_dir, "model1_inference", "autoregressive_predictions.zarr")
    zarr2 = os.path.join(inference_output_dir, "model2_inference", "autoregressive_predictions.zarr")
    if not os.path.isdir(zarr1):
        raise FileNotFoundError(f"Zarr not found: {zarr1}")
    if not os.path.isdir(zarr2):
        raise FileNotFoundError(f"Zarr not found: {zarr2}")

    min_pressure_model1 = _min_pressure_per_sample_from_zarr(
        zarr1, variable, lat_min, lat_max, lon_min, lon_max, time_index=time_index
    )
    min_pressure_model2 = _min_pressure_per_sample_from_zarr(
        zarr2, variable, lat_min, lat_max, lon_min, lon_max, time_index=time_index
    )
    n_sample = len(min_pressure_model1)
    if len(min_pressure_model2) != n_sample:
        raise ValueError(
            f"Model1 and model2 have different sample sizes: {n_sample} vs {len(min_pressure_model2)}"
        )

    px_full = np.load(px_path).squeeze()
    if px_full.ndim > 1:
        px_full = px_full.ravel()
    if candidate_time_indices is not None:
        candidate_time_indices = np.asarray(candidate_time_indices).ravel()
        if len(candidate_time_indices) != n_sample:
            raise ValueError(
                f"candidate_time_indices length {len(candidate_time_indices)} does not match "
                f"number of samples {n_sample}"
            )
        if np.any(candidate_time_indices < 0) or np.any(candidate_time_indices >= px_full.size):
            raise ValueError(
                f"candidate_time_indices must be in [0, {px_full.size}); got min={candidate_time_indices.min()}, max={candidate_time_indices.max()}"
            )
        px = px_full[candidate_time_indices]
    else:
        if px_full.size != n_sample:
            raise ValueError(
                f"px length {px_full.size} does not match number of samples {n_sample}. "
                "Pass candidate_time_indices to subsample px to the inference samples."
            )
        px = px_full
        candidate_time_indices = np.arange(n_sample)

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
    return candidate_time_indices[top_sample_indices], acquisition