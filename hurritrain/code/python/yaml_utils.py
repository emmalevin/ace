"""
Update YAML config files for training and inference indices.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import numpy as np


def update_yaml_training_indices(
    yaml_path: str,
    training_indices: list[int] | "np.ndarray",
    total_time_indices: int | None = None,
) -> None:
    """
    Update the indices in a YAML training config file.

    Args:
        yaml_path: Path to the YAML config file.
        training_indices: List of starting indices for each group (e.g., [0, 100]).
        total_time_indices: Optional total number of time indices for validation.
    """
    import numpy as np  # runtime import for isinstance check

    if isinstance(training_indices, np.ndarray):
        training_indices = training_indices.tolist()

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"YAML file {yaml_path} is empty or could not be parsed")

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

    config["train_loader"]["dataset"]["subset"] = training_indices

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_yaml_inference_indices(
    yaml_path: str,
    total_time_indices: int,
) -> None:
    """Update initial_condition.start_indices in an inference YAML config."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"YAML file {yaml_path} is empty or could not be parsed")

    if config.get("initial_condition") is None:
        raise ValueError(f"YAML file {yaml_path} does not have 'initial_condition' section")
    if "start_indices" not in config["initial_condition"]:
        raise ValueError(
            f"YAML file {yaml_path} does not have 'initial_condition.start_indices' section"
        )
    config["initial_condition"]["start_indices"]["n_initial_conditions"] = total_time_indices
    config["initial_condition"]["start_indices"]["first"] = 0
    config["initial_condition"]["start_indices"]["interval"] = 1

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_fast_inference_start_indices(yaml_path: str, n_indices: int) -> None:
    """
    Update base_evaluator_config.loader.start_indices.list in a fast-inference
    evaluator YAML to [0, 1, ..., n_indices-1].
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"YAML file {yaml_path} is empty or could not be parsed")
    if "base_evaluator_config" not in config:
        raise ValueError(f"YAML file {yaml_path} does not have 'base_evaluator_config'")
    be = config["base_evaluator_config"]
    if "loader" not in be or "start_indices" not in be["loader"]:
        raise ValueError(
            f"YAML file {yaml_path} does not have base_evaluator_config.loader.start_indices"
        )
    be["loader"]["start_indices"]["list"] = list(range(n_indices))
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_fast_inference_indices_for_both(
    model1_yaml_path: str,
    model2_yaml_path: str,
    n_indices: int,
) -> None:
    """Update both fast-inference YAMLs with start_indices list [0, 1, ..., n_indices-1]."""
    update_fast_inference_start_indices(model1_yaml_path, n_indices)
    update_fast_inference_start_indices(model2_yaml_path, n_indices)


# Base paths for batch inference YAMLs (same as template inference YAMLs)
_DEFAULT_DATA_PATH = "/scratch/gpfs/GVECCHI/el2358/ace/training_data/"
_MODEL1_CKPT = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/training_output/model1_wandb/training_checkpoints/ckpt.tar"
_MODEL2_CKPT = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/training_output/model2_wandb/training_checkpoints/ckpt.tar"
_INFERENCE_OUTPUT_BASE = "/scratch/gpfs/GVECCHI/el2358/ace/hurritrain/inference_output"


def write_inference_batch_yamls(
    total_time_indices: int,
    yaml_dir: str,
    batch_size: int = 50,
    data_path: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Write batched inference YAML files (10 initial conditions per file) for model1 and model2.
    Files are written to yaml_dir/yaml_batch/model1/ and yaml_dir/yaml_batch/model2/ with
    names m1_inf_0.yaml, m1_inf_10.yaml, ... and m2_inf_0.yaml, m2_inf_10.yaml, ...
    Each file has start_indices.first = 0, 10, 20, ... and n_initial_conditions = 10
    (or fewer for the last batch). Experiment dirs are .../model1_inference_folders/model1_0, etc.

    Returns:
        (model1_yaml_paths, model2_yaml_paths) for use when running inference.
    """
    data_path = data_path or _DEFAULT_DATA_PATH
    batch_dir1 = os.path.join(yaml_dir, "yaml_batch", "model1")
    batch_dir2 = os.path.join(yaml_dir, "yaml_batch", "model2")
    os.makedirs(batch_dir1, exist_ok=True)
    os.makedirs(batch_dir2, exist_ok=True)

    model1_paths: list[str] = []
    model2_paths: list[str] = []

    for first in range(0, total_time_indices, batch_size):
        n_ic = min(batch_size, total_time_indices - first)

        experiment_dir1 = f"{_INFERENCE_OUTPUT_BASE}/model1_inference_folders/model1_{first}"
        experiment_dir2 = f"{_INFERENCE_OUTPUT_BASE}/model1_inference_folders/model2_{first}"

        config = {
            "experiment_dir": experiment_dir1,
            "n_forward_steps": 1,
            "forward_steps_in_memory": 1,
            "checkpoint_path": _MODEL1_CKPT,
            "logging": {
                "log_to_screen": True,
                "log_to_wandb": False,
                "log_to_file": True,
                "project": "ace",
            },
            "initial_condition": {
                "dataset": {
                    "data_path": data_path,
                    "file_pattern": "*.nc",
                },
                "start_indices": {
                    "first": first,
                    "n_initial_conditions": n_ic,
                    "interval": 1,
                },
            },
            "forcing_loader": {
                "dataset": {
                    "data_path": data_path,
                },
            },
            "data_writer": {
                "save_prediction_files": True,
                "save_monthly_files": False,
                "names": ["PRESsfc"],
            },
        }
        path1 = os.path.join(batch_dir1, f"m1_inf_{first}.yaml")
        with open(path1, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        model1_paths.append(path1)

        config["experiment_dir"] = experiment_dir2
        config["checkpoint_path"] = _MODEL2_CKPT
        path2 = os.path.join(batch_dir2, f"m2_inf_{first}.yaml")
        with open(path2, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        model2_paths.append(path2)

    return model1_paths, model2_paths
