"""
Update YAML config files for training and inference indices.
"""

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
