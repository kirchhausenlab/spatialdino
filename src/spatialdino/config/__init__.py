from importlib.abc import Traversable
from importlib.resources import files
from pathlib import Path
from typing import Final, Optional

from omegaconf import DictConfig, OmegaConf

CONFIG_PATH: Final[Traversable] = files("spatialdino.config")
MODEL_CONFIG_PATH: Final[Traversable] = CONFIG_PATH.joinpath("models")


def parse_config(
    config_path: str,
) -> DictConfig:
    config_path = Path(config_path)
    assert config_path.exists(), f"Config file {config_path} does not exist"
    merged_config = OmegaConf.load(config_path)

    if hasattr(merged_config, "model"):
        model_config = MODEL_CONFIG_PATH.joinpath(f"{merged_config.model}.yaml")
        model_config = OmegaConf.load(model_config)
        merged_config = OmegaConf.merge(merged_config, model_config)

    merged_config.merge_with_cli()

    return merged_config
