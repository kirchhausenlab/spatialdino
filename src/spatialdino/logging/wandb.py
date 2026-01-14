from typing import Any, Dict
import wandb
from omegaconf import DictConfig, OmegaConf, SCMode
from wandb.sdk.wandb_run import Run


def init_wandb(
    cfg: DictConfig,
    **kwargs: Any,
) -> Run:
    config = OmegaConf.to_container(
        cfg, resolve=True, structured_config_mode=SCMode.DICT
    )
    run = wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        name=cfg.experiment_name,
        config=config,
        **kwargs,
    )

    return run
