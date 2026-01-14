from pathlib import Path
from typing import Optional, Union

import torch
import torch.amp
import torch.nn as nn

from spatialdino.distributed import save_on_master
from spatialdino.models.utils import save_backbone


def save_model(
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    checkpoint_dir = output_dir.joinpath(f"step={step}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }

    save_backbone(checkpoint_dir, model)

    if loss_scaler is not None:
        ckpt_dict["scaler"] = loss_scaler.state_dict()

    save_on_master(ckpt_dict, checkpoint_dir.joinpath("ckpt.pth"))


def load_model(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: Optional[torch.amp.GradScaler] = None,
) -> int:
    if checkpoint_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location="cpu", check_hash=True, weights_only=False
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint:
        loss_scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint["step"] + 1
