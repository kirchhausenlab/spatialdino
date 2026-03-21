from pathlib import Path

import torch
import torch.amp
import torch.nn as nn

from spatialdino.distributed import save_on_master


def save_model(
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.amp.GradScaler | None = None,  # type: ignore
) -> None:
    checkpoint_dir = output_dir.joinpath(f"step={step}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),  # type: ignore
        "step": step,
    }

    if loss_scaler is not None:
        ckpt_dict["scaler"] = loss_scaler.state_dict()

    save_on_master(model.state_dict(), checkpoint_dir.joinpath("segmentation.pth"))
    save_on_master(ckpt_dict, checkpoint_dir.joinpath("ckpt.pth"))
