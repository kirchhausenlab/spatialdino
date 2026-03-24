from pathlib import Path
from typing import Optional
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
    loss_scaler: Optional[torch.amp.GradScaler] = None,  # type: ignore
) -> None:
    checkpoint_dir = output_dir.joinpath(f"step={step}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "step_semantics": "optimizer_updates_completed",
    }

    save_backbone(checkpoint_dir, model.teacher.encoder)  # type: ignore

    if loss_scaler is not None:
        ckpt_dict["scaler"] = loss_scaler.state_dict()

    save_on_master(ckpt_dict, checkpoint_dir.joinpath("ckpt.pth"))
