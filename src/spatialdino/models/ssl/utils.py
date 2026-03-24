from pathlib import Path
from typing import Mapping, Optional
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
    extra_modules: Optional[Mapping[str, nn.Module]] = None,
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

    if extra_modules:
        for module in extra_modules.values():
            apply_center_update = getattr(module, "apply_center_update", None)
            if apply_center_update is not None:
                apply_center_update()
        ckpt_dict["extra_modules"] = {
            name: module.state_dict() for name, module in extra_modules.items()
        }

    save_on_master(ckpt_dict, checkpoint_dir.joinpath("ckpt.pth"))
