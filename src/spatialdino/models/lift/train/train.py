import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

from spatialdino.data import DTYPE_MAPPING
from spatialdino.logging import MetricLogger
from spatialdino.logging.utils import SmoothedValue
from spatialdino.models.lift.model import LiFT
from spatialdino.models.lift.model.extractor import ViTExtractor
from spatialdino.models.lift.train import get_feats_
from spatialdino.optim import lr_sched
from spatialdino.optim.lr_decay import apply_optim_scheduler

logger = logging.getLogger("LiFT")
torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)


def training_loop(
    cfg: DictConfig,
    scaler: GradScaler,
    dataloader: DataLoader,
    extractor: ViTExtractor,
    L1: tuple[int, int, int],
    L2: tuple[int, int, int],
    L4: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    lift: Union[DDP, LiFT],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_rank: int,
    metric_logger: MetricLogger | None,
    test_data_loader: Iterable[DataLoader] | None = None,
    run: Run | None = None,
    iteration: int = 0,
):
    if metric_logger:
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    assert cfg.accum_iter >= 1, "Accumulation steps should be greater than 1"
    (
        lr_schedule,
        wd_schedule,
        last_layer_lr_schedule,
    ) = lr_sched.build_schedulers_lift(cfg=cfg)

    for data in (
        metric_logger.log_every(
            iterable=dataloader,
            print_freq=cfg.log_interval,
            header="Training",
            n_iterations=cfg.lift_model.max_steps,
            start_iteration=iteration,
            rank=local_rank,
        )
        if metric_logger
        else dataloader
    ):
        if iteration >= cfg.lift_model.max_steps:
            return
        if iteration % cfg.accum_iter == 0:
            lr = lr_schedule[iteration]
            wd = wd_schedule[iteration]
            last_layer_lr = last_layer_lr_schedule[iteration]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # [8, 1, 64, 256, 256] = # [B, C, D, H, W]
        image_batch = data["image"].to(device, non_blocking=True)
        image_half = data["image_half"].to(device, non_blocking=True)
        image_quarter = data["image_quarter"].to(device, non_blocking=True)

        feat1 = get_feats_(
            extractor=extractor,
            image_tensor=image_batch,
            patches=L1,
            return_class_token=False,
            use_fp16=cfg.use_fp16,
            device_type=cfg.device_type,
            dtype=cfg.dtype,
        )
        feat2 = get_feats_(
            extractor=extractor,
            image_tensor=image_half,
            patches=L2,
            return_class_token=False,
            use_fp16=cfg.use_fp16,
            device_type=cfg.device_type,
            dtype=cfg.dtype,
        )
        feat3 = get_feats_(
            extractor=extractor,
            image_tensor=image_quarter,
            patches=L4,
            return_class_token=False,
            use_fp16=cfg.use_fp16,
            device_type=cfg.device_type,
            dtype=cfg.dtype,
        )

        with torch.autocast(
            enabled=cfg.use_fp16,
            device_type=cfg.device_type,
            dtype=DTYPE_MAPPING[cfg.dtype],
        ):
            # image_half - [B, 1, 64, 256, 256], feat1 - [B, 768, 4, 16, 16]
            gen1 = lift.module(image_half, feat2)
            gen2 = lift.module(image_quarter, feat3)
        # (B, C, D, H, W) -> (B, D, H, W, C) -> (B*D*H*W, C)
        gen1 = gen1.permute(0, 2, 3, 4, 1).reshape(-1, gen1.shape[-1])
        gen2 = gen2.permute(0, 2, 3, 4, 1).reshape(-1, gen2.shape[-1])
        feat1 = feat1.permute(0, 2, 3, 4, 1).reshape(-1, feat1.shape[-1])
        feat2 = feat2.permute(0, 2, 3, 4, 1).reshape(-1, feat2.shape[-1])
        loss_1 = loss_fn(gen1, feat1, torch.ones(gen1.shape[0]).to(device))
        loss_2 = loss_fn(gen2, feat2, torch.ones(gen2.shape[0]).to(device))
        loss = loss_1 + loss_2

        optimizer.zero_grad(set_to_none=True)
        if cfg.use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss = loss.detach().cpu().item()

        if metric_logger is not None and local_rank == 0 and run is not None:
            metric_logger.update(
                loss=loss,
            )
            metric_logger.update(
                iteration=iteration,
            )
            metric_logger.update(
                lr=cfg.lr,
            )
            run.log(
                {
                    "train_loss": loss,
                    "lr": cfg.lr,
                    "iteration": iteration,
                },
                step=iteration,
                commit=False,
            )

        # save checkpoint
        if (
            (iteration + 1) % cfg.save_interval == 0
            and local_rank == 0
            and run is not None
        ):
            output_dir = Path(cfg.lift_model.output_dir)
            ckpt_dir = output_dir.joinpath("checkpoints")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_path = ckpt_dir.joinpath(f"checkpoint_iteration_{iteration}.pth")
            logger.info(f"Saving checkpoint to {save_path} at iteration {iteration}")
            torch.save(
                {
                    "iteration": iteration,
                    "model": lift.state_dict(),  # type: ignore
                },
                save_path,
            )

        iteration += 1
    if metric_logger:
        metric_logger.synchronize_between_processes()
        return {k: v.global_avg for k, v in metric_logger.meters.items()}
    return {}
