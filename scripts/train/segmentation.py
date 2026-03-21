import datetime
import logging
import math
import sys
import time
from pathlib import Path

import torch
import webdataset as wds
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import spatialdino.distributed as dist
from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data import DTYPE_MAPPING
from spatialdino.data.collate import collate_fn_segmentation
from spatialdino.data.dataloader import make_dataloader
from spatialdino.data.dataset import custom_segmentation_unbatched, make_webdataset
from spatialdino.data.transforms import SegmentationTransform, remap_image
from spatialdino.logging import MetricLogger, SmoothedValue, setup_logging
from spatialdino.logging.wandb import init_wandb
from spatialdino.loss.charbonnier_loss import generalized_charbonnier_loss
from spatialdino.loss.soft_ncuts import SoftNCutsLoss
from spatialdino.models.segmentation import Segmentation
from spatialdino.models.segmentation.utils import save_model
from spatialdino.models.utils import build_segmentation_model, load_model
from spatialdino.utils.misc import set_seed

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
logger = logging.getLogger("segmentation")


def main():
    config = parse_config(CONFIG_PATH.joinpath("segmentation.yaml"))  # type: ignore

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank, rank, world_size = dist.setup(
        distributed=config.distributed,
        backend=config.backend,
    )
    torch.cuda.empty_cache()

    logger.info(
        f"OPTIONS -- local_rank: {local_rank}, rank: {rank}, world_size: {world_size}"
    )

    torch.cuda.set_device(local_rank)

    setup_logging(rank=rank)

    device = torch.cuda.current_device()

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    set_seed(seed)

    image_key = "image"
    mask_key = "mask"

    train_transform = SegmentationTransform(
        crop_size=config.crop_size,
        isotropic_scale_factor=config.isotropic_scale_factor,
        image_key=image_key,
        mask_key=mask_key,
        in_chans=config.in_chans,
        mean=config.mean,
        std=config.std,
        dtype=config.dtype,
        random_state=seed,
    )

    train_collate = collate_fn_segmentation

    train_dataset = make_webdataset(
        base_data_dir=config.base_data_dir,
        batch_size=config.batch_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
        nodesplitter=wds.split_by_node,
        collation_fn=train_collate,
        transform=train_transform,
    )

    train_dataloader = make_dataloader(
        train_dataset,
        batch_size=None,  # handled in web dataset
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
        drop_last=False,
        shuffle=False,  # web dataset already shuffled
        collate_fn=None,  # handled in web dataset
    )

    train_dataloader = (
        train_dataloader
        .compose(custom_segmentation_unbatched())
        .shuffle(config.shuffle_buffer_size)
        .batched(
            config.batch_size,
            collation_fn=collate_fn_segmentation,
        )
    )

    model = build_segmentation_model(config).to(device)
    logger.info("Model = %s" % str(model))

    eff_batch_size = config.batch_size * dist.get_world_size()

    if config.lr is None:  # only base_lr is specified
        config.lr = config.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (config.lr * 256 / eff_batch_size))
    logger.info("actual lr: %.2e" % config.lr)

    logger.info("effective batch size: %d" % eff_batch_size)

    model_ddp = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=config.find_unused_parameters,
    )

    param_groups = model_ddp.parameters()

    if config.optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    elif config.optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {config.optim} not supported")

    logger.info(f"Optimizer: {optimizer}")

    loss_scaler = (
        torch.amp.GradScaler(device=config.device_type) if config.use_amp else None  # type: ignore
    )

    step = 0
    if config.resume and config.ckpt_path:
        logger.info(f"Loading checkpoint from {config.ckpt_path}")
        step = load_model(
            checkpoint_path=config.ckpt_path,
            model=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    run = None
    if rank == 0:
        run = init_wandb(config)

    train(
        config=config,
        step=step,
        model=model,
        model_ddp=model_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        train_dataloader=train_dataloader,
        rank=rank,
        world_size=world_size,
        run=run,
    )

    dist.cleanup(distributed=config.distributed)


def train(
    config: DictConfig,
    step: int,
    model: Segmentation,
    model_ddp: DDP,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    rank: int,
    world_size: int,
    loss_scaler: torch.amp.GradScaler | None = None,  # type: ignore
    run: Run | None = None,
) -> dict[str, float]:
    start_time = time.time()
    model.train()

    device = torch.cuda.current_device()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir.joinpath("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir.joinpath("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = log_dir.joinpath("training_metrics.json")
    metric_logger = MetricLogger(
        logger=logger,
        delimiter="  ",
        output_file=str(metrics_file),
    )
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Segmentation"

    dtype = DTYPE_MAPPING[config.dtype]

    soft_n_cuts_loss_fn = SoftNCutsLoss(
        data_shape=config.crop_size,
        intensity_sigma=config.intensity_sigma,
        spatial_sigma=config.spatial_sigma,
        radius=config.radius,
    ).to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=config.lr_decay_factor,
        patience=config.lr_sched_patience,
    )

    for data in metric_logger.log_every(
        iterable=train_dataloader,
        print_freq=config.log_interval,
        header=header,
        n_iterations=config.max_steps,
        start_iteration=step,
        rank=rank,
    ):
        if step >= config.max_steps:
            break

        image_batch = data["collated_images"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss = 0.0
        loss_dict = {}

        out = model(
            x=image_batch,
            device_type=config.device_type,
            dtype=dtype,
            enabled=config.use_amp,
        )

        target = remap_image(image=image_batch)

        pred = out["decoder_recon"]
        if config.pix_loss_type == "l1":
            pix_loss = (pred - target).abs()
        elif config.pix_loss_type == "l2":
            pix_loss = (pred - target) ** 2
        elif config.pix_loss_type == "charbonnier":
            pix_loss = generalized_charbonnier_loss(pred, target)
        else:
            raise ValueError(f"Invalid loss type: {config.pix_loss_type}")

        pix_loss = pix_loss.mean()

        loss_dict["pix_loss"] = pix_loss

        loss += config.pix_loss_weight * pix_loss

        pred = out["decoder_ncuts"]
        n_cuts_loss = soft_n_cuts_loss_fn(
            labels=pred,
            inputs=target,
        )
        loss_dict["n_cuts_loss"] = n_cuts_loss
        loss += config.n_cuts_weight * n_cuts_loss

        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
            if config.clip_grad:
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            if config.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()

        if dist.get_world_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)

        loss_dict_reduced = {
            k: v.item() / dist.get_world_size() for k, v in loss_dict.items()
        }

        loss_value = loss.detach().cpu().item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training")
            logger.error(f"Loss dict: {loss_dict}")
            sys.exit(1)

        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(
            loss=loss_value,
            lr=lr,
            **loss_dict_reduced,
        )

        if run is not None and (step + 1) % config.log_interval == 0:
            run.log(
                {
                    "loss": loss_value,
                    "lr": lr,
                    **loss_dict_reduced,
                },
                step=step,
            )

        if (step + 1) % config.save_interval == 0 or (step + 1) == config.max_steps:
            logger.info(f"Saving checkpoint at step {step}")
            save_model(
                output_dir=ckpt_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
            )

        scheduler.step(loss_value)

        step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info("Averaged stats:", metric_logger)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    model.eval()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    main()
