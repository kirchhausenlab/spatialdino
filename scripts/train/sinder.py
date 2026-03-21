import datetime
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import webdataset as wds
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import spatialdino.distributed as dist
from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data import DTYPE_MAPPING
from spatialdino.data.collate import collate_fn_sinder
from spatialdino.data.dataloader import make_dataloader
from spatialdino.data.dataset import custom_sinder_unbatched, make_webdataset
from spatialdino.data.transforms import SinderTransform
from spatialdino.logging import setup_logging
from spatialdino.logging.utils import MetricLogger
from spatialdino.logging.wandb import init_wandb
from spatialdino.models.sinder.neighbor_loss import get_neighbor_loss
from spatialdino.models.sinder.repair import replace_linear_addition_noqk
from spatialdino.models.sinder.singular_defect import singular_defect_directions
from spatialdino.models.sinder.utils import save_model
from spatialdino.models.utils import init_backbone, load_model
from spatialdino.utils.misc import set_seed

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
logger = logging.getLogger("sinder")


def main():
    config = parse_config(CONFIG_PATH.joinpath("sinder.yaml"))

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

    train_transform = SinderTransform(
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

    train_collate = collate_fn_sinder

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
        .compose(custom_sinder_unbatched())
        .shuffle(config.shuffle_buffer_size)
        .batched(
            config.batch_size,
            collation_fn=train_collate,
        )
    )

    model = init_backbone(config).to(device)
    logger.info("Model = %s" % str(model))
    model.singular_defect_directions = singular_defect_directions(
        model=model, ffn_layer=config.ffn_layer
    )

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

    all_params = []
    for param in model.parameters():
        param.requires_grad = False

    replace_linear_addition_noqk(model, "model")
    for name, param in model.named_parameters():
        if ".epsilon" in name and param.requires_grad is True:
            all_params.append(param)

    grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_params.append(name)

    assert len(grad_params) == len(all_params)
    logger.info("grad_params: %d %s", len(grad_params), grad_params)
    logger.info("all_params: %d %s", len(all_params), all_params)
    optimizer = torch.optim.SGD(
        all_params,
        lr=config.lr,
        momentum=0.9,
    )

    logger.info(f"Optimizer: {optimizer}")

    loss_scaler = (
        torch.amp.GradScaler(device=config.device_type) if config.use_amp else None
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
    model: nn.Module,
    model_ddp: DDP,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    rank: int,
    world_size: int,
    loss_scaler: torch.amp.GradScaler | None = None,
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
    header = "Sinder"

    dtype = DTYPE_MAPPING[config.dtype]

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

        with torch.amp.autocast(
            enabled=config.use_amp,
            device_type=config.device_type,
            dtype=dtype,
        ):
            result = get_neighbor_loss(
                model=model,
                x=image_batch,
                skip_less_than=config.skip_less_than,
                temperature=config.temperature,
                mask_thr=config.mask_thr,
                kernel=config.kernel,
            )

        if result is None:
            # logger.info("No loss, skipping...")
            if run is not None and (step + 1) % config.log_interval == 0:
                run.log(
                    {
                        "loss": 0.0,  # no loss, skipping
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
            step += 1
            continue

        layer = result["i"]
        loss = result["loss_neighbor"]

        if torch.isnan(loss).any():
            logger.error(f"Loss is {loss.detach().cpu().item()}, stopping training")
            sys.exit(1)

        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
        else:
            loss.backward()

        if config.limit_layers:
            with torch.no_grad():
                for t in range(layer - config.limit_layers + 1):
                    for p in model.blocks[t].parameters():
                        p.grad = None

        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.info(f"NaN grad at {name}, skipping...")
                has_nan = True
        if has_nan:
            continue

        if loss_scaler is not None:
            if config.clip_grad:
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            if config.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()

        loss_value = loss.detach().cpu().item()

        metric_logger.update(
            loss=loss_value,
        )

        if run is not None and (step + 1) % config.log_interval == 0:
            run.log(
                {
                    "loss": loss_value,
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
