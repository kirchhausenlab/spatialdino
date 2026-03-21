"""DINOv2-style self-supervised pretraining with DINO, iBOT, and KoLeo losses.

This script orchestrates distributed self-supervised pretraining of a Vision
Transformer backbone using the DINOv2 framework. It combines three complementary
losses:
  - DINO loss: CLS-token distillation between student and teacher networks.
  - iBOT loss: Masked patch-token prediction via a teacher-student framework.
  - KoLeo loss: Kozachenko-Leonenko entropy regularizer for uniform feature
    distribution.

The training uses an exponential moving average (EMA) teacher, multi-crop
augmentation (global + local crops), mixed-precision support, gradient
accumulation, and WebDataset-based streaming data loading.
"""

import datetime
import logging
import math
import sys
import time
from functools import partial
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
from spatialdino.data.collate import collate_fn_train
from spatialdino.data.dataloader import make_dataloader
from spatialdino.data.dataset import custom_train_unbatched, make_webdataset
from spatialdino.data.transforms import PreTrainTransform
from spatialdino.logging import MetricLogger, SmoothedValue, setup_logging
from spatialdino.logging.wandb import init_wandb
from spatialdino.loss.dino_clstoken_loss import DINOLoss
from spatialdino.loss.ibot_patch_loss import iBOTPatchLoss
from spatialdino.loss.koleo_loss import KoLeoLoss
from spatialdino.models.ssl import SSL
from spatialdino.models.ssl.utils import save_model
from spatialdino.models.utils import build_ssl_model, load_model
from spatialdino.optim import lr_sched
from spatialdino.optim.lr_decay import (
    apply_optim_scheduler,
    get_params_groups_with_decay,
)
from spatialdino.utils.misc import make_3tuple, set_seed

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
logger = logging.getLogger("pretrain")


def main():
    """Entry point for DINOv2 self-supervised pretraining.

    Parses configuration, initializes distributed training, builds the SSL
    model and optimizer, constructs the WebDataset data pipeline with
    multi-crop transforms, and launches the training loop. Optionally
    resumes from a checkpoint and logs metrics to Weights & Biases.
    """
    config = parse_config(CONFIG_PATH.joinpath("pretrain.yaml"))

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

    logger.info(f"Global crop size: {config.global_crop_size}")
    logger.info(f"Local crop size: {config.local_crop_size}")
    logger.info(f"Isotropic scale factor: {config.isotropic_scale_factor}")

    image_key = "image"
    mask_key = "mask"

    train_transform = PreTrainTransform(
        global_crop_size=config.global_crop_size,
        local_crop_size=config.local_crop_size,
        isotropic_scale_factor=config.isotropic_scale_factor,
        n_global_crops=config.n_global_crops,
        n_local_crops=config.n_local_crops,
        image_key=image_key,
        mask_key=mask_key,
        in_chans=config.in_chans,
        mean=config.mean,
        std=config.std,
        dtype=config.dtype,
        random_state=seed,
    )

    global_crop_size = make_3tuple(config.global_crop_size)
    patch_size = make_3tuple(config.patch_size)

    # Compute the number of patches along each spatial dimension
    grid_size = tuple(
        global_crop_size[i] // patch_size[i] for i in range(len(global_crop_size))
    )

    train_collate = partial(
        collate_fn_train,
        grid_size=grid_size,
        patch_size=patch_size,
        min_mask_ratio=config.min_mask_ratio,
        max_mask_ratio=config.max_mask_ratio,
        mask_sample_probability=config.mask_sample_probability,
        mask_type=config.mask_type,
    )

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
        .compose(
            custom_train_unbatched(
                n_global_crops=config.n_global_crops,
                n_local_crops=config.n_local_crops,
            )
        )
        .shuffle(config.shuffle_buffer_size)
        .batched(
            config.batch_size,
            collation_fn=train_collate,
        )
    )

    # define the model
    model = build_ssl_model(
        config=config,
    ).to(device)

    logger.info("Model = %s" % str(model))

    assert config.accum_iter > 0, "accum_iter must be greater than 0"

    eff_batch_size = config.batch_size * config.accum_iter * dist.get_world_size()

    assert eff_batch_size > 1, "Effective batch size must be greater than 1"

    if config.lr is None:  # only base_lr is specified
        config.lr = config.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (config.lr * 256 / eff_batch_size))
    logger.info("actual lr: %.2e" % config.lr)

    logger.info("accumulate grad iterations: %d" % config.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    model_ddp = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=config.find_unused_parameters,
    )

    param_groups = get_params_groups_with_decay(
        model=model,
        lr_decay_rate=config.layerwise_decay,
        patch_embed_lr_mult=config.patch_embed_lr_mult,
    )

    if config.optim == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=config.betas)
    elif config.optim == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=config.lr, betas=config.betas)
    else:
        raise ValueError(f"Optimizer {config.optim} not supported")

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
    model: SSL,
    model_ddp: DDP,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    rank: int,
    world_size: int,
    loss_scaler: torch.amp.GradScaler | None = None,
    run: Run | None = None,
) -> dict[str, float]:
    """Run the main self-supervised pretraining loop.

    Iterates over the training data, computing DINO CLS-token loss, iBOT
    masked-patch loss, and KoLeo regularization (each gated by config
    weights). Applies learning-rate, weight-decay, momentum, and
    teacher-temperature schedules per step, performs gradient accumulation,
    and updates the teacher via EMA.

    Args:
        config: Hydra/OmegaConf configuration for the pretraining run.
        step: Global optimizer step to resume from (0 for fresh training).
        model: The underlying SSL model (student + teacher).
        model_ddp: DistributedDataParallel-wrapped model used for the
            student forward pass.
        optimizer: Optimizer instance for parameter updates.
        train_dataloader: WebDataset-backed streaming dataloader.
        rank: Process rank in the distributed group.
        world_size: Total number of distributed processes.
        loss_scaler: Optional AMP GradScaler for mixed-precision training.
        run: Optional Weights & Biases run for metric logging.

    Returns:
        Dictionary mapping metric names to their globally averaged values
        across all processes.
    """
    start_time = time.time()
    model.train()

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = lr_sched.build_schedulers(config)

    accum_iter = config.accum_iter

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
    header = "Pretraining"

    # Determine which loss components are active based on config weights
    do_dino = config.dino_loss_weight > 0
    do_ibot = config.ibot_loss_weight > 0
    do_koleo = config.koleo_loss_weight > 0

    latent_loss_fn = {}

    if do_dino:
        latent_loss_fn["cls_token"] = DINOLoss(
            out_dim=config.n_prototypes,
            student_temp=config.student_temp,
            center_momentum=config.center_momentum,
        ).to(device)

        if do_koleo:
            latent_loss_fn["koleo"] = KoLeoLoss().to(device)

    if do_ibot:
        latent_loss_fn["patch_tokens"] = iBOTPatchLoss(
            patch_out_dim=config.n_prototypes,
            student_temp=config.student_temp,
            center_momentum=config.center_momentum,
        ).to(device)

    iteration = step * accum_iter

    for batch in metric_logger.log_every(
        iterable=train_dataloader,
        print_freq=config.log_interval,
        header=header,
        n_iterations=config.max_steps,
        start_iteration=iteration,
        rank=rank,
    ):
        if iteration >= config.max_steps:
            break

        # Only update model weights after accumulating accum_iter gradients
        update = (iteration + 1) % accum_iter == 0

        lr = lr_schedule[step]
        wd = wd_schedule[step]
        mom = momentum_schedule[step]
        teacher_temp = teacher_temp_schedule[step]
        last_layer_lr = last_layer_lr_schedule[step]
        # Apply scheduler one iteration before the update step so the
        # optimizer uses the new LR/WD when it steps
        if (iteration + 2) % accum_iter == 0:
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        collated_global_crops = batch["collated_global_crops"].to(
            device, non_blocking=True
        )
        collated_local_crops = batch["collated_local_crops"].to(
            device, non_blocking=True
        )
        collated_masks = batch["collated_masks"].to(device, non_blocking=True)
        mask_indices_list = batch["mask_indices_list"].to(device, non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        masks_weight = batch["masks_weight"].to(device, non_blocking=True)
        upperbound = batch["upperbound"]
        n_masked_patches_tensor = batch["n_masked_patches_tensor"].to(
            device, non_blocking=True
        )

        loss = 0.0
        latent_loss = 0.0
        loss_dict = {}

        dtype = DTYPE_MAPPING[config.dtype]

        student_global_output, student_local_output = model_ddp(
            x={
                "collated_global_crops": collated_global_crops,
                "collated_local_crops": collated_local_crops,
            },
            masks={
                "collated_global_crops": collated_masks,
                "collated_local_crops": None,
            },
            upperbound=upperbound,
            n_masked_patches=n_masked_patches,
            mask_indices_list=mask_indices_list,
            device_type=config.device_type,
            dtype=dtype,
            enabled=config.use_amp,
        )

        teacher_global_output = model.forward_teacher(
            x=collated_global_crops,
            masks=None,
            upperbound=upperbound,
            n_masked_patches=n_masked_patches,
            mask_indices_list=mask_indices_list,
            n_masked_patches_tensor=n_masked_patches_tensor,
            teacher_temp=teacher_temp,
            latent_loss_fn=latent_loss_fn,
            device_type=config.device_type,
            dtype=dtype,
            enabled=config.use_amp,
        )

        # Normalization factors: total cross-view pairs for DINO loss averaging
        n_local_crops_loss_terms = max(config.n_local_crops * config.n_global_crops, 1)
        n_global_crops_loss_terms = (config.n_global_crops - 1) * config.n_global_crops

        if do_dino:
            dino_global_cls_loss = latent_loss_fn["cls_token"](
                student_output_list=[student_global_output["cls_token_after_head"]],
                teacher_out_softmaxed_centered_list=[
                    teacher_global_output["cls_token_softmaxed_centered"].flatten(0, 1)
                ],
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict["dino_global_cls_loss"] = dino_global_cls_loss
            latent_loss += config.dino_loss_weight * dino_global_cls_loss

            dino_local_cls_loss = latent_loss_fn["cls_token"](
                student_output_list=student_local_output["cls_token_after_head"].chunk(
                    config.n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_global_output[
                    "cls_token_softmaxed_centered"
                ],
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict["dino_local_cls_loss"] = dino_local_cls_loss

            latent_loss += config.dino_loss_weight * dino_local_cls_loss

            if do_koleo:
                koleo_loss = sum(
                    latent_loss_fn["koleo"](p.squeeze(0))
                    for p in student_global_output["cls_token_after_head"].chunk(
                        config.n_global_crops
                    )
                )
                loss_dict["koleo_loss"] = koleo_loss / config.n_global_crops
                latent_loss += config.koleo_loss_weight * koleo_loss

        if do_ibot:
            ibot_patch_loss = latent_loss_fn["patch_tokens"].forward_masked(
                student_global_output["patch_tokens_after_head"],
                teacher_global_output["patch_tokens_softmaxed_centered"],
                student_masks_flat=collated_masks,
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight,
            )

            loss_dict["ibot_patch_loss"] = ibot_patch_loss / config.n_global_crops

            latent_loss += config.ibot_loss_weight * ibot_patch_loss

        loss_dict["latent_loss"] = latent_loss
        loss += latent_loss

        loss /= accum_iter

        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()

        else:
            loss.backward()

        if update:
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

            optimizer.zero_grad(set_to_none=True)

            # perform teacher EMA update
            model.update_teacher(teacher_momentum=mom)

        # All-reduce losses across GPUs and compute per-process averages
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

        metric_logger.update(
            loss=loss_value,
            lr=lr,
            **loss_dict_reduced,
        )

        if run is not None and (iteration + 1) % config.log_interval == 0:
            run.log(
                {
                    "loss": loss_value,
                    "lr": lr,
                    **loss_dict_reduced,
                },
                step=iteration,
            )

        if (iteration + 1) % config.save_interval == 0 or (
            iteration + 1
        ) == config.max_steps:
            logger.info(f"Saving checkpoint at step {step}")
            save_model(
                output_dir=ckpt_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
            )

        if update:
            step += 1

        iteration += 1

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
