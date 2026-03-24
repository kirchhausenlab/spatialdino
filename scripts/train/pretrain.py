import datetime
import logging
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import webdataset as wds
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run
from spatialdino.models.utils import load_model
import spatialdino.distributed as dist
from spatialdino.models.ssl.utils import save_model
from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data import DTYPE_MAPPING
from spatialdino.data.collate import collate_fn_train
from spatialdino.data.dataloader import make_dataloader
from spatialdino.data.dataset import custom_train_unbatched, make_webdataset
from spatialdino.data.transforms import PreTrainTransform
from spatialdino.logging import MetricLogger, SmoothedValue, setup_logging
from spatialdino.logging.wandb import init_wandb
from spatialdino.loss.dino_clstoken_loss import DINOLoss, split_sample_major_batch
from spatialdino.loss.ibot_patch_loss import iBOTPatchLoss
from spatialdino.loss.koleo_loss import KoLeoLoss
from spatialdino.models.ssl import SSL
from spatialdino.models.utils import build_ssl_model
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


def _validate_crop_counts(config: DictConfig) -> None:
    assert config.n_global_crops > 0, "n_global_crops must be greater than 0"
    assert config.n_local_crops > 0, "n_local_crops must be greater than 0"


def _should_run_interval(step: int, interval: int, max_steps: int) -> bool:
    return step % interval == 0 or step == max_steps


def _log_pretrain_step(
    metric_logger: MetricLogger,
    header: str,
    step: int,
    max_steps: int,
    step_time: SmoothedValue,
    data_time: SmoothedValue,
    rank: int,
) -> None:
    metric_logger.dump_in_output_file(
        iteration=step,
        iter_time=step_time.avg,
        data_time=data_time.avg,
        rank=rank,
    )

    eta_seconds = step_time.global_avg * (max_steps - step)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    space_fmt = ":" + str(len(str(max_steps))) + "d"

    log_list = [
        header,
        "[{0" + space_fmt + "}/{1}]",
        "eta: {eta}",
        "{meters}",
        "time: {time}",
        "data: {data}",
    ]
    if torch.cuda.is_available():
        log_list += ["max mem: {memory:.0f}"]

    log_msg = metric_logger.delimiter.join(log_list)
    if torch.cuda.is_available():
        logger.info(
            log_msg.format(
                step,
                max_steps,
                eta=eta_string,
                meters=str(metric_logger),
                time=str(step_time),
                data=str(data_time),
                memory=torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            )
        )
    else:
        logger.info(
            log_msg.format(
                step,
                max_steps,
                eta=eta_string,
                meters=str(metric_logger),
                time=str(step_time),
                data=str(data_time),
            )
            )


def _should_update_optimizer_step(
    accum_iter: int,
    micro_steps_in_step: int,
) -> bool:
    return micro_steps_in_step + 1 >= accum_iter


def _wrap_model_for_training(
    model: SSL,
    distributed: bool,
    local_rank: int,
    find_unused_parameters: bool,
) -> Union[SSL, DDP]:
    if not distributed:
        return model
    return DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=find_unused_parameters,
    )


def _pairwise_dino_loss(
    loss_fn: DINOLoss,
    student_views: Tuple[torch.Tensor, ...],
    teacher_views: Tuple[torch.Tensor, ...],
    skip_matching_teacher_view: bool = False,
) -> torch.Tensor:
    if student_views:
        total_loss = student_views[0].new_zeros(())
    elif teacher_views:
        total_loss = teacher_views[0].new_zeros(())
    else:
        raise ValueError("Expected at least one student or teacher view.")

    for student_index, student_view in enumerate(student_views):
        current_teacher_views = teacher_views
        if skip_matching_teacher_view:
            current_teacher_views = (
                teacher_views[:student_index] + teacher_views[student_index + 1 :]
            )
        if not current_teacher_views:
            continue
        total_loss = total_loss + loss_fn(
            student_output_list=[student_view],
            teacher_out_softmaxed_centered_list=current_teacher_views,
        )

    return total_loss


def _reduce_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    reduced_loss_dict = {}
    world_size = dist.get_world_size()

    for key, value in loss_dict.items():
        reduced_value = value.detach().clone()
        if world_size > 1:
            torch.distributed.all_reduce(reduced_value)
            reduced_value /= world_size
        reduced_loss_dict[key] = reduced_value.item()

    return reduced_loss_dict


def _compute_weighted_loss_total(
    loss_dict: Dict[str, float],
    config: DictConfig,
) -> float:
    return (
        config.dino_loss_weight * loss_dict.get("dino_global_cls_loss", 0.0)
        + config.dino_loss_weight * loss_dict.get("dino_local_cls_loss", 0.0)
        + config.koleo_loss_weight * loss_dict.get("koleo_loss", 0.0)
        + config.ibot_loss_weight * loss_dict.get("ibot_patch_loss", 0.0)
    )


def main():
    config = parse_config(CONFIG_PATH.joinpath("pretrain.yaml"))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank, rank, world_size = dist.setup(
        distributed=config.distributed,
        backend=config.backend,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Pretraining requires CUDA. CPU execution is not supported.")
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
    _validate_crop_counts(config)

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
        train_dataloader.compose(
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

    train_model = _wrap_model_for_training(
        model=model,
        distributed=config.distributed,
        local_rank=local_rank,
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
        train_model=train_model,
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
    train_model: Union[SSL, DDP],
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    rank: int,
    world_size: int,
    loss_scaler: Optional[torch.amp.GradScaler] = None,
    run: Optional[Run] = None,
) -> Dict[str, float]:
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

    optimizer.zero_grad(set_to_none=True)
    step_loss_sums: Dict[str, float] = defaultdict(float)
    micro_steps_in_step = 0
    step_time = SmoothedValue(fmt="{avg:.6f}")
    data_time = SmoothedValue(fmt="{avg:.6f}")
    step_start_time = time.time()
    step_data_time = 0.0
    data_iterator = iter(train_dataloader)
    end = time.time()

    while step < config.max_steps:
        if micro_steps_in_step == 0:
            step_start_time = end
            step_data_time = 0.0

        try:
            batch = next(data_iterator)
        except StopIteration:
            break

        step_data_time += time.time() - end

        lr = lr_schedule[step]
        wd = wd_schedule[step]
        mom = momentum_schedule[step]
        teacher_temp = teacher_temp_schedule[step]
        last_layer_lr = last_layer_lr_schedule[step]

        if micro_steps_in_step == 0:
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
        should_sync_gradients = dist.should_sync_gradients(
            world_size=world_size,
            accum_iter=accum_iter,
            micro_steps_in_step=micro_steps_in_step,
        )
        update = _should_update_optimizer_step(
            accum_iter=accum_iter,
            micro_steps_in_step=micro_steps_in_step,
        )

        # Avoid per-microbatch all-reduces when accumulating gradients with DDP.
        with dist.maybe_no_sync(train_model, should_sync=should_sync_gradients):
            student_global_output, student_local_output = train_model(
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

            n_local_crops_loss_terms = config.n_local_crops * config.n_global_crops
            n_global_crops_loss_terms = max(config.n_global_crops - 1, 0) * (
                config.n_global_crops
            )
            n_dino_loss_terms = max(
                n_global_crops_loss_terms + n_local_crops_loss_terms,
                1,
            )

            if do_dino:
                student_global_cls_outputs = split_sample_major_batch(
                    student_global_output["cls_token_after_head"],
                    config.n_global_crops,
                )
                teacher_global_cls_outputs = split_sample_major_batch(
                    teacher_global_output["cls_token_softmaxed_centered"],
                    config.n_global_crops,
                )
                student_local_cls_outputs = split_sample_major_batch(
                    student_local_output["cls_token_after_head"],
                    config.n_local_crops,
                )

                dino_global_cls_loss = _pairwise_dino_loss(
                    loss_fn=latent_loss_fn["cls_token"],
                    student_views=student_global_cls_outputs,
                    teacher_views=teacher_global_cls_outputs,
                    skip_matching_teacher_view=True,
                ) / n_dino_loss_terms
                loss_dict["dino_global_cls_loss"] = dino_global_cls_loss
                latent_loss += config.dino_loss_weight * dino_global_cls_loss

                dino_local_cls_loss = _pairwise_dino_loss(
                    loss_fn=latent_loss_fn["cls_token"],
                    student_views=student_local_cls_outputs,
                    teacher_views=teacher_global_cls_outputs,
                ) / n_dino_loss_terms
                loss_dict["dino_local_cls_loss"] = dino_local_cls_loss

                latent_loss += config.dino_loss_weight * dino_local_cls_loss

                if do_koleo:
                    koleo_loss = sum(
                        latent_loss_fn["koleo"](p.squeeze(0))
                        for p in student_global_cls_outputs
                    ) / max(config.n_global_crops, 1)
                    loss_dict["koleo_loss"] = koleo_loss
                    latent_loss += config.koleo_loss_weight * koleo_loss

            if do_ibot:
                ibot_patch_loss = latent_loss_fn["patch_tokens"].forward_masked(
                    student_global_output["patch_tokens_after_head"],
                    teacher_global_output["patch_tokens_softmaxed_centered"],
                    student_masks_flat=collated_masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )

                loss_dict["ibot_patch_loss"] = ibot_patch_loss

                latent_loss += config.ibot_loss_weight * ibot_patch_loss

            loss += latent_loss

            local_has_nonfinite_loss = ~torch.isfinite(
                torch.stack(
                    [latent_loss.detach().float()]
                    + [value.detach().float() for value in loss_dict.values()]
                )
            ).all()
            has_nonfinite_loss = dist.any_true(local_has_nonfinite_loss)
            if has_nonfinite_loss:
                if local_has_nonfinite_loss.item():
                    local_loss_dict = {
                        k: float(v.detach().cpu()) for k, v in loss_dict.items()
                    }
                    local_loss_dict["latent_loss"] = float(latent_loss.detach().cpu())
                    logger.error(
                        "Encountered non-finite local loss on rank %d at optimizer step %d: %s",
                        rank,
                        step + 1,
                        local_loss_dict,
                    )
                else:
                    logger.error(
                        "Encountered non-finite loss on another rank at optimizer step %d; aborting all ranks.",
                        step + 1,
                    )
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("Encountered non-finite loss during pretraining.")

            loss /= accum_iter

            if loss_scaler is not None:
                loss_scaler.scale(loss).backward()

            else:
                loss.backward()

        micro_steps_in_step += 1

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

        loss_dict_reduced = _reduce_loss_dict(loss_dict)
        loss_dict_reduced["latent_loss"] = _compute_weighted_loss_total(
            loss_dict_reduced,
            config,
        )

        loss_value = loss_dict_reduced["latent_loss"]

        for key, value in loss_dict_reduced.items():
            step_loss_sums[key] += value

        if update:
            completed_step = step + 1
            averaged_step_losses = {
                key: value / micro_steps_in_step
                for key, value in step_loss_sums.items()
            }
            step_time.update(time.time() - step_start_time)
            data_time.update(step_data_time)

            metric_logger.update(
                loss=averaged_step_losses["latent_loss"],
                lr=lr,
                **averaged_step_losses,
            )

            if _should_run_interval(
                completed_step,
                config.log_interval,
                config.max_steps,
            ):
                _log_pretrain_step(
                    metric_logger=metric_logger,
                    header=header,
                    step=completed_step,
                    max_steps=config.max_steps,
                    step_time=step_time,
                    data_time=data_time,
                    rank=rank,
                )
                if run is not None:
                    run.log(
                        {
                            "loss": averaged_step_losses["latent_loss"],
                            "lr": lr,
                            **averaged_step_losses,
                        },
                        step=completed_step,
                    )

            if _should_run_interval(
                completed_step,
                config.save_interval,
                config.max_steps,
            ):
                logger.info(f"Saving checkpoint at step {completed_step}")
                save_model(
                    output_dir=ckpt_dir,
                    step=completed_step,
                    model=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                )

            step = completed_step
            micro_steps_in_step = 0
            step_loss_sums.clear()

        end = time.time()

    if micro_steps_in_step:
        logger.warning(
            "Dataloader exhausted with %d/%d accumulated microbatches for optimizer step %d; dropping the partial step.",
            micro_steps_in_step,
            accum_iter,
            step + 1,
        )
        optimizer.zero_grad(set_to_none=True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if not metric_logger.meters["lr"].count:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("No optimizer steps were run.")
        logger.info("Training time {}".format(total_time_str))
        model.eval()
        return {}

    logger.info("Averaged stats: %s", metric_logger)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    model.eval()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    main()
