import logging
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.amp
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data.transforms import LiFTtransforms
from spatialdino.logging import setup_logging
from spatialdino.logging.utils import MetricLogger
from spatialdino.logging.wandb import init_wandb
from spatialdino.models.dino3d.lift.model.extractor import ViTExtractor
from spatialdino.models.dino3d.lift.model.lift_model import LiFT
from spatialdino.models.dino3d.lift.train.train import training_loop
from spatialdino.models.dino3d.lift.utils.utils import setup_dataloader
from spatialdino.models.utils import init_backbone
from spatialdino.optim.lr_decay import get_params_groups_with_decay
from spatialdino.utils.misc import make_3tuple, set_seed

logger = logging.getLogger("LiFT")
torch.autograd.set_detect_anomaly(True)


def main() -> None:
    cfg = parse_config(CONFIG_PATH.joinpath("upsampler.yaml"))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.empty_cache()
    torch.cuda.set_device(local_rank)

    if rank == 0:
        logger.info(
            f"OPTIONS -- local_rank: {local_rank}, rank: {rank}, world_size: {world_size}\n\n"
        )
    set_seed(cfg.seed + dist.get_rank())
    setup_logging(rank=local_rank)
    device = torch.device(local_rank)

    model = init_backbone(config=cfg)
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    model.eval()
    if cfg.lift_model.compile:
        logger.info("Compiling model")
        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

    extractor = ViTExtractor(
        model=model,  # type: ignore
        stride=cfg.lift_model.stride,
        device=device,
    )
    lift = LiFT(
        in_channels=cfg.lift_model.embed_dim,
        patch_size=cfg.lift_model.patch_size,
        image_channels=cfg.lift_model.in_chans,
        latent_dim=cfg.lift_model.latent_dim,
        num_blocks=cfg.lift_model.num_blocks,
    )
    lift.to(device)

    lift = DDP(lift, device_ids=[local_rank], output_device=local_rank)
    logger.info(
        f"******************Running LiFT on device: {device}\n\n******************"
    )

    image_key = "image"
    mask_key = "mask"

    transforms = LiFTtransforms(
        isotropic_scale_factor=cfg.lift_model.isotropic_scale_factor,
        global_crops_size=cfg.lift_model.imsize,
        image_key=image_key,
        mask_key=mask_key,
        in_chans=cfg.in_chans,
        mean=cfg.mean,
        std=cfg.std,
        dtype=cfg.dtype,
        random_state=cfg.seed,
    )

    dataloader = setup_dataloader(
        config=cfg,
        transform=transforms,
    )

    scaler = GradScaler()
    param_groups = get_params_groups_with_decay(
        model=lift.module if cfg.distributed else lift,
        lr_decay_rate=cfg.weight_decay,
        patch_embed_lr_mult=cfg.patch_embed_lr_mult,
    )

    eff_batch_size = cfg.batch_size * cfg.accum_iter * dist.get_world_size()
    cfg.lr = cfg.blr * eff_batch_size / 256
    optimizer = AdamW(param_groups, lr=cfg.lr, betas=cfg.betas)
    logger.info(f"Optimizer: {optimizer}")
    loss_type = cfg.lift_model.loss_type
    if loss_type == "MSE":
        loss = nn.MSELoss()
    elif loss_type == "L1":
        loss = nn.L1Loss()
    else:
        loss = nn.CosineEmbeddingLoss()
    imsize = cfg.lift_model.imsize
    patch = make_3tuple(cfg.lift_model.patch_size)
    L1 = (
        imsize[0] // patch[0],
        imsize[1] // patch[1],
        imsize[2] // patch[2],
    )  # 64, 256, 256 -> 8, 32, 32
    L2 = (L1[0] // 2, L1[1] // 2, L1[2] // 2)
    L4 = (L1[0] // 4, L1[1] // 4, L1[2] // 4)

    assert all(L > 0 for L in L4)
    metadata = {"iteration": cfg.lift_model.start_step}
    if cfg.resume and cfg.ckpt_path:
        ckpt_path = Path(cfg.prediction.ckpt_path)
        state_dict = torch.load(ckpt_path)
        lift.load_state_dict(state_dict["model"])
        metadata["iteration"] = state_dict["iteration"]
    (metric_logger, run) = (None, None)

    if rank == 0:
        run = init_wandb(cfg=cfg)
        output_dir = Path(cfg.lift_model.output_dir)
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

    iteration = metadata["iteration"]
    training_loop(
        cfg=cfg,
        scaler=scaler,
        iteration=iteration,
        dataloader=dataloader,
        extractor=extractor,
        patch_size=make_3tuple(cfg.lift_model.patch_size),  # type: ignore
        L1=L1,
        L2=L2,
        L4=L4,
        lift=lift,
        loss_fn=loss,
        optimizer=optimizer,
        local_rank=local_rank,
        metric_logger=metric_logger,
        # test_data_loader=test_dataloader,
        device=device,
        run=run,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
