"""Distributed GPU deskewing of lattice light-sheet microscopy TIF stacks.

Lattice light-sheet data is acquired at an oblique angle and must be
geometrically corrected (deskewed) before downstream analysis. This script
loads raw TIF volumes, builds an affine deskew transform on the GPU, and
applies it in parallel across multiple GPUs using PyTorch
DistributedDataParallel. A binary mask is used to fill invalid border
voxels with the median intensity of the masked region.

Configuration is read from ``deskew.yaml`` via Hydra/OmegaConf.

Usage::

    torchrun --nproc_per_node=<N> deskew.py

"""

import argparse
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from natsort import natsorted
from skimage import io
from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data.deskew import (
    DeskewDataset,
    build_affine_transform,
    deskew,
    get_deskew_transform,
)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import spatialdino.distributed as dist
from spatialdino.logging import setup_logging
from spatialdino.utils.misc import set_seed

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

logger = logging.getLogger("deskew")


def main() -> None:
    """Run distributed deskewing of lattice light-sheet TIF volumes.

    Reads configuration from ``deskew.yaml``, initialises distributed
    processes, computes the affine deskew matrix from the first frame,
    builds a validity mask, then iterates over all frames via a
    DistributedSampler-backed DataLoader. Each deskewed volume is saved
    as a TIF file with border voxels filled by the median of the masked
    region.
    """
    config = parse_config(CONFIG_PATH.joinpath("deskew.yaml"))
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

    fnames = natsorted(list(Path(config.file_path).glob("*.tif")))
    start_frame = config.start_frame
    fnames = fnames[start_frame:]  # Skip the first file

    save_path = Path(config.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Compute the deskew transform from the first volume's geometry
    raw_volume = io.imread(fnames[0]).astype(np.float32)
    output_shape, deskew_matrix = get_deskew_transform(raw_volume)
    affine_transform = build_affine_transform()
    affine_transform = affine_transform.to(device)

    deskew_matrix = torch.from_numpy(deskew_matrix).to(device, non_blocking=True)

    # Build a validity mask by deskewing a dummy all-ones volume alongside
    # the real data; voxels that map outside the original field of view
    # will have interpolated values < 1, marking them as invalid.
    dummy_volume = np.ones_like(raw_volume)
    C = 1 if raw_volume.ndim == 3 else raw_volume.shape[0]
    inp = np.stack([raw_volume, dummy_volume], axis=0)
    inp = np.expand_dims(inp, axis=0)  # [B, C, Z_in, Y_in, X_in]
    inp = torch.from_numpy(inp).to(device, non_blocking=True)
    warped = deskew(
        inp,
        affine_transform=affine_transform,
        deskew_matrix=deskew_matrix,
        output_shape=output_shape,
    )
    # split back into data and mask
    vol = warped[:, :C]  # [B, C, Z_out, Y_out, X_out]
    mask_warped = warped[:, C:]  # [B, 1, Z_out, Y_out, X_out]
    # Threshold at 0.99 to account for interpolation artifacts
    valid = mask_warped > 0.99  # True where original data was sampled
    mask = ~valid
    mask = mask.squeeze_(0).squeeze_(0).cpu().numpy()  # [Z_out, Y_out, X_out]

    logger.info(f"Calculating and saving deskewed volumes to {save_path}")

    dataset = DeskewDataset(config.file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
        sampler=DistributedSampler(dataset),
    )
    dataloader.sampler.set_epoch(0)  # type: ignore

    # only rank 0 shows a tqdm bar
    if rank == 0:
        prog = tqdm(dataloader, desc="Deskew Progress", unit="batch")
    else:
        prog = dataloader

    for batch_idx, batch in prog:
        batch = batch.to(device, non_blocking=True)  # [B, C, Z_in, Y_in, X_in]
        vols = deskew(
            batch,
            affine_transform=affine_transform,
            deskew_matrix=deskew_matrix,
            output_shape=output_shape,
        )
        vols = vols.squeeze_(1).cpu().numpy()  # [B, Z_out, Y_out, X_out]
        for i, vol in enumerate(vols):
            fill_val = np.median(vol[mask])
            vol[mask] = fill_val
            frame_id = batch_idx[i]
            io.imsave(
                save_path.joinpath(f"frame_{frame_id + start_frame}.tif"),
                vol,
            )

    if rank == 0:
        prog.close()  # type: ignore

    dist.cleanup(distributed=config.distributed)


if __name__ == "__main__":
    main()
