import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from spatialdino.data import DTYPE_MAPPING
import spatialdino.distributed as dist
from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data.inference import (
    InferenceDataset,
    InferenceTransform,
    collate_fn,
)
from spatialdino.logging import setup_logging
from spatialdino.models.utils import init_backbone
from spatialdino.inference.input_files import list_tiff_paths
from spatialdino.inference.streaming import StreamingEncoder
from spatialdino.utils.misc import set_seed

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

logger = logging.getLogger("inference_3d")


def iter_batch_samples(
    batch: Dict[str, Any],
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for idx, vol_metadata in enumerate(batch["vol_metadata"]):
        yield (
            {
                "image": batch["images"][idx],
                "mask": batch["masks"][idx],
            },
            vol_metadata,
        )


def main() -> None:
    config = parse_config(CONFIG_PATH.joinpath("inference.yaml"))  # type: ignore
    local_rank, rank, world_size = dist.setup(
        distributed=config.distributed,
        backend=config.backend,
    )

    torch.cuda.empty_cache()
    torch.cuda.set_device(local_rank)

    setup_logging(rank=rank)

    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    dtype = DTYPE_MAPPING[config.dtype]

    model = init_backbone(config=config).eval().to(dtype).to(device)
    model.forward = model.predict  # type: ignore
    use_streaming = bool(getattr(config, "inference_route", "default") == "streaming")
    streaming_encoder = StreamingEncoder(model, device, config) if use_streaming else None

    file_path = config.file_path
    fnames = list_tiff_paths(file_path)
    file_start = int(getattr(config, "file_start", 0) or 0)
    file_end = getattr(config, "file_end", None)
    fnames = fnames[file_start:file_end]

    logger.info(f"Processing {len(fnames)} files")
    global_hist_min = getattr(config, "global_hist_min", None)
    global_hist_max = getattr(config, "global_hist_max", None)
    if global_hist_min is None and global_hist_max is None:
        logger.info("Using default per-volume normalization")
    else:
        logger.info(
            "Using global histogram normalization with global_hist_min=%s and global_hist_max=%s",
            global_hist_min,
            global_hist_max,
        )
    dataset = InferenceDataset(config=config, fnames=fnames)
    inference_transform = InferenceTransform(config=config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
        sampler=DistributedSampler(dataset, shuffle=False),
        collate_fn=collate_fn,
    )
    dataloader.sampler.set_epoch(0)  # type: ignore

    # only rank 0 shows a tqdm bar
    if rank == 0:
        prog = tqdm(dataloader, desc="Inference Progress", unit="batch")
    else:
        prog = dataloader

    save_special_tokens = bool(getattr(config, "save_special_tokens", False))

    def predict() -> None:
        """Run inference on the model."""
        for batch in prog:
            for data, vol_metadata in iter_batch_samples(batch):
                res = inference_transform(
                    data=data,
                    vol_metadata=vol_metadata,
                    chunk_interpolate=config.chunk_interpolate,
                    interpolate_chunk_size=config.interpolate_volume_chunk_size,
                    device="cpu" if use_streaming else device,
                )
                volume = torch.as_tensor(res["volume"])  # [C, Z, Y, X]

                feats_path = Path(vol_metadata["lr_feats_path"])
                feats_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving to {feats_path}")

                cls_token = None
                register_tokens = None

                if use_streaming:
                    result = streaming_encoder.predict(
                        volume,
                        vol_metadata=vol_metadata,
                        vit_feat="patch_attn",
                        norm_feat=config.norm_feat,
                        return_special_tokens=save_special_tokens,
                    )
                    if save_special_tokens:
                        lr_feats, cls_token, register_tokens = result
                    else:
                        lr_feats = result
                    lr_feats = lr_feats.squeeze(0).float()
                else:
                    result = model(
                        img=volume.unsqueeze_(0),
                        vit_feat="patch_attn",
                        norm_feat=config.norm_feat,
                        dtype=dtype,
                        use_amp=config.use_amp,
                        device_type=config.device_type,
                        device=device,
                        return_special_tokens=save_special_tokens,
                    )
                    if save_special_tokens:
                        lr_feats, cls_token, register_tokens = result
                        cls_token = cls_token.cpu().float()
                        if register_tokens is not None:
                            register_tokens = register_tokens.cpu().float()
                    else:
                        lr_feats = result
                    lr_feats = lr_feats.cpu().squeeze_(0).float()

                padding = vol_metadata["padding"]
                scale_factor = (
                    volume.shape[-3] / lr_feats.shape[-3],
                    volume.shape[-2] / lr_feats.shape[-2],
                    volume.shape[-1] / lr_feats.shape[-1],
                )
                padding_lr = (
                    math.floor(padding[0] / scale_factor[0]),
                    math.floor(padding[1] / scale_factor[1]),
                    math.floor(padding[2] / scale_factor[2]),
                )
                lr_feats = lr_feats[
                    :,
                    padding_lr[0] // 2 : lr_feats.shape[-3] - padding_lr[0] // 2,
                    padding_lr[1] // 2 : lr_feats.shape[-2] - padding_lr[1] // 2,
                    padding_lr[2] // 2 : lr_feats.shape[-1] - padding_lr[2] // 2,
                ]
                lr_feats = lr_feats.permute(1, 2, 3, 0)  # [Z, Y, X, C]
                np.save(feats_path, lr_feats.cpu().numpy())
                logger.info(f"Saved features to {feats_path}")

                if save_special_tokens and cls_token is not None:
                    cls_path = feats_path.parent / f"{feats_path.stem}_cls.npy"
                    np.save(cls_path, cls_token.numpy())
                    logger.info(f"Saved CLS token to {cls_path}")
                if save_special_tokens and register_tokens is not None:
                    reg_path = feats_path.parent / f"{feats_path.stem}_registers.npy"
                    np.save(reg_path, register_tokens.numpy())
                    logger.info(f"Saved register tokens to {reg_path}")

    predict()

    if rank == 0:
        prog.close()  # type: ignore

    dist.cleanup(distributed=config.distributed)


if __name__ == "__main__":
    main()
