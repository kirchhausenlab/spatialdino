from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pyclesperanto as cle
import tifffile
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

ATTENTION_HEAD_CHANNELS = 6
VORONOI_OTSU_DIRNAME = "voronoi-otsu"
SEGMENTATION_FILENAME = "instance_seg.tif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Voronoi-Otsu segmentation on saved spatialDINO features.")
    parser.add_argument("--input-path", required=True, help="Folder containing per-sample subfolders.")
    parser.add_argument(
        "--enable-voronoi-otsu",
        action="store_true",
        help="Enable the Voronoi-Otsu segmentation branch.",
    )
    parser.add_argument(
        "--gaussian-blur-sigma",
        type=int,
        default=3,
        help="Gaussian blur sigma applied before Voronoi-Otsu labeling.",
    )
    parser.add_argument(
        "--rolling-ball-radius",
        type=float,
        default=10.0,
        help="Rolling-ball radius used for background subtraction.",
    )
    return parser.parse_args()


def list_subfolders(input_path: Path) -> list[Path]:
    subfolders = [path for path in input_path.iterdir() if path.is_dir()]
    subfolders.sort(key=lambda path: (path.name.casefold(), path.name))
    return subfolders


def read_tiff_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path.name} does not contain a readable TIFF volume.")
        shape = tuple(int(dim) for dim in tif.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"{path.name} must be a 3D TIFF volume.")
    return shape


def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _log_kernel(size: tuple[int, int, int] = (7, 7, 7), sigma: float = 1.0) -> np.ndarray:
    x, y, z = size
    xc, yc, zc = (np.array(size) - 1) / 2.0
    xx, yy, zz = np.meshgrid(np.arange(x) - xc, np.arange(y) - yc, np.arange(z) - zc, indexing="ij")
    r2 = xx**2 + yy**2 + zz**2
    normalization = (r2 - 3 * sigma**2) / (sigma**5)
    log_kernel = normalization * np.exp(-r2 / (2 * sigma**2))
    log_kernel -= log_kernel.mean()
    return log_kernel


@torch.no_grad()
def upsample_scalar_volume(
    volume_zyx: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    input_tensor = torch.from_numpy(ensure_contiguous(volume_zyx)).to(device=device, dtype=torch.float32)
    output_tensor = F.interpolate(
        input_tensor.unsqueeze(0).unsqueeze(0),
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return output_tensor.cpu().numpy()


def validate_folder(subfolder: Path) -> tuple[Path, Path]:
    lr_path = subfolder / "lr_feats.npy"
    volume_path = subfolder / "volume_unnorm.tif"
    if not lr_path.is_file():
        raise FileNotFoundError(f"{subfolder.name} is missing lr_feats.npy.")
    if not volume_path.is_file():
        raise FileNotFoundError(f"{subfolder.name} is missing volume_unnorm.tif.")
    return lr_path, volume_path


def segment_subfolder(
    subfolder: Path,
    *,
    gaussian_blur_sigma: int,
    rolling_ball_radius: float,
    device: torch.device,
    cle_device: object,
    log_kernel: np.ndarray,
) -> None:
    lr_path, volume_path = validate_folder(subfolder)
    lr_feats = np.load(lr_path, mmap_mode="r")
    if lr_feats.ndim != 4:
        raise ValueError(f"{lr_path} must be a 4D array with shape [Z, Y, X, C].")

    target_shape = read_tiff_shape(volume_path)

    # Streaming inference appends attention channels after the patch embeddings.
    patch_tokens = lr_feats[..., :-ATTENTION_HEAD_CHANNELS] if lr_feats.shape[-1] > ATTENTION_HEAD_CHANNELS else lr_feats
    patch_tokens_sum = np.asarray(patch_tokens, dtype=np.float32).sum(axis=-1)
    patch_tokens_sum = upsample_scalar_volume(patch_tokens_sum, target_shape=target_shape, device=device)

    background = cle.opening_sphere(
        patch_tokens_sum.astype(np.float32),
        radius_x=rolling_ball_radius,
        radius_y=rolling_ball_radius,
        radius_z=rolling_ball_radius,
        device=cle_device,
    )
    image_bgsub = cle.subtract_images(
        patch_tokens_sum.astype(np.float32),
        background,
        device=cle_device,
    )
    convolved = -cle.convolve(image_bgsub, log_kernel)
    gaussian_blur = cle.gaussian_blur(
        convolved,
        sigma_x=gaussian_blur_sigma,
        sigma_y=gaussian_blur_sigma,
        sigma_z=gaussian_blur_sigma,
        device=cle_device,
    )
    seg = cle.voronoi_otsu_labeling(gaussian_blur, spot_sigma=2, outline_sigma=2, device=cle_device)
    seg_array = np.asarray(seg).astype(np.uint32, copy=False)
    output_dir = subfolder / VORONOI_OTSU_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        subfolder / SEGMENTATION_FILENAME,
        seg_array,
        bigtiff=True,
        metadata=None,
        photometric="minisblack",
    )
    tifffile.imwrite(
        output_dir / SEGMENTATION_FILENAME,
        seg_array,
        bigtiff=True,
        metadata=None,
        photometric="minisblack",
    )


def iter_subfolders(input_path: Path) -> Iterable[Path]:
    for subfolder in list_subfolders(input_path):
        yield subfolder


def main() -> None:
    args = parse_args()
    if not args.enable_voronoi_otsu:
        raise ValueError("Enable Voronoi-Otsu segmentation to run.")

    if args.gaussian_blur_sigma < 0:
        raise ValueError("Gaussian blur sigma must be nonnegative.")
    if args.rolling_ball_radius < 0:
        raise ValueError("Rolling ball radius must be nonnegative.")

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Segmentation requires one GPU.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cle_device = cle.get_device()
    kernel_fixed = np.transpose(_log_kernel((3, 3, 2), 1.0), (2, 1, 0))

    subfolders = list(iter_subfolders(input_path))
    if not subfolders:
        raise ValueError(f"Input folder contains no subfolders: {input_path}")

    print(f"[segmentation] Using torch device {device}", flush=True)
    print(f"[segmentation] Using pyclesperanto device {cle_device}", flush=True)
    print(f"[segmentation] Found {len(subfolders)} subfolders", flush=True)

    for index, subfolder in enumerate(subfolders, start=1):
        print(f"[segmentation] Processing {subfolder.name} ({index}/{len(subfolders)})", flush=True)
        segment_subfolder(
            subfolder,
            gaussian_blur_sigma=int(args.gaussian_blur_sigma),
            rolling_ball_radius=float(args.rolling_ball_radius),
            device=device,
            cle_device=cle_device,
            log_kernel=kernel_fixed,
        )
        print(f"[segmentation] Completed {subfolder.name}", flush=True)
        torch.cuda.empty_cache()

    print("[segmentation] Done", flush=True)


if __name__ == "__main__":
    main()
