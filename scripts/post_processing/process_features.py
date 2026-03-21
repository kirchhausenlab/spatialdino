"""PCA reduction and high-resolution feature export for spatialDINO outputs.

Given a folder of per-sample subfolders each containing ``lr_feats.npy``
(low-resolution features, shape [Z, Y, X, C]) and ``volume_unnorm.tif``
(the original 3D volume), this script can:

* Compute PCA on the feature channels and save upsampled PCA volumes.
* Upsample every individual feature channel to the original volume
  resolution and save it as a separate file.

Both outputs are written as TIFF or NumPy files, controlled via CLI flags.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

ALLOWED_SAVE_FORMATS = {".npy", ".tif"}
DEFAULT_MAX_UPSAMPLE_BYTES = 512 * 1024 * 1024
DEFAULT_PCA_BATCH_BYTES = 256 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature processing.

    Returns:
        Namespace with input/output paths, PCA settings, and I/O options.
    """
    parser = argparse.ArgumentParser(description="Process saved spatialDINO features.")
    parser.add_argument(
        "--input-path", required=True, help="Folder containing per-sample subfolders."
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Folder where processed outputs are written. Defaults to the input folder.",
    )
    parser.add_argument(
        "--save-pca",
        action="store_true",
        help="Save PCA volumes inside each sample folder.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of PCA components to save.",
    )
    parser.add_argument(
        "--pca-format", choices=sorted(ALLOWED_SAVE_FORMATS), default=".tif"
    )
    parser.add_argument(
        "--save-high-resolution-features",
        action="store_true",
        help="Upsample all low-resolution features and save them inside hr_feats/.",
    )
    parser.add_argument(
        "--high-resolution-format", choices=sorted(ALLOWED_SAVE_FORMATS), default=".tif"
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads used for writing output files.",
    )
    return parser.parse_args()


def list_subfolders(input_path: Path) -> list[Path]:
    """Return all immediate subdirectories of *input_path*, sorted case-insensitively.

    Args:
        input_path: Parent directory to scan.

    Returns:
        Sorted list of subdirectory paths.
    """
    subfolders = [path for path in input_path.iterdir() if path.is_dir()]
    subfolders.sort(key=lambda path: (path.name.casefold(), path.name))
    return subfolders


def read_tiff_shape(path: Path) -> tuple[int, int, int]:
    """Read the spatial shape of a 3D TIFF volume without loading its data.

    Args:
        path: Path to a 3D TIFF file.

    Returns:
        Tuple of (Z, Y, X) dimensions.

    Raises:
        ValueError: If the file is not a valid 3D volume.
    """
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path.name} does not contain a readable TIFF volume.")
        shape = tuple(int(dim) for dim in tif.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"{path.name} must be a 3D TIFF volume.")
    return shape


def choose_pca_batch_size(channel_count: int) -> int:
    """Choose the number of voxels to process per batch during PCA.

    Args:
        channel_count: Number of feature channels.

    Returns:
        Batch size in voxels that fits within ``DEFAULT_PCA_BATCH_BYTES``.
    """
    bytes_per_voxel = max(1, channel_count) * np.dtype(np.float32).itemsize
    return max(4096, DEFAULT_PCA_BATCH_BYTES // bytes_per_voxel)


def choose_feature_chunk_size(
    target_shape: tuple[int, int, int], channel_count: int
) -> int:
    """Choose how many channels to upsample at once during HR export.

    Args:
        target_shape: Target (Z, Y, X) volume dimensions.
        channel_count: Total number of feature channels.

    Returns:
        Number of channels per upsampling chunk.
    """
    voxels = max(1, math.prod(target_shape))
    bytes_per_channel = voxels * np.dtype(np.float32).itemsize
    return max(
        1, min(channel_count, DEFAULT_MAX_UPSAMPLE_BYTES // max(1, bytes_per_channel))
    )


def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous copy of *array* if it is not already contiguous.

    Args:
        array: Input NumPy array.

    Returns:
        C-contiguous array (may be the same object if already contiguous).
    """
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


@torch.no_grad()
def compute_pca_volume(
    lr_feats: np.ndarray,
    *,
    n_components: int,
    device: torch.device,
) -> np.ndarray:
    """Compute PCA on low-resolution features and project onto top components.

    Uses a batched covariance approach to avoid loading all voxels into
    GPU memory at once.  Eigendecomposition is performed on the full
    covariance matrix and the top *n_components* eigenvectors are used
    for projection.

    Args:
        lr_feats: Feature array of shape [Z, Y, X, C].
        n_components: Number of principal components to keep.
        device: Torch device for computation.

    Returns:
        Projected array of shape [Z, Y, X, n_components].
    """
    flat_feats = ensure_contiguous(lr_feats.reshape(-1, lr_feats.shape[-1]))
    voxel_count, channel_count = flat_feats.shape
    if n_components > channel_count:
        raise ValueError(
            f"PCA components ({n_components}) cannot exceed the number of feature channels ({channel_count})."
        )

    batch_size = choose_pca_batch_size(channel_count)
    sum_vec = torch.zeros(channel_count, dtype=torch.float32, device=device)
    sum_outer = torch.zeros(
        (channel_count, channel_count), dtype=torch.float32, device=device
    )

    for start in range(0, voxel_count, batch_size):
        end = min(voxel_count, start + batch_size)
        batch_np = ensure_contiguous(flat_feats[start:end])
        batch = torch.from_numpy(batch_np).to(
            device=device, dtype=torch.float32, non_blocking=False
        )
        sum_vec += batch.sum(dim=0)
        sum_outer += batch.transpose(0, 1) @ batch

    mean = sum_vec / max(1, voxel_count)
    denominator = max(1, voxel_count - 1)
    covariance = (sum_outer - voxel_count * torch.outer(mean, mean)) / denominator
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[
        :, torch.argsort(eigenvalues, descending=True)[:n_components]
    ]

    projected = np.empty((voxel_count, n_components), dtype=np.float32)
    for start in range(0, voxel_count, batch_size):
        end = min(voxel_count, start + batch_size)
        batch_np = ensure_contiguous(flat_feats[start:end])
        batch = torch.from_numpy(batch_np).to(
            device=device, dtype=torch.float32, non_blocking=False
        )
        transformed = (batch - mean) @ components
        projected[start:end] = transformed.cpu().numpy()

    return projected.reshape(*lr_feats.shape[:-1], n_components)


@torch.no_grad()
def upsample_channels(
    channels_zyx: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    """Upsample a multi-channel 3D volume to *target_shape* via trilinear interpolation.

    Args:
        channels_zyx: Array of shape [C, Z, Y, X].
        target_shape: Desired (Z, Y, X) output dimensions.
        device: Torch device for interpolation.

    Returns:
        Upsampled array of shape [C, Z', Y', X'].
    """
    input_tensor = torch.from_numpy(ensure_contiguous(channels_zyx)).to(
        device=device, dtype=torch.float32
    )
    output_tensor = F.interpolate(
        input_tensor.unsqueeze(0),
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    return output_tensor.cpu().numpy()


def normalize_channels_to_uint8(channels_zyx: np.ndarray) -> np.ndarray:
    """Min-max normalize each channel independently and scale to 0-255 uint8.

    Args:
        channels_zyx: Array of shape [C, Z, Y, X].

    Returns:
        uint8 array of the same shape, with each channel in [0, 255].
    """
    mins = channels_zyx.min(axis=(1, 2, 3), keepdims=True)
    maxs = channels_zyx.max(axis=(1, 2, 3), keepdims=True)
    scaled = (channels_zyx - mins) / (maxs - mins + 1e-6)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def save_volume(path: Path, array: np.ndarray, save_format: str) -> None:
    """Save a volume array to disk as either ``.npy`` or ``.tif``.

    Args:
        path: Destination file path.
        array: Volume data to write.
        save_format: ``".npy"`` or ``".tif"``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == ".npy":
        np.save(path, array)
        return
    if array.ndim == 4 and array.shape[-1] == 3:
        tifffile.imwrite(
            path,
            array,
            bigtiff=True,
            metadata=None,
            photometric="rgb",
            planarconfig="contig",
        )
        return
    tifffile.imwrite(path, array, bigtiff=True, metadata=None)


def wait_for_futures(futures: deque[Future[None]]) -> None:
    """Block until all queued futures have completed.

    Args:
        futures: Deque of pending futures to drain.
    """
    while futures:
        futures.popleft().result()


def submit_save_task(
    executor: ThreadPoolExecutor,
    futures: deque[Future[None]],
    *,
    limit: int,
    path: Path,
    array: np.ndarray,
    save_format: str,
) -> None:
    """Submit a save task to the thread pool, blocking if the queue is full.

    Args:
        executor: Thread pool used for async I/O.
        futures: Deque tracking outstanding writes.
        limit: Maximum number of outstanding futures before blocking.
        path: Destination file path.
        array: Data to save.
        save_format: ``".npy"`` or ``".tif"``.
    """
    futures.append(executor.submit(save_volume, path, array, save_format))
    if len(futures) >= limit:
        futures.popleft().result()


def export_pca(
    subfolder: Path,
    lr_feats: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    n_components: int,
    save_format: str,
    device: torch.device,
) -> None:
    """Compute PCA, upsample, normalize to uint8, and save.

    Args:
        subfolder: Output directory for the PCA file(s).
        lr_feats: Low-resolution features [Z, Y, X, C].
        target_shape: High-resolution (Z, Y, X) dimensions.
        n_components: Number of PCA components.
        save_format: ``".npy"`` or ``".tif"``.
        device: Torch device for computation.
    """
    pca_lr = compute_pca_volume(lr_feats, n_components=n_components, device=device)
    pca_lr_channels = np.moveaxis(pca_lr, -1, 0)
    pca_hr_channels = upsample_channels(
        pca_lr_channels, target_shape=target_shape, device=device
    )
    pca_hr_uint8 = normalize_channels_to_uint8(pca_hr_channels)
    save_path = subfolder / f"PCA_{n_components}{save_format}"
    if save_format == ".npy":
        save_volume(save_path, np.moveaxis(pca_hr_uint8, 0, -1), save_format)
        return

    if n_components == 1:
        save_volume(save_path, pca_hr_uint8[0], save_format)
        return

    if n_components == 3:
        save_volume(save_path, np.moveaxis(pca_hr_uint8, 0, -1), save_format)
        return

    component_width = max(2, len(str(n_components - 1)))
    for component_index in range(n_components):
        component_path = (
            subfolder
            / f"PCA_{n_components}_component_{component_index:0{component_width}d}{save_format}"
        )
        save_volume(component_path, pca_hr_uint8[component_index], save_format)


def export_high_resolution_features(
    subfolder: Path,
    lr_feats: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    save_format: str,
    device: torch.device,
    io_workers: int,
) -> None:
    """Upsample every feature channel to full resolution and save individually.

    Channels are processed in memory-bounded chunks and written to the
    ``hr_feats/`` subdirectory using a thread pool for overlapping I/O.

    Args:
        subfolder: Output directory (``hr_feats/`` is created inside it).
        lr_feats: Low-resolution features [Z, Y, X, C].
        target_shape: High-resolution (Z, Y, X) dimensions.
        save_format: ``".npy"`` or ``".tif"``.
        device: Torch device for interpolation.
        io_workers: Number of I/O threads for parallel writes.
    """
    channel_count = int(lr_feats.shape[-1])
    channel_width = max(3, len(str(channel_count - 1)))
    chunk_size = choose_feature_chunk_size(target_shape, channel_count)
    hr_dir = subfolder / "hr_feats"
    if hr_dir.exists():
        shutil.rmtree(hr_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)

    limit = max(1, io_workers * 2)
    futures: deque[Future[None]] = deque()
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        for start in range(0, channel_count, chunk_size):
            end = min(channel_count, start + chunk_size)
            lr_chunk = np.moveaxis(lr_feats[..., start:end], -1, 0)
            hr_chunk = upsample_channels(
                lr_chunk, target_shape=target_shape, device=device
            )
            hr_chunk_uint8 = normalize_channels_to_uint8(hr_chunk)
            for offset, feature_index in enumerate(range(start, end)):
                feature_path = (
                    hr_dir / f"feature_{feature_index:0{channel_width}d}{save_format}"
                )
                feature_volume = ensure_contiguous(hr_chunk_uint8[offset])
                submit_save_task(
                    executor,
                    futures,
                    limit=limit,
                    path=feature_path,
                    array=feature_volume,
                    save_format=save_format,
                )
        wait_for_futures(futures)


def validate_folder(subfolder: Path) -> tuple[Path, Path]:
    """Check that a sample subfolder contains the required input files.

    Args:
        subfolder: Path to a per-sample directory.

    Returns:
        Tuple of (lr_feats path, volume_unnorm path).

    Raises:
        FileNotFoundError: If either expected file is missing.
    """
    lr_path = subfolder / "lr_feats.npy"
    volume_path = subfolder / "volume_unnorm.tif"
    if not lr_path.is_file():
        raise FileNotFoundError(f"{subfolder.name} is missing lr_feats.npy.")
    if not volume_path.is_file():
        raise FileNotFoundError(f"{subfolder.name} is missing volume_unnorm.tif.")
    return lr_path, volume_path


def process_subfolder(
    input_subfolder: Path,
    output_subfolder: Path,
    *,
    save_pca: bool,
    pca_components: int,
    pca_format: str,
    save_high_resolution_features: bool,
    high_resolution_format: str,
    device: torch.device,
    io_workers: int,
) -> None:
    """Process a single sample: optionally export PCA and/or HR features.

    Args:
        input_subfolder: Directory containing ``lr_feats.npy`` and ``volume_unnorm.tif``.
        output_subfolder: Destination directory for outputs.
        save_pca: Whether to export PCA volumes.
        pca_components: Number of PCA components.
        pca_format: Save format for PCA output.
        save_high_resolution_features: Whether to export HR feature channels.
        high_resolution_format: Save format for HR features.
        device: Torch device.
        io_workers: Number of I/O threads.
    """
    lr_path, volume_path = validate_folder(input_subfolder)
    lr_feats = np.load(lr_path, mmap_mode="r")
    if lr_feats.ndim != 4:
        raise ValueError(f"{lr_path} must be a 4D array with shape [Z, Y, X, C].")

    target_shape = read_tiff_shape(volume_path)
    if save_pca:
        export_pca(
            output_subfolder,
            lr_feats,
            target_shape=target_shape,
            n_components=pca_components,
            save_format=pca_format,
            device=device,
        )
    if save_high_resolution_features:
        export_high_resolution_features(
            output_subfolder,
            lr_feats,
            target_shape=target_shape,
            save_format=high_resolution_format,
            device=device,
            io_workers=io_workers,
        )


def iter_subfolders(input_path: Path) -> Iterable[Path]:
    """Yield subfolders from *input_path* in sorted order.

    Args:
        input_path: Parent directory to iterate.

    Yields:
        Each subdirectory path.
    """
    for subfolder in list_subfolders(input_path):
        yield subfolder


def main() -> None:
    """Entry point: parse arguments and process all sample subfolders.

    Iterates over every subfolder in the input directory and runs PCA
    export and/or high-resolution feature export as requested.
    """
    args = parse_args()
    if not args.save_pca and not args.save_high_resolution_features:
        raise ValueError(
            "Choose at least one output: PCA and/or high-resolution features."
        )

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist or is not a directory: {input_path}"
        )
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else input_path
    )
    if output_path.exists() and not output_path.is_dir():
        raise FileNotFoundError(
            f"Output folder exists but is not a directory: {output_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Process-features requires one GPU.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    subfolders = list(iter_subfolders(input_path))
    if not subfolders:
        raise ValueError(f"Input folder contains no subfolders: {input_path}")

    print(f"[process-features] Using device {device}", flush=True)
    print(f"[process-features] Found {len(subfolders)} subfolders", flush=True)

    for index, subfolder in enumerate(subfolders, start=1):
        print(
            f"[process-features] Processing {subfolder.name} ({index}/{len(subfolders)})",
            flush=True,
        )
        destination_subfolder = (
            subfolder if output_path == input_path else output_path / subfolder.name
        )
        process_subfolder(
            subfolder,
            destination_subfolder,
            save_pca=bool(args.save_pca),
            pca_components=int(args.pca_components),
            pca_format=args.pca_format,
            save_high_resolution_features=bool(args.save_high_resolution_features),
            high_resolution_format=args.high_resolution_format,
            device=device,
            io_workers=max(1, int(args.io_workers)),
        )
        print(f"[process-features] Completed {subfolder.name}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[process-features] Done", flush=True)


if __name__ == "__main__":
    main()
