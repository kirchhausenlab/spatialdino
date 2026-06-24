from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from spatialdino.inference.output_layout import (
    FEATURE_STATS_DIRNAME,
    HR_FEATS_DIRNAME,
    discover_inference_timepoints,
    process_features_statistics_dir,
    process_features_hr_timepoint_dir,
    process_features_pca_dir,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

ALLOWED_SAVE_FORMATS = {".npy", ".tif"}
DEFAULT_MAX_UPSAMPLE_BYTES = 512 * 1024 * 1024
DEFAULT_PCA_BATCH_BYTES = 256 * 1024 * 1024
FEATURE_STAT_NAMES = ("mean", "max", "min", "median", "std", "l2_norm")
FEATURE_STATS_METADATA_FILENAME = "feature_stats_metadata.json"


@dataclass(frozen=True)
class PcaModel:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true or false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process saved spatialDINO features.")
    parser.add_argument(
        "--input-path",
        required=True,
        help="Inference output folder containing lr_feats/ and raw/.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Root folder where processed outputs are written. Defaults to the input folder.",
    )
    parser.add_argument("--save-pca", action="store_true", help="Save PCA volumes inside pca_<n_components>/.")
    parser.add_argument("--pca-components", type=int, default=3, help="Number of PCA components to save.")
    parser.add_argument("--pca-format", choices=sorted(ALLOWED_SAVE_FORMATS), default=".tif")
    parser.add_argument(
        "--global-pca",
        type=parse_bool,
        default=True,
        help="Fit one PCA basis and one intensity scale across all selected timepoints.",
    )
    parser.add_argument(
        "--save-high-resolution-features",
        action="store_true",
        help="Upsample all low-resolution features and save them inside hr_feats/."
    )
    parser.add_argument("--high-resolution-format", choices=sorted(ALLOWED_SAVE_FORMATS), default=".tif")
    parser.add_argument(
        "--save-feature-statistics",
        action="store_true",
        help="Save compact high-resolution feature statistics inside feature_stats/.",
    )
    parser.add_argument(
        "--ignore-trailing-channels",
        action="store_true",
        help="Ignore the last N feature channels before computing feature statistics.",
    )
    parser.add_argument(
        "--include-trailing-channels",
        dest="ignore_trailing_channels",
        action="store_false",
        help="Use all feature channels when computing feature statistics.",
    )
    parser.set_defaults(ignore_trailing_channels=True)
    parser.add_argument(
        "--trailing-channels",
        type=int,
        default=6,
        help="Number of trailing feature channels to ignore when --ignore-trailing-channels is enabled.",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads used for writing output files."
    )
    parser.add_argument(
        "--file-start",
        type=int,
        default=0,
        help="Zero-based index of the first discovered timepoint to process.",
    )
    parser.add_argument(
        "--file-end",
        type=int,
        default=None,
        help="Exclusive zero-based end index of discovered timepoints to process. Defaults to all remaining timepoints.",
    )
    return parser.parse_args()


def read_tiff_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path.name} does not contain a readable TIFF volume.")
        shape = tuple(int(dim) for dim in tif.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"{path.name} must be a 3D TIFF volume.")
    return shape


def choose_pca_batch_size(channel_count: int) -> int:
    bytes_per_voxel = max(1, channel_count) * np.dtype(np.float32).itemsize
    return max(4096, DEFAULT_PCA_BATCH_BYTES // bytes_per_voxel)


def choose_feature_chunk_size(target_shape: tuple[int, int, int], channel_count: int) -> int:
    voxels = max(1, math.prod(target_shape))
    bytes_per_channel = voxels * np.dtype(np.float32).itemsize
    return max(1, min(channel_count, DEFAULT_MAX_UPSAMPLE_BYTES // max(1, bytes_per_channel)))


def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def validate_lr_features(
    lr_feats: np.ndarray,
    *,
    source_name: str,
    expected_channel_count: int | None = None,
) -> int:
    if lr_feats.ndim != 4:
        raise ValueError(f"{source_name} must be a 4D array with shape [Z, Y, X, C].")
    channel_count = int(lr_feats.shape[-1])
    if expected_channel_count is not None and channel_count != expected_channel_count:
        raise ValueError(
            (
                "All selected timepoints must have the same feature channel count for global PCA. "
                f"Expected {expected_channel_count}, got {channel_count} for {source_name}."
            )
        )
    return channel_count


def iter_feature_batches(
    lr_feats: np.ndarray,
    *,
    channel_count: int,
    batch_size: int,
    device: torch.device,
):
    flat_feats = lr_feats.reshape(-1, channel_count)
    voxel_count = int(flat_feats.shape[0])
    for start in range(0, voxel_count, batch_size):
        end = min(voxel_count, start + batch_size)
        batch_np = ensure_contiguous(flat_feats[start:end])
        yield torch.from_numpy(batch_np).to(device=device, dtype=torch.float32, non_blocking=False)


@torch.no_grad()
def fit_pca_model_from_sources(
    sources,
    *,
    n_components: int,
    device: torch.device,
) -> PcaModel:
    channel_count: int | None = None
    batch_size: int | None = None
    total_count = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    for source_name, lr_feats in sources:
        channel_count = validate_lr_features(
            lr_feats,
            source_name=str(source_name),
            expected_channel_count=channel_count,
        )
        if n_components > channel_count:
            raise ValueError(
                f"PCA components ({n_components}) cannot exceed the number of feature channels ({channel_count})."
            )
        if batch_size is None:
            batch_size = choose_pca_batch_size(channel_count)
            mean = torch.zeros(channel_count, dtype=torch.float64, device=device)
            m2 = torch.zeros((channel_count, channel_count), dtype=torch.float64, device=device)

        assert batch_size is not None
        assert mean is not None
        assert m2 is not None
        for batch in iter_feature_batches(
            lr_feats,
            channel_count=channel_count,
            batch_size=batch_size,
            device=device,
        ):
            batch_count = int(batch.shape[0])
            if batch_count == 0:
                continue

            batch_mean = batch.sum(dim=0, dtype=torch.float64) / batch_count
            batch_mean_f32 = batch_mean.to(dtype=torch.float32)
            if device.type == "cpu":
                centered_batch = batch - batch_mean_f32
                batch_m2 = (centered_batch.transpose(0, 1) @ centered_batch).to(dtype=torch.float64)
            else:
                batch.sub_(batch_mean_f32)
                batch_m2 = (batch.transpose(0, 1) @ batch).to(dtype=torch.float64)

            if total_count == 0:
                mean.copy_(batch_mean)
                m2.copy_(batch_m2)
                total_count = batch_count
                continue

            new_count = total_count + batch_count
            delta = batch_mean - mean
            mean.add_(delta * (batch_count / new_count))
            m2.add_(batch_m2)
            m2.add_(torch.outer(delta, delta) * ((total_count * batch_count) / new_count))
            total_count = new_count

    if channel_count is None or mean is None or m2 is None or total_count == 0:
        raise ValueError("PCA requires at least one feature voxel.")

    denominator = max(1, total_count - 1)
    covariance = m2 / denominator
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:n_components]
    selected_eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order]

    max_abs_indices = torch.argmax(torch.abs(components), dim=0)
    signs = torch.sign(components[max_abs_indices, torch.arange(components.shape[1], device=device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    components = components * signs[None, :]

    return PcaModel(
        mean=mean.to(dtype=torch.float32).cpu().numpy(),
        components=components.to(dtype=torch.float32).cpu().numpy(),
        eigenvalues=selected_eigenvalues.to(dtype=torch.float32).cpu().numpy(),
    )


def fit_pca_model(
    lr_feats: np.ndarray,
    *,
    n_components: int,
    device: torch.device,
) -> PcaModel:
    return fit_pca_model_from_sources(
        [("lr_feats", lr_feats)],
        n_components=n_components,
        device=device,
    )


@torch.no_grad()
def project_pca_volume(
    lr_feats: np.ndarray,
    *,
    pca_model: PcaModel,
    device: torch.device,
) -> np.ndarray:
    channel_count = validate_lr_features(
        lr_feats,
        source_name="lr_feats",
        expected_channel_count=int(pca_model.mean.shape[0]),
    )
    voxel_count = int(np.prod(lr_feats.shape[:-1]))
    n_components = int(pca_model.components.shape[1])
    batch_size = choose_pca_batch_size(channel_count)
    projected = np.empty((voxel_count, n_components), dtype=np.float32)

    mean = torch.from_numpy(pca_model.mean).to(device=device, dtype=torch.float32)
    components = torch.from_numpy(pca_model.components).to(device=device, dtype=torch.float32)

    start = 0
    for batch in iter_feature_batches(
        lr_feats,
        channel_count=channel_count,
        batch_size=batch_size,
        device=device,
    ):
        end = start + int(batch.shape[0])
        transformed = (batch - mean) @ components
        projected[start:end] = transformed.cpu().numpy()
        start = end

    return projected.reshape(*lr_feats.shape[:-1], n_components)


def compute_pca_volume(
    lr_feats: np.ndarray,
    *,
    n_components: int,
    device: torch.device,
) -> np.ndarray:
    pca_model = fit_pca_model(lr_feats, n_components=n_components, device=device)
    return project_pca_volume(lr_feats, pca_model=pca_model, device=device)


@torch.no_grad()
def compute_global_pca_min_max_from_sources(
    sources,
    *,
    pca_model: PcaModel,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    channel_count = int(pca_model.mean.shape[0])
    n_components = int(pca_model.components.shape[1])
    batch_size = choose_pca_batch_size(channel_count)
    mean = torch.from_numpy(pca_model.mean).to(device=device, dtype=torch.float32)
    components = torch.from_numpy(pca_model.components).to(device=device, dtype=torch.float32)
    mins = torch.full((n_components,), torch.inf, dtype=torch.float32, device=device)
    maxs = torch.full((n_components,), -torch.inf, dtype=torch.float32, device=device)

    seen_voxels = 0
    for source_name, lr_feats in sources:
        validate_lr_features(
            lr_feats,
            source_name=str(source_name),
            expected_channel_count=channel_count,
        )
        for batch in iter_feature_batches(
            lr_feats,
            channel_count=channel_count,
            batch_size=batch_size,
            device=device,
        ):
            if batch.shape[0] == 0:
                continue
            transformed = (batch - mean) @ components
            mins = torch.minimum(mins, transformed.amin(dim=0))
            maxs = torch.maximum(maxs, transformed.amax(dim=0))
            seen_voxels += int(batch.shape[0])

    if seen_voxels == 0:
        raise ValueError("PCA range computation requires at least one feature voxel.")

    return mins.cpu().numpy(), maxs.cpu().numpy()


@torch.no_grad()
def upsample_channels(
    channels_zyx: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    input_tensor = torch.from_numpy(ensure_contiguous(channels_zyx)).to(device=device, dtype=torch.float32)
    output_tensor = F.interpolate(
        input_tensor.unsqueeze(0),
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    return output_tensor.cpu().numpy()


def normalize_channels_to_uint8(
    channels_zyx: np.ndarray,
    *,
    mins: np.ndarray | None = None,
    maxs: np.ndarray | None = None,
) -> np.ndarray:
    if mins is None or maxs is None:
        mins = channels_zyx.min(axis=(1, 2, 3), keepdims=True)
        maxs = channels_zyx.max(axis=(1, 2, 3), keepdims=True)
    else:
        mins = np.asarray(mins, dtype=np.float32).reshape(-1, 1, 1, 1)
        maxs = np.asarray(maxs, dtype=np.float32).reshape(-1, 1, 1, 1)
    scaled = (channels_zyx - mins) / (maxs - mins + 1e-6)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def save_volume(path: Path, array: np.ndarray, save_format: str) -> None:
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
    if array.ndim == 4:
        tifffile.imwrite(
            path,
            array,
            bigtiff=True,
            metadata=None,
            photometric="minisblack",
            planarconfig="contig",
        )
        return
    tifffile.imwrite(path, array, bigtiff=True, metadata=None, photometric="minisblack")


def wait_for_futures(futures: deque[Future[None]]) -> None:
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
    futures.append(executor.submit(save_volume, path, array, save_format))
    if len(futures) >= limit:
        futures.popleft().result()


def export_pca(
    output_root: Path,
    timepoint_name: str,
    lr_feats: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    n_components: int,
    save_format: str,
    device: torch.device,
    pca_model: PcaModel | None = None,
    pca_mins: np.ndarray | None = None,
    pca_maxs: np.ndarray | None = None,
) -> None:
    if pca_model is None:
        pca_lr = compute_pca_volume(lr_feats, n_components=n_components, device=device)
    else:
        pca_lr = project_pca_volume(lr_feats, pca_model=pca_model, device=device)
    pca_lr_channels = np.moveaxis(pca_lr, -1, 0)
    pca_hr_channels = upsample_channels(pca_lr_channels, target_shape=target_shape, device=device)
    pca_hr_uint8 = normalize_channels_to_uint8(pca_hr_channels, mins=pca_mins, maxs=pca_maxs)
    save_path = process_features_pca_dir(output_root, n_components) / f"{timepoint_name}{save_format}"
    if save_format == ".npy":
        save_volume(save_path, np.moveaxis(pca_hr_uint8, 0, -1), save_format)
        return

    if n_components == 1:
        save_volume(save_path, pca_hr_uint8[0], save_format)
        return

    save_volume(save_path, np.moveaxis(pca_hr_uint8, 0, -1), save_format)


def export_high_resolution_features(
    output_root: Path,
    timepoint_name: str,
    lr_feats: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    save_format: str,
    device: torch.device,
    io_workers: int,
) -> None:
    channel_count = int(lr_feats.shape[-1])
    channel_width = max(3, len(str(channel_count - 1)))
    chunk_size = choose_feature_chunk_size(target_shape, channel_count)
    hr_dir = process_features_hr_timepoint_dir(output_root, timepoint_name)
    if hr_dir.exists():
        shutil.rmtree(hr_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)

    limit = max(1, io_workers * 2)
    futures: deque[Future[None]] = deque()
    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        for start in range(0, channel_count, chunk_size):
            end = min(channel_count, start + chunk_size)
            lr_chunk = np.moveaxis(lr_feats[..., start:end], -1, 0)
            hr_chunk = upsample_channels(lr_chunk, target_shape=target_shape, device=device)
            hr_chunk_uint8 = normalize_channels_to_uint8(hr_chunk)
            for offset, feature_index in enumerate(range(start, end)):
                feature_path = hr_dir / f"feature_{feature_index:0{channel_width}d}{save_format}"
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


def feature_statistics_channel_slice(
    channel_count: int,
    *,
    ignore_trailing_channels: bool,
    trailing_channels: int,
) -> tuple[slice, int]:
    if trailing_channels < 0:
        raise ValueError("Trailing channels must be nonnegative.")
    if not ignore_trailing_channels or trailing_channels == 0:
        return slice(None), channel_count
    if trailing_channels >= channel_count:
        raise ValueError(
            f"Cannot ignore {trailing_channels} trailing channels from a feature array with {channel_count} channels."
        )
    return slice(0, channel_count - trailing_channels), channel_count - trailing_channels


def compute_feature_statistics_lr(
    lr_feats: np.ndarray,
    *,
    ignore_trailing_channels: bool,
    trailing_channels: int,
) -> tuple[np.ndarray, int, int]:
    channel_count = int(lr_feats.shape[-1])
    channel_slice, computed_channel_count = feature_statistics_channel_slice(
        channel_count,
        ignore_trailing_channels=ignore_trailing_channels,
        trailing_channels=trailing_channels,
    )
    features = np.asarray(lr_feats[..., channel_slice], dtype=np.float32)
    mean = np.mean(features, axis=-1, dtype=np.float32)
    max_values = np.max(features, axis=-1)
    min_values = np.min(features, axis=-1)
    median = np.median(features, axis=-1).astype(np.float32, copy=False)
    std = np.std(features, axis=-1, dtype=np.float32)
    l2_norm = np.sqrt(np.einsum("...c,...c->...", features, features, dtype=np.float32))
    stats = np.stack((mean, max_values, min_values, median, std, l2_norm), axis=0).astype(np.float32, copy=False)
    return stats, channel_count, computed_channel_count


def write_feature_statistics_metadata(
    output_root: Path,
    *,
    ignore_trailing_channels: bool,
    trailing_channels: int,
    input_channel_count: int,
    computed_channel_count: int,
) -> None:
    metadata = {
        "components": list(FEATURE_STAT_NAMES),
        "dtype": "float32",
        "source": "lr_feats",
        "ignored_trailing_channels": bool(ignore_trailing_channels),
        "trailing_channels": int(trailing_channels),
        "input_channel_count": int(input_channel_count),
        "computed_channel_count": int(computed_channel_count),
        "computed_before_upsampling": True,
    }
    output_path = process_features_statistics_dir(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / FEATURE_STATS_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_feature_statistics(
    output_root: Path,
    timepoint_name: str,
    lr_feats: np.ndarray,
    *,
    target_shape: tuple[int, int, int],
    ignore_trailing_channels: bool,
    trailing_channels: int,
    device: torch.device,
) -> tuple[int, int]:
    stats_lr_channels, input_channel_count, computed_channel_count = compute_feature_statistics_lr(
        lr_feats,
        ignore_trailing_channels=ignore_trailing_channels,
        trailing_channels=trailing_channels,
    )
    stats_hr_channels = upsample_channels(stats_lr_channels, target_shape=target_shape, device=device).astype(
        np.float32,
        copy=False,
    )
    save_path = process_features_statistics_dir(output_root) / f"{timepoint_name}.tif"
    save_volume(save_path, np.moveaxis(stats_hr_channels, 0, -1), ".tif")
    return input_channel_count, computed_channel_count


def remove_output_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


def list_requested_output_paths(
    output_path: Path,
    *,
    save_pca: bool,
    pca_components: int,
    save_high_resolution_features: bool,
    save_feature_statistics: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    if save_high_resolution_features:
        paths.append(output_path / HR_FEATS_DIRNAME)
    if save_pca:
        paths.append(process_features_pca_dir(output_path, pca_components))
    if save_feature_statistics:
        paths.append(output_path / FEATURE_STATS_DIRNAME)
    return paths


def cleanup_output_root(
    output_path: Path,
    *,
    save_pca: bool,
    pca_components: int,
    save_high_resolution_features: bool,
    save_feature_statistics: bool = False,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for path in list_requested_output_paths(
        output_path,
        save_pca=save_pca,
        pca_components=pca_components,
        save_high_resolution_features=save_high_resolution_features,
        save_feature_statistics=save_feature_statistics,
    ):
        remove_output_path(path)


def select_timepoints(timepoints: list, *, file_start: int, file_end: int | None) -> list:
    timepoint_count = len(timepoints)
    if file_start < 0 or file_start >= timepoint_count:
        raise ValueError(f"Start file must be between 0 and {timepoint_count - 1}.")
    if file_end is not None and (file_end < 0 or file_end > timepoint_count):
        raise ValueError(f"End file must be between 0 and {timepoint_count}.")

    effective_file_end = timepoint_count if file_end is None else file_end
    if effective_file_end <= file_start:
        raise ValueError("Chosen files leave zero timepoints to process.")

    return timepoints[file_start:effective_file_end]


def process_timepoint(
    timepoint_name: str,
    lr_path: Path,
    volume_path: Path,
    output_root: Path,
    *,
    save_pca: bool,
    pca_components: int,
    pca_format: str,
    save_high_resolution_features: bool,
    high_resolution_format: str,
    save_feature_statistics: bool,
    ignore_trailing_channels: bool,
    trailing_channels: int,
    device: torch.device,
    io_workers: int,
    pca_model: PcaModel | None = None,
    pca_mins: np.ndarray | None = None,
    pca_maxs: np.ndarray | None = None,
) -> None:
    lr_feats = np.load(lr_path, mmap_mode="r")
    validate_lr_features(lr_feats, source_name=str(lr_path))

    target_shape = read_tiff_shape(volume_path)
    if save_pca:
        export_pca(
            output_root,
            timepoint_name,
            lr_feats,
            target_shape=target_shape,
            n_components=pca_components,
            save_format=pca_format,
            device=device,
            pca_model=pca_model,
            pca_mins=pca_mins,
            pca_maxs=pca_maxs,
        )
    if save_high_resolution_features:
        export_high_resolution_features(
            output_root,
            timepoint_name,
            lr_feats,
            target_shape=target_shape,
            save_format=high_resolution_format,
            device=device,
            io_workers=io_workers,
        )
    if save_feature_statistics:
        export_feature_statistics(
            output_root,
            timepoint_name,
            lr_feats,
            target_shape=target_shape,
            ignore_trailing_channels=ignore_trailing_channels,
            trailing_channels=trailing_channels,
            device=device,
        )


def lr_feature_sources_for_timepoints(timepoints):
    for timepoint in timepoints:
        yield timepoint.name, np.load(timepoint.lr_path, mmap_mode="r")


def validate_feature_statistics_sources(
    timepoints,
    *,
    ignore_trailing_channels: bool,
    trailing_channels: int,
) -> tuple[int, int]:
    input_channel_count: int | None = None
    computed_channel_count: int | None = None
    for timepoint in timepoints:
        lr_feats = np.load(timepoint.lr_path, mmap_mode="r")
        channel_count = validate_lr_features(lr_feats, source_name=str(timepoint.lr_path))
        if input_channel_count is not None and channel_count != input_channel_count:
            raise ValueError(
                (
                    "All selected timepoints must have the same feature channel count for feature statistics. "
                    f"Expected {input_channel_count}, got {channel_count} for {timepoint.lr_path}."
                )
            )
        _channel_slice, current_computed_count = feature_statistics_channel_slice(
            channel_count,
            ignore_trailing_channels=ignore_trailing_channels,
            trailing_channels=trailing_channels,
        )
        input_channel_count = channel_count
        computed_channel_count = current_computed_count

    if input_channel_count is None or computed_channel_count is None:
        raise ValueError("Feature statistics require at least one feature volume.")
    return input_channel_count, computed_channel_count


def main() -> None:
    args = parse_args()
    if not args.save_pca and not args.save_high_resolution_features and not args.save_feature_statistics:
        raise ValueError("Choose at least one output: PCA, high-resolution features, and/or feature statistics.")
    if int(args.trailing_channels) < 0:
        raise ValueError("Trailing channels must be nonnegative.")

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_path}")
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else input_path
    if output_path.exists() and not output_path.is_dir():
        raise FileNotFoundError(f"Output folder exists but is not a directory: {output_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Process-features requires one GPU.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    discovered_timepoints = discover_inference_timepoints(input_path)
    timepoints = select_timepoints(
        discovered_timepoints,
        file_start=int(args.file_start),
        file_end=args.file_end,
    )
    cleanup_output_root(
        output_path,
        save_pca=bool(args.save_pca),
        pca_components=int(args.pca_components),
        save_high_resolution_features=bool(args.save_high_resolution_features),
        save_feature_statistics=bool(args.save_feature_statistics),
    )

    print(f"[process-features] Using device {device}", flush=True)
    print(f"[process-features] Found {len(timepoints)} timepoints", flush=True)

    pca_model: PcaModel | None = None
    pca_mins: np.ndarray | None = None
    pca_maxs: np.ndarray | None = None
    if bool(args.save_pca) and bool(args.global_pca):
        print("[process-features] Fitting global PCA", flush=True)
        pca_model = fit_pca_model_from_sources(
            lr_feature_sources_for_timepoints(timepoints),
            n_components=int(args.pca_components),
            device=device,
        )
        print("[process-features] Scanning global PCA ranges", flush=True)
        pca_mins, pca_maxs = compute_global_pca_min_max_from_sources(
            lr_feature_sources_for_timepoints(timepoints),
            pca_model=pca_model,
            device=device,
        )

    if bool(args.save_feature_statistics):
        input_channel_count, computed_channel_count = validate_feature_statistics_sources(
            timepoints,
            ignore_trailing_channels=bool(args.ignore_trailing_channels),
            trailing_channels=int(args.trailing_channels),
        )
        write_feature_statistics_metadata(
            output_path,
            ignore_trailing_channels=bool(args.ignore_trailing_channels),
            trailing_channels=int(args.trailing_channels),
            input_channel_count=input_channel_count,
            computed_channel_count=computed_channel_count,
        )

    for index, timepoint in enumerate(timepoints, start=1):
        print(f"[process-features] Processing {timepoint.name} ({index}/{len(timepoints)})", flush=True)
        process_timepoint(
            timepoint.name,
            timepoint.lr_path,
            timepoint.raw_path,
            output_path,
            save_pca=bool(args.save_pca),
            pca_components=int(args.pca_components),
            pca_format=args.pca_format,
            save_high_resolution_features=bool(args.save_high_resolution_features),
            high_resolution_format=args.high_resolution_format,
            save_feature_statistics=bool(args.save_feature_statistics),
            ignore_trailing_channels=bool(args.ignore_trailing_channels),
            trailing_channels=int(args.trailing_channels),
            device=device,
            io_workers=max(1, int(args.io_workers)),
            pca_model=pca_model,
            pca_mins=pca_mins,
            pca_maxs=pca_maxs,
        )
        print(f"[process-features] Completed {timepoint.name}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[process-features] Done", flush=True)


if __name__ == "__main__":
    main()
