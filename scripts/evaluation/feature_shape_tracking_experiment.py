from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm


METHOD_FEATURE_MEAN = "feature_mean"
METHOD_SLICED_WASSERSTEIN = "sliced_wasserstein"
METHOD_SLICED_WASSERSTEIN_SHAPE = "sliced_wasserstein_shape"
DEFAULT_METHODS = (
    METHOD_FEATURE_MEAN,
    METHOD_SLICED_WASSERSTEIN,
    METHOD_SLICED_WASSERSTEIN_SHAPE,
)
TRACK_COLUMNS = ("track_id", "start", "t", "x", "y", "z", "A", "track_length")


def _log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[feature-shape-tracking] {message}", flush=True)


def _progress_bar(
    iterable: Iterable[Any], *, total: int | None, desc: str, enabled: bool
) -> Iterable[Any]:
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


@dataclass(frozen=True)
class ExperimentConfig:
    n_features: int | str = 384
    samples_per_object: int = 128
    n_feature_projections: int = 64
    n_shape_pairs: int = 512
    n_shape_quantiles: int = 64
    shape_weight: float = 0.1
    methods: tuple[str, ...] = DEFAULT_METHODS
    seed: int = 12345
    feature_channel_block: int = 64
    object_batch_size: int = 256
    sample_batch_size: int = 131_072
    cost_block_rows: int = 1024
    mask_workers: int = max(1, min(8, os.cpu_count() or 1))
    signature_workers: int = 1
    devices: tuple[str, ...] = ("auto",)
    pair_device: str = "auto"
    use_float16_cost: bool = False
    torch_threads_per_worker: int = 1
    max_frames: int | None = None
    max_adjacent_pairs: int | None = None
    search_radius_enabled: bool = True
    search_radius_xy: float = 100.0
    search_radius_z: float = 50.0
    progress: bool = True


@dataclass(frozen=True)
class FramePaths:
    index: int
    name: str
    lr_features_path: Path
    segmentation_path: Path
    raw_path: Path | None


@dataclass
class FrameMetadata:
    index: int
    name: str
    high_shape_yxz: tuple[int, int, int]
    labels: np.ndarray
    volumes: np.ndarray
    amplitudes: np.ndarray
    centroids_yxz: np.ndarray
    sample_coords_yxz: np.ndarray
    shape_signature: np.ndarray


@dataclass
class FrameSignatures:
    index: int
    name: str
    labels: np.ndarray
    volumes: np.ndarray
    amplitudes: np.ndarray
    centroids_yxz: np.ndarray
    feature_mean: np.ndarray
    feature_sw_signature: np.ndarray
    shape_signature: np.ndarray


@dataclass(frozen=True)
class SearchRadiusGate:
    enabled: bool
    radius_xy: float
    radius_z: float
    allowed: np.ndarray
    strict_allowed: np.ndarray
    relaxed_ref_rows: np.ndarray
    delta_yxz: np.ndarray


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    frame_count: int
    summary: pd.DataFrame
    per_object: pd.DataFrame
    assignments: pd.DataFrame
    link_metrics: pd.DataFrame
    tracks: dict[str, pd.DataFrame]
    track_points: pd.DataFrame
    timings: pd.DataFrame


def _natural_key(value: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _read_tiff_yxz(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF at {path}, got shape {array.shape}.")
    return np.moveaxis(array, 0, -1)


def discover_frames(
    input_path: str | Path, segmentation_path: str | Path
) -> list[FramePaths]:
    input_path = Path(input_path).expanduser().resolve()
    segmentation_path = Path(segmentation_path).expanduser().resolve()
    lr_dir = input_path / "lr_feats"
    raw_dir = input_path / "raw"
    if not lr_dir.is_dir():
        raise FileNotFoundError(f"Missing feature folder: {lr_dir}")
    if not segmentation_path.is_dir():
        raise FileNotFoundError(f"Missing segmentation folder: {segmentation_path}")

    feature_paths = sorted(
        lr_dir.glob("*.npy"), key=lambda path: _natural_key(path.stem)
    )
    frames: list[FramePaths] = []
    for index, feature_path in enumerate(feature_paths):
        mask_candidates = [
            segmentation_path / f"{feature_path.stem}.tif",
            segmentation_path / f"{feature_path.stem}.tiff",
        ]
        mask_path = next((path for path in mask_candidates if path.is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(
                f"Missing GT segmentation mask for {feature_path.stem}."
            )
        raw_candidates = [
            raw_dir / f"{feature_path.stem}.tif",
            raw_dir / f"{feature_path.stem}.tiff",
        ]
        raw_path = next((path for path in raw_candidates if path.is_file()), None)
        frames.append(
            FramePaths(
                index=index,
                name=feature_path.stem,
                lr_features_path=feature_path,
                segmentation_path=mask_path,
                raw_path=raw_path,
            )
        )
    if len(frames) < 2:
        raise ValueError("At least two frames are required.")
    return frames


def infer_feature_count(frame_paths: Sequence[FramePaths]) -> int:
    if not frame_paths:
        raise ValueError("No frames available.")
    first = np.load(frame_paths[0].lr_features_path, mmap_mode="r")
    if first.ndim != 4:
        raise ValueError(
            f"Expected [Z, Y, X, C] features at {frame_paths[0].lr_features_path}."
        )
    return int(first.shape[-1])


def resolve_n_features(requested: int | str, available: int) -> int:
    if isinstance(requested, str):
        value = requested.strip().lower()
        if value == "all":
            return int(available)
        n_features = int(value)
    else:
        n_features = int(requested)
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if n_features > available:
        raise ValueError(
            f"Requested {n_features} features, but only {available} are available."
        )
    return n_features


def make_projection_directions(
    n_features: int, n_projections: int, seed: int
) -> np.ndarray:
    if n_projections <= 0:
        raise ValueError("n_feature_projections must be positive.")
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(int(n_features), int(n_projections))).astype(
        np.float32
    )
    norms = np.linalg.norm(directions, axis=0, keepdims=True)
    directions /= np.maximum(norms, 1.0e-12)
    return directions.astype(np.float32, copy=False)


def make_shape_pairs(samples_per_object: int, n_pairs: int, seed: int) -> np.ndarray:
    if samples_per_object < 2:
        raise ValueError("samples_per_object must be at least 2.")
    if n_pairs <= 0:
        raise ValueError("n_shape_pairs must be positive.")
    rng = np.random.default_rng(int(seed) + 17)
    first = rng.integers(0, samples_per_object, size=n_pairs, dtype=np.int32)
    second = rng.integers(0, samples_per_object - 1, size=n_pairs, dtype=np.int32)
    second = second + (second >= first)
    return np.column_stack((first, second)).astype(np.int32, copy=False)


def _stable_seed(base_seed: int, frame_index: int, label_id: int) -> int:
    mask = (1 << 64) - 1
    value = int(base_seed) & mask
    value ^= ((int(frame_index) + 1) * 0x9E3779B185EBCA87) & mask
    value ^= (int(label_id) * 0xC2B2AE3D27D4EB4F) & mask
    return int(value % (2**32 - 1))


def _sample_group_coords(
    coords: np.ndarray,
    *,
    samples_per_object: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    replace = coords.shape[0] < samples_per_object
    indices = rng.choice(coords.shape[0], size=samples_per_object, replace=replace)
    return coords[indices].astype(np.float32, copy=False)


def _label_mean_amplitudes(
    segmentation: np.ndarray,
    labels: np.ndarray,
    *,
    raw_path: Path | None,
) -> np.ndarray:
    if labels.size == 0:
        return np.empty((0,), dtype=np.float32)
    if raw_path is None:
        return np.ones((labels.size,), dtype=np.float32)

    raw = _read_tiff_yxz(raw_path)
    if raw.shape != segmentation.shape:
        raise ValueError(
            f"Raw/mask shape mismatch for {raw_path}: raw={raw.shape}, mask={segmentation.shape}."
        )

    flat_labels = segmentation.ravel()
    flat_raw = raw.ravel()
    foreground = flat_labels != 0
    if not np.any(foreground):
        return np.empty((0,), dtype=np.float32)

    foreground_labels = flat_labels[foreground].astype(np.int64, copy=False)
    raw_values = flat_raw[foreground].astype(np.float64, copy=False)
    unique_labels, inverse, counts = np.unique(
        foreground_labels, return_inverse=True, return_counts=True
    )
    sums = np.bincount(inverse, weights=raw_values, minlength=int(unique_labels.size))
    means = sums / np.maximum(counts, 1)
    if not np.array_equal(unique_labels.astype(np.int64, copy=False), labels):
        raise ValueError(f"Raw amplitude label mismatch for {raw_path}.")
    return means.astype(np.float32, copy=False)


def _shape_signature_from_coords(
    sample_coords_yxz: np.ndarray,
    pair_indices: np.ndarray,
    n_quantiles: int,
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    n_objects = int(sample_coords_yxz.shape[0])
    if n_objects == 0:
        return np.empty((0, n_quantiles), dtype=np.float32)
    pair_a = pair_indices[:, 0]
    pair_b = pair_indices[:, 1]
    quantile_indices = (
        np.linspace(0, pair_indices.shape[0] - 1, n_quantiles).round().astype(np.int64)
    )
    out = np.empty((n_objects, n_quantiles), dtype=np.float32)
    for start in range(0, n_objects, batch_size):
        end = min(n_objects, start + batch_size)
        coords = sample_coords_yxz[start:end].astype(np.float32, copy=False)
        diff = coords[:, pair_a, :] - coords[:, pair_b, :]
        distances = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))
        distances.sort(axis=1)
        signature = distances[:, quantile_indices]
        scale = np.median(distances, axis=1)
        scale = np.where(scale > 1.0e-6, scale, 1.0)
        out[start:end] = signature / scale[:, None]
    return out


def prepare_frame_metadata(
    frame: FramePaths,
    *,
    samples_per_object: int,
    shape_pair_indices: np.ndarray,
    n_shape_quantiles: int,
    seed: int,
) -> FrameMetadata:
    segmentation = _read_tiff_yxz(frame.segmentation_path)
    flat = segmentation.ravel()
    foreground_flat = np.flatnonzero(flat != 0)
    if foreground_flat.size == 0:
        return FrameMetadata(
            index=frame.index,
            name=frame.name,
            high_shape_yxz=tuple(int(dim) for dim in segmentation.shape),
            labels=np.empty((0,), dtype=np.int64),
            volumes=np.empty((0,), dtype=np.int64),
            amplitudes=np.empty((0,), dtype=np.float32),
            centroids_yxz=np.empty((0, 3), dtype=np.float32),
            sample_coords_yxz=np.empty((0, samples_per_object, 3), dtype=np.float32),
            shape_signature=np.empty((0, n_shape_quantiles), dtype=np.float32),
        )

    foreground_labels = flat[foreground_flat].astype(np.int64, copy=False)
    order = np.argsort(foreground_labels, kind="mergesort")
    foreground_flat = foreground_flat[order]
    foreground_labels = foreground_labels[order]
    labels, starts = np.unique(foreground_labels, return_index=True)
    counts = np.diff(np.r_[starts, foreground_labels.size])
    coords_all = np.column_stack(
        np.unravel_index(foreground_flat, segmentation.shape)
    ).astype(np.float32, copy=False)

    sample_coords = np.empty((labels.size, samples_per_object, 3), dtype=np.float32)
    centroids_yxz = np.empty((labels.size, 3), dtype=np.float32)
    for label_index, (label_id, start, count) in enumerate(
        zip(labels.tolist(), starts.tolist(), counts.tolist())
    ):
        coords = coords_all[start : start + count]
        centroids_yxz[label_index] = coords.mean(axis=0, dtype=np.float64)
        sample_coords[label_index] = _sample_group_coords(
            coords,
            samples_per_object=samples_per_object,
            seed=_stable_seed(seed, frame.index, int(label_id)),
        )

    shape_signature = _shape_signature_from_coords(
        sample_coords,
        shape_pair_indices,
        n_shape_quantiles,
    )
    labels = labels.astype(np.int64, copy=False)
    return FrameMetadata(
        index=frame.index,
        name=frame.name,
        high_shape_yxz=tuple(int(dim) for dim in segmentation.shape),
        labels=labels,
        volumes=counts.astype(np.int64, copy=False),
        amplitudes=_label_mean_amplitudes(
            segmentation,
            labels,
            raw_path=frame.raw_path,
        ),
        centroids_yxz=centroids_yxz,
        sample_coords_yxz=sample_coords,
        shape_signature=shape_signature,
    )


def _prepare_frame_metadata_worker(
    args: tuple[FramePaths, int, np.ndarray, int, int],
) -> FrameMetadata:
    frame, samples_per_object, shape_pair_indices, n_shape_quantiles, seed = args
    return prepare_frame_metadata(
        frame,
        samples_per_object=samples_per_object,
        shape_pair_indices=shape_pair_indices,
        n_shape_quantiles=n_shape_quantiles,
        seed=seed,
    )


def prepare_all_metadata(
    frames: Sequence[FramePaths],
    config: ExperimentConfig,
    *,
    shape_pair_indices: np.ndarray,
) -> list[FrameMetadata]:
    _log(
        f"preparing mask metadata for {len(frames)} frame(s) with {config.mask_workers} worker(s)",
        enabled=bool(config.progress),
    )
    jobs = [
        (
            frame,
            int(config.samples_per_object),
            shape_pair_indices,
            int(config.n_shape_quantiles),
            int(config.seed),
        )
        for frame in frames
    ]
    if config.mask_workers <= 1 or len(jobs) <= 1:
        return [
            _prepare_frame_metadata_worker(job)
            for job in _progress_bar(
                jobs,
                total=len(jobs),
                desc="mask metadata",
                enabled=bool(config.progress),
            )
        ]

    out: list[FrameMetadata] = []
    with ProcessPoolExecutor(max_workers=int(config.mask_workers)) as executor:
        futures = [executor.submit(_prepare_frame_metadata_worker, job) for job in jobs]
        for future in _progress_bar(
            as_completed(futures),
            total=len(futures),
            desc="mask metadata",
            enabled=bool(config.progress),
        ):
            out.append(future.result())
    out.sort(key=lambda item: item.index)
    return out


def _resolve_devices(devices: Sequence[str]) -> list[str]:
    requested = tuple(devices) if devices else ("auto",)
    if requested == ("auto",):
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        return ["cpu"]
    resolved: list[str] = []
    for device in requested:
        if device == "auto":
            resolved.extend(_resolve_devices(("auto",)))
        else:
            resolved.append(str(device))
    return resolved or ["cpu"]


def _resolve_single_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def _torch_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _coords_to_grid(
    coords_yxz: torch.Tensor, high_shape_yxz: tuple[int, int, int]
) -> torch.Tensor:
    y_size, x_size, z_size = (float(dim) for dim in high_shape_yxz)
    y = coords_yxz[:, 0]
    x = coords_yxz[:, 1]
    z = coords_yxz[:, 2]
    x_norm = ((x + 0.5) / x_size) * 2.0 - 1.0
    y_norm = ((y + 0.5) / y_size) * 2.0 - 1.0
    z_norm = ((z + 0.5) / z_size) * 2.0 - 1.0
    return torch.stack((x_norm, y_norm, z_norm), dim=-1).view(1, -1, 1, 1, 3)


def _sample_feature_chunk(
    chunk_zyxc: np.ndarray,
    coords_yxz: torch.Tensor,
    high_shape_yxz: tuple[int, int, int],
    *,
    device: torch.device,
    sample_batch_size: int,
) -> torch.Tensor:
    volume = (
        torch.from_numpy(np.asarray(chunk_zyxc, dtype=np.float32).copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
    )
    volume = volume.to(device=device, non_blocking=True)
    sampled_batches: list[torch.Tensor] = []
    for start in range(0, coords_yxz.shape[0], int(sample_batch_size)):
        end = min(coords_yxz.shape[0], start + int(sample_batch_size))
        grid = _coords_to_grid(coords_yxz[start:end], high_shape_yxz)
        sampled = F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled_batches.append(sampled[0, :, :, 0, 0].T.contiguous())
    return torch.cat(sampled_batches, dim=0)


@torch.no_grad()
def build_feature_signatures_for_frame(
    frame: FramePaths,
    metadata: FrameMetadata,
    *,
    projection_directions: np.ndarray,
    n_features: int,
    config: ExperimentConfig,
    device_name: str,
) -> FrameSignatures:
    if int(config.torch_threads_per_worker) > 0:
        torch.set_num_threads(int(config.torch_threads_per_worker))
    device = _torch_device(device_name)
    labels = metadata.labels.astype(np.int64, copy=False)
    n_objects = int(labels.size)
    n_samples = int(config.samples_per_object)
    n_projections = int(config.n_feature_projections)
    if n_objects == 0:
        return FrameSignatures(
            index=metadata.index,
            name=metadata.name,
            labels=labels,
            volumes=metadata.volumes.astype(np.int64, copy=False),
            amplitudes=metadata.amplitudes.astype(np.float32, copy=False),
            centroids_yxz=metadata.centroids_yxz.astype(np.float32, copy=False),
            feature_mean=np.empty((0, n_features), dtype=np.float32),
            feature_sw_signature=np.empty(
                (0, n_samples * n_projections), dtype=np.float32
            ),
            shape_signature=metadata.shape_signature.astype(np.float32, copy=False),
        )

    lr_features = np.load(frame.lr_features_path, mmap_mode="r")
    if lr_features.ndim != 4:
        raise ValueError(f"Expected [Z, Y, X, C] features at {frame.lr_features_path}.")
    if lr_features.shape[-1] < n_features:
        raise ValueError(
            f"{frame.lr_features_path} only has {lr_features.shape[-1]} channels."
        )

    directions_np = np.asarray(
        projection_directions[:, :n_projections], dtype=np.float32
    )
    mean_features = np.empty((n_objects, n_features), dtype=np.float32)
    sw_signature = np.empty((n_objects, n_samples * n_projections), dtype=np.float32)

    for object_start in range(0, n_objects, int(config.object_batch_size)):
        object_end = min(n_objects, object_start + int(config.object_batch_size))
        object_count = object_end - object_start
        coords_np = metadata.sample_coords_yxz[object_start:object_end].reshape(-1, 3)
        coords = torch.from_numpy(coords_np.astype(np.float32, copy=False)).to(
            device=device, non_blocking=True
        )
        sample_count = int(coords.shape[0])
        sample_norm_sq = torch.zeros(
            (sample_count,), dtype=torch.float32, device=device
        )

        for channel_start in range(0, n_features, int(config.feature_channel_block)):
            channel_end = min(
                n_features, channel_start + int(config.feature_channel_block)
            )
            chunk = np.asarray(
                lr_features[..., channel_start:channel_end], dtype=np.float32
            )
            values = _sample_feature_chunk(
                chunk,
                coords,
                metadata.high_shape_yxz,
                device=device,
                sample_batch_size=int(config.sample_batch_size),
            )
            sample_norm_sq += torch.sum(values * values, dim=1)

        sample_norm = torch.sqrt(torch.clamp(sample_norm_sq, min=1.0e-12))
        projected = torch.zeros(
            (sample_count, n_projections), dtype=torch.float32, device=device
        )

        for channel_start in range(0, n_features, int(config.feature_channel_block)):
            channel_end = min(
                n_features, channel_start + int(config.feature_channel_block)
            )
            chunk = np.asarray(
                lr_features[..., channel_start:channel_end], dtype=np.float32
            )
            values = _sample_feature_chunk(
                chunk,
                coords,
                metadata.high_shape_yxz,
                device=device,
                sample_batch_size=int(config.sample_batch_size),
            )
            values = values / sample_norm[:, None]
            directions = torch.from_numpy(directions_np[channel_start:channel_end]).to(
                device=device,
                non_blocking=True,
            )
            projected += values @ directions
            means = values.view(
                object_count, n_samples, channel_end - channel_start
            ).mean(dim=1)
            mean_features[object_start:object_end, channel_start:channel_end] = (
                means.cpu().numpy()
            )

        sorted_projected = torch.sort(
            projected.view(object_count, n_samples, n_projections), dim=1
        ).values
        sw_signature[object_start:object_end] = (
            sorted_projected.reshape(object_count, -1).cpu().numpy()
        )

    return FrameSignatures(
        index=metadata.index,
        name=metadata.name,
        labels=labels,
        volumes=metadata.volumes.astype(np.int64, copy=False),
        amplitudes=metadata.amplitudes.astype(np.float32, copy=False),
        centroids_yxz=metadata.centroids_yxz.astype(np.float32, copy=False),
        feature_mean=mean_features,
        feature_sw_signature=sw_signature,
        shape_signature=metadata.shape_signature.astype(np.float32, copy=False),
    )


def _signature_worker(
    args: tuple[
        str,
        list[tuple[FramePaths, FrameMetadata]],
        np.ndarray,
        int,
        ExperimentConfig,
        Any | None,
    ],
) -> list[FrameSignatures]:
    device_name, items, projection_directions, n_features, config, progress_queue = args
    out: list[FrameSignatures] = []
    for frame, metadata in items:
        out.append(
            build_feature_signatures_for_frame(
                frame,
                metadata,
                projection_directions=projection_directions,
                n_features=n_features,
                config=config,
                device_name=device_name,
            )
        )
        if progress_queue is not None:
            progress_queue.put(1)
    return out


def _build_signature_single_process(
    device_name: str,
    items: list[tuple[FramePaths, FrameMetadata]],
    *,
    projection_directions: np.ndarray,
    n_features: int,
    config: ExperimentConfig,
) -> list[FrameSignatures]:
    out: list[FrameSignatures] = []
    for frame, metadata in _progress_bar(
        items,
        total=len(items),
        desc="feature signatures",
        enabled=bool(config.progress),
    ):
        out.append(
            build_feature_signatures_for_frame(
                frame,
                metadata,
                projection_directions=projection_directions,
                n_features=n_features,
                config=config,
                device_name=device_name,
            )
        )
    return out


def build_all_signatures(
    frames: Sequence[FramePaths],
    metadata: Sequence[FrameMetadata],
    *,
    projection_directions: np.ndarray,
    n_features: int,
    config: ExperimentConfig,
) -> list[FrameSignatures]:
    items = list(zip(frames, metadata))
    devices = _resolve_devices(config.devices)
    if devices == ["cpu"] and config.signature_workers > 1:
        worker_devices = ["cpu"] * int(config.signature_workers)
    else:
        worker_devices = devices[
            : max(1, min(len(devices), int(config.signature_workers) or len(devices)))
        ]

    _log(
        (
            f"building feature signatures for {len(items)} frame(s) on "
            f"{', '.join(worker_devices)}"
        ),
        enabled=bool(config.progress),
    )
    if len(worker_devices) <= 1 or len(items) <= 1:
        device = worker_devices[0] if worker_devices else _resolve_single_device("auto")
        out = _build_signature_single_process(
            device,
            items,
            projection_directions=projection_directions,
            n_features=n_features,
            config=config,
        )
        out.sort(key=lambda item: item.index)
        return out

    shards: list[list[tuple[FramePaths, FrameMetadata]]] = [[] for _ in worker_devices]
    for item_index, item in enumerate(items):
        shards[item_index % len(worker_devices)].append(item)

    context = mp.get_context("spawn")
    out: list[FrameSignatures] = []
    with (
        context.Manager() as manager,
        ProcessPoolExecutor(
            max_workers=len(worker_devices),
            mp_context=context,
        ) as executor,
    ):
        progress_queue = manager.Queue() if config.progress else None
        futures = [
            executor.submit(
                _signature_worker,
                (
                    device,
                    shard,
                    projection_directions,
                    n_features,
                    config,
                    progress_queue,
                ),
            )
            for device, shard in zip(worker_devices, shards)
            if shard
        ]
        completed = 0
        if config.progress:
            with tqdm(
                total=len(items),
                desc="feature signatures",
                dynamic_ncols=True,
            ) as bar:
                while completed < len(items):
                    try:
                        progress_queue.get(timeout=0.25)
                        completed += 1
                        bar.update(1)
                    except Empty:
                        for future in futures:
                            if future.done():
                                exc = future.exception()
                                if exc is not None:
                                    raise exc
                        if all(future.done() for future in futures):
                            break
                if completed < len(items):
                    bar.update(len(items) - completed)
        for future in futures:
            out.extend(future.result())
    out.sort(key=lambda item: item.index)
    return out


def _pairwise_mse_cost(
    left: np.ndarray,
    right: np.ndarray,
    *,
    device_name: str,
    block_rows: int,
    use_float16: bool,
    progress: bool = False,
    desc: str = "pairwise cost",
) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.empty((left.shape[0], right.shape[0]), dtype=np.float32)

    device = _torch_device(_resolve_single_device(device_name))
    dtype = torch.float16 if use_float16 and device.type == "cuda" else torch.float32
    right_t = torch.from_numpy(np.asarray(right, dtype=np.float32)).to(
        device=device, dtype=dtype, non_blocking=True
    )
    right_norm = torch.sum(right_t.float() * right_t.float(), dim=1)
    costs = np.empty((left.shape[0], right.shape[0]), dtype=np.float32)
    denom = float(max(1, left.shape[1]))

    row_starts = range(0, left.shape[0], int(block_rows))
    for start in _progress_bar(
        row_starts,
        total=math.ceil(left.shape[0] / int(block_rows)),
        desc=desc,
        enabled=bool(progress),
    ):
        end = min(left.shape[0], start + int(block_rows))
        left_t = torch.from_numpy(np.asarray(left[start:end], dtype=np.float32)).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
        left_float = left_t.float()
        cross = left_float @ right_t.float().T
        left_norm = torch.sum(left_float * left_float, dim=1)
        block = (left_norm[:, None] + right_norm[None, :] - (2.0 * cross)) / denom
        costs[start:end] = torch.clamp(block, min=0.0).cpu().numpy()
    return costs


def _pairwise_cosine_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    device_name: str,
    block_rows: int,
    progress: bool = False,
    desc: str = "cosine cost",
) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.empty((left.shape[0], right.shape[0]), dtype=np.float32)

    left_norms = np.linalg.norm(left, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    left_normed = left / np.maximum(left_norms, 1.0e-12)
    right_normed = right / np.maximum(right_norms, 1.0e-12)
    similarity = 1.0 - _pairwise_mse_cost(
        left_normed,
        right_normed,
        device_name=device_name,
        block_rows=block_rows,
        use_float16=False,
        progress=progress,
        desc=desc,
    ) * (float(left.shape[1]) / 2.0)
    return np.clip(1.0 - similarity, 0.0, 2.0).astype(np.float32, copy=False)


def _robust_normalize_cost(cost: np.ndarray) -> np.ndarray:
    finite = cost[np.isfinite(cost)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return cost.astype(np.float32, copy=True)
    scale = float(np.median(finite))
    if not math.isfinite(scale) or scale <= 0.0:
        return cost.astype(np.float32, copy=True)
    return (cost / scale).astype(np.float32, copy=False)


def compute_method_costs(
    ref: FrameSignatures,
    cand: FrameSignatures,
    *,
    config: ExperimentConfig,
    device_name: str,
) -> dict[str, np.ndarray]:
    methods = set(config.methods)
    costs: dict[str, np.ndarray] = {}

    if METHOD_FEATURE_MEAN in methods:
        costs[METHOD_FEATURE_MEAN] = _pairwise_cosine_distance(
            ref.feature_mean,
            cand.feature_mean,
            device_name=device_name,
            block_rows=int(config.cost_block_rows),
            progress=bool(config.progress),
            desc=f"{ref.name}->{cand.name} mean",
        )

    needs_sw = bool(
        {METHOD_SLICED_WASSERSTEIN, METHOD_SLICED_WASSERSTEIN_SHAPE} & methods
    )
    sw_cost: np.ndarray | None = None
    if needs_sw:
        sw_cost = _pairwise_mse_cost(
            ref.feature_sw_signature,
            cand.feature_sw_signature,
            device_name=device_name,
            block_rows=int(config.cost_block_rows),
            use_float16=bool(config.use_float16_cost),
            progress=bool(config.progress),
            desc=f"{ref.name}->{cand.name} SW",
        )
        if METHOD_SLICED_WASSERSTEIN in methods:
            costs[METHOD_SLICED_WASSERSTEIN] = sw_cost

    if METHOD_SLICED_WASSERSTEIN_SHAPE in methods:
        if sw_cost is None:
            raise RuntimeError("Internal error: missing sliced-Wasserstein cost.")
        shape_cost = _pairwise_mse_cost(
            ref.shape_signature,
            cand.shape_signature,
            device_name=device_name,
            block_rows=int(config.cost_block_rows),
            use_float16=False,
            progress=bool(config.progress),
            desc=f"{ref.name}->{cand.name} shape",
        )
        costs[METHOD_SLICED_WASSERSTEIN_SHAPE] = _robust_normalize_cost(sw_cost) + (
            float(config.shape_weight) * _robust_normalize_cost(shape_cost)
        )

    return costs


def build_search_radius_gate(
    ref: FrameSignatures,
    cand: FrameSignatures,
    *,
    config: ExperimentConfig,
) -> SearchRadiusGate:
    shape = (int(ref.labels.size), int(cand.labels.size))
    if not config.search_radius_enabled:
        allowed = np.ones(shape, dtype=bool)
        delta_yxz = np.zeros((*shape, 3), dtype=np.float32)
        return SearchRadiusGate(
            enabled=False,
            radius_xy=float(config.search_radius_xy),
            radius_z=float(config.search_radius_z),
            allowed=allowed,
            strict_allowed=allowed.copy(),
            relaxed_ref_rows=np.zeros((shape[0],), dtype=bool),
            delta_yxz=delta_yxz,
        )

    if float(config.search_radius_xy) < 0.0 or float(config.search_radius_z) < 0.0:
        raise ValueError("search_radius_xy and search_radius_z must be non-negative.")

    if shape[0] == 0 or shape[1] == 0:
        allowed = np.zeros(shape, dtype=bool)
        delta_yxz = np.zeros((*shape, 3), dtype=np.float32)
        return SearchRadiusGate(
            enabled=True,
            radius_xy=float(config.search_radius_xy),
            radius_z=float(config.search_radius_z),
            allowed=allowed,
            strict_allowed=allowed.copy(),
            relaxed_ref_rows=np.zeros((shape[0],), dtype=bool),
            delta_yxz=delta_yxz,
        )

    delta_yxz = cand.centroids_yxz[None, :, :].astype(
        np.float32, copy=False
    ) - ref.centroids_yxz[:, None, :].astype(np.float32, copy=False)
    strict_allowed = (
        (np.abs(delta_yxz[..., 0]) <= float(config.search_radius_xy))
        & (np.abs(delta_yxz[..., 1]) <= float(config.search_radius_xy))
        & (np.abs(delta_yxz[..., 2]) <= float(config.search_radius_z))
    )
    relaxed_ref_rows = ~strict_allowed.any(axis=1)
    allowed = strict_allowed.copy()
    allowed[relaxed_ref_rows, :] = True
    return SearchRadiusGate(
        enabled=True,
        radius_xy=float(config.search_radius_xy),
        radius_z=float(config.search_radius_z),
        allowed=allowed,
        strict_allowed=strict_allowed,
        relaxed_ref_rows=relaxed_ref_rows,
        delta_yxz=delta_yxz.astype(np.float32, copy=False),
    )


def apply_search_radius_gate(cost: np.ndarray, gate: SearchRadiusGate) -> np.ndarray:
    if not gate.enabled or cost.size == 0:
        return cost
    gated = cost.astype(np.float32, copy=True)
    outside = ~gate.allowed
    if not np.any(outside):
        return gated

    finite = gated[np.isfinite(gated)]
    if finite.size == 0:
        outside_value = 1.0e12
    else:
        finite_min = float(np.min(finite))
        finite_max = float(np.max(finite))
        scale = max(1.0, abs(finite_max), finite_max - finite_min)
        outside_value = finite_max + (1.0e6 * scale)
    gated[outside] = np.float32(outside_value)
    return gated


def _gate_value(gate: SearchRadiusGate, row: int, col: int) -> bool:
    if not gate.enabled or gate.strict_allowed.size == 0:
        return True
    return bool(gate.strict_allowed[row, col])


def _gate_delta(
    gate: SearchRadiusGate, row: int, col: int
) -> tuple[float, float, float]:
    if gate.delta_yxz.size == 0:
        return (np.nan, np.nan, np.nan)
    delta = gate.delta_yxz[row, col]
    return float(delta[0]), float(delta[1]), float(delta[2])


def evaluate_cost_matrix(
    ref: FrameSignatures,
    cand: FrameSignatures,
    cost: np.ndarray,
    *,
    method: str,
    radius_gate: SearchRadiusGate,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ref_labels = ref.labels.astype(np.int64, copy=False)
    cand_labels = cand.labels.astype(np.int64, copy=False)
    cand_index_by_label = {
        int(label): index for index, label in enumerate(cand_labels.tolist())
    }

    if cost.shape != (ref_labels.size, cand_labels.size):
        raise ValueError(
            f"Cost shape {cost.shape} does not match labels {(ref_labels.size, cand_labels.size)}."
        )

    radius_pair_count = int(ref_labels.size * cand_labels.size)
    if radius_gate.enabled and radius_pair_count:
        radius_allowed_count = int(np.count_nonzero(radius_gate.strict_allowed))
        radius_allowed_fraction = float(radius_allowed_count / radius_pair_count)
    else:
        radius_allowed_count = radius_pair_count
        radius_allowed_fraction = float(1.0) if radius_pair_count else np.nan
    radius_relaxed_ref_count = (
        int(np.count_nonzero(radius_gate.relaxed_ref_rows))
        if radius_gate.enabled
        else 0
    )
    radius_relaxed_ref_fraction = (
        float(radius_relaxed_ref_count / ref_labels.size) if ref_labels.size else np.nan
    )

    assignment_by_row: dict[int, int] = {}
    if cost.size > 0 and ref_labels.size > 0 and cand_labels.size > 0:
        row_indices, col_indices = linear_sum_assignment(cost)
        assignment_by_row = {
            int(row): int(col)
            for row, col in zip(row_indices.tolist(), col_indices.tolist())
        }

    assignment_rows: list[dict[str, Any]] = []
    assigned_outside_radius_count = 0
    assigned_relaxed_ref_count = 0
    for row_index, col_index in sorted(assignment_by_row.items()):
        ref_label = int(ref_labels[row_index])
        cand_label = int(cand_labels[col_index])
        inside_radius = _gate_value(radius_gate, row_index, col_index)
        relaxed_for_ref = (
            bool(radius_gate.relaxed_ref_rows[row_index])
            if radius_gate.relaxed_ref_rows.size
            else False
        )
        delta_y, delta_x, delta_z = _gate_delta(radius_gate, row_index, col_index)
        assigned_outside_radius_count += int(radius_gate.enabled and not inside_radius)
        assigned_relaxed_ref_count += int(radius_gate.enabled and relaxed_for_ref)
        assignment_rows.append({
            "method": method,
            "ref_frame_index": int(ref.index),
            "cand_frame_index": int(cand.index),
            "ref_timepoint": ref.name,
            "cand_timepoint": cand.name,
            "ref_label": ref_label,
            "assigned_cand_label": cand_label,
            "cost": float(cost[row_index, col_index]),
            "is_true_link": bool(ref_label == cand_label),
            "inside_search_radius": inside_radius,
            "radius_relaxed_for_ref": relaxed_for_ref,
            "centroid_delta_y": delta_y,
            "centroid_delta_x": delta_x,
            "centroid_delta_z": delta_z,
            "centroid_distance_xy": float(math.hypot(delta_y, delta_x)),
        })

    per_object: list[dict[str, Any]] = []
    correct = 0
    rank_values: list[int] = []
    margins: list[float] = []
    true_costs: list[float] = []
    trackable_count = 0
    true_link_outside_radius_count = 0

    for row_index, ref_label in enumerate(ref_labels.tolist()):
        true_col = cand_index_by_label.get(int(ref_label))
        if true_col is None:
            continue
        trackable_count += 1
        true_inside_radius = _gate_value(radius_gate, row_index, true_col)
        true_link_outside_radius_count += int(
            radius_gate.enabled and not true_inside_radius
        )
        row_cost = cost[row_index]
        true_cost = float(row_cost[true_col])
        assigned_col = assignment_by_row.get(row_index)
        assigned_label = (
            int(cand_labels[assigned_col]) if assigned_col is not None else None
        )
        assigned_inside_radius = (
            _gate_value(radius_gate, row_index, assigned_col)
            if assigned_col is not None
            else False
        )
        relaxed_for_ref = (
            bool(radius_gate.relaxed_ref_rows[row_index])
            if radius_gate.relaxed_ref_rows.size
            else False
        )
        true_delta_y, true_delta_x, true_delta_z = _gate_delta(
            radius_gate, row_index, true_col
        )
        is_correct = assigned_col == true_col
        correct += int(is_correct)

        rank = int(np.count_nonzero(row_cost < true_cost) + 1)
        false_cost = np.array(row_cost, copy=True)
        false_cost[true_col] = np.inf
        best_false = float(np.min(false_cost)) if false_cost.size > 1 else np.nan
        margin = best_false - true_cost if math.isfinite(best_false) else np.nan
        rank_values.append(rank)
        true_costs.append(true_cost)
        if math.isfinite(margin):
            margins.append(float(margin))

        per_object.append({
            "method": method,
            "ref_frame_index": int(ref.index),
            "cand_frame_index": int(cand.index),
            "ref_timepoint": ref.name,
            "cand_timepoint": cand.name,
            "ref_label": int(ref_label),
            "true_cand_label": int(ref_label),
            "assigned_cand_label": assigned_label,
            "is_correct": bool(is_correct),
            "true_rank": rank,
            "true_cost": true_cost,
            "best_false_cost": best_false,
            "margin": margin,
            "true_inside_search_radius": true_inside_radius,
            "assigned_inside_search_radius": assigned_inside_radius,
            "radius_relaxed_for_ref": relaxed_for_ref,
            "true_centroid_delta_y": true_delta_y,
            "true_centroid_delta_x": true_delta_x,
            "true_centroid_delta_z": true_delta_z,
            "true_centroid_distance_xy": float(math.hypot(true_delta_y, true_delta_x)),
        })

    summary = {
        "method": method,
        "ref_frame_index": int(ref.index),
        "cand_frame_index": int(cand.index),
        "ref_timepoint": ref.name,
        "cand_timepoint": cand.name,
        "ref_count": int(ref_labels.size),
        "cand_count": int(cand_labels.size),
        "trackable_count": int(trackable_count),
        "hungarian_correct": int(correct),
        "hungarian_accuracy": float(correct / trackable_count)
        if trackable_count
        else np.nan,
        "top1_count": int(sum(1 for value in rank_values if value == 1)),
        "top1_accuracy": float(
            sum(1 for value in rank_values if value == 1) / trackable_count
        )
        if trackable_count
        else np.nan,
        "mean_true_rank": float(np.mean(rank_values)) if rank_values else np.nan,
        "median_true_rank": float(np.median(rank_values)) if rank_values else np.nan,
        "mean_true_cost": float(np.mean(true_costs)) if true_costs else np.nan,
        "median_margin": float(np.median(margins)) if margins else np.nan,
        "mean_margin": float(np.mean(margins)) if margins else np.nan,
        "search_radius_enabled": bool(radius_gate.enabled),
        "search_radius_xy": float(radius_gate.radius_xy),
        "search_radius_z": float(radius_gate.radius_z),
        "radius_pair_count": radius_pair_count,
        "radius_allowed_count": radius_allowed_count,
        "radius_allowed_fraction": radius_allowed_fraction,
        "radius_relaxed_ref_count": radius_relaxed_ref_count,
        "radius_relaxed_ref_fraction": radius_relaxed_ref_fraction,
        "true_link_outside_radius_count": int(true_link_outside_radius_count),
        "true_link_outside_radius_fraction": float(
            true_link_outside_radius_count / trackable_count
        )
        if trackable_count
        else np.nan,
        "assigned_outside_radius_count": int(assigned_outside_radius_count),
        "assigned_outside_radius_fraction": float(
            assigned_outside_radius_count / len(assignment_rows)
        )
        if assignment_rows
        else np.nan,
        "assigned_relaxed_ref_count": int(assigned_relaxed_ref_count),
    }
    pred_count = int(len(assignment_rows))
    tp = int(correct)
    fp = int(pred_count - tp)
    fn = int(trackable_count - tp)
    precision = float(tp / pred_count) if pred_count else np.nan
    recall = float(tp / trackable_count) if trackable_count else np.nan
    summary.update({
        "link_pred_count": pred_count,
        "link_tp": tp,
        "link_fp": fp,
        "link_fn": fn,
        "link_precision": precision,
        "link_recall": recall,
        "link_f1": float((2.0 * precision * recall) / (precision + recall))
        if math.isfinite(precision)
        and math.isfinite(recall)
        and (precision + recall) > 0.0
        else 0.0
        if math.isfinite(precision) and math.isfinite(recall)
        else np.nan,
    })
    return summary, per_object, assignment_rows


def evaluate_adjacent_pairs(
    signatures: Sequence[FrameSignatures],
    *,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_device = _resolve_single_device(config.pair_device)
    summary_rows: list[dict[str, Any]] = []
    per_object_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    link_metric_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    max_pairs = len(signatures) - 1
    if config.max_adjacent_pairs is not None:
        max_pairs = min(max_pairs, int(config.max_adjacent_pairs))

    _log(
        f"evaluating {max_pairs} adjacent frame pair(s) on {pair_device}",
        enabled=bool(config.progress),
    )
    pair_iter = _progress_bar(
        range(max_pairs),
        total=max_pairs,
        desc="adjacent pairs",
        enabled=bool(config.progress),
    )
    for pair_index in pair_iter:
        ref = signatures[pair_index]
        cand = signatures[pair_index + 1]
        _log(
            (
                f"pair {pair_index + 1}/{max_pairs}: {ref.name}->{cand.name} "
                f"dense matrix {ref.labels.size}x{cand.labels.size}"
            ),
            enabled=bool(config.progress),
        )
        start = time.perf_counter()
        method_costs = compute_method_costs(
            ref, cand, config=config, device_name=pair_device
        )
        radius_gate = build_search_radius_gate(ref, cand, config=config)
        cost_seconds = time.perf_counter() - start
        for method, cost in method_costs.items():
            eval_start = time.perf_counter()
            gated_cost = apply_search_radius_gate(cost, radius_gate)
            summary, per_object, assignments = evaluate_cost_matrix(
                ref,
                cand,
                gated_cost,
                method=method,
                radius_gate=radius_gate,
            )
            eval_seconds = time.perf_counter() - eval_start
            summary_rows.append(summary)
            per_object_rows.extend(per_object)
            assignment_rows.extend(assignments)
            link_metric_rows.append({
                key: summary[key]
                for key in (
                    "method",
                    "ref_frame_index",
                    "cand_frame_index",
                    "ref_timepoint",
                    "cand_timepoint",
                    "ref_count",
                    "cand_count",
                    "trackable_count",
                    "link_pred_count",
                    "link_tp",
                    "link_fp",
                    "link_fn",
                    "link_precision",
                    "link_recall",
                    "link_f1",
                    "search_radius_enabled",
                    "search_radius_xy",
                    "search_radius_z",
                    "radius_pair_count",
                    "radius_allowed_count",
                    "radius_allowed_fraction",
                    "radius_relaxed_ref_count",
                    "radius_relaxed_ref_fraction",
                    "true_link_outside_radius_count",
                    "true_link_outside_radius_fraction",
                    "assigned_outside_radius_count",
                    "assigned_outside_radius_fraction",
                    "assigned_relaxed_ref_count",
                )
            })
            timing_rows.append({
                "ref_frame_index": int(ref.index),
                "cand_frame_index": int(cand.index),
                "ref_timepoint": ref.name,
                "cand_timepoint": cand.name,
                "method": method,
                "cost_seconds": float(cost_seconds),
                "assignment_and_scoring_seconds": float(eval_seconds),
            })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(per_object_rows),
        pd.DataFrame(assignment_rows),
        pd.DataFrame(link_metric_rows),
        pd.DataFrame(timing_rows),
    )


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    def finite_series_mean(series: pd.Series) -> float:
        values = series.to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else np.nan

    def finite_series_median(series: pd.Series) -> float:
        values = series.to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else np.nan

    grouped = summary.groupby("method", sort=False)
    rows: list[dict[str, Any]] = []
    for method, group in grouped:
        trackable = group["trackable_count"].to_numpy(dtype=np.float64)
        correct = group["hungarian_correct"].to_numpy(dtype=np.float64)
        top1 = group["top1_count"].to_numpy(dtype=np.float64)
        pred = (
            float(group["link_pred_count"].sum())
            if "link_pred_count" in group
            else np.nan
        )
        tp = float(group["link_tp"].sum()) if "link_tp" in group else np.nan
        fp = float(group["link_fp"].sum()) if "link_fp" in group else np.nan
        fn = float(group["link_fn"].sum()) if "link_fn" in group else np.nan
        radius_pair_count = (
            float(group["radius_pair_count"].sum())
            if "radius_pair_count" in group
            else np.nan
        )
        radius_allowed_count = (
            float(group["radius_allowed_count"].sum())
            if "radius_allowed_count" in group
            else np.nan
        )
        radius_relaxed_ref_count = (
            float(group["radius_relaxed_ref_count"].sum())
            if "radius_relaxed_ref_count" in group
            else np.nan
        )
        true_link_outside_radius_count = (
            float(group["true_link_outside_radius_count"].sum())
            if "true_link_outside_radius_count" in group
            else np.nan
        )
        assigned_outside_radius_count = (
            float(group["assigned_outside_radius_count"].sum())
            if "assigned_outside_radius_count" in group
            else np.nan
        )
        assigned_relaxed_ref_count = (
            float(group["assigned_relaxed_ref_count"].sum())
            if "assigned_relaxed_ref_count" in group
            else np.nan
        )
        precision = tp / pred if pred and pred > 0.0 else np.nan
        recall = (
            tp / (tp + fn)
            if math.isfinite(tp) and math.isfinite(fn) and (tp + fn) > 0.0
            else np.nan
        )
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if math.isfinite(precision)
            and math.isfinite(recall)
            and (precision + recall) > 0.0
            else 0.0
            if math.isfinite(precision) and math.isfinite(recall)
            else np.nan
        )
        rows.append({
            "method": method,
            "frame_pairs": int(group.shape[0]),
            "trackable_count": int(np.nansum(trackable)),
            "hungarian_accuracy": float(np.nansum(correct) / np.nansum(trackable))
            if np.nansum(trackable) > 0
            else np.nan,
            "top1_accuracy": float(np.nansum(top1) / np.nansum(trackable))
            if np.nansum(trackable) > 0
            else np.nan,
            "mean_true_rank": finite_series_mean(group["mean_true_rank"]),
            "median_pair_margin": finite_series_median(group["median_margin"]),
            "mean_pair_margin": finite_series_mean(group["mean_margin"]),
            "link_pred_count": int(pred) if math.isfinite(pred) else 0,
            "link_tp": int(tp) if math.isfinite(tp) else 0,
            "link_fp": int(fp) if math.isfinite(fp) else 0,
            "link_fn": int(fn) if math.isfinite(fn) else 0,
            "link_precision": precision,
            "link_recall": recall,
            "link_f1": f1,
            "search_radius_enabled": bool(group["search_radius_enabled"].any())
            if "search_radius_enabled" in group
            else False,
            "search_radius_xy": finite_series_median(group["search_radius_xy"])
            if "search_radius_xy" in group
            else np.nan,
            "search_radius_z": finite_series_median(group["search_radius_z"])
            if "search_radius_z" in group
            else np.nan,
            "radius_pair_count": int(radius_pair_count)
            if math.isfinite(radius_pair_count)
            else 0,
            "radius_allowed_count": int(radius_allowed_count)
            if math.isfinite(radius_allowed_count)
            else 0,
            "radius_allowed_fraction": float(radius_allowed_count / radius_pair_count)
            if math.isfinite(radius_allowed_count)
            and math.isfinite(radius_pair_count)
            and radius_pair_count > 0.0
            else np.nan,
            "radius_relaxed_ref_count": int(radius_relaxed_ref_count)
            if math.isfinite(radius_relaxed_ref_count)
            else 0,
            "radius_relaxed_ref_fraction": float(
                radius_relaxed_ref_count / np.nansum(trackable)
            )
            if math.isfinite(radius_relaxed_ref_count) and np.nansum(trackable) > 0
            else np.nan,
            "true_link_outside_radius_count": int(true_link_outside_radius_count)
            if math.isfinite(true_link_outside_radius_count)
            else 0,
            "true_link_outside_radius_fraction": float(
                true_link_outside_radius_count / np.nansum(trackable)
            )
            if math.isfinite(true_link_outside_radius_count)
            and np.nansum(trackable) > 0
            else np.nan,
            "assigned_outside_radius_count": int(assigned_outside_radius_count)
            if math.isfinite(assigned_outside_radius_count)
            else 0,
            "assigned_outside_radius_fraction": float(
                assigned_outside_radius_count / pred
            )
            if math.isfinite(assigned_outside_radius_count)
            and math.isfinite(pred)
            and pred > 0.0
            else np.nan,
            "assigned_relaxed_ref_count": int(assigned_relaxed_ref_count)
            if math.isfinite(assigned_relaxed_ref_count)
            else 0,
        })
    return pd.DataFrame(rows)


def _object_lookup(
    signatures: Sequence[FrameSignatures],
) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for frame in signatures:
        for index, label_id in enumerate(frame.labels.tolist()):
            centroid = frame.centroids_yxz[index]
            lookup[(int(frame.index), int(label_id))] = {
                "frame_index": int(frame.index),
                "timepoint_name": frame.name,
                "label_id": int(label_id),
                "gt_track_id": int(label_id),
                "x": float(centroid[1]),
                "y": float(centroid[0]),
                "z": float(centroid[2]),
                "A": float(frame.amplitudes[index]),
                "volume": int(frame.volumes[index]),
            }
    return lookup


def _empty_track_points() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "method",
            "track_id",
            "start",
            "t",
            "frame_index",
            "timepoint_name",
            "label_id",
            "gt_track_id",
            "x",
            "y",
            "z",
            "A",
            "volume",
            "track_length",
        ]
    )


def _empty_tracks() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TRACK_COLUMNS))


def build_track_tables_from_assignments(
    signatures: Sequence[FrameSignatures],
    assignments: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if assignments.empty:
        return {}, _empty_track_points()

    object_lookup = _object_lookup(signatures)
    labels_by_frame = {
        int(frame.index): [int(label) for label in frame.labels.tolist()]
        for frame in signatures
    }
    rows: list[dict[str, Any]] = []

    for method, method_assignments in assignments.groupby("method", sort=False):
        pair_frames = (
            method_assignments[["ref_frame_index", "cand_frame_index"]]
            .drop_duplicates()
            .sort_values(["ref_frame_index", "cand_frame_index"], kind="mergesort")
        )
        if pair_frames.empty:
            continue

        next_track_id = 1
        active: dict[int, int] = {}
        seen_points: set[tuple[int, int, int]] = set()

        def append_point(track_id: int, frame_index: int, label_id: int) -> None:
            key = (int(track_id), int(frame_index), int(label_id))
            if key in seen_points:
                return
            info = object_lookup.get((int(frame_index), int(label_id)))
            if info is None:
                return
            seen_points.add(key)
            rows.append({"method": method, "track_id": int(track_id), **info})

        def start_track(frame_index: int, label_id: int) -> int:
            nonlocal next_track_id
            track_id = next_track_id
            next_track_id += 1
            append_point(track_id, frame_index, label_id)
            return track_id

        first_ref_frame = int(pair_frames.iloc[0]["ref_frame_index"])
        for label_id in labels_by_frame.get(first_ref_frame, []):
            active[int(label_id)] = start_track(first_ref_frame, int(label_id))

        for pair in pair_frames.itertuples(index=False):
            ref_frame = int(pair.ref_frame_index)
            cand_frame = int(pair.cand_frame_index)
            pair_assignments = method_assignments[
                (method_assignments["ref_frame_index"] == ref_frame)
                & (method_assignments["cand_frame_index"] == cand_frame)
            ]

            for label_id in labels_by_frame.get(ref_frame, []):
                if int(label_id) not in active:
                    active[int(label_id)] = start_track(ref_frame, int(label_id))

            next_active: dict[int, int] = {}
            assigned_cands: set[int] = set()
            for assignment in pair_assignments.itertuples(index=False):
                ref_label = int(assignment.ref_label)
                cand_label = int(assignment.assigned_cand_label)
                track_id = active.get(ref_label)
                if track_id is None:
                    track_id = start_track(ref_frame, ref_label)
                append_point(track_id, cand_frame, cand_label)
                next_active[cand_label] = track_id
                assigned_cands.add(cand_label)

            for cand_label in labels_by_frame.get(cand_frame, []):
                if int(cand_label) not in assigned_cands:
                    next_active[int(cand_label)] = start_track(
                        cand_frame, int(cand_label)
                    )

            active = next_active

    track_points = pd.DataFrame(rows)
    if track_points.empty:
        return {}, _empty_track_points()
    track_points = track_points.sort_values(
        ["method", "track_id", "frame_index"], kind="mergesort"
    ).reset_index(drop=True)
    track_points["start"] = (
        track_points.groupby(["method", "track_id"])["frame_index"].transform("min") + 1
    )
    track_points["t"] = (
        track_points.groupby(["method", "track_id"], sort=False).cumcount() + 1
    )
    track_points["track_length"] = track_points.groupby(["method", "track_id"])[
        "frame_index"
    ].transform("count")
    track_points = track_points[
        [
            "method",
            "track_id",
            "start",
            "t",
            "frame_index",
            "timepoint_name",
            "label_id",
            "gt_track_id",
            "x",
            "y",
            "z",
            "A",
            "volume",
            "track_length",
        ]
    ]

    tracks_by_method: dict[str, pd.DataFrame] = {}
    for method, method_points in track_points.groupby("method", sort=False):
        tracks = method_points.loc[:, TRACK_COLUMNS].copy()
        tracks = tracks.sort_values(["track_id", "t"], kind="mergesort").reset_index(
            drop=True
        )
        tracks_by_method[str(method)] = tracks
    return tracks_by_method, track_points


def run_adjacent_tracking_experiment(
    input_path: str | Path,
    segmentation_path: str | Path,
    *,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    config = ExperimentConfig() if config is None else config
    if config.search_radius_enabled and (
        float(config.search_radius_xy) < 0.0 or float(config.search_radius_z) < 0.0
    ):
        raise ValueError("search_radius_xy and search_radius_z must be non-negative.")
    total_start = time.perf_counter()
    frames = discover_frames(input_path, segmentation_path)
    discovered_count = len(frames)
    if config.max_frames is not None:
        if int(config.max_frames) < 2:
            raise ValueError("max_frames must be at least 2 when provided.")
        frames = frames[: int(config.max_frames)]
        if len(frames) < 2:
            raise ValueError(
                f"Only {len(frames)} frame(s) available after applying max_frames={config.max_frames}."
            )
        _log(
            f"limiting run to first {len(frames)}/{discovered_count} discovered frame(s)",
            enabled=bool(config.progress),
        )
    available_features = infer_feature_count(frames)
    n_features = resolve_n_features(config.n_features, available_features)
    _log(
        (
            f"found {len(frames)} frame(s); using {n_features}/{available_features} feature channel(s), "
            f"{config.samples_per_object} sample(s)/object, {config.n_feature_projections} projection(s)"
        ),
        enabled=bool(config.progress),
    )
    _log(
        (
            f"signature devices={_resolve_devices(config.devices)}; "
            f"pair_device={_resolve_single_device(config.pair_device)}"
        ),
        enabled=bool(config.progress),
    )
    if config.search_radius_enabled:
        _log(
            (
                "centroid search radius enabled: "
                f"xy={float(config.search_radius_xy):g}, z={float(config.search_radius_z):g}"
            ),
            enabled=bool(config.progress),
        )
    else:
        _log("centroid search radius disabled", enabled=bool(config.progress))
    projection_directions = make_projection_directions(
        n_features,
        int(config.n_feature_projections),
        int(config.seed),
    )
    shape_pair_indices = make_shape_pairs(
        int(config.samples_per_object),
        int(config.n_shape_pairs),
        int(config.seed),
    )

    metadata_start = time.perf_counter()
    metadata = prepare_all_metadata(
        frames, config, shape_pair_indices=shape_pair_indices
    )
    metadata_seconds = time.perf_counter() - metadata_start
    _log(
        f"mask metadata completed in {metadata_seconds:.2f}s",
        enabled=bool(config.progress),
    )

    signature_start = time.perf_counter()
    signatures = build_all_signatures(
        frames,
        metadata,
        projection_directions=projection_directions,
        n_features=n_features,
        config=config,
    )
    signature_seconds = time.perf_counter() - signature_start
    _log(
        f"feature signatures completed in {signature_seconds:.2f}s",
        enabled=bool(config.progress),
    )

    eval_start = time.perf_counter()
    summary, per_object, assignments, link_metrics, timings = evaluate_adjacent_pairs(
        signatures, config=config
    )
    tracks, track_points = build_track_tables_from_assignments(signatures, assignments)
    eval_seconds = time.perf_counter() - eval_start
    total_seconds = time.perf_counter() - total_start
    _log(
        f"adjacent pair evaluation completed in {eval_seconds:.2f}s; total {total_seconds:.2f}s",
        enabled=bool(config.progress),
    )

    timing_header = pd.DataFrame([
        {"stage": "metadata", "seconds": metadata_seconds},
        {"stage": "signatures", "seconds": signature_seconds},
        {"stage": "pair_costs_and_assignment", "seconds": eval_seconds},
        {"stage": "total", "seconds": total_seconds},
    ])
    timings = pd.concat([timing_header, timings], ignore_index=True, sort=False)
    return ExperimentResult(
        config=config,
        frame_count=len(frames),
        summary=summary,
        per_object=per_object,
        assignments=assignments,
        link_metrics=link_metrics,
        tracks=tracks,
        track_points=track_points,
        timings=timings,
    )


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("._") or "method"


def _write_tracks_csv(path: Path, tracks: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tracks.empty:
        _empty_tracks().to_csv(path, index=False)
        return
    tracks.loc[:, TRACK_COLUMNS].to_csv(path, index=False)


def save_result(result: ExperimentResult, output_path: str | Path) -> dict[str, Any]:
    output_path = Path(output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {
        "summary_csv": str(output_path / "summary.csv"),
        "aggregate_summary_csv": str(output_path / "aggregate_summary.csv"),
        "per_object_csv": str(output_path / "per_object.csv"),
        "assignments_csv": str(output_path / "assignments.csv"),
        "link_metrics_csv": str(output_path / "link_metrics.csv"),
        "track_points_csv": str(output_path / "track_points_debug.csv"),
        "timings_csv": str(output_path / "timings.csv"),
        "config_json": str(output_path / "config.json"),
    }
    result.summary.to_csv(paths["summary_csv"], index=False)
    aggregate_summary(result.summary).to_csv(
        paths["aggregate_summary_csv"], index=False
    )
    result.per_object.to_csv(paths["per_object_csv"], index=False)
    result.assignments.to_csv(paths["assignments_csv"], index=False)
    result.link_metrics.to_csv(paths["link_metrics_csv"], index=False)
    result.track_points.to_csv(paths["track_points_csv"], index=False)
    tracks_by_method_paths: dict[str, str] = {}
    default_method: str | None = None
    for method in result.config.methods:
        if method in result.tracks:
            default_method = str(method)
            break
    if default_method is None and result.tracks:
        default_method = next(iter(result.tracks))

    if default_method is not None:
        tracks_csv = output_path / "tracks.csv"
        _write_tracks_csv(tracks_csv, result.tracks[default_method])
        paths["tracks_csv"] = str(tracks_csv)
        paths["tracks_default_method"] = default_method

    for method, tracks in result.tracks.items():
        safe_method = _safe_filename_token(method)
        flat_path = output_path / f"tracks_{safe_method}.csv"
        nested_path = output_path / "tracks_by_method" / safe_method / "tracks.csv"
        _write_tracks_csv(flat_path, tracks)
        _write_tracks_csv(nested_path, tracks)
        paths[f"tracks_{safe_method}_csv"] = str(flat_path)
        tracks_by_method_paths[method] = str(nested_path)
    paths["tracks_by_method"] = tracks_by_method_paths
    result.timings.to_csv(paths["timings_csv"], index=False)
    Path(paths["config_json"]).write_text(
        json.dumps(asdict(result.config), indent=2), encoding="utf-8"
    )
    return paths


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(token.strip() for token in value.split(",") if token.strip())
    allowed = set(DEFAULT_METHODS)
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. Allowed methods: {sorted(allowed)}"
        )
    return methods


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental dense feature/shape GT tracking benchmark."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Inference output folder containing lr_feats/.",
    )
    parser.add_argument(
        "--segmentation-path",
        required=True,
        help="GT segmentation folder with <timepoint>.tif files.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Folder where result CSV files will be written.",
    )
    parser.add_argument(
        "--n-features", default="384", help="Feature channels to use, or 'all'."
    )
    parser.add_argument("--samples-per-object", type=int, default=128)
    parser.add_argument("--n-feature-projections", type=int, default=64)
    parser.add_argument("--shape-weight", type=float, default=0.1)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--mask-workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--signature-workers", type=int, default=1)
    parser.add_argument(
        "--devices",
        default="auto",
        help="Comma-separated devices, e.g. auto,cuda:0,cuda:1,cpu.",
    )
    parser.add_argument("--pair-device", default="auto")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N discovered timepoints. Must be at least 2.",
    )
    parser.add_argument("--max-adjacent-pairs", type=int, default=None)
    parser.add_argument(
        "--search-radius-xy",
        type=float,
        default=100.0,
        help="Optional centroid gate radius in x/y pixels. Defaults to 100.",
    )
    parser.add_argument(
        "--search-radius-z",
        type=float,
        default=50.0,
        help="Optional centroid gate radius in z slices. Defaults to 50.",
    )
    parser.add_argument(
        "--disable-search-radius",
        action="store_true",
        help="Disable centroid radius gating and evaluate fully dense matching.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and stage logs.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    n_features: int | str
    if str(args.n_features).strip().lower() == "all":
        n_features = "all"
    else:
        n_features = int(args.n_features)
    return ExperimentConfig(
        n_features=n_features,
        samples_per_object=int(args.samples_per_object),
        n_feature_projections=int(args.n_feature_projections),
        shape_weight=float(args.shape_weight),
        methods=_parse_methods(str(args.methods)),
        seed=int(args.seed),
        mask_workers=int(args.mask_workers),
        signature_workers=int(args.signature_workers),
        devices=tuple(
            token.strip() for token in str(args.devices).split(",") if token.strip()
        ),
        pair_device=str(args.pair_device),
        max_frames=args.max_frames,
        max_adjacent_pairs=args.max_adjacent_pairs,
        search_radius_enabled=not bool(args.disable_search_radius),
        search_radius_xy=float(args.search_radius_xy),
        search_radius_z=float(args.search_radius_z),
        progress=not bool(args.no_progress),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    result = run_adjacent_tracking_experiment(
        args.input_path,
        args.segmentation_path,
        config=config,
    )
    paths = save_result(result, args.output_path)
    print(aggregate_summary(result.summary).to_string(index=False))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
