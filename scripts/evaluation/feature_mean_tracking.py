from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm


METHOD_CENTROID = "centroid"
METHOD_FEATURE_MEAN = "feature_mean"
METHOD_CENTROID_FEATURE = "centroid_feature"
METHOD_CENTROID_FEATURE_3FRAME = "centroid_feature_3frame"
METHOD_CENTROID_FEATURE_PROTOTYPE = "centroid_feature_prototype"
PAIRWISE_METHODS = (METHOD_CENTROID, METHOD_FEATURE_MEAN, METHOD_CENTROID_FEATURE)
DEFAULT_METHODS = (METHOD_CENTROID, METHOD_FEATURE_MEAN, METHOD_CENTROID_FEATURE)
DEFAULT_TRACKS_METHOD = METHOD_CENTROID_FEATURE
TRACK_COLUMNS = ("track_id", "start", "t", "x", "y", "z", "A", "track_length")


@dataclass(frozen=True)
class FeatureMeanConfig:
    n_features: int | str = 384
    samples_per_object: int = 128
    seed: int = 12345
    device: str = "auto"
    methods: tuple[str, ...] = DEFAULT_METHODS
    tracks_method: str = DEFAULT_TRACKS_METHOD
    centroid_feature_weight: float = 0.1
    feature_norm_clip: float = 3.0
    z_weight: float = 1.0
    max_distance_xy: float = 20.0
    max_distance_z: float = 10.0
    three_frame_direct_weight: float = 0.25
    three_frame_candidate_top_k: int = 8
    three_frame_time_limit_seconds: float = 60.0
    max_frames: int | None = None
    compute_gt_metrics: bool = True
    progress: bool = True
    object_batch_size: int = 512
    feature_channel_block: int = 64
    sample_batch_size: int = 131_072


@dataclass(frozen=True)
class FramePaths:
    index: int
    name: str
    lr_features_path: Path
    segmentation_path: Path


@dataclass
class FrameObjects:
    index: int
    name: str
    high_shape_yxz: tuple[int, int, int]
    labels: np.ndarray
    volumes: np.ndarray
    centroids_yxz: np.ndarray
    sample_coords_yxz: np.ndarray
    feature_mean: np.ndarray


@dataclass
class PrototypeTrackState:
    last_frame_index: int
    last_timepoint: str
    last_label: int
    last_centroid_yxz: np.ndarray
    feature_sum: np.ndarray
    feature_count: int


@dataclass
class FeatureMeanResult:
    config: FeatureMeanConfig
    frame_count: int
    tracks: pd.DataFrame
    tracks_by_method: dict[str, pd.DataFrame]
    assignments: pd.DataFrame
    metrics: pd.DataFrame | None
    pair_metrics: pd.DataFrame | None
    timings: pd.DataFrame


def _log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[feature-mean-tracking] {message}", flush=True)


def _progress_bar(
    iterable: Iterable[Any], *, total: int | None, desc: str, enabled: bool
) -> Iterable[Any]:
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


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
    if not lr_dir.is_dir():
        raise FileNotFoundError(f"Missing feature folder: {lr_dir}")
    if not segmentation_path.is_dir():
        raise FileNotFoundError(f"Missing segmentation folder: {segmentation_path}")

    frames: list[FramePaths] = []
    feature_paths = sorted(
        lr_dir.glob("*.npy"), key=lambda path: _natural_key(path.stem)
    )
    for index, feature_path in enumerate(feature_paths):
        mask_candidates = [
            segmentation_path / f"{feature_path.stem}.tif",
            segmentation_path / f"{feature_path.stem}.tiff",
        ]
        mask_path = next((path for path in mask_candidates if path.is_file()), None)
        if mask_path is None:
            raise FileNotFoundError(
                f"Missing segmentation mask for {feature_path.stem}."
            )
        frames.append(
            FramePaths(
                index=index,
                name=feature_path.stem,
                lr_features_path=feature_path,
                segmentation_path=mask_path,
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


def _resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device_name)


def _torch_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(token.strip() for token in value.split(",") if token.strip())
    return validate_methods(methods)


def validate_methods(methods: Sequence[str]) -> tuple[str, ...]:
    allowed = set(PAIRWISE_METHODS) | {
        METHOD_CENTROID_FEATURE_3FRAME,
        METHOD_CENTROID_FEATURE_PROTOTYPE,
    }
    normalized = tuple(str(method).strip() for method in methods if str(method).strip())
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. Allowed methods: {sorted(allowed)}"
        )
    if not normalized:
        raise ValueError("At least one tracking method is required.")
    return normalized


def method_needs_features(methods: Sequence[str]) -> bool:
    return bool(
        {
            METHOD_FEATURE_MEAN,
            METHOD_CENTROID_FEATURE,
            METHOD_CENTROID_FEATURE_3FRAME,
            METHOD_CENTROID_FEATURE_PROTOTYPE,
        }
        & set(methods)
    )


def pairwise_methods_from(methods: Sequence[str]) -> tuple[str, ...]:
    return tuple(method for method in methods if method in set(PAIRWISE_METHODS))


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("._") or "method"


def _stable_seed(base_seed: int, frame_index: int, label_id: int) -> int:
    mask = (1 << 64) - 1
    value = int(base_seed) & mask
    value ^= ((int(frame_index) + 1) * 0x9E3779B185EBCA87) & mask
    value ^= (int(label_id) * 0xC2B2AE3D27D4EB4F) & mask
    return int(value % (2**32 - 1))


def _sample_group_coords(
    coords: np.ndarray, *, samples_per_object: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    replace = coords.shape[0] < samples_per_object
    indices = rng.choice(coords.shape[0], size=samples_per_object, replace=replace)
    return coords[indices].astype(np.float32, copy=False)


def _prepare_object_metadata(
    frame: FramePaths, *, samples_per_object: int, seed: int
) -> tuple[
    tuple[int, int, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    segmentation = _read_tiff_yxz(frame.segmentation_path)
    flat = segmentation.ravel()
    foreground_flat = np.flatnonzero(flat != 0)
    high_shape_yxz = tuple(int(dim) for dim in segmentation.shape)
    if foreground_flat.size == 0:
        return (
            high_shape_yxz,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, samples_per_object, 3), dtype=np.float32),
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

    return (
        high_shape_yxz,
        labels.astype(np.int64, copy=False),
        counts.astype(np.int64, copy=False),
        centroids_yxz,
        sample_coords,
    )


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
def _compute_feature_means(
    frame: FramePaths,
    high_shape_yxz: tuple[int, int, int],
    sample_coords_yxz: np.ndarray,
    *,
    n_features: int,
    config: FeatureMeanConfig,
    device: torch.device,
) -> np.ndarray:
    n_objects = int(sample_coords_yxz.shape[0])
    if n_objects == 0:
        return np.empty((0, n_features), dtype=np.float32)

    lr_features = np.load(frame.lr_features_path, mmap_mode="r")
    if lr_features.ndim != 4:
        raise ValueError(f"Expected [Z, Y, X, C] features at {frame.lr_features_path}.")
    if lr_features.shape[-1] < n_features:
        raise ValueError(
            f"{frame.lr_features_path} only has {lr_features.shape[-1]} channels."
        )

    means = np.empty((n_objects, n_features), dtype=np.float32)
    n_samples = int(config.samples_per_object)
    for object_start in range(0, n_objects, int(config.object_batch_size)):
        object_end = min(n_objects, object_start + int(config.object_batch_size))
        object_count = object_end - object_start
        coords_np = sample_coords_yxz[object_start:object_end].reshape(-1, 3)
        coords = torch.from_numpy(coords_np.astype(np.float32, copy=False)).to(
            device=device, non_blocking=True
        )
        sampled_features = torch.empty(
            (coords.shape[0], n_features), dtype=torch.float32, device=device
        )
        for channel_start in range(0, n_features, int(config.feature_channel_block)):
            channel_end = min(
                n_features, channel_start + int(config.feature_channel_block)
            )
            chunk = np.asarray(
                lr_features[..., channel_start:channel_end], dtype=np.float32
            )
            sampled_features[:, channel_start:channel_end] = _sample_feature_chunk(
                chunk,
                coords,
                high_shape_yxz,
                device=device,
                sample_batch_size=int(config.sample_batch_size),
            )

        sampled_features = F.normalize(sampled_features, dim=1, eps=1.0e-12)
        object_means = sampled_features.view(object_count, n_samples, n_features).mean(
            dim=1
        )
        means[object_start:object_end] = object_means.cpu().numpy()

    return means


def build_frame_objects(
    frame: FramePaths,
    *,
    n_features: int | None,
    config: FeatureMeanConfig,
    device: torch.device,
) -> FrameObjects:
    high_shape_yxz, labels, volumes, centroids_yxz, sample_coords_yxz = (
        _prepare_object_metadata(
            frame,
            samples_per_object=int(config.samples_per_object),
            seed=int(config.seed),
        )
    )
    if n_features is None:
        feature_mean = np.empty((labels.size, 0), dtype=np.float32)
    else:
        feature_mean = _compute_feature_means(
            frame,
            high_shape_yxz,
            sample_coords_yxz,
            n_features=n_features,
            config=config,
            device=device,
        )
    return FrameObjects(
        index=int(frame.index),
        name=frame.name,
        high_shape_yxz=high_shape_yxz,
        labels=labels,
        volumes=volumes,
        centroids_yxz=centroids_yxz,
        sample_coords_yxz=sample_coords_yxz,
        feature_mean=feature_mean,
    )


@torch.no_grad()
def feature_vectors_cost_matrix(
    ref_features: np.ndarray,
    cand_features: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    if ref_features.ndim != 2 or cand_features.ndim != 2:
        raise ValueError("Feature matrices must be 2D.")
    if ref_features.shape[1] != cand_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {ref_features.shape}, {cand_features.shape}."
        )
    if ref_features.shape[0] == 0 or cand_features.shape[0] == 0:
        return np.empty(
            (ref_features.shape[0], cand_features.shape[0]), dtype=np.float32
        )
    ref_tensor = torch.from_numpy(ref_features.astype(np.float32, copy=False)).to(
        device=device
    )
    cand_tensor = torch.from_numpy(cand_features.astype(np.float32, copy=False)).to(
        device=device
    )
    ref_tensor = F.normalize(ref_tensor, dim=1, eps=1.0e-12)
    cand_tensor = F.normalize(cand_tensor, dim=1, eps=1.0e-12)
    similarity = ref_tensor @ cand_tensor.T
    return torch.clamp(1.0 - similarity, min=0.0, max=2.0).cpu().numpy()


def feature_mean_cost_matrix(
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    device: torch.device,
) -> np.ndarray:
    return feature_vectors_cost_matrix(
        ref.feature_mean, cand.feature_mean, device=device
    )


def centroid_vectors_cost_matrix(
    ref_centroids_yxz: np.ndarray,
    cand_centroids_yxz: np.ndarray,
    *,
    z_weight: float,
) -> np.ndarray:
    if ref_centroids_yxz.ndim != 2 or cand_centroids_yxz.ndim != 2:
        raise ValueError("Centroid matrices must be 2D.")
    if ref_centroids_yxz.shape[1] != 3 or cand_centroids_yxz.shape[1] != 3:
        raise ValueError(
            f"Expected YXZ centroid matrices, got {ref_centroids_yxz.shape}, {cand_centroids_yxz.shape}."
        )
    if ref_centroids_yxz.shape[0] == 0 or cand_centroids_yxz.shape[0] == 0:
        return np.empty(
            (ref_centroids_yxz.shape[0], cand_centroids_yxz.shape[0]),
            dtype=np.float32,
        )
    delta = cand_centroids_yxz[None, :, :].astype(
        np.float32, copy=False
    ) - ref_centroids_yxz[:, None, :].astype(np.float32, copy=False)
    dy = delta[..., 0]
    dx = delta[..., 1]
    dz = float(z_weight) * delta[..., 2]
    return np.sqrt((dx * dx) + (dy * dy) + (dz * dz)).astype(np.float32, copy=False)


def centroid_cost_matrix(
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    z_weight: float,
) -> np.ndarray:
    return centroid_vectors_cost_matrix(
        ref.centroids_yxz,
        cand.centroids_yxz,
        z_weight=z_weight,
    )


def search_window_mask_from_centroids(
    ref_centroids_yxz: np.ndarray,
    cand_centroids_yxz: np.ndarray,
    *,
    max_distance_xy: float,
    max_distance_z: float,
) -> np.ndarray:
    if ref_centroids_yxz.ndim != 2 or cand_centroids_yxz.ndim != 2:
        raise ValueError("Centroid matrices must be 2D.")
    if ref_centroids_yxz.shape[1] != 3 or cand_centroids_yxz.shape[1] != 3:
        raise ValueError(
            f"Expected YXZ centroid matrices, got {ref_centroids_yxz.shape}, {cand_centroids_yxz.shape}."
        )
    if ref_centroids_yxz.shape[0] == 0 or cand_centroids_yxz.shape[0] == 0:
        return np.zeros(
            (ref_centroids_yxz.shape[0], cand_centroids_yxz.shape[0]),
            dtype=bool,
        )

    delta = np.abs(
        cand_centroids_yxz[None, :, :].astype(np.float32, copy=False)
        - ref_centroids_yxz[:, None, :].astype(np.float32, copy=False)
    )
    return (
        (delta[..., 0] <= float(max_distance_xy))
        & (delta[..., 1] <= float(max_distance_xy))
        & (delta[..., 2] <= float(max_distance_z))
    )


def search_window_mask(
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    config: FeatureMeanConfig,
) -> np.ndarray:
    return search_window_mask_from_centroids(
        ref.centroids_yxz,
        cand.centroids_yxz,
        max_distance_xy=float(config.max_distance_xy),
        max_distance_z=float(config.max_distance_z),
    )


def _linear_sum_assignment_with_allowed_pairs(
    cost: np.ndarray,
    allowed_pairs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if allowed_pairs is None:
        return linear_sum_assignment(cost)

    allowed = np.asarray(allowed_pairs, dtype=bool)
    if allowed.shape != cost.shape:
        raise ValueError(
            f"Allowed-pair mask shape {allowed.shape} does not match cost shape {cost.shape}."
        )
    valid = allowed & np.isfinite(cost)
    if not np.any(valid):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    if np.all(valid):
        return linear_sum_assignment(cost)

    finite_values = cost[valid].astype(np.float64, copy=False)
    finite_min = float(np.min(finite_values))
    finite_max = float(np.max(finite_values))
    span = max(1.0, finite_max - finite_min)
    min_dimension = min(cost.shape)
    sentinel = finite_max + (span * float(min_dimension + 1))
    if not math.isfinite(sentinel):
        sentinel = np.finfo(np.float64).max / 4.0

    blocked_cost = np.where(valid, cost.astype(np.float64, copy=False), sentinel)
    row_indices, col_indices = linear_sum_assignment(blocked_cost)
    keep = valid[row_indices, col_indices]
    return row_indices[keep], col_indices[keep]


def robust_cost_scale(cost: np.ndarray) -> float:
    finite = cost[np.isfinite(cost)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return 1.0
    scale = float(np.median(finite))
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def normalized_cost(cost: np.ndarray, scale: float) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (cost / float(scale)).astype(np.float32, copy=False)


def pair_cost_matrices(
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    methods: Sequence[str],
    config: FeatureMeanConfig,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    methods = validate_methods(methods)
    costs: dict[str, np.ndarray] = {}
    diagnostics: dict[str, float] = {}

    centroid_cost: np.ndarray | None = None
    centroid_norm: np.ndarray | None = None
    if METHOD_CENTROID in methods or METHOD_CENTROID_FEATURE in methods:
        centroid_cost = centroid_cost_matrix(ref, cand, z_weight=float(config.z_weight))
        centroid_scale = robust_cost_scale(centroid_cost)
        centroid_norm = normalized_cost(centroid_cost, centroid_scale)
        diagnostics["centroid_scale"] = centroid_scale
        costs[METHOD_CENTROID] = centroid_norm

    feature_cost: np.ndarray | None = None
    feature_norm: np.ndarray | None = None
    if METHOD_FEATURE_MEAN in methods or METHOD_CENTROID_FEATURE in methods:
        feature_cost = feature_mean_cost_matrix(ref, cand, device=device)
        feature_scale = robust_cost_scale(feature_cost)
        feature_norm = normalized_cost(feature_cost, feature_scale)
        diagnostics["feature_scale"] = feature_scale
        if METHOD_FEATURE_MEAN in methods:
            costs[METHOD_FEATURE_MEAN] = feature_norm

    if METHOD_CENTROID_FEATURE in methods:
        if centroid_norm is None or feature_norm is None:
            raise RuntimeError("Internal error: missing centroid or feature cost.")
        clipped_feature = np.clip(
            feature_norm,
            0.0,
            float(config.feature_norm_clip),
        )
        costs[METHOD_CENTROID_FEATURE] = centroid_norm + (
            float(config.centroid_feature_weight) * clipped_feature
        )
        diagnostics["centroid_feature_weight"] = float(config.centroid_feature_weight)
        diagnostics["feature_norm_clip"] = float(config.feature_norm_clip)

    return {method: costs[method] for method in methods}, diagnostics


def _top_k_smallest_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size == 0:
        return np.empty((0,), dtype=np.int64)
    k = max(1, min(int(k), int(values.size)))
    if k == values.size:
        return np.argsort(values, kind="mergesort").astype(np.int64, copy=False)
    unsorted = np.argpartition(values, k - 1)[:k]
    order = np.argsort(values[unsorted], kind="mergesort")
    return unsorted[order].astype(np.int64, copy=False)


def build_three_frame_candidates(
    cost01: np.ndarray,
    cost12: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    n0, n1 = cost01.shape
    n1_b, n2 = cost12.shape
    if n1 != n1_b:
        raise ValueError(
            f"Shape mismatch for 3-frame costs: {cost01.shape}, {cost12.shape}."
        )
    if n0 == 0 or n1 == 0 or n2 == 0:
        return np.empty((0, 3), dtype=np.int64)

    k = max(1, int(top_k))
    prev_by_middle = [
        _top_k_smallest_indices(cost01[:, middle_index], k)
        for middle_index in range(n1)
    ]
    next_by_middle = [
        _top_k_smallest_indices(cost12[middle_index, :], k)
        for middle_index in range(n1)
    ]
    middle_by_prev = [
        _top_k_smallest_indices(cost01[prev_index, :], k) for prev_index in range(n0)
    ]
    middle_by_next = [
        _top_k_smallest_indices(cost12[:, next_index], k) for next_index in range(n2)
    ]

    candidates: set[tuple[int, int, int]] = set()
    for middle_index in range(n1):
        for prev_index in prev_by_middle[middle_index]:
            for next_index in next_by_middle[middle_index]:
                candidates.add((int(prev_index), middle_index, int(next_index)))

    for prev_index in range(n0):
        for middle_index in middle_by_prev[prev_index]:
            for next_index in next_by_middle[int(middle_index)]:
                candidates.add((prev_index, int(middle_index), int(next_index)))

    for next_index in range(n2):
        for middle_index in middle_by_next[next_index]:
            for prev_index in prev_by_middle[int(middle_index)]:
                candidates.add((int(prev_index), int(middle_index), next_index))

    if not candidates:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(sorted(candidates), dtype=np.int64)


def solve_three_frame_assignment(
    cost01: np.ndarray,
    cost12: np.ndarray,
    cost02: np.ndarray,
    *,
    direct_weight: float,
    candidate_top_k: int,
    time_limit_seconds: float,
) -> np.ndarray:
    n0, n1 = cost01.shape
    n1_b, n2 = cost12.shape
    if cost02.shape != (n0, n2) or n1 != n1_b:
        raise ValueError(
            f"Shape mismatch for 3-frame assignment: {cost01.shape}, {cost12.shape}, {cost02.shape}."
        )
    triplet_count = min(n0, n1, n2)
    if triplet_count == 0:
        return np.empty((0, 3), dtype=np.int64)

    candidates = build_three_frame_candidates(
        cost01,
        cost12,
        top_k=int(candidate_top_k),
    )
    if candidates.shape[0] < triplet_count:
        raise RuntimeError(
            "3-frame candidate set is too small for a feasible assignment. "
            "Increase three_frame_candidate_top_k."
        )

    i = candidates[:, 0]
    j = candidates[:, 1]
    k = candidates[:, 2]
    objective = (
        cost01[i, j] + cost12[j, k] + (float(direct_weight) * cost02[i, k])
    ).astype(np.float64, copy=False)

    n_vars = int(candidates.shape[0])
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    total_row = n0 + n1 + n2
    for var_index, (prev_index, middle_index, next_index) in enumerate(candidates):
        row_indices.extend([
            int(prev_index),
            n0 + int(middle_index),
            n0 + n1 + int(next_index),
            total_row,
        ])
        col_indices.extend([var_index, var_index, var_index, var_index])
        data.extend([1.0, 1.0, 1.0, 1.0])

    constraint_matrix = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(total_row + 1, n_vars),
    ).tocsr()
    lower = np.zeros((total_row + 1,), dtype=np.float64)
    upper = np.ones((total_row + 1,), dtype=np.float64)
    lower[total_row] = float(triplet_count)
    upper[total_row] = float(triplet_count)

    result = milp(
        c=objective,
        integrality=np.ones((n_vars,), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(constraint_matrix, lower, upper),
        options={"time_limit": float(time_limit_seconds)},
    )
    if result.x is None or not result.success:
        raise RuntimeError(f"3-frame MILP failed: {result.message}")

    selected = np.flatnonzero(result.x > 0.5)
    if selected.size != triplet_count:
        raise RuntimeError(
            f"3-frame MILP selected {selected.size} triplets, expected {triplet_count}."
        )
    return candidates[selected]


def _f1(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall):
        return np.nan
    if precision + recall == 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _score_assignments(
    assignment_rows: list[dict[str, Any]],
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    method: str,
    diagnostics: dict[str, float],
) -> dict[str, Any]:
    trackable_count = len(set(ref.labels.tolist()) & set(cand.labels.tolist()))
    tp = sum(1 for row in assignment_rows if bool(row["is_true_link"]))
    pred_count = len(assignment_rows)
    fp = pred_count - tp
    fn = trackable_count - tp
    precision = float(tp / pred_count) if pred_count else np.nan
    recall = float(tp / trackable_count) if trackable_count else np.nan
    centroid_feature_methods = {
        METHOD_CENTROID_FEATURE,
        METHOD_CENTROID_FEATURE_3FRAME,
        METHOD_CENTROID_FEATURE_PROTOTYPE,
    }
    uses_centroid = method in {METHOD_CENTROID} | centroid_feature_methods
    uses_feature = method in {METHOD_FEATURE_MEAN} | centroid_feature_methods
    return {
        "method": method,
        "ref_frame_index": int(ref.index),
        "cand_frame_index": int(cand.index),
        "ref_timepoint": ref.name,
        "cand_timepoint": cand.name,
        "ref_count": int(ref.labels.size),
        "cand_count": int(cand.labels.size),
        "trackable_count": int(trackable_count),
        "link_pred_count": int(pred_count),
        "link_tp": int(tp),
        "link_fp": int(fp),
        "link_fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "centroid_scale": diagnostics.get("centroid_scale", np.nan)
        if uses_centroid
        else np.nan,
        "feature_scale": diagnostics.get("feature_scale", np.nan)
        if uses_feature
        else np.nan,
        "centroid_feature_weight": diagnostics.get("centroid_feature_weight", np.nan)
        if method in centroid_feature_methods
        else np.nan,
        "feature_norm_clip": diagnostics.get("feature_norm_clip", np.nan)
        if method in centroid_feature_methods
        else np.nan,
    }


def match_adjacent_pair(
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    methods: Sequence[str],
    config: FeatureMeanConfig,
    device: torch.device,
    compute_gt_metrics: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cost_matrices, diagnostics = pair_cost_matrices(
        ref,
        cand,
        methods=methods,
        config=config,
        device=device,
    )
    allowed_pairs = search_window_mask(ref, cand, config=config)
    assignment_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    for method, cost in cost_matrices.items():
        method_rows: list[dict[str, Any]] = []
        if cost.size > 0 and ref.labels.size > 0 and cand.labels.size > 0:
            row_indices, col_indices = _linear_sum_assignment_with_allowed_pairs(
                cost,
                allowed_pairs,
            )
            for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
                ref_label = int(ref.labels[row_index])
                cand_label = int(cand.labels[col_index])
                row: dict[str, Any] = {
                    "method": method,
                    "ref_frame_index": int(ref.index),
                    "cand_frame_index": int(cand.index),
                    "ref_timepoint": ref.name,
                    "cand_timepoint": cand.name,
                    "ref_label": ref_label,
                    "assigned_cand_label": cand_label,
                    "cost": float(cost[row_index, col_index]),
                }
                if compute_gt_metrics:
                    row["is_true_link"] = bool(ref_label == cand_label)
                method_rows.append(row)
        assignment_rows.extend(method_rows)
        if compute_gt_metrics:
            pair_metric_rows.append(
                _score_assignments(
                    method_rows,
                    ref,
                    cand,
                    method=method,
                    diagnostics=diagnostics,
                )
            )

    return assignment_rows, pair_metric_rows


def _normalized_feature_rows(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got {features.shape}.")
    if features.shape[0] == 0:
        return features.astype(np.float32, copy=True)
    values = features.astype(np.float32, copy=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(values, np.maximum(norms, 1.0e-12), out=values)
    return values


def _prototype_state_from_object(
    frame: FrameObjects,
    object_index: int,
    normalized_features: np.ndarray,
) -> PrototypeTrackState:
    return PrototypeTrackState(
        last_frame_index=int(frame.index),
        last_timepoint=frame.name,
        last_label=int(frame.labels[object_index]),
        last_centroid_yxz=frame.centroids_yxz[object_index].astype(
            np.float32, copy=True
        ),
        feature_sum=normalized_features[object_index].astype(np.float32, copy=True),
        feature_count=1,
    )


def _prototype_feature_matrix(states: Sequence[PrototypeTrackState]) -> np.ndarray:
    if not states:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(
        [
            state.feature_sum.astype(np.float32, copy=False)
            / max(1, int(state.feature_count))
            for state in states
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def prototype_cost_matrix(
    states: Sequence[PrototypeTrackState],
    cand: FrameObjects,
    *,
    config: FeatureMeanConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    diagnostics = {
        "centroid_feature_weight": float(config.centroid_feature_weight),
        "feature_norm_clip": float(config.feature_norm_clip),
    }
    if not states or cand.labels.size == 0:
        return (
            np.empty((len(states), cand.labels.size), dtype=np.float32),
            diagnostics,
        )

    ref_centroids = np.stack(
        [state.last_centroid_yxz for state in states], axis=0
    ).astype(np.float32, copy=False)
    centroid_cost = centroid_vectors_cost_matrix(
        ref_centroids,
        cand.centroids_yxz,
        z_weight=float(config.z_weight),
    )
    centroid_scale = robust_cost_scale(centroid_cost)
    centroid_norm = normalized_cost(centroid_cost, centroid_scale)
    diagnostics["centroid_scale"] = centroid_scale

    prototype_features = _prototype_feature_matrix(states)
    feature_cost = feature_vectors_cost_matrix(
        prototype_features,
        cand.feature_mean,
        device=device,
    )
    feature_scale = robust_cost_scale(feature_cost)
    feature_norm = normalized_cost(feature_cost, feature_scale)
    diagnostics["feature_scale"] = feature_scale

    clipped_feature = np.clip(feature_norm, 0.0, float(config.feature_norm_clip))
    cost = centroid_norm + (float(config.centroid_feature_weight) * clipped_feature)
    return cost.astype(np.float32, copy=False), diagnostics


def prototype_search_window_mask(
    states: Sequence[PrototypeTrackState],
    cand: FrameObjects,
    *,
    config: FeatureMeanConfig,
) -> np.ndarray:
    if not states:
        return np.zeros((0, cand.labels.size), dtype=bool)
    ref_centroids = np.stack(
        [state.last_centroid_yxz for state in states], axis=0
    ).astype(np.float32, copy=False)
    return search_window_mask_from_centroids(
        ref_centroids,
        cand.centroids_yxz,
        max_distance_xy=float(config.max_distance_xy),
        max_distance_z=float(config.max_distance_z),
    )


def match_prototype_tracks(
    frames: Sequence[FrameObjects],
    *,
    config: FeatureMeanConfig,
    device: torch.device,
    compute_gt_metrics: bool,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    if len(frames) < 2:
        return assignment_rows, pair_metric_rows

    first_features = _normalized_feature_rows(frames[0].feature_mean)
    active_states = [
        _prototype_state_from_object(frames[0], object_index, first_features)
        for object_index in range(frames[0].labels.size)
    ]

    frame_iter = enumerate(frames[1:], start=1)
    for frame_position, cand in _progress_bar(
        frame_iter,
        total=max(0, len(frames) - 1),
        desc="prototype pairs",
        enabled=progress,
    ):
        ref = frames[frame_position - 1]
        method_rows: list[dict[str, Any]] = []
        diagnostics: dict[str, float] = {
            "centroid_feature_weight": float(config.centroid_feature_weight),
            "feature_norm_clip": float(config.feature_norm_clip),
        }
        assigned_candidate_indices: set[int] = set()
        next_states: list[PrototypeTrackState] = []
        normalized_cand_features = _normalized_feature_rows(cand.feature_mean)

        cost, diagnostics = prototype_cost_matrix(
            active_states,
            cand,
            config=config,
            device=device,
        )
        allowed_pairs = prototype_search_window_mask(
            active_states,
            cand,
            config=config,
        )
        if cost.size > 0:
            row_indices, col_indices = _linear_sum_assignment_with_allowed_pairs(
                cost,
                allowed_pairs,
            )
            for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
                state = active_states[row_index]
                cand_label = int(cand.labels[col_index])
                row: dict[str, Any] = {
                    "method": METHOD_CENTROID_FEATURE_PROTOTYPE,
                    "ref_frame_index": int(state.last_frame_index),
                    "cand_frame_index": int(cand.index),
                    "ref_timepoint": state.last_timepoint,
                    "cand_timepoint": cand.name,
                    "ref_label": int(state.last_label),
                    "assigned_cand_label": cand_label,
                    "cost": float(cost[row_index, col_index]),
                    "prototype_observations": int(state.feature_count),
                }
                if compute_gt_metrics:
                    row["is_true_link"] = bool(int(state.last_label) == cand_label)
                method_rows.append(row)

                next_states.append(
                    PrototypeTrackState(
                        last_frame_index=int(cand.index),
                        last_timepoint=cand.name,
                        last_label=cand_label,
                        last_centroid_yxz=cand.centroids_yxz[col_index].astype(
                            np.float32, copy=True
                        ),
                        feature_sum=state.feature_sum
                        + normalized_cand_features[col_index],
                        feature_count=int(state.feature_count) + 1,
                    )
                )
                assigned_candidate_indices.add(int(col_index))

        for cand_index in range(cand.labels.size):
            if int(cand_index) not in assigned_candidate_indices:
                next_states.append(
                    _prototype_state_from_object(
                        cand,
                        int(cand_index),
                        normalized_cand_features,
                    )
                )

        assignment_rows.extend(method_rows)
        if compute_gt_metrics:
            pair_metric_rows.append(
                _score_assignments(
                    method_rows,
                    ref,
                    cand,
                    method=METHOD_CENTROID_FEATURE_PROTOTYPE,
                    diagnostics=diagnostics,
                )
            )
        active_states = next_states

    return assignment_rows, pair_metric_rows


def _assignment_rows_from_cost(
    ref: FrameObjects,
    cand: FrameObjects,
    cost: np.ndarray,
    *,
    method: str,
    compute_gt_metrics: bool,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if cost.size == 0 or ref.labels.size == 0 or cand.labels.size == 0:
        return rows
    row_indices, col_indices = linear_sum_assignment(cost)
    for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
        ref_label = int(ref.labels[row_index])
        cand_label = int(cand.labels[col_index])
        row: dict[str, Any] = {
            "method": method,
            "ref_frame_index": int(ref.index),
            "cand_frame_index": int(cand.index),
            "ref_timepoint": ref.name,
            "cand_timepoint": cand.name,
            "ref_label": ref_label,
            "assigned_cand_label": cand_label,
            "cost": float(cost[row_index, col_index]),
        }
        if extra:
            row.update(extra)
        if compute_gt_metrics:
            row["is_true_link"] = bool(ref_label == cand_label)
        rows.append(row)
    return rows


def _triplet_assignment_rows(
    first: FrameObjects,
    middle: FrameObjects,
    last: FrameObjects,
    triplets: np.ndarray,
    cost01: np.ndarray,
    cost12: np.ndarray,
    cost02: np.ndarray,
    *,
    compute_gt_metrics: bool,
    window_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for triplet_index, (first_index, middle_index, last_index) in enumerate(triplets):
        first_label = int(first.labels[first_index])
        middle_label = int(middle.labels[middle_index])
        last_label = int(last.labels[last_index])
        path_cost = float(
            cost01[first_index, middle_index]
            + cost12[middle_index, last_index]
            + cost02[first_index, last_index]
        )
        first_row: dict[str, Any] = {
            "method": METHOD_CENTROID_FEATURE_3FRAME,
            "ref_frame_index": int(first.index),
            "cand_frame_index": int(middle.index),
            "ref_timepoint": first.name,
            "cand_timepoint": middle.name,
            "ref_label": first_label,
            "assigned_cand_label": middle_label,
            "cost": float(cost01[first_index, middle_index]),
            "three_frame_window": int(window_index),
            "three_frame_triplet": int(triplet_index),
            "three_frame_path_cost": path_cost,
            "three_frame_direct_cost": float(cost02[first_index, last_index]),
        }
        second_row: dict[str, Any] = {
            "method": METHOD_CENTROID_FEATURE_3FRAME,
            "ref_frame_index": int(middle.index),
            "cand_frame_index": int(last.index),
            "ref_timepoint": middle.name,
            "cand_timepoint": last.name,
            "ref_label": middle_label,
            "assigned_cand_label": last_label,
            "cost": float(cost12[middle_index, last_index]),
            "three_frame_window": int(window_index),
            "three_frame_triplet": int(triplet_index),
            "three_frame_path_cost": path_cost,
            "three_frame_direct_cost": float(cost02[first_index, last_index]),
        }
        if compute_gt_metrics:
            first_row["is_true_link"] = bool(first_label == middle_label)
            second_row["is_true_link"] = bool(middle_label == last_label)
        rows.extend([first_row, second_row])
    return rows


def match_three_frame_blocks(
    frames: Sequence[FrameObjects],
    *,
    config: FeatureMeanConfig,
    device: torch.device,
    compute_gt_metrics: bool,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    if len(frames) < 2:
        return assignment_rows, pair_metric_rows

    window_starts = list(range(0, max(0, len(frames) - 2), 2))
    for window_number, start in enumerate(
        _progress_bar(
            window_starts,
            total=len(window_starts),
            desc="3-frame windows",
            enabled=progress,
        )
    ):
        first = frames[start]
        middle = frames[start + 1]
        last = frames[start + 2]
        cost01, _ = pair_cost_matrices(
            first,
            middle,
            methods=(METHOD_CENTROID_FEATURE,),
            config=config,
            device=device,
        )
        cost12, _ = pair_cost_matrices(
            middle,
            last,
            methods=(METHOD_CENTROID_FEATURE,),
            config=config,
            device=device,
        )
        cost02, _ = pair_cost_matrices(
            first,
            last,
            methods=(METHOD_CENTROID_FEATURE,),
            config=config,
            device=device,
        )
        matrix01 = cost01[METHOD_CENTROID_FEATURE]
        matrix12 = cost12[METHOD_CENTROID_FEATURE]
        matrix02 = cost02[METHOD_CENTROID_FEATURE]
        triplets = solve_three_frame_assignment(
            matrix01,
            matrix12,
            matrix02,
            direct_weight=float(config.three_frame_direct_weight),
            candidate_top_k=int(config.three_frame_candidate_top_k),
            time_limit_seconds=float(config.three_frame_time_limit_seconds),
        )
        assignment_rows.extend(
            _triplet_assignment_rows(
                first,
                middle,
                last,
                triplets,
                matrix01,
                matrix12,
                matrix02,
                compute_gt_metrics=compute_gt_metrics,
                window_index=window_number,
            )
        )

    covered_until = 0
    if window_starts:
        covered_until = window_starts[-1] + 2
    if covered_until < len(frames) - 1:
        ref = frames[covered_until]
        cand = frames[covered_until + 1]
        cost_matrices, _ = pair_cost_matrices(
            ref,
            cand,
            methods=(METHOD_CENTROID_FEATURE,),
            config=config,
            device=device,
        )
        assignment_rows.extend(
            _assignment_rows_from_cost(
                ref,
                cand,
                cost_matrices[METHOD_CENTROID_FEATURE],
                method=METHOD_CENTROID_FEATURE_3FRAME,
                compute_gt_metrics=compute_gt_metrics,
                extra={
                    "three_frame_window": np.nan,
                    "three_frame_triplet": np.nan,
                    "three_frame_path_cost": np.nan,
                    "three_frame_direct_cost": np.nan,
                },
            )
        )

    if compute_gt_metrics and assignment_rows:
        assignments = pd.DataFrame(assignment_rows)
        for (ref_index, cand_index), group in assignments.groupby(
            ["ref_frame_index", "cand_frame_index"], sort=True
        ):
            ref = next(frame for frame in frames if int(frame.index) == int(ref_index))
            cand = next(
                frame for frame in frames if int(frame.index) == int(cand_index)
            )
            pair_metric_rows.append(
                _score_assignments(
                    group.to_dict("records"),
                    ref,
                    cand,
                    method=METHOD_CENTROID_FEATURE_3FRAME,
                    diagnostics={
                        "centroid_feature_weight": float(
                            config.centroid_feature_weight
                        ),
                        "feature_norm_clip": float(config.feature_norm_clip),
                    },
                )
            )

    return assignment_rows, pair_metric_rows


def aggregate_pair_metrics(pair_metrics: pd.DataFrame) -> pd.DataFrame:
    if pair_metrics.empty:
        return pd.DataFrame()

    def finite_median(series: pd.Series) -> float:
        values = series.to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else np.nan

    rows: list[dict[str, Any]] = []
    for method, group in pair_metrics.groupby("method", sort=False):
        pred = int(group["link_pred_count"].sum())
        tp = int(group["link_tp"].sum())
        fp = int(group["link_fp"].sum())
        fn = int(group["link_fn"].sum())
        trackable = int(group["trackable_count"].sum())
        precision = float(tp / pred) if pred else np.nan
        recall = float(tp / trackable) if trackable else np.nan
        rows.append({
            "method": method,
            "frame_pairs": int(group.shape[0]),
            "trackable_count": trackable,
            "link_pred_count": pred,
            "link_tp": tp,
            "link_fp": fp,
            "link_fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "median_centroid_scale": finite_median(group["centroid_scale"])
            if "centroid_scale" in group
            else np.nan,
            "median_feature_scale": finite_median(group["feature_scale"])
            if "feature_scale" in group
            else np.nan,
            "centroid_feature_weight": finite_median(group["centroid_feature_weight"])
            if "centroid_feature_weight" in group
            else np.nan,
            "feature_norm_clip": finite_median(group["feature_norm_clip"])
            if "feature_norm_clip" in group
            else np.nan,
        })
    return pd.DataFrame(rows)


def build_tracks(
    frames: Sequence[FrameObjects],
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(TRACK_COLUMNS))

    object_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    labels_by_frame: dict[int, list[int]] = {}
    for frame in frames:
        labels_by_frame[int(frame.index)] = [int(label) for label in frame.labels]
        for label_index, label_id in enumerate(frame.labels.tolist()):
            centroid = frame.centroids_yxz[label_index]
            object_lookup[(int(frame.index), int(label_id))] = {
                "frame_index": int(frame.index),
                "label_id": int(label_id),
                "x": float(centroid[1]),
                "y": float(centroid[0]),
                "z": float(centroid[2]),
                "A": 1.0,
            }

    rows: list[dict[str, Any]] = []
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
        rows.append({"track_id": int(track_id), **info})

    def start_track(frame_index: int, label_id: int) -> int:
        nonlocal next_track_id
        track_id = next_track_id
        next_track_id += 1
        append_point(track_id, frame_index, label_id)
        return track_id

    first_frame = int(frames[0].index)
    for label_id in labels_by_frame.get(first_frame, []):
        active[int(label_id)] = start_track(first_frame, int(label_id))

    if assignments.empty:
        for frame in frames[1:]:
            for label_id in labels_by_frame.get(int(frame.index), []):
                start_track(int(frame.index), int(label_id))
    else:
        pair_frames = (
            assignments[["ref_frame_index", "cand_frame_index"]]
            .drop_duplicates()
            .sort_values(["ref_frame_index", "cand_frame_index"], kind="mergesort")
        )
        for pair in pair_frames.itertuples(index=False):
            ref_frame = int(pair.ref_frame_index)
            cand_frame = int(pair.cand_frame_index)
            for label_id in labels_by_frame.get(ref_frame, []):
                if int(label_id) not in active:
                    active[int(label_id)] = start_track(ref_frame, int(label_id))

            pair_assignments = assignments[
                (assignments["ref_frame_index"] == ref_frame)
                & (assignments["cand_frame_index"] == cand_frame)
            ]
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

    tracks = pd.DataFrame(rows)
    if tracks.empty:
        return pd.DataFrame(columns=list(TRACK_COLUMNS))
    tracks = tracks.sort_values(["track_id", "frame_index"], kind="mergesort")
    tracks["start"] = tracks.groupby("track_id")["frame_index"].transform("min") + 1
    tracks["t"] = tracks.groupby("track_id", sort=False).cumcount() + 1
    tracks["track_length"] = tracks.groupby("track_id")["frame_index"].transform(
        "count"
    )
    return tracks.loc[:, TRACK_COLUMNS].reset_index(drop=True)


def build_tracks_by_method(
    frames: Sequence[FrameObjects],
    assignments: pd.DataFrame,
    *,
    methods: Sequence[str],
) -> dict[str, pd.DataFrame]:
    if assignments.empty or "method" not in assignments.columns:
        return {
            str(method): build_tracks(frames, assignments) for method in methods
        }
    tracks_by_method: dict[str, pd.DataFrame] = {}
    for method in methods:
        method_assignments = assignments[assignments["method"] == method]
        tracks_by_method[str(method)] = build_tracks(frames, method_assignments)
    return tracks_by_method


def choose_tracks_method(
    tracks_by_method: dict[str, pd.DataFrame],
    *,
    requested: str,
    methods: Sequence[str],
) -> str:
    if requested in tracks_by_method:
        return requested
    for method in methods:
        if method in tracks_by_method:
            return str(method)
    if tracks_by_method:
        return next(iter(tracks_by_method))
    raise ValueError("No tracks are available.")


def run_feature_mean_tracking(
    input_path: str | Path,
    segmentation_path: str | Path,
    *,
    config: FeatureMeanConfig | None = None,
) -> FeatureMeanResult:
    config = FeatureMeanConfig() if config is None else config
    methods = validate_methods(config.methods)
    if int(config.samples_per_object) <= 0:
        raise ValueError("samples_per_object must be positive.")
    if int(config.object_batch_size) <= 0:
        raise ValueError("object_batch_size must be positive.")
    if int(config.feature_channel_block) <= 0:
        raise ValueError("feature_channel_block must be positive.")
    if float(config.centroid_feature_weight) < 0.0:
        raise ValueError("centroid_feature_weight must be non-negative.")
    if float(config.feature_norm_clip) <= 0.0:
        raise ValueError("feature_norm_clip must be positive.")
    if float(config.z_weight) <= 0.0:
        raise ValueError("z_weight must be positive.")
    if float(config.max_distance_xy) <= 0.0:
        raise ValueError("max_distance_xy must be positive.")
    if float(config.max_distance_z) <= 0.0:
        raise ValueError("max_distance_z must be positive.")
    if float(config.three_frame_direct_weight) < 0.0:
        raise ValueError("three_frame_direct_weight must be non-negative.")
    if int(config.three_frame_candidate_top_k) <= 0:
        raise ValueError("three_frame_candidate_top_k must be positive.")
    if float(config.three_frame_time_limit_seconds) <= 0.0:
        raise ValueError("three_frame_time_limit_seconds must be positive.")

    total_start = time.perf_counter()
    frames = discover_frames(input_path, segmentation_path)
    discovered_count = len(frames)
    if config.max_frames is not None:
        if int(config.max_frames) < 2:
            raise ValueError("max_frames must be at least 2 when provided.")
        frames = frames[: int(config.max_frames)]
        if len(frames) < 2:
            raise ValueError(
                f"Only {len(frames)} frame(s) available after max_frames={config.max_frames}."
            )
        _log(
            f"limiting run to first {len(frames)}/{discovered_count} discovered frame(s)",
            enabled=bool(config.progress),
        )

    needs_features = method_needs_features(methods)
    available_features = infer_feature_count(frames) if needs_features else 0
    n_features = (
        resolve_n_features(config.n_features, available_features)
        if needs_features
        else None
    )
    device_name = _resolve_device(config.device)
    device = _torch_device(device_name)
    if needs_features:
        _log(
            (
                f"found {len(frames)} frame(s); using {n_features}/{available_features} "
                f"feature channel(s), {config.samples_per_object} sample(s)/object, "
                f"methods={methods}, device={device_name}"
            ),
            enabled=bool(config.progress),
        )
    else:
        _log(
            f"found {len(frames)} frame(s); methods={methods}, device={device_name}",
            enabled=bool(config.progress),
        )

    feature_start = time.perf_counter()
    frame_objects = [
        build_frame_objects(
            frame,
            n_features=n_features,
            config=config,
            device=device,
        )
        for frame in _progress_bar(
            frames,
            total=len(frames),
            desc="feature means",
            enabled=bool(config.progress),
        )
    ]
    feature_seconds = time.perf_counter() - feature_start
    _log(
        f"feature means completed in {feature_seconds:.2f}s",
        enabled=bool(config.progress),
    )

    match_start = time.perf_counter()
    assignment_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    pairwise_methods = pairwise_methods_from(methods)
    if pairwise_methods:
        pair_iter = zip(frame_objects[:-1], frame_objects[1:])
        for ref, cand in _progress_bar(
            pair_iter,
            total=max(0, len(frame_objects) - 1),
            desc="adjacent pairs",
            enabled=bool(config.progress),
        ):
            assignments, pair_metrics = match_adjacent_pair(
                ref,
                cand,
                methods=pairwise_methods,
                config=config,
                device=device,
                compute_gt_metrics=bool(config.compute_gt_metrics),
            )
            assignment_rows.extend(assignments)
            pair_metric_rows.extend(pair_metrics)

    if METHOD_CENTROID_FEATURE_3FRAME in methods:
        assignments, pair_metrics = match_three_frame_blocks(
            frame_objects,
            config=config,
            device=device,
            compute_gt_metrics=bool(config.compute_gt_metrics),
            progress=bool(config.progress),
        )
        assignment_rows.extend(assignments)
        pair_metric_rows.extend(pair_metrics)

    if METHOD_CENTROID_FEATURE_PROTOTYPE in methods:
        assignments, pair_metrics = match_prototype_tracks(
            frame_objects,
            config=config,
            device=device,
            compute_gt_metrics=bool(config.compute_gt_metrics),
            progress=bool(config.progress),
        )
        assignment_rows.extend(assignments)
        pair_metric_rows.extend(pair_metrics)
    assignments = pd.DataFrame(assignment_rows)
    pair_metrics = pd.DataFrame(pair_metric_rows) if pair_metric_rows else None
    metrics = (
        aggregate_pair_metrics(pair_metrics)
        if pair_metrics is not None and not pair_metrics.empty
        else None
    )
    tracks_by_method = build_tracks_by_method(
        frame_objects, assignments, methods=methods
    )
    selected_tracks_method = choose_tracks_method(
        tracks_by_method,
        requested=str(config.tracks_method),
        methods=methods,
    )
    tracks = tracks_by_method[selected_tracks_method]
    match_seconds = time.perf_counter() - match_start
    total_seconds = time.perf_counter() - total_start

    timings = pd.DataFrame([
        {"stage": "feature_means", "seconds": feature_seconds},
        {"stage": "matching_and_tracks", "seconds": match_seconds},
        {"stage": "total", "seconds": total_seconds},
    ])
    _log(
        f"matching and tracks completed in {match_seconds:.2f}s; total {total_seconds:.2f}s",
        enabled=bool(config.progress),
    )

    return FeatureMeanResult(
        config=config,
        frame_count=len(frame_objects),
        tracks=tracks,
        tracks_by_method=tracks_by_method,
        assignments=assignments,
        metrics=metrics,
        pair_metrics=pair_metrics,
        timings=timings,
    )


def save_result(result: FeatureMeanResult, output_path: str | Path) -> dict[str, str]:
    output_path = Path(output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {
        "tracks_csv": str(output_path / "tracks.csv"),
        "assignments_csv": str(output_path / "assignments.csv"),
        "timings_csv": str(output_path / "timings.csv"),
        "config_json": str(output_path / "config.json"),
    }
    result.tracks.to_csv(paths["tracks_csv"], index=False)
    for method, tracks in result.tracks_by_method.items():
        safe_method = _safe_filename_token(method)
        flat_path = output_path / f"tracks_{safe_method}.csv"
        nested_path = output_path / "tracks_by_method" / safe_method / "tracks.csv"
        flat_path.parent.mkdir(parents=True, exist_ok=True)
        nested_path.parent.mkdir(parents=True, exist_ok=True)
        tracks.to_csv(flat_path, index=False)
        tracks.to_csv(nested_path, index=False)
        paths[f"tracks_{safe_method}_csv"] = str(flat_path)
        paths[f"tracks_{safe_method}_nested_csv"] = str(nested_path)
    paths["tracks_default_method"] = choose_tracks_method(
        result.tracks_by_method,
        requested=str(result.config.tracks_method),
        methods=result.config.methods,
    )
    result.assignments.to_csv(paths["assignments_csv"], index=False)
    result.timings.to_csv(paths["timings_csv"], index=False)
    if result.metrics is not None:
        paths["metrics_csv"] = str(output_path / "metrics.csv")
        result.metrics.to_csv(paths["metrics_csv"], index=False)
    if result.pair_metrics is not None:
        paths["pair_metrics_csv"] = str(output_path / "pair_metrics.csv")
        result.pair_metrics.to_csv(paths["pair_metrics_csv"], index=False)
    Path(paths["config_json"]).write_text(
        json.dumps(asdict(result.config), indent=2), encoding="utf-8"
    )
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barebones feature-mean tracker for segmented objects."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Inference output folder containing lr_feats/.",
    )
    parser.add_argument(
        "--segmentation-path",
        required=True,
        help="Segmentation folder with <timepoint>.tif files.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Folder where tracks.csv and optional metrics will be written.",
    )
    parser.add_argument(
        "--n-features", default="384", help="Feature channels to use, or 'all'."
    )
    parser.add_argument("--samples-per-object", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated methods to run: centroid,feature_mean,centroid_feature,"
            "centroid_feature_3frame,centroid_feature_prototype. Defaults to the "
            "three pairwise methods."
        ),
    )
    parser.add_argument(
        "--tracks-method",
        default=DEFAULT_TRACKS_METHOD,
        help="Method used for the top-level conventional tracks.csv.",
    )
    parser.add_argument(
        "--centroid-feature-weight",
        type=float,
        default=0.1,
        help="Weight for normalized clipped feature cost in centroid_feature.",
    )
    parser.add_argument(
        "--feature-norm-clip",
        type=float,
        default=3.0,
        help="Upper clip for normalized feature cost in centroid_feature.",
    )
    parser.add_argument(
        "--z-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to z centroid deltas before distance computation.",
    )
    parser.add_argument(
        "--max-distance-xy",
        type=float,
        default=20.0,
        help="Maximum candidate centroid displacement along X and Y, in voxels.",
    )
    parser.add_argument(
        "--max-distance-z",
        type=float,
        default=10.0,
        help="Maximum candidate centroid displacement along Z, in voxels.",
    )
    parser.add_argument(
        "--three-frame-direct-weight",
        type=float,
        default=0.25,
        help="Weight for the direct t->t+2 cost in centroid_feature_3frame.",
    )
    parser.add_argument(
        "--three-frame-candidate-top-k",
        type=int,
        default=8,
        help="Top-k candidate neighborhood used to build the 3-frame MILP triplet set.",
    )
    parser.add_argument(
        "--three-frame-time-limit-seconds",
        type=float,
        default=60.0,
        help="MILP time limit per 3-frame window.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-gt-metrics",
        action="store_true",
        help="Skip precision/recall/F1 scoring from persistent GT labels.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars and stage logs.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FeatureMeanConfig:
    n_features: int | str
    if str(args.n_features).strip().lower() == "all":
        n_features = "all"
    else:
        n_features = int(args.n_features)
    return FeatureMeanConfig(
        n_features=n_features,
        samples_per_object=int(args.samples_per_object),
        seed=int(args.seed),
        device=str(args.device),
        methods=_parse_methods(str(args.methods)),
        tracks_method=str(args.tracks_method),
        centroid_feature_weight=float(args.centroid_feature_weight),
        feature_norm_clip=float(args.feature_norm_clip),
        z_weight=float(args.z_weight),
        max_distance_xy=float(args.max_distance_xy),
        max_distance_z=float(args.max_distance_z),
        three_frame_direct_weight=float(args.three_frame_direct_weight),
        three_frame_candidate_top_k=int(args.three_frame_candidate_top_k),
        three_frame_time_limit_seconds=float(args.three_frame_time_limit_seconds),
        max_frames=args.max_frames,
        compute_gt_metrics=not bool(args.no_gt_metrics),
        progress=not bool(args.no_progress),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    result = run_feature_mean_tracking(
        args.input_path,
        args.segmentation_path,
        config=config,
    )
    paths = save_result(result, args.output_path)
    if result.metrics is not None:
        print(result.metrics.to_string(index=False))
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
