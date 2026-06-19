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
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm


METHOD_CENTROID = "centroid"
METHOD_FEATURE_MEAN = "feature_mean"
METHOD_CENTROID_FEATURE = "centroid_feature"
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
    allowed = set(DEFAULT_METHODS)
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
    return bool({METHOD_FEATURE_MEAN, METHOD_CENTROID_FEATURE} & set(methods))


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
def feature_mean_cost_matrix(
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    device: torch.device,
) -> np.ndarray:
    if ref.labels.size == 0 or cand.labels.size == 0:
        return np.empty((ref.labels.size, cand.labels.size), dtype=np.float32)
    ref_features = torch.from_numpy(ref.feature_mean).to(device=device)
    cand_features = torch.from_numpy(cand.feature_mean).to(device=device)
    ref_features = F.normalize(ref_features, dim=1, eps=1.0e-12)
    cand_features = F.normalize(cand_features, dim=1, eps=1.0e-12)
    similarity = ref_features @ cand_features.T
    return torch.clamp(1.0 - similarity, min=0.0, max=2.0).cpu().numpy()


def centroid_cost_matrix(
    ref: FrameObjects,
    cand: FrameObjects,
    *,
    z_weight: float,
) -> np.ndarray:
    if ref.labels.size == 0 or cand.labels.size == 0:
        return np.empty((ref.labels.size, cand.labels.size), dtype=np.float32)
    delta = cand.centroids_yxz[None, :, :].astype(
        np.float32, copy=False
    ) - ref.centroids_yxz[:, None, :].astype(np.float32, copy=False)
    dy = delta[..., 0]
    dx = delta[..., 1]
    dz = float(z_weight) * delta[..., 2]
    return np.sqrt((dx * dx) + (dy * dy) + (dz * dz)).astype(np.float32, copy=False)


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
    uses_centroid = method in {METHOD_CENTROID, METHOD_CENTROID_FEATURE}
    uses_feature = method in {METHOD_FEATURE_MEAN, METHOD_CENTROID_FEATURE}
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
        if method == METHOD_CENTROID_FEATURE
        else np.nan,
        "feature_norm_clip": diagnostics.get("feature_norm_clip", np.nan)
        if method == METHOD_CENTROID_FEATURE
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
    assignment_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []
    for method, cost in cost_matrices.items():
        method_rows: list[dict[str, Any]] = []
        if cost.size > 0 and ref.labels.size > 0 and cand.labels.size > 0:
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
    if assignments.empty:
        return {
            str(method): pd.DataFrame(columns=list(TRACK_COLUMNS)) for method in methods
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
            methods=methods,
            config=config,
            device=device,
            compute_gt_metrics=bool(config.compute_gt_metrics),
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
            "Comma-separated methods to run: centroid,feature_mean,centroid_feature. "
            "Defaults to all three."
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
