from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

DEFAULT_FEATURE_BATCH_BYTES = 256 * 1024 * 1024


class PipelineError(RuntimeError):
    pass


@dataclass
class LabelGeometry:
    label_id: int
    vox_coords: np.ndarray
    centroid: np.ndarray
    volume: int
    bbox_start: np.ndarray
    bbox_stop: np.ndarray
    local_mask: np.ndarray
    local_centroid: np.ndarray


@dataclass(frozen=True)
class AxisInterpolation:
    low: np.ndarray
    high: np.ndarray
    weight_low: np.ndarray
    weight_high: np.ndarray


def read_tiff_volume(path: str | Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise PipelineError(f"Expected a 3D TIFF volume, got shape {array.shape} for {path}.")
    return np.moveaxis(array, 0, -1)


def extract_label_geometries(segmentation_yxz: np.ndarray) -> dict[int, LabelGeometry]:
    flat = segmentation_yxz.ravel()
    foreground_indices = np.flatnonzero(flat != 0)
    if foreground_indices.size == 0:
        return {}

    labels = flat[foreground_indices].astype(np.int64, copy=False)
    order = np.argsort(labels, kind="mergesort")
    foreground_indices = foreground_indices[order]
    labels = labels[order]

    unique_labels, starts = np.unique(labels, return_index=True)
    counts = np.diff(np.r_[starts, len(labels)])
    coords_all = np.column_stack(np.unravel_index(foreground_indices, segmentation_yxz.shape)).astype(
        np.int32,
        copy=False,
    )

    geometries: dict[int, LabelGeometry] = {}
    for label_id, start, count in zip(unique_labels.tolist(), starts.tolist(), counts.tolist()):
        coords = coords_all[start : start + count]
        centroid = coords.mean(axis=0, dtype=np.float64)
        bbox_start = coords.min(axis=0)
        bbox_stop = coords.max(axis=0) + 1
        local_shape = tuple((bbox_stop - bbox_start).tolist())
        local_coords = coords - bbox_start

        local_mask = np.zeros(local_shape, dtype=bool)
        local_mask[tuple(local_coords.T)] = True

        geometries[int(label_id)] = LabelGeometry(
            label_id=int(label_id),
            vox_coords=coords,
            centroid=centroid,
            volume=int(coords.shape[0]),
            bbox_start=bbox_start.astype(np.int32, copy=False),
            bbox_stop=bbox_stop.astype(np.int32, copy=False),
            local_mask=local_mask,
            local_centroid=centroid - bbox_start,
        )

    return geometries


def compute_label_mean_intensities(segmentation_yxz: np.ndarray, raw_yxz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_labels = segmentation_yxz.ravel()
    flat_raw = raw_yxz.ravel()
    foreground = flat_labels != 0
    if not np.any(foreground):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    labels = flat_labels[foreground].astype(np.int64, copy=False)
    raw_values = flat_raw[foreground].astype(np.float64, copy=False)
    unique_labels, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=raw_values, minlength=len(unique_labels))
    means = sums / np.maximum(counts, 1)
    return unique_labels.astype(np.int64, copy=False), means.astype(np.float32, copy=False)


def anisotropic_distance(a: np.ndarray, b: np.ndarray, zratio: float) -> float:
    dyx = a[:2] - b[:2]
    dz = zratio * (a[2] - b[2])
    return float(np.sqrt(np.sum(dyx**2) + dz**2))


def compute_alignment_overlap(
    ref_geom: LabelGeometry,
    cand_geom: LabelGeometry,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    ref_shape = np.asarray(ref_geom.local_mask.shape, dtype=np.int32)
    cand_shape = np.asarray(cand_geom.local_mask.shape, dtype=np.int32)
    if np.any(ref_shape < 2) or np.any(cand_shape < 2):
        empty = np.empty((0, 3), dtype=np.int32)
        return 0.0, empty, empty, 0

    offset = np.rint(ref_geom.local_centroid - cand_geom.local_centroid).astype(np.int32)
    ref_start = np.maximum(0, -offset)
    cand_start = np.maximum(0, offset)
    ref_end = ref_start + ref_shape
    cand_end = cand_start + cand_shape

    overlap_start = np.maximum(ref_start, cand_start)
    overlap_end = np.minimum(ref_end, cand_end)
    if np.any(overlap_end <= overlap_start):
        empty = np.empty((0, 3), dtype=np.int32)
        return 0.0, empty, empty, 0

    ref_slices = tuple(
        slice(int(overlap_start[axis] - ref_start[axis]), int(overlap_end[axis] - ref_start[axis]))
        for axis in range(3)
    )
    cand_slices = tuple(
        slice(int(overlap_start[axis] - cand_start[axis]), int(overlap_end[axis] - cand_start[axis]))
        for axis in range(3)
    )

    ref_region = ref_geom.local_mask[ref_slices]
    cand_region = cand_geom.local_mask[cand_slices]
    overlap_mask = ref_region & cand_region

    intersection = int(overlap_mask.sum())
    union_count = ref_geom.volume + cand_geom.volume
    dice = 0.0 if union_count == 0 else float((2.0 * intersection) / union_count)
    if intersection <= 1:
        empty = np.empty((0, 3), dtype=np.int32)
        return dice, empty, empty, intersection

    overlap_local = np.argwhere(overlap_mask).astype(np.int32, copy=False)
    ref_offsets = np.array([part.start for part in ref_slices], dtype=np.int32)
    cand_offsets = np.array([part.start for part in cand_slices], dtype=np.int32)
    ref_global = overlap_local + ref_offsets[None, :] + ref_geom.bbox_start[None, :]
    cand_global = overlap_local + cand_offsets[None, :] + cand_geom.bbox_start[None, :]
    return (
        dice,
        ref_global.astype(np.int32, copy=False),
        cand_global.astype(np.int32, copy=False),
        intersection,
    )


def build_axis_interpolation(input_size: int, output_size: int) -> AxisInterpolation:
    if input_size <= 0 or output_size <= 0:
        raise PipelineError("Interpolation sizes must be positive.")
    src = ((np.arange(output_size, dtype=np.float32) + 0.5) * (input_size / output_size)) - 0.5
    low = np.floor(src).astype(np.int32)
    high = low + 1
    weight_high = src - low
    weight_low = 1.0 - weight_high
    low = np.clip(low, 0, input_size - 1)
    high = np.clip(high, 0, input_size - 1)
    return AxisInterpolation(
        low=low,
        high=high,
        weight_low=weight_low.astype(np.float32, copy=False),
        weight_high=weight_high.astype(np.float32, copy=False),
    )


def build_axis_maps(
    input_shape_yxz: tuple[int, int, int],
    output_shape_yxz: tuple[int, int, int],
) -> tuple[AxisInterpolation, AxisInterpolation, AxisInterpolation]:
    return (
        build_axis_interpolation(int(input_shape_yxz[0]), int(output_shape_yxz[0])),
        build_axis_interpolation(int(input_shape_yxz[1]), int(output_shape_yxz[1])),
        build_axis_interpolation(int(input_shape_yxz[2]), int(output_shape_yxz[2])),
    )


def choose_feature_batch_size(
    ref_lr_shape_zyx: tuple[int, int, int],
    cand_lr_shape_zyx: tuple[int, int, int],
    *,
    n_features: int,
) -> int:
    max_voxels = max(int(np.prod(ref_lr_shape_zyx)), int(np.prod(cand_lr_shape_zyx)))
    bytes_per_channel_pair = max_voxels * np.dtype(np.float32).itemsize * 2
    return max(1, min(n_features, DEFAULT_FEATURE_BATCH_BYTES // max(1, bytes_per_channel_pair)))


def load_feature_chunk_internal_yxz(
    lr_feats: np.ndarray,
    *,
    start: int,
    end: int,
) -> np.ndarray:
    chunk_zyxc = np.asarray(lr_feats[..., start:end], dtype=np.float32)
    if chunk_zyxc.ndim != 4:
        raise PipelineError(f"Expected lr_feats.npy to have shape [Z, Y, X, C], got {chunk_zyxc.shape}.")
    return np.moveaxis(np.transpose(chunk_zyxc, (1, 2, 0, 3)), -1, 0)


def sample_feature_chunk_at_internal_coords(
    feature_chunk_cyxz: np.ndarray,
    axis_maps: tuple[AxisInterpolation, AxisInterpolation, AxisInterpolation],
    coords_yxz: np.ndarray,
) -> np.ndarray:
    if coords_yxz.size == 0:
        return np.empty((feature_chunk_cyxz.shape[0], 0), dtype=np.float32)

    axis_y, axis_x, axis_z = axis_maps
    y_idx = coords_yxz[:, 0]
    x_idx = coords_yxz[:, 1]
    z_idx = coords_yxz[:, 2]

    y0 = axis_y.low[y_idx]
    y1 = axis_y.high[y_idx]
    x0 = axis_x.low[x_idx]
    x1 = axis_x.high[x_idx]
    z0 = axis_z.low[z_idx]
    z1 = axis_z.high[z_idx]

    wy0 = axis_y.weight_low[y_idx]
    wy1 = axis_y.weight_high[y_idx]
    wx0 = axis_x.weight_low[x_idx]
    wx1 = axis_x.weight_high[x_idx]
    wz0 = axis_z.weight_low[z_idx]
    wz1 = axis_z.weight_high[z_idx]

    w000 = (wy0 * wx0 * wz0)[None, :]
    w001 = (wy0 * wx0 * wz1)[None, :]
    w010 = (wy0 * wx1 * wz0)[None, :]
    w011 = (wy0 * wx1 * wz1)[None, :]
    w100 = (wy1 * wx0 * wz0)[None, :]
    w101 = (wy1 * wx0 * wz1)[None, :]
    w110 = (wy1 * wx1 * wz0)[None, :]
    w111 = (wy1 * wx1 * wz1)[None, :]

    return (
        (feature_chunk_cyxz[:, y0, x0, z0] * w000)
        + (feature_chunk_cyxz[:, y0, x0, z1] * w001)
        + (feature_chunk_cyxz[:, y0, x1, z0] * w010)
        + (feature_chunk_cyxz[:, y0, x1, z1] * w011)
        + (feature_chunk_cyxz[:, y1, x0, z0] * w100)
        + (feature_chunk_cyxz[:, y1, x0, z1] * w101)
        + (feature_chunk_cyxz[:, y1, x1, z0] * w110)
        + (feature_chunk_cyxz[:, y1, x1, z1] * w111)
    ).astype(np.float32, copy=False)


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    corr = np.full((a.shape[0],), np.nan, dtype=np.float32)
    if a.shape[1] <= 1:
        return corr

    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt(
        np.sum(a_centered * a_centered, axis=1, dtype=np.float64)
        * np.sum(b_centered * b_centered, axis=1, dtype=np.float64)
    )
    valid = denom > 0.0
    if np.any(valid):
        numer = np.sum(a_centered[valid] * b_centered[valid], axis=1, dtype=np.float64)
        corr[valid] = (numer / denom[valid]).astype(np.float32, copy=False)
    return corr
