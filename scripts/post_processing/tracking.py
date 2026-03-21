"""Multi-object tracking across timepoints using spatialDINO features.

Links segmented objects between consecutive timepoints by combining
spatial proximity (anisotropic centroid distance), shape overlap
(centroid-aligned Dice), and per-channel feature correlation.  The
assignment logic proceeds in three stages:

1. Distance pre-filter: immediately assigns very close centroids.
2. Feature voting: iterates over decreasing vote thresholds, assigning
   candidates that win a majority of per-feature correlation votes.
3. Global closest: assigns remaining objects by nearest distance.

The final tracks are exported as a CSV compatible with downstream
analysis tools.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

DEFAULT_FEATURE_BATCH_BYTES = 256 * 1024 * 1024
DEFAULT_N_PATCH_FEATURES = 384
DEFAULT_RAW_FILENAME = "volume_unnorm.tif"


@dataclass
class TimepointPaths:
    """Resolved file paths for a single timepoint."""

    name: str
    folder: Path
    raw_path: Path
    segmentation_path: Path
    lr_feats_path: Path


@dataclass
class LabelGeometry:
    """Spatial geometry for a single segmented object (label).

    Stores voxel coordinates, bounding box, centroid, and a cropped
    binary mask for efficient overlap computation.
    """

    label_id: int
    vox_coords: np.ndarray
    centroid: np.ndarray
    volume: int
    bbox_start: np.ndarray
    bbox_stop: np.ndarray
    local_mask: np.ndarray
    local_centroid: np.ndarray


@dataclass
class PreparedTimepoint:
    """Fully loaded timepoint with geometries and intensity statistics."""

    index: int
    paths: TimepointPaths
    shape_yxz: tuple[int, int, int]
    raw_z_size: int
    geometries: dict[int, LabelGeometry]
    label_ids: np.ndarray
    centroids: np.ndarray
    volumes: np.ndarray
    amplitudes: np.ndarray


@dataclass
class CandidateVote:
    """Accumulated feature-vote tally for one candidate-reference pair."""

    candidate_label: int
    wins: int = 0
    features: list[int] = field(default_factory=list)
    distance: float | None = None
    dice: float | None = None


@dataclass
class RefSummary:
    """Summary of all candidate votes for one reference label."""

    ref_label: int
    candidates: list[CandidateVote]


@dataclass
class AssignmentRecord:
    """Record of a single reference-to-candidate assignment with its method."""

    ref_label: int
    candidate_label: int
    method: str
    distance: float | None = None
    wins: int | None = None


@dataclass
class PairResult:
    """Complete matching result for one consecutive timepoint pair."""

    t_ref: int
    t_cand: int
    initial_summary: list[RefSummary]
    summary_current: list[RefSummary]
    summary_history: list[list[RefSummary]]
    assignments: list[AssignmentRecord]
    final_assignment: dict[int, int]
    summary_distance_prefilter: list[dict[str, float]]


@dataclass
class TrackPoint:
    """Single observation in a track: one object at one timepoint."""

    timepoint: int
    label_id: int
    centroid: tuple[float, float, float]
    volume: int
    amplitude: float


@dataclass
class Track:
    """A linked sequence of observations spanning multiple timepoints."""

    track_id: int
    points: list[TrackPoint]
    start_time: int
    length: int


@dataclass
class RefCandidateMetrics:
    """Spatial, shape, and feature metrics for one reference vs. its candidates."""

    ref_label: int
    candidate_ids: np.ndarray
    distances: np.ndarray
    dice: np.ndarray
    overlap_counts: np.ndarray
    corr: np.ndarray
    mse: np.ndarray


@dataclass(frozen=True)
class AxisInterpolation:
    """Precomputed trilinear interpolation indices and weights for one axis."""

    low: np.ndarray
    high: np.ndarray
    weight_low: np.ndarray
    weight_high: np.ndarray


@dataclass(frozen=True)
class TrackingParams:
    """User-configurable parameters controlling the tracking pipeline."""

    max_distance_xy: float
    max_distance_z: float
    z_distance_weight: float
    min_distance_to_remove_cand: float
    vote_thresholds: tuple[int, ...] | None
    dice_threshold: float
    corr_threshold: float
    invert_z: bool


class PipelineError(RuntimeError):
    """Raised when the tracking pipeline encounters an unrecoverable error."""

    pass


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tracking pipeline.

    Returns:
        Namespace with paths, distance thresholds, and tracking parameters.
    """
    parser = argparse.ArgumentParser(
        description="Track segmented objects across timepoints."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Folder containing per-timepoint subfolders.",
    )
    parser.add_argument(
        "--segmentation-path",
        required=True,
        help="Folder containing one segmentation mask per timepoint, named <timepoint>.tif.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Folder where tracks.csv will be written. Defaults to the input folder.",
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
        "--z-distance-weight",
        type=float,
        default=2.5,
        help="Anisotropy factor applied to Z in the centroid distance metric.",
    )
    parser.add_argument(
        "--min-distance-to-remove-cand",
        type=float,
        default=3.0,
        help="Immediate-assignment threshold from the original MATLAB logic.",
    )
    parser.add_argument(
        "--vote-thresholds",
        type=str,
        default="320,300,280,260",
        help="Comma-separated vote thresholds, e.g. '320,300,280,260'. Leave blank to use defaults.",
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=0.5,
        help="Minimum centroid-aligned Dice overlap required for feature voting.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.5,
        help="Minimum feature correlation required for a feature vote.",
    )
    invert_group = parser.add_mutually_exclusive_group()
    invert_group.add_argument(
        "--invert-z",
        dest="invert_z",
        action="store_true",
        help="Export z coordinates as z0 - z instead of keeping them as-is.",
    )
    invert_group.add_argument(
        "--no-invert-z",
        dest="invert_z",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(invert_z=False)
    return parser.parse_args()


def parse_thresholds(text: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated string of vote thresholds into a tuple.

    Args:
        text: Comma-separated integers, or ``None``/empty for default.

    Returns:
        Tuple of integer thresholds, or ``None`` if input is empty.
    """
    if text is None or text.strip() == "":
        return None
    thresholds: list[int] = []
    for token in text.split(","):
        value = token.strip()
        if not value:
            continue
        thresholds.append(int(value))
    return tuple(thresholds)


def natural_sort_key(value: str) -> list[Any]:
    """Generate a sort key that orders embedded numbers numerically.

    Args:
        value: String to generate the key for.

    Returns:
        List of alternating string/int segments for natural ordering.
    """
    parts = re.split(r"(\d+)", value)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_subfolders(
    input_path: Path, *, exclude_paths: Iterable[Path] = ()
) -> list[Path]:
    """List subdirectories of *input_path*, excluding specified paths.

    Args:
        input_path: Parent directory to scan.
        exclude_paths: Paths to exclude.

    Returns:
        Naturally sorted list of subdirectory paths.
    """
    excluded = {Path(os.path.realpath(path)) for path in exclude_paths}
    subfolders = [
        path
        for path in input_path.iterdir()
        if path.is_dir() and Path(os.path.realpath(path)) not in excluded
    ]
    subfolders.sort(key=lambda path: natural_sort_key(path.name))
    return subfolders


def read_tiff_volume(path: str | Path) -> np.ndarray:
    """Load a 3D TIFF volume and reorder axes from (Z, Y, X) to (Y, X, Z).

    Args:
        path: Path to the TIFF file.

    Returns:
        3D NumPy array with shape (Y, X, Z).

    Raises:
        PipelineError: If the volume is not 3D.
    """
    array = np.asarray(tifffile.imread(path))
    if array.ndim != 3:
        raise PipelineError(
            f"Expected a 3D TIFF volume, got shape {array.shape} for {path}."
        )
    return np.moveaxis(array, 0, -1)


def mask_path_for_timepoint(segmentation_path: Path, *, timepoint_name: str) -> Path:
    """Construct the expected segmentation mask path for a timepoint.

    Args:
        segmentation_path: Directory containing mask TIFFs.
        timepoint_name: Name of the timepoint subfolder.

    Returns:
        Path to ``<timepoint_name>.tif`` inside *segmentation_path*.
    """
    return segmentation_path / f"{timepoint_name}.tif"


def validate_subfolder(subfolder: Path, *, segmentation_path: Path) -> TimepointPaths:
    """Validate that a timepoint subfolder has all required files.

    Args:
        subfolder: Timepoint directory.
        segmentation_path: Directory containing segmentation masks.

    Returns:
        Validated ``TimepointPaths`` instance.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    raw_path = subfolder / DEFAULT_RAW_FILENAME
    segmentation_mask_path = mask_path_for_timepoint(
        segmentation_path, timepoint_name=subfolder.name
    )
    lr_feats_path = subfolder / "lr_feats.npy"
    if not raw_path.is_file():
        raise FileNotFoundError(f"{subfolder.name} is missing {DEFAULT_RAW_FILENAME}.")
    if not segmentation_mask_path.is_file():
        raise FileNotFoundError(
            f"Missing segmentation mask for {subfolder.name}: {segmentation_mask_path.name}."
        )
    if not lr_feats_path.is_file():
        raise FileNotFoundError(f"{subfolder.name} is missing lr_feats.npy.")
    return TimepointPaths(
        name=subfolder.name,
        folder=subfolder,
        raw_path=raw_path,
        segmentation_path=segmentation_mask_path,
        lr_feats_path=lr_feats_path,
    )


def discover_experiment(
    input_path: Path, *, segmentation_path: Path
) -> list[TimepointPaths]:
    """Discover and validate all timepoint subfolders in the experiment.

    Args:
        input_path: Root directory with timepoint subfolders.
        segmentation_path: Directory containing segmentation masks.

    Returns:
        List of validated ``TimepointPaths``.
    """
    discovered: list[TimepointPaths] = []
    for subfolder in list_subfolders(input_path, exclude_paths=(segmentation_path,)):
        discovered.append(
            validate_subfolder(subfolder, segmentation_path=segmentation_path)
        )
    return discovered


def extract_label_geometries(segmentation_yxz: np.ndarray) -> dict[int, LabelGeometry]:
    """Extract per-label geometry from a segmentation mask.

    Computes centroids, bounding boxes, and cropped binary masks for
    every non-zero label in the volume.

    Args:
        segmentation_yxz: Integer label volume of shape (Y, X, Z).

    Returns:
        Dictionary mapping label ID to its ``LabelGeometry``.
    """
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
    coords_all = np.column_stack(
        np.unravel_index(foreground_indices, segmentation_yxz.shape)
    ).astype(
        np.int32,
        copy=False,
    )

    geometries: dict[int, LabelGeometry] = {}
    for label_id, start, count in zip(
        unique_labels.tolist(), starts.tolist(), counts.tolist()
    ):
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


def compute_label_mean_intensities(
    segmentation_yxz: np.ndarray, raw_yxz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean raw intensity per foreground label.

    Args:
        segmentation_yxz: Integer label volume.
        raw_yxz: Raw intensity volume of the same shape.

    Returns:
        Tuple of (sorted label IDs, mean intensities per label).
    """
    flat_labels = segmentation_yxz.ravel()
    flat_raw = raw_yxz.ravel()
    foreground = flat_labels != 0
    if not np.any(foreground):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    labels = flat_labels[foreground].astype(np.int64, copy=False)
    raw_values = flat_raw[foreground].astype(np.float64, copy=False)
    unique_labels, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    sums = np.bincount(inverse, weights=raw_values, minlength=len(unique_labels))
    means = sums / np.maximum(counts, 1)
    return unique_labels.astype(np.int64, copy=False), means.astype(
        np.float32, copy=False
    )


def prepare_timepoint(index: int, paths: TimepointPaths) -> PreparedTimepoint:
    """Load raw volume and segmentation, extract geometries and amplitudes.

    Args:
        index: 1-based timepoint index.
        paths: File paths for this timepoint.

    Returns:
        Fully populated ``PreparedTimepoint``.
    """
    raw_yxz = read_tiff_volume(paths.raw_path)
    segmentation_yxz = read_tiff_volume(paths.segmentation_path)
    if raw_yxz.shape != segmentation_yxz.shape:
        raise PipelineError(
            f"Shape mismatch in {paths.name}: "
            f"{DEFAULT_RAW_FILENAME}={raw_yxz.shape}, {paths.segmentation_path.name}={segmentation_yxz.shape}."
        )

    geometries = extract_label_geometries(segmentation_yxz)
    label_ids = np.array(sorted(geometries.keys()), dtype=np.int64)
    if label_ids.size == 0:
        centroids = np.empty((0, 3), dtype=np.float64)
        volumes = np.empty((0,), dtype=np.int64)
        amplitudes = np.empty((0,), dtype=np.float32)
    else:
        centroids = np.vstack([
            geometries[int(label_id)].centroid for label_id in label_ids
        ])
        volumes = np.array(
            [geometries[int(label_id)].volume for label_id in label_ids], dtype=np.int64
        )
        amplitude_labels, amplitudes = compute_label_mean_intensities(
            segmentation_yxz, raw_yxz
        )
        if not np.array_equal(amplitude_labels, label_ids):
            raise PipelineError(f"Label intensity extraction mismatch in {paths.name}.")

    return PreparedTimepoint(
        index=index,
        paths=paths,
        shape_yxz=tuple(int(dim) for dim in segmentation_yxz.shape),
        raw_z_size=int(raw_yxz.shape[2]),
        geometries=geometries,
        label_ids=label_ids,
        centroids=centroids,
        volumes=volumes,
        amplitudes=amplitudes,
    )


def anisotropic_distance(a: np.ndarray, b: np.ndarray, zratio: float) -> float:
    """Compute anisotropic Euclidean distance between two 3D centroids.

    The Z component is weighted by *zratio* to account for anisotropic
    voxel spacing.

    Args:
        a: First centroid (Y, X, Z).
        b: Second centroid (Y, X, Z).
        zratio: Weight factor applied to the squared Z difference.

    Returns:
        Weighted Euclidean distance.
    """
    dyx = a[:2] - b[:2]
    dz = a[2] - b[2]
    return float(np.sqrt(np.sum(dyx**2) + zratio * (dz**2)))


def find_spatial_candidates(
    ref_tp: PreparedTimepoint,
    cand_tp: PreparedTimepoint,
    *,
    spatial_radius: tuple[float, float, float],
    zratio: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Find candidate labels within a spatial bounding box of each reference.

    Args:
        ref_tp: Reference timepoint.
        cand_tp: Candidate (next) timepoint.
        spatial_radius: (Y, X, Z) radius for the bounding-box filter.
        zratio: Anisotropy weight for the distance metric.

    Returns:
        Dict mapping each reference label to (candidate_ids, distances),
        sorted by ascending distance.
    """
    if cand_tp.label_ids.size == 0:
        return {
            int(ref_id): (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
            for ref_id in ref_tp.label_ids.tolist()
        }

    radius = np.asarray(spatial_radius, dtype=np.float64)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ref_id in ref_tp.label_ids.tolist():
        ref_centroid = ref_tp.geometries[int(ref_id)].centroid
        in_box = np.all(
            np.abs(cand_tp.centroids - ref_centroid[None, :]) <= radius[None, :],
            axis=1,
        )
        cand_ids = cand_tp.label_ids[in_box]
        distances = np.array(
            [
                anisotropic_distance(
                    ref_centroid, cand_tp.geometries[int(cand_id)].centroid, zratio
                )
                for cand_id in cand_ids.tolist()
            ],
            dtype=np.float32,
        )
        order = np.argsort(distances, kind="mergesort")
        out[int(ref_id)] = (
            cand_ids[order].astype(np.int64, copy=False),
            distances[order].astype(np.float32, copy=False),
        )
    return out


def compute_alignment_overlap(
    ref_geom: LabelGeometry,
    cand_geom: LabelGeometry,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """Compute centroid-aligned Dice overlap between two label masks.

    Translates the candidate mask so its centroid aligns with the
    reference centroid, then computes the overlapping voxels.

    Args:
        ref_geom: Geometry of the reference label.
        cand_geom: Geometry of the candidate label.

    Returns:
        Tuple of (dice_coefficient, ref_global_coords, cand_global_coords,
        intersection_count).
    """
    ref_shape = np.asarray(ref_geom.local_mask.shape, dtype=np.int32)
    cand_shape = np.asarray(cand_geom.local_mask.shape, dtype=np.int32)
    if np.any(ref_shape < 2) or np.any(cand_shape < 2):
        empty = np.empty((0, 3), dtype=np.int32)
        return 0.0, empty, empty, 0

    offset = np.rint(ref_geom.local_centroid - cand_geom.local_centroid).astype(
        np.int32
    )
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
        slice(
            int(overlap_start[axis] - ref_start[axis]),
            int(overlap_end[axis] - ref_start[axis]),
        )
        for axis in range(3)
    )
    cand_slices = tuple(
        slice(
            int(overlap_start[axis] - cand_start[axis]),
            int(overlap_end[axis] - cand_start[axis]),
        )
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
    """Build precomputed linear interpolation tables for one axis.

    Args:
        input_size: Number of elements in the low-resolution axis.
        output_size: Number of elements in the high-resolution axis.

    Returns:
        ``AxisInterpolation`` with index and weight arrays.
    """
    if input_size <= 0 or output_size <= 0:
        raise PipelineError("Interpolation sizes must be positive.")
    src = (
        (np.arange(output_size, dtype=np.float32) + 0.5) * (input_size / output_size)
    ) - 0.5
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
    """Build interpolation maps for all three axes (Y, X, Z).

    Args:
        input_shape_yxz: Low-resolution shape.
        output_shape_yxz: High-resolution shape.

    Returns:
        Tuple of (Y, X, Z) ``AxisInterpolation`` objects.
    """
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
    """Choose how many feature channels to load per batch.

    Args:
        ref_lr_shape_zyx: Reference low-resolution volume shape.
        cand_lr_shape_zyx: Candidate low-resolution volume shape.
        n_features: Total number of feature channels.

    Returns:
        Number of channels per batch that fits within the memory budget.
    """
    max_voxels = max(int(np.prod(ref_lr_shape_zyx)), int(np.prod(cand_lr_shape_zyx)))
    bytes_per_channel_pair = max_voxels * np.dtype(np.float32).itemsize * 2
    return max(
        1,
        min(n_features, DEFAULT_FEATURE_BATCH_BYTES // max(1, bytes_per_channel_pair)),
    )


def load_feature_chunk_internal_yxz(
    lr_feats: np.ndarray,
    *,
    start: int,
    end: int,
) -> np.ndarray:
    """Load a slice of feature channels and reorder to (C, Y, X, Z).

    Args:
        lr_feats: Memory-mapped features of shape [Z, Y, X, C].
        start: First channel index (inclusive).
        end: Last channel index (exclusive).

    Returns:
        Array of shape [end-start, Y, X, Z].
    """
    chunk_zyxc = np.asarray(lr_feats[..., start:end], dtype=np.float32)
    if chunk_zyxc.ndim != 4:
        raise PipelineError(
            f"Expected lr_feats.npy to have shape [Z, Y, X, C], got {chunk_zyxc.shape}."
        )
    return np.moveaxis(np.transpose(chunk_zyxc, (1, 2, 0, 3)), -1, 0)


def sample_feature_chunk_at_internal_coords(
    feature_chunk_cyxz: np.ndarray,
    axis_maps: tuple[AxisInterpolation, AxisInterpolation, AxisInterpolation],
    coords_yxz: np.ndarray,
) -> np.ndarray:
    """Sample feature values at high-res coordinates using trilinear interpolation.

    Uses precomputed axis maps to perform efficient trilinear lookup
    into the low-resolution feature volume at high-resolution voxel
    coordinates.

    Args:
        feature_chunk_cyxz: Feature channels of shape [C, Y, X, Z].
        axis_maps: Precomputed (Y, X, Z) interpolation tables.
        coords_yxz: High-resolution coordinates, shape [N, 3].

    Returns:
        Sampled values of shape [C, N].
    """
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
    """Compute Pearson correlation for each pair of corresponding rows.

    Args:
        a: Array of shape [N, M].
        b: Array of shape [N, M].

    Returns:
        1D array of length N with correlation coefficients (NaN where undefined).
    """
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


def compute_pair_metrics(
    ref_tp: PreparedTimepoint,
    cand_tp: PreparedTimepoint,
    *,
    n_features: int,
    spatial_radius: tuple[float, float, float],
    zratio: float,
) -> dict[int, RefCandidateMetrics]:
    """Compute all matching metrics between a reference and candidate timepoint.

    Finds spatial candidates, computes centroid-aligned Dice overlap,
    then batches through feature channels to compute per-channel
    correlation and MSE at overlapping voxels.

    Args:
        ref_tp: Reference (current) timepoint.
        cand_tp: Candidate (next) timepoint.
        n_features: Number of patch feature channels to compare.
        spatial_radius: Bounding-box search radius (Y, X, Z).
        zratio: Anisotropy weight for the distance metric.

    Returns:
        Dict mapping each reference label to its ``RefCandidateMetrics``.
    """
    spatial = find_spatial_candidates(
        ref_tp,
        cand_tp,
        spatial_radius=spatial_radius,
        zratio=zratio,
    )

    metrics_by_ref: dict[int, RefCandidateMetrics] = {}
    overlap_cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for ref_id in ref_tp.label_ids.tolist():
        candidate_ids, distances = spatial[int(ref_id)]
        candidate_count = len(candidate_ids)
        dice = np.full((candidate_count,), np.nan, dtype=np.float32)
        overlap_counts = np.zeros((candidate_count,), dtype=np.int32)
        overlaps: list[tuple[np.ndarray, np.ndarray]] = []

        ref_geom = ref_tp.geometries[int(ref_id)]
        for index, cand_id in enumerate(candidate_ids.tolist()):
            cand_geom = cand_tp.geometries[int(cand_id)]
            dice_value, ref_coords, cand_coords, overlap_count = (
                compute_alignment_overlap(ref_geom, cand_geom)
            )
            dice[index] = np.float32(dice_value)
            overlap_counts[index] = int(overlap_count)
            overlaps.append((ref_coords, cand_coords))

        metrics_by_ref[int(ref_id)] = RefCandidateMetrics(
            ref_label=int(ref_id),
            candidate_ids=candidate_ids,
            distances=distances.astype(np.float32, copy=False),
            dice=dice,
            overlap_counts=overlap_counts,
            corr=np.full((n_features, candidate_count), np.nan, dtype=np.float32),
            mse=np.full((n_features, candidate_count), np.nan, dtype=np.float32),
        )
        overlap_cache[int(ref_id)] = overlaps

    ref_lr_feats = np.load(ref_tp.paths.lr_feats_path, mmap_mode="r")
    cand_lr_feats = np.load(cand_tp.paths.lr_feats_path, mmap_mode="r")
    if ref_lr_feats.ndim != 4:
        raise PipelineError(
            f"{ref_tp.paths.lr_feats_path} must have shape [Z, Y, X, C]."
        )
    if cand_lr_feats.ndim != 4:
        raise PipelineError(
            f"{cand_tp.paths.lr_feats_path} must have shape [Z, Y, X, C]."
        )
    if ref_lr_feats.shape[-1] < n_features:
        raise PipelineError(
            f"{ref_tp.paths.lr_feats_path} only has {ref_lr_feats.shape[-1]} channels; tracking needs {n_features}."
        )
    if cand_lr_feats.shape[-1] < n_features:
        raise PipelineError(
            f"{cand_tp.paths.lr_feats_path} only has {cand_lr_feats.shape[-1]} channels; tracking needs {n_features}."
        )

    ref_input_shape_yxz = (
        int(ref_lr_feats.shape[1]),
        int(ref_lr_feats.shape[2]),
        int(ref_lr_feats.shape[0]),
    )
    cand_input_shape_yxz = (
        int(cand_lr_feats.shape[1]),
        int(cand_lr_feats.shape[2]),
        int(cand_lr_feats.shape[0]),
    )
    ref_axis_maps = build_axis_maps(ref_input_shape_yxz, ref_tp.shape_yxz)
    cand_axis_maps = build_axis_maps(cand_input_shape_yxz, cand_tp.shape_yxz)
    batch_size = choose_feature_batch_size(
        tuple(int(dim) for dim in ref_lr_feats.shape[:3]),
        tuple(int(dim) for dim in cand_lr_feats.shape[:3]),
        n_features=n_features,
    )

    for start in range(0, n_features, batch_size):
        end = min(n_features, start + batch_size)
        ref_chunk = load_feature_chunk_internal_yxz(ref_lr_feats, start=start, end=end)
        cand_chunk = load_feature_chunk_internal_yxz(
            cand_lr_feats, start=start, end=end
        )

        for ref_id, ref_metrics in metrics_by_ref.items():
            if ref_metrics.candidate_ids.size == 0:
                continue
            for candidate_index, _cand_id in enumerate(
                ref_metrics.candidate_ids.tolist()
            ):
                if ref_metrics.overlap_counts[candidate_index] <= 1:
                    continue

                ref_coords, cand_coords = overlap_cache[ref_id][candidate_index]
                if ref_coords.shape[0] <= 1:
                    continue

                ref_values = sample_feature_chunk_at_internal_coords(
                    ref_chunk, ref_axis_maps, ref_coords
                )
                cand_values = sample_feature_chunk_at_internal_coords(
                    cand_chunk, cand_axis_maps, cand_coords
                )

                diff = ref_values - cand_values
                ref_metrics.mse[start:end, candidate_index] = np.mean(
                    diff * diff,
                    axis=1,
                    dtype=np.float32,
                )
                ref_metrics.corr[start:end, candidate_index] = rowwise_correlation(
                    ref_values, cand_values
                )

    return metrics_by_ref


def build_summary_from_metrics(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    *,
    available_candidates: dict[int, set[int]] | None = None,
    dice_threshold: float,
    corr_threshold: float,
) -> list[RefSummary]:
    """Build vote summaries from metrics, applying Dice and correlation filters.

    For each feature channel, the candidate with the lowest MSE (among
    those passing both thresholds) receives a vote.  Candidates are then
    ranked by total wins.

    Args:
        metrics_by_ref: Per-reference metrics dict.
        available_candidates: Allowed candidate IDs per reference (``None`` = all).
        dice_threshold: Minimum Dice overlap to be eligible for a vote.
        corr_threshold: Minimum feature correlation to be eligible for a vote.

    Returns:
        List of ``RefSummary`` objects, one per reference label.
    """
    summaries: list[RefSummary] = []
    for ref_id in sorted(metrics_by_ref):
        metrics = metrics_by_ref[ref_id]
        if available_candidates is None:
            allowed = np.ones(metrics.candidate_ids.shape[0], dtype=bool)
        else:
            allowed_ids = available_candidates.get(ref_id, set())
            allowed = np.array([
                int(candidate_id) in allowed_ids
                for candidate_id in metrics.candidate_ids.tolist()
            ])

        candidates: dict[int, CandidateVote] = {}
        for candidate_id, distance, dice in zip(
            metrics.candidate_ids.tolist(),
            metrics.distances.tolist(),
            metrics.dice.tolist(),
        ):
            if available_candidates is not None and int(
                candidate_id
            ) not in available_candidates.get(ref_id, set()):
                continue
            candidates[int(candidate_id)] = CandidateVote(
                candidate_label=int(candidate_id),
                wins=0,
                features=[],
                distance=float(distance),
                dice=float(dice),
            )

        if np.any(allowed):
            for feature_index in range(metrics.corr.shape[0]):
                good = (
                    allowed
                    & np.isfinite(metrics.dice)
                    & (metrics.dice > dice_threshold)
                    & np.isfinite(metrics.corr[feature_index])
                    & (metrics.corr[feature_index] > corr_threshold)
                    & np.isfinite(metrics.mse[feature_index])
                )
                if not np.any(good):
                    continue
                good_indices = np.where(good)[0]
                chosen_index = int(
                    good_indices[np.argmin(metrics.mse[feature_index, good_indices])]
                )
                chosen_candidate = int(metrics.candidate_ids[chosen_index])
                candidates[chosen_candidate].wins += 1
                candidates[chosen_candidate].features.append(feature_index)

        ordered = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.wins,
                np.inf if candidate.distance is None else candidate.distance,
                candidate.candidate_label,
            ),
        )
        summaries.append(RefSummary(ref_label=ref_id, candidates=ordered))
    return summaries


def _remove_assigned_from_candidate_pool(
    candidate_pool: dict[int, set[int]],
    *,
    assigned_refs: Iterable[int],
    assigned_cands: Iterable[int],
) -> dict[int, set[int]]:
    """Return a copy of the candidate pool with assigned labels removed.

    Args:
        candidate_pool: Current pool mapping reference IDs to candidate sets.
        assigned_refs: Reference labels that have been assigned.
        assigned_cands: Candidate labels that have been assigned.

    Returns:
        Updated pool without the assigned labels.
    """
    assigned_refs_set = {int(value) for value in assigned_refs}
    assigned_cands_set = {int(value) for value in assigned_cands}
    out: dict[int, set[int]] = {}
    for ref_id, candidates in candidate_pool.items():
        if ref_id in assigned_refs_set:
            continue
        out[ref_id] = {
            int(candidate)
            for candidate in candidates
            if int(candidate) not in assigned_cands_set
        }
    return out


def run_assignment_logic(
    metrics_by_ref: dict[int, RefCandidateMetrics],
    *,
    min_distance_to_remove_cand: float,
    vote_thresholds: tuple[int, ...],
    dice_threshold: float,
    corr_threshold: float,
) -> tuple[
    list[AssignmentRecord],
    list[RefSummary],
    list[list[RefSummary]],
    list[dict[str, float]],
]:
    """Execute the three-stage assignment pipeline.

    1. Distance pre-filter: assigns pairs closer than *min_distance_to_remove_cand*.
    2. Vote thresholds: iteratively assigns unambiguous winners above
       each threshold (descending).
    3. Global closest: assigns remaining unmatched labels by distance.

    Args:
        metrics_by_ref: Per-reference matching metrics.
        min_distance_to_remove_cand: Immediate-assignment distance threshold.
        vote_thresholds: Descending sequence of vote thresholds.
        dice_threshold: Minimum Dice for voting eligibility.
        corr_threshold: Minimum correlation for voting eligibility.

    Returns:
        Tuple of (assignments, initial_summary, summary_history,
        distance_prefilter_info).
    """
    all_ref_ids = sorted(metrics_by_ref)
    candidate_pool: dict[int, set[int]] = {
        ref_id: set(metrics.candidate_ids.tolist())
        for ref_id, metrics in metrics_by_ref.items()
    }

    initial_summary = build_summary_from_metrics(
        metrics_by_ref,
        available_candidates=candidate_pool,
        dice_threshold=dice_threshold,
        corr_threshold=corr_threshold,
    )
    summary_history: list[list[RefSummary]] = [initial_summary]
    assignments: list[AssignmentRecord] = []
    summary_distance_prefilter: list[dict[str, float]] = []

    nearest_pairs: list[tuple[float, int, int]] = []
    for ref_id in all_ref_ids:
        metrics = metrics_by_ref[ref_id]
        if metrics.candidate_ids.size == 0:
            continue
        best_index = int(np.argmin(metrics.distances))
        min_distance = float(metrics.distances[best_index])
        if min_distance < min_distance_to_remove_cand:
            nearest_pairs.append((
                min_distance,
                ref_id,
                int(metrics.candidate_ids[best_index]),
            ))
        else:
            summary_distance_prefilter.append({
                "RefLabel": float(ref_id),
                "MinDistance": min_distance,
            })

    assigned_refs: set[int] = set()
    assigned_cands: set[int] = set()
    for distance, ref_id, cand_id in sorted(
        nearest_pairs, key=lambda item: (item[0], item[1], item[2])
    ):
        if ref_id in assigned_refs or cand_id in assigned_cands:
            continue
        assignments.append(
            AssignmentRecord(
                ref_label=ref_id,
                candidate_label=cand_id,
                method="distance_prefilter",
                distance=distance,
                wins=None,
            )
        )
        assigned_refs.add(ref_id)
        assigned_cands.add(cand_id)

    candidate_pool = _remove_assigned_from_candidate_pool(
        candidate_pool,
        assigned_refs=assigned_refs,
        assigned_cands=assigned_cands,
    )
    summary_current = build_summary_from_metrics(
        metrics_by_ref,
        available_candidates=candidate_pool,
        dice_threshold=dice_threshold,
        corr_threshold=corr_threshold,
    )
    summary_history.append(summary_current)

    used_vote_thresholds = tuple(
        sorted(
            {int(value) for value in vote_thresholds if int(value) > 0}, reverse=True
        )
    )
    for threshold in used_vote_thresholds:
        while True:
            if not summary_current:
                break

            top_candidates: list[tuple[int, int, int]] = []
            for item in summary_current:
                if not item.candidates:
                    continue
                top = item.candidates[0]
                if top.wins > threshold:
                    top_candidates.append((
                        item.ref_label,
                        top.candidate_label,
                        top.wins,
                    ))

            if not top_candidates:
                break

            counts = defaultdict(int)
            for _ref_id, cand_id, _wins in top_candidates:
                counts[cand_id] += 1

            accepted = [
                (ref_id, cand_id, wins)
                for ref_id, cand_id, wins in top_candidates
                if counts[cand_id] == 1
            ]
            if not accepted:
                break

            new_refs: set[int] = set()
            new_cands: set[int] = set()
            for ref_id, cand_id, wins in accepted:
                if ref_id in new_refs or cand_id in new_cands:
                    continue
                assignments.append(
                    AssignmentRecord(
                        ref_label=ref_id,
                        candidate_label=cand_id,
                        method=f"vote_threshold_{threshold}",
                        distance=None,
                        wins=wins,
                    )
                )
                new_refs.add(ref_id)
                new_cands.add(cand_id)

            if not new_refs:
                break

            candidate_pool = _remove_assigned_from_candidate_pool(
                candidate_pool,
                assigned_refs=new_refs,
                assigned_cands=new_cands,
            )
            summary_current = build_summary_from_metrics(
                metrics_by_ref,
                available_candidates=candidate_pool,
                dice_threshold=dice_threshold,
                corr_threshold=corr_threshold,
            )
            summary_history.append(summary_current)

    remaining_pairs: list[tuple[float, int, int]] = []
    for ref_id, remaining_candidates in candidate_pool.items():
        if not remaining_candidates:
            continue
        metrics = metrics_by_ref[ref_id]
        for candidate_index, candidate_id in enumerate(metrics.candidate_ids.tolist()):
            if int(candidate_id) in remaining_candidates:
                remaining_pairs.append((
                    float(metrics.distances[candidate_index]),
                    ref_id,
                    int(candidate_id),
                ))

    used_refs = {assignment.ref_label for assignment in assignments}
    used_cands = {assignment.candidate_label for assignment in assignments}
    for distance, ref_id, cand_id in sorted(
        remaining_pairs, key=lambda item: (item[0], item[1], item[2])
    ):
        if ref_id in used_refs or cand_id in used_cands:
            continue
        assignments.append(
            AssignmentRecord(
                ref_label=ref_id,
                candidate_label=cand_id,
                method="global_closest",
                distance=distance,
                wins=None,
            )
        )
        used_refs.add(ref_id)
        used_cands.add(cand_id)

    candidate_pool = _remove_assigned_from_candidate_pool(
        candidate_pool,
        assigned_refs=used_refs,
        assigned_cands=used_cands,
    )
    summary_current = build_summary_from_metrics(
        metrics_by_ref,
        available_candidates=candidate_pool,
        dice_threshold=dice_threshold,
        corr_threshold=corr_threshold,
    )
    summary_history.append(summary_current)

    assignments.sort(
        key=lambda assignment: (assignment.ref_label, assignment.candidate_label)
    )
    return assignments, initial_summary, summary_history, summary_distance_prefilter


def label_row_index(timepoint: PreparedTimepoint, label_id: int) -> int:
    """Find the row index of *label_id* in the timepoint's sorted label array.

    Args:
        timepoint: Prepared timepoint.
        label_id: Label to look up.

    Returns:
        Index into ``timepoint.label_ids``.

    Raises:
        PipelineError: If the label is not found.
    """
    row_index = int(np.searchsorted(timepoint.label_ids, label_id))
    if (
        row_index >= timepoint.label_ids.size
        or int(timepoint.label_ids[row_index]) != label_id
    ):
        raise PipelineError(f"Label {label_id} not found in {timepoint.paths.name}.")
    return row_index


def build_tracks(
    prepared_timepoints: list[PreparedTimepoint], pair_results: list[PairResult]
) -> list[Track]:
    """Assemble full tracks from pairwise assignment results.

    Chains consecutive-pair assignments into multi-timepoint tracks.
    Labels that appear only in the first or last timepoint without a
    match are started as single-point tracks.

    Args:
        prepared_timepoints: All prepared timepoints in order.
        pair_results: Pairwise assignment results for consecutive pairs.

    Returns:
        List of ``Track`` objects sorted by track ID.
    """
    if not prepared_timepoints:
        return []

    tracks: list[Track] = []
    active: dict[int, int] = {}

    def build_track_point(timepoint_idx: int, label_id: int) -> TrackPoint:
        prepared = prepared_timepoints[timepoint_idx - 1]
        row_index = label_row_index(prepared, label_id)
        geometry = prepared.geometries[int(label_id)]
        return TrackPoint(
            timepoint=timepoint_idx,
            label_id=int(label_id),
            centroid=tuple(float(value) for value in geometry.centroid.tolist()),
            volume=int(geometry.volume),
            amplitude=float(prepared.amplitudes[row_index]),
        )

    def start_track(timepoint_idx: int, label_id: int) -> int:
        track_id = len(tracks) + 1
        tracks.append(
            Track(
                track_id=track_id,
                points=[build_track_point(timepoint_idx, label_id)],
                start_time=timepoint_idx,
                length=1,
            )
        )
        return track_id

    def ensure_track_point(track_id: int, timepoint_idx: int, label_id: int) -> None:
        track = tracks[track_id - 1]
        if any(
            point.timepoint == timepoint_idx and point.label_id == label_id
            for point in track.points
        ):
            return
        track.points.append(build_track_point(timepoint_idx, label_id))

    if prepared_timepoints[0].label_ids.size > 0:
        for label_id in prepared_timepoints[0].label_ids.tolist():
            active[int(label_id)] = start_track(1, int(label_id))

    for pair_index, pair_result in enumerate(pair_results, start=1):
        current_tp = prepared_timepoints[pair_index - 1]
        next_tp = prepared_timepoints[pair_index]

        for label_id in current_tp.label_ids.tolist():
            if int(label_id) not in active:
                active[int(label_id)] = start_track(pair_index, int(label_id))

        next_active: dict[int, int] = {}
        for ref_label, cand_label in sorted(pair_result.final_assignment.items()):
            if int(ref_label) not in active:
                active[int(ref_label)] = start_track(pair_index, int(ref_label))
            track_id = active[int(ref_label)]
            ensure_track_point(track_id, pair_index, int(ref_label))
            ensure_track_point(track_id, pair_index + 1, int(cand_label))
            next_active[int(cand_label)] = track_id

        active = next_active
        if pair_index == len(pair_results):
            existing_last_frame = {
                point.label_id
                for track in tracks
                for point in track.points
                if point.timepoint == pair_index + 1
            }
            for label_id in next_tp.label_ids.tolist():
                if int(label_id) not in existing_last_frame:
                    start_track(pair_index + 1, int(label_id))

    for track in tracks:
        track.points.sort(key=lambda point: point.timepoint)
        track.start_time = min(point.timepoint for point in track.points)
        track.length = len(track.points)

    tracks.sort(key=lambda track: track.track_id)
    return tracks


def build_matlab_style_tracks(tracks: list[Track]) -> list[dict[str, Any]]:
    """Convert ``Track`` objects to MATLAB-compatible dictionaries.

    Args:
        tracks: List of tracks.

    Returns:
        List of dicts with keys: track_id, start, x, y, z, A.
    """
    matlab_tracks: list[dict[str, Any]] = []
    for track in tracks:
        if not track.points:
            continue
        ordered_points = sorted(track.points, key=lambda point: point.timepoint)
        matlab_tracks.append({
            "track_id": int(track.track_id),
            "start": int(track.start_time),
            "x": np.asarray(
                [point.centroid[1] for point in ordered_points], dtype=np.float64
            ),
            "y": np.asarray(
                [point.centroid[0] for point in ordered_points], dtype=np.float64
            ),
            "z": np.asarray(
                [point.centroid[2] for point in ordered_points], dtype=np.float64
            ),
            "A": np.asarray(
                [point.amplitude for point in ordered_points], dtype=np.float64
            ),
        })
    return matlab_tracks


def build_export_rows(
    matlab_tracks: Iterable[dict[str, Any]], *, z0: int, invert_z: bool
) -> list[dict[str, Any]]:
    """Flatten MATLAB-style tracks into per-frame rows for CSV export.

    Args:
        matlab_tracks: Tracks in MATLAB dict format.
        z0: Raw Z size for optional Z inversion.
        invert_z: If True, export Z as ``z0 - z`` instead of ``z``.

    Returns:
        List of row dicts suitable for ``csv.DictWriter``.
    """
    rows: list[dict[str, Any]] = []
    for track in matlab_tracks:
        x = np.asarray(track["x"], dtype=float)
        y = np.asarray(track["y"], dtype=float)
        z = np.asarray(track["z"], dtype=float)
        amplitudes = np.asarray(track["A"], dtype=float)
        start = int(track["start"])
        track_id = int(track["track_id"])

        if np.isnan(amplitudes).any():
            continue

        length = min(int(amplitudes.size), int(x.size), int(y.size), int(z.size))
        if length == 0:
            continue

        for frame_index in range(length):
            rows.append({
                "track_id": track_id,
                "start": start,
                "t": frame_index + 1,
                "x": float(x[frame_index]),
                "y": float(y[frame_index]),
                "z": float(z0 - z[frame_index]) if invert_z else float(z[frame_index]),
                "A": float(amplitudes[frame_index]),
                "track_length": length,
            })
    return rows


def save_tracks_csv(rows: Iterable[dict[str, Any]], *, output_path: Path) -> None:
    """Write track rows to a CSV file.

    Args:
        rows: Iterable of row dicts with standard track fields.
        output_path: Destination CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["track_id", "start", "t", "x", "y", "z", "A", "track_length"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def default_vote_thresholds(n_features: int) -> tuple[int, ...]:
    """Return default vote thresholds appropriate for the feature count.

    Args:
        n_features: Number of patch feature channels.

    Returns:
        Tuple of descending integer thresholds.
    """
    thresholds = tuple(
        threshold for threshold in (400, 320, 300, 280, 260) if threshold <= n_features
    )
    if thresholds:
        return thresholds
    return (max(1, int(round(0.8 * n_features))),)


def validate_timepoint_shapes(
    prepared_timepoints: list[PreparedTimepoint], *, require_same_raw_z_size: bool
) -> int:
    """Verify that all timepoints share the same volume shape.

    Args:
        prepared_timepoints: All prepared timepoints.
        require_same_raw_z_size: If True, also check raw Z sizes match.

    Returns:
        The common raw Z size.

    Raises:
        PipelineError: If shapes are inconsistent.
    """
    if not prepared_timepoints:
        raise PipelineError("No prepared timepoints available.")

    expected_shape = prepared_timepoints[0].shape_yxz
    expected_z0 = prepared_timepoints[0].raw_z_size
    for prepared in prepared_timepoints[1:]:
        if prepared.shape_yxz != expected_shape:
            raise PipelineError(
                "Tracking requires all timepoints to share the same volume shape. "
                f"Got {expected_shape} and {prepared.shape_yxz}."
            )
        if require_same_raw_z_size and prepared.raw_z_size != expected_z0:
            raise PipelineError(
                "Tracking requires all timepoints to share the same raw Z size for the final z inversion. "
                f"Got {expected_z0} and {prepared.raw_z_size}."
            )
    return expected_z0


def run_tracking(
    input_path: Path,
    *,
    segmentation_path: Path,
    output_path: Path | None = None,
    params: TrackingParams,
) -> Path:
    """Run the full tracking pipeline and save results as CSV.

    Discovers timepoints, prepares geometries, computes pairwise metrics
    and assignments, builds tracks, and exports to ``tracks.csv``.

    Args:
        input_path: Root directory with timepoint subfolders.
        segmentation_path: Directory with per-timepoint segmentation masks.
        output_path: Destination directory (defaults to *input_path*).
        params: Tracking hyperparameters.

    Returns:
        Path to the saved ``tracks.csv`` file.
    """
    segmentation_path = segmentation_path.expanduser().resolve()
    if not segmentation_path.is_dir():
        raise FileNotFoundError(
            f"Segmentation folder does not exist or is not a directory: {segmentation_path}"
        )
    output_dir = (
        input_path if output_path is None else output_path.expanduser().resolve()
    )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileNotFoundError(
            f"Output folder exists but is not a directory: {output_dir}"
        )

    discovered = discover_experiment(input_path, segmentation_path=segmentation_path)
    if len(discovered) < 2:
        raise PipelineError("Tracking requires at least 2 subfolders/timepoints.")

    prepared_timepoints: list[PreparedTimepoint] = []
    print(f"[tracking] Found {len(discovered)} subfolders", flush=True)
    for index, timepoint_paths in enumerate(discovered, start=1):
        print(
            f"[tracking] Processing {timepoint_paths.name} ({index}/{len(discovered)})",
            flush=True,
        )
        prepared_timepoints.append(prepare_timepoint(index, timepoint_paths))
        print(f"[tracking] Completed {timepoint_paths.name}", flush=True)

    z0 = validate_timepoint_shapes(
        prepared_timepoints, require_same_raw_z_size=bool(params.invert_z)
    )
    spatial_radius = (
        float(params.max_distance_xy),
        float(params.max_distance_xy),
        float(params.max_distance_z),
    )
    vote_thresholds = (
        params.vote_thresholds
        if params.vote_thresholds is not None
        else default_vote_thresholds(DEFAULT_N_PATCH_FEATURES)
    )

    pair_results: list[PairResult] = []
    for pair_index in range(len(prepared_timepoints) - 1):
        ref_tp = prepared_timepoints[pair_index]
        cand_tp = prepared_timepoints[pair_index + 1]
        print(
            f"[tracking] Matching {ref_tp.paths.name} -> {cand_tp.paths.name} ({pair_index + 1}/{len(prepared_timepoints) - 1})",
            flush=True,
        )

        metrics_by_ref = compute_pair_metrics(
            ref_tp,
            cand_tp,
            n_features=DEFAULT_N_PATCH_FEATURES,
            spatial_radius=spatial_radius,
            zratio=float(params.z_distance_weight),
        )
        assignments, initial_summary, summary_history, summary_distance_prefilter = (
            run_assignment_logic(
                metrics_by_ref,
                min_distance_to_remove_cand=float(params.min_distance_to_remove_cand),
                vote_thresholds=vote_thresholds,
                dice_threshold=float(params.dice_threshold),
                corr_threshold=float(params.corr_threshold),
            )
        )
        pair_results.append(
            PairResult(
                t_ref=ref_tp.index,
                t_cand=cand_tp.index,
                initial_summary=initial_summary,
                summary_current=summary_history[-1] if summary_history else [],
                summary_history=summary_history,
                assignments=assignments,
                final_assignment={
                    assignment.ref_label: assignment.candidate_label
                    for assignment in assignments
                },
                summary_distance_prefilter=summary_distance_prefilter,
            )
        )
        print(
            f"[tracking] Matched {ref_tp.paths.name} -> {cand_tp.paths.name} ({pair_index + 1}/{len(prepared_timepoints) - 1})",
            flush=True,
        )

    tracks = build_tracks(prepared_timepoints, pair_results)
    matlab_tracks = build_matlab_style_tracks(tracks)
    output_rows = build_export_rows(
        matlab_tracks, z0=z0, invert_z=bool(params.invert_z)
    )
    tracks_csv_path = output_dir / "tracks.csv"
    save_tracks_csv(output_rows, output_path=tracks_csv_path)
    print(f"[tracking] Saved tracks to {tracks_csv_path}", flush=True)
    print("[tracking] Done", flush=True)
    return tracks_csv_path


def main() -> None:
    """Entry point: validate CLI arguments and run the tracking pipeline."""
    args = parse_args()
    if args.max_distance_xy <= 0:
        raise ValueError("Max XY distance must be greater than 0.")
    if args.max_distance_z <= 0:
        raise ValueError("Max Z distance must be greater than 0.")
    if args.z_distance_weight <= 0:
        raise ValueError("Z distance weight must be greater than 0.")
    if args.min_distance_to_remove_cand < 0:
        raise ValueError("Immediate assignment distance must be nonnegative.")
    if args.dice_threshold < 0 or args.dice_threshold > 1:
        raise ValueError("Dice threshold must be between 0 and 1.")
    if args.corr_threshold < -1 or args.corr_threshold > 1:
        raise ValueError("Correlation threshold must be between -1 and 1.")

    input_path = Path(args.input_path).expanduser().resolve()
    segmentation_path = Path(args.segmentation_path).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist or is not a directory: {input_path}"
        )

    run_tracking(
        input_path,
        segmentation_path=segmentation_path,
        output_path=Path(args.output_path).expanduser().resolve()
        if args.output_path
        else None,
        params=TrackingParams(
            max_distance_xy=float(args.max_distance_xy),
            max_distance_z=float(args.max_distance_z),
            z_distance_weight=float(args.z_distance_weight),
            min_distance_to_remove_cand=float(args.min_distance_to_remove_cand),
            vote_thresholds=parse_thresholds(args.vote_thresholds),
            dice_threshold=float(args.dice_threshold),
            corr_threshold=float(args.corr_threshold),
            invert_z=bool(args.invert_z),
        ),
    )


if __name__ == "__main__":
    main()
