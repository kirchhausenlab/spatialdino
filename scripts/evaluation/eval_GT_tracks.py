from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import shutil
import sys
import warnings
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.is_dir() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import numpy as np  # noqa: E402
import zarr  # noqa: E402
from numcodecs import Blosc  # noqa: E402
from spatialdino.inference.output_layout import discover_inference_timepoints, natural_sort_key  # noqa: E402
from spatialdino.tracking.metrics import (  # noqa: E402
    LabelGeometry,
    PipelineError,
    anisotropic_distance,
    build_axis_maps,
    choose_feature_batch_size,
    compute_alignment_overlap,
    compute_label_mean_intensities,
    extract_label_geometries,
    load_feature_chunk_internal_yxz,
    read_tiff_volume,
    rowwise_correlation,
    sample_feature_chunk_at_internal_coords,
)

DEFAULT_N_TRACKING_FEATURES = 384
DEFAULT_FEATURE_MEAN_SAMPLE_BYTES = 256 * 1024 * 1024
DEFAULT_PAIR_METRIC_SAMPLE_BYTES = 256 * 1024 * 1024
DEFAULT_FEATURE_CHUNK_CACHE_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_CANDIDATE_BATCH_ROWS = 100_000
CANDIDATE_OUTPUT_MODES = {"none", "summary", "full"}
CSV_NAN = "NaN"
EMPTY_COORDS = np.empty((0, 3), dtype=np.int32)


@dataclass(frozen=True)
class EvaluationParams:
    max_gap: int = 10
    num_features: str = "tracking"
    max_distance_xy: float = 20.0
    max_distance_z: float = 10.0
    z_distance_weight: float = 2.5
    dice_threshold: float = 0.5
    corr_threshold: float = 0.5
    no_compression: bool = False
    export_wide_csv: bool = False
    csv_float_decimals: int = 2
    candidate_output: str = "full"
    include_candidate_pairs: bool = True
    overwrite: bool = False
    candidate_batch_rows: int = DEFAULT_CANDIDATE_BATCH_ROWS
    feature_cache_bytes: int = DEFAULT_FEATURE_CHUNK_CACHE_BYTES
    num_workers: int = max(1, min(8, os.cpu_count() or 1))


@dataclass
class EvalTimepoint:
    index: int
    name: str
    raw_path: Path
    gt_segmentation_path: Path
    lr_feats_path: Path
    shape_yxz: tuple[int, int, int]
    raw_z_size: int
    geometries: dict[int, LabelGeometry]
    label_ids: np.ndarray
    centroids: np.ndarray
    foreground_coords: np.ndarray
    foreground_label_indices: np.ndarray
    foreground_voxel_counts: np.ndarray
    foreground_label_starts: np.ndarray
    foreground_index_flat: np.ndarray | None
    amplitudes: dict[int, float]
    lr_feats: np.ndarray
    axis_maps: Any


@dataclass
class CandidateEntry:
    ref_row_index: int
    true_row_index: int | None
    cand_row_index: int
    ref_track_id: int
    cand_track_id: int
    distance: float
    dice: float
    overlap_voxels: int
    ref_coords: np.ndarray
    cand_coords: np.ndarray


@dataclass
class TrueGapPairResult:
    gap: int
    entries: list[CandidateEntry]
    corr: np.ndarray
    mse: np.ndarray


@dataclass
class PairEvaluationPlan:
    gap: int
    ref_frame: int
    cand_frame: int
    true_entries: list[CandidateEntry]
    true_metric_indices: list[int]
    candidate_entries: list[CandidateEntry]
    candidate_metric_indices: list[int]
    metric_count: int
    ref_indices: np.ndarray
    cand_indices: np.ndarray
    group_indices: np.ndarray
    counts: np.ndarray
    corr: np.ndarray
    mse: np.ndarray


class TableSink:
    def __init__(
        self,
        path_stem: Path,
        *,
        fieldnames: Sequence[str],
        schema_fields: Sequence[tuple[str, str]],
        no_compression: bool,
    ) -> None:
        self.fieldnames = list(fieldnames)
        self.path: Path
        self._writer: Any = None
        self._csv_handle: Any = None
        self._csv_writer: csv.DictWriter[str] | None = None
        self._pa: Any = None
        self._schema: Any = None
        self._schema_fields = list(schema_fields)

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.path = path_stem.with_suffix(".csv")
            self._csv_handle = self.path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.fieldnames)
            self._csv_writer.writeheader()
            return

        self.path = path_stem.with_suffix(".parquet")
        self._pa = pa
        self._schema = pa.schema([pa.field(name, _pyarrow_type(pa, kind)) for name, kind in schema_fields])
        self._writer = pq.ParquetWriter(
            self.path,
            schema=self._schema,
            compression=None if no_compression else "lz4",
        )

    def write_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        data = {name: [row.get(name) for row in rows] for name in self.fieldnames}
        self.write_columns(data)

    def write_columns(self, columns: Mapping[str, Any]) -> None:
        if not self.fieldnames:
            return
        row_count = _column_length(columns[self.fieldnames[0]])
        if row_count == 0:
            return
        if self._csv_writer is not None:
            for row_index in range(row_count):
                self._csv_writer.writerow(
                    {
                        name: _csv_value(_column_value_at(columns[name], row_index))
                        for name in self.fieldnames
                    }
                )
            return

        arrays = [
            _pyarrow_array(self._pa, columns[name], kind)
            for name, kind in self._schema_fields
        ]
        table = self._pa.Table.from_arrays(arrays, schema=self._schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._csv_handle is not None:
            self._csv_handle.close()


def _pyarrow_type(pa: Any, kind: str) -> Any:
    if kind == "int":
        return pa.int64()
    if kind == "float":
        return pa.float64()
    if kind == "bool":
        return pa.bool_()
    if kind == "str":
        return pa.string()
    raise ValueError(f"Unknown table field kind: {kind}")


def _masked_column(column: Any) -> tuple[Any, np.ndarray | None]:
    if isinstance(column, tuple) and len(column) == 2:
        values, mask = column
        return values, np.asarray(mask, dtype=bool)
    return column, None


def _column_length(column: Any) -> int:
    values, _mask = _masked_column(column)
    return int(len(values))


def _column_value_at(column: Any, index: int) -> Any:
    values, mask = _masked_column(column)
    if mask is not None and bool(mask[index]):
        return None
    value = values[index]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pyarrow_array(pa: Any, column: Any, kind: str) -> Any:
    values, mask = _masked_column(column)
    arrow_type = _pyarrow_type(pa, kind)
    if kind == "float":
        array = np.asarray(values, dtype=np.float64)
        invalid = ~np.isfinite(array)
        if mask is not None:
            invalid |= mask
        clean = np.where(invalid, 0.0, array)
        return pa.array(clean, type=arrow_type, mask=invalid)
    if kind == "int":
        if mask is not None:
            return pa.array(np.asarray(values, dtype=np.int64), type=arrow_type, mask=mask)
        return pa.array(values, type=arrow_type)
    if kind == "bool":
        if mask is not None:
            return pa.array(np.asarray(values, dtype=bool), type=arrow_type, mask=mask)
        return pa.array(values, type=arrow_type)
    if kind == "str":
        return pa.array(values, type=arrow_type, mask=mask)
    raise ValueError(f"Unknown table field kind: {kind}")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return CSV_NAN
    return value


def concat_column_batches(fieldnames: Sequence[str], batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batches:
        return {}
    merged: dict[str, Any] = {}
    for name in fieldnames:
        chunks = [batch[name] for batch in batches]
        first_values, first_mask = _masked_column(chunks[0])
        if first_mask is not None:
            values = [np.asarray(_masked_column(chunk)[0]) for chunk in chunks]
            masks = [np.asarray(_masked_column(chunk)[1], dtype=bool) for chunk in chunks]
            merged[name] = (np.concatenate(values), np.concatenate(masks))
            continue
        if isinstance(first_values, np.ndarray):
            merged[name] = np.concatenate([np.asarray(chunk) for chunk in chunks])
            continue
        values_list: list[Any] = []
        for chunk in chunks:
            values_list.extend(chunk)
        merged[name] = values_list
    return merged


def flush_column_batches(
    sink: TableSink,
    fieldnames: Sequence[str],
    batches: list[Mapping[str, Any]],
) -> None:
    if not batches:
        return
    sink.write_columns(concat_column_batches(fieldnames, batches))
    batches.clear()


def nullable_int_column(values: Sequence[int | None]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.fromiter((value is None for value in values), dtype=bool, count=len(values))
    data = np.fromiter((0 if value is None else int(value) for value in values), dtype=np.int64, count=len(values))
    return data, mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tracking metrics on ground-truth persistent-ID segmentation volumes."
    )
    parser.add_argument(
        "--gt-segmentation-path",
        "--gt-segmentation-folder",
        required=True,
        dest="gt_segmentation_path",
        help="Folder containing one persistent-ID GT segmentation TIFF per timepoint.",
    )
    parser.add_argument(
        "--input-path",
        "--inference-output-path",
        required=True,
        dest="input_path",
        help="Inference output folder containing lr_feats/ and raw/.",
    )
    parser.add_argument(
        "--output-path",
        "--output-dir",
        required=True,
        dest="output_path",
        help="Output directory for manifest, tables, and Zarr arrays.",
    )
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument(
        "--num-features",
        default="tracking",
        help="Number of feature channels to evaluate, 'tracking' for 384, or 'all'.",
    )
    parser.add_argument("--max-distance-xy", type=float, default=20.0)
    parser.add_argument("--max-distance-z", type=float, default=10.0)
    parser.add_argument("--z-distance-weight", type=float, default=2.5)
    parser.add_argument("--dice-threshold", type=float, default=0.5)
    parser.add_argument("--corr-threshold", type=float, default=0.5)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--export-wide-csv", action="store_true")
    parser.add_argument("--csv-float-decimals", type=int, default=2)
    parser.add_argument(
        "--candidate-output",
        choices=sorted(CANDIDATE_OUTPUT_MODES),
        default="full",
        help="Candidate diagnostics to write: none, summary, or full pair+summary tables.",
    )
    parser.add_argument(
        "--skip-candidate-pairs",
        action="store_true",
        help="Legacy alias for --candidate-output none.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--candidate-batch-rows", type=int, default=DEFAULT_CANDIDATE_BATCH_ROWS)
    parser.add_argument(
        "--feature-cache-bytes",
        type=int,
        default=DEFAULT_FEATURE_CHUNK_CACHE_BYTES,
        help="Approximate in-memory budget for cached feature chunks during pair metrics.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads for CPU-side pair plan construction.",
    )
    return parser.parse_args()


def validate_params(params: EvaluationParams) -> None:
    if params.max_gap <= 0:
        raise ValueError("Max gap must be greater than 0.")
    if params.max_distance_xy <= 0:
        raise ValueError("Max XY distance must be greater than 0.")
    if params.max_distance_z <= 0:
        raise ValueError("Max Z distance must be greater than 0.")
    if params.z_distance_weight <= 0:
        raise ValueError("Z distance weight must be greater than 0.")
    if not 0.0 <= params.dice_threshold <= 1.0:
        raise ValueError("Dice threshold must be between 0 and 1.")
    if not -1.0 <= params.corr_threshold <= 1.0:
        raise ValueError("Correlation threshold must be between -1 and 1.")
    if params.csv_float_decimals < 0:
        raise ValueError("CSV float decimals must be nonnegative.")
    if candidate_output_mode(params) not in CANDIDATE_OUTPUT_MODES:
        raise ValueError(f"Candidate output must be one of {sorted(CANDIDATE_OUTPUT_MODES)}.")
    if params.candidate_batch_rows <= 0:
        raise ValueError("Candidate batch rows must be greater than 0.")
    if params.feature_cache_bytes <= 0:
        raise ValueError("Feature cache bytes must be greater than 0.")
    if params.num_workers <= 0:
        raise ValueError("Number of workers must be greater than 0.")


def candidate_output_mode(params: EvaluationParams) -> str:
    if not params.include_candidate_pairs:
        return "none"
    return str(params.candidate_output).strip().lower()


def needs_candidate_diagnostics(params: EvaluationParams) -> bool:
    return candidate_output_mode(params) in {"summary", "full"}


def writes_full_candidate_pairs(params: EvaluationParams) -> bool:
    return candidate_output_mode(params) == "full"


def resolve_num_features(requested: str, channel_count: int) -> int:
    value = str(requested).strip().lower()
    if value == "tracking":
        n_features = DEFAULT_N_TRACKING_FEATURES
    elif value == "all":
        n_features = int(channel_count)
    else:
        n_features = int(value)
    if n_features <= 0:
        raise ValueError("Number of features must be greater than 0.")
    if n_features > channel_count:
        raise PipelineError(
            f"Requested {n_features} feature channels, but the first feature volume has {channel_count}."
        )
    return n_features


def prepare_output_dir(output_path: Path, *, overwrite: bool) -> None:
    managed = [
        "manifest.json",
        "gt_rows.parquet",
        "gt_rows.csv",
        "candidate_pairs.parquet",
        "candidate_pairs.csv",
        "candidate_ref_summary.parquet",
        "candidate_ref_summary.csv",
        "feature_means.zarr",
        "true_gap_corr.zarr",
        "true_gap_mse.zarr",
        "gt_wide.csv",
    ]
    if output_path.exists() and not output_path.is_dir():
        raise FileNotFoundError(f"Output path exists but is not a directory: {output_path}")
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty. Use --overwrite to replace managed outputs: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for name in managed:
        path = output_path / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def find_gt_mask_path(gt_segmentation_path: Path, timepoint_name: str) -> Path:
    matches = [
        gt_segmentation_path / f"{timepoint_name}.tif",
        gt_segmentation_path / f"{timepoint_name}.tiff",
    ]
    existing = [path for path in matches if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise PipelineError(f"Ambiguous GT masks for {timepoint_name}: {[path.name for path in existing]}")
    raise FileNotFoundError(f"Missing GT segmentation for {timepoint_name}: expected .tif or .tiff.")


def coerce_label_volume(segmentation_yxz: np.ndarray) -> np.ndarray:
    if np.issubdtype(segmentation_yxz.dtype, np.integer):
        return segmentation_yxz.astype(np.int64, copy=False)
    finite = np.nan_to_num(segmentation_yxz, nan=0.0, posinf=0.0, neginf=0.0)
    return finite.astype(np.int64, copy=False)


def flat_indices_from_coords(coords_yxz: np.ndarray, shape_yxz: tuple[int, int, int]) -> np.ndarray:
    if coords_yxz.size == 0:
        return np.empty((0,), dtype=np.int64)
    y_size, x_size, z_size = (int(dim) for dim in shape_yxz)
    coords64 = coords_yxz.astype(np.int64, copy=False)
    return (coords64[:, 0] * x_size + coords64[:, 1]) * z_size + coords64[:, 2]


def build_foreground_arrays(
    geometries: dict[int, LabelGeometry],
    label_ids: np.ndarray,
    shape_yxz: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_count = int(label_ids.size)
    total_voxels = int(sum(geometries[int(label_id)].volume for label_id in label_ids.tolist()))
    coords = np.empty((total_voxels, 3), dtype=np.int32)
    label_indices = np.empty((total_voxels,), dtype=np.int32)
    voxel_counts = np.empty((label_count,), dtype=np.float64)
    label_starts = np.empty((label_count,), dtype=np.int64)

    cursor = 0
    for local_index, label_id in enumerate(label_ids.tolist()):
        geom = geometries[int(label_id)]
        count = int(geom.volume)
        stop = cursor + count
        coords[cursor:stop] = geom.vox_coords
        label_indices[cursor:stop] = local_index
        voxel_counts[local_index] = float(count)
        label_starts[local_index] = int(cursor)
        cursor = stop

    foreground_index_flat = np.full((int(np.prod(shape_yxz)),), -1, dtype=np.int32)
    if total_voxels:
        flat = flat_indices_from_coords(coords, shape_yxz)
        foreground_index_flat[flat] = np.arange(total_voxels, dtype=np.int32)

    return coords, label_indices, voxel_counts, label_starts, foreground_index_flat


def discover_timepoints(input_path: Path, gt_segmentation_path: Path) -> list[tuple[Any, Path]]:
    discovered = discover_inference_timepoints(input_path)
    out: list[tuple[Any, Path]] = []
    for timepoint in discovered:
        out.append((timepoint, find_gt_mask_path(gt_segmentation_path, timepoint.name)))
    return out


def prepare_timepoints(
    discovered: Sequence[tuple[Any, Path]],
    *,
    n_features: int,
) -> list[EvalTimepoint]:
    prepared: list[EvalTimepoint] = []
    expected_shape: tuple[int, int, int] | None = None
    for index, (timepoint, gt_path) in enumerate(discovered, start=1):
        print(f"[eval_GT_tracks] Preparing {timepoint.name} ({index}/{len(discovered)})", flush=True)
        raw_yxz = read_tiff_volume(timepoint.raw_path)
        gt_yxz = coerce_label_volume(read_tiff_volume(gt_path))
        if raw_yxz.shape != gt_yxz.shape:
            raise PipelineError(
                f"Shape mismatch in {timepoint.name}: raw={raw_yxz.shape}, gt={gt_yxz.shape}."
            )
        if expected_shape is None:
            expected_shape = tuple(int(dim) for dim in gt_yxz.shape)
        elif tuple(int(dim) for dim in gt_yxz.shape) != expected_shape:
            raise PipelineError(
                f"All GT volumes must share the same shape. Got {expected_shape} and {gt_yxz.shape}."
            )

        lr_feats = np.load(timepoint.lr_path, mmap_mode="r")
        if lr_feats.ndim != 4:
            raise PipelineError(f"{timepoint.lr_path} must have shape [Z, Y, X, C].")
        if int(lr_feats.shape[-1]) < n_features:
            raise PipelineError(f"{timepoint.lr_path} has only {lr_feats.shape[-1]} feature channels.")

        geometries = extract_label_geometries(gt_yxz)
        label_ids = np.array(sorted(geometries.keys()), dtype=np.int64)
        centroids = (
            np.vstack([geometries[int(label_id)].centroid for label_id in label_ids])
            if label_ids.size
            else np.empty((0, 3), dtype=np.float64)
        )
        shape_yxz = tuple(int(dim) for dim in gt_yxz.shape)
        (
            foreground_coords,
            foreground_label_indices,
            foreground_voxel_counts,
            foreground_label_starts,
            foreground_index_flat,
        ) = build_foreground_arrays(geometries, label_ids, shape_yxz)
        amplitude_labels, amplitude_values = compute_label_mean_intensities(gt_yxz, raw_yxz)
        amplitudes = {int(label): float(value) for label, value in zip(amplitude_labels, amplitude_values)}
        input_shape_yxz = (int(lr_feats.shape[1]), int(lr_feats.shape[2]), int(lr_feats.shape[0]))
        axis_maps = build_axis_maps(input_shape_yxz, shape_yxz)
        prepared.append(
            EvalTimepoint(
                index=index,
                name=timepoint.name,
                raw_path=timepoint.raw_path,
                gt_segmentation_path=gt_path,
                lr_feats_path=timepoint.lr_path,
                shape_yxz=shape_yxz,
                raw_z_size=int(raw_yxz.shape[2]),
                geometries=geometries,
                label_ids=label_ids,
                centroids=centroids,
                foreground_coords=foreground_coords,
                foreground_label_indices=foreground_label_indices,
                foreground_voxel_counts=foreground_voxel_counts,
                foreground_label_starts=foreground_label_starts,
                foreground_index_flat=foreground_index_flat,
                amplitudes=amplitudes,
                lr_feats=lr_feats,
                axis_maps=axis_maps,
            )
        )
    return prepared


def build_gt_rows(
    timepoints: Sequence[EvalTimepoint],
    *,
    max_gap: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int]]:
    observations_by_track: dict[int, list[int]] = {}
    for frame_index, timepoint in enumerate(timepoints):
        for label_id in timepoint.label_ids.tolist():
            observations_by_track.setdefault(int(label_id), []).append(frame_index)

    row_by_frame_label: dict[tuple[int, int], int] = {}
    rows: list[dict[str, Any]] = []
    for frame_index, timepoint in enumerate(timepoints):
        for label_id in timepoint.label_ids.tolist():
            track_frames = observations_by_track[int(label_id)]
            observed_index = track_frames.index(frame_index)
            start_frame = track_frames[0] + 1
            end_frame = track_frames[-1] + 1
            geom = timepoint.geometries[int(label_id)]
            row_index = len(rows)
            row_by_frame_label[(frame_index, int(label_id))] = row_index
            row: dict[str, Any] = {
                "row_index": row_index,
                "track_id": int(label_id),
                "gt_label_id": int(label_id),
                "start": int(start_frame),
                "t": int(observed_index + 1),
                "frame": int(frame_index + 1),
                "frame_zero_based": int(frame_index),
                "timepoint_name": timepoint.name,
                "x": float(geom.centroid[1]),
                "y": float(geom.centroid[0]),
                "z": float(geom.centroid[2]),
                "A": float(timepoint.amplitudes.get(int(label_id), np.nan)),
                "track_length": int(len(track_frames)),
                "track_duration": int(end_frame - start_frame + 1),
                "num_voxels": int(geom.volume),
            }
            for gap in range(1, max_gap + 1):
                row[f"target_row_index_gap_{gap}"] = None
                row[f"dice_gap_{gap}"] = None
                row[f"anisotropic_distance_gap_{gap}"] = None
                row[f"overlap_voxels_gap_{gap}"] = None
                row[f"feat_corr_mean_gap_{gap}"] = None
                row[f"feat_corr_median_gap_{gap}"] = None
                row[f"feat_mse_mean_gap_{gap}"] = None
                row[f"feat_mse_median_gap_{gap}"] = None
                row[f"feat_pass_count_gap_{gap}"] = None
            rows.append(row)
    return rows, row_by_frame_label


def compressor_for_params(params: EvaluationParams) -> Any:
    if params.no_compression:
        return None
    return Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)


def zarr_version_kwargs() -> dict[str, int]:
    parameters = inspect.signature(zarr.open).parameters
    if "zarr_format" in parameters:
        return {"zarr_format": 2}
    if "zarr_version" in parameters:
        return {"zarr_version": 2}
    return {}


def create_metric_arrays(
    output_path: Path,
    *,
    n_rows: int,
    n_features: int,
    max_gap: int,
    params: EvaluationParams,
) -> tuple[Any, Any, Any]:
    compressor = compressor_for_params(params)
    version_kwargs = zarr_version_kwargs()
    row_chunk = max(1, min(256, n_rows))
    feature_chunk = max(1, min(64, n_features))
    feature_means = zarr.open(
        str(output_path / "feature_means.zarr"),
        mode="w",
        shape=(n_rows, n_features),
        chunks=(row_chunk, feature_chunk),
        dtype="f4",
        compressor=compressor,
        fill_value=np.nan,
        **version_kwargs,
    )
    gap_corr = np.full((n_rows, max_gap, n_features), np.nan, dtype=np.float32)
    gap_mse = np.full((n_rows, max_gap, n_features), np.nan, dtype=np.float32)
    return feature_means, gap_corr, gap_mse


def write_true_gap_metric_arrays(
    output_path: Path,
    gap_corr: np.ndarray,
    gap_mse: np.ndarray,
    *,
    params: EvaluationParams,
) -> None:
    compressor = compressor_for_params(params)
    version_kwargs = zarr_version_kwargs()
    n_rows, max_gap, n_features = gap_corr.shape
    if gap_mse.shape != gap_corr.shape:
        raise PipelineError(f"True-gap corr/mse shape mismatch: {gap_corr.shape} vs {gap_mse.shape}.")

    row_chunk = max(1, min(256, n_rows))
    feature_chunk = max(1, min(64, n_features))
    gap_corr_store = zarr.open(
        str(output_path / "true_gap_corr.zarr"),
        mode="w",
        shape=(n_rows, max_gap, n_features),
        chunks=(row_chunk, 1, feature_chunk),
        dtype="f4",
        compressor=compressor,
        fill_value=np.nan,
        **version_kwargs,
    )
    gap_mse_store = zarr.open(
        str(output_path / "true_gap_mse.zarr"),
        mode="w",
        shape=(n_rows, max_gap, n_features),
        chunks=(row_chunk, 1, feature_chunk),
        dtype="f4",
        compressor=compressor,
        fill_value=np.nan,
        **version_kwargs,
    )

    print("[eval_GT_tracks] Writing true-gap Zarr arrays", flush=True)
    for start in range(0, n_features, feature_chunk):
        end = min(n_features, start + feature_chunk)
        print(f"[eval_GT_tracks] Writing true-gap feature block {start}:{end}", flush=True)
        gap_corr_store[:, :, start:end] = gap_corr[:, :, start:end]
        gap_mse_store[:, :, start:end] = gap_mse[:, :, start:end]


def finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite, dtype=np.float64))


def finite_median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def compute_feature_corr_mse(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    ref_coords: np.ndarray,
    cand_coords: np.ndarray,
    *,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    corr = np.full((n_features,), np.nan, dtype=np.float32)
    mse = np.full((n_features,), np.nan, dtype=np.float32)
    if ref_coords.shape[0] <= 1 or cand_coords.shape[0] <= 1:
        return corr, mse

    batch_size = choose_feature_batch_size(
        tuple(int(dim) for dim in ref_tp.lr_feats.shape[:3]),
        tuple(int(dim) for dim in cand_tp.lr_feats.shape[:3]),
        n_features=n_features,
    )
    for start in range(0, n_features, batch_size):
        end = min(n_features, start + batch_size)
        ref_chunk = load_feature_chunk_internal_yxz(ref_tp.lr_feats, start=start, end=end)
        cand_chunk = load_feature_chunk_internal_yxz(cand_tp.lr_feats, start=start, end=end)
        ref_values = sample_feature_chunk_at_internal_coords(ref_chunk, ref_tp.axis_maps, ref_coords)
        cand_values = sample_feature_chunk_at_internal_coords(cand_chunk, cand_tp.axis_maps, cand_coords)
        diff = ref_values - cand_values
        mse[start:end] = np.mean(diff * diff, axis=1, dtype=np.float32)
        corr[start:end] = rowwise_correlation(ref_values, cand_values)
    return corr, mse


def build_foreground_coord_table(
    timepoint: EvalTimepoint,
    frame_index: int,
    row_by_frame_label: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_indices = np.empty((timepoint.label_ids.size,), dtype=np.int64)
    for local_index, label_id in enumerate(timepoint.label_ids.tolist()):
        row_indices[local_index] = row_by_frame_label[(frame_index, int(label_id))]
    return (
        timepoint.foreground_coords,
        timepoint.foreground_label_indices,
        row_indices,
        timepoint.foreground_voxel_counts,
    )


def choose_feature_mean_coord_batch_size(feature_count: int) -> int:
    bytes_per_voxel = max(1, int(feature_count)) * np.dtype(np.float32).itemsize
    return max(1, DEFAULT_FEATURE_MEAN_SAMPLE_BYTES // bytes_per_voxel)


def write_feature_mean_block(feature_means: Any, row_indices: np.ndarray, start: int, end: int, means: np.ndarray) -> None:
    if row_indices.size == 0:
        return
    first_row = int(row_indices[0])
    expected = np.arange(first_row, first_row + int(row_indices.size), dtype=np.int64)
    if np.array_equal(row_indices, expected):
        feature_means[first_row : first_row + int(row_indices.size), start:end] = means.astype(np.float32, copy=False)
        return
    for local_index, row_index in enumerate(row_indices.tolist()):
        feature_means[int(row_index), start:end] = means[local_index].astype(np.float32, copy=False)


def compute_feature_means(
    timepoints: Sequence[EvalTimepoint],
    row_by_frame_label: dict[tuple[int, int], int],
    feature_means: Any,
    *,
    n_features: int,
) -> None:
    for frame_index, timepoint in enumerate(timepoints):
        print(f"[eval_GT_tracks] Computing feature means for {timepoint.name}", flush=True)
        if timepoint.label_ids.size == 0:
            continue
        coords, label_indices, row_indices, voxel_counts = build_foreground_coord_table(
            timepoint,
            frame_index,
            row_by_frame_label,
        )
        if coords.shape[0] == 0:
            continue
        batch_size = choose_feature_batch_size(
            tuple(int(dim) for dim in timepoint.lr_feats.shape[:3]),
            tuple(int(dim) for dim in timepoint.lr_feats.shape[:3]),
            n_features=n_features,
        )
        for start in range(0, n_features, batch_size):
            end = min(n_features, start + batch_size)
            chunk = load_feature_chunk_internal_yxz(timepoint.lr_feats, start=start, end=end)
            chunk_width = int(end - start)
            sums = np.zeros((timepoint.label_ids.size, chunk_width), dtype=np.float64)
            coord_batch_size = choose_feature_mean_coord_batch_size(chunk_width)
            for coord_start in range(0, coords.shape[0], coord_batch_size):
                coord_end = min(coords.shape[0], coord_start + coord_batch_size)
                local_coords = coords[coord_start:coord_end]
                local_label_indices = label_indices[coord_start:coord_end]
                sampled = sample_feature_chunk_at_internal_coords(chunk, timepoint.axis_maps, local_coords)
                for feature_offset in range(chunk_width):
                    sums[:, feature_offset] += np.bincount(
                        local_label_indices,
                        weights=sampled[feature_offset],
                        minlength=int(timepoint.label_ids.size),
                    )
            means = sums / voxel_counts[:, None]
            write_feature_mean_block(feature_means, row_indices, start, end, means)


def choose_pair_metric_coord_batch_size(feature_count: int) -> int:
    bytes_per_coord = max(1, int(feature_count)) * np.dtype(np.float32).itemsize * 2
    return max(1, DEFAULT_PAIR_METRIC_SAMPLE_BYTES // bytes_per_coord)


def build_pair_coord_table(entries: Sequence[CandidateEntry]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros((len(entries),), dtype=np.float64)
    total_coords = 0
    for index, entry in enumerate(entries):
        if entry.overlap_voxels <= 1:
            continue
        count = int(entry.ref_coords.shape[0])
        if count <= 1:
            continue
        counts[index] = float(count)
        total_coords += count

    ref_coords = np.empty((total_coords, 3), dtype=np.int32)
    cand_coords = np.empty((total_coords, 3), dtype=np.int32)
    group_indices = np.empty((total_coords,), dtype=np.int32)
    cursor = 0
    for index, entry in enumerate(entries):
        count = int(counts[index])
        if count <= 1:
            continue
        stop = cursor + count
        ref_coords[cursor:stop] = entry.ref_coords
        cand_coords[cursor:stop] = entry.cand_coords
        group_indices[cursor:stop] = index
        cursor = stop
    return ref_coords, cand_coords, group_indices, counts


def coords_to_foreground_indices(timepoint: EvalTimepoint, coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.empty((0,), dtype=np.int32)
    if timepoint.foreground_index_flat is None:
        raise PipelineError(f"Foreground index map was already released for {timepoint.name}.")
    flat = flat_indices_from_coords(coords, timepoint.shape_yxz)
    indices = timepoint.foreground_index_flat[flat]
    if np.any(indices < 0):
        raise PipelineError(f"Overlap coordinates include non-foreground voxels in {timepoint.name}.")
    return indices.astype(np.int32, copy=False)


def build_pair_index_table(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    entries: Sequence[CandidateEntry],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros((len(entries),), dtype=np.float64)
    total_coords = 0
    for index, entry in enumerate(entries):
        if entry.overlap_voxels <= 1:
            continue
        count = int(entry.ref_coords.shape[0])
        if count <= 1:
            continue
        counts[index] = float(count)
        total_coords += count

    ref_indices = np.empty((total_coords,), dtype=np.int32)
    cand_indices = np.empty((total_coords,), dtype=np.int32)
    group_indices = np.empty((total_coords,), dtype=np.int32)
    cursor = 0
    for index, entry in enumerate(entries):
        count = int(counts[index])
        if count <= 1:
            continue
        stop = cursor + count
        ref_indices[cursor:stop] = coords_to_foreground_indices(ref_tp, entry.ref_coords)
        cand_indices[cursor:stop] = coords_to_foreground_indices(cand_tp, entry.cand_coords)
        group_indices[cursor:stop] = index
        cursor = stop
    return ref_indices, cand_indices, group_indices, counts


def release_foreground_index_maps(timepoints: Sequence[EvalTimepoint]) -> None:
    for timepoint in timepoints:
        timepoint.foreground_index_flat = None


def compute_pair_feature_metrics_from_table(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    *,
    ref_coords: np.ndarray,
    cand_coords: np.ndarray,
    group_indices: np.ndarray,
    counts: np.ndarray,
    metric_count: int,
    start: int,
    end: int,
    ref_chunk: np.ndarray,
    cand_chunk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chunk_width = int(end - start)
    corr = np.full((metric_count, chunk_width), np.nan, dtype=np.float32)
    mse = np.full((metric_count, chunk_width), np.nan, dtype=np.float32)
    if metric_count == 0 or ref_coords.shape[0] == 0:
        return corr, mse

    valid_counts = counts > 1.0
    sum_ref = np.zeros((metric_count, chunk_width), dtype=np.float64)
    sum_cand = np.zeros((metric_count, chunk_width), dtype=np.float64)
    sum_ref_sq = np.zeros((metric_count, chunk_width), dtype=np.float64)
    sum_cand_sq = np.zeros((metric_count, chunk_width), dtype=np.float64)
    sum_cross = np.zeros((metric_count, chunk_width), dtype=np.float64)
    sum_diff_sq = np.zeros((metric_count, chunk_width), dtype=np.float64)

    coord_batch_size = choose_pair_metric_coord_batch_size(chunk_width)
    for coord_start in range(0, ref_coords.shape[0], coord_batch_size):
        coord_end = min(ref_coords.shape[0], coord_start + coord_batch_size)
        local_groups = group_indices[coord_start:coord_end]
        ref_values = sample_feature_chunk_at_internal_coords(
            ref_chunk,
            ref_tp.axis_maps,
            ref_coords[coord_start:coord_end],
        )
        cand_values = sample_feature_chunk_at_internal_coords(
            cand_chunk,
            cand_tp.axis_maps,
            cand_coords[coord_start:coord_end],
        )
        diff = ref_values - cand_values
        for feature_offset in range(chunk_width):
            ref_feature = ref_values[feature_offset]
            cand_feature = cand_values[feature_offset]
            sum_ref[:, feature_offset] += np.bincount(
                local_groups,
                weights=ref_feature,
                minlength=metric_count,
            )
            sum_cand[:, feature_offset] += np.bincount(
                local_groups,
                weights=cand_feature,
                minlength=metric_count,
            )
            sum_ref_sq[:, feature_offset] += np.bincount(
                local_groups,
                weights=ref_feature * ref_feature,
                minlength=metric_count,
            )
            sum_cand_sq[:, feature_offset] += np.bincount(
                local_groups,
                weights=cand_feature * cand_feature,
                minlength=metric_count,
            )
            sum_cross[:, feature_offset] += np.bincount(
                local_groups,
                weights=ref_feature * cand_feature,
                minlength=metric_count,
            )
            sum_diff_sq[:, feature_offset] += np.bincount(
                local_groups,
                weights=diff[feature_offset] * diff[feature_offset],
                minlength=metric_count,
            )

    count_matrix = counts[:, None]
    valid_mse = counts > 0.0
    mse_block = np.full((metric_count, chunk_width), np.nan, dtype=np.float32)
    mse_block[valid_mse] = (sum_diff_sq[valid_mse] / count_matrix[valid_mse]).astype(np.float32, copy=False)
    mse[:, :] = mse_block

    with np.errstate(invalid="ignore", divide="ignore"):
        ref_ss = sum_ref_sq - (sum_ref * sum_ref) / count_matrix
        cand_ss = sum_cand_sq - (sum_cand * sum_cand) / count_matrix
        numerator = sum_cross - (sum_ref * sum_cand) / count_matrix
        denominator = np.sqrt(ref_ss * cand_ss)
    valid_corr = valid_counts[:, None] & (denominator > 0.0)
    corr_block = np.full((metric_count, chunk_width), np.nan, dtype=np.float32)
    corr_block[valid_corr] = (numerator[valid_corr] / denominator[valid_corr]).astype(np.float32, copy=False)
    corr[:, :] = corr_block
    return corr, mse


def compute_pair_feature_metrics(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    entries: Sequence[CandidateEntry],
    *,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    corr = np.full((len(entries), n_features), np.nan, dtype=np.float32)
    mse = np.full((len(entries), n_features), np.nan, dtype=np.float32)
    if not entries:
        return corr, mse

    ref_coords, cand_coords, group_indices, counts = build_pair_coord_table(entries)
    if ref_coords.shape[0] == 0:
        return corr, mse

    batch_size = choose_feature_batch_size(
        tuple(int(dim) for dim in ref_tp.lr_feats.shape[:3]),
        tuple(int(dim) for dim in cand_tp.lr_feats.shape[:3]),
        n_features=n_features,
    )
    for start in range(0, n_features, batch_size):
        end = min(n_features, start + batch_size)
        ref_chunk = load_feature_chunk_internal_yxz(ref_tp.lr_feats, start=start, end=end)
        cand_chunk = load_feature_chunk_internal_yxz(cand_tp.lr_feats, start=start, end=end)
        corr_block, mse_block = compute_pair_feature_metrics_from_table(
            ref_tp,
            cand_tp,
            ref_coords=ref_coords,
            cand_coords=cand_coords,
            group_indices=group_indices,
            counts=counts,
            metric_count=len(entries),
            start=start,
            end=end,
            ref_chunk=ref_chunk,
            cand_chunk=cand_chunk,
        )
        corr[:, start:end] = corr_block
        mse[:, start:end] = mse_block
    return corr, mse


def choose_global_pair_feature_batch_size(timepoints: Sequence[EvalTimepoint], n_features: int) -> int:
    if not timepoints:
        return max(1, int(n_features))
    max_voxels = max(int(np.prod(timepoint.lr_feats.shape[:3])) for timepoint in timepoints)
    bytes_per_channel_pair = max_voxels * np.dtype(np.float32).itemsize * 2
    return max(1, min(int(n_features), DEFAULT_FEATURE_MEAN_SAMPLE_BYTES // max(1, bytes_per_channel_pair)))


def estimate_feature_chunk_bytes(timepoint: EvalTimepoint, start: int, end: int) -> int:
    return int(np.prod(timepoint.lr_feats.shape[:3])) * int(end - start) * np.dtype(np.float32).itemsize


class FeatureChunkCache:
    def __init__(
        self,
        timepoints: Sequence[EvalTimepoint],
        *,
        start: int,
        end: int,
        max_bytes: int,
    ) -> None:
        self._timepoints = timepoints
        self._start = int(start)
        self._end = int(end)
        max_chunk_bytes = max(
            estimate_feature_chunk_bytes(timepoint, start, end)
            for timepoint in timepoints
        )
        self._max_items = max(1, min(len(timepoints), int(max_bytes) // max(1, max_chunk_bytes)))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def max_items(self) -> int:
        return self._max_items

    def get(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        cached = self._cache.get(frame_index)
        if cached is not None:
            self._cache.move_to_end(frame_index)
            return cached

        timepoint = self._timepoints[frame_index]
        chunk = load_feature_chunk_internal_yxz(timepoint.lr_feats, start=self._start, end=self._end)
        self._cache[frame_index] = chunk
        self._cache.move_to_end(frame_index)
        while len(self._cache) > self._max_items:
            self._cache.popitem(last=False)
        return chunk


def estimate_foreground_sample_bytes(timepoint: EvalTimepoint, start: int, end: int) -> int:
    return int(timepoint.foreground_coords.shape[0]) * int(end - start) * np.dtype(np.float32).itemsize


class ForegroundFeatureCache:
    def __init__(
        self,
        timepoints: Sequence[EvalTimepoint],
        *,
        start: int,
        end: int,
        max_bytes: int,
    ) -> None:
        self._timepoints = timepoints
        self._start = int(start)
        self._end = int(end)
        max_sample_bytes = max(
            estimate_foreground_sample_bytes(timepoint, start, end)
            for timepoint in timepoints
        )
        self._max_items = max(1, min(len(timepoints), int(max_bytes) // max(1, max_sample_bytes)))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def max_items(self) -> int:
        return self._max_items

    def get(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        cached = self._cache.get(frame_index)
        if cached is not None:
            self._cache.move_to_end(frame_index)
            return cached

        timepoint = self._timepoints[frame_index]
        chunk = load_feature_chunk_internal_yxz(timepoint.lr_feats, start=self._start, end=self._end)
        sampled = sample_feature_chunk_at_internal_coords(chunk, timepoint.axis_maps, timepoint.foreground_coords)
        self._cache[frame_index] = sampled
        self._cache.move_to_end(frame_index)
        while len(self._cache) > self._max_items:
            self._cache.popitem(last=False)
        return sampled


def grouped_pair_feature_metrics_from_samples(
    ref_samples: np.ndarray,
    cand_samples: np.ndarray,
    ref_indices: np.ndarray,
    cand_indices: np.ndarray,
    counts: np.ndarray,
    *,
    metric_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    chunk_width = int(ref_samples.shape[0])
    corr = np.full((metric_count, chunk_width), np.nan, dtype=np.float32)
    mse = np.full((metric_count, chunk_width), np.nan, dtype=np.float32)
    if metric_count == 0 or ref_indices.size == 0:
        return corr, mse

    counts_int = counts.astype(np.int64, copy=False)
    valid_mse = counts_int > 0
    valid_corr_counts = counts_int > 1
    if not np.any(valid_mse):
        return corr, mse

    starts_all = np.empty((metric_count,), dtype=np.int64)
    if metric_count:
        starts_all[0] = 0
        if metric_count > 1:
            starts_all[1:] = np.cumsum(counts_int[:-1])
    valid_indices = np.flatnonzero(valid_mse)
    starts = starts_all[valid_indices]
    count_matrix = counts[valid_indices, None]

    ref_values = ref_samples[:, ref_indices]
    cand_values = cand_samples[:, cand_indices]
    ref64 = ref_values.astype(np.float64, copy=False)
    cand64 = cand_values.astype(np.float64, copy=False)

    sum_ref = np.add.reduceat(ref64, starts, axis=1).T
    sum_cand = np.add.reduceat(cand64, starts, axis=1).T
    ref_sq = (ref_values * ref_values).astype(np.float64, copy=False)
    cand_sq = (cand_values * cand_values).astype(np.float64, copy=False)
    cross = (ref_values * cand_values).astype(np.float64, copy=False)
    diff = ref_values - cand_values
    diff_sq = (diff * diff).astype(np.float64, copy=False)

    sum_ref_sq = np.add.reduceat(ref_sq, starts, axis=1).T
    sum_cand_sq = np.add.reduceat(cand_sq, starts, axis=1).T
    sum_cross = np.add.reduceat(cross, starts, axis=1).T
    sum_diff_sq = np.add.reduceat(diff_sq, starts, axis=1).T

    mse[valid_indices] = (sum_diff_sq / count_matrix).astype(np.float32, copy=False)

    with np.errstate(invalid="ignore", divide="ignore"):
        ref_ss = sum_ref_sq - (sum_ref * sum_ref) / count_matrix
        cand_ss = sum_cand_sq - (sum_cand * sum_cand) / count_matrix
        numerator = sum_cross - (sum_ref * sum_cand) / count_matrix
        denominator = np.sqrt(ref_ss * cand_ss)

    valid_corr_rows = valid_corr_counts[valid_indices, None] & (denominator > 0.0)
    corr_valid = np.full((valid_indices.size, chunk_width), np.nan, dtype=np.float32)
    corr_valid[valid_corr_rows] = (numerator[valid_corr_rows] / denominator[valid_corr_rows]).astype(
        np.float32,
        copy=False,
    )
    corr[valid_indices] = corr_valid
    return corr, mse


def write_feature_mean_block_from_samples(
    timepoint: EvalTimepoint,
    frame_index: int,
    row_by_frame_label: dict[tuple[int, int], int],
    feature_means: Any,
    sampled: np.ndarray,
    *,
    start: int,
    end: int,
) -> None:
    if timepoint.label_ids.size == 0 or sampled.shape[1] == 0:
        return
    row_indices = np.array(
        [row_by_frame_label[(frame_index, int(label_id))] for label_id in timepoint.label_ids.tolist()],
        dtype=np.int64,
    )
    sums = np.add.reduceat(sampled.astype(np.float64, copy=False), timepoint.foreground_label_starts, axis=1).T
    means = sums / timepoint.foreground_voxel_counts[:, None]
    write_feature_mean_block(feature_means, row_indices, start, end, means)


def build_true_gap_entries_for_pair(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    *,
    ref_frame: int,
    gap: int,
    row_by_frame_label: dict[tuple[int, int], int],
    params: EvaluationParams,
) -> list[CandidateEntry]:
    cand_frame = ref_frame + gap
    common_labels = sorted(set(ref_tp.label_ids.tolist()) & set(cand_tp.label_ids.tolist()), key=int)
    entries: list[CandidateEntry] = []
    for label_id in common_labels:
        ref_geom = ref_tp.geometries[int(label_id)]
        cand_geom = cand_tp.geometries[int(label_id)]
        dice, ref_coords, cand_coords, overlap_count = compute_alignment_overlap(ref_geom, cand_geom)
        distance = anisotropic_distance(ref_geom.centroid, cand_geom.centroid, params.z_distance_weight)
        entries.append(
            CandidateEntry(
                ref_row_index=row_by_frame_label[(ref_frame, int(label_id))],
                true_row_index=row_by_frame_label[(cand_frame, int(label_id))],
                cand_row_index=row_by_frame_label[(cand_frame, int(label_id))],
                ref_track_id=int(label_id),
                cand_track_id=int(label_id),
                distance=float(distance),
                dice=float(dice),
                overlap_voxels=int(overlap_count),
                ref_coords=ref_coords,
                cand_coords=cand_coords,
            )
        )
    return entries


def compute_true_gap_pair_result(
    timepoints: Sequence[EvalTimepoint],
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    gap: int,
    ref_frame: int,
    params: EvaluationParams,
    n_features: int,
) -> TrueGapPairResult:
    entries = build_true_gap_entries_for_pair(
        timepoints[ref_frame],
        timepoints[ref_frame + gap],
        ref_frame=ref_frame,
        gap=gap,
        row_by_frame_label=row_by_frame_label,
        params=params,
    )
    corr, mse = compute_pair_feature_metrics(
        timepoints[ref_frame],
        timepoints[ref_frame + gap],
        entries,
        n_features=n_features,
    )
    return TrueGapPairResult(gap=gap, entries=entries, corr=corr, mse=mse)


def iter_true_gap_pair_results(
    timepoints: Sequence[EvalTimepoint],
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    gap: int,
    params: EvaluationParams,
    n_features: int,
) -> Iterable[TrueGapPairResult]:
    max_ref_frame = len(timepoints) - gap
    if params.num_workers <= 1 or max_ref_frame <= 1:
        for ref_frame in range(max_ref_frame):
            yield compute_true_gap_pair_result(
                timepoints,
                row_by_frame_label,
                gap=gap,
                ref_frame=ref_frame,
                params=params,
                n_features=n_features,
            )
        return

    next_frame = 0
    futures: set[Future[TrueGapPairResult]] = set()
    with ThreadPoolExecutor(max_workers=int(params.num_workers)) as executor:
        while next_frame < max_ref_frame and len(futures) < int(params.num_workers):
            futures.add(
                executor.submit(
                    compute_true_gap_pair_result,
                    timepoints,
                    row_by_frame_label,
                    gap=gap,
                    ref_frame=next_frame,
                    params=params,
                    n_features=n_features,
                )
            )
            next_frame += 1

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                if next_frame < max_ref_frame:
                    futures.add(
                        executor.submit(
                            compute_true_gap_pair_result,
                            timepoints,
                            row_by_frame_label,
                            gap=gap,
                            ref_frame=next_frame,
                            params=params,
                            n_features=n_features,
                        )
                    )
                    next_frame += 1


def apply_true_gap_pair_result(
    result: TrueGapPairResult,
    rows: list[dict[str, Any]],
    gap_corr: Any,
    gap_mse: Any,
    *,
    params: EvaluationParams,
) -> None:
    gap = int(result.gap)
    for entry_index, entry in enumerate(result.entries):
        corr = result.corr[entry_index]
        mse = result.mse[entry_index]
        row_index = int(entry.ref_row_index)
        row = rows[row_index]
        row[f"target_row_index_gap_{gap}"] = int(entry.cand_row_index)
        row[f"dice_gap_{gap}"] = float(entry.dice)
        row[f"anisotropic_distance_gap_{gap}"] = float(entry.distance)
        row[f"overlap_voxels_gap_{gap}"] = int(entry.overlap_voxels)
        gap_corr[row_index, gap - 1, :] = corr
        gap_mse[row_index, gap - 1, :] = mse
        row[f"feat_corr_mean_gap_{gap}"] = finite_mean(corr)
        row[f"feat_corr_median_gap_{gap}"] = finite_median(corr)
        row[f"feat_mse_mean_gap_{gap}"] = finite_mean(mse)
        row[f"feat_mse_median_gap_{gap}"] = finite_median(mse)
        good = (
            np.isfinite(corr)
            & np.isfinite(mse)
            & math.isfinite(float(entry.dice))
            & (float(entry.dice) > params.dice_threshold)
            & (corr > params.corr_threshold)
        )
        row[f"feat_pass_count_gap_{gap}"] = int(np.count_nonzero(good))


def compute_true_gap_metrics(
    timepoints: Sequence[EvalTimepoint],
    rows: list[dict[str, Any]],
    row_by_frame_label: dict[tuple[int, int], int],
    gap_corr: Any,
    gap_mse: Any,
    *,
    params: EvaluationParams,
    n_features: int,
) -> None:
    for gap in range(1, params.max_gap + 1):
        print(
            f"[eval_GT_tracks] Computing true GT gap {gap}/{params.max_gap} with {params.num_workers} worker(s)",
            flush=True,
        )
        for result in iter_true_gap_pair_results(
            timepoints,
            row_by_frame_label,
            gap=gap,
            params=params,
            n_features=n_features,
        ):
            apply_true_gap_pair_result(result, rows, gap_corr, gap_mse, params=params)


def candidate_field_schema() -> tuple[list[str], list[tuple[str, str]]]:
    fields = [
        ("ref_row_index", "int"),
        ("true_row_index", "int"),
        ("cand_row_index", "int"),
        ("ref_track_id", "int"),
        ("cand_track_id", "int"),
        ("ref_frame", "int"),
        ("cand_frame", "int"),
        ("gap", "int"),
        ("ref_timepoint_name", "str"),
        ("cand_timepoint_name", "str"),
        ("is_true_match", "bool"),
        ("anisotropic_distance", "float"),
        ("dice", "float"),
        ("overlap_voxels", "int"),
        ("feat_corr_mean", "float"),
        ("feat_corr_median", "float"),
        ("feat_mse_mean", "float"),
        ("feat_mse_median", "float"),
        ("vote_count", "int"),
        ("rank_distance", "int"),
        ("rank_dice", "int"),
        ("rank_corr_mean", "int"),
        ("rank_mse_mean", "int"),
        ("rank_vote_count", "int"),
        ("true_candidate_present", "bool"),
        ("true_rank_distance", "int"),
        ("true_rank_dice", "int"),
        ("true_rank_corr_mean", "int"),
        ("true_rank_mse_mean", "int"),
        ("true_rank_vote_count", "int"),
        ("true_distance_margin", "float"),
        ("true_dice_margin", "float"),
        ("true_corr_mean_margin", "float"),
        ("true_mse_mean_margin", "float"),
        ("true_vote_count_margin", "float"),
    ]
    return [name for name, _kind in fields], fields


def candidate_summary_field_schema() -> tuple[list[str], list[tuple[str, str]]]:
    fields = [
        ("ref_row_index", "int"),
        ("true_row_index", "int"),
        ("ref_track_id", "int"),
        ("ref_frame", "int"),
        ("cand_frame", "int"),
        ("gap", "int"),
        ("ref_timepoint_name", "str"),
        ("cand_timepoint_name", "str"),
        ("candidate_count", "int"),
        ("true_candidate_present", "bool"),
        ("true_rank_distance", "int"),
        ("true_rank_dice", "int"),
        ("true_rank_corr_mean", "int"),
        ("true_rank_mse_mean", "int"),
        ("true_rank_vote_count", "int"),
        ("true_distance_margin", "float"),
        ("true_dice_margin", "float"),
        ("true_corr_mean_margin", "float"),
        ("true_mse_mean_margin", "float"),
        ("true_vote_count_margin", "float"),
    ]
    return [name for name, _kind in fields], fields


def gt_field_schema(max_gap: int) -> tuple[list[str], list[tuple[str, str]]]:
    fields = [
        ("row_index", "int"),
        ("track_id", "int"),
        ("gt_label_id", "int"),
        ("start", "int"),
        ("t", "int"),
        ("frame", "int"),
        ("frame_zero_based", "int"),
        ("timepoint_name", "str"),
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("A", "float"),
        ("track_length", "int"),
        ("track_duration", "int"),
        ("num_voxels", "int"),
    ]
    for gap in range(1, max_gap + 1):
        fields.extend(
            [
                (f"target_row_index_gap_{gap}", "int"),
                (f"dice_gap_{gap}", "float"),
                (f"anisotropic_distance_gap_{gap}", "float"),
                (f"overlap_voxels_gap_{gap}", "int"),
                (f"feat_corr_mean_gap_{gap}", "float"),
                (f"feat_corr_median_gap_{gap}", "float"),
                (f"feat_mse_mean_gap_{gap}", "float"),
                (f"feat_mse_median_gap_{gap}", "float"),
                (f"feat_pass_count_gap_{gap}", "int"),
            ]
        )
    return [name for name, _kind in fields], fields


def rank_metric(values: Sequence[float | int | None], *, higher_better: bool) -> list[int | None]:
    finite_items: list[tuple[float, int]] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            finite_items.append((numeric, index))
    if higher_better:
        finite_items.sort(key=lambda item: (-item[0], item[1]))
    else:
        finite_items.sort(key=lambda item: (item[0], item[1]))
    ranks: list[int | None] = [None] * len(values)
    for rank, (_value, index) in enumerate(finite_items, start=1):
        ranks[index] = rank
    return ranks


def best_false_margin(
    true_value: float | int | None,
    false_values: Sequence[float | int | None],
    *,
    higher_better: bool,
) -> float | None:
    if true_value is None:
        return None
    true_float = float(true_value)
    if not math.isfinite(true_float):
        return None
    finite_false = [float(value) for value in false_values if value is not None and math.isfinite(float(value))]
    if not finite_false:
        return None
    best_false = max(finite_false) if higher_better else min(finite_false)
    return float(true_float - best_false if higher_better else best_false - true_float)


def finite_row_mean(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    array = values.astype(np.float64, copy=False)
    finite = np.isfinite(array)
    counts = np.count_nonzero(finite, axis=1)
    sums = np.where(finite, array, 0.0).sum(axis=1, dtype=np.float64)
    out = np.full((values.shape[0],), np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def finite_row_median(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    masked = np.array(values, copy=True)
    masked[~np.isfinite(masked)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(masked, axis=1)


def fill_rank_column(
    out: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    *,
    higher_better: bool,
) -> None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return
    finite_indices = indices[finite]
    finite_values = values[finite]
    order_values = -finite_values if higher_better else finite_values
    order = np.argsort(order_values, kind="mergesort")
    ranked_indices = finite_indices[order]
    out[ranked_indices] = np.arange(1, ranked_indices.size + 1, dtype=np.int64)
    mask[ranked_indices] = False


def false_margin_from_values(true_value: float, false_values: np.ndarray, *, higher_better: bool) -> float:
    if not math.isfinite(float(true_value)):
        return np.nan
    finite_false = false_values[np.isfinite(false_values)]
    if finite_false.size == 0:
        return np.nan
    best_false = float(np.max(finite_false) if higher_better else np.min(finite_false))
    return float(true_value - best_false if higher_better else best_false - true_value)


def candidate_group_ranges(ref_row_indices: np.ndarray) -> list[tuple[int, int]]:
    if ref_row_indices.size == 0:
        return []
    starts = np.concatenate(([0], np.flatnonzero(ref_row_indices[1:] != ref_row_indices[:-1]) + 1))
    ends = np.concatenate((starts[1:], [ref_row_indices.size]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def compute_vote_counts_vectorized(
    entries: Sequence[CandidateEntry],
    corr: np.ndarray,
    mse: np.ndarray,
    *,
    params: EvaluationParams,
    ref_row_indices: np.ndarray | None = None,
    dice_values: np.ndarray | None = None,
) -> np.ndarray:
    n_entries = len(entries)
    votes = np.zeros((n_entries,), dtype=np.int64)
    if n_entries == 0 or corr.shape[1] == 0:
        return votes
    if ref_row_indices is None:
        ref_row_indices = np.array([entry.ref_row_index for entry in entries], dtype=np.int64)
    if dice_values is None:
        dice_values = np.array([entry.dice for entry in entries], dtype=np.float64)

    for start, end in candidate_group_ranges(ref_row_indices):
        group_corr = corr[start:end]
        group_mse = mse[start:end]
        good = (
            np.isfinite(dice_values[start:end, None])
            & (dice_values[start:end, None] > params.dice_threshold)
            & np.isfinite(group_corr)
            & (group_corr > params.corr_threshold)
            & np.isfinite(group_mse)
        )
        if not np.any(good):
            continue
        scores = np.where(good, group_mse, np.inf)
        winners = np.argmin(scores, axis=0)
        winner_scores = scores[winners, np.arange(scores.shape[1])]
        valid_features = np.isfinite(winner_scores)
        if np.any(valid_features):
            np.add.at(votes, start + winners[valid_features], 1)
    return votes


def compute_candidate_entries(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    ref_frame: int,
    cand_frame: int,
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    params: EvaluationParams,
) -> list[CandidateEntry]:
    entries: list[CandidateEntry] = []
    radius = np.asarray(
        (float(params.max_distance_xy), float(params.max_distance_xy), float(params.max_distance_z)),
        dtype=np.float64,
    )
    for ref_id in ref_tp.label_ids.tolist():
        ref_geom = ref_tp.geometries[int(ref_id)]
        if cand_tp.label_ids.size == 0:
            continue
        in_box = np.all(np.abs(cand_tp.centroids - ref_geom.centroid[None, :]) <= radius[None, :], axis=1)
        candidate_ids = cand_tp.label_ids[in_box]
        distances = np.array(
            [
                anisotropic_distance(ref_geom.centroid, cand_tp.geometries[int(cand_id)].centroid, params.z_distance_weight)
                for cand_id in candidate_ids.tolist()
            ],
            dtype=np.float32,
        )
        order = np.argsort(distances, kind="mergesort")
        true_row_index = row_by_frame_label.get((cand_frame, int(ref_id)))
        for ordered_index in order.tolist():
            cand_id = int(candidate_ids[ordered_index])
            cand_geom = cand_tp.geometries[cand_id]
            dice, ref_coords, cand_coords, overlap_count = compute_alignment_overlap(ref_geom, cand_geom)
            entries.append(
                CandidateEntry(
                    ref_row_index=row_by_frame_label[(ref_frame, int(ref_id))],
                    true_row_index=true_row_index,
                    cand_row_index=row_by_frame_label[(cand_frame, cand_id)],
                    ref_track_id=int(ref_id),
                    cand_track_id=cand_id,
                    distance=float(distances[ordered_index]),
                    dice=float(dice),
                    overlap_voxels=int(overlap_count),
                    ref_coords=ref_coords,
                    cand_coords=cand_coords,
                )
            )
    return entries


def metric_entry_key(entry: CandidateEntry) -> tuple[int, int]:
    return (int(entry.ref_row_index), int(entry.cand_row_index))


def append_metric_entry(
    entry: CandidateEntry,
    metric_entries: list[CandidateEntry],
    metric_index_by_key: dict[tuple[int, int], int],
) -> int:
    key = metric_entry_key(entry)
    existing = metric_index_by_key.get(key)
    if existing is not None:
        return existing
    metric_index = len(metric_entries)
    metric_entries.append(entry)
    metric_index_by_key[key] = metric_index
    return metric_index


def clear_entry_coords(entries: Sequence[CandidateEntry]) -> None:
    for entry in entries:
        entry.ref_coords = EMPTY_COORDS
        entry.cand_coords = EMPTY_COORDS


def build_pair_evaluation_plan(
    timepoints: Sequence[EvalTimepoint],
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    gap: int,
    ref_frame: int,
    params: EvaluationParams,
    n_features: int,
) -> PairEvaluationPlan:
    cand_frame = ref_frame + gap
    true_entries = build_true_gap_entries_for_pair(
        timepoints[ref_frame],
        timepoints[cand_frame],
        ref_frame=ref_frame,
        gap=gap,
        row_by_frame_label=row_by_frame_label,
        params=params,
    )
    candidate_entries = (
        compute_candidate_entries(
            timepoints[ref_frame],
            timepoints[cand_frame],
            ref_frame,
            cand_frame,
            row_by_frame_label,
            params=params,
        )
        if needs_candidate_diagnostics(params)
        else []
    )

    metric_entries: list[CandidateEntry] = []
    metric_index_by_key: dict[tuple[int, int], int] = {}
    candidate_metric_indices = [
        append_metric_entry(entry, metric_entries, metric_index_by_key)
        for entry in candidate_entries
    ]
    true_metric_indices = [
        append_metric_entry(entry, metric_entries, metric_index_by_key)
        for entry in true_entries
    ]

    ref_indices, cand_indices, group_indices, counts = build_pair_index_table(
        timepoints[ref_frame],
        timepoints[cand_frame],
        metric_entries,
    )
    metric_count = len(metric_entries)
    corr = np.full((metric_count, n_features), np.nan, dtype=np.float32)
    mse = np.full((metric_count, n_features), np.nan, dtype=np.float32)

    clear_entry_coords(true_entries)
    clear_entry_coords(candidate_entries)

    return PairEvaluationPlan(
        gap=int(gap),
        ref_frame=int(ref_frame),
        cand_frame=int(cand_frame),
        true_entries=true_entries,
        true_metric_indices=true_metric_indices,
        candidate_entries=candidate_entries,
        candidate_metric_indices=candidate_metric_indices,
        metric_count=metric_count,
        ref_indices=ref_indices,
        cand_indices=cand_indices,
        group_indices=group_indices,
        counts=counts,
        corr=corr,
        mse=mse,
    )


def build_pair_evaluation_plans(
    timepoints: Sequence[EvalTimepoint],
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    params: EvaluationParams,
    n_features: int,
) -> list[PairEvaluationPlan]:
    plans: list[PairEvaluationPlan] = []
    for gap in range(1, params.max_gap + 1):
        max_ref_frame = len(timepoints) - gap
        if max_ref_frame <= 0:
            continue
        print(
            f"[eval_GT_tracks] Building pair plans for gap {gap}/{params.max_gap}",
            flush=True,
        )
        if params.num_workers <= 1 or max_ref_frame <= 1:
            for ref_frame in range(max_ref_frame):
                plans.append(
                    build_pair_evaluation_plan(
                        timepoints,
                        row_by_frame_label,
                        gap=gap,
                        ref_frame=ref_frame,
                        params=params,
                        n_features=n_features,
                    )
                )
            continue

        with ThreadPoolExecutor(max_workers=int(params.num_workers)) as executor:
            futures = [
                executor.submit(
                    build_pair_evaluation_plan,
                    timepoints,
                    row_by_frame_label,
                    gap=gap,
                    ref_frame=ref_frame,
                    params=params,
                    n_features=n_features,
                )
                for ref_frame in range(max_ref_frame)
            ]
            plans.extend(future.result() for future in futures)
    return plans


def compute_feature_means_and_metrics_for_plans(
    timepoints: Sequence[EvalTimepoint],
    plans: Sequence[PairEvaluationPlan],
    row_by_frame_label: dict[tuple[int, int], int],
    feature_means: Any,
    *,
    params: EvaluationParams,
    n_features: int,
) -> None:
    metric_pairs = sum(plan.metric_count for plan in plans)
    metric_coords = sum(int(plan.ref_indices.shape[0]) for plan in plans)
    print(
        f"[eval_GT_tracks] Computing feature metrics for {metric_pairs} planned pairs "
        f"over {metric_coords} aligned voxels",
        flush=True,
    )
    batch_size = choose_global_pair_feature_batch_size(timepoints, n_features)
    for start in range(0, n_features, batch_size):
        end = min(n_features, start + batch_size)
        cache = ForegroundFeatureCache(
            timepoints,
            start=start,
            end=end,
            max_bytes=params.feature_cache_bytes,
        )
        print(
            f"[eval_GT_tracks] Feature block {start}:{end} with up to {cache.max_items} cached foreground sample(s)",
            flush=True,
        )
        for frame_index, timepoint in enumerate(timepoints):
            sampled = cache.get(frame_index)
            write_feature_mean_block_from_samples(
                timepoint,
                frame_index,
                row_by_frame_label,
                feature_means,
                sampled,
                start=start,
                end=end,
            )
        for plan in plans:
            if plan.metric_count == 0 or plan.ref_indices.size == 0:
                continue
            ref_samples = cache.get(plan.ref_frame)
            cand_samples = cache.get(plan.cand_frame)
            corr_block, mse_block = grouped_pair_feature_metrics_from_samples(
                ref_samples,
                cand_samples,
                plan.ref_indices,
                plan.cand_indices,
                counts=plan.counts,
                metric_count=plan.metric_count,
            )
            plan.corr[:, start:end] = corr_block
            plan.mse[:, start:end] = mse_block


def apply_true_gap_plans(
    plans: Sequence[PairEvaluationPlan],
    rows: list[dict[str, Any]],
    gap_corr: Any,
    gap_mse: Any,
    *,
    params: EvaluationParams,
) -> None:
    for plan in plans:
        gap = int(plan.gap)
        for entry, metric_index in zip(plan.true_entries, plan.true_metric_indices):
            corr = plan.corr[int(metric_index)]
            mse = plan.mse[int(metric_index)]
            row_index = int(entry.ref_row_index)
            row = rows[row_index]
            row[f"target_row_index_gap_{gap}"] = int(entry.cand_row_index)
            row[f"dice_gap_{gap}"] = float(entry.dice)
            row[f"anisotropic_distance_gap_{gap}"] = float(entry.distance)
            row[f"overlap_voxels_gap_{gap}"] = int(entry.overlap_voxels)
            gap_corr[row_index, gap - 1, :] = corr
            gap_mse[row_index, gap - 1, :] = mse
            row[f"feat_corr_mean_gap_{gap}"] = finite_mean(corr)
            row[f"feat_corr_median_gap_{gap}"] = finite_median(corr)
            row[f"feat_mse_mean_gap_{gap}"] = finite_mean(mse)
            row[f"feat_mse_median_gap_{gap}"] = finite_median(mse)
            good = (
                np.isfinite(corr)
                & np.isfinite(mse)
                & math.isfinite(float(entry.dice))
                & (float(entry.dice) > params.dice_threshold)
                & (corr > params.corr_threshold)
            )
            row[f"feat_pass_count_gap_{gap}"] = int(np.count_nonzero(good))


def fill_candidate_feature_metrics(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    entries: Sequence[CandidateEntry],
    *,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    corr = np.full((len(entries), n_features), np.nan, dtype=np.float32)
    mse = np.full((len(entries), n_features), np.nan, dtype=np.float32)
    if not entries:
        return corr, mse
    batch_size = choose_feature_batch_size(
        tuple(int(dim) for dim in ref_tp.lr_feats.shape[:3]),
        tuple(int(dim) for dim in cand_tp.lr_feats.shape[:3]),
        n_features=n_features,
    )
    usable_indices = [index for index, entry in enumerate(entries) if entry.overlap_voxels > 1]
    if not usable_indices:
        return corr, mse

    for start in range(0, n_features, batch_size):
        end = min(n_features, start + batch_size)
        ref_chunk = load_feature_chunk_internal_yxz(ref_tp.lr_feats, start=start, end=end)
        cand_chunk = load_feature_chunk_internal_yxz(cand_tp.lr_feats, start=start, end=end)
        for index in usable_indices:
            entry = entries[index]
            ref_values = sample_feature_chunk_at_internal_coords(ref_chunk, ref_tp.axis_maps, entry.ref_coords)
            cand_values = sample_feature_chunk_at_internal_coords(cand_chunk, cand_tp.axis_maps, entry.cand_coords)
            diff = ref_values - cand_values
            mse[index, start:end] = np.mean(diff * diff, axis=1, dtype=np.float32)
            corr[index, start:end] = rowwise_correlation(ref_values, cand_values)
    return corr, mse


def compute_vote_counts(
    entries: Sequence[CandidateEntry],
    corr: np.ndarray,
    mse: np.ndarray,
    *,
    params: EvaluationParams,
) -> list[int]:
    return compute_vote_counts_vectorized(entries, corr, mse, params=params).astype(int).tolist()


def empty_candidate_pair_columns() -> dict[str, Any]:
    fieldnames, schema_fields = candidate_field_schema()
    columns: dict[str, Any] = {}
    for name, kind in schema_fields:
        if kind == "int":
            columns[name] = np.empty((0,), dtype=np.int64)
        elif kind == "float":
            columns[name] = np.empty((0,), dtype=np.float64)
        elif kind == "bool":
            columns[name] = np.empty((0,), dtype=bool)
        else:
            columns[name] = []
    return {name: columns[name] for name in fieldnames}


def candidate_columns_from_metrics(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    *,
    ref_frame: int,
    gap: int,
    row_by_frame_label: dict[tuple[int, int], int],
    params: EvaluationParams,
    entries: Sequence[CandidateEntry],
    corr: np.ndarray,
    mse: np.ndarray,
    include_pair_rows: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cand_frame = ref_frame + gap
    n_entries = len(entries)
    ref_label_ids = [int(label_id) for label_id in ref_tp.label_ids.tolist()]
    summary_count = len(ref_label_ids)
    summary_ref_rows = np.array(
        [row_by_frame_label[(ref_frame, label_id)] for label_id in ref_label_ids],
        dtype=np.int64,
    )
    summary_true_rows = nullable_int_column(
        [row_by_frame_label.get((cand_frame, label_id)) for label_id in ref_label_ids]
    )
    summary_ref_track_ids = np.array(ref_label_ids, dtype=np.int64)
    summary_candidate_count = np.zeros((summary_count,), dtype=np.int64)
    summary_true_present = np.zeros((summary_count,), dtype=bool)
    summary_rank_distance = np.zeros((summary_count,), dtype=np.int64)
    summary_rank_dice = np.zeros((summary_count,), dtype=np.int64)
    summary_rank_corr_mean = np.zeros((summary_count,), dtype=np.int64)
    summary_rank_mse_mean = np.zeros((summary_count,), dtype=np.int64)
    summary_rank_vote_count = np.zeros((summary_count,), dtype=np.int64)
    summary_rank_distance_mask = np.ones((summary_count,), dtype=bool)
    summary_rank_dice_mask = np.ones((summary_count,), dtype=bool)
    summary_rank_corr_mean_mask = np.ones((summary_count,), dtype=bool)
    summary_rank_mse_mean_mask = np.ones((summary_count,), dtype=bool)
    summary_rank_vote_count_mask = np.ones((summary_count,), dtype=bool)
    summary_distance_margin = np.full((summary_count,), np.nan, dtype=np.float64)
    summary_dice_margin = np.full((summary_count,), np.nan, dtype=np.float64)
    summary_corr_mean_margin = np.full((summary_count,), np.nan, dtype=np.float64)
    summary_mse_mean_margin = np.full((summary_count,), np.nan, dtype=np.float64)
    summary_vote_count_margin = np.full((summary_count,), np.nan, dtype=np.float64)
    summary_index_by_ref_row = {int(row): index for index, row in enumerate(summary_ref_rows.tolist())}

    if n_entries == 0:
        summary_columns = {
            "ref_row_index": summary_ref_rows,
            "true_row_index": summary_true_rows,
            "ref_track_id": summary_ref_track_ids,
            "ref_frame": np.full((summary_count,), int(ref_frame + 1), dtype=np.int64),
            "cand_frame": np.full((summary_count,), int(cand_frame + 1), dtype=np.int64),
            "gap": np.full((summary_count,), int(gap), dtype=np.int64),
            "ref_timepoint_name": [ref_tp.name] * summary_count,
            "cand_timepoint_name": [cand_tp.name] * summary_count,
            "candidate_count": summary_candidate_count,
            "true_candidate_present": summary_true_present,
            "true_rank_distance": (summary_rank_distance, summary_rank_distance_mask),
            "true_rank_dice": (summary_rank_dice, summary_rank_dice_mask),
            "true_rank_corr_mean": (summary_rank_corr_mean, summary_rank_corr_mean_mask),
            "true_rank_mse_mean": (summary_rank_mse_mean, summary_rank_mse_mean_mask),
            "true_rank_vote_count": (summary_rank_vote_count, summary_rank_vote_count_mask),
            "true_distance_margin": summary_distance_margin,
            "true_dice_margin": summary_dice_margin,
            "true_corr_mean_margin": summary_corr_mean_margin,
            "true_mse_mean_margin": summary_mse_mean_margin,
            "true_vote_count_margin": summary_vote_count_margin,
        }
        return empty_candidate_pair_columns() if include_pair_rows else None, summary_columns

    ref_row_indices = np.array([entry.ref_row_index for entry in entries], dtype=np.int64)
    true_row_indices = nullable_int_column([entry.true_row_index for entry in entries])
    cand_row_indices = np.array([entry.cand_row_index for entry in entries], dtype=np.int64)
    ref_track_ids = np.array([entry.ref_track_id for entry in entries], dtype=np.int64)
    cand_track_ids = np.array([entry.cand_track_id for entry in entries], dtype=np.int64)
    distance = np.array([entry.distance for entry in entries], dtype=np.float64)
    dice = np.array([entry.dice for entry in entries], dtype=np.float64)
    overlap_voxels = np.array([entry.overlap_voxels for entry in entries], dtype=np.int64)
    corr_mean = finite_row_mean(corr)
    corr_median = finite_row_median(corr)
    mse_mean = finite_row_mean(mse)
    mse_median = finite_row_median(mse)
    vote_counts = compute_vote_counts_vectorized(
        entries,
        corr,
        mse,
        params=params,
        ref_row_indices=ref_row_indices,
        dice_values=dice,
    )

    rank_distance = np.zeros((n_entries,), dtype=np.int64)
    rank_dice = np.zeros((n_entries,), dtype=np.int64)
    rank_corr_mean = np.zeros((n_entries,), dtype=np.int64)
    rank_mse_mean = np.zeros((n_entries,), dtype=np.int64)
    rank_vote_count = np.zeros((n_entries,), dtype=np.int64)
    rank_distance_mask = np.ones((n_entries,), dtype=bool)
    rank_dice_mask = np.ones((n_entries,), dtype=bool)
    rank_corr_mean_mask = np.ones((n_entries,), dtype=bool)
    rank_mse_mean_mask = np.ones((n_entries,), dtype=bool)
    rank_vote_count_mask = np.ones((n_entries,), dtype=bool)

    true_present = np.zeros((n_entries,), dtype=bool)
    true_rank_distance = np.zeros((n_entries,), dtype=np.int64)
    true_rank_dice = np.zeros((n_entries,), dtype=np.int64)
    true_rank_corr_mean = np.zeros((n_entries,), dtype=np.int64)
    true_rank_mse_mean = np.zeros((n_entries,), dtype=np.int64)
    true_rank_vote_count = np.zeros((n_entries,), dtype=np.int64)
    true_rank_distance_mask = np.ones((n_entries,), dtype=bool)
    true_rank_dice_mask = np.ones((n_entries,), dtype=bool)
    true_rank_corr_mean_mask = np.ones((n_entries,), dtype=bool)
    true_rank_mse_mean_mask = np.ones((n_entries,), dtype=bool)
    true_rank_vote_count_mask = np.ones((n_entries,), dtype=bool)
    true_distance_margin = np.full((n_entries,), np.nan, dtype=np.float64)
    true_dice_margin = np.full((n_entries,), np.nan, dtype=np.float64)
    true_corr_mean_margin = np.full((n_entries,), np.nan, dtype=np.float64)
    true_mse_mean_margin = np.full((n_entries,), np.nan, dtype=np.float64)
    true_vote_count_margin = np.full((n_entries,), np.nan, dtype=np.float64)

    all_indices = np.arange(n_entries, dtype=np.int64)
    for start, end in candidate_group_ranges(ref_row_indices):
        group = all_indices[start:end]
        fill_rank_column(rank_distance, rank_distance_mask, group, distance[start:end], higher_better=False)
        fill_rank_column(rank_dice, rank_dice_mask, group, dice[start:end], higher_better=True)
        fill_rank_column(rank_corr_mean, rank_corr_mean_mask, group, corr_mean[start:end], higher_better=True)
        fill_rank_column(rank_mse_mean, rank_mse_mean_mask, group, mse_mean[start:end], higher_better=False)
        fill_rank_column(
            rank_vote_count,
            rank_vote_count_mask,
            group,
            vote_counts[start:end].astype(np.float64, copy=False),
            higher_better=True,
        )

        ref_row_index = int(ref_row_indices[start])
        summary_index = summary_index_by_ref_row.get(ref_row_index)
        if summary_index is not None:
            summary_candidate_count[summary_index] = int(end - start)

        local_true = np.flatnonzero(cand_track_ids[start:end] == ref_track_ids[start:end])
        if local_true.size == 0:
            continue
        true_index = int(start + local_true[0])
        false = np.concatenate((all_indices[start:true_index], all_indices[true_index + 1 : end]))
        true_present[start:end] = True
        true_distance_margin[start:end] = false_margin_from_values(
            distance[true_index],
            distance[false],
            higher_better=False,
        )
        true_dice_margin[start:end] = false_margin_from_values(
            dice[true_index],
            dice[false],
            higher_better=True,
        )
        true_corr_mean_margin[start:end] = false_margin_from_values(
            corr_mean[true_index],
            corr_mean[false],
            higher_better=True,
        )
        true_mse_mean_margin[start:end] = false_margin_from_values(
            mse_mean[true_index],
            mse_mean[false],
            higher_better=False,
        )
        true_vote_count_margin[start:end] = false_margin_from_values(
            float(vote_counts[true_index]),
            vote_counts[false].astype(np.float64, copy=False),
            higher_better=True,
        )

        for source, source_mask, dest, dest_mask in (
            (rank_distance, rank_distance_mask, true_rank_distance, true_rank_distance_mask),
            (rank_dice, rank_dice_mask, true_rank_dice, true_rank_dice_mask),
            (rank_corr_mean, rank_corr_mean_mask, true_rank_corr_mean, true_rank_corr_mean_mask),
            (rank_mse_mean, rank_mse_mean_mask, true_rank_mse_mean, true_rank_mse_mean_mask),
            (rank_vote_count, rank_vote_count_mask, true_rank_vote_count, true_rank_vote_count_mask),
        ):
            if not bool(source_mask[true_index]):
                dest[start:end] = int(source[true_index])
                dest_mask[start:end] = False

        if summary_index is not None:
            summary_true_present[summary_index] = True
            summary_distance_margin[summary_index] = true_distance_margin[true_index]
            summary_dice_margin[summary_index] = true_dice_margin[true_index]
            summary_corr_mean_margin[summary_index] = true_corr_mean_margin[true_index]
            summary_mse_mean_margin[summary_index] = true_mse_mean_margin[true_index]
            summary_vote_count_margin[summary_index] = true_vote_count_margin[true_index]
            for source, source_mask, dest, dest_mask in (
                (rank_distance, rank_distance_mask, summary_rank_distance, summary_rank_distance_mask),
                (rank_dice, rank_dice_mask, summary_rank_dice, summary_rank_dice_mask),
                (rank_corr_mean, rank_corr_mean_mask, summary_rank_corr_mean, summary_rank_corr_mean_mask),
                (rank_mse_mean, rank_mse_mean_mask, summary_rank_mse_mean, summary_rank_mse_mean_mask),
                (rank_vote_count, rank_vote_count_mask, summary_rank_vote_count, summary_rank_vote_count_mask),
            ):
                if not bool(source_mask[true_index]):
                    dest[summary_index] = int(source[true_index])
                    dest_mask[summary_index] = False

    pair_columns = None
    if include_pair_rows:
        pair_columns = {
            "ref_row_index": ref_row_indices,
            "true_row_index": true_row_indices,
            "cand_row_index": cand_row_indices,
            "ref_track_id": ref_track_ids,
            "cand_track_id": cand_track_ids,
            "ref_frame": np.full((n_entries,), int(ref_frame + 1), dtype=np.int64),
            "cand_frame": np.full((n_entries,), int(cand_frame + 1), dtype=np.int64),
            "gap": np.full((n_entries,), int(gap), dtype=np.int64),
            "ref_timepoint_name": [ref_tp.name] * n_entries,
            "cand_timepoint_name": [cand_tp.name] * n_entries,
            "is_true_match": cand_track_ids == ref_track_ids,
            "anisotropic_distance": distance,
            "dice": dice,
            "overlap_voxels": overlap_voxels,
            "feat_corr_mean": corr_mean,
            "feat_corr_median": corr_median,
            "feat_mse_mean": mse_mean,
            "feat_mse_median": mse_median,
            "vote_count": vote_counts,
            "rank_distance": (rank_distance, rank_distance_mask),
            "rank_dice": (rank_dice, rank_dice_mask),
            "rank_corr_mean": (rank_corr_mean, rank_corr_mean_mask),
            "rank_mse_mean": (rank_mse_mean, rank_mse_mean_mask),
            "rank_vote_count": (rank_vote_count, rank_vote_count_mask),
            "true_candidate_present": true_present,
            "true_rank_distance": (true_rank_distance, true_rank_distance_mask),
            "true_rank_dice": (true_rank_dice, true_rank_dice_mask),
            "true_rank_corr_mean": (true_rank_corr_mean, true_rank_corr_mean_mask),
            "true_rank_mse_mean": (true_rank_mse_mean, true_rank_mse_mean_mask),
            "true_rank_vote_count": (true_rank_vote_count, true_rank_vote_count_mask),
            "true_distance_margin": true_distance_margin,
            "true_dice_margin": true_dice_margin,
            "true_corr_mean_margin": true_corr_mean_margin,
            "true_mse_mean_margin": true_mse_mean_margin,
            "true_vote_count_margin": true_vote_count_margin,
        }

    summary_columns = {
        "ref_row_index": summary_ref_rows,
        "true_row_index": summary_true_rows,
        "ref_track_id": summary_ref_track_ids,
        "ref_frame": np.full((summary_count,), int(ref_frame + 1), dtype=np.int64),
        "cand_frame": np.full((summary_count,), int(cand_frame + 1), dtype=np.int64),
        "gap": np.full((summary_count,), int(gap), dtype=np.int64),
        "ref_timepoint_name": [ref_tp.name] * summary_count,
        "cand_timepoint_name": [cand_tp.name] * summary_count,
        "candidate_count": summary_candidate_count,
        "true_candidate_present": summary_true_present,
        "true_rank_distance": (summary_rank_distance, summary_rank_distance_mask),
        "true_rank_dice": (summary_rank_dice, summary_rank_dice_mask),
        "true_rank_corr_mean": (summary_rank_corr_mean, summary_rank_corr_mean_mask),
        "true_rank_mse_mean": (summary_rank_mse_mean, summary_rank_mse_mean_mask),
        "true_rank_vote_count": (summary_rank_vote_count, summary_rank_vote_count_mask),
        "true_distance_margin": summary_distance_margin,
        "true_dice_margin": summary_dice_margin,
        "true_corr_mean_margin": summary_corr_mean_margin,
        "true_mse_mean_margin": summary_mse_mean_margin,
        "true_vote_count_margin": summary_vote_count_margin,
    }
    return pair_columns, summary_columns


def candidate_rows_from_metrics(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    *,
    ref_frame: int,
    gap: int,
    row_by_frame_label: dict[tuple[int, int], int],
    params: EvaluationParams,
    entries: Sequence[CandidateEntry],
    corr: np.ndarray,
    mse: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cand_frame = ref_frame + gap
    if not entries:
        summary_rows = []
        for ref_id in ref_tp.label_ids.tolist():
            summary_rows.append(
                {
                    "ref_row_index": int(row_by_frame_label[(ref_frame, int(ref_id))]),
                    "true_row_index": row_by_frame_label.get((cand_frame, int(ref_id))),
                    "ref_track_id": int(ref_id),
                    "ref_frame": int(ref_frame + 1),
                    "cand_frame": int(cand_frame + 1),
                    "gap": int(gap),
                    "ref_timepoint_name": ref_tp.name,
                    "cand_timepoint_name": cand_tp.name,
                    "candidate_count": 0,
                    "true_candidate_present": False,
                    "true_rank_distance": None,
                    "true_rank_dice": None,
                    "true_rank_corr_mean": None,
                    "true_rank_mse_mean": None,
                    "true_rank_vote_count": None,
                    "true_distance_margin": None,
                    "true_dice_margin": None,
                    "true_corr_mean_margin": None,
                    "true_mse_mean_margin": None,
                    "true_vote_count_margin": None,
                }
            )
        return [], summary_rows
    corr_mean = [finite_mean(corr[index]) for index in range(len(entries))]
    corr_median = [finite_median(corr[index]) for index in range(len(entries))]
    mse_mean = [finite_mean(mse[index]) for index in range(len(entries))]
    mse_median = [finite_median(mse[index]) for index in range(len(entries))]
    vote_counts = compute_vote_counts(entries, corr, mse, params=params)

    rows: list[dict[str, Any]] = []
    by_ref: dict[int, list[int]] = {}
    for index, entry in enumerate(entries):
        by_ref.setdefault(entry.ref_row_index, []).append(index)

    rank_distance: list[int | None] = [None] * len(entries)
    rank_dice: list[int | None] = [None] * len(entries)
    rank_corr_mean: list[int | None] = [None] * len(entries)
    rank_mse_mean: list[int | None] = [None] * len(entries)
    rank_vote_count: list[int | None] = [None] * len(entries)
    true_stats_by_ref: dict[int, dict[str, Any]] = {}

    for ref_row_index, indices in by_ref.items():
        distance_ranks = rank_metric([entries[index].distance for index in indices], higher_better=False)
        dice_ranks = rank_metric([entries[index].dice for index in indices], higher_better=True)
        corr_ranks = rank_metric([corr_mean[index] for index in indices], higher_better=True)
        mse_ranks = rank_metric([mse_mean[index] for index in indices], higher_better=False)
        vote_ranks = rank_metric([vote_counts[index] for index in indices], higher_better=True)
        for local_index, entry_index in enumerate(indices):
            rank_distance[entry_index] = distance_ranks[local_index]
            rank_dice[entry_index] = dice_ranks[local_index]
            rank_corr_mean[entry_index] = corr_ranks[local_index]
            rank_mse_mean[entry_index] = mse_ranks[local_index]
            rank_vote_count[entry_index] = vote_ranks[local_index]

        true_indices = [index for index in indices if entries[index].cand_track_id == entries[index].ref_track_id]
        if not true_indices:
            true_stats_by_ref[ref_row_index] = {"present": False}
            continue
        true_index = true_indices[0]
        false_indices = [index for index in indices if index != true_index]
        true_stats_by_ref[ref_row_index] = {
            "present": True,
            "rank_distance": rank_distance[true_index],
            "rank_dice": rank_dice[true_index],
            "rank_corr_mean": rank_corr_mean[true_index],
            "rank_mse_mean": rank_mse_mean[true_index],
            "rank_vote_count": rank_vote_count[true_index],
            "distance_margin": best_false_margin(
                entries[true_index].distance,
                [entries[index].distance for index in false_indices],
                higher_better=False,
            ),
            "dice_margin": best_false_margin(
                entries[true_index].dice,
                [entries[index].dice for index in false_indices],
                higher_better=True,
            ),
            "corr_mean_margin": best_false_margin(
                corr_mean[true_index],
                [corr_mean[index] for index in false_indices],
                higher_better=True,
            ),
            "mse_mean_margin": best_false_margin(
                mse_mean[true_index],
                [mse_mean[index] for index in false_indices],
                higher_better=False,
            ),
            "vote_count_margin": best_false_margin(
                vote_counts[true_index],
                [vote_counts[index] for index in false_indices],
                higher_better=True,
            ),
        }

    for index, entry in enumerate(entries):
        true_stats = true_stats_by_ref[entry.ref_row_index]
        rows.append(
            {
                "ref_row_index": int(entry.ref_row_index),
                "true_row_index": entry.true_row_index,
                "cand_row_index": int(entry.cand_row_index),
                "ref_track_id": int(entry.ref_track_id),
                "cand_track_id": int(entry.cand_track_id),
                "ref_frame": int(ref_frame + 1),
                "cand_frame": int(cand_frame + 1),
                "gap": int(gap),
                "ref_timepoint_name": ref_tp.name,
                "cand_timepoint_name": cand_tp.name,
                "is_true_match": bool(entry.cand_track_id == entry.ref_track_id),
                "anisotropic_distance": float(entry.distance),
                "dice": float(entry.dice),
                "overlap_voxels": int(entry.overlap_voxels),
                "feat_corr_mean": corr_mean[index],
                "feat_corr_median": corr_median[index],
                "feat_mse_mean": mse_mean[index],
                "feat_mse_median": mse_median[index],
                "vote_count": int(vote_counts[index]),
                "rank_distance": rank_distance[index],
                "rank_dice": rank_dice[index],
                "rank_corr_mean": rank_corr_mean[index],
                "rank_mse_mean": rank_mse_mean[index],
                "rank_vote_count": rank_vote_count[index],
                "true_candidate_present": bool(true_stats.get("present", False)),
                "true_rank_distance": true_stats.get("rank_distance"),
                "true_rank_dice": true_stats.get("rank_dice"),
                "true_rank_corr_mean": true_stats.get("rank_corr_mean"),
                "true_rank_mse_mean": true_stats.get("rank_mse_mean"),
                "true_rank_vote_count": true_stats.get("rank_vote_count"),
                "true_distance_margin": true_stats.get("distance_margin"),
                "true_dice_margin": true_stats.get("dice_margin"),
                "true_corr_mean_margin": true_stats.get("corr_mean_margin"),
                "true_mse_mean_margin": true_stats.get("mse_mean_margin"),
                "true_vote_count_margin": true_stats.get("vote_count_margin"),
            }
        )
    summary_rows: list[dict[str, Any]] = []
    for ref_id in ref_tp.label_ids.tolist():
        ref_row_index = row_by_frame_label[(ref_frame, int(ref_id))]
        true_stats = true_stats_by_ref.get(ref_row_index, {"present": False})
        summary_rows.append(
            {
                "ref_row_index": int(ref_row_index),
                "true_row_index": row_by_frame_label.get((cand_frame, int(ref_id))),
                "ref_track_id": int(ref_id),
                "ref_frame": int(ref_frame + 1),
                "cand_frame": int(cand_frame + 1),
                "gap": int(gap),
                "ref_timepoint_name": ref_tp.name,
                "cand_timepoint_name": cand_tp.name,
                "candidate_count": int(len(by_ref.get(ref_row_index, []))),
                "true_candidate_present": bool(true_stats.get("present", False)),
                "true_rank_distance": true_stats.get("rank_distance"),
                "true_rank_dice": true_stats.get("rank_dice"),
                "true_rank_corr_mean": true_stats.get("rank_corr_mean"),
                "true_rank_mse_mean": true_stats.get("rank_mse_mean"),
                "true_rank_vote_count": true_stats.get("rank_vote_count"),
                "true_distance_margin": true_stats.get("distance_margin"),
                "true_dice_margin": true_stats.get("dice_margin"),
                "true_corr_mean_margin": true_stats.get("corr_mean_margin"),
                "true_mse_mean_margin": true_stats.get("mse_mean_margin"),
                "true_vote_count_margin": true_stats.get("vote_count_margin"),
            }
        )
    return rows, summary_rows


def candidate_rows_for_pair(
    ref_tp: EvalTimepoint,
    cand_tp: EvalTimepoint,
    *,
    ref_frame: int,
    gap: int,
    row_by_frame_label: dict[tuple[int, int], int],
    params: EvaluationParams,
    n_features: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cand_frame = ref_frame + gap
    entries = compute_candidate_entries(
        ref_tp,
        cand_tp,
        ref_frame,
        cand_frame,
        row_by_frame_label,
        params=params,
    )
    corr, mse = compute_pair_feature_metrics(ref_tp, cand_tp, entries, n_features=n_features)
    return candidate_rows_from_metrics(
        ref_tp,
        cand_tp,
        ref_frame=ref_frame,
        gap=gap,
        row_by_frame_label=row_by_frame_label,
        params=params,
        entries=entries,
        corr=corr,
        mse=mse,
    )


def compute_candidate_rows_for_frame(
    timepoints: Sequence[EvalTimepoint],
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    gap: int,
    ref_frame: int,
    params: EvaluationParams,
    n_features: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return candidate_rows_for_pair(
        timepoints[ref_frame],
        timepoints[ref_frame + gap],
        ref_frame=ref_frame,
        gap=gap,
        row_by_frame_label=row_by_frame_label,
        params=params,
        n_features=n_features,
    )


def iter_candidate_pair_results(
    timepoints: Sequence[EvalTimepoint],
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    gap: int,
    params: EvaluationParams,
    n_features: int,
) -> Iterable[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    max_ref_frame = len(timepoints) - gap
    if params.num_workers <= 1 or max_ref_frame <= 1:
        for ref_frame in range(max_ref_frame):
            yield compute_candidate_rows_for_frame(
                timepoints,
                row_by_frame_label,
                gap=gap,
                ref_frame=ref_frame,
                params=params,
                n_features=n_features,
            )
        return

    next_frame = 0
    futures: set[Future[tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = set()
    with ThreadPoolExecutor(max_workers=int(params.num_workers)) as executor:
        while next_frame < max_ref_frame and len(futures) < int(params.num_workers):
            futures.add(
                executor.submit(
                    compute_candidate_rows_for_frame,
                    timepoints,
                    row_by_frame_label,
                    gap=gap,
                    ref_frame=next_frame,
                    params=params,
                    n_features=n_features,
                )
            )
            next_frame += 1

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                if next_frame < max_ref_frame:
                    futures.add(
                        executor.submit(
                            compute_candidate_rows_for_frame,
                            timepoints,
                            row_by_frame_label,
                            gap=gap,
                            ref_frame=next_frame,
                            params=params,
                            n_features=n_features,
                        )
                    )
                    next_frame += 1


def write_candidate_pairs(
    timepoints: Sequence[EvalTimepoint],
    output_path: Path,
    row_by_frame_label: dict[tuple[int, int], int],
    *,
    params: EvaluationParams,
    n_features: int,
) -> tuple[Path, Path] | None:
    if not params.include_candidate_pairs:
        return None
    fieldnames, schema_fields = candidate_field_schema()
    sink = TableSink(
        output_path / "candidate_pairs",
        fieldnames=fieldnames,
        schema_fields=schema_fields,
        no_compression=params.no_compression,
    )
    summary_fieldnames, summary_schema_fields = candidate_summary_field_schema()
    summary_sink = TableSink(
        output_path / "candidate_ref_summary",
        fieldnames=summary_fieldnames,
        schema_fields=summary_schema_fields,
        no_compression=params.no_compression,
    )
    try:
        batch: list[dict[str, Any]] = []
        summary_batch: list[dict[str, Any]] = []
        for gap in range(1, params.max_gap + 1):
            print(
                f"[eval_GT_tracks] Computing candidate diagnostics for gap {gap}/{params.max_gap} "
                f"with {params.num_workers} worker(s)",
                flush=True,
            )
            for pair_rows, summary_rows in iter_candidate_pair_results(
                timepoints,
                row_by_frame_label,
                gap=gap,
                params=params,
                n_features=n_features,
            ):
                batch.extend(pair_rows)
                summary_batch.extend(summary_rows)
                if len(batch) >= params.candidate_batch_rows:
                    sink.write_rows(batch)
                    batch.clear()
                if len(summary_batch) >= params.candidate_batch_rows:
                    summary_sink.write_rows(summary_batch)
                    summary_batch.clear()
        sink.write_rows(batch)
        summary_sink.write_rows(summary_batch)
    finally:
        sink.close()
        summary_sink.close()
    return sink.path, summary_sink.path


def write_candidate_pairs_from_plans(
    timepoints: Sequence[EvalTimepoint],
    output_path: Path,
    row_by_frame_label: dict[tuple[int, int], int],
    plans: Sequence[PairEvaluationPlan],
    *,
    params: EvaluationParams,
) -> tuple[Path | None, Path | None] | None:
    mode = candidate_output_mode(params)
    if mode == "none":
        return None
    include_pair_rows = mode == "full"
    fieldnames: list[str] = []
    sink: TableSink | None = None
    if include_pair_rows:
        fieldnames, schema_fields = candidate_field_schema()
        sink = TableSink(
            output_path / "candidate_pairs",
            fieldnames=fieldnames,
            schema_fields=schema_fields,
            no_compression=params.no_compression,
        )
    summary_fieldnames, summary_schema_fields = candidate_summary_field_schema()
    summary_sink = TableSink(
        output_path / "candidate_ref_summary",
        fieldnames=summary_fieldnames,
        schema_fields=summary_schema_fields,
        no_compression=params.no_compression,
    )
    try:
        print(
            f"[eval_GT_tracks] Writing candidate diagnostics ({mode}) "
            f"in batches of {params.candidate_batch_rows} row(s)",
            flush=True,
        )
        batch: list[Mapping[str, Any]] = []
        summary_batch: list[Mapping[str, Any]] = []
        batch_rows = 0
        summary_batch_rows = 0
        for plan_index, plan in enumerate(plans, start=1):
            metric_indices = np.asarray(plan.candidate_metric_indices, dtype=np.int64)
            corr = plan.corr[metric_indices] if metric_indices.size else np.empty((0, plan.corr.shape[1]), dtype=np.float32)
            mse = plan.mse[metric_indices] if metric_indices.size else np.empty((0, plan.mse.shape[1]), dtype=np.float32)
            pair_columns, summary_columns = candidate_columns_from_metrics(
                timepoints[plan.ref_frame],
                timepoints[plan.cand_frame],
                ref_frame=plan.ref_frame,
                gap=plan.gap,
                row_by_frame_label=row_by_frame_label,
                params=params,
                entries=plan.candidate_entries,
                corr=corr,
                mse=mse,
                include_pair_rows=include_pair_rows,
            )
            if pair_columns is not None and sink is not None:
                rows = _column_length(pair_columns[fieldnames[0]])
                if rows:
                    batch.append(pair_columns)
                    batch_rows += rows
                if batch_rows >= params.candidate_batch_rows:
                    flush_column_batches(sink, fieldnames, batch)
                    batch_rows = 0
            summary_rows = _column_length(summary_columns[summary_fieldnames[0]])
            if summary_rows:
                summary_batch.append(summary_columns)
                summary_batch_rows += summary_rows
            if summary_batch_rows >= params.candidate_batch_rows:
                flush_column_batches(summary_sink, summary_fieldnames, summary_batch)
                summary_batch_rows = 0
            if plan_index % 100 == 0:
                print(
                    f"[eval_GT_tracks] Candidate diagnostics processed {plan_index}/{len(plans)} pair plan(s)",
                    flush=True,
                )
        if sink is not None:
            flush_column_batches(sink, fieldnames, batch)
        flush_column_batches(summary_sink, summary_fieldnames, summary_batch)
    finally:
        if sink is not None:
            sink.close()
        summary_sink.close()
    return (sink.path if sink is not None else None), summary_sink.path


def write_gt_rows(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    params: EvaluationParams,
) -> Path:
    fieldnames, schema_fields = gt_field_schema(params.max_gap)
    sink = TableSink(
        output_path / "gt_rows",
        fieldnames=fieldnames,
        schema_fields=schema_fields,
        no_compression=params.no_compression,
    )
    try:
        sink.write_rows(rows)
    finally:
        sink.close()
    return sink.path


def round_float(value: Any, decimals: int) -> Any:
    if value is None:
        return CSV_NAN
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return CSV_NAN
        return round(value, decimals)
    return value


def export_wide_csv(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    feature_means: Any,
    gap_corr: Any,
    gap_mse: Any,
    *,
    params: EvaluationParams,
    n_features: int,
) -> Path:
    path = output_path / "gt_wide.csv"
    base_fields, _schema = gt_field_schema(params.max_gap)
    fields = list(base_fields)
    fields.extend([f"feat_{feature_index}_mean" for feature_index in range(n_features)])
    for gap in range(1, params.max_gap + 1):
        fields.extend([f"feat_{feature_index}_gap_{gap}_corr" for feature_index in range(n_features)])
        fields.extend([f"feat_{feature_index}_gap_{gap}_mse" for feature_index in range(n_features)])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row_index = int(row["row_index"])
            out = {field: round_float(row.get(field), params.csv_float_decimals) for field in base_fields}
            means = np.asarray(feature_means[row_index, :])
            for feature_index in range(n_features):
                out[f"feat_{feature_index}_mean"] = round_float(float(means[feature_index]), params.csv_float_decimals)
            for gap in range(1, params.max_gap + 1):
                corr = np.asarray(gap_corr[row_index, gap - 1, :])
                mse = np.asarray(gap_mse[row_index, gap - 1, :])
                for feature_index in range(n_features):
                    out[f"feat_{feature_index}_gap_{gap}_corr"] = round_float(
                        float(corr[feature_index]),
                        params.csv_float_decimals,
                    )
                for feature_index in range(n_features):
                    out[f"feat_{feature_index}_gap_{gap}_mse"] = round_float(
                        float(mse[feature_index]),
                        params.csv_float_decimals,
                    )
            writer.writerow(out)
    return path


def write_manifest(
    output_path: Path,
    *,
    input_path: Path,
    gt_segmentation_path: Path,
    rows_path: Path,
    candidate_outputs: tuple[Path | None, Path | None] | None,
    wide_csv_path: Path | None,
    params: EvaluationParams,
    n_features: int,
    n_rows: int,
    n_timepoints: int,
) -> Path:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "gt_segmentation_path": str(gt_segmentation_path),
        "n_timepoints": int(n_timepoints),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "max_gap": int(params.max_gap),
        "max_distance_xy": float(params.max_distance_xy),
        "max_distance_z": float(params.max_distance_z),
        "z_distance_weight": float(params.z_distance_weight),
        "dice_threshold": float(params.dice_threshold),
        "corr_threshold": float(params.corr_threshold),
        "num_workers": int(params.num_workers),
        "feature_cache_bytes": int(params.feature_cache_bytes),
        "candidate_output": candidate_output_mode(params),
        "compression": None if params.no_compression else "lz4",
        "outputs": {
            "gt_rows": rows_path.name,
            "feature_means": "feature_means.zarr",
            "true_gap_corr": "true_gap_corr.zarr",
            "true_gap_mse": "true_gap_mse.zarr",
            "candidate_pairs": (
                candidate_outputs[0].name
                if candidate_outputs is not None and candidate_outputs[0] is not None
                else None
            ),
            "candidate_ref_summary": (
                candidate_outputs[1].name
                if candidate_outputs is not None and candidate_outputs[1] is not None
                else None
            ),
            "wide_csv": wide_csv_path.name if wide_csv_path is not None else None,
        },
        "array_shapes": {
            "feature_means": [int(n_rows), int(n_features)],
            "true_gap_corr": [int(n_rows), int(params.max_gap), int(n_features)],
            "true_gap_mse": [int(n_rows), int(params.max_gap), int(n_features)],
        },
        "notes": [
            "GT track_id is the persistent integer label in the GT segmentation volume.",
            "Per-feature true-gap arrays are stored in Zarr and indexed by gt_rows.row_index.",
            "Candidate-pair diagnostics store aggregate/rank metrics for spatial candidates, not per-feature arrays.",
        ],
    }
    path = output_path / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def infer_initial_feature_count(discovered: Sequence[tuple[Any, Path]]) -> int:
    if not discovered:
        raise PipelineError("No inference timepoints found.")
    first = np.load(discovered[0][0].lr_path, mmap_mode="r")
    if first.ndim != 4:
        raise PipelineError(f"{discovered[0][0].lr_path} must have shape [Z, Y, X, C].")
    return int(first.shape[-1])


def run_evaluation(
    input_path: str | Path,
    *,
    gt_segmentation_path: str | Path,
    output_path: str | Path,
    params: EvaluationParams,
) -> Path:
    validate_params(params)
    input_path = Path(input_path).expanduser().resolve()
    gt_segmentation_path = Path(gt_segmentation_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")
    if not gt_segmentation_path.is_dir():
        raise FileNotFoundError(f"GT segmentation folder does not exist: {gt_segmentation_path}")
    prepare_output_dir(output_path, overwrite=params.overwrite)

    discovered = discover_timepoints(input_path, gt_segmentation_path)
    discovered.sort(key=lambda item: natural_sort_key(item[0].name))
    if not discovered:
        raise PipelineError("No matching timepoints found.")
    initial_channels = infer_initial_feature_count(discovered)
    n_features = resolve_num_features(params.num_features, initial_channels)
    print(
        f"[eval_GT_tracks] Found {len(discovered)} timepoints; evaluating {n_features} feature channels",
        flush=True,
    )

    timepoints = prepare_timepoints(discovered, n_features=n_features)
    rows, row_by_frame_label = build_gt_rows(timepoints, max_gap=params.max_gap)
    print(f"[eval_GT_tracks] Built {len(rows)} GT object rows", flush=True)
    feature_means, gap_corr, gap_mse = create_metric_arrays(
        output_path,
        n_rows=len(rows),
        n_features=n_features,
        max_gap=params.max_gap,
        params=params,
    )
    pair_plans = build_pair_evaluation_plans(
        timepoints,
        row_by_frame_label,
        params=params,
        n_features=n_features,
    )
    release_foreground_index_maps(timepoints)
    compute_feature_means_and_metrics_for_plans(
        timepoints,
        pair_plans,
        row_by_frame_label,
        feature_means,
        params=params,
        n_features=n_features,
    )
    apply_true_gap_plans(
        pair_plans,
        rows,
        gap_corr,
        gap_mse,
        params=params,
    )
    write_true_gap_metric_arrays(output_path, gap_corr, gap_mse, params=params)
    gt_rows_path = write_gt_rows(rows, output_path, params=params)
    candidate_outputs = write_candidate_pairs_from_plans(
        timepoints,
        output_path,
        row_by_frame_label,
        pair_plans,
        params=params,
    )
    wide_csv_path = None
    if params.export_wide_csv:
        print("[eval_GT_tracks] Exporting wide rounded CSV", flush=True)
        wide_csv_path = export_wide_csv(
            rows,
            output_path,
            feature_means,
            gap_corr,
            gap_mse,
            params=params,
            n_features=n_features,
        )
    manifest_path = write_manifest(
        output_path,
        input_path=input_path,
        gt_segmentation_path=gt_segmentation_path,
        rows_path=gt_rows_path,
        candidate_outputs=candidate_outputs,
        wide_csv_path=wide_csv_path,
        params=params,
        n_features=n_features,
        n_rows=len(rows),
        n_timepoints=len(timepoints),
    )
    print(f"[eval_GT_tracks] Saved outputs under {output_path}", flush=True)
    print("[eval_GT_tracks] Done", flush=True)
    return manifest_path


def main() -> None:
    args = parse_args()
    run_evaluation(
        args.input_path,
        gt_segmentation_path=args.gt_segmentation_path,
        output_path=args.output_path,
        params=EvaluationParams(
            max_gap=int(args.max_gap),
            num_features=str(args.num_features),
            max_distance_xy=float(args.max_distance_xy),
            max_distance_z=float(args.max_distance_z),
            z_distance_weight=float(args.z_distance_weight),
            dice_threshold=float(args.dice_threshold),
            corr_threshold=float(args.corr_threshold),
            no_compression=bool(args.no_compression),
            export_wide_csv=bool(args.export_wide_csv),
            csv_float_decimals=int(args.csv_float_decimals),
            candidate_output="none" if bool(args.skip_candidate_pairs) else str(args.candidate_output),
            include_candidate_pairs=not bool(args.skip_candidate_pairs),
            overwrite=bool(args.overwrite),
            candidate_batch_rows=int(args.candidate_batch_rows),
            feature_cache_bytes=int(args.feature_cache_bytes),
            num_workers=int(args.num_workers),
        ),
    )


if __name__ == "__main__":
    main()
