from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import tifffile

DEFAULT_THRESHOLD = 0.05
DEFAULT_RAW_GLOB = "*.tif"
DEFAULT_SIGMA_Z_COLUMN = "s_1"
DEFAULT_SIGMA_XY_COLUMN = "s_2"


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    z: float
    label: int


@dataclass(frozen=True)
class FrameTask:
    frame: int
    raw_path: Path
    output_path: Path
    shape_zyx: tuple[int, int, int]
    detections: tuple[Detection, ...]
    sigma_z: float
    sigma_xy: float
    threshold: float
    dtype_name: str
    zero_based: bool
    compression: str | None


@dataclass(frozen=True)
class FrameResult:
    frame: int
    raw_path: Path
    output_path: Path
    detection_count: int
    nonzero_voxels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Detection3D detection CSV outputs into per-frame instance "
            "segmentation masks."
        )
    )
    parser.add_argument(
        "detections_csv",
        type=Path,
        help="Detection CSV produced by convert_detection3d_mat_to_csv.py.",
    )
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=None,
        help=(
            "Frame metadata CSV containing sigma columns. Defaults to replacing "
            "'_detections.csv' with '_frames.csv' next to detections_csv."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing one raw TIFF volume per frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for output instance masks. Output filenames exactly match "
            "the naturally sorted raw TIFF filenames."
        ),
    )
    parser.add_argument(
        "--raw-glob",
        default=DEFAULT_RAW_GLOB,
        help=f"Glob used to discover raw volumes. Defaults to {DEFAULT_RAW_GLOB!r}.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Relative Gaussian threshold for the ellipsoid footprint. "
            f"Defaults to {DEFAULT_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--sigma-z-column",
        default=DEFAULT_SIGMA_Z_COLUMN,
        help=(
            f"Frame CSV column used as Z sigma. Defaults to {DEFAULT_SIGMA_Z_COLUMN!r}."
        ),
    )
    parser.add_argument(
        "--sigma-xy-column",
        default=DEFAULT_SIGMA_XY_COLUMN,
        help=(
            "Frame CSV column used as both X and Y sigma. Defaults to "
            f"{DEFAULT_SIGMA_XY_COLUMN!r}."
        ),
    )
    parser.add_argument(
        "--sigma-z",
        type=float,
        default=None,
        help="Constant Z sigma override. Must be supplied with --sigma-xy.",
    )
    parser.add_argument(
        "--sigma-xy",
        type=float,
        default=None,
        help="Constant XY sigma override. Must be supplied with --sigma-z.",
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Treat CSV coordinates as zero-based. By default MATLAB one-based coordinates are converted to zero-based.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "uint16", "uint32"),
        default="auto",
        help="Output label dtype. Defaults to auto.",
    )
    parser.add_argument(
        "--compression",
        default="zlib",
        help="TIFF compression passed to tifffile. Use 'none' for uncompressed output. Defaults to zlib.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of frame-level worker processes. Defaults to min(8, cpu_count).",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower()) for part in parts
    )


def default_frames_csv_path(detections_csv: Path) -> Path:
    name = detections_csv.name
    if name.endswith("_detections.csv"):
        return detections_csv.with_name(
            f"{name.removesuffix('_detections.csv')}_frames.csv"
        )
    return detections_csv.with_name(f"{detections_csv.stem}_frames.csv")


def normalize_compression(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower() in {"", "none", "no", "false", "0"}:
        return None
    return normalized


def parse_positive_float(value: str, *, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(
            f"{field_name} must be a positive finite number; got {value!r}."
        )
    return parsed


def parse_frame(value: str) -> int:
    frame = int(value)
    if frame <= 0:
        raise ValueError(f"Frame numbers must be positive; got {value!r}.")
    return frame


def parse_label(value: str) -> int:
    label = int(value)
    if label <= 0:
        raise ValueError(f"Detection labels must be positive; got {value!r}.")
    return label


def ensure_columns(
    fieldnames: list[str] | None, required_columns: set[str], *, path: Path
) -> None:
    if fieldnames is None:
        raise ValueError(f"{path} does not contain a CSV header.")
    missing = sorted(required_columns.difference(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_detections(
    detections_csv: Path,
) -> tuple[dict[int, list[Detection]], dict[int, int], int]:
    required_columns = {"frame", "detection_index", "x", "y", "z"}
    detections_by_frame: dict[int, list[Detection]] = {}
    counts_by_frame: dict[int, int] = {}
    max_label = 0

    with detections_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        ensure_columns(reader.fieldnames, required_columns, path=detections_csv)
        for row in reader:
            frame = parse_frame(row["frame"])
            label = parse_label(row["detection_index"])
            detection = Detection(
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                label=label,
            )
            detections_by_frame.setdefault(frame, []).append(detection)
            counts_by_frame[frame] = counts_by_frame.get(frame, 0) + 1
            max_label = max(max_label, label)

    return detections_by_frame, counts_by_frame, max_label


def read_frame_sigmas(
    frames_csv: Path,
    *,
    sigma_z_column: str,
    sigma_xy_column: str,
) -> tuple[dict[int, tuple[float, float]], dict[int, int]]:
    required_columns = {"frame", "n_detections", sigma_z_column, sigma_xy_column}
    sigmas_by_frame: dict[int, tuple[float, float]] = {}
    counts_by_frame: dict[int, int] = {}

    with frames_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        ensure_columns(reader.fieldnames, required_columns, path=frames_csv)
        for row in reader:
            frame = parse_frame(row["frame"])
            sigma_z = parse_positive_float(
                row[sigma_z_column],
                field_name=sigma_z_column,
            )
            sigma_xy = parse_positive_float(
                row[sigma_xy_column],
                field_name=sigma_xy_column,
            )
            sigmas_by_frame[frame] = (sigma_z, sigma_xy)
            counts_by_frame[frame] = int(row["n_detections"])

    return sigmas_by_frame, counts_by_frame


def discover_raw_volumes(raw_dir: Path, raw_glob: str) -> list[Path]:
    files = sorted(raw_dir.glob(raw_glob), key=natural_sort_key)
    if not files:
        raise ValueError(f"No raw volumes matching {raw_glob!r} found in {raw_dir}.")
    return files


def read_tiff_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path} does not contain a readable TIFF series.")
        shape = tuple(int(dim) for dim in tif.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"{path} must be a 3D TIFF volume; got shape {shape}.")
    return shape


def choose_dtype(dtype_name: str, max_label: int) -> str:
    if dtype_name != "auto":
        dtype = np.dtype(dtype_name)
        max_value = int(np.iinfo(dtype).max)
        if max_label > max_value:
            raise ValueError(
                f"Max detection label {max_label} does not fit in {dtype_name}."
            )
        return dtype_name
    return "uint16" if max_label <= np.iinfo(np.uint16).max else "uint32"


def build_frame_tasks(
    *,
    raw_files: list[Path],
    output_dir: Path,
    detections_by_frame: dict[int, list[Detection]],
    detection_counts_by_frame: dict[int, int],
    frame_counts_by_frame: dict[int, int],
    sigmas_by_frame: dict[int, tuple[float, float]],
    threshold: float,
    dtype_name: str,
    zero_based: bool,
    compression: str | None,
) -> list[FrameTask]:
    tasks: list[FrameTask] = []
    raw_count = len(raw_files)
    max_detection_frame = max(detections_by_frame, default=0)
    if max_detection_frame > raw_count:
        raise ValueError(
            (
                f"Detections reference frame {max_detection_frame}, but only "
                f"{raw_count} raw volumes were found."
            )
        )

    for frame, expected_count in frame_counts_by_frame.items():
        actual_count = detection_counts_by_frame.get(frame, 0)
        if actual_count != expected_count:
            raise ValueError(
                (
                    f"Frame {frame}: detections CSV has {actual_count} rows, "
                    f"but frames CSV reports {expected_count}."
                )
            )

    for index, raw_path in enumerate(raw_files, start=1):
        if index not in sigmas_by_frame:
            raise ValueError(f"No sigma metadata available for frame {index}.")
        sigma_z, sigma_xy = sigmas_by_frame[index]
        tasks.append(
            FrameTask(
                frame=index,
                raw_path=raw_path,
                output_path=output_dir / raw_path.name,
                shape_zyx=read_tiff_shape(raw_path),
                detections=tuple(detections_by_frame.get(index, ())),
                sigma_z=sigma_z,
                sigma_xy=sigma_xy,
                threshold=threshold,
                dtype_name=dtype_name,
                zero_based=zero_based,
                compression=compression,
            )
        )
    return tasks


def detection_center(
    detection: Detection, *, zero_based: bool
) -> tuple[float, float, float]:
    offset = 0.0 if zero_based else 1.0
    return detection.z - offset, detection.y - offset, detection.x - offset


def draw_detection(
    mask: np.ndarray, detection: Detection, task: FrameTask, radius_factor: float
) -> None:
    center_z, center_y, center_x = detection_center(
        detection,
        zero_based=task.zero_based,
    )
    shape_z, shape_y, shape_x = task.shape_zyx
    radius_z = radius_factor * task.sigma_z
    radius_xy = radius_factor * task.sigma_xy

    z0 = max(0, int(math.floor(center_z - radius_z)))
    z1 = min(shape_z, int(math.ceil(center_z + radius_z)) + 1)
    y0 = max(0, int(math.floor(center_y - radius_xy)))
    y1 = min(shape_y, int(math.ceil(center_y + radius_xy)) + 1)
    x0 = max(0, int(math.floor(center_x - radius_xy)))
    x1 = min(shape_x, int(math.ceil(center_x + radius_xy)) + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return

    zz = np.arange(z0, z1, dtype=np.float32) - np.float32(center_z)
    yy = np.arange(y0, y1, dtype=np.float32) - np.float32(center_y)
    xx = np.arange(x0, x1, dtype=np.float32) - np.float32(center_x)
    distance2 = (
        (zz[:, None, None] / np.float32(task.sigma_z)) ** 2
        + (yy[None, :, None] / np.float32(task.sigma_xy)) ** 2
        + (xx[None, None, :] / np.float32(task.sigma_xy)) ** 2
    )
    footprint = distance2 <= np.float32(radius_factor * radius_factor)
    mask[z0:z1, y0:y1, x0:x1][footprint] = detection.label


def write_frame_mask(task: FrameTask) -> FrameResult:
    dtype = np.dtype(task.dtype_name)
    mask = np.zeros(task.shape_zyx, dtype=dtype)
    radius_factor = math.sqrt(-2.0 * math.log(task.threshold))
    for detection in task.detections:
        draw_detection(mask, detection, task, radius_factor)

    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.output_path.with_name(
        f".{task.output_path.stem}.tmp{task.output_path.suffix}"
    )
    try:
        tifffile.imwrite(
            tmp_path,
            mask,
            bigtiff=True,
            metadata=None,
            photometric="minisblack",
            compression=task.compression,
        )
        tmp_path.replace(task.output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return FrameResult(
        frame=task.frame,
        raw_path=task.raw_path,
        output_path=task.output_path,
        detection_count=len(task.detections),
        nonzero_voxels=int(np.count_nonzero(mask)),
    )


def run_tasks(tasks: list[FrameTask], *, workers: int) -> list[FrameResult]:
    if workers <= 1:
        return [write_frame_mask(task) for task in tasks]

    results: list[FrameResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_frame = {
            executor.submit(write_frame_mask, task): task.frame for task in tasks
        }
        total = len(future_to_frame)
        for completed, future in enumerate(as_completed(future_to_frame), start=1):
            results.append(future.result())
            if completed == total or completed % 10 == 0:
                print(f"[csv_to_mask] Completed {completed}/{total} frames", flush=True)

    return sorted(results, key=lambda result: result.frame)


def validate_paths(raw_dir: Path, output_dir: Path) -> None:
    raw_resolved = raw_dir.expanduser().resolve()
    output_resolved = output_dir.expanduser().resolve()
    if raw_resolved == output_resolved:
        raise ValueError(
            "output-dir must be different from raw-dir because output filenames match raw filenames."
        )


def main() -> None:
    args = parse_args()
    if args.threshold <= 0.0 or args.threshold >= 1.0:
        raise ValueError("--threshold must be greater than 0 and less than 1.")
    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    detections_csv = args.detections_csv.expanduser()
    raw_dir = args.raw_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    validate_paths(raw_dir, output_dir)

    use_constant_sigmas = args.sigma_z is not None or args.sigma_xy is not None
    if use_constant_sigmas and (args.sigma_z is None or args.sigma_xy is None):
        raise ValueError("--sigma-z and --sigma-xy must be supplied together.")

    detections_by_frame, detection_counts_by_frame, max_label = read_detections(
        detections_csv
    )
    raw_files = discover_raw_volumes(raw_dir, args.raw_glob)

    if use_constant_sigmas:
        sigma_z = parse_positive_float(str(args.sigma_z), field_name="--sigma-z")
        sigma_xy = parse_positive_float(str(args.sigma_xy), field_name="--sigma-xy")
        sigmas_by_frame = {
            frame: (sigma_z, sigma_xy) for frame in range(1, len(raw_files) + 1)
        }
        frame_counts_by_frame: dict[int, int] = {}
    else:
        frames_csv = args.frames_csv or default_frames_csv_path(detections_csv)
        if not frames_csv.is_file():
            raise FileNotFoundError(
                (
                    f"Frame metadata CSV not found at {frames_csv}. Provide "
                    "--frames-csv, or provide constant --sigma-z and --sigma-xy."
                )
            )
        sigmas_by_frame, frame_counts_by_frame = read_frame_sigmas(
            frames_csv,
            sigma_z_column=args.sigma_z_column,
            sigma_xy_column=args.sigma_xy_column,
        )

    dtype_name = choose_dtype(args.dtype, max_label)
    compression = normalize_compression(args.compression)
    tasks = build_frame_tasks(
        raw_files=raw_files,
        output_dir=output_dir,
        detections_by_frame=detections_by_frame,
        detection_counts_by_frame=detection_counts_by_frame,
        frame_counts_by_frame=frame_counts_by_frame,
        sigmas_by_frame=sigmas_by_frame,
        threshold=args.threshold,
        dtype_name=dtype_name,
        zero_based=args.zero_based,
        compression=compression,
    )

    print(
        (
            f"[csv_to_mask] Writing {len(tasks)} masks to {output_dir} "
            f"with dtype={dtype_name}, threshold={args.threshold}, "
            f"workers={args.workers}"
        ),
        flush=True,
    )
    results = run_tasks(tasks, workers=args.workers)
    print(
        (
            f"[csv_to_mask] Saved {len(results)} masks; "
            f"{sum(result.detection_count for result in results)} detections, "
            f"{sum(result.nonzero_voxels for result in results)} labeled voxels"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
