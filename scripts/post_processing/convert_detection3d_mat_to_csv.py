from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

DEFAULT_VARIABLE_NAME = "frameInfo"

DETECTION_VECTOR_FIELDS = (
    "x",
    "y",
    "z",
    "A",
    "c",
    "x_pstd",
    "y_pstd",
    "z_pstd",
    "A_pstd",
    "c_pstd",
    "x_init",
    "y_init",
    "z_init",
    "sigma_r",
    "SE_sigma_r",
    "RSS",
    "pval_Ar",
    "hval_Ar",
    "hval_AD",
    "isPSF",
)
DETECTION_MATRIX_FIELDS = {
    "xCoord": 2,
    "yCoord": 2,
    "zCoord": 2,
    "amp": 2,
}
FRAME_VECTOR_FIELDS = {
    "s": 2,
    "dRange": 2,
}


@dataclass(frozen=True)
class ConversionResult:
    detections_path: Path
    frames_path: Path
    frame_count: int
    detection_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Detection3D.mat MATLAB frameInfo struct into separate "
            "per-detection and per-frame CSV files."
        )
    )
    parser.add_argument(
        "mat_path",
        type=Path,
        help="Path to Detection3D.mat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for CSV outputs. Defaults to the MAT file's parent folder.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Filename prefix for CSV outputs. Defaults to the MAT file stem.",
    )
    parser.add_argument(
        "--variable",
        default=DEFAULT_VARIABLE_NAME,
        help=f"MATLAB variable to convert. Defaults to {DEFAULT_VARIABLE_NAME}.",
    )
    return parser.parse_args()


def public_mat_variables(data: dict[str, Any]) -> list[str]:
    return [name for name in data if not name.startswith("__")]


def load_frame_info(mat_path: Path, variable_name: str) -> tuple[np.ndarray, str]:
    data = loadmat(mat_path, squeeze_me=False, struct_as_record=True)
    if variable_name not in data:
        variables = public_mat_variables(data)
        raise ValueError(
            (
                f"{mat_path} does not contain variable {variable_name!r}. "
                f"Available variables: {variables}"
            )
        )

    frame_info = np.asarray(data[variable_name])
    if frame_info.dtype.names is None:
        raise ValueError(f"Variable {variable_name!r} is not a MATLAB struct array.")

    required_fields = set(DETECTION_VECTOR_FIELDS)
    required_fields.update(DETECTION_MATRIX_FIELDS)
    required_fields.update(FRAME_VECTOR_FIELDS)
    missing_fields = sorted(required_fields.difference(frame_info.dtype.names))
    if missing_fields:
        raise ValueError(
            f"Variable {variable_name!r} is missing required fields: {missing_fields}"
        )

    return frame_info.ravel(order="F"), variable_name


def unwrap_matlab_value(value: Any) -> np.ndarray:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return np.asarray(value)


def as_vector(value: Any, *, field_name: str) -> np.ndarray:
    array = unwrap_matlab_value(value)
    if array.dtype == object:
        raise ValueError(f"Field {field_name!r} contains nested object data.")
    if array.size == 0:
        return np.asarray([], dtype=array.dtype)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    raise ValueError(
        f"Field {field_name!r} must be a scalar or vector; got shape {array.shape}."
    )


def as_fixed_width_vector(value: Any, *, field_name: str, width: int) -> np.ndarray:
    vector = as_vector(value, field_name=field_name)
    if vector.size != width:
        raise ValueError(
            f"Field {field_name!r} must contain {width} values; got {vector.size}."
        )
    return vector


def as_matrix(
    value: Any,
    *,
    field_name: str,
    rows: int,
    columns: int,
) -> np.ndarray:
    array = unwrap_matlab_value(value)
    if array.dtype == object:
        raise ValueError(f"Field {field_name!r} contains nested object data.")
    if rows == 0 and array.size == 0:
        return np.empty((0, columns), dtype=array.dtype)
    if array.ndim != 2 or array.shape != (rows, columns):
        raise ValueError(
            (
                f"Field {field_name!r} must have shape ({rows}, {columns}); "
                f"got {array.shape}."
            )
        )
    return array


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def detection_fieldnames() -> list[str]:
    fieldnames = ["frame", "detection_index"]
    fieldnames.extend(DETECTION_VECTOR_FIELDS)
    for field_name, columns in DETECTION_MATRIX_FIELDS.items():
        fieldnames.extend(f"{field_name}_{index}" for index in range(1, columns + 1))
    return fieldnames


def frame_fieldnames() -> list[str]:
    fieldnames = ["frame", "n_detections"]
    for field_name, width in FRAME_VECTOR_FIELDS.items():
        fieldnames.extend(f"{field_name}_{index}" for index in range(1, width + 1))
    return fieldnames


def detection_arrays(
    frame: np.void, *, frame_number: int
) -> tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]:
    vectors = {
        field_name: as_vector(frame[field_name], field_name=field_name)
        for field_name in DETECTION_VECTOR_FIELDS
    }
    detection_count = int(vectors["x"].size)
    for field_name, vector in vectors.items():
        if vector.size != detection_count:
            raise ValueError(
                (
                    f"Frame {frame_number}: field {field_name!r} has {vector.size} "
                    f"values; expected {detection_count}."
                )
            )

    matrices = {
        field_name: as_matrix(
            frame[field_name],
            field_name=field_name,
            rows=detection_count,
            columns=columns,
        )
        for field_name, columns in DETECTION_MATRIX_FIELDS.items()
    }
    return detection_count, vectors, matrices


def write_frames_csv(frames: np.ndarray, output_path: Path) -> int:
    detection_total = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=frame_fieldnames())
        writer.writeheader()
        for frame_index, frame in enumerate(frames, start=1):
            detection_count, _, _ = detection_arrays(
                frame,
                frame_number=frame_index,
            )
            row: dict[str, Any] = {
                "frame": frame_index,
                "n_detections": detection_count,
            }
            for field_name, width in FRAME_VECTOR_FIELDS.items():
                values = as_fixed_width_vector(
                    frame[field_name],
                    field_name=field_name,
                    width=width,
                )
                for value_index, value in enumerate(values, start=1):
                    row[f"{field_name}_{value_index}"] = csv_value(value)
            writer.writerow(row)
            detection_total += detection_count
    return detection_total


def write_detections_csv(frames: np.ndarray, output_path: Path) -> int:
    detection_total = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detection_fieldnames())
        writer.writeheader()
        for frame_index, frame in enumerate(frames, start=1):
            detection_count, vectors, matrices = detection_arrays(
                frame,
                frame_number=frame_index,
            )
            for detection_index in range(detection_count):
                row: dict[str, Any] = {
                    "frame": frame_index,
                    "detection_index": detection_index + 1,
                }
                for field_name, vector in vectors.items():
                    row[field_name] = csv_value(vector[detection_index])
                for field_name, matrix in matrices.items():
                    for value_index, value in enumerate(
                        matrix[detection_index],
                        start=1,
                    ):
                        row[f"{field_name}_{value_index}"] = csv_value(value)
                writer.writerow(row)
            detection_total += detection_count
    return detection_total


def convert_detection3d_mat(
    mat_path: Path,
    *,
    output_dir: Path | None = None,
    output_prefix: str | None = None,
    variable_name: str = DEFAULT_VARIABLE_NAME,
) -> ConversionResult:
    mat_path = mat_path.expanduser().resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(f"{mat_path} does not exist or is not a file.")

    frames, _ = load_frame_info(mat_path, variable_name)
    output_dir = mat_path.parent if output_dir is None else output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix or mat_path.stem

    detections_path = output_dir / f"{output_prefix}_detections.csv"
    frames_path = output_dir / f"{output_prefix}_frames.csv"
    tmp_detections_path = detections_path.with_name(f"{detections_path.name}.tmp")
    tmp_frames_path = frames_path.with_name(f"{frames_path.name}.tmp")

    try:
        detection_count = write_detections_csv(frames, tmp_detections_path)
        frame_detection_count = write_frames_csv(frames, tmp_frames_path)
        if detection_count != frame_detection_count:
            raise ValueError(
                (
                    "Detection and frame CSV writers disagreed on the detection "
                    f"count: {detection_count} vs {frame_detection_count}."
                )
            )
        tmp_detections_path.replace(detections_path)
        tmp_frames_path.replace(frames_path)
    except Exception:
        tmp_detections_path.unlink(missing_ok=True)
        tmp_frames_path.unlink(missing_ok=True)
        raise

    return ConversionResult(
        detections_path=detections_path,
        frames_path=frames_path,
        frame_count=int(frames.size),
        detection_count=detection_count,
    )


def main() -> None:
    args = parse_args()
    result = convert_detection3d_mat(
        args.mat_path,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        variable_name=args.variable,
    )
    print(
        (
            f"[convert_detection3d] Saved {result.detection_count} detections "
            f"from {result.frame_count} frames to {result.detections_path}"
        ),
        flush=True,
    )
    print(
        f"[convert_detection3d] Saved frame metadata to {result.frames_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
