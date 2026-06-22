from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_VARIABLE_NAME = "tracks"
DEFAULT_DETECTION_CSV_NAME = "Detection3D_detections.csv"
TRACK_COLUMNS = ("track_id", "start", "t", "x", "y", "z", "A", "track_length")
POINT_FIELDS = ("f", "x", "y", "z", "A")
SOURCE_FIELDS = ("start", "tracksFeatIndxCG")
Z_INFERENCE_FIELDS = ("f", "z", "tracksFeatIndxCG", "nSeg")
Z_INFERENCE_MIN_MATCHES = 10
Z_INFERENCE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ConversionResult:
    output_csv: Path
    track_count: int
    exported_track_count: int
    point_count: int
    skipped_point_count: int
    export_mode: str
    duplicate_source_count: int
    missing_source_count: int
    z_flip_constant: float | None
    z_flip_source: str
    z_inference_match_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a MATLAB v7.3 ProcessedTracks.mat file into the simple "
            "spatialDINO tracks CSV schema."
        )
    )
    parser.add_argument(
        "mat_path",
        type=Path,
        help="Path to ProcessedTracks.mat.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <mat-file-stem>.csv next to the MAT file.",
    )
    parser.add_argument(
        "--variable",
        default=DEFAULT_VARIABLE_NAME,
        help=f"Top-level MATLAB struct variable to convert. Defaults to {DEFAULT_VARIABLE_NAME!r}.",
    )
    parser.add_argument(
        "--detection-csv",
        type=Path,
        default=None,
        help=(
            "Detection3D detections CSV used to export source-linked detections. "
            f"Defaults to sibling {DEFAULT_DETECTION_CSV_NAME!r}."
        ),
    )
    parser.add_argument(
        "--include-gap-filled",
        action="store_true",
        help=(
            "Export ProcessedTracks coordinate arrays, including gap-filled/interpolated "
            "points. By default only source-linked Detection3D detections are exported."
        ),
    )
    parser.add_argument(
        "--keep-duplicate-source-detections",
        action="store_true",
        help=(
            "Keep duplicate frame/detection references from compound tracks. "
            "By default duplicates are written only once."
        ),
    )
    z_group = parser.add_mutually_exclusive_group()
    z_group.add_argument(
        "--raw-z-size",
        type=int,
        default=None,
        help=(
            "Raw volume z size. Writes z as raw_z_size + 2 - z, matching MATLAB one-based "
            "Detection3D coordinates."
        ),
    )
    z_group.add_argument(
        "--z-flip-constant",
        type=float,
        default=None,
        help="Write z as z_flip_constant - z.",
    )
    z_group.add_argument(
        "--no-invert-z",
        action="store_true",
        help="Keep the z coordinate exactly as stored in ProcessedTracks.mat.",
    )
    return parser.parse_args()


def matlab_field_names(group: h5py.Group) -> list[str]:
    fields = group.attrs.get("MATLAB_fields")
    if fields is None:
        return []
    names: list[str] = []
    for field in fields:
        names.append(b"".join(np.asarray(field).ravel()).decode("utf-8"))
    return names


def ensure_tracks_group(file: h5py.File, variable_name: str) -> h5py.Group:
    if variable_name not in file:
        variables = [name for name in file.keys() if name != "#refs#"]
        raise ValueError(
            (
                f"{file.filename} does not contain variable {variable_name!r}. "
                f"Available variables: {variables}"
            )
        )
    tracks = file[variable_name]
    if not isinstance(tracks, h5py.Group):
        raise ValueError(f"Variable {variable_name!r} is not a MATLAB struct group.")

    missing = sorted(set((*POINT_FIELDS, "start")).difference(tracks.keys()))
    if missing:
        fields = matlab_field_names(tracks) or list(tracks.keys())
        raise ValueError(
            f"Variable {variable_name!r} is missing required fields {missing}. Available fields: {fields}"
        )
    return tracks


def require_tracks_fields(tracks: h5py.Group, field_names: tuple[str, ...]) -> None:
    missing = sorted(set(field_names).difference(tracks.keys()))
    if missing:
        fields = matlab_field_names(tracks) or list(tracks.keys())
        raise ValueError(
            f"Variable {tracks.name!r} is missing required fields {missing}. Available fields: {fields}"
        )


def read_ref_array(file: h5py.File, ref: h5py.Reference) -> np.ndarray:
    if not ref:
        return np.asarray([], dtype=np.float64)
    obj = file[ref]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"Expected HDF5 dataset reference, got {obj.name}.")
    attrs = obj.attrs
    if bool(attrs.get("MATLAB_empty", False)):
        return np.asarray([], dtype=np.float64)
    return np.asarray(obj[()]).ravel(order="F")


def read_ref_dataset(file: h5py.File, ref: h5py.Reference) -> np.ndarray:
    if not ref:
        return np.asarray([], dtype=np.float64)
    obj = file[ref]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"Expected HDF5 dataset reference, got {obj.name}.")
    attrs = obj.attrs
    if bool(attrs.get("MATLAB_empty", False)):
        return np.asarray([], dtype=np.float64)
    return np.asarray(obj[()])


def read_track_field(
    file: h5py.File,
    tracks: h5py.Group,
    field_name: str,
    track_index: int,
) -> np.ndarray:
    field = tracks[field_name]
    if field.ndim != 2 or field.shape[1] != 1:
        raise ValueError(
            f"Field {field_name!r} must be a column reference array; got shape {field.shape}."
        )
    return read_ref_array(file, field[track_index, 0])


def read_track_dataset(
    file: h5py.File,
    tracks: h5py.Group,
    field_name: str,
    track_index: int,
) -> np.ndarray:
    field = tracks[field_name]
    if field.ndim != 2 or field.shape[1] != 1:
        raise ValueError(
            f"Field {field_name!r} must be a column reference array; got shape {field.shape}."
        )
    return read_ref_dataset(file, field[track_index, 0])


def read_scalar(
    file: h5py.File,
    tracks: h5py.Group,
    field_name: str,
    track_index: int,
) -> float:
    values = read_track_field(file, tracks, field_name, track_index)
    if values.size == 0:
        return float("nan")
    return float(values[0])


def finite_point_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    lengths = {name: values.size for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Point fields have inconsistent lengths: {lengths}")
    length = next(iter(lengths.values()), 0)
    if length == 0:
        return np.zeros(0, dtype=bool)

    mask = np.ones(length, dtype=bool)
    for values in arrays.values():
        mask &= np.isfinite(values.astype(np.float64, copy=False))
    return mask


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_detection_csv(mat_path: Path, detection_csv: Path | None) -> Path:
    return (
        mat_path.with_name(DEFAULT_DETECTION_CSV_NAME)
        if detection_csv is None
        else detection_csv.expanduser()
    )


def read_detection_rows_by_key(
    detection_csv: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    detection_csv = detection_csv.expanduser().resolve()
    if not detection_csv.is_file():
        raise FileNotFoundError(f"{detection_csv} does not exist or is not a file.")

    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    with detection_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"frame", "detection_index", "x", "y", "z", "A"}
        missing_columns = sorted(required_columns.difference(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"{detection_csv} is missing required columns: {missing_columns}"
            )
        for row in reader:
            frame = int(row["frame"])
            detection_index = int(row["detection_index"])
            rows_by_key[(frame, detection_index)] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "A": float(row["A"]),
            }
    return rows_by_key


def read_detection_z_by_key(detection_csv: Path) -> dict[tuple[int, int], float]:
    return {
        key: row["z"] for key, row in read_detection_rows_by_key(detection_csv).items()
    }


def clean_z_flip_constant(value: float) -> float:
    rounded = round(value)
    if np.isclose(value, rounded, atol=Z_INFERENCE_TOLERANCE):
        return float(rounded)
    return float(value)


def infer_z_flip_constant(
    file: h5py.File,
    tracks: h5py.Group,
    *,
    detection_csv: Path,
) -> tuple[float, int]:
    require_tracks_fields(tracks, Z_INFERENCE_FIELDS)
    detection_z_by_key = read_detection_z_by_key(detection_csv)
    z_sums: list[float] = []
    track_count = int(tracks["z"].shape[0])

    for track_index in range(track_count):
        n_segments = read_scalar(file, tracks, "nSeg", track_index)
        if not np.isfinite(n_segments) or int(round(n_segments)) != 1:
            continue

        frames = read_track_field(file, tracks, "f", track_index).astype(
            np.float64,
            copy=False,
        )
        track_z = read_track_field(file, tracks, "z", track_index).astype(
            np.float64,
            copy=False,
        )
        feature_indices = read_track_field(
            file,
            tracks,
            "tracksFeatIndxCG",
            track_index,
        ).astype(np.float64, copy=False)
        length = min(frames.size, track_z.size, feature_indices.size)

        for point_index in range(length):
            frame = frames[point_index]
            z_value = track_z[point_index]
            feature_index = feature_indices[point_index]
            if not (
                np.isfinite(frame)
                and np.isfinite(z_value)
                and np.isfinite(feature_index)
                and feature_index > 0
            ):
                continue
            detection_z = detection_z_by_key.get((
                int(round(frame)),
                int(round(feature_index)),
            ))
            if detection_z is not None:
                z_sums.append(float(z_value + detection_z))

    if len(z_sums) < Z_INFERENCE_MIN_MATCHES:
        raise ValueError(
            (
                f"Only found {len(z_sums)} source-matched detection points in "
                f"{detection_csv}; need at least {Z_INFERENCE_MIN_MATCHES} to infer z flip."
            )
        )

    sums = np.asarray(z_sums, dtype=np.float64)
    z_flip_constant = clean_z_flip_constant(float(np.median(sums)))
    max_error = float(np.max(np.abs(sums - z_flip_constant)))
    if max_error > Z_INFERENCE_TOLERANCE:
        raise ValueError(
            (
                "Could not infer a consistent z flip constant. "
                f"Median was {z_flip_constant}, but max absolute error was {max_error}."
            )
        )
    return z_flip_constant, len(z_sums)


def resolve_z_flip(
    file: h5py.File,
    tracks: h5py.Group,
    *,
    mat_path: Path,
    detection_csv: Path | None,
    raw_z_size: int | None,
    z_flip_constant: float | None,
    no_invert_z: bool,
) -> tuple[float | None, str, int]:
    if no_invert_z:
        return None, "stored ProcessedTracks z", 0
    if z_flip_constant is not None:
        return float(z_flip_constant), "--z-flip-constant", 0
    if raw_z_size is not None:
        if raw_z_size <= 0:
            raise ValueError(f"--raw-z-size must be positive; got {raw_z_size}.")
        return float(raw_z_size + 2), "--raw-z-size", 0

    resolved_detection_csv = resolve_detection_csv(mat_path, detection_csv)
    constant, match_count = infer_z_flip_constant(
        file,
        tracks,
        detection_csv=resolved_detection_csv,
    )
    return constant, f"inferred from {resolved_detection_csv}", match_count


def source_detection_rows(
    file: h5py.File,
    tracks: h5py.Group,
    track_index: int,
    *,
    detection_rows_by_key: dict[tuple[int, int], dict[str, Any]],
    seen_sources: set[tuple[int, int]],
    first_track_id: int,
    keep_duplicate_source_detections: bool,
) -> tuple[list[dict[str, Any]], int, int, int]:
    require_tracks_fields(tracks, SOURCE_FIELDS)
    start_value = read_scalar(file, tracks, "start", track_index)
    if not np.isfinite(start_value):
        return [], 0, 0, 0
    start = int(round(start_value))

    feature_indices = read_track_dataset(
        file,
        tracks,
        "tracksFeatIndxCG",
        track_index,
    ).astype(np.float64, copy=False)
    if feature_indices.size == 0:
        return [], 0, 0, 0
    if feature_indices.ndim == 1:
        feature_indices = feature_indices.reshape(-1, 1)
    if feature_indices.ndim != 2:
        raise ValueError(
            (
                "Field 'tracksFeatIndxCG' must reference a vector or matrix; "
                f"got shape {feature_indices.shape} for track {track_index + 1}."
            )
        )

    segment_rows: list[list[dict[str, Any]]] = [
        [] for _segment_index in range(feature_indices.shape[1])
    ]
    duplicate_source_count = 0
    missing_source_count = 0
    for frame_offset in range(feature_indices.shape[0]):
        frame = start + frame_offset
        for segment_index in range(feature_indices.shape[1]):
            detection_index_value = feature_indices[frame_offset, segment_index]
            if not np.isfinite(detection_index_value) or detection_index_value <= 0:
                continue
            source_key = (frame, int(round(detection_index_value)))
            if source_key in seen_sources and not keep_duplicate_source_detections:
                duplicate_source_count += 1
                continue
            detection_row = detection_rows_by_key.get(source_key)
            if detection_row is None:
                missing_source_count += 1
                continue
            seen_sources.add(source_key)
            segment_rows[segment_index].append({
                "frame": frame,
                "x": detection_row["x"],
                "y": detection_row["y"],
                "z": detection_row["z"],
                "A": detection_row["A"],
            })

    rows: list[dict[str, Any]] = []
    exported_track_count = 0
    for segment in segment_rows:
        if not segment:
            continue
        exported_track_count += 1
        segment.sort(key=lambda row: row["frame"])
        segment_start = int(segment[0]["frame"])
        track_length = len(segment)
        for point in segment:
            frame = int(point["frame"])
            rows.append({
                "track_id": first_track_id + exported_track_count - 1,
                "start": segment_start,
                "t": frame - segment_start + 1,
                "x": point["x"],
                "y": point["y"],
                "z": point["z"],
                "A": point["A"],
                "track_length": track_length,
            })
    return rows, duplicate_source_count, missing_source_count, exported_track_count


def processed_track_rows(
    file: h5py.File,
    tracks: h5py.Group,
    track_index: int,
    *,
    z_flip_constant: float | None,
) -> tuple[list[dict[str, Any]], int]:
    arrays = {
        field_name: read_track_field(file, tracks, field_name, track_index).astype(
            np.float64,
            copy=False,
        )
        for field_name in POINT_FIELDS
    }
    finite_mask = finite_point_mask(arrays)
    skipped_point_count = int((~finite_mask).sum())
    if not finite_mask.any():
        return [], skipped_point_count

    finite_indices = np.flatnonzero(finite_mask)
    frame_values = arrays["f"][finite_indices]
    order = np.argsort(frame_values, kind="mergesort")
    ordered_indices = finite_indices[order]

    start_value = read_scalar(file, tracks, "start", track_index)
    if not np.isfinite(start_value):
        start_value = float(np.nanmin(arrays["f"][ordered_indices]))
    start = int(round(start_value))
    track_length = int(ordered_indices.size)
    rows: list[dict[str, Any]] = []
    for point_number, source_index in enumerate(ordered_indices, start=1):
        z_value = arrays["z"][source_index]
        if z_flip_constant is not None:
            z_value = z_flip_constant - z_value
        rows.append({
            "track_id": track_index + 1,
            "start": start,
            "t": point_number,
            "x": csv_value(arrays["x"][source_index]),
            "y": csv_value(arrays["y"][source_index]),
            "z": csv_value(z_value),
            "A": csv_value(arrays["A"][source_index]),
            "track_length": track_length,
        })
    return rows, skipped_point_count


def convert_processed_tracks_mat(
    mat_path: Path,
    *,
    output_csv: Path | None = None,
    variable_name: str = DEFAULT_VARIABLE_NAME,
    detection_csv: Path | None = None,
    include_gap_filled: bool = False,
    keep_duplicate_source_detections: bool = False,
    raw_z_size: int | None = None,
    z_flip_constant: float | None = None,
    no_invert_z: bool = False,
) -> ConversionResult:
    mat_path = mat_path.expanduser().resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(f"{mat_path} does not exist or is not a file.")

    output_csv = (
        mat_path.with_suffix(".csv") if output_csv is None else output_csv.expanduser()
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_name(f"{output_csv.name}.tmp")

    point_count = 0
    skipped_point_count = 0
    duplicate_source_count = 0
    missing_source_count = 0
    exported_track_count = 0
    resolved_z_flip_constant: float | None = None
    z_flip_source = "not used"
    z_inference_match_count = 0
    export_mode = (
        "processed coordinates with gap-filled/interpolated points"
        if include_gap_filled
        else "source-linked Detection3D detections"
    )
    try:
        with h5py.File(mat_path, "r") as file:
            tracks = ensure_tracks_group(file, variable_name)
            track_count = int(tracks["x"].shape[0])
            detection_rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
            seen_sources: set[tuple[int, int]] = set()
            next_source_track_id = 1
            if include_gap_filled:
                (
                    resolved_z_flip_constant,
                    z_flip_source,
                    z_inference_match_count,
                ) = resolve_z_flip(
                    file,
                    tracks,
                    mat_path=mat_path,
                    detection_csv=detection_csv,
                    raw_z_size=raw_z_size,
                    z_flip_constant=z_flip_constant,
                    no_invert_z=no_invert_z,
                )
            else:
                resolved_detection_csv = resolve_detection_csv(mat_path, detection_csv)
                detection_rows_by_key = read_detection_rows_by_key(
                    resolved_detection_csv
                )
            with tmp_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRACK_COLUMNS)
                writer.writeheader()
                for track_index in range(track_count):
                    if include_gap_filled:
                        rows, skipped = processed_track_rows(
                            file,
                            tracks,
                            track_index,
                            z_flip_constant=resolved_z_flip_constant,
                        )
                        skipped_point_count += skipped
                    else:
                        (
                            rows,
                            duplicates,
                            missing,
                            track_segments,
                        ) = source_detection_rows(
                            file,
                            tracks,
                            track_index,
                            detection_rows_by_key=detection_rows_by_key,
                            seen_sources=seen_sources,
                            first_track_id=next_source_track_id,
                            keep_duplicate_source_detections=keep_duplicate_source_detections,
                        )
                        next_source_track_id += track_segments
                        exported_track_count += track_segments
                        duplicate_source_count += duplicates
                        missing_source_count += missing
                    for row in rows:
                        writer.writerow(row)
                    point_count += len(rows)
        tmp_path.replace(output_csv)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return ConversionResult(
        output_csv=output_csv,
        track_count=track_count,
        exported_track_count=(
            track_count if include_gap_filled else exported_track_count
        ),
        point_count=point_count,
        skipped_point_count=skipped_point_count,
        export_mode=export_mode,
        duplicate_source_count=duplicate_source_count,
        missing_source_count=missing_source_count,
        z_flip_constant=resolved_z_flip_constant,
        z_flip_source=z_flip_source,
        z_inference_match_count=z_inference_match_count,
    )


def main() -> None:
    args = parse_args()
    result = convert_processed_tracks_mat(
        args.mat_path,
        output_csv=args.output_csv,
        variable_name=args.variable,
        detection_csv=args.detection_csv,
        include_gap_filled=args.include_gap_filled,
        keep_duplicate_source_detections=args.keep_duplicate_source_detections,
        raw_z_size=args.raw_z_size,
        z_flip_constant=args.z_flip_constant,
        no_invert_z=args.no_invert_z,
    )
    print(
        (
            f"[convert_processed_tracks] Saved {result.point_count} points "
            f"from {result.track_count} MATLAB tracks to {result.output_csv}"
        ),
        flush=True,
    )
    if result.exported_track_count != result.track_count:
        print(
            (
                f"[convert_processed_tracks] Wrote {result.exported_track_count} "
                "simple track segments"
            ),
            flush=True,
        )
    print(f"[convert_processed_tracks] Export mode: {result.export_mode}", flush=True)
    if result.export_mode == "source-linked Detection3D detections":
        if result.duplicate_source_count:
            print(
                (
                    f"[convert_processed_tracks] Skipped {result.duplicate_source_count} "
                    "duplicate frame/detection references"
                ),
                flush=True,
            )
        if result.missing_source_count:
            print(
                (
                    f"[convert_processed_tracks] Skipped {result.missing_source_count} "
                    "source references not found in Detection3D CSV"
                ),
                flush=True,
            )
    elif result.z_flip_constant is None:
        print(
            "[convert_processed_tracks] Kept z coordinates as stored in ProcessedTracks.mat",
            flush=True,
        )
    else:
        message = (
            f"[convert_processed_tracks] Wrote z as {result.z_flip_constant:g} - z "
            f"({result.z_flip_source})"
        )
        if result.z_inference_match_count:
            message += f" using {result.z_inference_match_count} matched detections"
        print(message, flush=True)
    if result.skipped_point_count:
        print(
            (
                f"[convert_processed_tracks] Skipped {result.skipped_point_count} "
                "non-finite separator/incomplete points"
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
