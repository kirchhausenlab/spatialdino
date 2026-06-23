from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy import ndimage
import tifffile

from spatialdino.inference.output_layout import (
    inference_raw_dir,
    natural_sort_key,
    probability_map_dir,
    segmentation_probmap_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment normalized foreground probability maps.")
    parser.add_argument("--input-path", required=True, help="Inference output folder containing probmap/.")
    parser.add_argument("--output-path", required=True, help="Root folder where segmentation outputs are written.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Foreground probability threshold applied to normalized probmap volumes.",
    )
    parser.add_argument("--run-ccl", dest="run_ccl", action="store_true", help="Save connected-component labels.")
    parser.add_argument("--skip-ccl", dest="run_ccl", action="store_false", help="Save binary semantic masks.")
    parser.set_defaults(run_ccl=True)
    return parser.parse_args()


def read_tiff_volume(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path} does not contain a readable TIFF volume.")
        array = tif.asarray()
    if array.ndim != 3:
        raise ValueError(f"{path} must be a 3D TIFF volume.")
    return np.asarray(array, dtype=np.float32)


def semantic_to_instance_seg(semantic_seg: np.ndarray) -> np.ndarray:
    semantic_bool = np.asarray(semantic_seg > 0, dtype=bool)
    labeled, _ = ndimage.label(semantic_bool, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return labeled.astype(np.uint32, copy=False)


def named_tiff_file_map(directory: Path) -> dict[str, Path]:
    names_to_paths: dict[str, Path] = {}
    if not directory.is_dir():
        return names_to_paths

    for path in directory.iterdir():
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        existing = names_to_paths.get(path.stem)
        if existing is not None:
            raise ValueError(
                f"Duplicate TIFF files map to the same timepoint name {path.stem!r}: "
                f"{existing.name} and {path.name}."
            )
        names_to_paths[path.stem] = path
    return names_to_paths


def list_probability_map_timepoints(input_path: Path) -> list[tuple[str, Path]]:
    raw_root = inference_raw_dir(input_path)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw folder: {raw_root}")

    probmap_root = probability_map_dir(input_path)
    if not probmap_root.is_dir():
        raise FileNotFoundError(f"Missing probability-map folder: {probmap_root}")

    raw_paths = named_tiff_file_map(raw_root)
    probmap_paths = named_tiff_file_map(probmap_root)
    if not raw_paths:
        raise FileNotFoundError(f"{raw_root} contains no TIFF files.")
    if not probmap_paths:
        raise FileNotFoundError(f"{probmap_root} contains no TIFF files.")

    missing_probmap = sorted(set(raw_paths) - set(probmap_paths), key=natural_sort_key)
    if missing_probmap:
        missing_name = missing_probmap[0]
        raise FileNotFoundError(f"Missing probability-map volume for {missing_name}: {probmap_root / f'{missing_name}.tif'}")

    missing_raw = sorted(set(probmap_paths) - set(raw_paths), key=natural_sort_key)
    if missing_raw:
        missing_name = missing_raw[0]
        raise FileNotFoundError(f"Missing raw volume for {missing_name}: {raw_root / f'{missing_name}.tif'}")

    names = sorted(raw_paths, key=natural_sort_key)
    return [(name, probmap_paths[name]) for name in names]


def segment_probability_maps(
    input_path: Path,
    *,
    output_path: Path,
    threshold: float,
    run_ccl: bool,
) -> Path:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_path}")
    if output_path.exists() and not output_path.is_dir():
        raise FileNotFoundError(f"Output folder exists but is not a directory: {output_path}")

    timepoints = list_probability_map_timepoints(input_path)

    output_root = segmentation_probmap_dir(output_path)
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[segmentation] Found {len(timepoints)} timepoints", flush=True)
    for index, (timepoint_name, probmap_path) in enumerate(timepoints, start=1):
        print(f"[segmentation] Processing {timepoint_name} ({index}/{len(timepoints)})", flush=True)
        probmap = read_tiff_volume(probmap_path)
        semantic = (probmap >= threshold).astype(np.uint8, copy=False)
        output_array = semantic_to_instance_seg(semantic) if run_ccl else semantic
        mask_path = output_root / f"{timepoint_name}.tif"
        tifffile.imwrite(mask_path, output_array, bigtiff=True, metadata=None, photometric="minisblack")
        print(f"[segmentation] Completed {timepoint_name}", flush=True)

    print("[segmentation] Done", flush=True)
    return output_root


def main() -> None:
    args = parse_args()
    segment_probability_maps(
        Path(args.input_path).expanduser().resolve(),
        output_path=Path(args.output_path).expanduser().resolve(),
        threshold=float(args.threshold),
        run_ccl=bool(args.run_ccl),
    )


if __name__ == "__main__":
    main()
