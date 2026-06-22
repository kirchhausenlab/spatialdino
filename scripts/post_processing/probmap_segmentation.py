from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy import ndimage
import tifffile

from spatialdino.inference.output_layout import (
    discover_inference_timepoints,
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

    timepoints = discover_inference_timepoints(input_path)
    probmap_root = probability_map_dir(input_path)
    if not probmap_root.is_dir():
        raise FileNotFoundError(f"Missing probability-map folder: {probmap_root}")

    output_root = segmentation_probmap_dir(output_path)
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[segmentation] Found {len(timepoints)} timepoints", flush=True)
    for index, timepoint in enumerate(timepoints, start=1):
        print(f"[segmentation] Processing {timepoint.name} ({index}/{len(timepoints)})", flush=True)
        probmap_path = probmap_root / f"{timepoint.name}.tif"
        if not probmap_path.is_file():
            raise FileNotFoundError(f"Missing probability-map volume for {timepoint.name}: {probmap_path}")

        probmap = read_tiff_volume(probmap_path)
        semantic = (probmap >= threshold).astype(np.uint8, copy=False)
        output_array = semantic_to_instance_seg(semantic) if run_ccl else semantic
        mask_path = output_root / f"{timepoint.name}.tif"
        tifffile.imwrite(mask_path, output_array, bigtiff=True, metadata=None, photometric="minisblack")
        print(f"[segmentation] Completed {timepoint.name}", flush=True)

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
