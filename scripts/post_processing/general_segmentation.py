from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import tifffile

from spatialdino.inference.output_layout import (
    inference_raw_dir,
    natural_sort_key,
    probability_map_dir,
    segmentation_general_dir,
)
from spatialdino.segmentation.general import (
    DATA_BACKEND_CPU,
    DATA_BACKEND_GPU,
    DATA_BACKENDS,
    INSTANCE_METHOD_CONNECTED_COMPONENTS,
    INSTANCE_METHODS,
    apply_data_operations,
    apply_mask_operations,
    instance_segmentation,
    normalize_data_operations,
    normalize_distance_transform_connectivity,
    normalize_distance_transform_dynamic,
    normalize_distance_transform_spacing,
    normalize_intensity_normalization_percentiles,
    normalize_intensity_prominence,
    normalize_intensity_smoothing_sigma,
    normalize_instance_method,
    normalize_mask_operations,
    normalize_watershed_connectivity,
    threshold_to_semantic,
)


SOURCE_KINDS = {"raw", "probmap", "pca", "feature_stats"}
COMPONENT_SOURCE_KINDS = {"pca", "feature_stats"}
TIFF_SUFFIXES = {".tif", ".tiff"}
ARRAY_SUFFIXES = {".tif", ".tiff", ".npy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run general threshold-based segmentation.")
    parser.add_argument("--input-path", required=True, help="Inference output folder containing raw/.")
    parser.add_argument("--output-path", required=True, help="Root folder where segmentation outputs are written.")
    parser.add_argument("--source-kind", choices=sorted(SOURCE_KINDS), required=True)
    parser.add_argument(
        "--source-folder",
        default=None,
        help="Optional source folder name. Defaults to raw or probmap for those source kinds.",
    )
    parser.add_argument("--threshold", type=float, required=True, help="Scalar threshold in source data units.")
    parser.add_argument("--component-index", type=int, default=0, help="Zero-based component index for component sources.")
    parser.add_argument("--invert-mask", action="store_true", help="Use values below threshold as foreground.")
    parser.add_argument(
        "--data-operations-json",
        default="[]",
        help="JSON list of data operations applied before thresholding.",
    )
    parser.add_argument(
        "--data-backend",
        choices=sorted(DATA_BACKENDS),
        default=DATA_BACKEND_CPU,
        help="Backend used for data operations.",
    )
    parser.add_argument("--gpu-index", type=int, default=None, help="GPU index used when --data-backend gpu is selected.")
    parser.add_argument(
        "--mask-operations-json",
        default="[]",
        help="JSON list of mask operations applied after thresholding.",
    )
    parser.add_argument(
        "--instance-method",
        choices=sorted(INSTANCE_METHODS),
        default=INSTANCE_METHOD_CONNECTED_COMPONENTS,
        help="How to convert the final binary mask into the saved output.",
    )
    parser.add_argument("--voronoi-spot-sigma", type=float, default=2.0)
    parser.add_argument("--voronoi-outline-sigma", type=float, default=2.0)
    parser.add_argument("--distance-transform-dynamic", type=float, default=1.0)
    parser.add_argument("--distance-transform-connectivity", type=int, choices=[6, 26], default=6)
    parser.add_argument("--distance-transform-spacing-z", type=float, default=1.0)
    parser.add_argument("--distance-transform-spacing-y", type=float, default=1.0)
    parser.add_argument("--distance-transform-spacing-x", type=float, default=1.0)
    parser.add_argument("--intensity-prominence", type=float, default=0.15)
    parser.add_argument("--intensity-smoothing-sigma", type=float, default=0.0)
    parser.add_argument("--intensity-low-percentile", type=float, default=1.0)
    parser.add_argument("--intensity-high-percentile", type=float, default=99.0)
    parser.add_argument("--intensity-connectivity", type=int, choices=[6, 26], default=6)
    parser.add_argument("--run-ccl", dest="run_ccl", action="store_true", help="Save connected-component labels.")
    parser.add_argument("--skip-ccl", dest="run_ccl", action="store_false", help="Save binary semantic masks.")
    parser.set_defaults(run_ccl=True)
    return parser.parse_args()


def named_file_map(directory: Path, *, suffixes: set[str]) -> dict[str, Path]:
    names_to_paths: dict[str, Path] = {}
    if not directory.is_dir():
        return names_to_paths

    for path in directory.iterdir():
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        existing = names_to_paths.get(path.stem)
        if existing is not None:
            raise ValueError(
                f"Duplicate files map to the same timepoint name {path.stem!r}: "
                f"{existing.name} and {path.name}."
            )
        names_to_paths[path.stem] = path
    return names_to_paths


def read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, mmap_mode="r"))
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path} does not contain a readable TIFF volume.")
        return np.asarray(tif.asarray())


def read_scalar_volume(path: Path, *, source_kind: str, component_index: int) -> np.ndarray:
    array = read_array(path)
    if source_kind in COMPONENT_SOURCE_KINDS:
        if component_index < 0:
            raise ValueError("Component index must be nonnegative.")
        if array.ndim == 3:
            if component_index != 0:
                raise ValueError(f"{path.name} has one component, but component {component_index + 1} was requested.")
            return np.asarray(array, dtype=np.float32)
        if array.ndim == 4:
            component_count = int(array.shape[-1])
            if component_index >= component_count:
                raise ValueError(
                    f"{path.name} has {component_count} components, but component {component_index + 1} was requested."
                )
            return np.asarray(array[..., component_index], dtype=np.float32)
        raise ValueError(f"{path.name} must be a 3D or 4D component array.")

    if array.ndim != 3:
        raise ValueError(f"{path.name} must be a 3D volume.")
    return np.asarray(array, dtype=np.float32)


def source_directory(input_path: Path, *, source_kind: str, source_folder: str | None) -> Path:
    if source_folder:
        return input_path / source_folder
    if source_kind == "raw":
        return inference_raw_dir(input_path)
    if source_kind == "probmap":
        return probability_map_dir(input_path)
    raise ValueError(f"{source_kind} segmentation requires --source-folder.")


def segment_general(
    input_path: Path,
    *,
    output_path: Path,
    source_kind: str,
    source_folder: str | None,
    threshold: float,
    component_index: int,
    invert_mask: bool,
    run_ccl: bool,
    data_operations: list[dict[str, object]] | None = None,
    mask_operations: list[dict[str, object]] | None = None,
    data_backend: str = DATA_BACKEND_CPU,
    gpu_index: int | None = None,
    instance_method: str | None = None,
    voronoi_spot_sigma: float = 2.0,
    voronoi_outline_sigma: float = 2.0,
    distance_transform_dynamic: float = 1.0,
    distance_transform_connectivity: int = 6,
    distance_transform_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity_prominence: float = 0.15,
    intensity_smoothing_sigma: float = 0.0,
    intensity_percentiles: tuple[float, float] = (1.0, 99.0),
    intensity_connectivity: int = 6,
) -> Path:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind: {source_kind}.")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    normalized_data_operations = normalize_data_operations(data_operations)
    normalized_mask_operations = normalize_mask_operations(mask_operations)
    if data_backend not in DATA_BACKENDS:
        raise ValueError(f"Unsupported data backend: {data_backend}.")
    if data_backend == DATA_BACKEND_GPU and normalized_data_operations and gpu_index is None:
        raise ValueError("GPU data processing requires --gpu-index.")
    resolved_instance_method = (
        normalize_instance_method(instance_method)
        if instance_method is not None
        else ("connected_components" if run_ccl else "none")
    )
    if not np.isfinite(voronoi_spot_sigma) or voronoi_spot_sigma < 0:
        raise ValueError("Voronoi-Otsu spot sigma must be nonnegative.")
    if not np.isfinite(voronoi_outline_sigma) or voronoi_outline_sigma < 0:
        raise ValueError("Voronoi-Otsu outline sigma must be nonnegative.")
    normalized_distance_dynamic = normalize_distance_transform_dynamic(distance_transform_dynamic)
    normalized_distance_connectivity = normalize_distance_transform_connectivity(distance_transform_connectivity)
    normalized_distance_spacing = normalize_distance_transform_spacing(*distance_transform_spacing)
    normalized_intensity_prominence = normalize_intensity_prominence(intensity_prominence)
    normalized_intensity_smoothing_sigma = normalize_intensity_smoothing_sigma(intensity_smoothing_sigma)
    normalized_intensity_percentiles = normalize_intensity_normalization_percentiles(*intensity_percentiles)
    normalized_intensity_connectivity = normalize_watershed_connectivity(
        intensity_connectivity,
        label="Intensity-prominence watershed connectivity",
    )
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_path}")
    if output_path.exists() and not output_path.is_dir():
        raise FileNotFoundError(f"Output folder exists but is not a directory: {output_path}")

    raw_paths = named_file_map(inference_raw_dir(input_path), suffixes=TIFF_SUFFIXES)
    if not raw_paths:
        raise FileNotFoundError(f"{inference_raw_dir(input_path)} contains no TIFF files.")

    source_dir = source_directory(input_path, source_kind=source_kind, source_folder=source_folder)
    source_paths = named_file_map(source_dir, suffixes=ARRAY_SUFFIXES)
    if not source_paths:
        raise FileNotFoundError(f"{source_dir} contains no supported source files.")

    missing_source = sorted(set(raw_paths) - set(source_paths), key=natural_sort_key)
    if missing_source:
        missing_name = missing_source[0]
        raise FileNotFoundError(f"Source folder is missing {missing_name}.")

    missing_raw = sorted(set(source_paths) - set(raw_paths), key=natural_sort_key)
    if missing_raw:
        missing_name = missing_raw[0]
        raise FileNotFoundError(f"Raw folder is missing {missing_name}.tif.")

    output_root = segmentation_general_dir(output_path)
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    names = sorted(raw_paths, key=natural_sort_key)
    print(f"[segmentation] Found {len(names)} timepoints", flush=True)
    if normalized_data_operations:
        backend_label = data_backend if data_backend == DATA_BACKEND_CPU else f"{data_backend}:{gpu_index}"
        print(f"[segmentation] Data processing backend: {backend_label}", flush=True)
    for index, timepoint_name in enumerate(names, start=1):
        print(f"[segmentation] Processing {timepoint_name} ({index}/{len(names)})", flush=True)
        values = read_scalar_volume(source_paths[timepoint_name], source_kind=source_kind, component_index=component_index)
        processed_values = apply_data_operations(
            values,
            normalized_data_operations,
            source_kind=source_kind,
            backend=data_backend,
            gpu_index=gpu_index,
        )
        semantic = threshold_to_semantic(processed_values, threshold=threshold, invert_mask=invert_mask)
        semantic = apply_mask_operations(semantic, normalized_mask_operations)
        output_array = instance_segmentation(
            processed_values,
            semantic,
            method=resolved_instance_method,
            voronoi_spot_sigma=voronoi_spot_sigma,
            voronoi_outline_sigma=voronoi_outline_sigma,
            distance_dynamic=normalized_distance_dynamic,
            distance_connectivity=normalized_distance_connectivity,
            distance_spacing_zyx=normalized_distance_spacing,
            intensity_prominence=normalized_intensity_prominence,
            intensity_smoothing_sigma=normalized_intensity_smoothing_sigma,
            intensity_low_percentile=normalized_intensity_percentiles[0],
            intensity_high_percentile=normalized_intensity_percentiles[1],
            intensity_connectivity=normalized_intensity_connectivity,
        )
        mask_path = output_root / f"{timepoint_name}.tif"
        tifffile.imwrite(mask_path, output_array, bigtiff=True, metadata=None, photometric="minisblack")
        print(f"[segmentation] Completed {timepoint_name}", flush=True)

    print("[segmentation] Done", flush=True)
    return output_root


def main() -> None:
    args = parse_args()
    data_operations = normalize_data_operations(json.loads(args.data_operations_json))
    mask_operations = normalize_mask_operations(json.loads(args.mask_operations_json))
    instance_method = normalize_instance_method(args.instance_method)
    segment_general(
        Path(args.input_path).expanduser().resolve(),
        output_path=Path(args.output_path).expanduser().resolve(),
        source_kind=str(args.source_kind),
        source_folder=args.source_folder,
        threshold=float(args.threshold),
        component_index=int(args.component_index),
        invert_mask=bool(args.invert_mask),
        run_ccl=bool(args.run_ccl),
        data_operations=data_operations,
        mask_operations=mask_operations,
        data_backend=str(args.data_backend),
        gpu_index=args.gpu_index,
        instance_method=instance_method,
        voronoi_spot_sigma=float(args.voronoi_spot_sigma),
        voronoi_outline_sigma=float(args.voronoi_outline_sigma),
        distance_transform_dynamic=float(args.distance_transform_dynamic),
        distance_transform_connectivity=int(args.distance_transform_connectivity),
        distance_transform_spacing=(
            float(args.distance_transform_spacing_z),
            float(args.distance_transform_spacing_y),
            float(args.distance_transform_spacing_x),
        ),
        intensity_prominence=float(args.intensity_prominence),
        intensity_smoothing_sigma=float(args.intensity_smoothing_sigma),
        intensity_percentiles=(
            float(args.intensity_low_percentile),
            float(args.intensity_high_percentile),
        ),
        intensity_connectivity=int(args.intensity_connectivity),
    )


if __name__ == "__main__":
    main()
