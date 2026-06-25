from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima, local_maxima
from skimage.segmentation import watershed


DATA_OPERATION_INVERT_LUT = "invert_lut"
DATA_OPERATION_SUBTRACT_BACKGROUND = "subtract_background"
DATA_OPERATION_GAUSSIAN_SMOOTHING = "gaussian_smoothing"
DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN = "laplacian_of_gaussian"
DATA_OPERATION_PERCENTILE_CLIPPING = "percentile_clipping"
DATA_OPERATION_MEDIAN_FILTER = "median_filter"
DATA_OPERATION_DIFFERENCE_OF_GAUSSIANS = "difference_of_gaussians"
DATA_OPERATION_TOP_HAT = "top_hat"
DATA_OPERATION_BLACK_HAT = "black_hat"
DATA_OPERATION_TYPES = {
    DATA_OPERATION_INVERT_LUT,
    DATA_OPERATION_SUBTRACT_BACKGROUND,
    DATA_OPERATION_GAUSSIAN_SMOOTHING,
    DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN,
    DATA_OPERATION_PERCENTILE_CLIPPING,
    DATA_OPERATION_MEDIAN_FILTER,
    DATA_OPERATION_DIFFERENCE_OF_GAUSSIANS,
    DATA_OPERATION_TOP_HAT,
    DATA_OPERATION_BLACK_HAT,
}
DATA_BACKEND_CPU = "cpu"
DATA_BACKEND_GPU = "gpu"
DATA_BACKENDS = {DATA_BACKEND_CPU, DATA_BACKEND_GPU}
LOG_RESPONSE_BRIGHT = "bright"
LOG_RESPONSE_DARK = "dark"
LOG_RESPONSE_TYPES = {LOG_RESPONSE_BRIGHT, LOG_RESPONSE_DARK}

MASK_OPERATION_REMOVE_SMALL_OBJECTS = "remove_small_objects"
MASK_OPERATION_FILL_SMALL_HOLES = "fill_small_holes"
MASK_OPERATION_BINARY_CLOSING = "binary_closing"
MASK_OPERATION_BINARY_OPENING = "binary_opening"
MASK_OPERATION_DILATE = "dilate"
MASK_OPERATION_ERODE = "erode"
MASK_OPERATION_REMOVE_BORDER_OBJECTS = "remove_border_objects"
MASK_OPERATION_SIZE_RANGE = "size_range"
MASK_OPERATION_TYPES = {
    MASK_OPERATION_REMOVE_SMALL_OBJECTS,
    MASK_OPERATION_FILL_SMALL_HOLES,
    MASK_OPERATION_BINARY_CLOSING,
    MASK_OPERATION_BINARY_OPENING,
    MASK_OPERATION_DILATE,
    MASK_OPERATION_ERODE,
    MASK_OPERATION_REMOVE_BORDER_OBJECTS,
    MASK_OPERATION_SIZE_RANGE,
}

INSTANCE_METHOD_NONE = "none"
INSTANCE_METHOD_CONNECTED_COMPONENTS = "connected_components"
INSTANCE_METHOD_VORONOI_OTSU = "voronoi_otsu"
INSTANCE_METHOD_DISTANCE_TRANSFORM_WATERSHED = "distance_transform_watershed"
INSTANCE_METHOD_INTENSITY_PROMINENCE_WATERSHED = "intensity_prominence_watershed"
INSTANCE_METHODS = {
    INSTANCE_METHOD_NONE,
    INSTANCE_METHOD_CONNECTED_COMPONENTS,
    INSTANCE_METHOD_VORONOI_OTSU,
    INSTANCE_METHOD_DISTANCE_TRANSFORM_WATERSHED,
    INSTANCE_METHOD_INTENSITY_PROMINENCE_WATERSHED,
}
DISTANCE_TRANSFORM_CONNECTIVITIES = {6, 26}
WATERSHED_CONNECTIVITIES = {6, 26}

SOURCE_KIND_PROBMAP = "probmap"
SOURCE_KIND_PCA = "pca"


def finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return 0.0, 0.0
    finite_values = np.asarray(values[finite], dtype=np.float32)
    return float(np.min(finite_values)), float(np.max(finite_values))


def finite_float32(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    low, high = finite_min_max(array)
    return np.nan_to_num(array, nan=low, neginf=low, posinf=high).astype(np.float32, copy=False)


@lru_cache(maxsize=32)
def spherical_footprint(radius_key: int) -> np.ndarray:
    radius = max(1, int(radius_key))
    coords = np.ogrid[
        -radius : radius + 1,
        -radius : radius + 1,
        -radius : radius + 1,
    ]
    squared_distance = sum(axis.astype(np.float32) ** 2 for axis in coords)
    return np.asarray(squared_distance <= float(radius * radius), dtype=bool)


@lru_cache(maxsize=128)
def ellipsoid_footprint(radius_z_key: int, radius_y_key: int, radius_x_key: int) -> np.ndarray:
    radius_z = max(0, int(radius_z_key))
    radius_y = max(0, int(radius_y_key))
    radius_x = max(0, int(radius_x_key))
    if radius_z == 0 and radius_y == 0 and radius_x == 0:
        return np.ones((1, 1, 1), dtype=bool)

    z_coords, y_coords, x_coords = np.ogrid[
        -radius_z : radius_z + 1,
        -radius_y : radius_y + 1,
        -radius_x : radius_x + 1,
    ]
    squared_distance = np.zeros((2 * radius_z + 1, 2 * radius_y + 1, 2 * radius_x + 1), dtype=np.float32)
    for coords, radius in ((z_coords, radius_z), (y_coords, radius_y), (x_coords, radius_x)):
        if radius <= 0:
            squared_distance = squared_distance + np.where(coords == 0, 0.0, np.inf)
        else:
            squared_distance = squared_distance + (coords.astype(np.float32) / float(radius)) ** 2
    return np.asarray(squared_distance <= 1.0, dtype=bool)


def _finite_nonnegative(value: Any, *, label: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be nonnegative.")
    return parsed


def _finite_float(value: Any, *, label: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _axis_values_from_raw(
    raw_operation: dict[str, Any],
    *,
    base_key: str,
    default: float,
    label: str,
) -> tuple[bool, float, float, float]:
    axis_keys = (f"{base_key}_z", f"{base_key}_y", f"{base_key}_x")
    isotropic = _finite_nonnegative(raw_operation.get(base_key, default), label=label)
    has_axis_values = any(axis_key in raw_operation for axis_key in axis_keys)
    if not has_axis_values:
        return False, isotropic, isotropic, isotropic
    z_value = _finite_nonnegative(raw_operation.get(axis_keys[0], isotropic), label=f"{label} Z")
    y_value = _finite_nonnegative(raw_operation.get(axis_keys[1], isotropic), label=f"{label} Y")
    x_value = _finite_nonnegative(raw_operation.get(axis_keys[2], isotropic), label=f"{label} X")
    return True, z_value, y_value, x_value


def _axis_values_from_operation(operation: dict[str, Any], *, base_key: str) -> tuple[float, float, float]:
    axis_keys = (f"{base_key}_z", f"{base_key}_y", f"{base_key}_x")
    if all(axis_key in operation for axis_key in axis_keys):
        return float(operation[axis_keys[0]]), float(operation[axis_keys[1]]), float(operation[axis_keys[2]])
    value = float(operation[base_key])
    return value, value, value


def _append_scale_operation(
    operations: list[dict[str, Any]],
    *,
    operation_type: str,
    base_key: str,
    default: float,
    label: str,
    raw_operation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    has_axis_values, z_value, y_value, x_value = _axis_values_from_raw(
        raw_operation,
        base_key=base_key,
        default=default,
        label=label,
    )
    operation: dict[str, Any] = {"type": operation_type}
    if has_axis_values:
        operation.update({
            f"{base_key}_z": z_value,
            f"{base_key}_y": y_value,
            f"{base_key}_x": x_value,
        })
    else:
        operation[base_key] = z_value
    if extra:
        operation.update(extra)
    operations.append(operation)


def _footprint_from_radii(radius_z: float, radius_y: float, radius_x: float) -> np.ndarray:
    return ellipsoid_footprint(int(np.ceil(radius_z)), int(np.ceil(radius_y)), int(np.ceil(radius_x)))


def _data_operation_axis_triplet(operation: dict[str, Any], *, base_key: str) -> tuple[float, float, float]:
    return _axis_values_from_operation(operation, base_key=base_key)


def normalize_data_operations(raw_operations: Any) -> list[dict[str, Any]]:
    if raw_operations is None:
        return []
    if not isinstance(raw_operations, list):
        raise ValueError("Data operations must be a list.")

    operations: list[dict[str, Any]] = []
    for index, raw_operation in enumerate(raw_operations, start=1):
        if not isinstance(raw_operation, dict):
            raise ValueError(f"Data operation {index} must be an object.")
        operation_type = str(raw_operation.get("type", "")).strip()
        if operation_type not in DATA_OPERATION_TYPES:
            raise ValueError(f"Data operation {index} has an unsupported type: {operation_type}.")

        if operation_type == DATA_OPERATION_INVERT_LUT:
            operations.append({"type": operation_type})
        elif operation_type == DATA_OPERATION_SUBTRACT_BACKGROUND:
            _append_scale_operation(
                operations,
                operation_type=operation_type,
                base_key="radius",
                default=10.0,
                label=f"Data operation {index} rolling-ball radius",
                raw_operation=raw_operation,
            )
        elif operation_type == DATA_OPERATION_GAUSSIAN_SMOOTHING:
            _append_scale_operation(
                operations,
                operation_type=operation_type,
                base_key="sigma",
                default=1.0,
                label=f"Data operation {index} Gaussian sigma",
                raw_operation=raw_operation,
            )
        elif operation_type == DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN:
            response = str(raw_operation.get("response", LOG_RESPONSE_BRIGHT)).strip()
            if response not in LOG_RESPONSE_TYPES:
                raise ValueError(f"Data operation {index} LoG response must be bright or dark.")
            _append_scale_operation(
                operations,
                operation_type=operation_type,
                base_key="sigma",
                default=1.0,
                label=f"Data operation {index} LoG sigma",
                raw_operation=raw_operation,
                extra={"response": response},
            )
        elif operation_type == DATA_OPERATION_PERCENTILE_CLIPPING:
            low_percentile = _finite_float(
                raw_operation.get("low_percentile", 1.0),
                label=f"Data operation {index} low percentile",
            )
            high_percentile = _finite_float(
                raw_operation.get("high_percentile", 99.0),
                label=f"Data operation {index} high percentile",
            )
            if low_percentile < 0 or low_percentile > 100:
                raise ValueError(f"Data operation {index} low percentile must be between 0 and 100.")
            if high_percentile < 0 or high_percentile > 100:
                raise ValueError(f"Data operation {index} high percentile must be between 0 and 100.")
            if low_percentile >= high_percentile:
                raise ValueError(f"Data operation {index} low percentile must be smaller than high percentile.")
            rescale = bool(raw_operation.get("rescale", False))
            output_min = _finite_float(
                raw_operation.get("output_min", 0.0),
                label=f"Data operation {index} output minimum",
            )
            output_max = _finite_float(
                raw_operation.get("output_max", 1.0),
                label=f"Data operation {index} output maximum",
            )
            if rescale and output_min >= output_max:
                raise ValueError(f"Data operation {index} output minimum must be smaller than output maximum.")
            operations.append(
                {
                    "type": operation_type,
                    "low_percentile": low_percentile,
                    "high_percentile": high_percentile,
                    "rescale": rescale,
                    "output_min": output_min,
                    "output_max": output_max,
                }
            )
        elif operation_type == DATA_OPERATION_MEDIAN_FILTER:
            _append_scale_operation(
                operations,
                operation_type=operation_type,
                base_key="radius",
                default=1.0,
                label=f"Data operation {index} median radius",
                raw_operation=raw_operation,
            )
        elif operation_type == DATA_OPERATION_DIFFERENCE_OF_GAUSSIANS:
            response = str(raw_operation.get("response", LOG_RESPONSE_BRIGHT)).strip()
            if response not in LOG_RESPONSE_TYPES:
                raise ValueError(f"Data operation {index} DoG response must be bright or dark.")
            small_has_axis, small_z, small_y, small_x = _axis_values_from_raw(
                raw_operation,
                base_key="sigma",
                default=1.0,
                label=f"Data operation {index} DoG small sigma",
            )
            large_has_axis, large_z, large_y, large_x = _axis_values_from_raw(
                raw_operation,
                base_key="sigma2",
                default=2.0,
                label=f"Data operation {index} DoG large sigma",
            )
            operation = {"type": operation_type, "response": response}
            if small_has_axis:
                operation.update({"sigma_z": small_z, "sigma_y": small_y, "sigma_x": small_x})
            else:
                operation["sigma"] = small_z
            if large_has_axis:
                operation.update({"sigma2_z": large_z, "sigma2_y": large_y, "sigma2_x": large_x})
            else:
                operation["sigma2"] = large_z
            operations.append(operation)
        elif operation_type in {DATA_OPERATION_TOP_HAT, DATA_OPERATION_BLACK_HAT}:
            _append_scale_operation(
                operations,
                operation_type=operation_type,
                base_key="radius",
                default=10.0,
                label=f"Data operation {index} hat radius",
                raw_operation=raw_operation,
            )

    return operations


def normalize_mask_operations(raw_operations: Any) -> list[dict[str, Any]]:
    if raw_operations is None:
        return []
    if not isinstance(raw_operations, list):
        raise ValueError("Mask operations must be a list.")

    operations: list[dict[str, Any]] = []
    for index, raw_operation in enumerate(raw_operations, start=1):
        if not isinstance(raw_operation, dict):
            raise ValueError(f"Mask operation {index} must be an object.")
        operation_type = str(raw_operation.get("type", "")).strip()
        if operation_type not in MASK_OPERATION_TYPES:
            raise ValueError(f"Mask operation {index} has an unsupported type: {operation_type}.")
        if operation_type in {MASK_OPERATION_REMOVE_SMALL_OBJECTS, MASK_OPERATION_FILL_SMALL_HOLES}:
            size = int(raw_operation.get("size", 64))
            if size < 0:
                raise ValueError(f"Mask operation {index} size must be nonnegative.")
            operations.append({"type": operation_type, "size": size})
        elif operation_type in {
            MASK_OPERATION_BINARY_CLOSING,
            MASK_OPERATION_BINARY_OPENING,
            MASK_OPERATION_DILATE,
            MASK_OPERATION_ERODE,
        }:
            radius = float(raw_operation.get("radius", 1.0))
            if not np.isfinite(radius) or radius < 0:
                raise ValueError(f"Mask operation {index} radius must be nonnegative.")
            operations.append({"type": operation_type, "radius": radius})
        elif operation_type == MASK_OPERATION_REMOVE_BORDER_OBJECTS:
            operations.append({"type": operation_type})
        elif operation_type == MASK_OPERATION_SIZE_RANGE:
            min_size = int(raw_operation.get("min_size", 0))
            max_size = int(raw_operation.get("max_size", 0))
            if min_size < 0:
                raise ValueError(f"Mask operation {index} minimum size must be nonnegative.")
            if max_size < 0:
                raise ValueError(f"Mask operation {index} maximum size must be nonnegative.")
            if max_size > 0 and min_size > max_size:
                raise ValueError(f"Mask operation {index} minimum size must be smaller than or equal to maximum size.")
            operations.append({"type": operation_type, "min_size": min_size, "max_size": max_size})

    return operations


def normalize_instance_method(raw_method: Any) -> str:
    method = str(raw_method or INSTANCE_METHOD_CONNECTED_COMPONENTS).strip()
    if method not in INSTANCE_METHODS:
        raise ValueError(f"Unsupported instance segmentation method: {method}.")
    return method


def normalize_distance_transform_dynamic(raw_dynamic: Any) -> float:
    return _finite_nonnegative(raw_dynamic if raw_dynamic is not None else 1.0, label="Distance-transform dynamic")


def normalize_distance_transform_connectivity(raw_connectivity: Any) -> int:
    connectivity = int(raw_connectivity if raw_connectivity is not None else 6)
    if connectivity not in DISTANCE_TRANSFORM_CONNECTIVITIES:
        raise ValueError("Distance-transform connectivity must be 6 or 26.")
    return connectivity


def normalize_distance_transform_spacing(
    raw_spacing_z: Any,
    raw_spacing_y: Any,
    raw_spacing_x: Any,
) -> tuple[float, float, float]:
    spacing_z = _finite_float(raw_spacing_z if raw_spacing_z is not None else 1.0, label="Distance-transform Z spacing")
    spacing_y = _finite_float(raw_spacing_y if raw_spacing_y is not None else 1.0, label="Distance-transform Y spacing")
    spacing_x = _finite_float(raw_spacing_x if raw_spacing_x is not None else 1.0, label="Distance-transform X spacing")
    if spacing_z <= 0 or spacing_y <= 0 or spacing_x <= 0:
        raise ValueError("Distance-transform spacing values must be positive.")
    return spacing_z, spacing_y, spacing_x


def normalize_watershed_connectivity(raw_connectivity: Any, *, label: str = "Watershed connectivity") -> int:
    connectivity = int(raw_connectivity if raw_connectivity is not None else 6)
    if connectivity not in WATERSHED_CONNECTIVITIES:
        raise ValueError(f"{label} must be 6 or 26.")
    return connectivity


def normalize_intensity_prominence(raw_prominence: Any) -> float:
    prominence = _finite_nonnegative(
        raw_prominence if raw_prominence is not None else 0.15,
        label="Intensity-prominence watershed prominence",
    )
    if prominence > 1:
        raise ValueError("Intensity-prominence watershed prominence must be between 0 and 1.")
    return prominence


def normalize_intensity_smoothing_sigma(raw_sigma: Any) -> float:
    return _finite_nonnegative(
        raw_sigma if raw_sigma is not None else 0.0,
        label="Intensity-prominence watershed smoothing sigma",
    )


def normalize_intensity_normalization_percentiles(
    raw_low_percentile: Any,
    raw_high_percentile: Any,
) -> tuple[float, float]:
    low_percentile = _finite_float(
        raw_low_percentile if raw_low_percentile is not None else 1.0,
        label="Intensity-prominence watershed low percentile",
    )
    high_percentile = _finite_float(
        raw_high_percentile if raw_high_percentile is not None else 99.0,
        label="Intensity-prominence watershed high percentile",
    )
    if low_percentile < 0 or low_percentile > 100:
        raise ValueError("Intensity-prominence watershed low percentile must be between 0 and 100.")
    if high_percentile < 0 or high_percentile > 100:
        raise ValueError("Intensity-prominence watershed high percentile must be between 0 and 100.")
    if low_percentile >= high_percentile:
        raise ValueError("Intensity-prominence watershed low percentile must be smaller than high percentile.")
    return low_percentile, high_percentile


def invert_lut(values: np.ndarray, *, source_kind: str) -> np.ndarray:
    if source_kind == SOURCE_KIND_PROBMAP:
        low, high = 0.0, 1.0
    elif source_kind == SOURCE_KIND_PCA:
        low, high = 0.0, 255.0
    else:
        low, high = finite_min_max(values)
    return np.asarray((low + high) - values, dtype=np.float32)


def subtract_background(values: np.ndarray, *, radius_z: float, radius_y: float, radius_x: float) -> np.ndarray:
    if radius_z <= 0 and radius_y <= 0 and radius_x <= 0:
        return np.asarray(values, dtype=np.float32)
    footprint = _footprint_from_radii(radius_z, radius_y, radius_x)
    background = ndimage.grey_opening(values, footprint=footprint)
    return np.asarray(values - background, dtype=np.float32)


def percentile_clip_or_rescale(
    values: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
    rescale: bool,
    output_min: float,
    output_max: float,
) -> np.ndarray:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite_values.size == 0:
        return np.asarray(values, dtype=np.float32)
    low_value, high_value = np.percentile(finite_values, [low_percentile, high_percentile])
    clipped = np.asarray(np.clip(values, low_value, high_value), dtype=np.float32)
    if not rescale:
        return clipped
    denominator = float(high_value - low_value)
    if denominator <= 0:
        return np.full(clipped.shape, float(output_min), dtype=np.float32)
    scaled = (clipped - float(low_value)) / denominator
    return np.asarray(scaled * float(output_max - output_min) + float(output_min), dtype=np.float32)


def median_filter(values: np.ndarray, *, radius_z: float, radius_y: float, radius_x: float) -> np.ndarray:
    if radius_z <= 0 and radius_y <= 0 and radius_x <= 0:
        return np.asarray(values, dtype=np.float32)
    footprint = _footprint_from_radii(radius_z, radius_y, radius_x)
    return np.asarray(ndimage.median_filter(values, footprint=footprint), dtype=np.float32)


def difference_of_gaussians(
    values: np.ndarray,
    *,
    sigma_z: float,
    sigma_y: float,
    sigma_x: float,
    sigma2_z: float,
    sigma2_y: float,
    sigma2_x: float,
    response: str,
) -> np.ndarray:
    small = np.asarray(ndimage.gaussian_filter(values, sigma=(sigma_z, sigma_y, sigma_x)), dtype=np.float32)
    large = np.asarray(ndimage.gaussian_filter(values, sigma=(sigma2_z, sigma2_y, sigma2_x)), dtype=np.float32)
    dog = np.asarray(small - large, dtype=np.float32)
    return -dog if response == LOG_RESPONSE_DARK else dog


def top_hat(values: np.ndarray, *, radius_z: float, radius_y: float, radius_x: float) -> np.ndarray:
    if radius_z <= 0 and radius_y <= 0 and radius_x <= 0:
        return np.asarray(values, dtype=np.float32)
    footprint = _footprint_from_radii(radius_z, radius_y, radius_x)
    opened = ndimage.grey_opening(values, footprint=footprint)
    return np.asarray(values - opened, dtype=np.float32)


def black_hat(values: np.ndarray, *, radius_z: float, radius_y: float, radius_x: float) -> np.ndarray:
    if radius_z <= 0 and radius_y <= 0 and radius_x <= 0:
        return np.asarray(values, dtype=np.float32)
    footprint = _footprint_from_radii(radius_z, radius_y, radius_x)
    closed = ndimage.grey_closing(values, footprint=footprint)
    return np.asarray(closed - values, dtype=np.float32)


def _apply_data_operations_cpu(
    values: np.ndarray,
    operations: list[dict[str, Any]],
    *,
    source_kind: str,
) -> np.ndarray:
    processed = finite_float32(values)
    for operation in operations:
        operation_type = str(operation["type"])
        if operation_type == DATA_OPERATION_INVERT_LUT:
            processed = invert_lut(processed, source_kind=source_kind)
        elif operation_type == DATA_OPERATION_SUBTRACT_BACKGROUND:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            processed = subtract_background(
                processed,
                radius_z=radius_z,
                radius_y=radius_y,
                radius_x=radius_x,
            )
        elif operation_type == DATA_OPERATION_GAUSSIAN_SMOOTHING:
            sigma_z, sigma_y, sigma_x = _data_operation_axis_triplet(operation, base_key="sigma")
            processed = np.asarray(ndimage.gaussian_filter(processed, sigma=(sigma_z, sigma_y, sigma_x)), dtype=np.float32)
        elif operation_type == DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN:
            sigma_z, sigma_y, sigma_x = _data_operation_axis_triplet(operation, base_key="sigma")
            log_values = np.asarray(ndimage.gaussian_laplace(processed, sigma=(sigma_z, sigma_y, sigma_x)), dtype=np.float32)
            processed = -log_values if operation.get("response") == LOG_RESPONSE_BRIGHT else log_values
        elif operation_type == DATA_OPERATION_PERCENTILE_CLIPPING:
            processed = percentile_clip_or_rescale(
                processed,
                low_percentile=float(operation["low_percentile"]),
                high_percentile=float(operation["high_percentile"]),
                rescale=bool(operation["rescale"]),
                output_min=float(operation["output_min"]),
                output_max=float(operation["output_max"]),
            )
        elif operation_type == DATA_OPERATION_MEDIAN_FILTER:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            processed = median_filter(processed, radius_z=radius_z, radius_y=radius_y, radius_x=radius_x)
        elif operation_type == DATA_OPERATION_DIFFERENCE_OF_GAUSSIANS:
            sigma_z, sigma_y, sigma_x = _data_operation_axis_triplet(operation, base_key="sigma")
            sigma2_z, sigma2_y, sigma2_x = _data_operation_axis_triplet(operation, base_key="sigma2")
            processed = difference_of_gaussians(
                processed,
                sigma_z=sigma_z,
                sigma_y=sigma_y,
                sigma_x=sigma_x,
                sigma2_z=sigma2_z,
                sigma2_y=sigma2_y,
                sigma2_x=sigma2_x,
                response=str(operation.get("response", LOG_RESPONSE_BRIGHT)),
            )
        elif operation_type == DATA_OPERATION_TOP_HAT:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            processed = top_hat(processed, radius_z=radius_z, radius_y=radius_y, radius_x=radius_x)
        elif operation_type == DATA_OPERATION_BLACK_HAT:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            processed = black_hat(processed, radius_z=radius_z, radius_y=radius_y, radius_x=radius_x)
        else:
            raise ValueError(f"Unsupported data operation: {operation_type}.")
    return finite_float32(processed)


def _import_pyclesperanto() -> Any:
    try:
        import pyclesperanto as cle
    except Exception as exc:  # pragma: no cover - exercised only when GPU backend is requested.
        raise RuntimeError("pyclesperanto is required for GPU data processing.") from exc
    return cle


def _select_cle_device(cle: Any, gpu_index: int | None) -> Any:
    if gpu_index is None:
        return cle.get_device()
    try:
        return cle.select_device(int(gpu_index))
    except Exception:
        return cle.select_device(f"gpu:{int(gpu_index)}")


def _cle_min_max(cle: Any, image: Any) -> tuple[float, float]:
    return float(cle.minimum_of_all_pixels(image)), float(cle.maximum_of_all_pixels(image))


def _apply_data_operations_gpu(
    values: np.ndarray,
    operations: list[dict[str, Any]],
    *,
    source_kind: str,
    gpu_index: int | None,
) -> np.ndarray:
    cle = _import_pyclesperanto()
    cle_device = _select_cle_device(cle, gpu_index)
    processed = cle.push(finite_float32(values), device=cle_device)

    for operation in operations:
        operation_type = str(operation["type"])
        if operation_type == DATA_OPERATION_INVERT_LUT:
            if source_kind == SOURCE_KIND_PROBMAP:
                low, high = 0.0, 1.0
            elif source_kind == SOURCE_KIND_PCA:
                low, high = 0.0, 255.0
            else:
                low, high = _cle_min_max(cle, processed)
            processed = cle.subtract_image_from_scalar(processed, scalar=float(low + high), device=cle_device)
        elif operation_type == DATA_OPERATION_SUBTRACT_BACKGROUND:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            if radius_z > 0 or radius_y > 0 or radius_x > 0:
                background = cle.opening_sphere(
                    processed,
                    radius_x=radius_x,
                    radius_y=radius_y,
                    radius_z=radius_z,
                    device=cle_device,
                )
                processed = cle.subtract_images(processed, background, device=cle_device)
        elif operation_type == DATA_OPERATION_GAUSSIAN_SMOOTHING:
            sigma_z, sigma_y, sigma_x = _data_operation_axis_triplet(operation, base_key="sigma")
            processed = cle.gaussian_blur(
                processed,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                sigma_z=sigma_z,
                device=cle_device,
            )
        elif operation_type == DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN:
            sigma_z, sigma_y, sigma_x = _data_operation_axis_triplet(operation, base_key="sigma")
            smoothed = cle.gaussian_blur(
                processed,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                sigma_z=sigma_z,
                device=cle_device,
            )
            processed = cle.laplace_box(smoothed, device=cle_device)
            if operation.get("response") == LOG_RESPONSE_BRIGHT:
                processed = cle.multiply_image_and_scalar(processed, scalar=-1.0, device=cle_device)
        elif operation_type == DATA_OPERATION_PERCENTILE_CLIPPING:
            low_value = float(cle.percentile(processed, percentile=float(operation["low_percentile"]), device=cle_device))
            high_value = float(cle.percentile(processed, percentile=float(operation["high_percentile"]), device=cle_device))
            processed = cle.clip(processed, min_intensity=low_value, max_intensity=high_value, device=cle_device)
            if bool(operation["rescale"]):
                denominator = high_value - low_value
                output_min = float(operation["output_min"])
                output_max = float(operation["output_max"])
                if denominator <= 0:
                    processed = cle.multiply_image_and_scalar(processed, scalar=0.0, device=cle_device)
                    processed = cle.add_image_and_scalar(processed, scalar=output_min, device=cle_device)
                else:
                    processed = cle.add_image_and_scalar(processed, scalar=-low_value, device=cle_device)
                    processed = cle.multiply_image_and_scalar(
                        processed,
                        scalar=(output_max - output_min) / denominator,
                        device=cle_device,
                    )
                    processed = cle.add_image_and_scalar(processed, scalar=output_min, device=cle_device)
        elif operation_type == DATA_OPERATION_MEDIAN_FILTER:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            if radius_z > 0 or radius_y > 0 or radius_x > 0:
                processed = cle.median_sphere(
                    processed,
                    radius_x=radius_x,
                    radius_y=radius_y,
                    radius_z=radius_z,
                    device=cle_device,
                )
        elif operation_type == DATA_OPERATION_DIFFERENCE_OF_GAUSSIANS:
            sigma_z, sigma_y, sigma_x = _data_operation_axis_triplet(operation, base_key="sigma")
            sigma2_z, sigma2_y, sigma2_x = _data_operation_axis_triplet(operation, base_key="sigma2")
            processed = cle.difference_of_gaussian(
                processed,
                sigma1_x=sigma_x,
                sigma1_y=sigma_y,
                sigma1_z=sigma_z,
                sigma2_x=sigma2_x,
                sigma2_y=sigma2_y,
                sigma2_z=sigma2_z,
                device=cle_device,
            )
            if operation.get("response") == LOG_RESPONSE_DARK:
                processed = cle.multiply_image_and_scalar(processed, scalar=-1.0, device=cle_device)
        elif operation_type == DATA_OPERATION_TOP_HAT:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            if radius_z > 0 or radius_y > 0 or radius_x > 0:
                processed = cle.top_hat_sphere(
                    processed,
                    radius_x=radius_x,
                    radius_y=radius_y,
                    radius_z=radius_z,
                    device=cle_device,
                )
        elif operation_type == DATA_OPERATION_BLACK_HAT:
            radius_z, radius_y, radius_x = _data_operation_axis_triplet(operation, base_key="radius")
            if radius_z > 0 or radius_y > 0 or radius_x > 0:
                closed = cle.closing_sphere(
                    processed,
                    radius_x=radius_x,
                    radius_y=radius_y,
                    radius_z=radius_z,
                    device=cle_device,
                )
                processed = cle.subtract_images(closed, processed, device=cle_device)
        else:
            raise ValueError(f"Unsupported data operation: {operation_type}.")

    return finite_float32(np.asarray(cle.pull(processed)))


def apply_data_operations(
    values: np.ndarray,
    operations: list[dict[str, Any]],
    *,
    source_kind: str,
    backend: str = DATA_BACKEND_CPU,
    gpu_index: int | None = None,
) -> np.ndarray:
    if backend not in DATA_BACKENDS:
        raise ValueError(f"Unsupported data processing backend: {backend}.")
    if backend == DATA_BACKEND_GPU and operations:
        return _apply_data_operations_gpu(values, operations, source_kind=source_kind, gpu_index=gpu_index)
    return _apply_data_operations_cpu(values, operations, source_kind=source_kind)


def threshold_to_semantic(values: np.ndarray, *, threshold: float, invert_mask: bool) -> np.ndarray:
    finite = np.isfinite(values)
    if invert_mask:
        return np.asarray(finite & (values < threshold), dtype=bool)
    return np.asarray(finite & (values >= threshold), dtype=bool)


def connected_components(mask: np.ndarray) -> np.ndarray:
    labeled, _ = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    return np.asarray(labeled, dtype=np.uint32)


def remove_small_objects(mask: np.ndarray, *, size: int) -> np.ndarray:
    if size <= 0:
        return np.asarray(mask, dtype=bool)
    labels = connected_components(mask)
    if int(labels.max()) == 0:
        return np.zeros(labels.shape, dtype=bool)
    counts = np.bincount(labels.reshape(-1))
    keep = counts >= int(size)
    keep[0] = False
    return np.asarray(keep[labels], dtype=bool)


def _border_label_ids(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros(0, dtype=labels.dtype)
    border_values = [
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ]
    return np.unique(np.concatenate([np.ravel(values) for values in border_values]))


def fill_small_holes(mask: np.ndarray, *, size: int) -> np.ndarray:
    processed = np.asarray(mask, dtype=bool)
    if size <= 0:
        return processed
    labels = connected_components(~processed)
    if int(labels.max()) == 0:
        return processed
    counts = np.bincount(labels.reshape(-1))
    border_ids = _border_label_ids(labels)
    fill = counts <= int(size)
    fill[0] = False
    fill[border_ids] = False
    return np.asarray(processed | fill[labels], dtype=bool)


def binary_morphology(mask: np.ndarray, *, operation_type: str, radius: float) -> np.ndarray:
    processed = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return processed
    footprint = _footprint_from_radii(radius, radius, radius)
    if operation_type == MASK_OPERATION_BINARY_CLOSING:
        return np.asarray(ndimage.binary_closing(processed, structure=footprint), dtype=bool)
    if operation_type == MASK_OPERATION_BINARY_OPENING:
        return np.asarray(ndimage.binary_opening(processed, structure=footprint), dtype=bool)
    if operation_type == MASK_OPERATION_DILATE:
        return np.asarray(ndimage.binary_dilation(processed, structure=footprint), dtype=bool)
    if operation_type == MASK_OPERATION_ERODE:
        return np.asarray(ndimage.binary_erosion(processed, structure=footprint), dtype=bool)
    raise ValueError(f"Unsupported binary morphology operation: {operation_type}.")


def remove_border_objects(mask: np.ndarray) -> np.ndarray:
    labels = connected_components(mask)
    if int(labels.max()) == 0:
        return np.asarray(mask, dtype=bool)
    remove = np.zeros(int(labels.max()) + 1, dtype=bool)
    border_ids = _border_label_ids(labels)
    remove[border_ids] = True
    remove[0] = True
    return np.asarray(~remove[labels], dtype=bool)


def filter_objects_by_size_range(mask: np.ndarray, *, min_size: int, max_size: int) -> np.ndarray:
    labels = connected_components(mask)
    if int(labels.max()) == 0:
        return np.zeros(labels.shape, dtype=bool)
    counts = np.bincount(labels.reshape(-1))
    keep = counts >= int(min_size)
    if max_size > 0:
        keep &= counts <= int(max_size)
    keep[0] = False
    return np.asarray(keep[labels], dtype=bool)


def apply_mask_operations(mask: np.ndarray, operations: list[dict[str, Any]]) -> np.ndarray:
    processed = np.asarray(mask, dtype=bool)
    for operation in operations:
        operation_type = str(operation["type"])
        if operation_type == MASK_OPERATION_REMOVE_SMALL_OBJECTS:
            processed = remove_small_objects(processed, size=int(operation["size"]))
        elif operation_type == MASK_OPERATION_FILL_SMALL_HOLES:
            processed = fill_small_holes(processed, size=int(operation["size"]))
        elif operation_type in {
            MASK_OPERATION_BINARY_CLOSING,
            MASK_OPERATION_BINARY_OPENING,
            MASK_OPERATION_DILATE,
            MASK_OPERATION_ERODE,
        }:
            processed = binary_morphology(processed, operation_type=operation_type, radius=float(operation["radius"]))
        elif operation_type == MASK_OPERATION_REMOVE_BORDER_OBJECTS:
            processed = remove_border_objects(processed)
        elif operation_type == MASK_OPERATION_SIZE_RANGE:
            processed = filter_objects_by_size_range(
                processed,
                min_size=int(operation["min_size"]),
                max_size=int(operation["max_size"]),
            )
        else:
            raise ValueError(f"Unsupported mask operation: {operation_type}.")
    return np.asarray(processed, dtype=bool)


def voronoi_otsu_instances(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    spot_sigma: float,
    outline_sigma: float,
) -> np.ndarray:
    semantic = np.asarray(mask, dtype=bool)
    if not bool(semantic.any()):
        return np.zeros(semantic.shape, dtype=np.uint32)

    safe_values = finite_float32(values)
    spot_image = ndimage.gaussian_filter(safe_values, sigma=max(0.0, float(spot_sigma)))
    min_distance = max(1, int(round(max(1.0, float(spot_sigma)))))
    coordinates = peak_local_max(
        spot_image,
        labels=semantic.astype(np.uint8),
        min_distance=min_distance,
        exclude_border=False,
    )
    if coordinates.size == 0:
        return connected_components(semantic)

    markers = np.zeros(semantic.shape, dtype=np.uint32)
    for marker_index, coordinate in enumerate(coordinates, start=1):
        markers[tuple(int(item) for item in coordinate)] = marker_index

    outline_image = ndimage.gaussian_filter(safe_values, sigma=max(0.0, float(outline_sigma)))
    labels = watershed(-outline_image, markers=markers, mask=semantic)
    return np.asarray(labels, dtype=np.uint32)


def distance_transform_connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return np.asarray(ndimage.generate_binary_structure(3, 1), dtype=bool)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    raise ValueError("Distance-transform connectivity must be 6 or 26.")


def watershed_connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return np.asarray(ndimage.generate_binary_structure(3, 1), dtype=bool)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=bool)
    raise ValueError("Watershed connectivity must be 6 or 26.")


def _positions_from_maximum_position(raw_positions: Any) -> list[tuple[int, int, int]]:
    if isinstance(raw_positions, tuple) and len(raw_positions) == 3 and all(np.isscalar(item) for item in raw_positions):
        return [tuple(int(item) for item in raw_positions)]
    return [tuple(int(item) for item in position) for position in raw_positions]


def distance_transform_watershed_instances(
    mask: np.ndarray,
    *,
    dynamic: float,
    connectivity: int,
    spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    semantic = np.asarray(mask, dtype=bool)
    if not bool(semantic.any()):
        return np.zeros(semantic.shape, dtype=np.uint32)

    structure = distance_transform_connectivity_structure(connectivity)
    component_labels, component_count = ndimage.label(semantic, structure=structure)
    if int(component_count) == 0:
        return np.zeros(semantic.shape, dtype=np.uint32)

    distance = np.asarray(ndimage.distance_transform_edt(semantic, sampling=spacing_zyx), dtype=np.float32)
    if dynamic <= 0:
        maxima = np.asarray(local_maxima(distance, footprint=structure), dtype=bool) & semantic
    else:
        maxima = np.asarray(h_maxima(distance, float(dynamic), footprint=structure), dtype=bool) & semantic

    marker_labels, marker_count = ndimage.label(maxima, structure=structure)
    if int(marker_count) == 0:
        return np.asarray(component_labels, dtype=np.uint32)

    markers = np.asarray(marker_labels, dtype=np.uint32)
    marker_component_ids = np.unique(component_labels[markers > 0])
    marker_component_ids = marker_component_ids[marker_component_ids > 0]
    all_component_ids = np.arange(1, int(component_count) + 1, dtype=component_labels.dtype)
    missing_component_ids = np.setdiff1d(all_component_ids, marker_component_ids, assume_unique=True)
    if missing_component_ids.size:
        missing_positions = _positions_from_maximum_position(
            ndimage.maximum_position(distance, labels=component_labels, index=missing_component_ids.tolist())
        )
        next_marker = int(markers.max()) + 1
        for position in missing_positions:
            markers[position] = next_marker
            next_marker += 1

    labels = watershed(-distance, markers=markers, mask=semantic, connectivity=structure)
    return np.asarray(labels, dtype=np.uint32)


def foreground_robust_normalize(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray | None:
    semantic = np.asarray(mask, dtype=bool)
    safe_values = finite_float32(values)
    foreground_values = np.asarray(safe_values[semantic], dtype=np.float32)
    if foreground_values.size == 0:
        return None
    low_value, high_value = np.percentile(foreground_values, [low_percentile, high_percentile])
    denominator = float(high_value - low_value)
    if denominator <= 0:
        return None
    normalized = (safe_values - float(low_value)) / denominator
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~semantic] = 0.0
    return np.asarray(normalized, dtype=np.float32)


def intensity_prominence_watershed_instances(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    prominence: float,
    smoothing_sigma: float,
    low_percentile: float,
    high_percentile: float,
    connectivity: int,
) -> np.ndarray:
    semantic = np.asarray(mask, dtype=bool)
    if not bool(semantic.any()):
        return np.zeros(semantic.shape, dtype=np.uint32)

    structure = watershed_connectivity_structure(connectivity)
    component_labels, component_count = ndimage.label(semantic, structure=structure)
    if int(component_count) == 0:
        return np.zeros(semantic.shape, dtype=np.uint32)

    normalized = foreground_robust_normalize(
        values,
        semantic,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    if normalized is None:
        return np.asarray(component_labels, dtype=np.uint32)

    elevation_source = normalized
    if smoothing_sigma > 0:
        elevation_source = np.asarray(ndimage.gaussian_filter(elevation_source, sigma=float(smoothing_sigma)), dtype=np.float32)
        elevation_source[~semantic] = 0.0

    if prominence <= 0:
        maxima = np.asarray(local_maxima(elevation_source, footprint=structure), dtype=bool) & semantic
    else:
        maxima = np.asarray(h_maxima(elevation_source, float(prominence), footprint=structure), dtype=bool) & semantic

    marker_labels, marker_count = ndimage.label(maxima, structure=structure)
    if int(marker_count) == 0:
        return np.asarray(component_labels, dtype=np.uint32)

    markers = np.asarray(marker_labels, dtype=np.uint32)
    marker_component_ids = np.unique(component_labels[markers > 0])
    marker_component_ids = marker_component_ids[marker_component_ids > 0]
    all_component_ids = np.arange(1, int(component_count) + 1, dtype=component_labels.dtype)
    missing_component_ids = np.setdiff1d(all_component_ids, marker_component_ids, assume_unique=True)
    if missing_component_ids.size:
        missing_positions = _positions_from_maximum_position(
            ndimage.maximum_position(elevation_source, labels=component_labels, index=missing_component_ids.tolist())
        )
        next_marker = int(markers.max()) + 1
        for position in missing_positions:
            markers[position] = next_marker
            next_marker += 1

    labels = watershed(-elevation_source, markers=markers, mask=semantic, connectivity=structure)
    return np.asarray(labels, dtype=np.uint32)


def instance_segmentation(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    method: str,
    voronoi_spot_sigma: float,
    voronoi_outline_sigma: float,
    distance_dynamic: float = 1.0,
    distance_connectivity: int = 6,
    distance_spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity_prominence: float = 0.15,
    intensity_smoothing_sigma: float = 0.0,
    intensity_low_percentile: float = 1.0,
    intensity_high_percentile: float = 99.0,
    intensity_connectivity: int = 6,
) -> np.ndarray:
    if method == INSTANCE_METHOD_NONE:
        return np.asarray(mask, dtype=np.uint8)
    if method == INSTANCE_METHOD_CONNECTED_COMPONENTS:
        return connected_components(mask)
    if method == INSTANCE_METHOD_VORONOI_OTSU:
        return voronoi_otsu_instances(
            values,
            mask,
            spot_sigma=voronoi_spot_sigma,
            outline_sigma=voronoi_outline_sigma,
        )
    if method == INSTANCE_METHOD_DISTANCE_TRANSFORM_WATERSHED:
        return distance_transform_watershed_instances(
            mask,
            dynamic=distance_dynamic,
            connectivity=distance_connectivity,
            spacing_zyx=distance_spacing_zyx,
        )
    if method == INSTANCE_METHOD_INTENSITY_PROMINENCE_WATERSHED:
        return intensity_prominence_watershed_instances(
            values,
            mask,
            prominence=intensity_prominence,
            smoothing_sigma=intensity_smoothing_sigma,
            low_percentile=intensity_low_percentile,
            high_percentile=intensity_high_percentile,
            connectivity=intensity_connectivity,
        )
    raise ValueError(f"Unsupported instance segmentation method: {method}.")
