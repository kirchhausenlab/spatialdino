from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


DATA_OPERATION_INVERT_LUT = "invert_lut"
DATA_OPERATION_SUBTRACT_BACKGROUND = "subtract_background"
DATA_OPERATION_GAUSSIAN_SMOOTHING = "gaussian_smoothing"
DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN = "laplacian_of_gaussian"
DATA_OPERATION_TYPES = {
    DATA_OPERATION_INVERT_LUT,
    DATA_OPERATION_SUBTRACT_BACKGROUND,
    DATA_OPERATION_GAUSSIAN_SMOOTHING,
    DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN,
}
DATA_BACKEND_CPU = "cpu"
DATA_BACKEND_GPU = "gpu"
DATA_BACKENDS = {DATA_BACKEND_CPU, DATA_BACKEND_GPU}
LOG_RESPONSE_BRIGHT = "bright"
LOG_RESPONSE_DARK = "dark"
LOG_RESPONSE_TYPES = {LOG_RESPONSE_BRIGHT, LOG_RESPONSE_DARK}

MASK_OPERATION_REMOVE_SMALL_OBJECTS = "remove_small_objects"
MASK_OPERATION_TYPES = {MASK_OPERATION_REMOVE_SMALL_OBJECTS}

INSTANCE_METHOD_NONE = "none"
INSTANCE_METHOD_CONNECTED_COMPONENTS = "connected_components"
INSTANCE_METHOD_VORONOI_OTSU = "voronoi_otsu"
INSTANCE_METHODS = {
    INSTANCE_METHOD_NONE,
    INSTANCE_METHOD_CONNECTED_COMPONENTS,
    INSTANCE_METHOD_VORONOI_OTSU,
}

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
            radius = float(raw_operation.get("radius", 10.0))
            if not np.isfinite(radius) or radius < 0:
                raise ValueError(f"Data operation {index} rolling-ball radius must be nonnegative.")
            operations.append({"type": operation_type, "radius": radius})
        elif operation_type == DATA_OPERATION_GAUSSIAN_SMOOTHING:
            sigma = float(raw_operation.get("sigma", 1.0))
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError(f"Data operation {index} Gaussian sigma must be nonnegative.")
            operations.append({"type": operation_type, "sigma": sigma})
        elif operation_type == DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN:
            sigma = float(raw_operation.get("sigma", 1.0))
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError(f"Data operation {index} LoG sigma must be nonnegative.")
            response = str(raw_operation.get("response", LOG_RESPONSE_BRIGHT)).strip()
            if response not in LOG_RESPONSE_TYPES:
                raise ValueError(f"Data operation {index} LoG response must be bright or dark.")
            operations.append({"type": operation_type, "sigma": sigma, "response": response})

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
        size = int(raw_operation.get("size", 64))
        if size < 0:
            raise ValueError(f"Mask operation {index} size must be nonnegative.")
        operations.append({"type": operation_type, "size": size})

    return operations


def normalize_instance_method(raw_method: Any) -> str:
    method = str(raw_method or INSTANCE_METHOD_CONNECTED_COMPONENTS).strip()
    if method not in INSTANCE_METHODS:
        raise ValueError(f"Unsupported instance segmentation method: {method}.")
    return method


def invert_lut(values: np.ndarray, *, source_kind: str) -> np.ndarray:
    if source_kind == SOURCE_KIND_PROBMAP:
        low, high = 0.0, 1.0
    elif source_kind == SOURCE_KIND_PCA:
        low, high = 0.0, 255.0
    else:
        low, high = finite_min_max(values)
    return np.asarray((low + high) - values, dtype=np.float32)


def subtract_background(values: np.ndarray, *, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.asarray(values, dtype=np.float32)
    footprint = spherical_footprint(int(np.ceil(radius)))
    background = ndimage.grey_opening(values, footprint=footprint)
    return np.asarray(values - background, dtype=np.float32)


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
            processed = subtract_background(processed, radius=float(operation["radius"]))
        elif operation_type == DATA_OPERATION_GAUSSIAN_SMOOTHING:
            sigma = float(operation["sigma"])
            processed = np.asarray(ndimage.gaussian_filter(processed, sigma=sigma), dtype=np.float32)
        elif operation_type == DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN:
            sigma = float(operation["sigma"])
            log_values = np.asarray(ndimage.gaussian_laplace(processed, sigma=sigma), dtype=np.float32)
            processed = -log_values if operation.get("response") == LOG_RESPONSE_BRIGHT else log_values
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
            radius = float(operation["radius"])
            if radius > 0:
                background = cle.opening_sphere(
                    processed,
                    radius_x=radius,
                    radius_y=radius,
                    radius_z=radius,
                    device=cle_device,
                )
                processed = cle.subtract_images(processed, background, device=cle_device)
        elif operation_type == DATA_OPERATION_GAUSSIAN_SMOOTHING:
            sigma = float(operation["sigma"])
            processed = cle.gaussian_blur(
                processed,
                sigma_x=sigma,
                sigma_y=sigma,
                sigma_z=sigma,
                device=cle_device,
            )
        elif operation_type == DATA_OPERATION_LAPLACIAN_OF_GAUSSIAN:
            sigma = float(operation["sigma"])
            smoothed = cle.gaussian_blur(
                processed,
                sigma_x=sigma,
                sigma_y=sigma,
                sigma_z=sigma,
                device=cle_device,
            )
            processed = cle.laplace_box(smoothed, device=cle_device)
            if operation.get("response") == LOG_RESPONSE_BRIGHT:
                processed = cle.multiply_image_and_scalar(processed, scalar=-1.0, device=cle_device)
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


def apply_mask_operations(mask: np.ndarray, operations: list[dict[str, Any]]) -> np.ndarray:
    processed = np.asarray(mask, dtype=bool)
    for operation in operations:
        operation_type = str(operation["type"])
        if operation_type == MASK_OPERATION_REMOVE_SMALL_OBJECTS:
            processed = remove_small_objects(processed, size=int(operation["size"]))
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


def instance_segmentation(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    method: str,
    voronoi_spot_sigma: float,
    voronoi_outline_sigma: float,
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
    raise ValueError(f"Unsupported instance segmentation method: {method}.")
