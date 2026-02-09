# NOTE: Assumes Channel first ordering
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from spatialdino.data import EMPTY_VOXEL_THRESHOLD, Image2D, Image3D, ZFlattenMode

RANDOM_SEED = 42
RNG = np.random.default_rng(RANDOM_SEED)
CONST_AUTO_THRESHOLD = 5000


def validate_crop_params(
    crop_params: Tuple[int, int, int, int, int, int],
    raw_volume_shape: Tuple[int, int, int],
) -> None:
    """
    Validate that crop parameters are valid for the given volume shape.

    Args:
        crop_params: (z_start, z_end, y_start, y_end, x_start, x_end)
        raw_volume_shape: (Z, Y, X) dimensions of the volume
    """
    start_z, end_z, start_y, end_y, start_x, end_x = crop_params
    assert start_z < end_z, "start_z must be less than end_z"
    assert start_y < end_y, "start_y must be less than end_y"
    assert start_x < end_x, "start_x must be less than end_x"
    assert start_z < raw_volume_shape[0], (
        "start_z must be less than the number of z-stacks"
    )
    assert start_y < raw_volume_shape[1], (
        "start_y must be less than the number of y-stacks"
    )
    assert start_x < raw_volume_shape[2], (
        "start_x must be less than the number of x-stacks"
    )
    assert end_z <= raw_volume_shape[0], (
        "end_z must be less than or equal to the number of z-stacks"
    )
    assert end_y <= raw_volume_shape[1], (
        "end_y must be less than or equal to the number of y-stacks"
    )
    assert end_x <= raw_volume_shape[2], (
        "end_x must be less than or equal to the number of x-stacks"
    )

def median_fill(raw_volume: np.ndarray) -> np.ndarray:
    """Create a median fill for the volume.

    Returns
    -------
    np.ndarray
        Median pad for the volume.
    """

    zero_mask = np.isclose(raw_volume, 0, atol=EMPTY_VOXEL_THRESHOLD)
    if np.any(zero_mask):
        median_val = np.median(raw_volume[~zero_mask])
        raw_volume[zero_mask] = median_val
    return raw_volume


def auto_extract_crop_params(
    volume: np.ndarray,
    k: int = 4,
    lower_percentile: float = 99.75,
    upper_percentile: float = 99.9,
    min_bbox_ratio: float = 0.3,
    max_bbox_ratio: float = 0.6,
    median_filter_size: int = 3,
) -> Optional[np.ndarray]:
    assert min_bbox_ratio <= max_bbox_ratio, (
        "min_bbox_ratio must be less than or equal to max_bbox_ratio"
    )
    assert min_bbox_ratio >= 0.1 and max_bbox_ratio <= 1.0, (
        "min_bbox_ratio and max_bbox_ratio must be greater than 0 and less than or equal to 1.0"
    )

    lower_bound, upper_bound = np.percentile(
        volume, [lower_percentile, upper_percentile]
    )
    mask = ((volume > lower_bound) & (volume < upper_bound)).astype(bool)
    filtered = np.zeros_like(mask)

    for i in range(mask.shape[0]):
        filtered[i] = median_filter(mask[i], size=median_filter_size)

    scaler = MinMaxScaler()
    obj_coords = np.argwhere(filtered)

    if obj_coords.shape[0] < k:
        return None

    obj_coords = scaler.fit_transform(obj_coords)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED)
    kmeans.fit(obj_coords)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    Z, Y, X = filtered.shape
    N = centroids.shape[0]
    rand_z_ratio, rand_y_ratio, rand_x_ratio = (
        RNG.uniform(min_bbox_ratio, max_bbox_ratio, size=N),
        RNG.uniform(min_bbox_ratio, max_bbox_ratio, size=N),
        RNG.uniform(min_bbox_ratio, max_bbox_ratio, size=N),
    )
    z_offset, y_offset, x_offset = (
        rand_z_ratio * Z,
        rand_y_ratio * Y,
        rand_x_ratio * X,
    )

    z_start, z_end, y_start, y_end, x_start, x_end = (
        np.maximum(centroids[:, 0] - z_offset / 2, 0).astype(np.uint32),
        np.minimum(centroids[:, 0] + z_offset / 2, Z).astype(np.uint32),
        np.maximum(centroids[:, 1] - y_offset / 2, 0).astype(np.uint32),
        np.minimum(centroids[:, 1] + y_offset / 2, Y).astype(np.uint32),
        np.maximum(centroids[:, 2] - x_offset / 2, 0).astype(np.uint32),
        np.minimum(centroids[:, 2] + x_offset / 2, X).astype(np.uint32),
    )
    crop_params = np.stack([z_start, z_end, y_start, y_end, x_start, x_end], axis=1)
    return crop_params


def percentile_min_max(
    image: np.ndarray,
    channel: Optional[int] = None,
    pmin: Tuple[float, float] = (1, 3),
    pmax: Tuple[float, float] = (99.5, 99.9),
) -> Tuple[float, float]:
    percentile_axes = (
        None
        if channel is None
        else tuple((d for d in range(image.ndim) if d != channel))
    )
    min_percentile = np.random.uniform(*pmin)
    max_percentile = np.random.uniform(*pmax)
    min_val, max_val = np.percentile(
        image, q=[min_percentile, max_percentile], axis=percentile_axes, keepdims=True
    )
    return min_val, max_val


def histogram_min_max(
    image: np.ndarray,
    threshold: int = CONST_AUTO_THRESHOLD,
    bins: int = 256,
    bins_range: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    if bins_range is None:
        bins_range = image.min(), image.max()
    histogram, bin_edges = np.histogram(image, bins=bins, range=bins_range)
    pixel_count = image.size
    threshold = int(pixel_count / threshold)

    # Find min bin
    min_bin = next((i for i in range(bins) if histogram[i] > threshold), 0)
    max_bin = next(
        (i for i in range(bins - 1, -1, -1) if histogram[i] > threshold), bins - 1
    )

    hmin, hmax = bin_edges[min_bin], bin_edges[max_bin]

    return hmin, hmax


def convert_16bit_to_8bit(
    image: np.ndarray,
    non_zero_stacks: Optional[np.ndarray] = None,
    flatten_mode: ZFlattenMode = ZFlattenMode.NONE,
    bins_range: Optional[Tuple[int, int]] = (0, 256),
) -> np.ndarray:
    if non_zero_stacks is None:
        non_zero_stacks = np.any(image > 1e-6, axis=(-2, -1))

    im = flatten_z_axis(image[non_zero_stacks], flatten_mode)
    image = flatten_z_axis(image, flatten_mode)

    min_val, max_val = histogram_min_max(im, bins_range=bins_range)

    # Apply contrast
    image = np.clip(
        (image - min_val) / (max_val - min_val + np.finfo(np.float32).eps) * 255,
        0,
        255,
    ).astype(np.uint8)

    return image


def wavelength_to_rgb_channel_index(
    wavelength: int, upper_freq_limit: int, lower_freq_limit: int
) -> int:
    # Returns [0, 1, 2] for [R, G, B]
    if wavelength >= upper_freq_limit and wavelength < 440:
        return 2
    elif wavelength >= 440 and wavelength < 490:
        return 1
    elif wavelength >= 490 and wavelength < 510:
        return 1
    elif wavelength >= 510 and wavelength < 580:
        return 0
    elif wavelength >= 580 and wavelength < 645:
        return 0
    elif wavelength >= 645 and wavelength < lower_freq_limit:
        return 0
    return 0


def image_to_color_mode(image: np.ndarray) -> np.ndarray:
    # input shape: [H, W]
    # output shape: [H, W, 3]
    img = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    # img = np.where(img > CellDataset2D.BINARY_THRESHOLD, img, 0)
    return img


def flatten_z_axis(image: np.ndarray, mode: ZFlattenMode) -> np.ndarray:
    # input shape: [Z, Y, X]
    # output shape: [Z, Y, X] if mode == ZFlattenMode.NONE else [1, Y, X]
    if mode == ZFlattenMode.MAX:
        return np.max(image, axis=0, keepdims=True)
    elif mode == ZFlattenMode.MEAN:
        return np.mean(image, axis=0, keepdims=True)
    elif mode == ZFlattenMode.NONE:
        return image


def center_pad_rgb(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    height, width, _ = image.shape

    pad_height = max(target_height - height, 0)
    pad_width = max(target_width - width, 0)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return padded_image


def downsample_image(
    image_2d: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    image = Image.fromarray(image_2d, mode="RGB")  # shape: [H, W, C]
    width, height = image.size
    if width != target_width and height != target_height:
        image.thumbnail(
            (target_height, target_width), resample=Image.Resampling.LANCZOS
        )
    image = np.asarray(image)
    height, width = image.shape[:2]

    # pad if needed
    if width < target_width or height < target_height:
        image = center_pad_rgb(image, target_height, target_width)

    return image  # shape: [H, W, C]


def extract_frame(file_name: str) -> int:
    frame_number = [part for part in file_name.split("_") if "stack" in part][0]
    frame = int(
        frame_number[5:]
    )  # e.g. stack0145 -> 145 (will later be used as index for sparse tensor)
    return frame


def reconstruct_3d_image_from_voxels(frame_dir: Path) -> np.ndarray:
    """Recovers a 3D image from a directory containing voxels.
    Input: [Z, Y, X]
    Output: [New_Z, Y, X]
    Parameters
    ----------
    frame_dir: Path
        Path to the directory containing the voxels

    Returns
    -------
    reconstructed_3d_image : np.ndarray
        The reconstructed 3D image.
    """
    image_3d_dict = {}
    max_z, max_y, max_x = 0, 0, 0

    for voxel_path in frame_dir.iterdir():
        voxel = Image3D(torch.load(voxel_path, weights_only=False))
        position = tuple(np.array(voxel["data"]["position"], dtype=np.uint32))
        values = np.array(voxel["data"]["values"], dtype=np.float32)
        image_3d_dict[position] = values

        z, y, x = position
        dz, dy, dx = values.shape
        max_z = max(max_z, z + dz)
        max_y = max(max_y, y + dy)
        max_x = max(max_x, x + dx)

    # Create the 3D image with the correct dimensions (Z, Y, X)
    reconstructed_3d_image = np.zeros((max_z, max_y, max_x), dtype=np.float32)

    # Reconstruct the 3D image from the voxels
    for (z, y, x), sub_volume in image_3d_dict.items():
        dz, dy, dx = sub_volume.shape
        reconstructed_3d_image[z : z + dz, y : y + dy, x : x + dx] = sub_volume

    return reconstructed_3d_image


def reconstruct_2d_image_from_patches(
    frame_dir: Path,
) -> np.ndarray:
    """_summary_
    Input shape: [3, H, W]
    Output shape: [3, New_H, New_W]
    Parameters
    ----------
    frame_dir : Path
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    image_2d_dict = {}
    max_y, max_x = 0, 0

    for patch in frame_dir.iterdir():
        image_2d = Image2D(torch.load(patch, weights_only=False))
        position = tuple(np.array(image_2d["data"]["position"], dtype=np.uint32))
        values = np.array(image_2d["data"]["values"], dtype=np.float32)
        image_2d_dict[position] = values
        y, x = position
        _, dy, dx = values.shape
        max_y = max(max_y, y + dy)
        max_x = max(max_x, x + dx)

    reconstructed_2d_image = np.zeros((3, max_y, max_x), dtype=np.float32)

    for (y, x), patch in image_2d_dict.items():
        _, dy, dx = patch.shape
        reconstructed_2d_image[:, y : y + dy, x : x + dx] = patch

    return reconstructed_2d_image


def get_channels_last(image: np.ndarray) -> np.ndarray:
    return image.transpose(1, 2, 0)


def image_is_background(image: np.ndarray, threshold: float = 10) -> np.bool_:
    """
    Determines if an image is a background image by checking if the sum of the pixel values is below a threshold.

    Parameters
    ----------
    image : np.ndarray
        The image to check.
    threshold : float, optional
        The threshold below which the image is considered a background image, by default 0.1

    Returns
    -------
    bool
        True if the image is a background image, False otherwise.
    """
    return np.sum(image) < threshold
