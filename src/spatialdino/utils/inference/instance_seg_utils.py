from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyclesperanto as cle
import torch
from torch.nn import functional as F
from numba import njit
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import exposure, io
from sklearn.cluster import KMeans
from typing_extensions import TypedDict
from skimage import filters
from spatialdino.utils.utils import PCA, cosine_dist_func  # generate_centroids
from skimage.filters import threshold_otsu
from scipy import ndimage


def threshold_otsu_3d(
    volume: np.ndarray,
    nbins: int = 256,
    footprint: Tuple[int, int, int] = (5, 5, 5),
    device: cle.Device = None,
) -> float:
    """
    Compute the single foreground/background threshold for 3D Otsu segmentation
    based on the dimension decomposition rule (Eq. 6, Feng et al. 2017).

    From: https://www.sciencedirect.com/science/article/pii/S1051200416301191

    Parameters
    ----------
    volume : ndarray
        3D image volume to segment. It should be a 3D numpy array of shape (Z, Y, X).
    nbins : int, optional
        Number of quantization bins for intensity values. Default is 256.
    footprint : tuple of int, optional
        Size of the neighborhood for spatial features (mean and median).
        Default is (5, 5, 5). Where each element corresponds to the size in
        the (Z, Y, X) dimensions respectively.

    Returns the intensity threshold separating foreground from background.
    """
    if volume.dtype != np.float32 and volume.dtype != np.float64:
        raise ValueError("Input volume must be of type float32 or float64.")

    # 1. Spatial features: neighborhood mean and median
    # g = uniform_filter(volume.astype(float), size=footprint)
    radius_z, radius_y, radius_x = (
        footprint[0] // 2,
        footprint[1] // 2,
        footprint[2] // 2,
    )

    g = np.asarray(
        cle.mean_filter(
            volume,
            radius_x=radius_x,
            radius_y=radius_y,
            radius_z=radius_z,
            device=device,
        ),
        dtype=np.float32,
    )
    # h = median_filter(volume, size=footprint)
    h = np.asarray(
        cle.median_box(
            volume,
            radius_x=radius_x,
            radius_y=radius_y,
            radius_z=radius_z,
            device=device,
        ),
        dtype=np.float32,
    )

    # 2. Quantize to [0, nbins-1]
    v_min, v_max = volume.min(), volume.max()
    scale = (nbins - 1) / (v_max - v_min) if v_max > v_min else 1.0
    f_q = ((volume - v_min) * scale).astype(np.uint32)
    g_q = ((g - v_min) * scale).astype(np.uint32)
    h_q = ((h - v_min) * scale).astype(np.uint32)

    # 3. 1D Otsu thresholds on each feature
    tf = threshold_otsu(f_q, nbins=nbins)
    tg = threshold_otsu(g_q, nbins=nbins)
    th = threshold_otsu(h_q, nbins=nbins)

    # 4. Fuse thresholds (average) and convert back to intensity scale
    thr_q = np.mean([tf, tg, th])
    threshold = v_min + thr_q / scale
    return threshold


def pseudo_flat_field(
    volume: np.ndarray,
    radius: float = 3.0,
    pixel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: cle.Device = None,
) -> np.ndarray:
    """
    Pseudo flat-field correction on a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        Input image stack, shape (Z, Y, X)
    radius : float
        Gaussian blur radius along X (in pixels).
    pixel_size : tuple of float
        (pixel_depth, pixel_height, pixel_width). Used to compute anisotropic blur.

    Returns
    -------
    np.ndarray
        Flat-field corrected volume.
    """
    if volume.dtype != np.float32 and volume.dtype != np.float64:
        raise ValueError("Input volume must be of type float32 or float64.")

    # compute aspect ratios
    pd, ph, pw = pixel_size
    x_y_ratio = pw / ph
    z_x_ratio = pd / pw

    # compute blur sigmas
    sigma_x = radius
    sigma_y = radius * x_y_ratio
    sigma_z = radius / z_x_ratio

    background = np.asarray(
        cle.gaussian_blur(
            volume, sigma_z=sigma_z, sigma_y=sigma_y, sigma_x=sigma_x, device=device
        )
    )

    # normalize: original / background * mean(background)
    mean_bg = background.mean()
    corrected = volume / (background + 1e-8) * mean_bg

    return corrected


@staticmethod
@njit(
    fastmath=True,
    cache=True,
)
def compute_indices(
    mask: np.ndarray, non_zero_count: int, D: int
) -> Tuple[np.ndarray, np.ndarray]:
    row_inds = np.empty(non_zero_count, dtype=np.uint64)
    col_inds = np.empty(non_zero_count, dtype=np.uint64)

    idx = 0
    for i in range(mask.size):
        if mask[i]:
            for j in range(D):
                row_inds[idx] = i
                col_inds[idx] = j
                idx += 1

    return row_inds, col_inds


def sobel(img: np.ndarray, device: cle.Device = None) -> np.ndarray:
    """
    Apply Sobel filter to 2D, 3D, or 4D numpy array.

    Args:
        img: numpy array of shape [(Z), Y, X, (C)]
        device: device to use for computation, refer to pyclesperanto documentation for more details
    """
    res = np.zeros_like(img)
    if img.ndim == 2:
        res = cle.sobel(img, device=device)
    elif img.ndim == 3:
        res = cle.sobel(img, device=device)
    else:
        for i in range(img.shape[-1]):
            res[..., i] = np.asarray(cle.sobel(img[..., i], device=device))
    return res


def intensity_adjustment_and_background_removal(
    img_3d: np.ndarray,
    device: cle.Device,
    radius: Optional[int] = 5,
) -> np.ndarray:
    # https://github.com/clEsperanto/pyclesperanto/blob/main/demos/examples/Segmentation_3D.ipynb
    if img_3d.dtype != np.float32:
        img_3d = img_3d.astype(np.float32)
    equalized_intensities_stack = cle.create_like(img_3d, device=device)
    a_slice = cle.create([img_3d.shape[1], img_3d.shape[2]], device=device)
    num_slices = img_3d.shape[0]
    mean_intensity_stack = cle.mean_of_all_pixels(img_3d, device=device)  # type: ignore

    corrected_slices = None
    for z in range(num_slices):
        cle.copy_slice(img_3d, a_slice, z, device=device)  # type: ignore
        mean_intensity_slice = cle.mean_of_all_pixels(a_slice, device=device)  # type: ignore
        correction_factor = mean_intensity_slice / mean_intensity_stack  # type: ignore
        corrected_slices = cle.multiply_image_and_scalar(
            a_slice, corrected_slices, correction_factor, device=device
        )
        cle.copy_slice(corrected_slices, equalized_intensities_stack, z, device=device)  # type: ignore

    background_subtracted = cle.top_hat_box(
        equalized_intensities_stack,
        radius_x=5 if radius is None else radius,
        radius_y=5 if radius is None else radius,
        radius_z=5 if radius is None else radius,
        device=device,
    )
    return background_subtracted


def refine_mask(
    img_3d: np.ndarray,
    fg_seg_3d: np.ndarray,
    device: cle.Device,
    normalize_fn: Callable,
    clip_limit: float = 0.9,
) -> np.ndarray:
    refined_mask_3d = cle.dilate_box(fg_seg_3d, device=device)
    refined_mask_3d = sobel(img_3d, device=device) * refined_mask_3d
    refined_mask_3d = normalize_fn(np.asarray(refined_mask_3d))
    kernel_size = np.array([max(img_3d.shape[0] // 25, 11) for _ in range(img_3d.ndim)])
    refined_mask_3d = exposure.equalize_adapthist(
        refined_mask_3d, kernel_size=kernel_size, clip_limit=clip_limit
    )
    return refined_mask_3d


@njit(
    fastmath=True,
    cache=True,
)
def get_attn_density_optim(
    labels_arr: np.ndarray, attn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """For each cluster, get attention per unit area and return a density arr
    where each entry is the pixel's cluster attention density.

    :param labels_arr: arr shape (z, y, x) where each entry is the cluster index
    :type labels_arr: np.ndarray
    :param attn: arr shape (z, y, x) of ]the attention (usually sum(CLS))
    :type attn: np.ndarray
    :return: attention density map and list of all cluster densities
    :rtype: Tuple[np.ndarray, List[float]]
    """
    unique_labels = np.unique(labels_arr)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    num_pixels = np.zeros_like(unique_labels, dtype=np.uint64)
    cluster_attn = np.zeros_like(unique_labels, dtype=np.float64)

    for z in range(labels_arr.shape[0]):
        for y in range(labels_arr.shape[1]):
            for x in range(labels_arr.shape[2]):
                label_val = labels_arr[z, y, x]
                idx = label_to_idx[label_val]
                num_pixels[idx] += 1
                cluster_attn[idx] += attn[z, y, x]
    density = cluster_attn / num_pixels
    attention_density_map = np.zeros_like(labels_arr, dtype=np.float64)

    for z in range(labels_arr.shape[0]):
        for y in range(labels_arr.shape[1]):
            for x in range(labels_arr.shape[2]):
                label_val = labels_arr[z, y, x]
                idx = label_to_idx[label_val]
                attention_density_map[z, y, x] += density[idx]
    return attention_density_map, density


def get_attn_density(
    labels_arr: np.ndarray, attn: np.ndarray
) -> Tuple[np.ndarray, List[float]]:
    """For each cluster, get attention per unit area and return a density arr
    where each entry is the pixel's cluster attention density.

    :param labels_arr: arr shape (h, w) where each entry is the cluster index
    :type labels_arr: np.ndarray
    :param attn: arr shape (h, w) of ]the attention (usually sum(CLS))
    :type attn: np.ndarray
    :return: attention density map and list of all cluster densities
    :rtype: Tuple[np.ndarray, List[float]]
    """
    densities = []
    attention_density_map = np.zeros_like(labels_arr, dtype=np.float32)
    for n in np.unique(labels_arr):
        binary_mask = np.where(labels_arr == n, 1, 0)
        n_pix = np.sum(binary_mask)
        cluster_attn = np.sum(attn * binary_mask)
        cluster_attn_density = cluster_attn / n_pix
        densities.append(cluster_attn_density)
        attention_density_map += cluster_attn_density * binary_mask
    return attention_density_map, densities


def _generate_instance_segmentation(
    fg: np.ndarray,
    normalize_fn: Callable,
    device: cle.Device,
    sigma_vals: float = 2.0,
    spot_sigma: float = 1.0,
    outline_sigma: float = 1.0,
) -> np.ndarray:
    blobs_laplacian_of_gaussian = cle.laplace(
        cle.gaussian_blur(
            fg,
            sigma_x=sigma_vals,
            sigma_y=sigma_vals,
            sigma_z=sigma_vals,
            device=device,
        ),
        device=device,
    )
    blobs_laplacian_of_gaussian = normalize_fn(blobs_laplacian_of_gaussian)
    segmented = cle.voronoi_otsu_labeling(
        blobs_laplacian_of_gaussian,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        device=device,
    )
    return np.asarray(segmented, dtype=np.uint64)


def get_3d_mask_and_density(
    img_3d: np.ndarray,
    labels: np.ndarray,
    attn_feats: np.ndarray,
    nbins: int = 65536,
) -> np.ndarray:
    sum_cls = np.sum(attn_feats, axis=-1)
    density_map, densities = get_attn_density_optim(labels_arr=labels, attn=sum_cls)

    threshold = filters.threshold_otsu(densities, nbins=nbins)
    seg_3d_mask = (density_map <= threshold).astype(np.float64)
    seg_3d_mask = np.clip(seg_3d_mask * 255, 0, 255).astype(np.uint8)
    return seg_3d_mask


def _log_kernel(
    size: Tuple[int, int, int] = (7, 7, 7),
    sigma: float = 1.0,
) -> np.ndarray:
    x, y, z = size
    xc, yc, zc = (np.array(size) - 1) / 2.0
    xx, yy, zz = np.meshgrid(
        np.arange(x) - xc, np.arange(y) - yc, np.arange(z) - zc, indexing="ij"
    )
    r2 = xx**2 + yy**2 + zz**2
    normalization = (r2 - 3 * sigma**2) / (sigma**5)
    log_kernel = normalization * np.exp(-r2 / (2 * sigma**2))
    log_kernel -= log_kernel.mean()
    return log_kernel


def get_3d_mask_and_foreground(
    normalize_fn: Callable,
    img_3d: np.ndarray,
    seg_3d_mask: np.ndarray,
    device: cle.Device,
    default_kernel_size: int = 11,
    clip_limit: float = 0.9,
    use_raw_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_raw_mask:
        fg = normalize_fn(
            np.asarray(img_3d * seg_3d_mask.astype(np.float32), dtype=np.float32)
        )
        return seg_3d_mask, fg

    refined_mask_3d = cle.dilate_box(seg_3d_mask, device=device)
    refined_mask_3d = sobel(img_3d, device=device) * refined_mask_3d
    refined_mask_3d = normalize_fn(np.asarray(refined_mask_3d))
    kernel_size = np.array([
        max(img_3d.shape[0] // 25, default_kernel_size) for _ in range(img_3d.ndim)
    ])
    refined_mask_3d = np.asarray(
        exposure.equalize_adapthist(
            refined_mask_3d, kernel_size=kernel_size, clip_limit=clip_limit
        ),
        dtype=np.float32,
    )
    fg = normalize_fn(np.asarray(img_3d * refined_mask_3d, dtype=np.float32))

    return refined_mask_3d, fg


def outlier_utils(detections: Dict[int, int]) -> Tuple[np.ndarray, int]:
    print(detections)
    plt.hist(list(detections.values()), bins=20)
    min_detections = min(detections.values())
    max_detections = max(detections.values())
    average_detections = sum(detections.values()) / len(detections)
    print(
        f"Min: {min_detections}, Max: {max_detections}, Average: {average_detections}"
    )

    data = list(detections.values())
    kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(data).reshape(-1, 1))
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.flatten()
    high_label = np.argmax(centroids)

    high_vals = np.array(data)[labels == high_label]
    low_vals = np.array(data)[labels != high_label]

    print(f"High: {high_vals}, mean {high_vals.mean()}, std {high_vals.std()}")
    print(f"Low: {low_vals}, mean {low_vals.mean()}, std {low_vals.std()}")

    return labels, int(high_label)


def exclude_outliers(
    detections: Dict[int, int],
    labels: np.ndarray,
    high_label: int,
    file_paths: List[Path],
) -> List[Path]:
    high_value_frames = [
        frame
        for frame, count in detections.items()
        if labels[list(detections.keys()).index(frame)] == high_label
    ]
    print(
        f"Excluding {len(high_value_frames)} frames with high detections: {high_value_frames}"
    )
    # Filter out high value frames
    filtered_file_paths = [
        path for i, path in enumerate(file_paths) if i not in high_value_frames
    ]
    print(
        f"Using {len(filtered_file_paths)} frames out of {len(file_paths)} total frames"
    )
    return filtered_file_paths


def _normalize_3d(arr: np.ndarray, EPS: float = 1e-8) -> np.ndarray:
    min_val, max_val = np.min(arr), np.max(arr)
    return (arr - min_val) / (max_val - min_val + EPS)


def interpolate_fn_and_normalize(
    patch_tokens: np.ndarray,
    volume: np.ndarray,
    normalize_: bool = True,
) -> np.ndarray:
    interpolated_tokens = (
        F.interpolate(
            torch.from_numpy(patch_tokens)
            .unsqueeze_(0)
            .unsqueeze_(0)
            .cuda(non_blocking=True),
            size=volume.shape[-3:],
            mode="trilinear",
        )
        .squeeze_(0)
        .squeeze_(0)
        .cpu()
        .numpy()
    )
    if normalize_:
        interpolated_tokens = _normalize_3d(interpolated_tokens)
    return interpolated_tokens


def _laplacian_3d(
    img_bgsub: np.ndarray, size: List[int], sigma: float = 1.0
) -> np.ndarray:
    sz_x, sz_y, sz_z = size

    # Create coordinate grids
    x = np.arange(-sz_x // 2, sz_x // 2 + 1)
    y = np.arange(-sz_y // 2, sz_y // 2 + 1)
    z = np.arange(-sz_z // 2, sz_z // 2 + 1)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r2 = X**2 + Y**2 + Z**2
    # Laplacian of Gaussian formula
    h = (r2 - 3 * sigma**2) * np.exp(-r2 / (2 * sigma**2))
    h /= h.sum()
    return h


def _generate_mask(
    patch_tokens_3d: np.ndarray,
    attn_feats_3d: np.ndarray,
    attention_weight: float = 0.3,
    cle_device: cle.Device = None,
    nbins: int = 65536,
) -> np.ndarray:
    if cle_device is None:
        cle_device = cle.get_device()
    fused_features = (
        patch_tokens_3d * (1 - attention_weight) + attn_feats_3d * attention_weight
    )
    thresh_fused = threshold_otsu_3d(
        fused_features.max(0), nbins=nbins, device=cle_device
    )
    return fused_features > thresh_fused


class Track(TypedDict):
    track_id: int
    t: int
    x: float
    y: float
    z: float
    t_0: int
    track_length: int


def track_objects(
    file_paths: List[Path],
    max_distance: float = float("inf"),
    max_gap: int = 1,
) -> pd.DataFrame:
    assert max_distance > 0, "max_distance must be greater than 0"

    tracks = []  # list to store track updates (one per matched frame)
    next_track_id = 0  # counter to assign unique track IDs

    # active_tracks holds info for currently active tracks:
    # track_id -> {"last_t": int, "track_length": int, "t_0": int, "centroid": np.array}
    active_tracks = {}

    # Process each frame by index t and its centroids
    for t, file_path in enumerate(file_paths):
        centroids, features = (
            io.imread(file_path.joinpath("centroids.tif")),
            io.imread(file_path.joinpath("features.tif")),
        )

        # If this is the first frame or no active tracks, initialize new tracks
        if t == 0 or not active_tracks:
            for centroid, feature in zip(centroids, features):
                track_id = next_track_id
                next_track_id += 1

                tracks.append(
                    Track(
                        track_id=track_id,
                        t=t,
                        z=centroid[0],
                        y=centroid[1],
                        x=centroid[2],
                        t_0=t,
                        track_length=1,
                    )
                )
                active_tracks[track_id] = {
                    "last_t": t,
                    "track_length": 1,
                    "t_0": t,
                    "centroid": centroid,
                    "feature": feature,
                }
            continue

        # Gather centroids from active tracks
        prev_track_ids = list(active_tracks.keys())
        prev_features = np.array([
            active_tracks[tid]["feature"] for tid in prev_track_ids
        ])
        prev_centroids = np.array([
            active_tracks[tid]["centroid"] for tid in prev_track_ids
        ])

        # Compute the cost matrix between active tracks and current centroids
        cost_matrix = cosine_dist_func(prev_features, features) + cdist(
            prev_centroids, centroids, metric="euclidean"
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_cols = set()
        for r, c in zip(row_ind, col_ind):
            track_id = prev_track_ids[r]
            centroid = centroids[c]
            feature = features[c]
            matched_cols.add(c)

            # Update the active track with the new centroid and time
            active_tracks[track_id]["last_t"] = t
            active_tracks[track_id]["track_length"] += 1
            active_tracks[track_id]["centroid"] = centroid
            active_tracks[track_id]["feature"] = feature

            tracks.append(
                Track(
                    track_id=track_id,
                    t=t,
                    z=centroid[0],
                    y=centroid[1],
                    x=centroid[2],
                    t_0=active_tracks[track_id]["t_0"],
                    track_length=active_tracks[track_id]["track_length"],
                )
            )

        # Create new tracks for any centroids that were not matched
        for c in range(len(centroids)):
            if c not in matched_cols:
                centroid = centroids[c]
                feature = features[c]
                track_id = next_track_id
                next_track_id += 1

                tracks.append(
                    Track(
                        track_id=track_id,
                        t=t,
                        z=centroid[0],
                        y=centroid[1],
                        x=centroid[2],
                        t_0=t,
                        track_length=1,
                    )
                )
                active_tracks[track_id] = {
                    "last_t": t,
                    "track_length": 1,
                    "t_0": t,
                    "centroid": centroid,
                    "feature": feature,
                }

        # Remove tracks that have not been updated within the allowed gap (max_gap frames)
        inactive_track_ids = [
            tid
            for tid in list(active_tracks.keys())
            if t - active_tracks[tid]["last_t"] > max_gap
        ]
        for tid in inactive_track_ids:
            del active_tracks[tid]

        # print(np.mean(cost_matrix, axis=1))
    # remove tracks with track_length 1
    tracks = [t for t in tracks if t["track_length"] > 1]

    tracks = pd.DataFrame(tracks)
    tracks.sort_values(by=["track_id", "t"], inplace=True)
    tracks.reset_index(drop=True, inplace=True)
    tracks["track_length"] = tracks.groupby("track_id")["track_length"].transform("max")

    return tracks


from typing import List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from sklearn.decomposition import PCA


def get_track_features(
    file_paths: List[Path],
    explained_variance: Optional[np.ndarray] = None,
    centroid_type: str = "centroids_new",
    run_pca: bool = False,
    pca_fn: PCA = None,  # type: ignore
) -> pd.DataFrame:
    """
    Track objects across multiple time frames.

    Parameters:
    -----------
    file_paths : list of Path objects
        List of paths to folders containing the relevant TIFF files.
    explained_variance : Optional[np.ndarray]
        Variance array to weight features, if provided.
    centroid_type : str
        Base name for centroid TIFF files (without extension).
    run_pca : bool
        Whether to apply PCA on features and include PCA components.
    pca_fn : PCA
        scikit-learn PCA instance, required if run_pca is True.

    Returns:
    --------
    pd.DataFrame
        DataFrame of track features across frames.
    """
    track_features = []

    for t, file_path in enumerate(file_paths):
        centroid_file = file_path.joinpath(f"{centroid_type}.tif")
        feature_file = file_path.joinpath("object_features.tif")
        if not (centroid_file.exists() and feature_file.exists()):
            print(f"Skipping {file_path} - missing centroid or feature TIFF")
            continue

        if t == 0:
            print(f"Centroids: {centroid_file}, Features: {feature_file} at t=0")

        centroids = io.imread(centroid_file)
        features = io.imread(feature_file)

        # Load intensity arrays if they exist
        intensity_arrays = {}
        filenames = {
            "correct_intensities.tif": "correct_intensity",
            "intensities.tif": "intensity",
            "total_intensities.tif": "total_intensity",
        }
        for fname, col in filenames.items():
            fpath = file_path.joinpath(fname)
            if fpath.exists():
                intensity_arrays[col] = io.imread(fpath)

        # Optionally run PCA
        if run_pca and pca_fn is not None:
            pca_res = pca_fn.fit_transform(features)

        # Build feature records
        num_objs = features.shape[0]
        for i in range(num_objs):
            rec = {
                "z": centroids[i, 0],
                "y": centroids[i, 1],
                "x": centroids[i, 2],
                "t": t,
            }
            # Add intensity metrics
            for col, arr in intensity_arrays.items():
                if i < len(arr):
                    rec[col] = arr[i]

            # Add features (optionally weighted)
            for j in range(features.shape[1]):
                key = f"feature_{j}"
                rec[key] = (
                    features[i, j] * explained_variance[j]
                    if explained_variance is not None
                    else features[i, j]
                )

            # Add PCA features if requested
            if run_pca and pca_fn is not None:
                for j in range(pca_res.shape[1]):  # type: ignore
                    rec[f"pca_feature_{j}"] = pca_res[i, j]  # type: ignore

            track_features.append(rec)

    return pd.DataFrame(track_features)


# def get_track_features(
#     file_paths: List[Path],
#     explained_variance: Optional[np.ndarray] = None,
#     centroid_type: str = "centroids_new",
#     run_pca: bool = False,
#     pca_fn: PCA = None,  # type: ignore
# ) -> pd.DataFrame:
#     """
#     Track objects across multiple time frames.

#     Parameters:
#     -----------
#     file_paths : list of Path objects
#         List of paths to folders containing in_seg.tif.
#     max_distance : float
#         Maximum distance between frames for considering a match between frames. Must be greater than 0.
#     max_gap : int
#         Maximum allowed number of consecutive frames in which an object can be missed
#         before its track is terminated.
#     centroid_weight : float
#         Weight for the centroid distance in the cost matrix.
#     feature_weight : float
#         Weight for the feature distance in the cost matrix.

#     Returns:
#     --------
#     tracks : pd.DataFrame
#         A DataFrame of Track objects representing track updates.
#     """

#     track_features = []  # list to store track updates (one per matched frame)
#     # Process each frame by index t and its centroids
#     for t, file_path in enumerate(file_paths):
#         if not file_path.joinpath(f"{centroid_type}.tif").exists():
#             print(f"Skipping {file_path} - {centroid_type}.tif does not exist")
#             continue
#         centroids, features = (
#             io.imread(file_path.joinpath(f"{centroid_type}.tif")),
#             io.imread(file_path.joinpath("object_features.tif")),
#         )
#         if t == 0:
#             print(f"Centroids: {centroids.shape}, Features: {features.shape}")
#         # Check if intensity data is available
#         intensity_file = file_path.joinpath("intensity_new.tif")
#         intensity_raw = file_path.joinpath("intensities.tif")
#         counts_file = file_path.joinpath("counts.tif")
#         total_intensity_file = file_path.joinpath("total_intensities.tif")
#         counts = None
#         intensities = None
#         intensities_raw = None
#         total_intensities = None
#         if intensity_file.exists():
#             intensities = io.imread(intensity_file)
#         if intensity_raw.exists():
#             intensities_raw = io.imread(intensity_raw)
#         if counts_file.exists():
#             counts = io.imread(counts_file)
#         if total_intensity_file.exists():
#             total_intensities = io.imread(total_intensity_file)
#         if run_pca:
#             pca_res = pca_fn.fit_transform(features)

#         # Use minimum to avoid IndexError when arrays have different sizes
#         for i in range(features.shape[0]):
#             track_feature = {
#                 "z": centroids[i, 0],
#                 "y": centroids[i, 1],
#                 "x": centroids[i, 2],
#                 "t": t,
#             }

#             if intensities is not None and i < len(intensities):
#                 track_feature["intensity"] = intensities[i]
#             if intensities_raw is not None and i < len(intensities_raw):
#                 track_feature["intensity_raw"] = intensities_raw[i]
#             if counts is not None and i < len(counts):
#                 track_feature["counts"] = counts[i]
#             if total_intensities is not None and i < len(total_intensities):
#                 track_feature["total_intensity"] = total_intensities[i]

#             for j in range(features.shape[1]):
#                 if explained_variance is not None:
#                     track_feature[f"feature_{j}"] = (
#                         features[i, j] * explained_variance[j]
#                     )
#                 else:
#                     track_feature[f"feature_{j}"] = features[i, j]
#             if run_pca:
#                 for j in range(pca_res.shape[1]):  # type: ignore
#                     track_feature[f"pca_feature_{j}"] = pca_res[i, j]  # type: ignore
#             track_features.append(track_feature)
#     return pd.DataFrame(track_features)


def _get_top_variance_features(
    lr_feats: torch.Tensor,
    variance_cutoff: float = 0.001,
) -> List[int]:
    Z, Y, X, num_features = lr_feats.shape
    feature_variances = []
    for feat_idx in range(num_features):
        single_feature = lr_feats[:, :, :, feat_idx].numpy()
        var = np.var(single_feature)
        feature_variances.append((feat_idx, var))
    feature_variances.sort(key=lambda x: x[1], reverse=True)

    top_variance_features = [
        feat_idx for feat_idx, var in feature_variances if var > variance_cutoff
    ]
    return top_variance_features


def get_features_table(
    file_paths: List[Path],
    explained_variance: Optional[np.ndarray] = None,
    centroid_type: str = "centroids_new",
    run_pca: bool = False,
    pca_fn=None,  # type: ignore
    attention_heads: int = 6,
    use_top_variance_features: bool = True,
    variance_cutoff: float = 0.001,
    load_features: Union[
        str, List[str]
    ] = "object_features",  # Can be single string or list of feature types
) -> pd.DataFrame:
    track_features = []  # list to store track updates (one per matched frame)
    top_variance_features = None  # Will be set from timepoint 0

    # Normalize load_features to list format
    if isinstance(load_features, str):
        if load_features == "all":
            feature_types = ["object_features", "feature_medians", "feature_sums"]
        else:
            feature_types = [load_features]
    else:
        feature_types = load_features

    # Validate feature types
    valid_features = ["object_features", "feature_medians", "feature_sums"]
    for feat_type in feature_types:
        if feat_type not in valid_features:
            raise ValueError(
                f"Invalid feature type: {feat_type}. Must be one of {valid_features}"
            )

    for t, file_path in enumerate(file_paths):
        centroids = io.imread(file_path.joinpath(f"{centroid_type}.tif"))

        # Initialize feature containers
        loaded_features = {}
        primary_features = None

        # Load requested feature types
        for feat_type in feature_types:
            try:
                loaded_features[feat_type] = io.imread(
                    file_path.joinpath(f"{feat_type}.tif")
                )
                if (
                    primary_features is None
                ):  # Use first loaded features as primary for PCA
                    primary_features = loaded_features[feat_type]
            except FileNotFoundError:
                if t == 0:
                    print(
                        f"Warning: {feat_type}.tif not found in {file_path}, skipping this feature type"
                    )
                continue

        # Set the main features variable for backward compatibility with PCA
        features = primary_features
        if features is None:
            raise ValueError(f"No valid feature files found in {file_path}")

        # Only compute top variance features from timepoint 0 if enabled
        if t == 0 and use_top_variance_features:
            lr_feats = torch.load(file_path.joinpath("lr_feats.pt"))
            top_variance_features = _get_top_variance_features(
                lr_feats, variance_cutoff=variance_cutoff
            )
            print(
                f"Selected {len(top_variance_features)} top variance features from t=0"
            )
        if t == 0:
            print(f"Centroids: {centroids.shape}, Features: {features.shape}")
        if run_pca:
            features = pca_fn.fit_transform(features)

        # Use minimum to avoid IndexError when arrays have different sizes
        for i in range(features.shape[0]):
            track_feature = {
                "z": centroids[i, 0],
                "y": centroids[i, 1],
                "x": centroids[i, 2],
                "t": t,
            }

            # Add features from primary feature set (with PCA if applied)
            for j in range(features.shape[1]):
                if explained_variance is not None:
                    if run_pca:
                        track_feature[f"pca_{j}"] = (
                            features[i, j] * explained_variance[j]
                        )
                    else:
                        track_feature[f"feature_{j}"] = (
                            features[i, j] * explained_variance[j]
                        )
                else:
                    if run_pca:
                        track_feature[f"pca_{j}"] = features[i, j]
                    else:
                        track_feature[f"feature_{j}"] = features[i, j]

            # Add all other loaded feature types
            for feat_type, feat_data in loaded_features.items():
                if (
                    feat_data is not primary_features and i < feat_data.shape[0]
                ):  # Skip primary features to avoid duplication
                    if use_top_variance_features and top_variance_features is not None:
                        # Only use selected high-variance features
                        for j in top_variance_features:
                            if j < feat_data.shape[1]:
                                if feat_type == "object_features":
                                    track_feature[f"obj_{j}"] = feat_data[i, j]
                                elif feat_type == "feature_medians":
                                    track_feature[f"median_{j}"] = feat_data[i, j]
                                elif feat_type == "feature_sums":
                                    track_feature[f"sum_{j}"] = feat_data[i, j]
                    else:
                        # Use all features
                        for j in range(feat_data.shape[1]):
                            if feat_type == "object_features":
                                track_feature[f"obj_{j}"] = feat_data[i, j]
                            elif feat_type == "feature_medians":
                                track_feature[f"median_{j}"] = feat_data[i, j]
                            elif feat_type == "feature_sums":
                                track_feature[f"sum_{j}"] = feat_data[i, j]

            track_features.append(track_feature)

    # Convert to DataFrame
    df = pd.DataFrame(track_features)

    # Add sum of sums if feature_sums were loaded
    if "feature_sums" in feature_types:
        sum_columns = [col for col in df.columns if col.startswith("sum_")]
        if sum_columns:
            df["sum_of_sum_per_feature"] = df[sum_columns].sum(axis=1)

    return df


def save_sparse_lr_feats(
    lr_feats: torch.Tensor,
    refined_mask_3d: np.ndarray,
    save_path: Path,
) -> None:
    Z, Y, X, C = lr_feats.shape
    flatten_lr_feats = lr_feats.view(Z * Y * X, C).numpy()
    N, D = flatten_lr_feats.shape
    fg_binary_mask_flat = refined_mask_3d.flatten()
    data = flatten_lr_feats[fg_binary_mask_flat].flatten().astype(np.float32)
    non_zero_count = data.size
    row_inds, col_inds = compute_indices(
        mask=fg_binary_mask_flat, non_zero_count=non_zero_count, D=D
    )
    csr_mat = csr_matrix((data, (row_inds, col_inds)), shape=(N, D), dtype=data.dtype)
    sp.save_npz(save_path.joinpath("lr_feats.npz"), csr_mat)


def enhanced_flat_field_correction(
    volume: np.ndarray,
    patch_size: tuple[int, int, int] = (448, 448, 448),
    blur_radius: float = 50.0,
    pixel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: cle.Device = None,
    method: str = "gaussian",  # "gaussian", "median", "rolling_ball"
) -> np.ndarray:
    """
    Enhanced flat-field correction for large 3D volumes with patch-wise contrast variations.

    This function applies correction to handle illumination gradients and contrast differences
    across different regions of the volume, particularly useful for [448, 448, 448] patches.

    Parameters
    ----------
    volume : np.ndarray
        Input image stack, shape (Z, Y, X)
    patch_size : tuple of int
        Size of patches for local correction
    blur_radius : float
        Radius for background estimation (larger values = more global correction)
    pixel_size : tuple of float
        (pixel_depth, pixel_height, pixel_width) for anisotropic correction
    device : cle.Device
        GPU device for computation
    method : str
        Method for background estimation: "gaussian", "median", "rolling_ball"

    Returns
    -------
    np.ndarray
        Flat-field corrected volume
    """
    if volume.dtype != np.float32 and volume.dtype != np.float64:
        volume = volume.astype(np.float32)

    corrected_volume = np.zeros_like(volume)

    # Calculate patch coordinates with overlap
    pz, py, px = patch_size
    overlap = 0.1  # 10% overlap
    stride_z = int(pz * (1 - overlap))
    stride_y = int(py * (1 - overlap))
    stride_x = int(px * (1 - overlap))

    Z, Y, X = volume.shape

    # Store weights for blending overlapping regions
    weight_volume = np.zeros_like(volume)

    for z_start in range(0, Z, stride_z):
        for y_start in range(0, Y, stride_y):
            for x_start in range(0, X, stride_x):
                z_end = min(z_start + pz, Z)
                y_end = min(y_start + py, Y)
                x_end = min(x_start + px, X)

                # Extract patch
                patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]

                if patch.size == 0:
                    continue

                # Apply flat field correction to patch
                if method == "gaussian":
                    corrected_patch = _gaussian_flat_field(
                        patch, blur_radius, pixel_size, device
                    )
                elif method == "median":
                    corrected_patch = _median_flat_field(patch, blur_radius, device)
                elif method == "rolling_ball":
                    corrected_patch = _rolling_ball_flat_field(
                        patch, blur_radius, device
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Create weight map for blending (higher in center, lower at edges)
                patch_shape = corrected_patch.shape
                z_weights = np.linspace(0.1, 1.0, patch_shape[0] // 2)
                z_weights = np.concatenate([z_weights, z_weights[::-1]])
                if len(z_weights) < patch_shape[0]:
                    z_weights = np.concatenate([z_weights, [1.0]])

                y_weights = np.linspace(0.1, 1.0, patch_shape[1] // 2)
                y_weights = np.concatenate([y_weights, y_weights[::-1]])
                if len(y_weights) < patch_shape[1]:
                    y_weights = np.concatenate([y_weights, [1.0]])

                x_weights = np.linspace(0.1, 1.0, patch_shape[2] // 2)
                x_weights = np.concatenate([x_weights, x_weights[::-1]])
                if len(x_weights) < patch_shape[2]:
                    x_weights = np.concatenate([x_weights, [1.0]])

                weights = np.outer(z_weights, np.outer(y_weights, x_weights)).reshape(
                    patch_shape
                )

                # Add to corrected volume with weights
                corrected_volume[z_start:z_end, y_start:y_end, x_start:x_end] += (
                    corrected_patch * weights
                )
                weight_volume[z_start:z_end, y_start:y_end, x_start:x_end] += weights

    # Normalize by weights to handle overlapping regions
    weight_volume[weight_volume == 0] = 1  # Avoid division by zero
    corrected_volume = corrected_volume / weight_volume

    return corrected_volume


def _gaussian_flat_field(
    patch: np.ndarray,
    blur_radius: float,
    pixel_size: tuple[float, float, float],
    device: cle.Device = None,
) -> np.ndarray:
    """Apply Gaussian-based flat field correction to a patch"""
    pd, ph, pw = pixel_size
    x_y_ratio = pw / ph
    z_x_ratio = pd / pw

    sigma_x = blur_radius
    sigma_y = blur_radius * x_y_ratio
    sigma_z = blur_radius / z_x_ratio

    background = np.asarray(
        cle.gaussian_blur(
            patch, sigma_z=sigma_z, sigma_y=sigma_y, sigma_x=sigma_x, device=device
        )
    )

    mean_bg = background.mean()
    corrected = patch / (background + 1e-8) * mean_bg
    return corrected


def _median_flat_field(
    patch: np.ndarray,
    blur_radius: float,
    device: cle.Device = None,
) -> np.ndarray:
    """Apply median-based flat field correction to a patch"""
    radius = int(blur_radius)
    background = np.asarray(
        cle.median_box(
            patch,
            radius_x=radius,
            radius_y=radius,
            radius_z=radius,
            device=device,
        )
    )

    mean_bg = background.mean()
    corrected = patch / (background + 1e-8) * mean_bg
    return corrected


def _rolling_ball_flat_field(
    patch: np.ndarray,
    radius: float,
    device: cle.Device = None,
) -> np.ndarray:
    """Apply rolling ball-based flat field correction to a patch"""
    # Using top-hat filter as approximation for rolling ball
    radius_int = int(radius)
    background = np.asarray(
        cle.top_hat_box(
            patch,
            radius_x=radius_int,
            radius_y=radius_int,
            radius_z=radius_int,
            device=device,
        )
    )

    # Invert the top-hat result to get background estimation
    background = patch - background

    mean_bg = background.mean()
    corrected = patch / (background + 1e-8) * mean_bg
    return corrected


# @njit(fastmath=True, cache=True)
def index_by_instance_segmentation(
    new_img: np.ndarray,
    instance_seg: np.ndarray,
    instance_ids: Optional[List[int]] = None,
    return_coordinates: bool = False,
    background_value: int = 0,
) -> Union[
    np.ndarray, Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]
]:
    unique_ids = np.unique(instance_seg)
    unique_ids = unique_ids[unique_ids != background_value]

    if instance_ids is not None:
        unique_ids = np.array([uid for uid in unique_ids if uid in instance_ids])

    mask = np.zeros_like(instance_seg, dtype=bool)
    coordinates_list = []

    for instance_id in unique_ids:
        instance_mask = instance_seg == instance_id
        mask |= instance_mask

        if return_coordinates:
            coords = np.nonzero(instance_mask)
            coordinates_list.append(coords)

    if return_coordinates:
        return mask, coordinates_list
    else:
        return mask


def analyze_longest_tracks(
    linked_df: pd.DataFrame, top_n: int = 10, show_intensity: bool = True
) -> None:
    """
    Analyze the longest tracks in the linked dataframe.

    Parameters:
    -----------
    linked_df : pd.DataFrame
        The tracking dataframe with columns like track_id, t, track_length, t_0, etc.
    top_n : int
        Number of longest tracks to analyze (default: 10)
    show_intensity : bool
        Whether to show intensity data if available
    """

    # Total number of unique particles
    total_particles = linked_df["track_id"].nunique()
    print(f"Total number of particles in linked dataframe: {total_particles}")
    print(f"Total number of track points: {len(linked_df)}")

    # Get track summary statistics
    track_stats = (
        linked_df.groupby("track_id")
        .agg({
            "track_length": "first",  # track_length is the same for all rows of a track
            "t_0": "first",  # starting frame
            "t": ["min", "max"],  # first and last time points
            "z": ["mean", "std"],  # movement statistics
            "y": ["mean", "std"],
            "x": ["mean", "std"],
        })
        .round(3)
    )

    # Flatten column names
    track_stats.columns = [
        "_".join(col).strip() if col[1] else col[0] for col in track_stats.columns
    ]
    track_stats = track_stats.rename(columns={"t_min": "t_start", "t_max": "t_end"})

    # Add intensity statistics if available
    if "intensity" in linked_df.columns and show_intensity:
        intensity_stats = (
            linked_df.groupby("track_id")["intensity"]
            .agg(["mean", "std", "min", "max"])
            .round(3)
        )
        intensity_stats.columns = [
            "intensity_" + col for col in intensity_stats.columns
        ]
        track_stats = track_stats.join(intensity_stats)

    # Sort by track length (descending) and get top N
    track_stats_sorted = track_stats.sort_values("track_length_first", ascending=False)
    longest_tracks = track_stats_sorted.head(top_n)

    print(f"\nTop {top_n} longest tracks:")
    print("=" * 80)

    for idx, (track_id, stats) in enumerate(longest_tracks.iterrows(), 1):
        print(f"\n{idx}. Track ID: {track_id}")
        print(f"   Track Length: {int(stats['track_length_first'])} frames")
        print(f"   Start Frame: {int(stats['t_0_first'])}")
        print(f"   Time Range: {int(stats['t_start'])} - {int(stats['t_end'])}")
        print("   Position (mean ± std):")
        print(f"     Z: {stats['z_mean']:.2f} ± {stats['z_std']:.2f}")
        print(f"     Y: {stats['y_mean']:.2f} ± {stats['y_std']:.2f}")
        print(f"     X: {stats['x_mean']:.2f} ± {stats['x_std']:.2f}")

        if "intensity_mean" in stats:
            print(
                f"   Intensity: {stats['intensity_mean']:.3f} ± {stats['intensity_std']:.3f}"
            )
            print(
                f"   Intensity Range: {stats['intensity_min']:.3f} - {stats['intensity_max']:.3f}"
            )

    # Show detailed time series for longest tracks
    print(f"\n\nDetailed time series for top {min(3, top_n)} longest tracks:")
    print("=" * 80)

    longest_track_ids = longest_tracks.head(3).index.tolist()

    for track_id in longest_track_ids:
        track_data = linked_df[linked_df["track_id"] == track_id].sort_values("t")
        print(f"\nTrack ID {track_id} (Length: {len(track_data)} frames):")

        # Show subset of columns for readability
        display_cols = ["t", "z", "y", "x"]
        if "intensity" in track_data.columns and show_intensity:
            display_cols.append("intensity")

        print(track_data[display_cols].to_string(index=False))

    # Summary statistics
    print("\n\nSummary Statistics:")
    print("=" * 50)
    print("Track length distribution:")
    print(f"  Mean: {track_stats['track_length_first'].mean():.1f}")
    print(f"  Median: {track_stats['track_length_first'].median():.1f}")
    print(f"  Min: {track_stats['track_length_first'].min()}")
    print(f"  Max: {track_stats['track_length_first'].max()}")
    print(f"  Std: {track_stats['track_length_first'].std():.1f}")

    return track_stats_sorted  # type: ignore


def plot_track_analysis(
    linked_df: pd.DataFrame, track_stats: pd.DataFrame, top_n: int = 10
) -> None:
    """
    Create plots for track analysis.

    Parameters:
    -----------
    linked_df : pd.DataFrame
        The tracking dataframe
    track_stats : pd.DataFrame
        Track statistics from analyze_longest_tracks
    top_n : int
        Number of tracks to highlight in plots
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1. Track length distribution
    axes[0].hist(
        track_stats["track_length_first"], bins=50, alpha=0.7, edgecolor="black"
    )
    axes[0].set_xlabel("Track Length (frames)")
    axes[0].set_ylabel("Number of Tracks")
    axes[0].set_title("Distribution of Track Lengths")
    axes[0].axvline(
        track_stats["track_length_first"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {track_stats['track_length_first'].mean():.1f}",
    )
    axes[0].legend()

    # 2. Start frame distribution
    axes[1].hist(track_stats["t_0_first"], bins=30, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Start Frame")
    axes[1].set_ylabel("Number of Tracks")
    axes[1].set_title("Distribution of Track Start Frames")

    # 3. Intensity over time for longest tracks (if available)
    if "intensity" in linked_df.columns:
        longest_track_ids = track_stats.head(top_n).index.tolist()
        for i, track_id in enumerate(longest_track_ids):
            track_data = linked_df[linked_df["track_id"] == track_id].sort_values("t")
            axes[2].plot(
                track_data["t"],
                track_data["intensity"],
                label=f"Track {track_id}",
                alpha=0.7,
                linewidth=2,
            )
        axes[2].set_xlabel("Time Frame")
        axes[2].set_ylabel("Intensity")
        axes[2].set_title("Intensity Over Time (Top 5 Longest Tracks)")
        axes[2].legend()
    plt.tight_layout()
    plt.show()
