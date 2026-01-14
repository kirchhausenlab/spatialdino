from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import sklearn.decomposition as skd
import torch
import torch.nn.functional as F
import torch_pca
from monai.config.type_definitions import NdarrayOrTensor
from numba import njit


def cosine_dist_func(a: NdarrayOrTensor, b: NdarrayOrTensor) -> torch.Tensor:
    # Normalize the vectors
    is_numpy = False
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
        b = torch.from_numpy(b)
        is_numpy = True

    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)  # type: ignore
    # Cosine similarity is dot product of normalized vectors
    # Convert to distance: 1 - cosine_similarity
    dist = 1 - torch.mm(a_norm, b_norm.t())
    if is_numpy:
        return dist.cpu().numpy()  # type: ignore
    return dist


def euclidean_dist_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2)


@torch.no_grad()
def kmeans_fit_predict(
    feats: NdarrayOrTensor,
    n_samples: int = 256_000,
    n_clusters: int = 80,
    probs: Optional[np.ndarray] = None,
    max_samples_per_batch: int = 256_000,
    max_iter: int = 300,
    device: Union[str, torch.device] = "cuda",
    init: Literal["kmeans++", "random", "precomputed"] = "kmeans++",
    centroids: Optional[np.ndarray] = None,
    dist_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cosine_dist_func,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assumes `feats` is an (N, D) array of unit-length vectors (already L2-normalized).
    Returns:
        labels: (N,) cluster assignments
        centroids: (k, D) unit-length cluster centers
    """
    N = feats.shape[0]

    # Sample a subset of data for training centroids
    sample_size = min(n_samples, N)

    if probs is not None:
        non_zero_probs = (probs > 0).sum()
        sample_size = min(sample_size, non_zero_probs)

    random_inds = np.random.choice(N, sample_size, replace=False, p=probs)

    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)

    # For training, it's acceptable to move the sampled subset to GPU
    train_set = feats[random_inds].float().to(device, non_blocking=True)

    if init == "kmeans++":
        # Replace random initialization with k-means++ initialization
        means = torch.zeros(
            (n_clusters, train_set.shape[1]), device=device, dtype=torch.float32
        )
        # Choose first centroid randomly
        first_centroid = train_set[torch.randint(0, train_set.shape[0], (1,))]
        means[0] = first_centroid

        # Choose remaining centroids
        for k in range(1, n_clusters):
            # Compute distances to existing centroids
            dists = dist_func(train_set, means[:k])
            # Get minimum distance for each point
            min_dists = torch.min(dists, dim=1)[0]
            # Square distances to get probabilities
            selection_probs = min_dists**2
            selection_probs = selection_probs / (torch.sum(selection_probs) + 1e-6)
            # Sample next centroid based on probabilities
            next_centroid_idx = torch.multinomial(selection_probs, 1)
            means[k] = train_set[next_centroid_idx]
    elif init == "random":
        # Initialize centroids by randomly picking points from the training set.
        means = train_set[torch.randint(0, train_set.shape[0], (n_clusters,))].clone()
    elif init == "precomputed":
        means = torch.from_numpy(centroids)
    else:
        raise ValueError(f"Invalid initialization method: {init}")

    n_train = train_set.shape[0]
    assignments = torch.zeros(n_train, device=device, dtype=torch.int64)

    means = F.normalize(means, p=2, dim=-1)
    means = means.to(device, non_blocking=True)

    center_shift_total = 0.0
    for iteration in range(max_iter):
        # Assignment step: process training set in batches
        for start in range(0, n_train, max_samples_per_batch):
            batch = train_set[start : start + max_samples_per_batch]  # already on GPU
            # Compute Euclidean distances between batch and centroids
            dists = dist_func(batch, means)
            assignments[start : start + max_samples_per_batch] = torch.argmin(
                dists, dim=1
            )

        # Update step: compute new centroids by averaging assigned points.
        new_means = torch.zeros_like(means)
        counts = torch.zeros(n_clusters, device=device, dtype=torch.int64)
        new_means.index_add_(0, assignments, train_set)
        counts.index_add_(
            0,
            assignments,
            torch.ones_like(assignments, dtype=counts.dtype, device=device),
        )
        # handle empty clusters: re‑seed to random points
        empty = counts == 0
        if empty.any():
            num_empty = int(empty.sum().item())
            reinit_idx = torch.randint(0, n_train, (num_empty,), device=device)
            new_means[empty] = train_set[reinit_idx]
            counts[empty] = 1  # avoid div‑by‑0

        new_means = new_means / counts.unsqueeze_(1)
        new_means = F.normalize(new_means, p=2, dim=-1)

        center_shift_total = ((new_means - means) ** 2).sum()

        means = new_means

        # convergence check
        if center_shift_total <= tol:
            break

    # Inference: assign every point in feats_flat to its nearest centroid.
    # We do not move the entire feats_flat to GPU at once.
    labels = np.zeros(N, dtype=np.int64)
    distances = np.zeros((N, n_clusters), dtype=np.float32)
    for start in range(0, N, max_samples_per_batch):
        # Convert the current mini-batch from CPU (NumPy) to a GPU tensor
        batch = feats[start : start + max_samples_per_batch].float().to(device)
        dists = dist_func(batch, means)
        min_vals, min_indices = torch.min(dists, dim=1)
        distances[start : start + max_samples_per_batch] = dists.cpu().numpy()
        labels[start : start + max_samples_per_batch] = min_indices.cpu().numpy()

    centroids = means.cpu().numpy()

    return labels, centroids, distances


class PCA(object):
    def __init__(
        self,
        n_components: int,
        use_torch_pca: bool = True,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ):
        self.n_components = n_components
        self.use_torch_pca = use_torch_pca
        self.device = device
        # Initialize PCA once, avoiding repeated string comparison
        if self.use_torch_pca:
            self.pca = torch_pca.PCA(n_components=self.n_components, **kwargs)
        else:
            self.pca = skd.PCA(n_components=self.n_components, **kwargs)

    def _prepare_input(self, x):
        if self.use_torch_pca:
            # Combine conditional operations to reduce checks
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            elif x.dtype != torch.float32:
                x = x.float()
            # Only move to device if needed
            if self.device is not None and x.device != self.device:
                x = x.to(self.device, non_blocking=True)
        else:
            # For sklearn PCA, ensure we have numpy array
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        return x

    def fit(self, x):
        x = self._prepare_input(x)
        self.pca.fit(x)  # type: ignore
        return self

    def transform(self, x):
        x = self._prepare_input(x)
        transformed = self.pca.transform(x)  # type: ignore
        # Only convert to numpy if necessary
        return (
            transformed.detach().cpu().numpy()
            if isinstance(transformed, torch.Tensor)
            else transformed
        )

    def fit_transform(self, x):
        # Avoid duplicate preparation by leveraging existing methods
        x = self._prepare_input(x)
        transformed = self.pca.fit_transform(x)  # type: ignore
        return (
            transformed.detach().cpu().numpy()
            if isinstance(transformed, torch.Tensor)
            else transformed
        )


def run_pca(
    pca: PCA,
    feats: np.ndarray,
    non_zero_mask: np.ndarray,
    input_shape: Tuple[int, int, int],
    n_components: int,
    dtype: str = "float32",
) -> np.ndarray:
    Z, Y, X = input_shape
    pca_features_rgb = np.zeros((Z * Y * X, n_components), dtype=dtype)
    pca_features_rgb_fg = pca.fit_transform(feats)
    min_val = np.min(pca_features_rgb_fg, axis=0, keepdims=True)
    max_val = np.max(pca_features_rgb_fg, axis=0, keepdims=True)
    pca_features_rgb_fg = (pca_features_rgb_fg - min_val) / (max_val - min_val + 1e-6)
    pca_features_rgb[non_zero_mask] = pca_features_rgb_fg
    pca_features_rgb = pca_features_rgb.reshape(Z, Y, X, n_components)

    min_val = pca_features_rgb.min()
    max_val = pca_features_rgb.max()
    pca_features_rgb = (pca_features_rgb - min_val) / (max_val - min_val + 1e-6)
    pca_features_rgb = np.clip(pca_features_rgb * 255, 0, 255).astype(np.uint8)
    return pca_features_rgb


def convert_instance_seg_to_csr_mat(
    instance_seg: np.ndarray,
) -> Tuple[sp.csr_matrix, Tuple[int, int, int]]:
    Z, Y, X = instance_seg.shape
    instance_seg_flat = instance_seg.flatten()
    sparse_matrix = sp.csr_matrix(instance_seg_flat)
    return sparse_matrix, (Z, Y, X)


def convert_csr_mat_to_instance_seg(
    csr_mat: sp.csr_matrix, shape: Tuple[int, int, int]
) -> np.ndarray:
    return csr_mat.toarray().reshape(shape)


@njit(
    fastmath=True,
    cache=True,
)
def _calculate_features_with_local_background(
    instance_segmentation: np.ndarray,
    lr_feats_flat: np.ndarray,
    fg_mask: np.ndarray,
    z_ratio: float,
    y_ratio: float,
    x_ratio: float,
    lr_shape: Tuple[int, int, int],
    unnormalized_vol: np.ndarray,
    shell_radius: int = 4,
    inner_buffer: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate features with local background correction using shell sampling.
    """
    z_ind, y_ind, x_ind = np.nonzero(fg_mask)
    fg_mask_flat = fg_mask.flatten()
    unique_labels = np.unique(instance_segmentation.flatten()[fg_mask_flat])

    # Create mapping from label to index using array indexing
    max_label = int(unique_labels.max())
    label_to_idx = np.full(max_label + 1, -1, dtype=np.int32)
    for i in range(len(unique_labels)):
        label_to_idx[int(unique_labels[i])] = i

    Z_lr, Y_lr, X_lr = lr_shape
    Z_hr, Y_hr, X_hr = instance_segmentation.shape

    centroids = np.zeros((len(unique_labels), 3), dtype=np.float32)
    m_0 = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_z = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_y = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_x = np.zeros(len(unique_labels), dtype=np.float32)

    counts = np.zeros(len(unique_labels), dtype=np.uint64)
    intensities = np.zeros(len(unique_labels), dtype=np.float32)
    local_backgrounds = np.zeros(len(unique_labels), dtype=np.float32)

    object_features = np.zeros(
        (len(unique_labels), lr_feats_flat.shape[-1]),
        dtype=np.float32,
    )

    # First pass: calculate features and accumulate intensities
    for idx, (z, y, x) in enumerate(zip(z_ind, y_ind, x_ind)):
        label_val = instance_segmentation[z, y, x]
        label_idx = label_to_idx[int(label_val)]
        counts[label_idx] += 1

        # Map high-res coordinates to low-res coordinates
        z_lr = int(z * z_ratio)
        y_lr = int(y * y_ratio)
        x_lr = int(x * x_ratio)

        # Clamp to valid indices
        z_lr = min(z_lr, Z_lr - 1)
        y_lr = min(y_lr, Y_lr - 1)
        x_lr = min(x_lr, X_lr - 1)

        lr_flat_idx = z_lr * Y_lr * X_lr + y_lr * X_lr + x_lr

        # Accumulate features from low-res space
        object_features[label_idx] += lr_feats_flat[lr_flat_idx]

        intensities[label_idx] += unnormalized_vol[z, y, x]

        weight = unnormalized_vol[z, y, x]
        m_0[label_idx] += weight
        m_1_z[label_idx] += z * weight
        m_1_y[label_idx] += y * weight
        m_1_x[label_idx] += x * weight

    # Calculate centroids
    centroids[:, 0] = m_1_z / (m_0 + 1e-6)
    centroids[:, 1] = m_1_y / (m_0 + 1e-6)
    centroids[:, 2] = m_1_x / (m_0 + 1e-6)

    # Average features by count
    object_features /= counts[:, None]

    # Second pass: calculate local background for each object
    for label_idx in range(len(unique_labels)):
        current_label = unique_labels[label_idx]

        # Calculate shell background for this object
        bg_sum = 0.0
        bg_count = 0

        # Go through all object pixels and check their shell
        for idx, (z, y, x) in enumerate(zip(z_ind, y_ind, x_ind)):
            if instance_segmentation[z, y, x] != current_label:
                continue

            # Check shell coordinates around this pixel (ring sampling with buffer)
            for dz in range(-shell_radius, shell_radius + 1):
                for dy in range(-shell_radius, shell_radius + 1):
                    for dx in range(-shell_radius, shell_radius + 1):
                        # Skip if it's the center point
                        if dz == 0 and dy == 0 and dx == 0:
                            continue

                        # Check if within shell ring (between inner_buffer and shell_radius)
                        distance_sq = dz * dz + dy * dy + dx * dx
                        inner_sq = inner_buffer * inner_buffer
                        outer_sq = shell_radius * shell_radius

                        # Only sample if in the ring: distance > inner_buffer AND distance <= shell_radius
                        if distance_sq <= inner_sq or distance_sq > outer_sq:
                            continue

                        z_shell = z + dz
                        y_shell = y + dy
                        x_shell = x + dx

                        # Check bounds
                        if (
                            z_shell >= 0
                            and z_shell < Z_hr
                            and y_shell >= 0
                            and y_shell < Y_hr
                            and x_shell >= 0
                            and x_shell < X_hr
                        ):
                            # Check if this shell coordinate is background (label = 0)
                            if instance_segmentation[z_shell, y_shell, x_shell] == 0:
                                bg_sum += unnormalized_vol[z_shell, y_shell, x_shell]
                                bg_count += 1

        # Calculate local background average
        if bg_count > 0:
            local_backgrounds[label_idx] = bg_sum / bg_count
        else:
            # Fallback to global background if no shell background found
            local_backgrounds[label_idx] = 0.0

    # Apply local background correction to intensities
    corrected_intensities = (intensities / counts) - local_backgrounds
    total_intensities = intensities - local_backgrounds * counts
    return (
        centroids,
        object_features,
        corrected_intensities,
        total_intensities,
    )


def calculate_features_with_local_background(
    instance_segmentation: np.ndarray,
    lr_feats_flat: np.ndarray,
    fg_mask: np.ndarray,
    z_ratio: float,
    y_ratio: float,
    x_ratio: float,
    lr_shape: Tuple[int, int, int],
    unnormalized_vol: np.ndarray,
    pca_3: PCA,
    shell_radius: int = 4,
    inner_buffer: int = 1,
    return_object_features: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    (
        centroids,
        object_features,
        corrected_intensities,
        intensities,
    ) = _calculate_features_with_local_background(
        instance_segmentation=instance_segmentation,
        lr_feats_flat=lr_feats_flat,
        fg_mask=fg_mask,
        z_ratio=z_ratio,
        y_ratio=y_ratio,
        x_ratio=x_ratio,
        lr_shape=lr_shape,
        unnormalized_vol=unnormalized_vol,
        shell_radius=shell_radius,
        inner_buffer=inner_buffer,
    )
    pca_3_res = pca_3.fit_transform(object_features)
    if return_object_features:
        return (
            centroids,
            corrected_intensities,
            pca_3_res,
            intensities,
            object_features,
        )
    else:
        return centroids, corrected_intensities, pca_3_res, intensities


@njit(
    fastmath=True,
    cache=True,
)
def _calculate_features_with_local_background_with_matched_background(
    instance_segmentation: np.ndarray,
    lr_feats_flat: np.ndarray,
    fg_mask: np.ndarray,
    z_ratio: float,
    y_ratio: float,
    x_ratio: float,
    lr_shape: Tuple[int, int, int],
    unnormalized_vol: np.ndarray,
    shell_radius: int = 4,
    inner_buffer: int = 2,
    random_seed: int = 42,  # Added for reproducibility
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate features with local background correction using shell sampling.
    Also creates matched background objects with same voxel counts.
    """
    # Set random seed for reproducible background sampling
    np.random.seed(random_seed)

    z_ind, y_ind, x_ind = np.nonzero(fg_mask)
    fg_mask_flat = fg_mask.flatten()
    unique_labels = np.unique(instance_segmentation.flatten()[fg_mask_flat])

    # Create mapping from label to index using array indexing
    max_label = int(unique_labels.max())
    label_to_idx = np.full(max_label + 1, -1, dtype=np.int32)
    for i in range(len(unique_labels)):
        label_to_idx[int(unique_labels[i])] = i

    Z_lr, Y_lr, X_lr = lr_shape
    Z_hr, Y_hr, X_hr = instance_segmentation.shape

    centroids = np.zeros((len(unique_labels), 3), dtype=np.float32)
    m_0 = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_z = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_y = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_x = np.zeros(len(unique_labels), dtype=np.float32)

    counts = np.zeros(len(unique_labels), dtype=np.uint64)
    intensities = np.zeros(len(unique_labels), dtype=np.float32)
    local_backgrounds = np.zeros(len(unique_labels), dtype=np.float32)

    object_features = np.zeros(
        (len(unique_labels), lr_feats_flat.shape[-1]),
        dtype=np.float32,
    )

    # First pass: calculate features and accumulate intensities
    for idx, (z, y, x) in enumerate(zip(z_ind, y_ind, x_ind)):
        label_val = instance_segmentation[z, y, x]
        label_idx = label_to_idx[int(label_val)]
        counts[label_idx] += 1

        # Map high-res coordinates to low-res coordinates
        z_lr = int(z * z_ratio)
        y_lr = int(y * y_ratio)
        x_lr = int(x * x_ratio)

        # Clamp to valid indices
        z_lr = min(z_lr, Z_lr - 1)
        y_lr = min(y_lr, Y_lr - 1)
        x_lr = min(x_lr, X_lr - 1)

        lr_flat_idx = z_lr * Y_lr * X_lr + y_lr * X_lr + x_lr

        # Accumulate features from low-res space
        object_features[label_idx] += lr_feats_flat[lr_flat_idx]

        intensities[label_idx] += unnormalized_vol[z, y, x]

        weight = unnormalized_vol[z, y, x]
        m_0[label_idx] += weight
        m_1_z[label_idx] += z * weight
        m_1_y[label_idx] += y * weight
        m_1_x[label_idx] += x * weight

    # Calculate centroids
    centroids[:, 0] = m_1_z / (m_0 + 1e-6)
    centroids[:, 1] = m_1_y / (m_0 + 1e-6)
    centroids[:, 2] = m_1_x / (m_0 + 1e-6)

    # Average features by count
    object_features /= counts[:, None]

    # Second pass: calculate local background for each object
    for label_idx in range(len(unique_labels)):
        current_label = unique_labels[label_idx]

        # Calculate shell background for this object
        bg_sum = 0.0
        bg_count = 0

        # Go through all object pixels and check their shell
        for idx, (z, y, x) in enumerate(zip(z_ind, y_ind, x_ind)):
            if instance_segmentation[z, y, x] != current_label:
                continue

            # Check shell coordinates around this pixel (ring sampling with buffer)
            for dz in range(-shell_radius, shell_radius + 1):
                for dy in range(-shell_radius, shell_radius + 1):
                    for dx in range(-shell_radius, shell_radius + 1):
                        # Skip if it's the center point
                        if dz == 0 and dy == 0 and dx == 0:
                            continue

                        # Check if within shell ring (between inner_buffer and shell_radius)
                        distance_sq = dz * dz + dy * dy + dx * dx
                        inner_sq = inner_buffer * inner_buffer
                        outer_sq = shell_radius * shell_radius

                        # Only sample if in the ring: distance > inner_buffer AND distance <= shell_radius
                        if distance_sq <= inner_sq or distance_sq > outer_sq:
                            continue

                        z_shell = z + dz
                        y_shell = y + dy
                        x_shell = x + dx

                        # Check bounds
                        if (
                            z_shell >= 0
                            and z_shell < Z_hr
                            and y_shell >= 0
                            and y_shell < Y_hr
                            and x_shell >= 0
                            and x_shell < X_hr
                        ):
                            # Check if this shell coordinate is background (label = 0)
                            if instance_segmentation[z_shell, y_shell, x_shell] == 0:
                                bg_sum += unnormalized_vol[z_shell, y_shell, x_shell]
                                bg_count += 1

        # Calculate local background average
        if bg_count > 0:
            local_backgrounds[label_idx] = bg_sum / bg_count
        else:
            # Fallback to global background if no shell background found
            local_backgrounds[label_idx] = 0.0

    # Apply local background correction to intensities
    corrected_intensities = (intensities / counts) - local_backgrounds
    total_intensities = intensities - local_backgrounds * counts

    # NEW: Create matched background objects
    # Pre-compute all background coordinates (where instance_segmentation == 0)
    bg_coords = []
    for z in range(Z_hr):
        for y in range(Y_hr):
            for x in range(X_hr):
                if instance_segmentation[z, y, x] == 0:
                    bg_coords.append((z, y, x))

    # Convert to numpy array for easier indexing
    bg_coords_array = np.array(bg_coords, dtype=np.int32)
    total_bg_voxels = len(bg_coords_array)

    # Initialize background features array
    background_features = np.zeros(
        (len(unique_labels), lr_feats_flat.shape[-1]),
        dtype=np.float32,
    )

    # Sample background voxels for each object to match their voxel counts
    bg_start_idx = 0
    for label_idx in range(len(unique_labels)):
        voxel_count = int(counts[label_idx])

        # Ensure we don't exceed available background voxels
        if bg_start_idx + voxel_count > total_bg_voxels:
            # If we run out of background voxels, sample with replacement
            bg_indices = np.random.choice(
                total_bg_voxels, size=voxel_count, replace=True
            )
        else:
            # Sample without replacement for better spatial distribution
            bg_indices = np.arange(bg_start_idx, bg_start_idx + voxel_count)
            # Shuffle to randomize spatial distribution
            np.random.shuffle(bg_indices)
            bg_start_idx += voxel_count

        # Extract features for sampled background voxels
        bg_feature_sum = np.zeros(lr_feats_flat.shape[-1], dtype=np.float32)

        for i in range(voxel_count):
            coord_idx = bg_indices[i]
            z_bg, y_bg, x_bg = bg_coords_array[coord_idx]

            # Map to low-res coordinates (same logic as for objects)
            z_lr = int(z_bg * z_ratio)
            y_lr = int(y_bg * y_ratio)
            x_lr = int(x_bg * x_ratio)

            # Clamp to valid indices
            z_lr = min(z_lr, Z_lr - 1)
            y_lr = min(y_lr, Y_lr - 1)
            x_lr = min(x_lr, X_lr - 1)

            lr_flat_idx = z_lr * Y_lr * X_lr + y_lr * X_lr + x_lr
            bg_feature_sum += lr_feats_flat[lr_flat_idx]

        # Average features by count (same as for objects)
        background_features[label_idx] = bg_feature_sum / voxel_count

    return (
        centroids,
        object_features,
        corrected_intensities,
        total_intensities,
        background_features,  # NEW: Background object features
        counts,  # NEW: Also return counts for reference
    )


@njit(
    fastmath=True,
    cache=True,
)
def calculate_features_per_particle(
    instance_segmentation: np.ndarray,
    lr_feats_flat: np.ndarray,
    fg_mask: np.ndarray,
    z_ratio: float,
    y_ratio: float,
    x_ratio: float,
    lr_shape: Tuple[int, int, int],
    unnormalized_vol: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    np.ndarray,
]:
    z_ind, y_ind, x_ind = np.nonzero(fg_mask)
    fg_mask_flat = fg_mask.flatten()
    unique_labels = np.unique(instance_segmentation.flatten()[fg_mask_flat])

    # Create mapping from label to index using array indexing
    max_label = int(unique_labels.max())
    label_to_idx = np.full(max_label + 1, -1, dtype=np.int32)
    for i in range(len(unique_labels)):
        label_to_idx[int(unique_labels[i])] = i

    Z_lr, Y_lr, X_lr = lr_shape
    Z_hr, Y_hr, X_hr = instance_segmentation.shape

    centroids = np.zeros((len(unique_labels), 3), dtype=np.float32)
    m_0 = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_z = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_y = np.zeros(len(unique_labels), dtype=np.float32)
    m_1_x = np.zeros(len(unique_labels), dtype=np.float32)
    counts = np.zeros(len(unique_labels), dtype=np.uint64)
    intensities = np.zeros(len(unique_labels), dtype=np.float32)

    # Feature statistics arrays
    feature_sums = np.zeros(
        (len(unique_labels), lr_feats_flat.shape[-1]), dtype=np.float32
    )
    feature_means = np.zeros(
        (len(unique_labels), lr_feats_flat.shape[-1]), dtype=np.float32
    )
    feature_medians = np.zeros(
        (len(unique_labels), lr_feats_flat.shape[-1]), dtype=np.float32
    )

    # Store per-voxel features for each particle
    # Since numba has issues with List[np.ndarray], we'll use a different approach
    max_voxels_per_particle = 4000  # Adjust based on your data
    per_voxel_features = np.zeros(
        (len(unique_labels), max_voxels_per_particle, lr_feats_flat.shape[-1]),
        dtype=np.float32,
    )
    per_voxel_counts = np.zeros(len(unique_labels), dtype=np.int32)

    # First pass: collect all features per particle

    for idx, (z, y, x) in enumerate(zip(z_ind, y_ind, x_ind)):
        label_val = instance_segmentation[z, y, x]
        label_idx = label_to_idx[int(label_val)]
        counts[label_idx] += 1

        # Map high-res coordinates to low-res coordinates
        z_lr = int(z * z_ratio)
        y_lr = int(y * y_ratio)
        x_lr = int(x * x_ratio)

        # Clamp to valid indices
        z_lr = min(z_lr, Z_lr - 1)
        y_lr = min(y_lr, Y_lr - 1)
        x_lr = min(x_lr, X_lr - 1)

        lr_flat_idx = z_lr * Y_lr * X_lr + y_lr * X_lr + x_lr

        # Get features for this voxel
        voxel_features = lr_feats_flat[lr_flat_idx]

        # Accumulate sums for mean calculation
        feature_sums[label_idx] += voxel_features

        # Store per-voxel features
        if per_voxel_counts[label_idx] < max_voxels_per_particle:
            per_voxel_features[label_idx, per_voxel_counts[label_idx]] = voxel_features
        per_voxel_counts[label_idx] += 1

        intensities[label_idx] += unnormalized_vol[z, y, x]
        weight = unnormalized_vol[z, y, x]
        m_0[label_idx] += weight
        m_1_z[label_idx] += z * weight
        m_1_y[label_idx] += y * weight
        m_1_x[label_idx] += x * weight

    # Calculate centroids
    centroids[:, 0] = m_1_z / (m_0 + 1e-6)
    centroids[:, 1] = m_1_y / (m_0 + 1e-6)
    centroids[:, 2] = m_1_x / (m_0 + 1e-6)

    feature_means = feature_sums / counts[:, None]

    per_voxel_arrays = []
    for i in range(len(unique_labels)):
        actual_count = min(per_voxel_counts[i], max_voxels_per_particle)
        per_voxel_arrays.append(per_voxel_features[i, :actual_count, :].copy())
        if actual_count > 0:
            for j in range(lr_feats_flat.shape[-1]):
                particle_features = per_voxel_features[i, :actual_count, j]
                sorted_features = np.sort(particle_features)
                median_idx = actual_count // 2
                feature_medians[i, j] = sorted_features[median_idx]

    # Create list of per-voxel arrays for return (trimmed to actual counts)
    # per_voxel_arrays = []
    # for i in range(len(unique_labels)):
    #     actual_count = min(per_voxel_counts[i], max_voxels_per_particle)
    #     per_voxel_arrays.append(per_voxel_features[i, :actual_count, :].copy())

    corrected_intensities = intensities / counts

    return (
        centroids,
        corrected_intensities,
        feature_medians,  # New: median of features per particle
        feature_sums,  # New: sum of features per particle
        per_voxel_arrays,  # New: per-voxel feature arrays for each particle
        feature_means,  # New: mean of features per particle (replaces old average_features)
    )
