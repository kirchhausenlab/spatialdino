from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import Tensor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DEFAULT_DENSITIES_FILENAME = "probmap_densities.npz"
DEFAULT_Z_CHUNK_SIZE = 4


@dataclass(frozen=True)
class TimepointPaths:
    name: str
    subfolder: Path
    lr_path: Path
    raw_path: Path


@dataclass(frozen=True)
class ProbabilityMapParams:
    run_density_estimation: bool
    training_timepoint: str | None
    seg_tif: Path | None
    valid_mask_tif: Path | None
    densities_path: Path
    output_path: Path
    density_method: str
    feature_batch: int
    kde_points: int
    kde_max_samples: int
    kde_bandwidth: float | None
    hist_sigma_bins: float
    bg_prob_threshold: float
    fg_prob_threshold: float
    seed: int
    device_name: str | None


@dataclass(frozen=True)
class PackedDens:
    x1: Tensor
    f1: Tensor
    len1: Tensor
    x2: Tensor
    f2: Tensor
    len2: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probability-map post-processing from lr_feats.npy.")
    parser.add_argument("--input-path", required=True, help="Folder containing per-timepoint subfolders.")
    parser.add_argument("--output-path", required=True, help="Folder where segmentation masks are written.")
    parser.add_argument(
        "--run-density-estimation",
        action="store_true",
        help="Run Stage 1+2 (density estimation) before probability-map estimation.",
    )
    parser.add_argument(
        "--training-timepoint",
        default=None,
        help="Timepoint subfolder name used for density estimation.",
    )
    parser.add_argument("--seg-tif", default=None, help="Training segmentation TIFF used for density estimation.")
    parser.add_argument(
        "--valid-mask-tif",
        default=None,
        help="Optional valid-mask TIFF used during density estimation.",
    )
    parser.add_argument(
        "--densities-path",
        default=None,
        help=f"Path to the saved densities cache. Default: <input-path>/{DEFAULT_DENSITIES_FILENAME}",
    )
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if omitted).")
    parser.add_argument("--feature-batch", type=int, default=32, help="Feature batch size for Stage 3 classification.")
    parser.add_argument("--kde-points", type=int, default=512, help="Density grid size.")
    parser.add_argument("--kde-max-samples", type=int, default=200000, help="Max samples per class in Stage 1.")
    parser.add_argument("--kde-bandwidth", type=float, default=None, help="Manual KDE bandwidth; omit for auto.")
    parser.add_argument(
        "--density-method",
        choices=["kde", "gpu-hist"],
        default="gpu-hist",
        help="Stage 2 density estimator.",
    )
    parser.add_argument(
        "--hist-sigma-bins",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma in histogram bins for gpu-hist.",
    )
    parser.add_argument(
        "--bg-prob-threshold",
        type=float,
        default=0.4,
        help="Background probability threshold used in probability-map estimation.",
    )
    parser.add_argument(
        "--fg-prob-threshold",
        type=float,
        default=0.95,
        help="Foreground probability threshold used in probability-map estimation.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for Stage 1 sampling.")
    return parser.parse_args()


def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_tiff_volume(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path} does not contain a readable TIFF volume.")
        array = tif.asarray()
    if array.ndim != 3:
        raise ValueError(f"{path} must be a 3D TIFF volume.")
    return np.asarray(array)


def read_tiff_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"{path} does not contain a readable TIFF volume.")
        shape = tuple(int(dim) for dim in tif.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"{path} must be a 3D TIFF volume.")
    return shape


def load_valid_mask(valid_mask_tif: Path, *, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    with tifffile.TiffFile(valid_mask_tif) as tif:
        if not tif.series:
            raise ValueError(f"{valid_mask_tif} does not contain a readable TIFF volume.")
        valid_mask = tif.asarray()
    z_size, y_size, x_size = shape_zyx
    if valid_mask.ndim == 2:
        if valid_mask.shape != (y_size, x_size):
            raise ValueError(f"Valid mask shape {valid_mask.shape} != {(y_size, x_size)}.")
        valid_mask_2d = valid_mask > 0
        return np.broadcast_to(valid_mask_2d, shape_zyx)

    if valid_mask.ndim != 3:
        raise ValueError(f"Valid mask must be 2D or 3D, got ndim={valid_mask.ndim}.")
    if valid_mask.shape != shape_zyx:
        raise ValueError(f"Valid mask shape {valid_mask.shape} != {shape_zyx}.")
    return valid_mask > 0


def list_subfolders(input_path: Path, *, exclude_paths: Iterable[Path] = ()) -> list[Path]:
    excluded = {Path(os.path.realpath(path)) for path in exclude_paths}
    subfolders = [path for path in input_path.iterdir() if path.is_dir() and Path(os.path.realpath(path)) not in excluded]
    subfolders.sort(key=lambda path: (path.name.casefold(), path.name))
    return subfolders


def discover_timepoints(input_path: Path, *, exclude_paths: Iterable[Path] = ()) -> list[TimepointPaths]:
    discovered: list[TimepointPaths] = []
    for subfolder in list_subfolders(input_path, exclude_paths=exclude_paths):
        lr_path = subfolder / "lr_feats.npy"
        raw_path = subfolder / "volume_unnorm.tif"
        if not lr_path.is_file():
            raise FileNotFoundError(f"{subfolder.name} is missing lr_feats.npy.")
        if not raw_path.is_file():
            raise FileNotFoundError(f"{subfolder.name} is missing volume_unnorm.tif.")
        discovered.append(TimepointPaths(name=subfolder.name, subfolder=subfolder, lr_path=lr_path, raw_path=raw_path))
    if not discovered:
        raise ValueError(f"Input folder contains no subfolders: {input_path}")
    return discovered


def build_feature_names(channel_count: int) -> list[str]:
    width = max(3, len(str(max(0, channel_count - 1))))
    return [f"feature_{index:0{width}d}" for index in range(channel_count)]


def reorder_lists_to_current(
    current_names: Sequence[str],
    saved_names: Sequence[str],
    lists: Sequence[Sequence[np.ndarray]],
) -> list[list[np.ndarray]]:
    if len(current_names) != len(saved_names):
        raise RuntimeError("Feature count mismatch between cached densities and current run.")
    if len(set(current_names)) != len(current_names) or len(set(saved_names)) != len(saved_names):
        if list(current_names) != list(saved_names):
            raise RuntimeError("Cannot align cached densities to current feature order.")
        index_order = list(range(len(current_names)))
    else:
        positions = {name: index for index, name in enumerate(saved_names)}
        try:
            index_order = [positions[name] for name in current_names]
        except KeyError as exc:
            raise RuntimeError(f"Cached densities are missing feature {exc}.") from exc
    reordered: list[list[np.ndarray]] = []
    for values in lists:
        reordered.append([np.asarray(values[index], dtype=np.float32) for index in index_order])
    return reordered


def save_densities_npz(
    path: Path,
    x1_all: Sequence[np.ndarray],
    f1_all: Sequence[np.ndarray],
    x2_all: Sequence[np.ndarray],
    f2_all: Sequence[np.ndarray],
    feature_names: Sequence[str],
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        x1=np.array(x1_all, dtype=object),
        f1=np.array(f1_all, dtype=object),
        x2=np.array(x2_all, dtype=object),
        f2=np.array(f2_all, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
    )


def load_densities_npz(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    loaded = np.load(path, allow_pickle=True)
    x1_all = [np.asarray(value, dtype=np.float32) for value in loaded["x1"]]
    f1_all = [np.asarray(value, dtype=np.float32) for value in loaded["f1"]]
    x2_all = [np.asarray(value, dtype=np.float32) for value in loaded["x2"]]
    f2_all = [np.asarray(value, dtype=np.float32) for value in loaded["f2"]]
    feature_names = [str(value) for value in loaded["feature_names"]]
    return x1_all, f1_all, x2_all, f2_all, feature_names


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


class UpsampledFeatureSampler:
    def __init__(self, lr_feats: np.ndarray, target_shape: tuple[int, int, int], device: torch.device):
        if lr_feats.ndim != 4:
            raise ValueError("lr_feats.npy must be a 4D array with shape [Z, Y, X, C].")
        self.lr_feats = lr_feats
        self.target_shape = tuple(int(dim) for dim in target_shape)
        self.device = device
        self.channel_count = int(lr_feats.shape[-1])
        self._xy_grid = self._build_xy_grid()
        self._grid_cache: dict[tuple[int, ...], Tensor] = {}
        self._feature_range_cache: tuple[int, int, Tensor] | None = None

    def _build_xy_grid(self) -> Tensor:
        _, y_size, x_size = self.target_shape
        x_coords = ((torch.arange(x_size, device=self.device, dtype=torch.float32) + 0.5) * 2.0 / x_size) - 1.0
        y_coords = ((torch.arange(y_size, device=self.device, dtype=torch.float32) + 0.5) * 2.0 / y_size) - 1.0
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    def _grid_for_z_indices(self, z_indices: Sequence[int]) -> Tensor:
        key = tuple(int(index) for index in z_indices)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        z_size, y_size, x_size = self.target_shape
        xy_grid = self._xy_grid.unsqueeze(0).expand(len(key), y_size, x_size, 2)
        z_coords = ((torch.tensor(key, device=self.device, dtype=torch.float32) + 0.5) * 2.0 / z_size) - 1.0
        z_grid = z_coords.view(len(key), 1, 1, 1).expand(len(key), y_size, x_size, 1)
        grid = torch.cat((xy_grid, z_grid), dim=-1).unsqueeze(0)
        self._grid_cache[key] = grid
        return grid

    def sample_feature_range(self, start: int, end: int, z_indices: Sequence[int]) -> Tensor:
        if start < 0 or end > self.channel_count or start >= end:
            raise ValueError("Invalid feature range requested.")
        input_tensor = self._get_feature_range_tensor(start, end)
        grid = self._grid_for_z_indices(z_indices)
        sampled = F.grid_sample(
            input_tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(0)

    def clear_feature_range_cache(self) -> None:
        self._feature_range_cache = None

    def feature_range_min_max(self, start: int, end: int) -> tuple[Tensor, Tensor]:
        input_tensor = self._get_feature_range_tensor(start, end)
        mins_t = input_tensor.amin(dim=(0, 2, 3, 4)).view(-1, 1, 1, 1)
        maxs_t = input_tensor.amax(dim=(0, 2, 3, 4)).view(-1, 1, 1, 1)
        same_mask = maxs_t <= mins_t
        maxs_t = torch.where(same_mask, mins_t + 1.0, maxs_t)
        return mins_t, maxs_t

    def _get_feature_range_tensor(self, start: int, end: int) -> Tensor:
        cached = self._feature_range_cache
        if cached is not None and cached[0] == start and cached[1] == end:
            return cached[2]

        lr_chunk = np.moveaxis(np.asarray(self.lr_feats[..., start:end], dtype=np.float32), -1, 0)
        input_tensor = torch.from_numpy(ensure_contiguous(lr_chunk)).unsqueeze(0).to(self.device, dtype=torch.float32)
        self._feature_range_cache = (start, end, input_tensor)
        return input_tensor


def compute_feature_min_max_from_lr(
    lr_feats: np.ndarray,
    *,
    feature_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Using lr_feats ranges avoids a second full pass over the upsampled HR volume.
    if lr_feats.ndim != 4:
        raise ValueError("lr_feats.npy must be a 4D array with shape [Z, Y, X, C].")
    channel_count = int(lr_feats.shape[-1])
    mins = np.full(channel_count, np.inf, dtype=np.float32)
    maxs = np.full(channel_count, -np.inf, dtype=np.float32)

    for start in range(0, channel_count, feature_batch):
        end = min(channel_count, start + feature_batch)
        lr_chunk = np.asarray(lr_feats[..., start:end], dtype=np.float32)
        mins[start:end] = lr_chunk.min(axis=(0, 1, 2)).astype(np.float32, copy=False)
        maxs[start:end] = lr_chunk.max(axis=(0, 1, 2)).astype(np.float32, copy=False)

    mins = np.where(np.isfinite(mins), mins, 0.0).astype(np.float32, copy=False)
    maxs = np.where(np.isfinite(maxs), maxs, 1.0).astype(np.float32, copy=False)
    same_mask = maxs <= mins
    maxs[same_mask] = mins[same_mask] + 1.0
    return mins, maxs


def quantize_feature_batch(sampled: Tensor, mins_t: Tensor, maxs_t: Tensor) -> Tensor:
    scaled = (sampled - mins_t) / (maxs_t - mins_t + 1e-6)
    quantized = torch.clamp(scaled * 255.0, 0.0, 255.0).to(torch.uint8)
    return quantized.to(torch.float32)


def precompute_mask_indices(bg_mask: np.ndarray, fg_mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    bg_idx_per_z: list[np.ndarray] = []
    fg_idx_per_z: list[np.ndarray] = []
    for z_index in range(bg_mask.shape[0]):
        bg_idx_per_z.append(np.flatnonzero(bg_mask[z_index].ravel()))
        fg_idx_per_z.append(np.flatnonzero(fg_mask[z_index].ravel()))
    return bg_idx_per_z, fg_idx_per_z


def iter_z_index_chunks(z_size: int, z_chunk_size: int = DEFAULT_Z_CHUNK_SIZE) -> Sequence[list[int]]:
    return [list(range(z_start, min(z_size, z_start + z_chunk_size))) for z_start in range(0, z_size, z_chunk_size)]


def collect_samples_from_timepoint(
    timepoint: TimepointPaths,
    *,
    seg_tif: Path,
    valid_mask_tif: Path | None,
    max_samples_per_class: int,
    feature_batch: int,
    seed: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    lr_feats = np.load(timepoint.lr_path, mmap_mode="r")
    raw_shape = read_tiff_shape(timepoint.raw_path)

    seg = read_tiff_volume(seg_tif)
    if seg.shape != raw_shape:
        raise ValueError(f"Training segmentation shape {seg.shape} does not match raw shape {raw_shape}.")

    valid_mask = (
        load_valid_mask(valid_mask_tif, shape_zyx=raw_shape) if valid_mask_tif is not None else np.ones(raw_shape, dtype=bool)
    )
    bg_label = int(np.min(seg))
    bg_mask = (seg == bg_label) & valid_mask
    fg_mask = (seg != bg_label) & valid_mask
    bg_idx_per_z, fg_idx_per_z = precompute_mask_indices(bg_mask, fg_mask)

    sampler = UpsampledFeatureSampler(lr_feats, raw_shape, device)
    feature_names = build_feature_names(sampler.channel_count)
    bg_list: list[np.ndarray] = [np.empty((0,), dtype=np.float32) for _ in range(sampler.channel_count)]
    fg_list: list[np.ndarray] = [np.empty((0,), dtype=np.float32) for _ in range(sampler.channel_count)]
    per_z_budget = int(math.ceil(max_samples_per_class / max(1, raw_shape[0])))
    rngs = [np.random.default_rng(seed + feature_index) for feature_index in range(sampler.channel_count)]
    z_chunks = iter_z_index_chunks(raw_shape[0])

    for start in range(0, sampler.channel_count, feature_batch):
        end = min(sampler.channel_count, start + feature_batch)
        bg_parts = [[] for _ in range(end - start)]
        fg_parts = [[] for _ in range(end - start)]
        batch_mins_t, batch_maxs_t = sampler.feature_range_min_max(start, end)

        for z_indices in z_chunks:
            sampled = sampler.sample_feature_range(start, end, z_indices)
            quantized = quantize_feature_batch(sampled, batch_mins_t, batch_maxs_t)
            quantized_np = quantized.reshape(end - start, len(z_indices), -1).detach().cpu().numpy()
            for local_index, z_index in enumerate(z_indices):
                bg_indices = bg_idx_per_z[z_index]
                fg_indices = fg_idx_per_z[z_index]
                for offset, feature_index in enumerate(range(start, end)):
                    feature_rng = rngs[feature_index]
                    if bg_indices.size > 0:
                        bg_count = min(per_z_budget, int(bg_indices.size))
                        picked_bg = feature_rng.choice(bg_indices, size=bg_count, replace=False)
                        bg_parts[offset].append(quantized_np[offset, local_index, picked_bg])
                    if fg_indices.size > 0:
                        fg_count = min(per_z_budget, int(fg_indices.size))
                        picked_fg = feature_rng.choice(fg_indices, size=fg_count, replace=False)
                        fg_parts[offset].append(quantized_np[offset, local_index, picked_fg])

            del sampled
            del quantized

        for offset, feature_index in enumerate(range(start, end)):
            feature_rng = rngs[feature_index]
            bg_values = (
                np.concatenate(bg_parts[offset]).astype(np.float32, copy=False)
                if bg_parts[offset]
                else np.empty((0,), dtype=np.float32)
            )
            fg_values = (
                np.concatenate(fg_parts[offset]).astype(np.float32, copy=False)
                if fg_parts[offset]
                else np.empty((0,), dtype=np.float32)
            )
            if bg_values.size > max_samples_per_class:
                bg_values = bg_values[feature_rng.permutation(bg_values.size)[:max_samples_per_class]]
            if fg_values.size > max_samples_per_class:
                fg_values = fg_values[feature_rng.permutation(fg_values.size)[:max_samples_per_class]]
            bg_list[feature_index] = bg_values
            fg_list[feature_index] = fg_values
        sampler.clear_feature_range_cache()

    return bg_list, fg_list, feature_names


def bw_auto(values: np.ndarray) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    count = numeric.size
    if count < 2:
        return 0.2
    std = np.std(numeric)
    iqr = np.subtract(*np.percentile(numeric, [75, 25]))
    sigma = min(std, iqr / 1.349) if std > 0 else (iqr / 1.349 if iqr > 0 else 1.0)
    return float(max(1e-6, 1.06 * sigma * (count ** (-1 / 5))))


def kde_pdf(values: np.ndarray, grid: np.ndarray, bandwidth: float | None) -> np.ndarray:
    from sklearn.neighbors import KernelDensity

    numeric = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if numeric.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    resolved_bandwidth = bw_auto(numeric) if bandwidth is None else float(bandwidth)
    kde = KernelDensity(kernel="gaussian", bandwidth=resolved_bandwidth)
    kde.fit(numeric)
    log_density = kde.score_samples(grid.reshape(-1, 1))
    density = np.exp(log_density)
    return np.maximum(density, 0.0)


def fit_densities_from_samples(
    bg_list: Sequence[np.ndarray],
    fg_list: Sequence[np.ndarray],
    *,
    num_points: int,
    bandwidth: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    x1_all: list[np.ndarray] = []
    f1_all: list[np.ndarray] = []
    x2_all: list[np.ndarray] = []
    f2_all: list[np.ndarray] = []
    for bg_values, fg_values in zip(bg_list, fg_list, strict=True):
        if bg_values.size and fg_values.size:
            value_min = float(min(bg_values.min(), fg_values.min()))
            value_max = float(max(bg_values.max(), fg_values.max()))
        elif bg_values.size:
            value_min = float(bg_values.min())
            value_max = float(bg_values.max())
        elif fg_values.size:
            value_min = float(fg_values.min())
            value_max = float(fg_values.max())
        else:
            value_min, value_max = 0.0, 1.0
        if not np.isfinite(value_min) or not np.isfinite(value_max) or value_min == value_max:
            value_min, value_max = 0.0, 1.0
        grid = np.linspace(value_min, value_max, num_points, dtype=np.float64)
        f_bg = kde_pdf(bg_values, grid, bandwidth)
        f_fg = kde_pdf(fg_values, grid, bandwidth)
        x1_all.append(grid.astype(np.float32))
        f1_all.append(f_bg.astype(np.float32))
        x2_all.append(grid.astype(np.float32))
        f2_all.append(f_fg.astype(np.float32))
    return x1_all, f1_all, x2_all, f2_all


def gaussian_kernel_1d(sigma_bins: float, device: torch.device) -> torch.Tensor:
    kernel_size = int(max(3, 2 * round(3 * sigma_bins) + 1))
    xs = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (xs / float(sigma_bins)) ** 2)
    kernel = kernel / (kernel.sum() + 1e-20)
    return kernel.view(1, 1, kernel_size)


def fit_densities_gpu_hist(
    bg_list: Sequence[np.ndarray],
    fg_list: Sequence[np.ndarray],
    *,
    num_points: int,
    sigma_bins: float,
    device: torch.device,
    batch_features: int = 64,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    feature_total = len(bg_list)
    bin_count = int(num_points)
    kernel = gaussian_kernel_1d(sigma_bins=max(0.5, float(sigma_bins)), device=device)
    x1_all: list[np.ndarray] = []
    f1_all: list[np.ndarray] = []
    x2_all: list[np.ndarray] = []
    f2_all: list[np.ndarray] = []

    for start in range(0, feature_total, batch_features):
        end = min(feature_total, start + batch_features)
        for feature_index in range(start, end):
            bg_values = bg_list[feature_index]
            fg_values = fg_list[feature_index]
            if bg_values.size and fg_values.size:
                value_min = float(min(bg_values.min(), fg_values.min()))
                value_max = float(max(bg_values.max(), fg_values.max()))
            elif bg_values.size:
                value_min = float(bg_values.min())
                value_max = float(bg_values.max())
            elif fg_values.size:
                value_min = float(fg_values.min())
                value_max = float(fg_values.max())
            else:
                grid = np.linspace(0.0, 1.0, bin_count, dtype=np.float32)
                zeros = np.zeros_like(grid, dtype=np.float32)
                x1_all.append(grid)
                f1_all.append(zeros)
                x2_all.append(grid)
                f2_all.append(zeros)
                continue

            if not np.isfinite(value_min) or not np.isfinite(value_max) or value_min == value_max:
                value_min, value_max = 0.0, 1.0

            bin_width = (value_max - value_min) / bin_count
            x_grid = torch.linspace(value_min, value_max, bin_count, device=device, dtype=torch.float32)

            def histogram_1d(values: np.ndarray) -> torch.Tensor:
                if values.size == 0:
                    return torch.zeros(bin_count, device=device, dtype=torch.float32)
                tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device=device, dtype=torch.float32)
                indices = torch.clamp(
                    ((tensor - value_min) / (value_max - value_min + 1e-20) * (bin_count - 1)).long(),
                    0,
                    bin_count - 1,
                )
                return torch.bincount(indices, minlength=bin_count).to(torch.float32)

            h_bg = histogram_1d(bg_values)
            h_fg = histogram_1d(fg_values)
            pad = (kernel.shape[-1] - 1) // 2
            s_bg = F.conv1d(h_bg.view(1, 1, bin_count), kernel, padding=pad).view(bin_count)
            s_fg = F.conv1d(h_fg.view(1, 1, bin_count), kernel, padding=pad).view(bin_count)
            n_bg = max(1.0, float(bg_values.size))
            n_fg = max(1.0, float(fg_values.size))
            f_bg = torch.clamp(s_bg / (n_bg * (bin_width + 1e-20)), min=0.0)
            f_fg = torch.clamp(s_fg / (n_fg * (bin_width + 1e-20)), min=0.0)
            x_cpu = x_grid.detach().cpu().numpy().astype(np.float32, copy=False)
            fbg_cpu = f_bg.detach().cpu().numpy().astype(np.float32, copy=False)
            ffg_cpu = f_fg.detach().cpu().numpy().astype(np.float32, copy=False)
            x1_all.append(x_cpu)
            f1_all.append(fbg_cpu)
            x2_all.append(x_cpu)
            f2_all.append(ffg_cpu)
            del h_bg, h_fg, s_bg, s_fg, f_bg, f_fg, x_grid
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return x1_all, f1_all, x2_all, f2_all


def pack_curves(curves: Sequence[np.ndarray], *, pad_value: float) -> tuple[Tensor, Tensor]:
    lengths = np.array([len(curve) for curve in curves], dtype=np.int64)
    max_length = int(lengths.max()) if lengths.size else 1
    packed = np.full((len(curves), max_length), pad_value, dtype=np.float32)
    for index, curve in enumerate(curves):
        packed[index, : len(curve)] = np.asarray(curve, dtype=np.float32)
    return torch.from_numpy(packed), torch.from_numpy(lengths)


def pack_densities(
    x1_all: Sequence[np.ndarray],
    f1_all: Sequence[np.ndarray],
    x2_all: Sequence[np.ndarray],
    f2_all: Sequence[np.ndarray],
    *,
    device: torch.device,
) -> PackedDens:
    x1, len1 = pack_curves(x1_all, pad_value=np.inf)
    f1, _ = pack_curves(f1_all, pad_value=0.0)
    x2, len2 = pack_curves(x2_all, pad_value=np.inf)
    f2, _ = pack_curves(f2_all, pad_value=0.0)
    return PackedDens(
        x1=x1.to(device),
        f1=f1.to(device),
        len1=len1.to(device),
        x2=x2.to(device),
        f2=f2.to(device),
        len2=len2.to(device),
    )


@torch.no_grad()
def interp1d_per_feature(values: Tensor, xp: Tensor, fp: Tensor, lengths: Tensor) -> Tensor:
    batch_size, flat_count = values.shape
    max_length = xp.shape[1]
    indices = torch.searchsorted(xp, values, right=False).clamp_(1, max_length - 1)
    left = indices - 1
    x0 = torch.gather(xp, 1, left)
    x1 = torch.gather(xp, 1, indices)
    y0 = torch.gather(fp, 1, left)
    y1 = torch.gather(fp, 1, indices)
    denom = x1 - x0
    t = torch.where(denom != 0, (values - x0) / denom, torch.zeros_like(values))
    interpolated = y0 + (y1 - y0) * t
    xp_min = xp[:, :1]
    last_index = (lengths - 1).clamp(min=0).view(batch_size, 1)
    xp_max = torch.gather(xp, 1, last_index)
    in_range = (values >= xp_min) & (values <= xp_max)
    return torch.where(in_range, interpolated, torch.zeros_like(interpolated))


def semantic_to_instance_seg(semantic_seg: np.ndarray) -> np.ndarray:
    semantic_bool = np.asarray(semantic_seg > 0, dtype=bool)
    labeled, _ = ndimage.label(semantic_bool, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return labeled.astype(np.uint32, copy=False)


def classify_timepoint(
    timepoint: TimepointPaths,
    *,
    output_path: Path,
    densities: PackedDens,
    feature_batch: int,
    bg_prob_threshold: float,
    fg_prob_threshold: float,
    device: torch.device,
) -> Path:
    lr_feats = np.load(timepoint.lr_path, mmap_mode="r")
    raw_shape = read_tiff_shape(timepoint.raw_path)
    sampler = UpsampledFeatureSampler(lr_feats, raw_shape, device)
    if sampler.channel_count != int(densities.x1.shape[0]):
        raise ValueError(
            f"{timepoint.name} has {sampler.channel_count} feature channels, but densities expect {int(densities.x1.shape[0])}."
    )
    z_size, y_size, x_size = raw_shape
    stack_semantic = np.empty((z_size, y_size, x_size), dtype=np.uint8)

    for z_indices in iter_z_index_chunks(z_size):
        chunk_depth = len(z_indices)
        cnt_bg = torch.zeros((chunk_depth, y_size, x_size), device=device, dtype=torch.int32)
        cnt_fg = torch.zeros((chunk_depth, y_size, x_size), device=device, dtype=torch.int32)

        for start in range(0, sampler.channel_count, feature_batch):
            end = min(sampler.channel_count, start + feature_batch)
            batch_mins_t, batch_maxs_t = sampler.feature_range_min_max(start, end)
            sampled = sampler.sample_feature_range(start, end, z_indices)
            quantized = quantize_feature_batch(sampled, batch_mins_t, batch_maxs_t)
            flat_values = quantized.reshape(end - start, chunk_depth * y_size * x_size)
            x1_batch = densities.x1[start:end]
            f1_batch = densities.f1[start:end]
            len1_batch = densities.len1[start:end]
            x2_batch = densities.x2[start:end]
            f2_batch = densities.f2[start:end]
            len2_batch = densities.len2[start:end]

            p_bg = interp1d_per_feature(flat_values, x1_batch, f1_batch, len1_batch)
            p_fg = interp1d_per_feature(flat_values, x2_batch, f2_batch, len2_batch)
            denom = p_bg + p_fg
            zero_mask = denom == 0
            prob_bg = torch.where(zero_mask, 0.5, p_bg / denom).view(end - start, chunk_depth, y_size, x_size)
            prob_fg = torch.where(zero_mask, 0.5, p_fg / denom).view(end - start, chunk_depth, y_size, x_size)
            cnt_bg += (prob_bg > bg_prob_threshold).sum(dim=0)
            cnt_fg += (prob_fg > fg_prob_threshold).sum(dim=0)
            sampler.clear_feature_range_cache()

        semantic_chunk = (cnt_fg > cnt_bg).to(torch.uint8)
        stack_semantic[z_indices] = semantic_chunk.detach().cpu().numpy()

    ensure_dir(output_path)
    instance_path = output_path / f"{timepoint.name}.tif"
    instance_seg = semantic_to_instance_seg(stack_semantic)
    tifffile.imwrite(instance_path, instance_seg, bigtiff=True, metadata=None, photometric="minisblack")
    return instance_path


def load_or_compute_densities(
    timepoints: Sequence[TimepointPaths],
    params: ProbabilityMapParams,
    *,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    first_timepoint = timepoints[0]
    first_lr = np.load(first_timepoint.lr_path, mmap_mode="r")
    if first_lr.ndim != 4:
        raise ValueError(f"{first_timepoint.lr_path} must be a 4D array with shape [Z, Y, X, C].")
    current_feature_names = build_feature_names(int(first_lr.shape[-1]))

    if not params.run_density_estimation:
        if not params.densities_path.is_file():
            raise FileNotFoundError(f"Missing densities file: {params.densities_path}")
        x1_all, f1_all, x2_all, f2_all, saved_names = load_densities_npz(params.densities_path)
        reordered = reorder_lists_to_current(current_feature_names, saved_names, [x1_all, f1_all, x2_all, f2_all])
        return reordered[0], reordered[1], reordered[2], reordered[3]

    if params.training_timepoint is None:
        raise ValueError("Density estimation requires a training timepoint.")
    if params.seg_tif is None:
        raise ValueError("Density estimation requires seg_tif.")

    selected_timepoint = next((item for item in timepoints if item.name == params.training_timepoint), None)
    if selected_timepoint is None:
        raise ValueError(f"Training timepoint {params.training_timepoint!r} was not found in the input folder.")

    print("[probmap] Processing density samples (1/2)", flush=True)
    bg_list, fg_list, feature_names = collect_samples_from_timepoint(
        selected_timepoint,
        seg_tif=params.seg_tif,
        valid_mask_tif=params.valid_mask_tif,
        max_samples_per_class=params.kde_max_samples,
        feature_batch=params.feature_batch,
        seed=params.seed,
        device=device,
    )
    print("[probmap] Completed density samples", flush=True)
    print("[probmap] Processing density fitting (2/2)", flush=True)
    if params.density_method == "kde":
        x1_all, f1_all, x2_all, f2_all = fit_densities_from_samples(
            bg_list,
            fg_list,
            num_points=params.kde_points,
            bandwidth=params.kde_bandwidth,
        )
    else:
        x1_all, f1_all, x2_all, f2_all = fit_densities_gpu_hist(
            bg_list,
            fg_list,
            num_points=params.kde_points,
            sigma_bins=params.hist_sigma_bins,
            device=device,
        )
    save_densities_npz(params.densities_path, x1_all, f1_all, x2_all, f2_all, feature_names)
    print("[probmap] Completed density fitting", flush=True)
    return x1_all, f1_all, x2_all, f2_all


def run_probability_map(input_path: Path, *, params: ProbabilityMapParams) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist or is not a directory: {input_path}")
    output_path = params.output_path.expanduser().resolve()
    if output_path.exists() and not output_path.is_dir():
        raise FileNotFoundError(f"Output folder exists but is not a directory: {output_path}")
    device = resolve_device(params.device_name)
    timepoints = discover_timepoints(input_path, exclude_paths=(output_path,))
    print(f"[probmap] Using device {device}", flush=True)
    print(f"[probmap] Found {len(timepoints)} subfolders", flush=True)
    ensure_dir(output_path)

    x1_all, f1_all, x2_all, f2_all = load_or_compute_densities(timepoints, params, device=device)
    packed_densities = pack_densities(x1_all, f1_all, x2_all, f2_all, device=device)

    for index, timepoint in enumerate(timepoints, start=1):
        print(f"[probmap] Processing {timepoint.name} ({index}/{len(timepoints)})", flush=True)
        classify_timepoint(
            timepoint,
            output_path=output_path,
            densities=packed_densities,
            feature_batch=params.feature_batch,
            bg_prob_threshold=params.bg_prob_threshold,
            fg_prob_threshold=params.fg_prob_threshold,
            device=device,
        )
        print(f"[probmap] Completed {timepoint.name}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[probmap] Done", flush=True)
    return params.densities_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    densities_path = (
        Path(args.densities_path).expanduser().resolve()
        if args.densities_path
        else (input_path / DEFAULT_DENSITIES_FILENAME).resolve()
    )
    seg_tif = Path(args.seg_tif).expanduser().resolve() if args.seg_tif else None
    valid_mask_tif = Path(args.valid_mask_tif).expanduser().resolve() if args.valid_mask_tif else None
    params = ProbabilityMapParams(
        run_density_estimation=bool(args.run_density_estimation),
        training_timepoint=args.training_timepoint,
        seg_tif=seg_tif,
        valid_mask_tif=valid_mask_tif,
        densities_path=densities_path,
        output_path=output_path,
        density_method=args.density_method,
        feature_batch=max(1, int(args.feature_batch)),
        kde_points=max(2, int(args.kde_points)),
        kde_max_samples=max(1, int(args.kde_max_samples)),
        kde_bandwidth=None if args.kde_bandwidth is None else float(args.kde_bandwidth),
        hist_sigma_bins=float(args.hist_sigma_bins),
        bg_prob_threshold=float(args.bg_prob_threshold),
        fg_prob_threshold=float(args.fg_prob_threshold),
        seed=int(args.seed),
        device_name=args.device,
    )
    run_probability_map(input_path, params=params)


if __name__ == "__main__":
    main()
