import logging
from skimage import io
from functools import partial
from typing import Any, Dict, Final, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from pathlib import Path
from spatialdino.data import DTYPE_MAPPING, EMPTY_VOXEL_THRESHOLD
from spatialdino.data.transforms import (
    HistogramNormalize,
    MinMaxNormalize,
    make_inference_transform,
)
from spatialdino.data.utils import (
    median_fill,
    validate_crop_params,
)
from spatialdino.inference.output_layout import (
    inference_lr_feats_dir,
    inference_lr_feats_path,
    inference_raw_dir,
    inference_raw_path,
    timepoint_name_for_input,
)
from spatialdino.utils.misc import make_3tuple
import torch.nn.functional as F

logger = logging.getLogger("inference_data")


def get_global_hist_bounds(config: DictConfig) -> Optional[Tuple[float, float]]:
    global_hist_min = getattr(config, "global_hist_min", None)
    global_hist_max = getattr(config, "global_hist_max", None)

    if global_hist_min is None and global_hist_max is None:
        return None
    if global_hist_min is None or global_hist_max is None:
        raise ValueError(
            "Both global_hist_min and global_hist_max must be provided together."
        )

    global_hist_min = float(global_hist_min)
    global_hist_max = float(global_hist_max)
    if global_hist_max <= global_hist_min:
        raise ValueError(
            "global_hist_max must be greater than global_hist_min."
        )

    return global_hist_min, global_hist_max


def sanitize_nonfinite_volume(
    volume: np.ndarray,
    *,
    volume_path: Union[Path, str],
) -> np.ndarray:
    finite_mask = np.isfinite(volume)
    if finite_mask.all():
        return volume

    finite_values = volume[finite_mask]
    if finite_values.size == 0:
        raise ValueError(f"{volume_path} contains no finite voxels.")

    finite_min = float(finite_values.min())
    finite_max = float(finite_values.max())
    finite_median = float(np.median(finite_values))

    nan_mask = np.isnan(volume)
    posinf_mask = np.isposinf(volume)
    neginf_mask = np.isneginf(volume)

    if np.any(nan_mask):
        volume[nan_mask] = finite_median
    if np.any(posinf_mask):
        volume[posinf_mask] = finite_max
    if np.any(neginf_mask):
        volume[neginf_mask] = finite_min

    logger.warning(
        "Sanitized non-finite voxels in %s (%d NaN, %d +inf, %d -inf).",
        volume_path,
        int(nan_mask.sum()),
        int(posinf_mask.sum()),
        int(neginf_mask.sum()),
    )

    return volume


def get_inference_pad_value(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    volume_path: Union[Path, str],
) -> float:
    masked_values = image[mask]
    masked_finite_values = masked_values[np.isfinite(masked_values)]
    if masked_finite_values.size > 0:
        return float(np.median(masked_finite_values))

    finite_values = image[np.isfinite(image)]
    if finite_values.size > 0:
        logger.warning(
            "Inference mask is empty or non-finite for %s; using whole-volume finite median for padding.",
            volume_path,
        )
        return float(np.median(finite_values))

    raise ValueError(
        f"{volume_path} contains no finite voxels after preprocessing; cannot determine pad value."
    )


class InferenceDataset(Dataset):
    EMPTY_VOXEL_THRESHOLD: Final[float] = EMPTY_VOXEL_THRESHOLD
    DTYPE_MAPPING: Final[Dict[str, torch.dtype]] = DTYPE_MAPPING

    def __init__(
        self,
        config: DictConfig,
        fnames: List[Path],
    ) -> None:
        super().__init__()
        self.config = config
        self.file_path = self.config.file_path
        self.fnames = fnames
        self.save_path = Path(self.config.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.lr_feats_dir = inference_lr_feats_dir(self.save_path)
        self.raw_dir = inference_raw_dir(self.save_path)
        self.lr_feats_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.upsample_factor = make_3tuple(self.config.upsample_factor)
        self.isotropic_scale_factor = make_3tuple(self.config.isotropic_scale_factor)
        self.patch_size = make_3tuple(self.config.patch_size)
        self.stride = make_3tuple(self.config.stride)
        self.config.patch_size = self.patch_size
        self.global_hist_bounds = get_global_hist_bounds(self.config)
        assert len(self.config.crop_params) == 6, (
            "crop_params must be a tuple of 6 elements in the form (z_start, z_end, y_start, y_end, x_start, x_end)"
        )
        if self.global_hist_bounds is None:
            self.hist_normalize = HistogramNormalize(
                b_min=None,
                b_max=None,
                clip=False,
                max_val=65535,
                threshold_divisor=1.0 / 5000,
                channel_wise=False,
            )
            self.normalize_transform = MinMaxNormalize(channel_wise=False)
        else:
            global_hist_min, global_hist_max = self.global_hist_bounds
            self.hist_normalize = HistogramNormalize(
                b_min=0.0,
                b_max=1.0,
                global_min=global_hist_min,
                global_max=global_hist_max,
                clip=True,
                max_val=65535,
                threshold_divisor=1.0 / 5000,
                channel_wise=False,
            )
            self.normalize_transform = None
        self.crop_params = tuple(self.config.crop_params)

    def __len__(self) -> int:
        return len(self.fnames)

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = self.hist_normalize(volume)
        if self.normalize_transform is not None:
            volume = self.normalize_transform(volume)
        return volume

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file = self.fnames[idx]
        timepoint_name = timepoint_name_for_input(file)
        raw_path = inference_raw_path(self.save_path, timepoint_name)
        lr_path = inference_lr_feats_path(self.save_path, timepoint_name)
        raw_volume_original = io.imread(file)
        raw_volume_unnorm = raw_volume_original

        assert raw_volume_unnorm.ndim == 3, (
            f"raw_volume dim {raw_volume_unnorm.ndim} != 3, expected [Z, Y, X] dims"
        )

        if self.crop_params != (0, 0, 0, 0, 0, 0):
            # Use provided crop parameters with existing logic
            start_z, end_z, start_y, end_y, start_x, end_x = self.crop_params
            end_z = end_z if end_z > 0 else raw_volume_unnorm.shape[0]
            end_y = end_y if end_y > 0 else raw_volume_unnorm.shape[1]
            end_x = end_x if end_x > 0 else raw_volume_unnorm.shape[2]
            crop_params = (start_z, end_z, start_y, end_y, start_x, end_x)

            validate_crop_params(crop_params, raw_volume_unnorm.shape)

            # Extract the crop coordinates for slicing
            start_z, end_z, start_y, end_y, start_x, end_x = crop_params
            raw_volume_unnorm = raw_volume_unnorm[
                start_z:end_z,
                start_y:end_y,
                start_x:end_x,
            ]

        io.imsave(
            raw_path,
            raw_volume_unnorm,
            check_contrast=False,
        )

        raw_volume = raw_volume_unnorm.astype(np.float32)
        raw_volume = sanitize_nonfinite_volume(raw_volume, volume_path=file)
        raw_volume = median_fill(raw_volume)

        if self.isotropic_scale_factor != (1.0, 1.0, 1.0):
            volume = (
                F.interpolate(
                    torch.from_numpy(raw_volume).unsqueeze(0).unsqueeze(0),
                    scale_factor=self.isotropic_scale_factor,
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )
        else:
            volume = raw_volume.copy()

        volume = self._normalize_volume(volume)

        # Store original dimensions before inference padding
        original_dims = volume.shape[-3:]

        Z, Y, X = original_dims

        assert all(factor >= 1 for factor in self.upsample_factor)

        Z = int(Z * self.upsample_factor[0])
        Y = int(Y * self.upsample_factor[1])
        X = int(X * self.upsample_factor[2])

        if isinstance(self.config.chunk_size, (int, float)):
            assert 0.0 < self.config.chunk_size <= 1.0, (
                "Chunk size must be between 0 and 1"
            )
            chunk_size = (
                int(Z * self.config.chunk_size + 1),
                int(Y * self.config.chunk_size + 1),
                int(X * self.config.chunk_size + 1),
            )
        else:
            chunk_size = tuple(self.config.chunk_size)

        # ensure chunk size is divisible by patch size
        chunk_size = (
            ((chunk_size[0] + self.patch_size[0] - 1) // self.patch_size[0])  # type: ignore
            * self.patch_size[0],
            ((chunk_size[1] + self.patch_size[1] - 1) // self.patch_size[1])  # type: ignore
            * self.patch_size[1],
            ((chunk_size[2] + self.patch_size[2] - 1) // self.patch_size[2])  # type: ignore
            * self.patch_size[2],
        )

        # Ensure total volume size is divisible by chunk size
        target_shape = (
            ((Z + chunk_size[0] - 1) // chunk_size[0]) * chunk_size[0],
            ((Y + chunk_size[1] - 1) // chunk_size[1]) * chunk_size[1],
            ((X + chunk_size[2] - 1) // chunk_size[2]) * chunk_size[2],
        )

        padding = (
            max(0, target_shape[0] - Z),
            max(0, target_shape[1] - Y),
            max(0, target_shape[2] - X),
        )

        target_vol_size = (
            Z + padding[0],
            Y + padding[1],
            X + padding[2],
        )

        mask = np.any(volume > InferenceDataset.EMPTY_VOXEL_THRESHOLD, axis=(-2, -1))

        data = {
            "image": volume,
            "mask": mask,
            "vol_metadata": {
                "target_vol_size": target_vol_size,
                "padding": padding,
                "target_shape": target_shape,
                "chunk_size": chunk_size,
                "save_path": str(self.save_path),
                "timepoint_name": timepoint_name,
                "raw_path": str(raw_path),
                "lr_feats_path": str(lr_path),
            },
        }

        return data


class InferenceTransform:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.file_path = self.config.file_path
        self.save_path = Path(self.config.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.upsample_factor = make_3tuple(self.config.upsample_factor)
        self.isotropic_scale_factor = make_3tuple(self.config.isotropic_scale_factor)
        self.patch_size = make_3tuple(self.config.patch_size)
        self.stride = make_3tuple(self.config.stride)
        self.config.patch_size = self.patch_size
        self.global_hist_bounds = get_global_hist_bounds(self.config)
        # Convert to float tuples for type compatibility
        upsample_factor_float = (
            float(self.upsample_factor[0]),
            float(self.upsample_factor[1]),
            float(self.upsample_factor[2]),
        )  # type: ignore

        self.transform_fn = partial(
            make_inference_transform,
            upsample_factor=upsample_factor_float,
            # The dataset already applies anisotropy correction before normalization.
            isotropic_scale_factor=(1.0, 1.0, 1.0),
            chunk_interpolate=True,
            antialias=False,
            image_key="image",
            mask_key="mask",
            in_chans=self.config.in_chans,
            mean=self.config.mean,
            std=self.config.std,
            dtype=self.config.dtype,
            normalize=self.global_hist_bounds is None,
        )

    def __call__(
        self,
        data: Dict[str, Any],
        vol_metadata: Dict[str, Any],
        chunk_interpolate: bool = True,
        interpolate_chunk_size: Tuple[int, int, int, int] = (1, 448, 448, 448),
        device: Union[torch.device, str] = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Apply the transform to the data."""
        target_vol_size = vol_metadata["target_vol_size"]
        median = get_inference_pad_value(
            data["image"],
            data["mask"],
            volume_path=vol_metadata.get("save_path", "<unknown volume>"),
        )

        transform_fn = self.transform_fn(
            target_img_size=target_vol_size,
            chunk_interpolate=chunk_interpolate,
            device=device,
            interpolate_chunk_size=interpolate_chunk_size,
            pad_value=median,
        )
        res = transform_fn(data)
        volume = res["image"].to(  # type: ignore
            InferenceDataset.DTYPE_MAPPING[self.config.dtype]
        )  # [C, Z, Y, X]

        return {
            "volume": volume,
        }


def collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader."""

    batch_dict = {
        "images": np.stack([b["image"] for b in batch], axis=0),
        "masks": np.stack([b["mask"] for b in batch], axis=0),
        "vol_metadata": [{k: v for k, v in b["vol_metadata"].items()} for b in batch],
    }
    return batch_dict
