# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import math
import random
from typing import Any, Callable, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2

from cell_interactome.utils.utils import make_3tuple

logger = logging.getLogger("dino3d")


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Compose(v2.Compose):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__(transforms)

    def __call__(
        self, img: Any, mask: Optional[np.ndarray] = None
    ) -> Tuple[Any, Optional[np.ndarray]]:
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask

    def __getitem__(self, index: int) -> Callable:
        return self.transforms[index]


class Normalize(v2.Transform):
    CONST_AUTO_THRESHOLD: Final[int] = 5000
    MAX_VAL: Final[int] = 65536  # 2 ** 16
    THRESHOLD_DIVISOR: Final[float] = 1.0 / CONST_AUTO_THRESHOLD
    EPS: Final[float] = torch.finfo(torch.float32).eps

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _get_min_max(
        img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[int, int]:
        im = img if mask is None else img[mask]
        histogram = torch.histc(
            im.float(), bins=Normalize.MAX_VAL, min=0, max=Normalize.MAX_VAL - 1
        )
        threshold = int(im.numel() * Normalize.THRESHOLD_DIVISOR)

        cumsum = torch.cumsum(histogram, 0)
        cumsum_reverse = torch.cumsum(histogram.flip(0), 0)

        # Find first value exceeding threshold, or use min/max if none found
        min_indices = torch.nonzero(cumsum > threshold)
        max_indices = torch.nonzero(cumsum_reverse > threshold)

        hmin = min_indices[0].item() if len(min_indices) > 0 else 0
        hmax = (
            (Normalize.MAX_VAL - 1 - max_indices[0].item())
            if len(max_indices) > 0
            else Normalize.MAX_VAL - 1
        )

        return hmin, hmax  # type: ignore

    @staticmethod
    def min_max_normalize(
        img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        min_val, max_val = Normalize._get_min_max(img, mask)
        return (
            img.sub_(min_val).div_(max_val - min_val + Normalize.EPS).clamp_(0.0, 1.0)
        ), mask

    def __call__(
        self, image: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return self.min_max_normalize(image, mask)


def make_normalize_transform() -> Normalize:
    return Normalize()


class RandomResizedCrop(v2.Transform):
    """
    Randomly crop and resize 3D volumes with proper aspect ratio handling.

    Args:
        size (Union[int, Tuple[int, int, int]]): Desired output size
        scale (Tuple[float, float]): Range of size of the origin size cropped (as fraction of volume)
        ratio (Tuple[float, float]): Range of aspect ratio of the origin ratio cropped
        interpolation (str): Interpolation method (only 'trilinear' supported)
        max_attempts (int): Maximum number of attempts to get a valid crop (default: 10)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int, int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333),
    ):
        super().__init__()
        self.size = make_3tuple(size)

        if not 0 < scale[0] <= scale[1]:
            raise ValueError("Scale values must be increasing and positive")
        if not 0 < ratio[0] <= ratio[1]:
            raise ValueError("Ratio values must be increasing and positive")

        self.scale = scale
        self.ratio = ratio

        self.interpolation = "trilinear"
        self.max_attempts = 10

    def get_params(
        self,
        D: int,
        H: int,
        W: int,
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Get crop parameters maintaining proper 3D aspect ratios.

        Args:
            D: Depth dimension size
            H: Height dimension size
            W: Width dimension size

        Returns:
            Tuple of (crop_d, crop_h, crop_w, start_d, start_h, start_w)
        """
        volume = D * H * W
        target_volume = volume * random.uniform(*self.scale)

        # Calculate target dimensions based on desired output size ratios
        target_ratios = (
            self.size[1] / self.size[0],  # H/D ratio
            self.size[2] / self.size[0],  # W/D ratio
        )

        for _ in range(self.max_attempts):
            # Apply random perturbation to target ratios within allowed range
            ratio_h = target_ratios[0] * math.exp(
                random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            )
            ratio_w = target_ratios[1] * math.exp(
                random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            )

            # Calculate crop dimensions maintaining target volume
            crop_d = int(round((target_volume / (ratio_h * ratio_w)) ** (1.0 / 3)))
            crop_h = int(round(crop_d * ratio_h))
            crop_w = int(round(crop_d * ratio_w))

            if 0 < crop_d <= D and 0 < crop_h <= H and 0 < crop_w <= W:
                # Calculate random starting positions
                start_d = random.randint(0, D - crop_d) if D > crop_d else 0
                start_h = random.randint(0, H - crop_h) if H > crop_h else 0
                start_w = random.randint(0, W - crop_w) if W > crop_w else 0

                return crop_d, crop_h, crop_w, start_d, start_h, start_w

        # Fallback to central crop that preserves target aspect ratios
        scale = min(D / self.size[0], H / self.size[1], W / self.size[2])

        crop_d = min(D, int(self.size[0] * scale))
        crop_h = min(H, int(self.size[1] * scale))
        crop_w = min(W, int(self.size[2] * scale))

        start_d = (D - crop_d) >> 1
        start_h = (H - crop_h) >> 1
        start_w = (W - crop_w) >> 1

        return crop_d, crop_h, crop_w, start_d, start_h, start_w

    def __call__(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Optimized random resized crop application.
        """

        if volume.ndim != 3:
            raise ValueError(f"Expected 3D tensor [Z, Y, X], got {volume.ndim}D")

        # Minimize tensor conversions
        is_tensor = isinstance(volume, torch.Tensor)
        if is_tensor:
            volume = volume.numpy()

        D, H, W = volume.shape
        if mask is not None:
            if mask.shape != (D,):
                raise ValueError(f"mask must have shape [{D}], got {mask.shape}")
            D = int(mask.sum())

        crop_d, crop_h, crop_w, start_d, start_h, start_w = self.get_params(D, H, W)

        if mask is None:
            cropped = volume[
                start_d : start_d + crop_d,
                start_h : start_h + crop_h,
                start_w : start_w + crop_w,
            ]
        else:
            cropped = volume[
                mask,
                start_h : start_h + crop_h,
                start_w : start_w + crop_w,
            ][start_d : start_d + crop_d]
            mask = mask[start_d : start_d + crop_d]

        resized = (
            F.interpolate(
                torch.from_numpy(cropped).unsqueeze_(0).unsqueeze_(0),
                size=self.size,
                mode=self.interpolation,
                align_corners=False,
            )
            .squeeze_(0)
            .squeeze_(0)
        )

        if mask is not None:
            z = cropped.shape[-3]
            pad_z = self.size[0] - z
            if pad_z > 0:
                mask = np.pad(mask, (0, pad_z), mode="constant", constant_values=0)
            else:
                mask = mask[: self.size[0]]

        return resized, mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"scale={tuple(round(s, 4) for s in self.scale)}, "
            f"ratio={tuple(round(math.exp(r), 4) for r in self.ratio)}, "
            f"interpolation={self.interpolation}, "
            f"max_attempts={self.max_attempts})"
        )


class CropOrPad(v2.Transform):
    """Modify the volume by cropping or padding to match a target shape.

    Args:
        target_shape: Tuple (..., D, H, W). If a single value N is provided, then D = H = W = N.
        padding_mode: Value used for padding (default: 0)
    """

    def __init__(
        self,
        target_shape: Union[int, Tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.target_shape = np.array(make_3tuple(target_shape))

    @staticmethod
    @torch.jit.script
    def _apply_crop(
        volume: torch.Tensor, starts: List[int], ends: List[int]
    ) -> torch.Tensor:
        """Optimized cropping using tensor slicing"""
        return volume[
            ..., starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]
        ]

    def _compute_crop_or_pad_params(
        self,
        source_shape: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized computation of padding and cropping parameters"""
        diff_shape = self.target_shape - source_shape

        # Vectorized computation for both padding and cropping
        pad_mask = diff_shape > 0
        crop_mask = ~pad_mask

        # Calculate padding
        pad_before = np.where(pad_mask, diff_shape // 2, 0)
        pad_after = np.where(pad_mask, diff_shape - pad_before, 0)

        # Calculate cropping
        crop_size = np.abs(diff_shape) * crop_mask
        crop_before = np.where(crop_mask, crop_size // 2, 0)
        crop_after = np.where(crop_mask, crop_size - crop_before, 0)

        return pad_before, pad_after, crop_before, crop_after

    def __call__(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        mask: Optional[np.ndarray] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        # Efficient conversion and mask handling
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()

        # Get spatial dimensions (last 3 dimensions)
        spatial_shape = volume.shape[-3:]
        Z, Y, X = spatial_shape
        source_shape = np.array([Z, Y, X])

        pad_before, pad_after, crop_before, crop_after = (
            self._compute_crop_or_pad_params(source_shape)
        )

        # Apply cropping if needed (any crop values > 0)
        if np.any(crop_before > 0) or np.any(crop_after > 0):
            ends = source_shape - crop_after
            # Use ellipsis to handle arbitrary leading dimensions
            volume = volume[
                ...,
                crop_before[0] : ends[0],
                crop_before[1] : ends[1],
                crop_before[2] : ends[2],
            ]
            if mask is not None:
                mask = mask[crop_before[0] : ends[0]]

        # Apply padding if needed (any pad values > 0)
        if np.any(pad_before > 0) or np.any(pad_after > 0):
            # Create pad_width array with correct number of dimensions
            leading_dims = [(0, 0)] * (volume.ndim - 3)
            pad_width = leading_dims + [(b, a) for b, a in zip(pad_before, pad_after)]

            if mask is not None:
                # Index the Z dimension correctly when calculating pad_value
                if volume.ndim > 3:
                    # For [..., Z, Y, X] format, select the masked Z slices first
                    masked_volume = volume[..., mask, :, :]
                else:
                    # For [Z, Y, X] format
                    masked_volume = volume[mask]
                pad_value = np.median(masked_volume)
            else:
                pad_value = np.median(volume)

            volume = np.pad(
                volume, pad_width=pad_width, mode="constant", constant_values=pad_value
            )

            if mask is not None:
                mask = np.pad(
                    mask,
                    pad_width=pad_width[-3],  # Only pad the Z dimension for mask
                    mode="constant",
                    constant_values=0,
                )

        return torch.from_numpy(volume), mask


def pad_to_patch_size(
    img: torch.Tensor,
    patch_size: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    img = img.numpy()  # type: ignore
    assert img.ndim == 3, "Expected 3D array [Z, Y, X]"
    z, y, x = img.shape[-3:]
    pad_z = (patch_size[0] - z % patch_size[0]) % patch_size[0]
    pad_y = (patch_size[1] - y % patch_size[1]) % patch_size[1]
    pad_x = (patch_size[2] - x % patch_size[2]) % patch_size[2]

    padding = ((0, pad_z), (0, pad_y), (0, pad_x))

    if mask is not None:
        pad_value = np.median(img[mask])
    else:
        pad_value = np.median(img)

    img = np.pad(img, padding, mode="constant", constant_values=pad_value)  # type: ignore

    if mask is not None:
        mask = np.pad(mask, (0, pad_z), mode="constant", constant_values=0)

    return torch.from_numpy(img), mask


def center_crop_to_patch_size(
    img: torch.Tensor,
    patch_size: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    assert img.ndim == 3, "Expected 3D array [Z, Y, X]"
    z, y, x = img.shape[-3:]
    # Calculate new dimensions that are divisible by patch_size
    new_z = (z // patch_size[0]) * patch_size[0]
    new_y = (y // patch_size[1]) * patch_size[1]
    new_x = (x // patch_size[2]) * patch_size[2]

    # Calculate start indices for center crop
    start_z = (z - new_z) // 2
    start_y = (y - new_y) // 2
    start_x = (x - new_x) // 2

    img = img[
        start_z : start_z + new_z,
        start_y : start_y + new_y,
        start_x : start_x + new_x,
    ]

    if mask is not None:
        mask = mask[start_z : start_z + new_z]

    return img, mask


class CropOrPadToPatchSize(v2.Transform):
    """
    Pad the image to the patch size.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]],
        crop: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = make_3tuple(patch_size)
        self.crop = crop

    def __call__(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        if self.crop:
            return center_crop_to_patch_size(img, patch_size=self.patch_size, mask=mask)
        else:
            return pad_to_patch_size(
                img,
                patch_size=self.patch_size,
                mask=mask,
            )


class GaussianBlur3D(nn.Module):
    """
    Apply Gaussian Blur to 3D images of shape [Z, Y, X]
    Optimized for CPU usage with separable convolutions.

    Args:
        kernel_size: Size of the gaussian kernel, either single int or sequence of 3 ints
        sigma: Either a single float for fixed sigma, or a sequence of (min, max) for random sigma
    """

    def __init__(
        self,
        kernel_size: Union[int, Sequence[int]],
        sigma: Union[float, Sequence[float]],
    ) -> None:
        super().__init__()
        dim = 3
        self.kernel_size = make_3tuple(kernel_size)  # type: ignore

        # Convert all sigma inputs to per-dimension ranges
        if isinstance(sigma, (int, float)):
            self.sigma_ranges = [(float(sigma), float(sigma))] * dim
            self.random_sigma = False
        elif len(sigma) == 2:
            self.sigma_ranges = [(float(sigma[0]), float(sigma[1]))] * dim
            self.random_sigma = True
        else:
            raise ValueError("sigma must be a float or a (min, max) tuple")

        if len(self.kernel_size) != dim:
            raise ValueError(f"kernel_size must have length {dim}")

        self.padding = tuple(k // 2 for k in self.kernel_size)

        # Validate kernel sizes
        for size in self.kernel_size:
            if size % 2 == 0:
                raise ValueError("Kernel size must be odd")

        # Precompute fixed kernels if using fixed sigma
        if not self.random_sigma:
            self.register_buffer(
                "fixed_kernels",
                self._create_separable_kernels([s[0] for s in self.sigma_ranges]),
            )
        else:
            self.fixed_kernels = None

    def _create_1d_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create a 1D Gaussian kernel efficiently."""
        x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel = (-0.5 * (x / sigma).pow(2)).exp()
        return kernel / kernel.sum()

    def _create_separable_kernels(self, sigmas: Sequence[float]) -> torch.Tensor:
        """Create separable 1D kernels for each dimension."""
        return torch.stack([
            self._create_1d_kernel(size, sigma)
            for size, sigma in zip(self.kernel_size, sigmas)
        ])

    def _apply_separable_conv(
        self, x: torch.Tensor, kernels: torch.Tensor
    ) -> torch.Tensor:
        """Apply separable convolution using the given kernels."""
        # Apply convolutions separately in each dimension
        for i, (kernel, padding) in enumerate(zip(kernels, self.padding)):
            # Reshape kernel for the current dimension
            kernel_size = [1] * 5
            kernel_size[i + 2] = len(kernel)  # offset by 2 for batch and channel dims
            weight = kernel.view(kernel_size)

            # Apply convolution with appropriate padding
            x = F.conv3d(
                x,
                weight=weight.to(x.dtype),
                padding=(
                    padding if i == 0 else 0,  # depth
                    padding if i == 1 else 0,  # height
                    padding if i == 2 else 0,
                ),  # width
            )

        return x

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Z, Y, X]
        Returns:
            Blurred tensor of the same shape
        """
        # Add batch and channel dims for conv3d
        needs_squeeze = x.dim() == 3
        if needs_squeeze:
            x = x.unsqueeze_(0).unsqueeze_(0)  # [1, 1, Z, Y, X]

        # Get sigma values and create kernels
        if self.random_sigma:
            sigmas = [
                random.uniform(min_s, max_s) for min_s, max_s in self.sigma_ranges
            ]
            kernels = self._create_separable_kernels(sigmas)
        else:
            kernels = self.fixed_kernels

        # Apply separable convolution
        out = self._apply_separable_conv(x, kernels)  # type: ignore

        # Remove extra dimensions if they were added
        if needs_squeeze:
            out = out.squeeze_(0).squeeze_(0)  # Back to [Z, Y, X]

        return out


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ) -> None:
        transform = GaussianBlur3D(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)

    def __call__(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return super().__call__(img), mask


class GaussianNoise(v2.RandomApply):
    """Apply random Gaussian noise to a 3D volume.

    Args:
        std_range (tuple[float, float]): Range for noise standard deviation as a fraction of the image's dynamic range.
            Default: (0.01, 0.05) means noise std will be between 1% and 5% of the image's range.
        p (float): Probability of applying the transform. Default: 0.5
    """

    def __init__(
        self, std_range: Tuple[float, float] = (0.01, 0.05), p: float = 0.5
    ) -> None:
        self.std_min, self.std_max = std_range
        super().__init__(transforms=[self._add_noise], p=p)

    def _add_noise(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        # Calculate dynamic range of the image
        img_range = img.max() - img.min()

        # Generate random std within specified range
        noise_std = random.uniform(self.std_min, self.std_max) * img_range

        # Generate noise with the same shape as the input
        noise = torch.randn_like(img) * noise_std

        # Add noise and clamp to maintain original range
        noisy_img = (img + noise).clamp(img.min(), img.max())

        return noisy_img, mask


class RandomBrightness(v2.RandomApply):
    """Apply random brightness adjustment to a 3D volume.

    Args:
        brightness_factor (tuple[float, float]): Range for brightness adjustment.
            Values > 1 increase brightness, values < 1 decrease brightness.
        p (float): Probability of applying the transform. Default: 0.5
    """

    def __init__(
        self, brightness_factor: Tuple[float, float] = (0.6, 1.4), p: float = 0.5
    ) -> None:
        self.brightness_min, self.brightness_max = brightness_factor
        super().__init__(transforms=[self._adjust_brightness], p=p)

    def _adjust_brightness(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        factor = random.uniform(self.brightness_min, self.brightness_max)
        return img.mul(factor), mask


class RandomContrast(v2.RandomApply):
    """Apply random contrast adjustment to a 3D volume using gamma correction.

    Args:
        gamma_range (tuple[float, float]): Range for gamma values.
            gamma > 1 decreases contrast (darkens image)
            gamma < 1 increases contrast (brightens image)
        p (float): Probability of applying the transform. Default: 0.5
    """

    def __init__(
        self, gamma_range: Tuple[float, float] = (0.7, 1.3), p: float = 0.5
    ) -> None:
        self.gamma_min, self.gamma_max = gamma_range
        super().__init__(transforms=[self._adjust_contrast], p=p)

    def _adjust_contrast(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        # Normalize to [0,1] range for gamma correction
        img_min, img_max = img.min(), img.max()
        normalized = (img - img_min) / (img_max - img_min + torch.finfo(img.dtype).eps)
        # Apply gamma correction
        adjusted = normalized.pow(gamma)
        # Scale back to original range
        result = adjusted * (img_max - img_min) + img_min
        return result, mask


class ToDtype(v2.Transform):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def __call__(
        self, img: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return torch.from_numpy(img).to(self.dtype), mask


class ToTensor(v2.Transform):
    """
    Convert a ``numpy.ndarray`` to tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.transform = v2.Compose([ToDtype(torch.float32)])

    def __call__(
        self, pic: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return self.transform(pic, mask)


class AddChannels(v2.Transform):
    """Add channels to the image"""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def __call__(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return img.unsqueeze_(0).repeat(self.num_channels, 1, 1, 1), mask


class MaybeToTensor(ToTensor):
    """
    Convert a ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(
        self, pic: Union[np.ndarray, torch.Tensor], mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Args:
            pic (numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic, mask
        return super().__call__(pic, mask)


class RandomRotation(v2.RandomApply):
    def __init__(
        self, axes: Union[Tuple[int, ...], None], degrees: float = 10.0, p: float = 0.5
    ) -> None:
        self.degrees = degrees
        super().__init__(transforms=[RandomRotation3D(axes, degrees)], p=p)

    def __call__(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return super().__call__(img, mask)


class RandomRotation3D:
    axes: Final[List[Tuple[int, ...]]] = [
        (0, 1),  # XY plane rotation
        (0, 2),  # XZ plane rotation
        (1, 2),  # YZ plane rotation
    ]

    def __init__(
        self,
        axes: Union[Tuple[int, ...], None],
        degrees: float = 90.0,
    ) -> None:
        if axes is None:
            axes = random.choice(RandomRotation3D.axes)
        self.axes = axes  # type: ignore
        self.degrees = degrees

    def __call__(
        self,
        img: torch.Tensor,
        mask: Optional[np.ndarray] = None,
        num_rotations: int = 1,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return torch.rot90(input=img, k=num_rotations, dims=self.axes), mask  # type: ignore


class RandomFlip3D:
    COMBINATIONS: Final[List[Tuple[int, ...]]] = [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]

    def __init__(self, axes: Union[int, Tuple[int, ...], None]) -> None:
        if isinstance(axes, int):
            axes = (axes,)
        elif axes is None:
            axes = random.choice(RandomFlip3D.COMBINATIONS)

        self.axes = axes

    def __call__(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return torch.flip(img, self.axes), mask


class RandomFlip(v2.RandomApply):
    def __init__(self, axes: Union[int, Tuple[int, ...], None], p: float = 0.5) -> None:
        super().__init__(transforms=[RandomFlip3D(axes)], p=p)

    def __call__(
        self, img: torch.Tensor, mask: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        return super().__call__(img, mask)


def make_inference_transform(
    *,
    in_chans: int = 1,
) -> Compose:
    transforms_list = [
        MaybeToTensor(),
        make_normalize_transform(),
        AddChannels(num_channels=in_chans),
    ]
    return Compose(transforms_list)


def make_inference_transform_for_lift(
    *,
    target_size: Tuple[int, int, int],
    in_chans: int = 1,
) -> Compose:
    transforms_list = [
        MaybeToTensor(),
        make_normalize_transform(),
    ]
    return Compose(transforms_list)
