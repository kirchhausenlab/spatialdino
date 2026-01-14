import math
import random
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from numba import njit
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import median_filter
import torch
import torch.nn.functional as F
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms import (
    Compose,
    CopyItemsd,
    DeleteItemsd,
    MapTransform,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd,
    SpatialPadd,
    ToTensord,
    Transform,
)
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from monai.utils.misc import ensure_tuple
from spatialdino.data import DTYPE_MAPPING
from spatialdino.utils.misc import make_3tuple
import logging
# warnings.filterwarnings(
#     "ignore",
#     message="Divide by zero \(a_min == a_max\)",
#     category=Warning,
#     module="monai.transforms.intensity.array",
# )

CELL3D_DEFAULT_MEAN = 0.163
CELL3D_DEFAULT_STD = 0.0988
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
ORIGINAL_VOXEL_SIZE = (0.25, 0.104, 0.104)  # [Z, Y, X]

logging.getLogger("numba").setLevel(logging.WARNING)


def unnormalize(
    img: NdarrayOrTensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    img = convert_to_tensor(img)
    assert img.dim() == 5, (
        f"Expected 5D tensor of shape [B, C, Z, Y, X], got {img.size()}"
    )

    dims = (1, -1, 1, 1, 1)

    return img.mul(std.view(*dims)).add(mean.view(*dims))


class HistogramNormalize(Transform):
    def __init__(
        self,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        global_min: Optional[float] = None,
        global_max: Optional[float] = None,
        clip: bool = False,
        max_val: int = 65535,
        threshold_divisor: float = 1.0 / 5000,
        channel_wise: bool = True,
    ):
        super().__init__()
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.max_val = max_val
        self.global_min = global_min
        self.global_max = global_max
        self.threshold_divisor = threshold_divisor
        self.channel_wise = channel_wise

    def get_min_max(
        self, img: NdarrayOrTensor
    ) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
        is_numpy = False
        if isinstance(img, np.ndarray):
            is_numpy = True
            img = convert_to_tensor(img)

        if self.channel_wise and img.dim() > 3:  # Has channel dimension
            C = img.shape[0]
            # Initialize tensors with proper shape [C, 1, 1, 1]
            channel_mins = torch.zeros(C, 1, 1, 1)
            channel_maxs = torch.zeros(C, 1, 1, 1)

            # Process each channel separately
            for c in range(C):
                channel_img = img[c]
                bins = self.max_val + 1
                histogram = torch.histc(
                    channel_img.float(), bins=bins, min=0, max=self.max_val
                )
                threshold = int(channel_img.numel() * self.threshold_divisor)

                cumsum = torch.cumsum(histogram, 0)
                cumsum_reverse = torch.cumsum(histogram.flip(0), 0)

                min_indices = torch.nonzero(cumsum > threshold)
                max_indices = torch.nonzero(cumsum_reverse > threshold)

                channel_mins[c] = min_indices[0].item() if len(min_indices) > 0 else 0
                channel_maxs[c] = (
                    bins - 1 - max_indices[0].item()
                    if len(max_indices) > 0
                    else bins - 1
                )

            return channel_mins, channel_maxs
        else:
            # Original behavior for non-channel-wise operation
            bins = self.max_val + 1
            histogram = torch.histc(img.float(), bins=bins, min=0, max=self.max_val)  # type: ignore
            threshold = int(img.numel() * self.threshold_divisor)  # type: ignore

            cumsum = torch.cumsum(histogram, 0)
            cumsum_reverse = torch.cumsum(histogram.flip(0), 0)

            min_indices = torch.nonzero(cumsum > threshold)
            max_indices = torch.nonzero(cumsum_reverse > threshold)

            hmin = min_indices[0].item() if len(min_indices) > 0 else 0
            hmax = (
                bins - 1 - max_indices[0].item() if len(max_indices) > 0 else bins - 1
            )

            channel_mins = torch.tensor(hmin).view(1, 1, 1)
            channel_maxs = torch.tensor(hmax).view(1, 1, 1)

        if is_numpy:
            channel_mins = convert_to_numpy(channel_mins)
            channel_maxs = convert_to_numpy(channel_maxs)

        return channel_mins, channel_maxs

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        channel_mins, channel_maxs = self.get_min_max(img)
        if self.global_min is not None and self.global_max is not None:
            channel_mins = self.global_min
            channel_maxs = self.global_max
        is_numpy = False
        if isinstance(img, np.ndarray):
            img = convert_to_tensor(img)
            is_numpy = True

        img = (img - channel_mins) / (channel_maxs - channel_mins + 1e-6)
        if (self.b_min is not None) and (self.b_max is not None):
            img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = torch.clip(img, self.b_min, self.b_max)  # type: ignore
        if is_numpy:
            img = convert_to_numpy(img)

        return img


class HistogramNormalized(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        global_min_val: Optional[float] = None,
        global_max_val: Optional[float] = None,
        clip: bool = False,
        max_val: int = 65535,
        threshold_divisor: float = 1.0 / 5000,
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalize = HistogramNormalize(
            b_min=b_min,
            b_max=b_max,
            clip=clip,
            global_max=global_max_val,
            global_min=global_min_val,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=channel_wise,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalize(d[key])
        return d


class MinMaxNormalize(Transform):
    def __init__(
        self,
        channel_wise: bool = True,
    ):
        super().__init__()
        self.channel_wise = channel_wise

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.channel_wise:
            if isinstance(img, np.ndarray):
                axis = tuple(range(img.ndim))[1:]
                min_val, max_val = (
                    img.min(axis=axis, keepdims=True),
                    img.max(axis=axis, keepdims=True),
                )
            else:
                C = img.size(0)
                flat = img.reshape(C, -1)
                # compute min and max along the flattened spatial dims
                min_vals, _ = flat.min(dim=1, keepdim=True)  # shape [C, 1]
                max_vals, _ = flat.max(dim=1, keepdim=True)  # shape [C, 1]
                min_val = min_vals.reshape(C, 1, 1, 1)
                max_val = max_vals.reshape(C, 1, 1, 1)
        else:
            if isinstance(img, np.ndarray):
                min_val, max_val = (
                    img.min(),
                    img.max(),
                )
            else:
                min_val = img.min()
                max_val = img.max()

        img = (img - min_val) / (max_val - min_val + 1e-6)

        return img


class MinMaxNormalizeGlobal(Transform):
    def __init__(
        self,
        channel_wise: bool = True,
        global_min: Optional[float] = None,
        global_max: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.channel_wise = channel_wise
        self.global_min = global_min
        self.global_max = global_max

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.channel_wise:
            if isinstance(img, np.ndarray):
                axis = tuple(range(img.ndim))[1:]
                min_val, max_val = (
                    img.min(axis=axis, keepdims=True),
                    img.max(axis=axis, keepdims=True),
                )
            else:
                C = img.size(0)
                dim = 1
                min_val = (
                    img.view(C, -1).min(dim=dim, keepdim=True).values.view(C, 1, 1, 1)
                )
                max_val = (
                    img.view(C, -1).max(dim=dim, keepdim=True).values.view(C, 1, 1, 1)
                )
        else:
            if isinstance(img, np.ndarray):
                min_val, max_val = (
                    img.min(),
                    img.max(),
                )
            else:
                min_val = img.min()
                max_val = img.max()
        if self.global_min is not None and self.global_max is not None:
            img = (img - self.global_min) / (self.global_max - self.global_min + 1e-6)
        else:
            img = (img - min_val) / (max_val - min_val + 1e-6)

        return img


class MinMaxNormalizedGlobal(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """Min-max normalization of the image. Assumes if image has channels, that it is the first dimension.

        Parameters
        ----------
        keys : KeysCollection
            Keys of the corresponding items to be transformed.
        channel_wise : bool, optional
            If True, normalize each channel separately.
        global_min : float, optional
            Global minimum value for normalization.
        global_max : float, optional
            Global maximum value for normalization.
        allow_missing_keys : bool, optional
            If True, allow missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.channel_wise = channel_wise
        self.normalize = MinMaxNormalizeGlobal(
            channel_wise=channel_wise,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalize(d[key])
        return d


class MinMaxNormalized(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """Min-max normalization of the image. Assumes if image has channels, that it is the first dimension.

        Parameters
        ----------
        keys : KeysCollection
            Keys of the corresponding items to be transformed.
        channel_wise : bool, optional
            If True, normalize each channel separately.
        allow_missing_keys : bool, optional
            If True, allow missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.channel_wise = channel_wise
        self.normalize = MinMaxNormalize(channel_wise=channel_wise)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalize(d[key])
        return d


def remap_image(
    image: NdarrayOrTensor,
    new_min: float = 0,
    new_max: float = 100,
) -> NdarrayOrTensor:
    """Remap the image to a new range.
    From: https://adaptivemotorcontrollab.github.io/CellSeg3D/source/guides/training_wnet.html
    """
    image = image * (new_max - new_min) + new_min
    return image


class Maskd(MapTransform):
    """Given a mask, apply the mask to the image."""

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        mask = d[self.mask_key]
        for key in self.key_iterator(d):
            d[key] = d[key][mask]
        return d


def _gaussian_kernel1d(
    kernel_size: int, sigma: float, device=None, dtype=None
) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        kernel_size (int): The size of the kernel. Must be odd and positive.
        sigma (float): The standard deviation of the Gaussian. Must be positive.
        device: The device to create the tensor on.
        dtype: The data type of the tensor.

    Returns:
        torch.Tensor: A 1D Gaussian kernel.
    """
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, but got {kernel_size}"
        )
    if not isinstance(sigma, (float, int)) or sigma <= 0:
        raise ValueError(f"sigma must be a positive number, but got {sigma}")

    # Create a range of coordinates for the kernel
    k = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    # Calculate Gaussian values
    k = torch.exp(-(k**2) / (2 * sigma**2))
    # Normalize the kernel so that its elements sum to 1
    return k / k.sum()


def _gaussian_kernel3d(
    kernel_size: int, sigma: float, channels: int, device=None, dtype=None
) -> torch.Tensor:
    """
    Creates a 3D Gaussian kernel suitable for depthwise convolution in conv3d.

    The kernel is designed to be applied independently to each channel.

    Args:
        kernel_size (int): The size of the kernel for each dimension (depth, height, width). Must be odd.
        sigma (float): The standard deviation of the Gaussian for each dimension.
        channels (int): The number of input channels. The kernel will be shaped for grouped convolution
                        (groups=channels) so each channel is convolved with its own filter.
        device: The device to create the tensor on.
        dtype: The data type of the tensor.

    Returns:
        torch.Tensor: A 3D Gaussian kernel of shape (channels, 1, kernel_size, kernel_size, kernel_size).
    """
    # Create a 1D Gaussian kernel
    kernel_1d = _gaussian_kernel1d(kernel_size, sigma, device=device, dtype=dtype)

    # Create a 3D kernel by taking the outer product of three 1D kernels
    # This results in a kernel where the Gaussian is applied independently along each axis
    kernel_3d = torch.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)

    # Reshape for conv3d: (out_channels, in_channels/groups, D_k, H_k, W_k)
    # For depthwise separable convolution (groups = channels), out_channels = channels, and in_channels/groups = 1.
    kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(
        0
    )  # Shape: (1, 1, kernel_size, kernel_size, kernel_size)

    # Repeat the kernel for each channel to apply it depthwise
    kernel_3d = kernel_3d.repeat(
        channels, 1, 1, 1, 1
    )  # Shape: (channels, 1, kernel_size, kernel_size, kernel_size)
    return kernel_3d


def trilinear_interpolate_with_antialias(
    input_tensor: torch.Tensor,
    size: Optional[Tuple[int, int, int]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float, float]]] = None,
    antialias: bool = True,
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    gaussian_kernel_size: int = 3,
    gaussian_sigma: float = 0.5,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Performs trilinear interpolation on a 5D input tensor (N, C, D, H, W)
    with an optional antialiasing step using Gaussian blur for downsampling.

    Antialiasing is applied via a Gaussian blur if `antialias` is True and the operation
    involves downsampling in any of the spatial dimensions (D, H, W). This is a
    manual implementation as torch.nn.functional.interpolate's `antialias`
    parameter does not directly support 'trilinear' mode.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, C, D, H, W).
        size (tuple[int, int, int] | None): Desired output spatial size (D_out, H_out, W_out).
                                            Exactly one of `size` or `scale_factor` must be specified.
        scale_factor (float | tuple[float, float, float] | None):
                                            Multiplier for spatial size. Can be a single float for all
                                            spatial dimensions or a tuple (scale_D, scale_H, scale_W).
                                            Exactly one of `size` or `scale_factor` must be specified.
        antialias (bool): If True, applies Gaussian blur before downsampling to reduce aliasing.
                          Defaults to True.
        align_corners (bool | None): Passed to `torch.nn.functional.interpolate`.
                                     See PyTorch documentation for details. Defaults to None.
        recompute_scale_factor (bool | None): Passed to `torch.nn.functional.interpolate`.
                                              See PyTorch documentation. Defaults to None.
        gaussian_kernel_size (int): Kernel size for the Gaussian blur. Must be a positive odd integer.
                                    Defaults to 3.
        gaussian_sigma (float): Standard deviation for the Gaussian blur. Must be positive.
                                Defaults to 0.5. A larger sigma or kernel_size might be needed
                                for stronger downsampling ratios.

    Returns:
        torch.Tensor: The interpolated tensor.
    """
    # --- Input Validations ---
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(
            f"Expected input_tensor to be a torch.Tensor, but got {type(input_tensor)}."
        )
    if not input_tensor.dim() == 5:
        raise ValueError(
            f"Expected 5D input tensor (N, C, D, H, W), but got {input_tensor.dim()}D."
        )

    mode = kwargs.pop("mode", "trilinear")

    assert mode == "trilinear", f"Only trilinear mode is supported, got {mode}"

    if not ((size is None) ^ (scale_factor is None)):  # XOR condition
        raise ValueError("Exactly one of 'size' or 'scale_factor' must be specified.")

    if not (
        isinstance(gaussian_kernel_size, int)
        and gaussian_kernel_size > 0
        and gaussian_kernel_size % 2 != 0
    ):
        raise ValueError(
            f"gaussian_kernel_size must be a positive odd integer, but got {gaussian_kernel_size}."
        )
    if not (isinstance(gaussian_sigma, (float, int)) and gaussian_sigma > 0):
        raise ValueError(
            f"gaussian_sigma must be a positive number, but got {gaussian_sigma}."
        )

    N, C, D_in, H_in, W_in = input_tensor.shape

    # --- Determine Target Output Size and Check for Downsampling ---
    is_downsampling = False
    if size is not None:
        if not (
            isinstance(size, tuple)
            and len(size) == 3
            and all(isinstance(s, int) and s > 0 for s in size)
        ):
            raise ValueError(
                "size must be a tuple of 3 positive integers (D_out, H_out, W_out)."
            )
        D_out, H_out, W_out = size
        if D_out < D_in or H_out < H_in or W_out < W_in:
            is_downsampling = True
    else:  # scale_factor is not None
        if isinstance(scale_factor, (float, int)):
            if scale_factor <= 0:
                raise ValueError(
                    "If scale_factor is a single number, it must be positive."
                )
            sf_d = sf_h = sf_w = float(scale_factor)
        elif isinstance(scale_factor, tuple) and len(scale_factor) == 3:
            if not all(isinstance(sf, (float, int)) and sf > 0 for sf in scale_factor):
                raise ValueError(
                    "Each element in scale_factor tuple must be a positive number."
                )
            sf_d, sf_h, sf_w = (float(sf) for sf in scale_factor)
        else:
            raise ValueError(
                "scale_factor must be a positive float or a tuple of 3 positive floats."
            )

        # Calculate output dimensions as F.interpolate would.
        # Note: F.interpolate's actual output size calculation can depend on recompute_scale_factor.
        # For simplicity in determining 'is_downsampling', we use direct multiplication.
        # A more precise check would involve how F.interpolate computes output size.
        D_out_est = math.floor(D_in * sf_d)
        H_out_est = math.floor(H_in * sf_h)
        W_out_est = math.floor(W_in * sf_w)

        if D_out_est <= 0 or H_out_est <= 0 or W_out_est <= 0:
            raise ValueError(
                f"Computed estimated output size ({D_out_est}, {H_out_est}, {W_out_est}) is not valid. "
                "Ensure scale_factor results in positive dimensions."
            )
        if D_out_est < D_in or H_out_est < H_in or W_out_est < W_in:
            is_downsampling = True

    # --- Apply Antialiasing (Gaussian Blur) if Necessary ---
    processed_tensor = input_tensor
    if antialias and is_downsampling:
        # Warn if input dimensions are too small for the kernel, as padding might not be enough
        # or the blur might not be meaningful.
        if (
            D_in < gaussian_kernel_size
            or H_in < gaussian_kernel_size
            or W_in < gaussian_kernel_size
        ):
            import warnings

            warnings.warn(
                f"Input spatial dimension ({D_in}, {H_in}, {W_in}) is smaller than "
                f"gaussian_kernel_size ({gaussian_kernel_size}) in at least one dimension. "
                "Gaussian blurring might behave unexpectedly or lead to errors if padding cannot "
                "compensate. Consider using a smaller kernel or disabling antialias if input is very small.",
                UserWarning,
            )

        # Create the 3D Gaussian kernel
        kernel = _gaussian_kernel3d(
            kernel_size=gaussian_kernel_size,
            sigma=gaussian_sigma,
            channels=C,  # Number of input channels
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

        # Calculate padding to maintain spatial dimensions ('same' padding)
        # For a kernel of size K, padding is (K-1)/2 on each side.
        padding_d = (gaussian_kernel_size - 1) // 2
        padding_h = (gaussian_kernel_size - 1) // 2
        padding_w = (gaussian_kernel_size - 1) // 2

        # Apply 3D convolution with the Gaussian kernel.
        # 'groups=C' makes it a depthwise convolution, applying each filter
        # of the kernel to its corresponding input channel.
        processed_tensor = F.conv3d(
            input_tensor, kernel, padding=(padding_d, padding_h, padding_w), groups=C
        )

    # --- Perform Trilinear Interpolation ---
    # Prepare arguments for F.interpolate
    interpolate_args = {"mode": mode}
    if size is not None:
        interpolate_args["size"] = size
    if scale_factor is not None:
        interpolate_args["scale_factor"] = (
            scale_factor  # Pass the original scale_factor
        )
    if align_corners is not None:
        interpolate_args["align_corners"] = align_corners
    if recompute_scale_factor is not None:
        interpolate_args["recompute_scale_factor"] = recompute_scale_factor
    # Note: The 'antialias' argument of F.interpolate is NOT used here because
    # it's not supported for 'trilinear' mode. Our manual blur handles it.

    interpolated_tensor = F.interpolate(processed_tensor, **interpolate_args)

    return interpolated_tensor


def chunked_trilinear_interpolate(
    inp: torch.Tensor,
    device: Union[torch.device, str],
    chunk_size: Tuple[int, int, int, int] = (64, 224, 448, 448),
    progress: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Chunk [B,C,Z,Y,X] in C, Z, Y, X to avoid huge allocations.

    Args:
      inp         (torch.Tensor): input on CPU, shape [B,C,Z,Y,X]
      device           (str or torch.device): where to do the interpolation
      chunk_size      (tuple):        (chunkC, chunkZ, chunkY, chunkX)
      kwargs           (dict):        kwargs for F.interpolate

    Returns:
      out (torch.Tensor): upsampled volume [B,C,newZ,newY,newX] on CPU
    """
    B, C, Z, Y, X = inp.shape
    mode = kwargs.pop("mode", "trilinear")
    assert mode == "trilinear", f"Only trilinear mode is supported, got {mode}"
    if "size" in kwargs:
        newZ, newY, newX = kwargs["size"]
        kwargs.pop("size")
    elif "scale_factor" in kwargs:
        newZ = int(Z * kwargs["scale_factor"][0])
        newY = int(Y * kwargs["scale_factor"][1])
        newX = int(X * kwargs["scale_factor"][2])
        kwargs.pop("scale_factor")
    else:
        raise ValueError("Either size or scale_factor must be provided in kwargs.")

    chunkC, chunkZ, chunkY, chunkX = chunk_size

    # allocate output
    out = torch.zeros(B, C, newZ, newY, newX, dtype=inp.dtype)

    num_chunks = (
        math.ceil(C / chunkC)
        * math.ceil(Z / chunkZ)
        * math.ceil(Y / chunkY)
        * math.ceil(X / chunkX)
    )

    pbar = (
        tqdm(total=num_chunks, desc="Interpolating chunks", unit="chunk")
        if progress
        else None
    )

    # loop over chunks
    for c0 in range(0, C, chunkC):
        c1 = min(C, c0 + chunkC)
        for z0 in range(0, Z, chunkZ):
            z1 = min(Z, z0 + chunkZ)
            # map original z‐range to upsampled z‐range (use ceil on the upper bound!)
            hz0 = math.floor(z0 * newZ / Z)
            hz1 = min(newZ, math.ceil(z1 * newZ / Z))

            for y0 in range(0, Y, chunkY):
                y1 = min(Y, y0 + chunkY)
                hy0 = math.floor(y0 * newY / Y)
                hy1 = min(newY, math.ceil(y1 * newY / Y))

                for x0 in range(0, X, chunkX):
                    x1 = min(X, x0 + chunkX)
                    hx0 = math.floor(x0 * newX / X)
                    hx1 = min(newX, math.ceil(x1 * newX / X))

                    # compute target size
                    out_size = (hz1 - hz0, hy1 - hy0, hx1 - hx0)
                    # skip any truly zero‐sized blocks
                    if out_size[0] <= 0 or out_size[1] <= 0 or out_size[2] <= 0:
                        continue

                    # slice out the small block
                    block = inp[
                        :,  # all batches
                        c0:c1,  # channel slice
                        z0:z1,
                        y0:y1,
                        x0:x1,  # spatial slice
                    ].to(device, non_blocking=True)

                    # compute the target size for this block
                    # interpolate that block
                    # block = F.interpolate(
                    #     block,
                    #     size=out_size,
                    #     mode="trilinear",
                    #     **kwargs,
                    # ).cpu()

                    block = trilinear_interpolate_with_antialias(
                        block, size=out_size, mode="trilinear", **kwargs
                    ).cpu()

                    # write it back into the big tensor
                    out[
                        :,  # all batches
                        c0:c1,  # same channel slice
                        hz0:hz1,
                        hy0:hy1,
                        hx0:hx1,
                    ] = block
                    if pbar is not None:
                        pbar.update(1)

    if pbar is not None:
        pbar.close()

    return out


class Interpolated(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
        chunk_interpolate: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.kwargs = kwargs
        self.channel_wise = channel_wise
        self.chunk_interpolate = chunk_interpolate
        self.mode = self.kwargs["mode"]

        if self.mode == "trilinear":
            if self.chunk_interpolate:
                self.interpolate_fn = partial(chunked_trilinear_interpolate, **kwargs)
            else:
                self.interpolate_fn = partial(
                    trilinear_interpolate_with_antialias,
                    **kwargs,
                )
        else:
            self.interpolate_fn = partial(
                F.interpolate,
                **kwargs,
            )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        with torch.no_grad():
            d = dict(data)
            for key in self.key_iterator(d):
                is_numpy = False
                if isinstance(d[key], np.ndarray):
                    d[key] = convert_to_tensor(d[key])
                    is_numpy = True
                d[key] = self.interpolate_fn(
                    d[key].unsqueeze_(0)
                    if self.channel_wise
                    else d[key].unsqueeze_(0).unsqueeze_(0),
                    **self.kwargs,
                )
                d[key] = (
                    d[key].squeeze_(0)  # type: ignore
                    if self.channel_wise
                    else d[key].squeeze_(0).squeeze_(0)  # type: ignore
                )
                if is_numpy:
                    d[key] = convert_to_numpy(d[key])
        return d


class RandomResizedCropd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Tuple[int, int, int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        if not 0 < scale[0] <= scale[1]:
            raise ValueError("Scale values must be increasing and positive")
        if not 0 < ratio[0] <= ratio[1]:
            raise ValueError("Ratio values must be increasing and positive")
        self.roi_size = roi_size
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
            self.roi_size[1] / self.roi_size[0],  # H/D ratio
            self.roi_size[2] / self.roi_size[0],  # W/D ratio
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
        scale = min(D / self.roi_size[0], H / self.roi_size[1], W / self.roi_size[2])

        crop_d = min(D, int(self.roi_size[0] * scale))
        crop_h = min(H, int(self.roi_size[1] * scale))
        crop_w = min(W, int(self.roi_size[2] * scale))

        start_d = (D - crop_d) >> 1
        start_h = (H - crop_h) >> 1
        start_w = (W - crop_w) >> 1

        return crop_d, crop_h, crop_w, start_d, start_h, start_w

    def _apply(
        self,
        volume: NdarrayOrTensor,
    ) -> NdarrayOrTensor:
        """
        Optimized random resized crop application.
        """

        # Minimize tensor conversions
        is_numpy = isinstance(volume, np.ndarray)
        if is_numpy:
            volume = convert_to_tensor(volume)

        D, H, W = volume.shape[-3:]

        crop_d, crop_h, crop_w, start_d, start_h, start_w = self.get_params(D, H, W)

        volume = volume[
            ...,
            start_d : start_d + crop_d,
            start_h : start_h + crop_h,
            start_w : start_w + crop_w,
        ]

        ndims = 5
        if volume.dim() == 3:
            volume = volume.unsqueeze_(0).unsqueeze_(0)
            ndims = 3
        elif volume.dim() == 4:
            volume = volume.unsqueeze_(0)
            ndims = 4
        else:
            assert volume[0] == 1, (
                "Expected 5D tensor of shape [1, C, Z, Y, X] if has batch dim, got {volume.shape}"
            )

        resized = F.interpolate(
            volume,
            size=self.roi_size,
            mode=self.interpolation,
            align_corners=False,
        )

        if ndims == 3:
            resized = resized.squeeze_(0).squeeze_(0)
        elif ndims == 4:
            resized = resized.squeeze_(0)

        if is_numpy:
            resized = convert_to_numpy(resized)

        return resized

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self._apply(d[key])
        return d


class AddChannelsd(MapTransform):
    """Add channels to the image"""

    def __init__(
        self, num_channels: int, keys: KeysCollection, allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.num_channels = num_channels

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if isinstance(d[key], torch.Tensor):
                d[key] = d[key].unsqueeze_(0).repeat(self.num_channels, 1, 1, 1)  # type: ignore
            elif isinstance(d[key], np.ndarray):
                d[key] = d[key][None, ...]
                d[key] = np.repeat(d[key], self.num_channels, axis=0)
            else:
                raise ValueError(f"Unsupported type: {type(d[key])}")
        return d


class KMeansSpatialCropSamplesd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        save_keys: KeysCollection,
        roi_size: Tuple[int, int, int],
        num_samples: int,
        pmin: float = 99.75,
        pmax: float = 99.9,
        median_filter_size: int = 3,
        sample_size: int = 5000,
        channel_wise: bool = True,
        random_state: int = 42,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.save_keys = save_keys
        self.roi_size = roi_size
        self.num_samples = num_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.kmeans = KMeans(
            n_clusters=self.num_samples,
            random_state=self.random_state,
        )
        self.scaler = MinMaxScaler(copy=False)
        self.sample_size = sample_size
        self.pmin = pmin
        self.pmax = pmax
        self.median_filter_size = median_filter_size
        self.channel_wise = channel_wise

        self.save_keys = ensure_tuple(save_keys)

        assert len(self.keys) == len(self.save_keys), (
            f"Number of keys ({len(self.keys)}) must match number of save_keys ({len(self.save_keys)})"
        )

    @staticmethod
    @njit(fastmath=True, cache=True)
    def argwhere(a: np.ndarray) -> np.ndarray:
        count = 0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    if a[i, j, k]:
                        count += 1

        result = np.empty((count, 3), dtype=np.int32)
        idx = 0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    if a[i, j, k]:
                        result[idx, 0] = i
                        result[idx, 1] = j
                        result[idx, 2] = k
                        idx += 1
        return result

    def get_crop_params(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        is_tensor = False
        if isinstance(img, torch.Tensor):
            is_tensor = True
            img = convert_to_numpy(img)

        axis = None
        if self.channel_wise:
            axis = tuple(range(img.ndim))[1:]

        lower_bound, upper_bound = np.percentile(
            img,
            [self.pmin, self.pmax],
            axis=axis,
            keepdims=True,
        )

        Z, Y, X = img.shape[-3:]
        roi_z, roi_y, roi_x = self.roi_size

        # Handle both [C, Z, Y, X] and [Z, Y, X] cases
        if self.channel_wise:  # [C, Z, Y, X]
            mask = np.all(
                (img > lower_bound) & (img < upper_bound), axis=0
            )  # [Z, Y, X]
        else:  # [Z, Y, X]
            mask = (img > lower_bound) & (img < upper_bound)  # [Z, Y, X]

        obj_coords = KMeansSpatialCropSamplesd.argwhere(mask)  # [N, 3]

        N = obj_coords.shape[0]

        assert N > 0, (
            f"No valid coordinates found in the image with shape {img.shape} and bounds ({lower_bound}, {upper_bound})"
        )

        if N < self.num_samples:
            idx = self.rng.choice(
                N,
                size=self.num_samples,
                replace=True,
            ).astype(np.uint32)

            centroids = obj_coords[idx]

        else:
            idx = self.rng.choice(
                N,
                size=min(self.sample_size, N),
                replace=False,
            ).astype(np.uint32)

            obj_coords = obj_coords[idx].astype(np.float32)

            obj_coords = self.scaler.fit_transform(obj_coords)

            self.kmeans.fit(obj_coords)

            centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
            centroids = np.round(centroids).astype(
                np.int32
            )  # Convert to int coordinates

            # reset state of kmeans

            self.kmeans.cluster_centers_ = None
            self.kmeans.labels_ = None
            self.kmeans.inertia_ = None
            self.kmeans.n_iter_ = None

        # Initialize crop parameters array
        crop_params = np.empty((centroids.shape[0], 6), dtype=np.int32)

        for i, centroid in enumerate(centroids):
            # Ensure centroid is within image bounds
            z, y, x = np.clip(centroid, [0, 0, 0], [Z - 1, Y - 1, X - 1])

            # Calculate center crop positions that keep the centroid in the center
            z_start = max(0, min(Z - roi_z, z - roi_z // 2))
            y_start = max(0, min(Y - roi_y, y - roi_y // 2))
            x_start = max(0, min(X - roi_x, x - roi_x // 2))

            # Calculate end positions
            z_end = z_start + roi_z
            y_end = y_start + roi_y
            x_end = x_start + roi_x

            crop_params[i] = [z_start, z_end, y_start, y_end, x_start, x_end]

        if is_tensor:
            crop_params = convert_to_tensor(crop_params)

        return crop_params

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor],
    ) -> List[Dict[Hashable, NdarrayOrTensor]]:
        ret = [{} for _ in range(self.num_samples)]
        keys_to_copy = set(data) - set(self.keys)

        # copy over the non‐keys
        for i in range(self.num_samples):
            for k in keys_to_copy:
                ret[i][k] = deepcopy(data[k])

        roi_z, roi_y, roi_x = self.roi_size

        for idx, key in enumerate(self.key_iterator(data)):
            im = data[key]
            save_key = self.save_keys[idx]

            for i, (sz, ez, sy, ey, sx, ex) in enumerate(self.get_crop_params(im)):
                # 1) Extract the raw patch
                patch = im[
                    ..., sz:ez, sy:ey, sx:ex
                ]  # shape either [Z,Y,X] or [C,Z,Y,X]

                # 2) Ensure it's a torch.Tensor with a channel dim
                if not torch.is_tensor(patch):
                    patch = convert_to_tensor(patch)
                if not self.channel_wise:
                    patch = patch.unsqueeze_(0)

                # 3) Compute the one uniform scale
                Z, Y, X = patch.shape[-3:]
                scale = max(roi_z / Z, roi_y / Y, roi_x / X)
                new_z = math.ceil(Z * scale)
                new_y = math.ceil(Y * scale)
                new_x = math.ceil(X * scale)

                # 4) Interpolate this patch
                patch = F.interpolate(
                    patch.unsqueeze_(0),
                    size=(new_z, new_y, new_x),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze_(0)  # back to [C, Z', Y', X']

                # 5) Center‐crop any overshoot
                z0 = (new_z - roi_z) // 2
                y0 = (new_y - roi_y) // 2
                x0 = (new_x - roi_x) // 2
                patch = patch[
                    :,
                    z0 : z0 + roi_z,
                    y0 : y0 + roi_y,
                    x0 : x0 + roi_x,
                ]  # [C, roi_z, roi_y, roi_x]

                # 6) Squeeze channel if necessary and put back
                if not self.channel_wise:  # original was [Z,Y,X]
                    patch = patch.squeeze_(0)

                ret[i][save_key] = (
                    convert_to_numpy(patch) if isinstance(im, np.ndarray) else patch
                )

        return ret


def make_normalize_transform_global(
    image_key: KeysCollection,
    global_hist_max: float,
    global_hist_min: float,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    b_min: Optional[float] = None,
    b_max: Optional[float] = None,
    clip: bool = False,
    channel_wise: bool = True,
    allow_missing_keys: bool = False,
) -> List[Transform]:
    return [
        HistogramNormalized(
            keys=image_key,
            b_min=b_min,
            b_max=b_max,
            global_max_val=global_hist_max,
            global_min_val=global_hist_min,
            clip=clip,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=channel_wise,
            allow_missing_keys=allow_missing_keys,
        ),
        MinMaxNormalizedGlobal(
            keys=image_key,
            channel_wise=channel_wise,
            allow_missing_keys=allow_missing_keys,
        ),
        # NormalizeIntensityd(
        #     keys=image_key,
        #     subtrahend=np.asarray(mean),
        #     divisor=np.asarray(std),
        #     nonzero=False,  # normalize all values
        #     channel_wise=channel_wise,
        #     dtype=None,
        #     allow_missing_keys=allow_missing_keys,
        # ),
    ]


def make_normalize_transform(
    image_key: KeysCollection,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    b_min: Optional[float] = None,
    b_max: Optional[float] = None,
    clip: bool = False,
    channel_wise: bool = True,
    allow_missing_keys: bool = False,
) -> List[Transform]:
    return [
        HistogramNormalized(
            keys=image_key,
            b_min=b_min,
            b_max=b_max,
            clip=clip,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=channel_wise,
            allow_missing_keys=allow_missing_keys,
        ),
        MinMaxNormalized(
            keys=image_key,
            channel_wise=channel_wise,
            allow_missing_keys=allow_missing_keys,
        ),
        # NormalizeIntensityd(
        #     keys=image_key,
        #     subtrahend=np.asarray(mean),
        #     divisor=np.asarray(std),
        #     nonzero=False,  # normalize all values
        #     channel_wise=channel_wise,
        #     dtype=None,
        #     allow_missing_keys=allow_missing_keys,
        # ),
    ]


def make_global_transform(
    image_key: str,
    original_image_key: str = "image",
    img_size: Tuple[int, int, int] = (96, 96, 96),
    n_crops: int = 2,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    dtype: torch.dtype = torch.float32,
    random_state: int = 42,
    allow_missing_keys: bool = False,
) -> Compose:
    return Compose([
        # CopyItemsd(
        #     keys=original_image_key,
        #     names=image_key,
        #     allow_missing_keys=allow_missing_keys,
        # ),
        # DeleteItemsd(
        #     keys=original_image_key,
        # ),
        # RandSpatialCropSamplesd(
        #     keys=image_key,
        #     roi_size=img_size,
        #     num_samples=n_crops,
        #     random_center=True,
        #     random_size=False,
        #     allow_missing_keys=allow_missing_keys,
        # ),
        KMeansSpatialCropSamplesd(
            keys=original_image_key,
            save_keys=image_key,
            roi_size=img_size,
            num_samples=n_crops,
            channel_wise=True,
            random_state=random_state,
            allow_missing_keys=allow_missing_keys,
        ),
        RandFlipd(
            keys=image_key,
            prob=0.5,
            spatial_axis=-3,
            allow_missing_keys=allow_missing_keys,
        ),
        RandFlipd(
            keys=image_key,
            prob=0.5,
            spatial_axis=-2,
            allow_missing_keys=allow_missing_keys,
        ),
        RandFlipd(
            keys=image_key,
            prob=0.5,
            spatial_axis=-1,
            allow_missing_keys=allow_missing_keys,
        ),
        RandRotate90d(
            keys=image_key,
            prob=0.5,
            max_k=3,
            spatial_axes=(-3, -2),
            allow_missing_keys=allow_missing_keys,
        ),
        RandAdjustContrastd(
            keys=image_key,
            prob=0.8,
            gamma=(0.25, 2.0),
            invert_image=False,
            retain_stats=True,
            allow_missing_keys=allow_missing_keys,
        ),
        RandGaussianNoised(
            keys=image_key,
            prob=1.0,
            mean=0.0,
            std=0.1,
            sample_std=False,
            allow_missing_keys=allow_missing_keys,
        ),
        *make_normalize_transform(
            image_key=image_key,
            mean=mean,
            std=std,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=True,
            allow_missing_keys=allow_missing_keys,
        ),
        ToTensord(
            keys=image_key,
            dtype=dtype,
            allow_missing_keys=allow_missing_keys,
        ),
    ])


def make_local_transform(
    image_key: str,
    original_image_key: str = "image",
    img_size: Tuple[int, int, int] = (48, 48, 48),
    n_crops: int = 8,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    dtype: torch.dtype = torch.float32,
    random_state: int = 42,
    allow_missing_keys: bool = False,
) -> Compose:
    return Compose([
        # CopyItemsd(
        #     keys=original_image_key,
        #     names=image_key,
        #     allow_missing_keys=allow_missing_keys,
        # ),
        # DeleteItemsd(
        #     keys=original_image_key,
        # ),
        # RandSpatialCropSamplesd(
        #     keys=image_key,
        #     roi_size=img_size,
        #     num_samples=n_crops,
        #     random_center=True,
        #     random_size=False,
        #     allow_missing_keys=allow_missing_keys,
        # ),
        KMeansSpatialCropSamplesd(
            keys=original_image_key,
            save_keys=image_key,
            roi_size=img_size,
            num_samples=n_crops,
            channel_wise=True,
            random_state=random_state,
            allow_missing_keys=allow_missing_keys,
        ),
        RandFlipd(
            keys=image_key,
            prob=0.5,
            spatial_axis=-3,
            allow_missing_keys=allow_missing_keys,
        ),
        RandFlipd(
            keys=image_key,
            prob=0.5,
            spatial_axis=-2,
            allow_missing_keys=allow_missing_keys,
        ),
        RandFlipd(
            keys=image_key,
            prob=0.5,
            spatial_axis=-1,
            allow_missing_keys=allow_missing_keys,
        ),
        RandRotate90d(
            keys=image_key,
            prob=0.5,
            max_k=3,
            spatial_axes=(-3, -2),
            allow_missing_keys=allow_missing_keys,
        ),
        RandAdjustContrastd(
            keys=image_key,
            prob=0.8,
            gamma=(0.25, 2.0),
            invert_image=False,
            retain_stats=True,
            allow_missing_keys=allow_missing_keys,
        ),
        *make_normalize_transform(
            image_key=image_key,
            mean=mean,
            std=std,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=True,
            allow_missing_keys=allow_missing_keys,
        ),
        ToTensord(
            keys=image_key,
            dtype=dtype,
            allow_missing_keys=allow_missing_keys,
        ),
    ])


def make_base_sinder_transform(
    image_key: str,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    dtype: torch.dtype = torch.float32,
    random_state: int = 42,
    allow_missing_keys: bool = False,
) -> Compose:
    return Compose([
        # RandSpatialCropSamplesd(
        #     keys=image_key,
        #     roi_size=img_size,
        #     num_samples=1,
        #     random_center=True,
        #     random_size=False,
        #     allow_missing_keys=allow_missing_keys,
        # ),
        KMeansSpatialCropSamplesd(
            keys=image_key,
            save_keys=image_key,
            roi_size=img_size,
            num_samples=1,
            channel_wise=True,
            random_state=random_state,
            allow_missing_keys=allow_missing_keys,
        ),
        *make_normalize_transform(
            image_key=image_key,
            mean=mean,
            std=std,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=True,
            allow_missing_keys=allow_missing_keys,
        ),
        ToTensord(
            keys=image_key,
            dtype=dtype,
            allow_missing_keys=allow_missing_keys,
        ),
    ])


def make_base_segmentation_transform(
    image_key: str,
    img_size: Tuple[int, int, int] = (96, 96, 96),
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    dtype: torch.dtype = torch.float32,
    random_state: int = 42,
    allow_missing_keys: bool = False,
) -> Compose:
    return Compose([
        # RandSpatialCropSamplesd(
        #     keys=image_key,
        #     roi_size=img_size,
        #     num_samples=1,
        #     random_center=True,
        #     random_size=False,
        #     allow_missing_keys=allow_missing_keys,
        # ),
        KMeansSpatialCropSamplesd(
            keys=image_key,
            save_keys=image_key,
            roi_size=img_size,
            num_samples=1,
            channel_wise=True,
            random_state=random_state,
            allow_missing_keys=allow_missing_keys,
        ),
        *make_normalize_transform(
            image_key=image_key,
            mean=mean,
            std=std,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=True,
            allow_missing_keys=allow_missing_keys,
        ),
        ToTensord(
            keys=image_key,
            dtype=dtype,
            allow_missing_keys=allow_missing_keys,
        ),
    ])


class PreTrainTransform(object):
    def __init__(
        self,
        global_crop_size: Tuple[int, int, int] = (96, 96, 96),
        local_crop_size: Tuple[int, int, int] = (48, 48, 48),
        isotropic_scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        antialias: bool = False,
        n_global_crops: int = 2,
        n_local_crops: int = 8,
        image_key: str = "image",
        mask_key: str = "mask",
        in_chans: int = 3,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
        max_val: int = 65535,
        threshold_divisor: float = 1.0 / 5000,
        dtype: str = "fp32",
        random_state: int = 42,
        allow_missing_keys: bool = False,
    ) -> None:
        self.global_crop_size = make_3tuple(global_crop_size)
        self.local_crop_size = make_3tuple(local_crop_size)
        self.isotropic_scale_factor = make_3tuple(isotropic_scale_factor)
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.image_key = image_key
        self.mask_key = mask_key
        self.in_chans = in_chans
        self.mean = mean
        self.std = std
        self.max_val = max_val
        self.threshold_divisor = threshold_divisor
        self.dtype = DTYPE_MAPPING[dtype]
        self.allow_missing_keys = allow_missing_keys
        self.random_state = random_state
        assert n_global_crops > 0 or n_local_crops > 0, "No crops to copy"

        self.base_transform = []

        if mask_key:
            self.base_transform.extend([
                Maskd(
                    keys=[image_key, mask_key],
                    mask_key=mask_key,
                    allow_missing_keys=allow_missing_keys,
                ),
                DeleteItemsd(keys=mask_key),
            ])

        if isotropic_scale_factor != (1.0, 1.0, 1.0):
            self.base_transform.append(
                Interpolated(
                    keys=image_key,
                    channel_wise=False,
                    allow_missing_keys=allow_missing_keys,
                    scale_factor=self.isotropic_scale_factor,
                    mode="trilinear",
                    antialias=antialias,
                ),
            )

        spatial_size = tuple(
            max(self.global_crop_size[i], self.local_crop_size[i])
            for i in range(len(self.global_crop_size))
        )

        self.base_transform.extend([
            AddChannelsd(
                num_channels=in_chans,
                keys=image_key,
                allow_missing_keys=allow_missing_keys,
            )
        ])
        self.base_transform = Compose(self.base_transform)

        self.crop_keys = []

        if self.n_global_crops > 0:
            self.global_transform = make_global_transform(
                image_key="global_crops",
                original_image_key=image_key,
                img_size=self.global_crop_size,
                n_crops=self.n_global_crops,
                mean=self.mean,
                std=self.std,
                max_val=self.max_val,
                threshold_divisor=self.threshold_divisor,
                dtype=self.dtype,
                random_state=self.random_state,
                allow_missing_keys=self.allow_missing_keys,
            )
            self.crop_keys.append("global_crops")
        if self.n_local_crops > 0:
            self.local_transform = make_local_transform(
                image_key="local_crops",
                original_image_key=image_key,
                img_size=self.local_crop_size,
                n_crops=self.n_local_crops,
                mean=self.mean,
                std=self.std,
                max_val=self.max_val,
                threshold_divisor=self.threshold_divisor,
                dtype=self.dtype,
                random_state=self.random_state,
                allow_missing_keys=self.allow_missing_keys,
            )
            self.crop_keys.append("local_crops")

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[str, torch.Tensor]:
        data = self.base_transform(data)

        output = {}

        if self.n_global_crops > 0:
            global_crops_data = self.global_transform(data)
            output["global_crops"] = torch.stack(
                [item["global_crops"] for item in global_crops_data], dim=0
            )
        if self.n_local_crops > 0:
            local_crops_data = self.local_transform(data)
            output["local_crops"] = torch.stack(
                [item["local_crops"] for item in local_crops_data], dim=0
            )

        return output


class SegmentationTransform(object):
    def __init__(
        self,
        crop_size: Tuple[int, int, int] = (256, 256, 256),
        isotropic_scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        antialias: bool = False,
        image_key: str = "image",
        mask_key: str = "mask",
        in_chans: int = 1,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        max_val: int = 65535,
        threshold_divisor: float = 1.0 / 5000,
        dtype: str = "fp32",
        random_state: int = 42,
        allow_missing_keys: bool = False,
    ) -> None:
        self.crop_size = make_3tuple(crop_size)
        self.isotropic_scale_factor = make_3tuple(isotropic_scale_factor)
        self.image_key = image_key
        self.mask_key = mask_key
        self.in_chans = in_chans
        self.mean = mean
        self.std = std
        self.max_val = max_val
        self.threshold_divisor = threshold_divisor
        self.dtype = DTYPE_MAPPING[dtype]
        self.allow_missing_keys = allow_missing_keys
        self.random_state = random_state

        self.base_transform = []
        if mask_key:
            self.base_transform.extend([
                Maskd(
                    keys=[image_key, mask_key],
                    mask_key=mask_key,
                    allow_missing_keys=allow_missing_keys,
                ),
                DeleteItemsd(keys=[mask_key]),
            ])

        if isotropic_scale_factor != (1.0, 1.0, 1.0):
            self.base_transform.append(
                Interpolated(
                    keys=image_key,
                    channel_wise=False,
                    allow_missing_keys=allow_missing_keys,
                    scale_factor=self.isotropic_scale_factor,
                    mode="trilinear",
                    antialias=antialias,
                ),
            )

        spatial_size = tuple(self.crop_size[i] for i in range(len(self.crop_size)))

        self.base_transform.extend([
            AddChannelsd(
                num_channels=in_chans,
                keys=image_key,
                allow_missing_keys=allow_missing_keys,
            ),
            SpatialPadd(
                keys=image_key,
                spatial_size=spatial_size,
                mode="median",
                allow_missing_keys=allow_missing_keys,
            ),
        ])
        self.base_transform = Compose(self.base_transform)
        self.segmentation_transform = make_base_segmentation_transform(
            image_key="image",
            img_size=self.crop_size,
            mean=self.mean,
            std=self.std,
            max_val=self.max_val,
            threshold_divisor=self.threshold_divisor,
            dtype=self.dtype,
            random_state=self.random_state,
            allow_missing_keys=self.allow_missing_keys,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[str, torch.Tensor]:
        data = self.base_transform(data)
        output = self.segmentation_transform(data)[0]

        return output


class SinderTransform(object):
    def __init__(
        self,
        crop_size: Tuple[int, int, int] = (96, 96, 96),
        isotropic_scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        antialias: bool = False,
        image_key: str = "image",
        mask_key: str = "mask",
        in_chans: int = 1,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        max_val: int = 65535,
        threshold_divisor: float = 1.0 / 5000,
        dtype: str = "fp32",
        random_state: int = 42,
        allow_missing_keys: bool = False,
    ) -> None:
        self.crop_size = make_3tuple(crop_size)
        self.isotropic_scale_factor = make_3tuple(isotropic_scale_factor)
        self.image_key = image_key
        self.mask_key = mask_key
        self.in_chans = in_chans
        self.mean = mean
        self.std = std
        self.max_val = max_val
        self.threshold_divisor = threshold_divisor
        self.dtype = DTYPE_MAPPING[dtype]
        self.allow_missing_keys = allow_missing_keys
        self.random_state = random_state

        self.base_transform = []
        if mask_key:
            self.base_transform.extend([
                Maskd(
                    keys=[image_key, mask_key],
                    mask_key=mask_key,
                    allow_missing_keys=allow_missing_keys,
                ),
                DeleteItemsd(keys=[mask_key]),
            ])

        if isotropic_scale_factor != (1.0, 1.0, 1.0):
            self.base_transform.append(
                Interpolated(
                    keys=image_key,
                    channel_wise=False,
                    allow_missing_keys=allow_missing_keys,
                    scale_factor=self.isotropic_scale_factor,
                    mode="trilinear",
                    antialias=antialias,
                ),
            )

        spatial_size = tuple(self.crop_size[i] for i in range(len(self.crop_size)))

        self.base_transform.extend([
            AddChannelsd(
                num_channels=in_chans,
                keys=image_key,
                allow_missing_keys=allow_missing_keys,
            ),
            SpatialPadd(
                keys=image_key,
                spatial_size=spatial_size,
                mode="median",
                allow_missing_keys=allow_missing_keys,
            ),
        ])
        self.base_transform = Compose(self.base_transform)
        self.sinder_transform = make_base_sinder_transform(
            image_key="image",
            img_size=self.crop_size,
            mean=self.mean,
            std=self.std,
            max_val=self.max_val,
            threshold_divisor=self.threshold_divisor,
            dtype=self.dtype,
            random_state=self.random_state,
            allow_missing_keys=self.allow_missing_keys,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[str, torch.Tensor]:
        data = self.base_transform(data)
        output = self.sinder_transform(data)[0]

        return output


class LiFTtransforms(object):
    def __init__(
        self,
        isotropic_scale_factor: Tuple[float, float, float],
        antialias: bool = False,
        global_crops_size: Tuple[int, int, int] = (64, 64, 64),
        image_key: str = "image",
        mask_key: str = "mask",
        in_chans: int = 1,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        max_val: int = 65535,
        threshold_divisor: float = 1.0 / 5000,
        dtype: str = "fp32",
        random_state: int = 42,
        allow_missing_keys: bool = False,
    ) -> None:
        self.global_crops_size = make_3tuple(global_crops_size)  # type: ignore
        self.isotropic_scale_factor = make_3tuple(isotropic_scale_factor)  # type: ignore
        self.image_key, self.mask_key = image_key, mask_key
        self.in_chans = in_chans
        self.mean, self.std = mean, std
        self.max_val = max_val
        self.threshold_divisor = threshold_divisor
        self.dtype = DTYPE_MAPPING[dtype]
        self.random_state = random_state
        self.allow_missing_keys = allow_missing_keys

        self.base_transform = []
        if mask_key:
            self.base_transform.extend([
                Maskd(
                    keys=[image_key, mask_key],
                    mask_key=mask_key,
                    allow_missing_keys=allow_missing_keys,
                ),
                DeleteItemsd(keys=[mask_key]),
            ])

        if isotropic_scale_factor != (1.0, 1.0, 1.0):
            self.base_transform.append(
                Interpolated(
                    keys=image_key,
                    channel_wise=False,
                    allow_missing_keys=allow_missing_keys,
                    scale_factor=self.isotropic_scale_factor,
                    mode="trilinear",
                    antialias=antialias,
                ),
            )

        spatial_size = tuple(
            self.global_crops_size[i] for i in range(len(self.global_crops_size))
        )

        self.base_transform.extend([
            AddChannelsd(
                num_channels=in_chans,
                keys=image_key,
                allow_missing_keys=allow_missing_keys,
            ),
            SpatialPadd(
                keys=image_key,
                spatial_size=spatial_size,
                mode="median",
                allow_missing_keys=allow_missing_keys,
            ),
        ])
        self.base_transform = Compose(self.base_transform)
        self.lift_transform = make_lift_transform(
            image_key="image",
            img_size=self.global_crops_size,
            mean=self.mean,
            std=self.std,
            max_val=self.max_val,
            threshold_divisor=self.threshold_divisor,
            dtype=self.dtype,
            allow_missing_keys=self.allow_missing_keys,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[str, torch.Tensor]:
        data = self.base_transform(data)
        image_data = self.lift_transform(data)
        item = {
            "image": image_data[0]["image"],  # type: ignore
            "image_half": image_data[0]["image"][..., ::2, ::2, ::2],  # type: ignore
            "image_quarter": image_data[0]["image"][..., ::4, ::4, ::4],  # type: ignore
        }

        return item


def make_inference_transform(
    target_img_size: Tuple[int, int, int],
    upsample_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    isotropic_scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk_interpolate: bool = True,
    antialias: bool = False,
    device: Union[torch.device, str] = "cpu",
    interpolate_chunk_size: Tuple[int, int, int, int] = (1, 448, 448, 448),
    pad_value: float = 0.0,
    image_key: str = "image",
    mask_key: str = "mask",
    in_chans: int = 3,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    dtype: str = "fp32",
    allow_missing_keys: bool = False,
) -> Compose:
    dtype = DTYPE_MAPPING[dtype]  # type: ignore

    transforms_list = []

    if mask_key:
        keys = [image_key, mask_key]
        transforms_list.extend([
            Maskd(
                keys=keys,
                mask_key=mask_key,
                allow_missing_keys=allow_missing_keys,
            ),
            DeleteItemsd(
                keys=mask_key,
            ),
        ])

    if upsample_factor != (1.0, 1.0, 1.0) or isotropic_scale_factor != (1.0, 1.0, 1.0):
        scale_factor = tuple(
            upsample_factor[i] * isotropic_scale_factor[i]
            for i in range(len(upsample_factor))
        )
        transforms_list.append(
            Interpolated(
                keys=image_key,
                channel_wise=False,
                allow_missing_keys=allow_missing_keys,
                scale_factor=scale_factor,
                mode="trilinear",
                antialias=antialias,
                chunk_interpolate=chunk_interpolate,
                device=device,
                chunk_size=interpolate_chunk_size,
            ),
        )

    transforms_list.extend([
        AddChannelsd(
            num_channels=in_chans,
            keys=image_key,
            allow_missing_keys=allow_missing_keys,
        ),
        ResizeWithPadOrCropd(
            keys=image_key,
            spatial_size=target_img_size,
            mode="constant",  # "median" is passed in as a constant for efficiency
            constant_values=pad_value,
            allow_missing_keys=allow_missing_keys,
        ),
        *make_normalize_transform(
            image_key=image_key,
            mean=mean,
            std=std,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=True,
            allow_missing_keys=allow_missing_keys,
        ),
        ToTensord(
            keys=image_key,
            dtype=dtype,  # type: ignore
            allow_missing_keys=allow_missing_keys,
        ),
    ])

    return Compose(transforms_list)


def make_inference_transform_global(
    target_img_size: Tuple[int, int, int],
    global_hist_max: float,
    global_hist_min: float,
    upsample_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    isotropic_scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    chunk_interpolate: bool = True,
    antialias: bool = False,
    device: Union[torch.device, str] = "cpu",
    interpolate_chunk_size: Tuple[int, int, int, int] = (1, 448, 448, 448),
    pad_value: float = 0.0,
    image_key: str = "image",
    mask_key: str = "mask",
    in_chans: int = 3,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    dtype: str = "fp32",
    allow_missing_keys: bool = False,
) -> Compose:
    dtype = DTYPE_MAPPING[dtype]  # type: ignore

    transforms_list = []

    if mask_key:
        keys = [image_key, mask_key]
        transforms_list.extend([
            Maskd(
                keys=keys,
                mask_key=mask_key,
                allow_missing_keys=allow_missing_keys,
            ),
            DeleteItemsd(
                keys=mask_key,
            ),
        ])

    if upsample_factor != (1.0, 1.0, 1.0) or isotropic_scale_factor != (1.0, 1.0, 1.0):
        scale_factor = tuple(
            upsample_factor[i] * isotropic_scale_factor[i]
            for i in range(len(upsample_factor))
        )
        transforms_list.append(
            Interpolated(
                keys=image_key,
                channel_wise=False,
                allow_missing_keys=allow_missing_keys,
                scale_factor=scale_factor,
                mode="trilinear",
                antialias=antialias,
                chunk_interpolate=chunk_interpolate,
                device=device,
                chunk_size=interpolate_chunk_size,
            ),
        )

    transforms_list.extend([
        AddChannelsd(
            num_channels=in_chans,
            keys=image_key,
            allow_missing_keys=allow_missing_keys,
        ),
        ResizeWithPadOrCropd(
            keys=image_key,
            spatial_size=target_img_size,
            mode="constant",  # "median" is passed in as a constant for efficiency
            constant_values=pad_value,
            allow_missing_keys=allow_missing_keys,
        ),
        *make_normalize_transform_global(
            image_key=image_key,
            global_hist_max=global_hist_max,
            global_hist_min=global_hist_min,
            mean=mean,
            std=std,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=True,
            allow_missing_keys=allow_missing_keys,
        ),
        ToTensord(
            keys=image_key,
            dtype=dtype,  # type: ignore
            allow_missing_keys=allow_missing_keys,
        ),
    ])

    return Compose(transforms_list)


def make_lift_transform(
    image_key: str,
    img_size: Tuple[int, int, int] = (32, 96, 96),
    n_crops: int = 1,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    max_val: int = 65535,
    threshold_divisor: float = 1.0 / 5000,
    dtype: torch.dtype = torch.float32,
    allow_missing_keys: bool = False,
) -> Compose:
    return Compose([
        RandSpatialCropSamplesd(
            keys=image_key,
            roi_size=img_size,
            num_samples=n_crops,
            random_center=True,
            random_size=False,
            allow_missing_keys=allow_missing_keys,
        ),
        RandAdjustContrastd(
            keys=image_key,
            prob=0.4,
            gamma=(0.25, 2.0),
            invert_image=False,
            retain_stats=True,
            allow_missing_keys=allow_missing_keys,
        ),
        *make_normalize_transform(
            image_key=image_key,
            mean=mean,
            std=std,
            max_val=max_val,
            threshold_divisor=threshold_divisor,
            channel_wise=True,
            allow_missing_keys=allow_missing_keys,
        ),
        ToTensord(
            keys=image_key,
            dtype=dtype,
            allow_missing_keys=allow_missing_keys,
        ),
    ])


class PatchNormalize(Transform):
    """
    Normalize each patch independently to handle contrast variations across patches.
    This is particularly useful for large 3D volumes where different regions may have
    different contrast characteristics.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = (448, 448, 448),
        overlap_factor: float = 0.1,
        method: str = "minmax",  # "minmax", "histogram", "adaptive"
        channel_wise: bool = True,
        blend_boundaries: bool = True,
        threshold_divisor: float = 1.0 / 5000,
        max_val: int = 65535,
    ):
        super().__init__()
        self.patch_size = make_3tuple(patch_size)
        self.overlap_factor = overlap_factor
        self.method = method
        self.channel_wise = channel_wise
        self.blend_boundaries = blend_boundaries
        self.threshold_divisor = threshold_divisor
        self.max_val = max_val

        if method == "minmax":
            self.normalizer = MinMaxNormalize(channel_wise=channel_wise)
        elif method == "histogram":
            self.normalizer = HistogramNormalize(
                channel_wise=channel_wise,
                threshold_divisor=threshold_divisor,
                max_val=max_val,
            )
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def _get_patch_coordinates(
        self, volume_shape: Tuple[int, ...]
    ) -> List[Tuple[slice, ...]]:
        """Generate overlapping patch coordinates"""
        if len(volume_shape) == 3:  # No channel dimension
            Z, Y, X = volume_shape
        else:  # Has channel dimension
            _, Z, Y, X = volume_shape

        pz, py, px = self.patch_size
        overlap_z = int(pz * self.overlap_factor)
        overlap_y = int(py * self.overlap_factor)
        overlap_x = int(px * self.overlap_factor)

        stride_z = pz - overlap_z
        stride_y = py - overlap_y
        stride_x = px - overlap_x

        coordinates = []

        for z_start in range(0, Z, stride_z):
            for y_start in range(0, Y, stride_y):
                for x_start in range(0, X, stride_x):
                    z_end = min(z_start + pz, Z)
                    y_end = min(y_start + py, Y)
                    x_end = min(x_start + px, X)

                    if len(volume_shape) == 3:
                        coordinates.append((
                            slice(z_start, z_end),
                            slice(y_start, y_end),
                            slice(x_start, x_end),
                        ))
                    else:
                        coordinates.append((
                            slice(None),  # All channels
                            slice(z_start, z_end),
                            slice(y_start, y_end),
                            slice(x_start, x_end),
                        ))

        return coordinates

    def _blend_overlaps(
        self,
        normalized_volume: NdarrayOrTensor,
        patch_coords: List[Tuple[slice, ...]],
        patch_data: List[NdarrayOrTensor],
    ) -> NdarrayOrTensor:
        """Blend overlapping regions using weighted averaging"""
        is_numpy = isinstance(normalized_volume, np.ndarray)
        if is_numpy:
            weight_volume = np.zeros_like(normalized_volume)
            blended_volume = np.zeros_like(normalized_volume)
        else:
            weight_volume = torch.zeros_like(normalized_volume)
            blended_volume = torch.zeros_like(normalized_volume)

        for coords, patch in zip(patch_coords, patch_data):
            # Create weight map for this patch (higher in center, lower at edges)
            patch_shape = patch.shape
            if len(patch_shape) == 3:
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
            else:
                # Handle channel dimension
                z_weights = np.linspace(0.1, 1.0, patch_shape[1] // 2)
                z_weights = np.concatenate([z_weights, z_weights[::-1]])
                if len(z_weights) < patch_shape[1]:
                    z_weights = np.concatenate([z_weights, [1.0]])

                y_weights = np.linspace(0.1, 1.0, patch_shape[2] // 2)
                y_weights = np.concatenate([y_weights, y_weights[::-1]])
                if len(y_weights) < patch_shape[2]:
                    y_weights = np.concatenate([y_weights, [1.0]])

                x_weights = np.linspace(0.1, 1.0, patch_shape[3] // 2)
                x_weights = np.concatenate([x_weights, x_weights[::-1]])
                if len(x_weights) < patch_shape[3]:
                    x_weights = np.concatenate([x_weights, [1.0]])

                spatial_weights = np.outer(
                    z_weights, np.outer(y_weights, x_weights)
                ).reshape(patch_shape[1:])
                weights = np.tile(spatial_weights, (patch_shape[0], 1, 1, 1))

            if not is_numpy:
                weights = torch.from_numpy(weights).type_as(patch)

            blended_volume[coords] += patch * weights
            weight_volume[coords] += weights

        # Normalize by total weights
        if is_numpy:
            weight_volume[weight_volume == 0] = 1  # Avoid division by zero
        else:
            weight_volume = torch.where(
                weight_volume == 0, torch.ones_like(weight_volume), weight_volume
            )

        return blended_volume / weight_volume

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        is_numpy = isinstance(img, np.ndarray)
        original_img = img

        if is_numpy:
            img = convert_to_tensor(img)

        # Get patch coordinates
        patch_coords = self._get_patch_coordinates(img.shape)

        # Process each patch
        normalized_patches = []
        for coords in patch_coords:
            patch = img[coords]
            normalized_patch = self.normalizer(patch)
            normalized_patches.append(normalized_patch)

        if self.blend_boundaries and len(patch_coords) > 1:
            # Create output volume with same shape as input
            normalized_volume = torch.zeros_like(img)
            # Blend overlapping patches
            result = self._blend_overlaps(
                normalized_volume, patch_coords, normalized_patches
            )
        else:
            # Simple non-overlapping assignment
            result = torch.zeros_like(img)
            for coords, normalized_patch in zip(patch_coords, normalized_patches):
                result[coords] = normalized_patch

        if is_numpy:
            result = convert_to_numpy(result)

        return result


class PatchNormalized(MapTransform):
    """Dictionary-based patch normalization transform"""

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Union[int, Tuple[int, int, int]] = (448, 448, 448),
        overlap_factor: float = 0.1,
        method: str = "minmax",
        channel_wise: bool = True,
        blend_boundaries: bool = True,
        threshold_divisor: float = 1.0 / 5000,
        max_val: int = 65535,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalize = PatchNormalize(
            patch_size=patch_size,
            overlap_factor=overlap_factor,
            method=method,
            channel_wise=channel_wise,
            blend_boundaries=blend_boundaries,
            threshold_divisor=threshold_divisor,
            max_val=max_val,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalize(d[key])
        return d