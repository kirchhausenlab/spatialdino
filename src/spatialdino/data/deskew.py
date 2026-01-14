from skimage import io
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
import numpy.linalg as LA
import torch
import torch.nn.functional as F
from monai.utils.enums import GridSampleMode, GridSamplePadMode
from typing import (
    Any,
    Final,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    Iterable,
    Collection,
    Callable,
)
import warnings
import matplotlib.pyplot as plt
from monai.networks.layers import AffineTransform
from torch.utils.data import Dataset


class DeskewDataset(Dataset):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path
        self.fnames = natsorted(list(Path(file_path).glob("*.tif")))

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray]:
        fname = self.fnames[idx]
        raw_volume = io.imread(fname).astype(np.float32)
        raw_volume = np.expand_dims(raw_volume, axis=0)  # [C, Z, Y, X]
        return idx, raw_volume


def rot_around_y(angle_deg: float) -> np.ndarray:
    """create affine matrix for rotation around y axis

    Parameters
    ----------
    angle_deg : float
        rotation angle in degrees

    Returns
    -------
    np.ndarray
        4x4 affine rotation matrix
    """
    arad = angle_deg * np.pi / 180.0
    roty = np.array(
        [
            [np.cos(arad), 0, np.sin(arad), 0],
            [0, 1, 0, 0],
            [-np.sin(arad), 0, np.cos(arad), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return roty


def get_transformed_corners(
    aff: np.ndarray,
    vol_or_shape: Union[np.ndarray, Iterable[float]],
    zeroindex: bool = True,
):
    """Input
    aff: an affine transformation matrix
    vol_or_shape: a numpy volume or shape of a volume.

    This function will return the positions of the corner points of the volume (or volume with
    provided shape) after applying the affine transform.
    """
    # get the dimensions of the array.
    # see whether we got a volume
    if np.array(vol_or_shape).ndim == 3:
        d0, d1, d2 = np.array(vol_or_shape).shape
    elif np.array(vol_or_shape).ndim == 1:
        d0, d1, d2 = vol_or_shape
    else:
        raise ValueError
    # By default we calculate where the corner points in
    # zero-indexed (numpy) arrays will be transformed to.
    # set zeroindex to False if you want to perform the calculation
    # for Matlab-style arrays.
    if zeroindex:
        d0 -= 1
        d1 -= 1
        d2 -= 1
    # all corners of the input volume (maybe there is
    # a more concise way to express this with itertools?)
    corners_in = [
        (0, 0, 0, 1),
        (d0, 0, 0, 1),
        (0, d1, 0, 1),
        (0, 0, d2, 1),
        (d0, d1, 0, 1),
        (d0, 0, d2, 1),
        (0, d1, d2, 1),
        (d0, d1, d2, 1),
    ]
    corners_out = list(map(lambda c: aff @ np.array(c, dtype=np.float32), corners_in))
    corner_array = np.concatenate(corners_out).reshape((-1, 4))
    # print(corner_array)
    return corner_array


def ceil_to_multiple(x, base: int = 4):
    """rounds up to the nearest integer multiple of base

    Parameters
    ----------
    x : scalar or np.array
        value/s to round up from
    base : int, optional
        round up to multiples of base (the default is 4)

    Returns
    -------

    scalar or np.array:
        rounded up value/s

    """

    return (np.int32(base) * np.ceil(np.array(x).astype(np.float64) / base)).astype(
        np.int32
    )


def shift_center(matrix_shape: Collection[float], direction=-1.0) -> np.ndarray:
    """create an affine matrix for shifting centre of a volume to the origin

    Parameters
    ----------
    matrix_shape : Collection[float]
        shape of the volume to be shifted (typically a 3-tuple)
    direction : float, optional
        [description] (the default is -1.0, which [default_description])

    Returns
    -------
    np.ndarray
        4x4 affine translation matrix
    """
    assert len(matrix_shape) == 3
    center = np.array(matrix_shape, dtype=np.float32) / 2
    shift = np.eye(4, dtype=np.float32)
    shift[:3, 3] = direction * center
    return shift


def unshift_center(matrix_shape: Collection[float]) -> np.ndarray:
    """like shift_center but with inverse tranlation direction"""
    return shift_center(matrix_shape, 1.0)


def get_output_dimensions(
    aff: np.ndarray, vol_or_shape: Union[np.ndarray, Iterable[float]]
):
    """given an 4x4 affine transformation matrix aff and
    a 3d input volume (numpy array) or volumen shape (iterable with 3 elements)
    this function returns the output dimensions required for the array after the
    transform. Rounds up to create an integer result.
    """
    corners = get_transformed_corners(aff, vol_or_shape, zeroindex=True)
    # +1 to avoid fencepost error
    dims = np.max(corners, axis=0) - np.min(corners, axis=0) + 1
    dims = ceil_to_multiple(dims, 2)
    return dims[:3].astype(np.int32)


class DeskewConfig:
    def __init__(self, dx: float, dy: float, dz_stage: float, angle: float):
        assert dx == dy, "dx and dy are assumed to be equal"
        self.dx = dx
        self.dy = dy
        self.dz_stage = dz_stage
        self.angle = angle
        self.dz = np.sin(self.angle * np.pi / 180.0) * self.dz_stage
        self.deskew_factor = (
            np.cos(self.angle * np.pi / 180.0) * self.dz_stage / self.dx
        )
        self.isotropic_scale = self.dz / self.dx


DEFAULT_DESKEW_CONFIG: Final[DeskewConfig] = DeskewConfig(
    dx=0.104,
    dy=0.104,
    dz_stage=0.5,
    angle=31.5,
)


def get_deskew_transform(
    vol: np.ndarray,
    config: DeskewConfig = DEFAULT_DESKEW_CONFIG,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """returns the affine transformation matrix for deskewing
    a volume with the default parameters"""
    dx = config.dx
    dy = config.dy
    dz = config.dz

    angle = config.angle
    deskew_factor = config.deskew_factor
    isotropic_scale = config.isotropic_scale

    if verbose:
        print("Parameter summary:")
        print("==================")
        print("dz:", dz)
        print("dy:", dy)
        print("dx:", dx)
        print("deskew_factor:", deskew_factor)
        print("voxel aspect ratio z-voxel/xy-voxel (isotropic_scale):", isotropic_scale)

    skew = np.eye(4, dtype=np.float32)
    skew[2, 0] = deskew_factor

    scale = np.eye(4, dtype=np.float32)
    scale[0, 0] = isotropic_scale

    rot = rot_around_y(-angle)

    shift_original = shift_center(vol.shape)
    combined = rot @ scale @ skew @ shift_original
    output_shape = get_output_dimensions(combined, vol)
    unshift_final = unshift_center(output_shape)
    all_in_one = unshift_final @ rot @ scale @ skew @ shift_original
    all_in_one_inv = LA.inv(all_in_one)
    return output_shape, all_in_one_inv


def build_affine_transform(
    normalized: bool = False,
    reverse_indexing: bool = True,
    interpolation: str = GridSampleMode.BILINEAR,
    padding_mode: str = GridSamplePadMode.ZEROS,
    **kwargs: Any,
) -> AffineTransform:
    """Builds an affine transform for deskewing."""
    affine_transform = AffineTransform(
        normalized=normalized,
        reverse_indexing=reverse_indexing,
        padding_mode=padding_mode,
        mode=interpolation,
        **kwargs,
    )
    return affine_transform


@torch.no_grad()
def deskew(
    vol: torch.Tensor,
    affine_transform: AffineTransform,
    deskew_matrix: torch.Tensor,
    output_shape: Optional[np.ndarray] = None,
) -> torch.Tensor:
    assert vol.dim() == 5, (
        "deskew function only works with 5D volumes of shape (B, C, Z, Y, X)"
    )

    result = affine_transform(vol, theta=deskew_matrix, spatial_size=output_shape)

    return result
