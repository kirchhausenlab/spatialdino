from enum import Enum
from typing import Annotated, Dict, Final, List, Literal, Tuple, TypedDict

import numpy as np
import torch
from pyparsing import Optional

EMPTY_VOXEL_THRESHOLD: Final[float] = 1e-6
Values2D = Annotated[np.ndarray, Literal["Y", "X", 3]]
Position2D = Annotated[np.ndarray, Literal[2]]
Values3D = Annotated[np.ndarray, Literal["Z", "Y", "X"]]
Position3D = Annotated[np.ndarray, Literal[3]]


class BaseMetadata(TypedDict):
    path: str
    frame: int
    stack: int
    wavelength: int
    camera: str
    channel: str
    dtype: str
    shape: Tuple[int, ...]


class Metadata2D(BaseMetadata):
    pass


class Metadata3D(BaseMetadata):
    dx: float
    dy: float
    dz: float
    non_zero_stacks: np.ndarray


class Image2DData(TypedDict):
    values: Values2D
    position: Position2D


class Image2D(TypedDict):
    data: Image2DData
    metadata: Metadata2D


class DINOv2Image2D(TypedDict):
    global_crops: List[torch.Tensor]
    global_crops_teacher: List[torch.Tensor]
    local_crops: List[torch.Tensor]
    offsets: List[torch.Tensor]


class Image3DData(TypedDict):
    values: Values3D
    position: Position3D


class Image3D(TypedDict):
    data: Image3DData
    metadata: Metadata3D


class Dimension(Enum):
    DIM_2D = "2D"
    DIM_3D = "3D"


class ZFlattenMode(Enum):
    MAX = "MAX"
    MEAN = "MEAN"
    NONE = "NONE"


DTYPE_MAPPING: Final[Dict[str, torch.dtype]] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
    "u8": torch.uint8,
}