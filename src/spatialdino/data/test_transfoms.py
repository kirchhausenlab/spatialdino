from typing import List, Callable, Tuple
import torch


def identity_transform(x: torch.Tensor) -> torch.Tensor:
    return x


def flip_transform(x: torch.Tensor, spatial_axis: Tuple[int, ...]) -> torch.Tensor:
    return torch.flip(x, dims=spatial_axis)


def make_transforms() -> List[Callable[[torch.Tensor], torch.Tensor]]:
    return [
        identity_transform,
        # partial(flip_transform, spatial_axis=(-3,)),
        # partial(flip_transform, spatial_axis=(-2,)),
        # partial(flip_transform, spatial_axis=(-1,)),
    ]


def make_inverse_transforms() -> List[Callable[[torch.Tensor], torch.Tensor]]:
    return [
        identity_transform,
        # partial(flip_transform, spatial_axis=(-3,)),
        # partial(flip_transform, spatial_axis=(-2,)),
        # partial(flip_transform, spatial_axis=(-1,)),
    ]
