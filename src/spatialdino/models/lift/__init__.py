from typing import List, Tuple

import torch

MEAN: list[float] = [0.485, 0.456, 0.406]
STD: list[float] = [0.229, 0.224, 0.225]


def get_mean(
    is_dino: bool = True,
) -> list[float]:
    if not is_dino:
        return [0.5, 0.5, 0.5]
    return MEAN


def get_std(
    is_dino: bool = True,
) -> list[float]:
    if not is_dino:
        return [0.5, 0.5, 0.5]
    return STD


# [B, T, C] --> [B, C, D, H, W]
def convert_shape(
    x: torch.Tensor,
    patches: tuple[int, int, int],
) -> torch.Tensor:
    x = x.permute(0, 2, 1)  # B, C, T
    x = x.reshape(x.shape[0], -1, patches[0], patches[1], patches[2])
    return x
