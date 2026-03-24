# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import random
from typing import Any, Dict, List, Optional, Tuple
from scipy import ndimage as ndi
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.measure import block_reduce


def block_mask(
    mask: np.ndarray,
    max_mask_patches: int,
    grid_size: Tuple[int, int, int],
    log_aspect_ratio: Tuple[float, float],
    min_num_patches: int = 4,
) -> int:
    for _ in range(10):
        target_area = random.uniform(min_num_patches, max_mask_patches)
        # Generate random aspect ratios for depth and height relative to width
        aspect_ratio_d = math.exp(random.uniform(*log_aspect_ratio))
        aspect_ratio_h = math.exp(random.uniform(*log_aspect_ratio))

        # Calculate dimensions for 3D block
        d = int(round(math.pow(target_area * aspect_ratio_d * aspect_ratio_h, 1 / 3)))
        h = int(round(math.pow(target_area * aspect_ratio_h / aspect_ratio_d, 1 / 3)))
        w = int(round(math.pow(target_area / (aspect_ratio_d * aspect_ratio_h), 1 / 3)))

        if d < grid_size[0] and h < grid_size[1] and w < grid_size[2]:
            # Random position for 3D block
            z = random.randint(0, grid_size[0] - d)
            top = random.randint(0, grid_size[1] - h)
            left = random.randint(0, grid_size[2] - w)

            # Get the 3D region to be masked
            region = mask[z : z + d, top : top + h, left : left + w]
            num_masked = region.sum()

            # Check overlap condition
            if 0 < d * h * w - num_masked <= max_mask_patches:
                # Use boolean indexing to update the mask in 3D
                mask[z : z + d, top : top + h, left : left + w] = True
                delta = d * h * w - num_masked
                return int(delta)

    return 0


def random_mask(
    high: int,
    grid_size: Tuple[int, int, int],
) -> np.ndarray:
    mask = np.hstack([
        np.zeros(np.prod(grid_size) - high),
        np.ones(high),
    ]).astype(bool)
    np.random.shuffle(mask)
    mask = mask.reshape(grid_size)
    return mask


def _normalize_mask_type(mask_type: str) -> str:
    if mask_type == "random":
        return "rand"
    return mask_type


def mask_generator(
    high: int,
    grid_size: Tuple[int, int, int],
    min_aspect: float = 0.3,
    max_aspect: Optional[float] = None,
    min_num_patches: int = 4,
    mask_type: str = "block",
) -> torch.Tensor:
    mask_type = _normalize_mask_type(mask_type)

    if mask_type == "block":
        max_aspect = max_aspect or 1 / min_aspect
        log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        mask = np.zeros(grid_size, dtype=bool)
        mask_count = 0
        while mask_count < high:
            max_mask_patches = high - mask_count
            delta = block_mask(
                mask=mask,
                max_mask_patches=max_mask_patches,
                grid_size=grid_size,
                log_aspect_ratio=log_aspect_ratio,
                min_num_patches=min_num_patches,
            )
            if delta == 0:
                break
            else:
                mask_count += delta
    elif mask_type == "rand":
        mask = random_mask(
            high=high,
            grid_size=grid_size,
        )
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")

    return torch.from_numpy(mask)


def collate_fn_train(
    samples_list: List[Dict[str, Any]],
    grid_size: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    min_mask_ratio: float = 0.1,
    max_mask_ratio: float = 0.5,
    mask_sample_probability: float = 0.5,
    min_aspect: float = 0.3,
    max_aspect: Optional[float] = None,
    min_num_patches: int = 4,
    mask_type: str = "block",
) -> Dict[str, Any]:
    batch_size = len(samples_list)

    dtype = samples_list[0]["global_crops"].dtype
    n_global_crops = samples_list[0]["global_crops"].size(0)
    collated_global_crops = torch.empty(
        (batch_size * n_global_crops, *samples_list[0]["global_crops"].shape[1:]),
        dtype=dtype,
    )
    for i in range(batch_size):
        collated_global_crops[i * n_global_crops : (i + 1) * n_global_crops] = (
            samples_list[i]["global_crops"]
        )

    dtype = samples_list[0]["local_crops"].dtype
    n_local_crops = samples_list[0]["local_crops"].size(0)
    collated_local_crops = torch.empty(
        (batch_size * n_local_crops, *samples_list[0]["local_crops"].shape[1:]),
        dtype=dtype,
    )

    for i in range(batch_size):
        collated_local_crops[i * n_local_crops : (i + 1) * n_local_crops] = (
            samples_list[i]["local_crops"]
        )

    B = len(collated_global_crops)
    L = np.prod(grid_size)

    n_samples_masked = int(B * mask_sample_probability)
    probs = torch.linspace(min_mask_ratio, max_mask_ratio, n_samples_masked + 1)
    masks_list = []
    upperbound = 0
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            mask_generator(
                int(L * random.uniform(prob_min.item(), prob_max.item())),
                grid_size=grid_size,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                min_num_patches=min_num_patches,
                mask_type=mask_type,
            )
        )
        upperbound += int(L * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(
            mask_generator(
                0,
                grid_size=grid_size,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                min_num_patches=min_num_patches,
                mask_type=mask_type,
            )
        )

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)  # [B, L]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()  # [B*L]

    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze_(-1)
        .expand_as(collated_masks)[collated_masks]
    )  # [B, L]

    n_masked_patches_tensor = torch.full(
        (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
    )

    return {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches_tensor": n_masked_patches_tensor,
    }


def collate_fn_sinder(
    image_batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dtype = image_batch[0]["image"].dtype
    collated_images = torch.empty(
        (len(image_batch), *image_batch[0]["image"].shape), dtype=dtype
    )
    for i, item in enumerate(image_batch):
        collated_images[i] = item["image"]

    res: Dict[str, Any] = {
        "collated_images": collated_images,
    }

    return res


def collate_fn_segmentation(
    image_batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dtype = image_batch[0]["image"].dtype
    collated_images = torch.empty(
        (len(image_batch), *image_batch[0]["image"].shape), dtype=dtype
    )
    for i, item in enumerate(image_batch):
        collated_images[i] = item["image"]

    res: Dict[str, Any] = {
        "collated_images": collated_images,
    }

    return res
