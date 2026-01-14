# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from typing import Optional, Tuple, Union

import numpy as np
import torch.nn as nn

from spatialdino.utils.misc import make_3tuple


class PatchEmbed(nn.Module):
    """3D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]],
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ):
        super().__init__()
        img_size = make_3tuple(img_size)
        patch_size = make_3tuple(patch_size)
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = make_3tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, Z, H, W = x.shape

        assert Z % self.patch_size[0] == 0, (
            f"Input image depth {Z} is not a multiple of patch depth: {self.patch_size[0]}"
        )
        assert H % self.patch_size[1] == 0, (
            f"Input image height {H} is not a multiple of patch height {self.patch_size[1]}"
        )
        assert W % self.patch_size[2] == 0, (
            f"Input image width {W} is not a multiple of patch width: {self.patch_size[2]}"
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCZHW -> BNC
        x = self.norm(x)
        return x
