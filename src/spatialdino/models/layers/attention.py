# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
from typing import Literal
import warnings
import torch
import torch.nn as nn


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
if XFORMERS_ENABLED:
    try:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True

    except ImportError:
        raise ImportError("xFormers is not available (Attention)")
else:
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        vit_feat: str = "patch",
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # [3, B, H, N, C // H]

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if vit_feat == "attn":
            return attn[:, :, 0].transpose(1, 2)  # [B, N, H]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if vit_feat == "patch_attn":
            x = torch.cat([x, attn[:, :, 0].transpose(1, 2)], dim=-1)

        return x


class MemEffAttention(Attention):
    def forward(
        self,
        x: torch.Tensor,
        attn_bias=None,
        vit_feat: str = "patch",
    ) -> torch.Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, vit_feat=vit_feat)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)  # [B, N, H, C // H]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        if vit_feat == "attn" or vit_feat == "patch_attn":
            B, N, H, _ = q.shape
            x_a_cls = x[:, :1, :, :]
            cls_attn = x_a_cls.permute(0, 2, 1, 3) @ v.permute(0, 2, 3, 1)
            cls_attn = cls_attn.squeeze(2).permute(0, 2, 1)
            attn = cls_attn.reshape(B, N, H)
            if vit_feat == "attn":
                return attn

        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        if vit_feat == "patch_attn":
            x = torch.cat([x, attn], dim=-1)

        return x
