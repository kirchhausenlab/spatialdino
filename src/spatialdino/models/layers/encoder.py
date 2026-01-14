# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import logging
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers.utils import sliding_window_inference
from monai.data.utils import get_valid_patch_size
from spatialdino.data.transforms import trilinear_interpolate_with_antialias
from spatialdino.models.layers.attention import MemEffAttention
from spatialdino.models.layers.block import NestedTensorBlock as Block
from spatialdino.models.layers.mlp import Mlp
from spatialdino.models.layers.patch_embed import PatchEmbed
from spatialdino.models.layers.pos_embed import get_3d_sincos_pos_embed
from spatialdino.models.layers.swiglu_ffn import SwiGLUFFNFused
from spatialdino.utils.misc import make_3tuple

logger = logging.getLogger("pretrain")


class Encoder(nn.Module):
    """_summary_
    - 3D Vision Transformer encoder for volumetric data processing
    - Converts raw volumes to patch tokens via 3D convolution
    - 12-layer transformer with multi-head attention and FFN
    - Supports sliding window inference for large volumes
    - Handles positional encoding interpolation for variable input sizes
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: Optional[
            float
        ] = None,  # for layerscale: None or 0 => no layerscale
        embed_layer: nn.Module = PatchEmbed,
        act_layer: nn.Module = nn.GELU,
        block_fn: nn.Module = partial(Block, attn_class=MemEffAttention),
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        ffn_layer: str = "mlp",
        num_tt_register_tokens: int = 0,
        interpolate_offset: float = 0.1,
        interpolate_antialias: bool = True,
        interpolate_align_corners: bool = True,
        pos_embed_type: Literal["sincos", "learned", "none"] = "none",
    ) -> None:
        super().__init__()

        # store interpolation settings for positional encoding
        self.in_chans = in_chans
        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_align_corners = interpolate_align_corners

        # convert sizes to 3D tuples, default stride = patch_size (non-overlapping)
        patch_size = make_3tuple(patch_size)
        if stride is None:
            self.stride = patch_size
        else:
            self.stride = make_3tuple(stride)
        img_size = make_3tuple(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_tokens = 1  # CLS token
        self.num_tt_register_tokens = num_tt_register_tokens
        # calculate patch grid dimensions
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )

        self.embed_dim = embed_dim
        self.n_blocks = depth

        # learnable mask token for masked autoencoding
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))

        # 3D patch embedding layer - converts volume patches to embeddings
        self.patch_embed = embed_layer(
            img_size, patch_size, stride, in_chans, embed_dim
        )  # these are needed regardless of the patch sampling method
        self.num_patches = self.patch_embed.num_patches

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        assert self.num_tt_register_tokens >= 0
        if self.num_tt_register_tokens:
            self.register_buffer(
                "tt_register_tokens",
                torch.zeros(1, self.num_tt_register_tokens, self.embed_dim),
            )
        else:
            self.tt_register_tokens = None

        # stochastic depth - either uniform or linearly increasing
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        # FFN layer selection - MLP, SwiGLU, or Identity
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        self.num_heads = num_heads

        # transformer blocks - attention + FFN with residual connections
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(self.n_blocks)
        ])

        self.pos_embed_type = pos_embed_type
        self.use_pos_embed = pos_embed_type != "none"

        if self.pos_embed_type == "sincos":
            # use sincos positional encoding
            logger.info("using sincos positional encoding")
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim),
                requires_grad=False,
            )
            sincos = get_3d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                grid_depth=self.grid_size[0],
                grid_height=self.grid_size[1],
                grid_width=self.grid_size[2],
                cls_token=True,
            )
            self.pos_embed.data.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
        elif self.pos_embed_type == "learned":
            # use learnable positional embeddings
            logger.info("using learned positional encoding")
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # no positional encoding
            logger.info("no positional encoding")
            self.pos_embed = None

        # final layer norm
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """_summary_
        - Initialize positional embeddings with truncated normal
        - CLS and register tokens with small normal distribution
        - Apply weight init to all modules recursively
        """
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """_summary_
        - Standard ViT weight initialization
        - Linear layers: truncated normal weights, zero biases
        - Prevents vanishing/exploding gradients in deep transformer
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(
        self, pos_embed: nn.Parameter, x: torch.Tensor, z: int, h: int, w: int
    ) -> torch.Tensor:
        """_summary_
        - Interpolate positional encodings for variable input sizes
        - Separates CLS token encoding from patch encodings
        - Uses trilinear interpolation with optional antialiasing
        - Calculates target patch grid size based on input dimensions
        - Essential for handling different volume sizes than training
        """
        previous_dtype = x.dtype

        # work in float32 for numerical stability
        pos_embed = pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        # Calculate dimensions in patch space - how many patches we'll get
        z0 = 1 + (z - self.patch_size[0]) // self.stride[0]
        h0 = 1 + (h - self.patch_size[1]) // self.stride[1]
        w0 = 1 + (w - self.patch_size[2]) // self.stride[2]

        kwargs = {}
        if self.interpolate_offset:
            # Apply offset to all dimensions
            sz, sy, sx = (
                float(z0 + self.interpolate_offset) / self.grid_size[0],
                float(h0 + self.interpolate_offset) / self.grid_size[1],
                float(w0 + self.interpolate_offset) / self.grid_size[2],
            )
            kwargs["scale_factor"] = (sz, sy, sx)
        else:
            kwargs["size"] = (z0, h0, w0)

        # Reshape to 3D and interpolate
        patch_pos_embed = trilinear_interpolate_with_antialias(
            patch_pos_embed.reshape(
                1, self.grid_size[0], self.grid_size[1], self.grid_size[2], dim
            ).permute(0, 4, 1, 2, 3),
            antialias=self.interpolate_antialias,  # from https://arxiv.org/pdf/2309.16588
            align_corners=self.interpolate_align_corners,  # from https://github.com/facebookresearch/dinov2/issues/468
            **kwargs,
        )

        assert (z0, h0, w0) == patch_pos_embed.shape[-3:], "Output size mismatch"
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """_summary_
        - Take raw volumetric data [B, C, Z, Y, X], divide it into non overlapping patches by calling the patch_embed.py (src/spatialdino/models/layers/patch_embed.py) class.
        - The forward method passes the tensor into a 3D conv layer, which outputs [B, N_patches, embed_dim], N Patches determined by a grid_size. Norm set to true by default.
        - if the mask is proviced, replace it with a learnable mask token.
        - add a CLS token and interpolate the positional encoding to the patch tokens.
        - if register tokens are provided, add them to the tensor.
        - return the tensor.
        """
        B, C, Z, H, W = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.use_pos_embed:
            x = x + self.interpolate_pos_encoding(self.pos_embed, x, Z, H, W)

        return x

    def forward_features_list(
        self, x_list: List[torch.Tensor], masks_list: List[torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """_summary_
        - Process list of tensors through transformer pipeline
        - Each tensor: raw volumetric data [B, C, Z, Y, X] + corresponding masks
        - Convert to patch tokens, add CLS/register tokens, apply positional encoding
        - Pass through 12 transformer blocks with self-attention and FFN
        - Return both normalized and pre-normalized outputs for different use cases
        - Supports batch processing of multiple volumes simultaneously
        """
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append({
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_patchtokens": x_norm[:, 1:],
                "x_prenorm_clstoken": x[:, 0],
                "x_prenorm_patchtokens": x[:, 1:],
                "x_prenorm": x,
                "masks": masks,
            })
        return output

    def forward_features(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Union[torch.Tensor, List[torch.Tensor], None] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """_summary_
        - Main feature extraction method - handles both single tensors and lists
        - If list input: delegates to forward_features_list for batch processing
        - If single tensor: processes through patch embedding → transformer → normalization
        - Returns comprehensive output dict with different token types and normalization states

        **Prenorm**
        for SSL learning objectives
        **Norm**
        for downstream tasks of classification, segmentation, etc.
        """
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm_clstoken": x[:, 0],
            "x_prenorm_patchtokens": x[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Union[torch.Tensor, List[torch.Tensor], None] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """_summary_
        - If a list, call the forward_features_list method.
        - If a single tensor, call the forward_features method.
        - return the tensor.
        """
        return self.forward_features(x, masks)

    def _predict(
        self,
        x: torch.Tensor,
        vit_feat: Literal["patch", "patch_attn", "attn"] = "patch_attn",
        norm_feat: Literal["prenorm", "norm"] = "prenorm",
    ) -> torch.Tensor:
        # assert isinstance(x, torch.Tensor)

        Z, Y, X = x.shape[-3:]

        z0, y0, x0 = (
            1 + (Z - self.patch_size[0]) // self.stride[0],
            1 + (Y - self.patch_size[1]) // self.stride[1],
            1 + (X - self.patch_size[2]) // self.stride[2],
        )

        x = self.patch_embed(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.use_pos_embed:
            x = x + self.interpolate_pos_encoding(self.pos_embed, x, Z, Y, X)

        if self.tt_register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.tt_register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        # Run through model, at the last block just return the attention.
        for i in range(len(self.blocks) - 1):
            x = self.blocks[i](x)

        res = self.blocks[-1](x, vit_feat=vit_feat)

        if norm_feat == "norm":
            if vit_feat == "patch":
                res = self.norm(res)
            elif vit_feat == "patch_attn":
                res[..., : -self.num_heads] = self.norm(res[..., : -self.num_heads])

        res = res[:, 1 + self.num_tt_register_tokens :].transpose(1, 2)  # [B, C, L]

        res = res.reshape(
            1,
            -1,
            z0,
            y0,
            x0,
        )  # [B, C, Z, Y, X]

        return res

    def predict(
        self,
        img: torch.Tensor,
        vit_feat: Literal["patch", "patch_attn", "attn"] = "patch_attn",
        norm_feat: Literal["prenorm", "norm"] = "prenorm",
        dtype: torch.dtype = torch.bfloat16,
        use_amp: bool = True,
        device_type: str = "cuda",
        **kwargs: Any,
    ) -> torch.Tensor:
        with torch.no_grad():
            with torch.amp.autocast(  # type: ignore
                enabled=use_amp,
                dtype=dtype,
                device_type=device_type,
            ):
                num_spatial_dims = len(img.shape) - 2
                image_size_ = img.shape[-3:]
                roi_size = kwargs["roi_size"]
                image_size = tuple(
                    max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims)
                )
                patch_size = get_valid_patch_size(image_size, roi_size)
                lr_feats = convert_to_tensor(
                    sliding_window_inference(
                        inputs=img,
                        predictor=lambda x: self._predict(
                            x,
                            vit_feat=vit_feat,
                            norm_feat=norm_feat,
                        ),
                        mode="constant",
                        roi_weight_map=torch.ones(
                            patch_size, device=kwargs["device"], dtype=torch.float32
                        ),
                        **kwargs,
                    )
                )

        return lr_feats
