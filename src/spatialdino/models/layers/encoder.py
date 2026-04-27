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
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from spatialdino.data.transforms import trilinear_interpolate_with_antialias
from spatialdino.models.layers.attention import MemEffAttention
from spatialdino.models.layers.block import NestedTensorBlock as Block
from spatialdino.models.layers.mlp import Mlp
from spatialdino.models.layers.patch_embed import PatchEmbed
from spatialdino.models.layers.pos_embed import get_3d_sincos_pos_embed
from spatialdino.models.layers.rope import RoPE3D, RoPECache, build_3d_rope_cache
from spatialdino.models.layers.swiglu_ffn import XFORMERS_AVAILABLE as _SWIGLU_XFORMERS
from spatialdino.models.layers.swiglu_ffn import SwiGLUFFN

if _SWIGLU_XFORMERS:
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
        num_register_tokens: int = 0,
        num_tt_register_tokens: int = 0,
        interpolate_offset: float = 0.1,
        interpolate_antialias: bool = True,
        interpolate_align_corners: bool = True,
        pos_embed_type: Literal["sincos", "learned", "rope", "none"] = "none",
        rope_theta: float = 10000.0,
        rope_normalize_coords: bool = False,
        rope_coord_shift: Optional[float] = None,
        rope_coord_jitter: Optional[float] = None,
        rope_coord_rescale: Optional[float] = None,
        rope_drop_prob: float = 0.0,
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
        self.num_register_tokens = num_register_tokens
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

        # Learnable register tokens (DINOv2-reg) — participate in training & inference
        assert self.num_register_tokens >= 0
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, self.embed_dim)
            )
        else:
            self.register_tokens = None

        # Test-time register tokens (non-learnable) — inference only
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
        elif ffn_layer == "swiglufused":
            if _SWIGLU_XFORMERS:
                logger.info("using SwiGLU (xFormers fused) layer as FFN")
                ffn_layer = SwiGLUFFNFused
            else:
                logger.warning(
                    "xFormers not available, falling back to pure-PyTorch SwiGLU"
                )
                ffn_layer = SwiGLUFFN
        elif ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFN
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        self.num_heads = num_heads

        # transformer blocks - attention + FFN with residual connections
        self.blocks = nn.ModuleList(
            [
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
            ]
        )

        self.pos_embed_type = pos_embed_type
        self.use_pos_embed = pos_embed_type not in ("none", "rope")
        self.use_rope = pos_embed_type == "rope"

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
        elif self.pos_embed_type == "rope":
            logger.info("using 3D rotary position embedding (RoPE)")
            self.pos_embed = None
            head_dim = embed_dim // num_heads
            self.rope = RoPE3D(
                head_dim=head_dim,
                theta=rope_theta,
                normalize_coords=rope_normalize_coords,
                coord_shift=rope_coord_shift,
                coord_jitter=rope_coord_jitter,
                coord_rescale=rope_coord_rescale,
                drop_prob=rope_drop_prob,
            )
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
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(self._init_weights)

    def _ensure_tt_registers_supported(self) -> None:
        if self.num_tt_register_tokens:
            raise ValueError(
                "Test-time register tokens are only supported in predict() / streaming inference."
            )

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

        # Skip interpolation entirely at training resolution — return raw weights
        if (z0, h0, w0) == self.grid_size:
            return pos_embed.to(previous_dtype)

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

    def _build_rope_cache(
        self,
        Z: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        num_prefix_tokens: int = 1,
    ) -> Optional[RoPECache]:
        """Build a 3D RoPE frequency table via the :class:`RoPE3D` module.

        Built in the compute dtype (e.g. bfloat16 under autocast) so the cache
        already holds the correct dtype and no cast is needed at apply time.
        Caching, train/eval mode, and coordinate augmentation are handled
        automatically by the module's ``forward()`` method.
        """
        if not self.use_rope:
            return None
        z0 = 1 + (Z - self.patch_size[0]) // self.stride[0]
        h0 = 1 + (H - self.patch_size[1]) // self.stride[1]
        w0 = 1 + (W - self.patch_size[2]) // self.stride[2]

        return self.rope(
            grid_size=(z0, h0, w0),
            num_prefix_tokens=num_prefix_tokens,
            device=device,
            dtype=dtype,
        )

    def clear_rope_cache(self) -> None:
        """Drop all cached RoPE tables (e.g. after ``.to(device)`` or dtype change)."""
        if self.use_rope:
            self.rope.clear_cache()

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

        # Insert learnable register tokens after CLS, before patches
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(
        self,
        x_list: List[torch.Tensor],
        masks_list: List[torch.Tensor],
        equiv_share_rope: bool = False,
    ) -> List[Dict[str, Any]]:
        """_summary_
        - Process list of tensors through transformer pipeline
        - Each tensor: raw volumetric data [B, C, Z, Y, X] + corresponding masks
        - Convert to patch tokens, add CLS/register tokens, apply positional encoding
        - Pass through 12 transformer blocks with self-attention and FFN
        - Return both normalized and pre-normalized outputs for different use cases
        - Supports batch processing of multiple volumes simultaneously

        When ``equiv_share_rope`` is True, the last crop in ``x_list`` reuses the
        exact RoPE cache built for the first crop. This is required by the
        rotation-equivariance loss: the first global crop and the rotated
        equivariance crop must see identical positional encodings so the cosine
        comparison only measures rotation error, not augmentation drift.
        """
        self._ensure_tt_registers_supported()
        x = [
            self.prepare_tokens_with_masks(xi, masks)
            for xi, masks in zip(x_list, masks_list)
        ]

        # Build per-crop RoPE caches (None list if RoPE is disabled)
        # Register tokens are prefix tokens that get identity rotation
        num_prefix = 1 + self.num_register_tokens
        rope_list: Optional[List[RoPECache]] = None
        if self.use_rope:
            rope_list = []
            for i, xi in enumerate(x_list):
                if (
                    equiv_share_rope
                    and i == len(x_list) - 1
                    and len(x_list) >= 2
                    and xi.shape[2:] == x_list[0].shape[2:]
                ):
                    rope_list.append(rope_list[0])
                    continue
                rope_list.append(
                    self._build_rope_cache(
                        xi.shape[2],
                        xi.shape[3],
                        xi.shape[4],
                        device=x[0].device,
                        dtype=x[0].dtype,
                        num_prefix_tokens=num_prefix,
                    )
                )

        for blk in self.blocks:
            x = blk(x, rope=rope_list)

        all_x = x
        output = []
        # Skip CLS + register tokens to get patch tokens
        patch_start = 1 + self.num_register_tokens
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1:patch_start],
                    "x_norm_patchtokens": x_norm[:, patch_start:],
                    "x_prenorm_clstoken": x[:, 0],
                    "x_prenorm_regtokens": x[:, 1:patch_start],
                    "x_prenorm_patchtokens": x[:, patch_start:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Union[torch.Tensor, List[torch.Tensor], None] = None,
        equiv_share_rope: bool = False,
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
        self._ensure_tt_registers_supported()
        if isinstance(x, list):
            return self.forward_features_list(x, masks, equiv_share_rope=equiv_share_rope)

        Z, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = self.prepare_tokens_with_masks(x, masks)
        num_prefix = 1 + self.num_register_tokens
        rope = self._build_rope_cache(
            Z, H, W, device=x.device, dtype=x.dtype, num_prefix_tokens=num_prefix
        )

        for blk in self.blocks:
            x = blk(x, rope=rope)

        patch_start = 1 + self.num_register_tokens
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:patch_start],
            "x_norm_patchtokens": x_norm[:, patch_start:],
            "x_prenorm_clstoken": x[:, 0],
            "x_prenorm_regtokens": x[:, 1:patch_start],
            "x_prenorm_patchtokens": x[:, patch_start:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        masks: Union[torch.Tensor, List[torch.Tensor], None] = None,
        equiv_share_rope: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """_summary_
        - If a list, call the forward_features_list method.
        - If a single tensor, call the forward_features method.
        - return the tensor.
        """
        return self.forward_features(x, masks, equiv_share_rope=equiv_share_rope)

    def _predict(
        self,
        x: torch.Tensor,
        vit_feat: Literal["patch", "patch_attn", "attn"] = "patch_attn",
        norm_feat: Literal["prenorm", "norm"] = "prenorm",
        return_special_tokens: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
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

        # Insert learnable register tokens (always present if configured)
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        # Insert test-time register tokens (inference-only, non-learnable)
        if self.tt_register_tokens is not None:
            insert_pos = 1 + self.num_register_tokens
            x = torch.cat(
                (
                    x[:, :insert_pos],
                    self.tt_register_tokens.expand(x.shape[0], -1, -1),
                    x[:, insert_pos:],
                ),
                dim=1,
            )

        # Build RoPE cache (CLS + all register tokens get identity rotation)
        num_prefix = 1 + self.num_register_tokens + self.num_tt_register_tokens
        rope = self._build_rope_cache(
            Z, Y, X, device=x.device, dtype=x.dtype, num_prefix_tokens=num_prefix
        )

        # Run through model, at the last block just return the attention.
        for i in range(len(self.blocks) - 1):
            x = self.blocks[i](x, rope=rope)

        res = self.blocks[-1](x, vit_feat=vit_feat, rope=rope)

        if norm_feat == "norm":
            if vit_feat == "patch":
                res = self.norm(res)
            elif vit_feat == "patch_attn":
                res[..., : -self.num_heads] = self.norm(res[..., : -self.num_heads])

        # Optionally capture CLS and register features before stripping.
        # Always slice only the embed_dim feature dims (safe for both "patch" and
        # "patch_attn" layouts where res may have extra attention columns appended).
        if return_special_tokens:
            special = res[0, :num_prefix, : self.embed_dim]  # [num_prefix, D]
            cls_token = special[0]  # [D]
            num_regs = self.num_register_tokens + self.num_tt_register_tokens
            register_tokens: Optional[torch.Tensor] = (
                special[1 : 1 + num_regs] if num_regs > 0 else None
            )

        # Strip CLS + all register tokens to get only patch tokens
        res = res[:, num_prefix:].transpose(1, 2)  # [B, C, L]

        res = res.reshape(
            1,
            -1,
            z0,
            y0,
            x0,
        )  # [B, C, Z, Y, X]

        if return_special_tokens:
            return res, cls_token, register_tokens
        return res

    def predict(
        self,
        img: torch.Tensor,
        vit_feat: Literal["patch", "patch_attn", "attn"] = "patch_attn",
        norm_feat: Literal["prenorm", "norm"] = "prenorm",
        dtype: torch.dtype = torch.bfloat16,
        use_amp: bool = True,
        device_type: str = "cuda",
        device: torch.device | str | None = None,
        return_special_tokens: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        with torch.no_grad():
            with torch.amp.autocast(  # type: ignore
                enabled=use_amp,
                dtype=dtype,
                device_type=device_type,
            ):
                result = self._predict(
                    img.to(device),
                    vit_feat=vit_feat,
                    norm_feat=norm_feat,
                    return_special_tokens=return_special_tokens,
                )

        return result
