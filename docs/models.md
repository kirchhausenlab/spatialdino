# Model Architecture Reference

This document describes every neural network module in the `spatialdino` library,
how they compose into the full training and inference pipelines, and the key
design decisions behind each component.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Encoder (3D Vision Transformer)](#encoder-3d-vision-transformer)
3. [Patch Embedding](#patch-embedding)
4. [Positional Embeddings](#positional-embeddings)
5. [Transformer Block](#transformer-block)
6. [Attention](#attention)
7. [Feed-Forward Networks (MLP / SwiGLU)](#feed-forward-networks)
8. [Layer Scale & Drop Path](#layer-scale--drop-path)
9. [DINO Head](#dino-head)
10. [SSL (Self-Supervised Learning) Wrapper](#ssl-wrapper)
11. [Decoder (U-Net)](#decoder-u-net)
12. [LiFT (Learned image Feature Transform)](#lift)
13. [SinDer (Singular Defect Directions)](#sinder)
14. [Loss Functions](#loss-functions)
15. [Streaming Inference](#streaming-inference)

---

## Architecture Overview

```
Input Volume [B, C, Z, Y, X]
        |
        v
  +-----------+
  | PatchEmbed |  3D Conv -> [B, N_patches, embed_dim]
  +-----------+
        |
  + CLS token + positional encoding
        |
        v
  +-------------------+
  | Transformer Blocks | x depth (12 for ViT-S, 12 for ViT-B)
  | (Attention + FFN)  |
  +-------------------+
        |
        v
  +----------+       +----------+
  | DINO Head |       | iBOT Head|   (SSL training only)
  +----------+       +----------+
        |
        v
  +---------+
  | Decoder  |  Linear unpatchify -> U-Net segmentation
  +---------+
        |
        v
  Segmentation Map [B, 2, Z, Y, X]
```

---

## Encoder (3D Vision Transformer)

**Module**: `spatialdino.models.layers.encoder.Encoder`

The core backbone. Converts raw 3D volumetric data into dense feature
representations through a standard Vision Transformer pipeline adapted for 3D.

| Parameter | ViT-S/8 | ViT-B/8 |
|-----------|---------|---------|
| `embed_dim` | 384 | 768 |
| `depth` | 12 | 12 |
| `num_heads` | 6 | 12 |
| `patch_size` | (8,8,8) | (8,8,8) |
| `mlp_ratio` | 4.0 | 4.0 |

**Key methods**:

- `prepare_tokens_with_masks(x, masks)` -- Patch-embed the input, optionally
  replace masked patches with a learnable `mask_token`, prepend CLS token,
  add positional encoding.
- `forward_features(x, masks)` -- Full forward pass returning a dict with
  `x_norm_clstoken`, `x_norm_patchtokens`, `x_prenorm_clstoken`,
  `x_prenorm_patchtokens`, and `masks`.
- `predict(img, ...)` -- Inference-mode forward with AMP, returns
  `[B, C, Z', Y', X']` feature volume.

**Output dict keys**:
- `x_norm_clstoken` -- LayerNorm-ed CLS token `[B, D]`
- `x_norm_patchtokens` -- LayerNorm-ed patch tokens `[B, N, D]`
- `x_prenorm_clstoken` -- Pre-norm CLS token (for SSL objectives)
- `x_prenorm_patchtokens` -- Pre-norm patch tokens (for SSL objectives)

---

## Patch Embedding

**Module**: `spatialdino.models.layers.patch_embed.PatchEmbed`

Converts a 3D volume into a sequence of patch tokens via a single 3D
convolution.

```
Input:  [B, C, Z, Y, X]
Conv3d: kernel_size=patch_size, stride=stride
Output: [B, N, embed_dim]  where N = (Z/pz) * (Y/py) * (X/px)
```

- Supports overlapping patches when `stride < patch_size`.
- Optional `norm_layer` applied after projection (default: `nn.Identity`).

---

## Positional Embeddings

**Module**: `spatialdino.models.layers.pos_embed`

Three modes controlled by `pos_embed_type`:

| Mode | Description |
|------|-------------|
| `sincos` | 3D sinusoidal (fixed). Depth gets `D/2` dims; height and width each get `D/4`. |
| `learned` | Learnable parameters initialized with truncated normal `std=0.02`. |
| `none` | No positional encoding (relies on patch content only). |

For variable-size inputs at inference, the encoder's
`interpolate_pos_encoding` method uses **trilinear interpolation with
anti-aliasing** to resize the positional grid.

---

## Transformer Block

**Module**: `spatialdino.models.layers.block.Block` and `NestedTensorBlock`

Standard pre-norm transformer block:

```
x = x + DropPath(LayerScale(Attention(LayerNorm(x))))
x = x + DropPath(LayerScale(FFN(LayerNorm(x))))
```

`NestedTensorBlock` extends `Block` to handle **lists of tensors** with
different sequence lengths by using xFormers' `BlockDiagonalMask` for
efficient batched attention across global and local crops.

**Stochastic depth** is implemented in three regimes:
- `drop_path_rate > 0.1` -- optimized subset-sampling path
- `0 < drop_path_rate <= 0.1` -- standard DropPath
- `drop_path_rate == 0` -- deterministic residual

---

## Attention

**Module**: `spatialdino.models.layers.attention.Attention` and `MemEffAttention`

Standard multi-head self-attention with three output modes via `vit_feat`:

| `vit_feat` | Returns |
|------------|---------|
| `"patch"` | Standard attention output `[B, N, D]` |
| `"attn"` | CLS attention weights `[B, N, H]` |
| `"patch_attn"` | Concatenation of both `[B, N, D+H]` |

`MemEffAttention` uses xFormers' `memory_efficient_attention` for O(1) memory
in the attention computation. Falls back to vanilla attention when xFormers is
unavailable.

---

## Feed-Forward Networks

### MLP

**Module**: `spatialdino.models.layers.mlp.Mlp`

Two-layer MLP: `Linear -> GELU -> Dropout -> Linear -> Dropout`.
Hidden dimension = `embed_dim * mlp_ratio` (typically 4x).

### SwiGLU

**Module**: `spatialdino.models.layers.swiglu_ffn.SwiGLUFFN` and `SwiGLUFFNFused`

Gated linear unit with SiLU activation: `SiLU(W1 x) * W2 x`, followed by a
projection `W3`. `SwiGLUFFNFused` uses xFormers' fused kernel for better
throughput. Hidden dimension is rounded to multiples of 8 for hardware
efficiency.

---

## Layer Scale & Drop Path

### LayerScale

**Module**: `spatialdino.models.layers.layer_scale.LayerScale`

Per-channel learnable scaling `gamma * x`, initialized to a small value
(default `1e-5`) to stabilize early training of deep transformers.

### DropPath

**Module**: `spatialdino.models.layers.drop_path.DropPath`

Stochastic depth: randomly drops entire residual branches during training
with probability `drop_prob`. Applied per-sample (not per-token).

---

## DINO Head

**Module**: `spatialdino.models.layers.dino_head.DINOHead`

Projection head for self-supervised contrastive learning:

```
Linear -> GELU -> Linear -> GELU -> Linear -> L2-Normalize -> WeightNorm(Linear)
```

- 3-layer MLP with `hidden_dim=2048`, `bottleneck_dim=256`.
- L2 normalization before the final weight-normed projection.
- Output dimension = `n_prototypes` (default 32768).
- Casts to float32 via `@custom_fwd` for numerical stability.

---

## SSL Wrapper

**Module**: `spatialdino.models.ssl.SSL`

Orchestrates the student-teacher self-supervised training loop:

- **Student**: `Encoder` + `DINOHead` (CLS) + `DINOHead` (patch/iBOT)
- **Teacher**: Identical architecture, **no gradients**, updated via EMA

**Forward path (student)**:
1. Process global and local crops through the encoder
2. DINO head on CLS tokens (both crops)
3. iBOT head on masked patch tokens (global crops only)

**Forward path (teacher)**:
1. Process global crops through the teacher encoder
2. Flip global crops for cross-assignment (crop A matched to B)
3. Apply centering (Sinkhorn-Knopp or mean centering) + temperature scaling
4. iBOT head on masked patches

**EMA update**: `teacher = m * teacher + (1 - m) * student`
where momentum `m` ramps from 0.994 to 1.0 during training.

---

## Decoder (U-Net)

**Module**: `spatialdino.models.layers.decoder.Decoder`

Reconstructs volumetric data from ViT patch tokens for downstream
segmentation.

**Pipeline**:
1. `decoder_pred`: Linear projection `[embed_dim -> pz*py*px*C]`
2. `unpatchify`: Rearrange patches back to volume `[B, C, Z, Y, X]`
3. `UNet`: 3D U-Net with GroupNorm for binary segmentation

The U-Net (`spatialdino.models.layers.unet.UNet`) uses:
- 3 encoder levels: `[64, 128, 256]` channels
- 1 bottleneck at 512 channels
- Skip connections via concatenation
- GroupNorm (4 groups) instead of BatchNorm
- Dropout = 0.65 (regularization for small datasets)

**Output dict**:
- `decoder_recon` -- Reconstructed volume `[B, C, Z, Y, X]`
- `decoder_ncuts` -- Segmentation probabilities `[B, 2, Z, Y, X]`

The `predict` method wraps inference with MONAI's `SlidingWindowInferer`
for processing volumes larger than training crops.

---

## LiFT

**Module**: `spatialdino.models.lift.model.lift_model.LiFT`

Learned image Feature Transform -- upsamples ViT patch features to a
higher-resolution feature map by fusing with multi-scale image features.

```
Image [B,C,Z,Y,X] --> Conv layers --> latent_1 (stride-4) + latent_2 (stride-8)
ViT features [B,D,z,y,x] + latent_2 --> ConvTranspose3d --> concat with latent_1 --> Conv --> output
```

- `image_convs_1`: Two strided Conv3d layers (stride 2 each) with BatchNorm + GELU
- `scale_adapter`: Adapts spatial resolution when patch_size != 8
- `image_convs_2`: One more strided Conv3d layer
- `Up` block: ConvTranspose3d + skip connection + DoubleConv

Output: `[B, embed_dim, 2z, 2y, 2x]` -- 2x spatial upsampling from patch grid.

---

## SinDer

**Module**: `spatialdino.models.sinder.singular_defect`

**Sin**gular **D**efect di**r**ections analysis. Computes the dominant
singular vectors of the composed linear approximations of each transformer
block to identify "defect" directions -- rank-1 structures that emerge in
the representation space.

**Key functions**:
- `anomaly_dir_attn(blk)` -- SVD of the attention residual branch:
  `A = LayerScale * Proj * V_proj * LayerNorm`
- `anomaly_dir_mlp_ls(blk)` -- Least-squares linear approximation of the
  MLP branch, then SVD
- `anomaly_dir(blk)` -- Composes both branches: `C @ A` where C is the FFN
  approximation and A is the attention approximation
- `singular_defect_directions(model)` -- Accumulates block-wise compositions
  across all layers, returning per-layer dominant singular vectors

Used for model interpretability and feature repair
(`spatialdino.models.sinder.repair`).

---

## Loss Functions

**Module**: `spatialdino.loss`

| Loss | Module | Description |
|------|--------|-------------|
| DINO CLS | `dino_clstoken_loss` | Cross-entropy between student/teacher CLS projections with centering |
| iBOT Patch | `ibot_patch_loss` | Cross-entropy on masked patch tokens |
| KoLeo | `koleo_loss` | Kozachenko-Leonenko entropy regularizer for uniform feature coverage |
| Charbonnier | `charbonnier_loss` | Smooth L1-like pixel reconstruction loss |
| Fourier | `fourier_loss` | Frequency-domain reconstruction loss |
| Soft NCuts | `soft_ncuts` | Differentiable normalized cuts for unsupervised segmentation |

---

## Streaming Inference

**Module**: `spatialdino.inference.streaming`

For volumes too large to fit in GPU memory:

- `StreamingEncoder` -- Processes the volume in overlapping sliding windows,
  accumulates patch features into a pre-allocated output tensor
- `TritonKernels` -- Custom Triton kernels for efficient patch-level
  operations during streaming
- `Storage` -- Manages memory-mapped or CPU-backed tensors for accumulation

The streaming pipeline enables inference on arbitrarily large volumes
(tested up to 2048^3 voxels) with bounded GPU memory usage.
