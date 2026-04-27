"""3D Rotary Position Embedding (RoPE) for volumetric Vision Transformers.

Splits head dimensions across Z, Y, X spatial axes and applies axis-specific
rotary frequencies.  Prefix tokens (CLS, register) receive identity rotation
so they remain position-agnostic.

Supports two coordinate modes:

* **Raw positions** (``normalize_coords=False``, legacy): integer indices
  ``0, 1, …, N-1`` with standard RoPE theta-based frequencies.
* **Normalized coordinates** (``normalize_coords=True``): cell-centered values
  in ``[-1, +1]`` with ``2π / θ^(i/n)`` frequencies (DiNOv3-style).  This makes
  the frequency spectrum independent of grid size — critical for SSL with
  variable crop sizes.

Training-time coordinate augmentations (*coord_shift*, *coord_jitter*,
*coord_rescale*) can further reduce the model's reliance on absolute position.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn

RoPECache = Tuple[torch.Tensor, torch.Tensor]  # (cos, sin)


def build_3d_rope_cache(
    grid_size: Tuple[int, int, int],
    head_dim: int,
    num_prefix_tokens: int = 1,
    theta: float = 10000.0,
    normalize_coords: bool = False,
    coord_shift: float | None = None,
    coord_jitter: float | None = None,
    coord_rescale: float | None = None,
    drop_prob: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> RoPECache:
    """Build cos/sin frequency tables for 3D Rotary Position Embedding.

    The *head_dim* is halved (RoPE rotates pairs), then split across Z, Y, X.
    Any remainder pairs go to the X axis.

    With raw positions (``normalize_coords=False``), ``freq = 1 / θ^(i/n)``
    (standard RoPE).  With normalized coords, ``freq = 2π / θ^(i/n)`` where
    theta acts as the period base (analogous to DiNOv3's ``base`` — a value
    of ~100 is typical for microscopy data).

    Args:
        grid_size: (grid_z, grid_y, grid_x) patch grid dimensions.
        head_dim: per-head dimension (must be even).
        num_prefix_tokens: number of non-spatial prefix tokens (CLS, registers)
            that receive identity rotation.
        theta: frequency base.  With raw positions this is the standard RoPE
            theta; with normalized coords it serves as the period base (like
            DiNOv3's ``base`` — use ~100 for microscopy data).
        normalize_coords: map positions to cell-centered coords in ``[-1, +1]``.
        coord_shift: if not None, add a uniform random shift in
            ``[-coord_shift, +coord_shift]`` per axis (training augmentation).
        coord_jitter: if not None, multiply coords by a log-uniform random
            factor in ``[1/coord_jitter, coord_jitter]`` per axis.
        coord_rescale: if not None, multiply all coords by a single
            log-uniform random factor in ``[1/coord_rescale, coord_rescale]``.
        drop_prob: probability of zeroing all RoPE angles for a patch token
            (identity rotation → no position info).  Applied before prefix
            tokens are prepended, so CLS/registers are never dropped.

    Returns:
        (cos, sin) each of shape ``[num_prefix_tokens + num_patches, head_dim // 2]``.
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    half_dim = head_dim // 2
    grid_z, grid_y, grid_x = grid_size

    # Split frequency pairs evenly across the three spatial axes.
    # Remainder pairs go to Z, then Y (round-robin) to avoid over-representing
    # any single axis — the previous X-heavy split caused PCA artifacts.

    pairs_per_axis = half_dim // 3
    remainder = half_dim - 3 * pairs_per_axis

    # pairs_z = half_dim //3 # Using old method for consistency with previous runs; 
    # pairs_y = half_dim //3
    # pairs_x = half_dim - pairs_z - pairs_y

    pairs_z = pairs_per_axis + (1 if remainder > 0 else 0)
    pairs_y = pairs_per_axis + (1 if remainder > 1 else 0)
    pairs_x = pairs_per_axis

    def _freqs(n_pairs: int) -> torch.Tensor:
        t = torch.arange(n_pairs, device=device, dtype=dtype)
        if normalize_coords:
            # Theta-as-base with normalized coords (DiNOv3 style):
            # periods = theta^(i/n), freqs = 2π / period
            periods = theta ** (t / n_pairs)
            return 2.0 * math.pi / periods
        else:
            # Original RoPE theta-based with raw integer positions
            return 1.0 / (theta ** (t / n_pairs))

    freqs_z = _freqs(pairs_z)
    freqs_y = _freqs(pairs_y)
    freqs_x = _freqs(pairs_x)

    # Position coordinates
    if normalize_coords:
        # Cell-centered coords in [-1, +1], independent of grid size
        pos_z = (
            torch.arange(grid_z, device=device, dtype=dtype) + 0.5
        ) / grid_z * 2 - 1
        pos_y = (
            torch.arange(grid_y, device=device, dtype=dtype) + 0.5
        ) / grid_y * 2 - 1
        pos_x = (
            torch.arange(grid_x, device=device, dtype=dtype) + 0.5
        ) / grid_x * 2 - 1
    else:
        pos_z = torch.arange(grid_z, device=device, dtype=dtype)
        pos_y = torch.arange(grid_y, device=device, dtype=dtype)
        pos_x = torch.arange(grid_x, device=device, dtype=dtype)

    # Meshgrid → flatten to (num_patches, 3)
    mesh_z, mesh_y, mesh_x = torch.meshgrid(pos_z, pos_y, pos_x, indexing="ij")
    coords = torch.stack(
        [mesh_z.reshape(-1), mesh_y.reshape(-1), mesh_x.reshape(-1)], dim=1
    )  # [num_patches, 3]

    # Training-time coordinate augmentations
    if coord_shift is not None:
        shift = torch.empty(3, device=device, dtype=dtype).uniform_(
            -coord_shift, coord_shift
        )
        coords = coords + shift[None, :]

    if coord_jitter is not None:
        jitter_max = math.log(coord_jitter)
        jitter = (
            torch.empty(3, device=device, dtype=dtype)
            .uniform_(-jitter_max, jitter_max)
            .exp()
        )
        coords = coords * jitter[None, :]

    if coord_rescale is not None:
        rescale_max = math.log(coord_rescale)
        rescale = (
            torch.empty(1, device=device, dtype=dtype)
            .uniform_(-rescale_max, rescale_max)
            .exp()
        )
        coords = coords * rescale

    # Outer products → angles for each axis, then concatenate
    angles = torch.cat(
        [
            coords[:, 0:1] * freqs_z[None, :],
            coords[:, 1:2] * freqs_y[None, :],
            coords[:, 2:3] * freqs_x[None, :],
        ],
        dim=1,
    )  # [num_patches, half_dim]

    # RoPE angle dropout: zero all angles for a random subset of patch tokens,
    # giving them identity rotation (cos=1, sin=0 → no position info).
    if drop_prob > 0.0:
        keep = torch.bernoulli(
            torch.full((angles.shape[0], 1), 1.0 - drop_prob, device=device, dtype=dtype)
        )
        angles = angles * keep

    # Prefix tokens (CLS / registers) get identity rotation (angle = 0)
    if num_prefix_tokens > 0:
        prefix = torch.zeros(num_prefix_tokens, half_dim, device=device, dtype=dtype)
        angles = torch.cat([prefix, angles], dim=0)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to *x*.

    Uses the half-split convention: the first ``D//2`` dims pair with the last
    ``D//2`` dims.

    Args:
        x: tensor of shape ``[..., D]`` (any leading dims).
        cos, sin: broadcastable to ``[..., D//2]``.

    Returns:
        Rotated tensor with the same shape as *x*.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    x_out = torch.empty_like(x)
    x_out[..., :half] = x1 * cos - x2 * sin
    x_out[..., half:] = x1 * sin + x2 * cos
    return x_out


def concat_rope_for_nested(
    rope_list: list[RoPECache],
    batch_sizes: list[int],
    seq_lens: list[int],
) -> RoPECache:
    """Tile and concatenate per-crop RoPE caches to match ``get_attn_bias_and_cat``.

    Each crop's ``(cos, sin)`` of shape ``[N_i, D]`` is repeated ``B_i`` times
    and the results are concatenated along the token dimension.
    """
    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for (cos_i, sin_i), bs, N in zip(rope_list, batch_sizes, seq_lens):
        # cos_i may have been built for a different N (e.g. with fewer prefix
        # tokens) — slice to the actual seq len used in the concatenated tensor.
        cos_i = cos_i[:N]
        sin_i = sin_i[:N]
        # Tile for each batch element: [N, D] → [B*N, D]
        cos_parts.append(
            cos_i.unsqueeze(0).expand(bs, -1, -1).reshape(-1, cos_i.shape[-1])
        )
        sin_parts.append(
            sin_i.unsqueeze(0).expand(bs, -1, -1).reshape(-1, sin_i.shape[-1])
        )
    return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)


class RoPE3D(nn.Module):
    """3D Rotary Position Embedding module with train-time augmentation.

    Wraps :func:`build_3d_rope_cache` into an ``nn.Module`` so that:

    * **Training** — coordinate augmentations (shift / jitter / rescale) produce
      fresh stochastic embeddings every forward call.
    * **Eval** — augmentations are disabled and results are cached by
      ``(grid, num_prefix, dtype)`` for zero-cost reuse.

    The module carries no learnable parameters and excludes the cache from
    ``state_dict`` (it's deterministic given the config).
    """

    def __init__(
        self,
        head_dim: int,
        *,
        theta: float = 10000.0,
        normalize_coords: bool = False,
        coord_shift: float | None = None,
        coord_jitter: float | None = None,
        coord_rescale: float | None = None,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.normalize_coords = normalize_coords
        self.coord_shift = coord_shift
        self.coord_jitter = coord_jitter
        self.coord_rescale = coord_rescale
        self.drop_prob = drop_prob
        # Eval-mode cache: maps (grid, num_prefix, dtype) → RoPECache
        self._cache: dict[tuple, RoPECache] = {}

    def forward(
        self,
        grid_size: Tuple[int, int, int],
        num_prefix_tokens: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> RoPECache:
        """Return ``(cos, sin)`` for the given grid.

        During training with augmentation enabled, a fresh stochastic cache is
        built each call.  During eval (or when no augmentation is configured),
        the result is cached by ``(grid_size, num_prefix_tokens, dtype)``.
        """
        augmenting = self.training and any(
            [
                self.coord_shift,
                self.coord_jitter,
                self.coord_rescale,
                self.drop_prob,
            ]
        )

        # Resolve None → default device/dtype so cache lookups are consistent
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        key = (*grid_size, num_prefix_tokens, dtype)
        if not augmenting:
            cached = self._cache.get(key)
            if cached is not None:
                # Move to requested device if needed (e.g. after .cuda())
                if cached[0].device != device:
                    cached = (cached[0].to(device), cached[1].to(device))
                    self._cache[key] = cached
                return cached

        rope = build_3d_rope_cache(
            grid_size=grid_size,
            head_dim=self.head_dim,
            num_prefix_tokens=num_prefix_tokens,
            theta=self.theta,
            normalize_coords=self.normalize_coords,
            coord_shift=self.coord_shift if self.training else None,
            coord_jitter=self.coord_jitter if self.training else None,
            coord_rescale=self.coord_rescale if self.training else None,
            drop_prob=self.drop_prob if self.training else 0.0,
            device=device,
            dtype=dtype,
        )

        if not augmenting:
            self._cache[key] = rope
        return rope

    def clear_cache(self) -> None:
        """Drop all cached RoPE tables (e.g. after ``.to(device)`` or dtype change)."""
        self._cache.clear()

    def extra_repr(self) -> str:
        parts = [f"head_dim={self.head_dim}", f"theta={self.theta}"]
        if self.normalize_coords:
            parts.append("normalize_coords=True")
        if self.coord_shift is not None:
            parts.append(f"coord_shift={self.coord_shift}")
        if self.coord_jitter is not None:
            parts.append(f"coord_jitter={self.coord_jitter}")
        if self.coord_rescale is not None:
            parts.append(f"coord_rescale={self.coord_rescale}")
        if self.drop_prob > 0:
            parts.append(f"drop_prob={self.drop_prob}")
        return ", ".join(parts)
