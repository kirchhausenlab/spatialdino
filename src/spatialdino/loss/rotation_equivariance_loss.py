"""Rotation Equivariance Loss for RoPE artifact suppression.

Compares patch features from an original crop with features from a rotated
copy that has been inverse-rotated back in feature space.  If the model
encodes absolute position (e.g. via RoPE), the two representations will
disagree -- this loss penalises that disagreement, pushing the encoder
toward position-invariant content features.

Only applied to a **single** global crop to limit memory overhead (~25%
extra encoder compute).  Gradients flow only through the rotated path;
the original features are detached as the target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# Mapping from axis name to (input_dims, feature_dims) for torch.rot90.
# Input: [B, C, Z, Y, X]  ->  Feature grid: [B, gZ, gY, gX, D]
#   Z-axis rotation: rotate in the Y-X plane  -> input dims (-2, -1), feature dims (2, 3)
#   Y-axis rotation: rotate in the Z-X plane  -> input dims (-3, -1), feature dims (1, 3)
#   X-axis rotation: rotate in the Z-Y plane  -> input dims (-3, -2), feature dims (1, 2)
_AXIS_CONFIGS = [
    ((-2, -1), (2, 3)),  # Z-axis
    ((-3, -1), (1, 3)),  # Y-axis
    ((-3, -2), (1, 2)),  # X-axis
]


class RotationEquivarianceLoss(nn.Module):
    """1 - cosine similarity between original and inverse-rotated features.

    Usage::

        loss_fn = RotationEquivarianceLoss()
        axis, input_dims, feat_dims, k = loss_fn.sample_rotation()
        rotated_crop = torch.rot90(crop, k, input_dims)
        # ... run rotated_crop through encoder ...
        loss = loss_fn(orig_tokens.detach(), rot_tokens, grid_size, feat_dims, k)
    """

    @staticmethod
    def sample_rotation() -> Tuple[int, Tuple[int, int], Tuple[int, int], int]:
        """Sample a random 90-degree rotation.

        Returns:
            (axis_index, input_dims, feat_dims, k) where k in {1, 2, 3}.
        """
        axis = torch.randint(0, 3, ()).item()
        k = torch.randint(1, 4, ()).item()  # 1, 2, or 3
        input_dims, feat_dims = _AXIS_CONFIGS[axis]
        return axis, input_dims, feat_dims, k

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        orig_tokens: torch.Tensor,
        rot_tokens: torch.Tensor,
        grid_size: Tuple[int, int, int],
        feat_dims: Tuple[int, int],
        k: int,
    ) -> torch.Tensor:
        """Compute 1 - cosine_similarity between original and inverse-rotated features.

        Args:
            orig_tokens: ``[B, N, D]`` patch tokens from original crop (**detached**).
            rot_tokens:  ``[B, N, D]`` patch tokens from rotated crop.
            grid_size:   ``(gZ, gY, gX)`` patch grid dimensions.
            feat_dims:   which dims of ``[B, gZ, gY, gX, D]`` to rotate.
            k:           number of 90-degree rotations applied to input.

        Returns:
            Scalar loss (mean over batch and patches).
        """
        B, N, D = rot_tokens.shape
        gZ, gY, gX = grid_size

        # 90-degree rotation requires the rotated-plane dims to match.
        dim_a, dim_b = feat_dims
        axis_to_size = {1: gZ, 2: gY, 3: gX}
        assert axis_to_size[dim_a] == axis_to_size[dim_b], (
            f"RotationEquivarianceLoss requires equal grid extents on the "
            f"rotated plane {feat_dims}, got grid_size={grid_size}"
        )

        # Reshape to spatial grid: [B, gZ, gY, gX, D]
        rot_grid = rot_tokens.reshape(B, gZ, gY, gX, D)

        # Inverse rotation: rotate by (4 - k) in the same plane
        rot_grid = torch.rot90(rot_grid, 4 - k, feat_dims)

        # Flatten back to [B, N, D]
        rot_aligned = rot_grid.reshape(B, N, D)

        # Cosine similarity per patch token, mean over all
        cos_sim = F.cosine_similarity(orig_tokens, rot_aligned, dim=-1)  # [B, N]
        return (1.0 - cos_sim).mean()
