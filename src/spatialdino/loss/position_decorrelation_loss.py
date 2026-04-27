"""Position Decorrelation Loss for RoPE artifact suppression.

Penalises linear correlation between patch features and absolute spatial
coordinates.  When RoPE theta is too small, features become dominated by
position — this loss provides a direct corrective gradient that pushes content
signal above the positional floor.

Designed to be used on the student's global-crop patch tokens alongside the
existing DINO / iBOT / KoLeo losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionDecorrelationLoss(nn.Module):
    """Squared Frobenius norm of the cross-covariance between L2-normalised
    patch features and cell-centred grid coordinates in [-1, +1].
    """

    def __init__(self, grid_size: tuple[int, int, int]) -> None:
        super().__init__()
        gz, gy, gx = grid_size
        pos_z = (torch.arange(gz, dtype=torch.float32) + 0.5) / gz * 2 - 1
        pos_y = (torch.arange(gy, dtype=torch.float32) + 0.5) / gy * 2 - 1
        pos_x = (torch.arange(gx, dtype=torch.float32) + 0.5) / gx * 2 - 1
        mesh_z, mesh_y, mesh_x = torch.meshgrid(pos_z, pos_y, pos_x, indexing="ij")
        coords = torch.stack(
            [mesh_z.reshape(-1), mesh_y.reshape(-1), mesh_x.reshape(-1)], dim=1
        )  # [N_patches, 3]
        self.register_buffer("coords", coords)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """Compute the position decorrelation loss.

        Args:
            patch_feats: ``[B, N, D]`` patch tokens from the student encoder.

        Returns:
            Scalar loss (mean squared cross-covariance, normalised by D).
        """
        B, N, D = patch_feats.shape
        feats = F.normalize(patch_feats, dim=-1).reshape(-1, D)  # [B*N, D]
        coords = self.coords.repeat(B, 1)  # [B*N, 3]

        # Centre features (coords are already zero-mean by construction)
        feats = feats - feats.mean(0)

        # Cross-covariance matrix [D, 3]
        cross_cov = feats.T @ coords / (feats.shape[0] - 1)
        return cross_cov.pow(2).sum() / D
