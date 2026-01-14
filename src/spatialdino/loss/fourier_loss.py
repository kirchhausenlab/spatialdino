# adapted from https://github.com/recursionpharma/maes_microscopy/blob/f9505d5a1e9c4716abda6ebb59b850888f569f5d/loss.py#L46
import torch
import torch.nn as nn
from torch.amp import custom_fwd
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np
from spatialdino.utils.misc import make_3tuple


class FourierLoss(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        norm: str = "ortho",
    ) -> None:
        """
        Fourier transform loss is only sound when using L1 loss to compare the frequency domains
        between the images.
        """
        super().__init__()

        self.patch_size = make_3tuple(patch_size)
        self.img_size = make_3tuple(img_size)
        self.in_chans = in_chans
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.norm = norm

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, Z, Y, X = imgs.size()
        Z_patches = Z // self.patch_size[0]
        Y_patches = Y // self.patch_size[1]
        X_patches = X // self.patch_size[2]
        x = imgs.reshape((
            B,
            C,
            Z_patches,
            self.patch_size[0],
            Y_patches,
            self.patch_size[1],
            X_patches,
            self.patch_size[2],
        ))
        x = torch.einsum("bczpyqxr->bzyxpqrc", x)
        x = x.reshape(
            shape=(
                B,
                Z_patches * Y_patches * X_patches,
                np.prod(self.patch_size) * C,
            )
        )
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, pz*py*px*C] where N = gz*gy*gx
        B, N, D = x.shape
        pz, py, px = self.patch_size
        gz, gy, gx = self.grid_size
        C = self.in_chans
        assert N == gz * gy * gx, f"Expected {gz * gy * gx} patches, got {N}"
        assert D == pz * py * px * C, f"Expected patch dim {pz * py * px * C}, got {D}"
        # → [B, gz, gy, gx, pz, py, px, C]
        x = x.view(B, gz, gy, gx, pz, py, px, C)
        # → [B, C, gz, pz, gy, py, gx, px]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # → [B, C, gz*pz, gy*py, gx*px] == [B, C, Z, Y, X]
        return x.reshape(B, C, gz * pz, gy * py, gx * px)

    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.unpatchify(pred)
        target = self.unpatchify(target)

        fft_pred = torch.fft.fftn(pred.float(), dim=(-3, -2, -1), norm=self.norm).abs()
        fft_target = torch.fft.fftn(
            target.float(), dim=(-3, -2, -1), norm=self.norm
        ).abs()

        loss = F.l1_loss(fft_pred, fft_target, reduction="none")

        loss = self.patchify(loss)

        return loss
