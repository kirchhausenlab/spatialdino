from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer

from spatialdino.models.layers.unet import UNet
from spatialdino.utils.misc import make_3tuple


class Decoder(nn.Module):
    """_summary_
    - Takes patch tokens from encoder and reconstructs volumetric data
    - Linear layer maps embedding dim back to patch pixel values
    - UNet processes reconstructed volume for semantic segmentation
    - Outputs both raw reconstruction and segmentation probabilities
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.65,
    ) -> None:
        super().__init__()
        self.num_classes = 2
        self.patch_size = make_3tuple(patch_size)
        self.img_size = make_3tuple(img_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        # linear layer to project from embedding space back to patch pixel space
        # each patch has patch_size³ * channels pixels
        self.decoder_pred = nn.Linear(
            self.embed_dim,
            np.prod(self.patch_size) * self.in_chans,
            bias=True,
        )  # decoder to patch
        # UNet for semantic segmentation on reconstructed volume
        self.unet = UNet(
            in_channels=self.in_chans, out_channels=self.num_classes, dropout=dropout
        )
        # alternative: simple 1x1 conv instead of UNet
        # self.unet = nn.Conv3d(
        #     in_channels=self.in_chans,
        #     out_channels=self.num_classes,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=True,
        # )
        self.inferer = SlidingWindowInferer(
            roi_size=(32, 32, 32),
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """_summary_
        - Initialize all model weights recursively
        - Applies _init_weights to each module in the network
        """
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """_summary_
        - Weight initialization for linear layers
        - Truncated normal for weights (std=0.02), zeros for biases
        - This is standard ViT initialization scheme
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """_summary_
        - Break down volumetric image into non-overlapping patches
        - Input: [B, C, Z, Y, X] volumetric data
        - Output: [B, N_patches, patch_dim] where patch_dim = pz*py*px*C
        - This is the inverse operation of unpatchify
        """
        B, C, Z, Y, X = imgs.size()
        # calculate how many patches we get in each dimension
        Z_patches = Z // self.patch_size[0]
        Y_patches = Y // self.patch_size[1]
        X_patches = X // self.patch_size[2]
        # reshape to separate patch grid from patch content
        # [B, C, Z, Y, X] -> [B, C, Z_patches, pz, Y_patches, py, X_patches, px]
        x = imgs.reshape((
            B,
            C,
            Z_patches,
            self.patch_size[0],
            Y_patches,
            self.patch_size[1],
            X_patches,
            self.patch_size[2],
        ))  # type: ignore
        # rearrange dimensions to group patch coordinates together
        # bczpyqxr -> bzyxpqrc (put patch grid coords first, then patch content)
        x = torch.einsum("bczpyqxr->bzyxpqrc", x)
        # flatten to patch sequence: [B, N_patches, patch_content]
        x = x.reshape(
            shape=(
                B,
                Z_patches * Y_patches * X_patches,
                np.prod(self.patch_size) * C,  # type: ignore
            )
        )
        return x

    def unpatchify(
        self, x: torch.Tensor, grid_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """_summary_
        - Reconstruct volumetric data from patch tokens
        - Input: [B, N, patch_dim] where N = gz*gy*gx, patch_dim = pz*py*px*C
        - Reshaping [B, N, patch_dim] -> [B, gz, gy, gx, pz, py, px, C]
        - Permuting [B, C, gz, pz, gy, py, gx, px] -> [B, C, Z, Y, X]
        - Finally reshape to [B, C, Z, Y, X]
        """
        # x: [B, N, pz*py*px*C] where N = gz*gy*gx
        B, N, D = x.shape
        pz, py, px = self.patch_size
        gz, gy, gx = grid_size

        C = self.in_chans
        assert N == gz * gy * gx, f"Expected {gz * gy * gx} patches, got {N}"
        assert D == pz * py * px * C, f"Expected patch dim {pz * py * px * C}, got {D}"
        # reshape to separate patch grid coordinates from patch content
        # → [B, gz, gy, gx, pz, py, px, C]
        x = x.view(B, gz, gy, gx, pz, py, px, C)
        # permute to interleave grid and patch dimensions
        # → [B, C, gz, pz, gy, py, gx, px]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # final reshape to merge grid*patch dimensions → volumetric format
        # → [B, C, gz*pz, gy*py, gx*px] == [B, C, Z, Y, X]
        return x.reshape(B, C, gz * pz, gy * py, gx * px)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> Dict[str, torch.Tensor]:
        """_summary_
        - Input is patch tokens [B, N, embed_dim], use linear projection to map embed_dim to patch pixel space
        - decoder_pred transforms embeddings back to patch pixel values
        - unpatchify reconstructs patches into volumetric format [B, C, Z, Y, X]
        - unet predicts segmentation masks with softmax over 2 classes
        - Returns dict with both reconstruction and segmentation outputs
        """
        # predictor projection
        out = {}
        # project embeddings back to patch pixel space, then reconstruct volume
        x = self.unpatchify(self.decoder_pred(x), grid_size)
        out["decoder_recon"] = x
        # apply UNet for segmentation, softmax over classes
        x = F.softmax(self.unet(x), dim=1)
        out["decoder_ncuts"] = x
        return out

    def predict(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int],
        dtype: torch.dtype = torch.bfloat16,
        use_amp: bool = True,
        device_type: str = "cuda",
        roi_size: Union[int, Tuple[int, int, int]] = (32, 32, 32),
    ) -> Dict[str, torch.Tensor]:
        """_summary_"""
        self.inferer.roi_size = roi_size
        out = {}
        # sliding window inference doesnt work with unpatchify because its not in a CNN
        # so we need to unpatchify first, then apply sliding window inference
        x = self.unpatchify(self.decoder_pred(x), grid_size)
        out["decoder_recon"] = x
        with torch.inference_mode():
            with torch.amp.autocast(  # type: ignore
                enabled=use_amp,
                dtype=dtype,
                device_type=device_type,
            ):
                out["decoder_ncuts"] = self.inferer(
                    x,
                    lambda x: F.softmax(self.unet(x), dim=1),
                ).contiguous()  # [C, Z, Y, X] # type: ignore
        return out
