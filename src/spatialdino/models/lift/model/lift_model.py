import logging

import torch
import torch.nn as nn

logger = logging.getLogger("LiFT")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        kernel_size: int = 3,
        bias: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
        stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 4):
        super().__init__()

        self.double_conv = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
                dropout=0.1,
                batch_norm=True,
            ),
            *[
                ConvBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                    dropout=0.0,
                    batch_norm=False,
                )
                for _ in range(num_blocks - 1)
            ],
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv_1 = DoubleConv(
            in_channels=in_channels // 2 + latent_dim,
            out_channels=out_channels // 2,
            num_blocks=num_blocks,
        )

    def forward(self, x: torch.Tensor, imgs_1: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


class LiFT(nn.Module):
    def __init__(
        self,
        patch_size: int,
        image_channels: int = 3,
        in_channels: int = 768,
        latent_dim: int = 256,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.up1 = Up(
            in_channels=in_channels + latent_dim,
            out_channels=in_channels,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
        )
        self.outc = nn.Conv3d(
            in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1
        )
        self.image_channels = image_channels

        self.image_convs_1 = nn.Sequential(
            nn.Conv3d(image_channels, latent_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(latent_dim),
            nn.GELU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(latent_dim),
            nn.GELU(),
        )

        self._set_scale_adapter(patch_size=patch_size)
        self.image_convs_2 = nn.Sequential(
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(latent_dim),
            nn.GELU(),
        )

    def _set_scale_adapter(
        self,
        patch_size: int,
        in_channels: int = 32,
    ) -> None:
        kernel_size = max(8 // patch_size, patch_size // 8)
        scale_factor = max(8 / patch_size, patch_size / 8)
        if patch_size == 8:
            self.scale_adapter = nn.Identity()
        else:
            scale_adapter: list[nn.Module] = []
            if patch_size % 8 != 0:
                scale_adapter.append(
                    nn.Upsample(
                        scale_factor=scale_factor,
                        mode="trilinear",
                    )
                )

            if patch_size < 8:
                scale_adapter.append(
                    nn.Sequential(
                        nn.Upsample(
                            scale_factor=scale_factor,
                            mode="trilinear",
                        ),
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=kernel_size + 1,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    )
                )

            else:
                scale_adapter.append(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=kernel_size,
                        stride=kernel_size,
                    )
                )
            self.scale_adapter = nn.Sequential(*scale_adapter)

    def forward(
        self,
        imgs: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        imgs_1 = self.image_convs_1(imgs)
        imgs_1 = self.scale_adapter(imgs_1)
        imgs_2 = self.image_convs_2(imgs_1)
        x = torch.cat([x, imgs_2], dim=1)
        # imgs_1 - (torch.Size([1, 96, 38, 154, 138]), imgs_2 - torch.Size([1, 96, 19, 77, 69]), x - torch.Size([1, 768, 19, 77, 69]))
        x = self.up1(x, imgs_1)
        logits = self.outc(x)
        return logits
