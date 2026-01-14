# From https://github.com/AdaptiveMotorControlLab/CellSeg3D/blob/6de4b86a671ffcd4b5535277a53082ac5ecc00a1/napari_cellseg3d/code_models/models/wnet/model.py

from typing import List, Final, Optional
import torch
import torch.nn as nn

NUM_GROUPS: Final[int] = 4


class InBlock(nn.Module):
    """Input block of the U-Net architecture."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.65
    ) -> None:
        """Create the input block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float, optional): Dropout probability. Defaults to 0.65.
        """
        super().__init__()
        # self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the input block."""
        return self.module(x)


class Block(nn.Module):
    """Basic block of the U-Net architecture."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.65
    ) -> None:
        """Initialize the basic block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float, optional): Dropout probability. Defaults to 0.65.
        """
        super().__init__()
        # self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.Conv3d(out_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the basic block."""
        return self.module(x)


class OutBlock(nn.Module):
    """Output block of the U-Net architecture."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.65
    ) -> None:
        """Initialize the output block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float, optional): Dropout probability. Defaults to 0.65.
        """
        super().__init__()
        # self.device = device
        self.module = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(64),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=64),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.BatchNorm3d(64),
            nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=64),
            nn.Conv3d(64, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the output block."""
        return self.module(x)


class UNet(nn.Module):
    """Half of the W-Net model, based on the U-Net architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Optional[List[int]] = None,
        dropout: float = 0.65,
    ) -> None:
        """Creates a U-Net model, which is half of the W-Net model."""
        if channels is None:
            channels = [64, 128, 256, 512, 1024]
        if len(channels) != 5:
            raise ValueError(
                "Channels must be a list of channels in the form: [64, 128, 256, 512, 1024]"
            )
        super(UNet, self).__init__()
        # self.device = device
        self.channels = channels
        self.max_pool = nn.MaxPool3d(2)
        self.in_b = InBlock(in_channels, self.channels[0], dropout=dropout)
        self.conv1 = Block(channels[0], self.channels[1], dropout=dropout)
        self.conv2 = Block(channels[1], self.channels[2], dropout=dropout)
        # self.conv3 = Block(channels[2], self.channels[3], dropout=dropout)
        # self.bot = Block(channels[3], self.channels[4], dropout=dropout)
        self.bot = Block(channels[2], self.channels[3], dropout=dropout)
        # self.bot = Block(channels[1], self.channels[2], dropout=dropout)
        # self.bot = Block(channels[0], self.channels[1], dropout=dropout)
        # self.deconv1 = Block(channels[4], self.channels[3], dropout=dropout)
        self.deconv2 = Block(channels[3], self.channels[2], dropout=dropout)
        self.deconv3 = Block(channels[2], self.channels[1], dropout=dropout)
        self.out_b = OutBlock(channels[1], out_channels, dropout=dropout)
        # self.conv_trans1 = nn.ConvTranspose3d(
        #     self.channels[4], self.channels[3], 2, stride=2
        # )
        self.conv_trans2 = nn.ConvTranspose3d(
            self.channels[3], self.channels[2], 2, stride=2
        )
        self.conv_trans3 = nn.ConvTranspose3d(
            self.channels[2], self.channels[1], 2, stride=2
        )
        self.conv_trans_out = nn.ConvTranspose3d(
            self.channels[1], self.channels[0], 2, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the U-Net model."""
        in_b = self.in_b(x)
        c1 = self.conv1(self.max_pool(in_b))
        c2 = self.conv2(self.max_pool(c1))
        # c3 = self.conv3(self.max_pool(c2))
        # x = self.bot(self.max_pool(c3))
        x = self.bot(self.max_pool(c2))
        # x = self.bot(self.max_pool(c1))
        # x = self.bot(self.max_pool(in_b))
        # x = self.deconv1(
        #     torch.cat(
        #         [
        #             c3,
        #             self.conv_trans1(x),
        #         ],
        #         dim=1,
        #     )
        # )
        x = self.deconv2(
            torch.cat(
                [
                    c2,
                    self.conv_trans2(x),
                ],
                dim=1,
            )
        )
        x = self.deconv3(
            torch.cat(
                [
                    c1,
                    self.conv_trans3(x),
                ],
                dim=1,
            )
        )
        x = self.out_b(
            torch.cat(
                [
                    in_b,
                    self.conv_trans_out(x),
                ],
                dim=1,
            )
        )
        return x
