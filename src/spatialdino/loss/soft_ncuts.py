"""Implementation of a 3D Soft N-Cuts loss based on https://arxiv.org/abs/1711.08506 and https://ieeexplore.ieee.org/document/868688.

The implementation was adapted and approximated to reduce computational and memory cost.
This faster version was proposed on https://github.com/fkodom/wnet-unsupervised-image-segmentation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd


class SoftNCutsLoss(nn.Module):
    """Implementation of a 3D Soft N-Cuts loss based on https://arxiv.org/abs/1711.08506 and https://ieeexplore.ieee.org/document/868688.

    Args:
        data_shape (H, W, D): shape of the images as a tuple.
        intensity_sigma (scalar): scale of the gaussian kernel of pixels brightness.
        spatial_sigma (scalar): scale of the gaussian kernel of pixels spacial distance.
        radius (scalar): radius of pixels for which we compute the weights
    """

    def __init__(
        self,
        data_shape,
        intensity_sigma,
        spatial_sigma,
        radius=None,
    ):
        """Initialize the Soft N-Cuts loss.

        Args:
            data_shape (H, W, D): shape of the images as a tuple.
            device (torch.device): device on which the loss is computed.
            intensity_sigma (scalar): scale of the gaussian kernel of pixels brightness.
            spatial_sigma (scalar): scale of the gaussian kernel of pixels spacial distance.
            radius (scalar): radius of pixels for which we compute the weights
        """
        super().__init__()
        self.intensity_sigma = intensity_sigma
        self.spatial_sigma = spatial_sigma
        self.radius = radius
        self.D = data_shape[0]
        self.H = data_shape[1]
        self.W = data_shape[2]

        if self.radius is None:
            # NOTE: check radius b/c psf is 5 voxels
            self.radius = min(
                max(5, math.ceil(min(self.D, self.H, self.W) / 20)),
                self.D,
                self.H,
                self.W,
            )

        self.register_buffer(
            "kernel",
            SoftNCutsLoss.gaussian_kernel(
                self.radius, self.spatial_sigma, dtype=torch.float32
            ),
        )

    @custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, labels, inputs) -> torch.Tensor:
        """Forward pass of the Soft N-Cuts loss.

        Args:
        # NOTE: labels and inputs are originally (N, C, H, W, D) in the original impl
            labels (torch.Tensor): Tensor of shape (N, K, D, H, W) containing the predicted class probabilities for each pixel.
            inputs (torch.Tensor): Tensor of shape (N, C, D, H, W) containing the input images.

        Returns:
            The Soft N-Cuts loss of shape (N,).
        """
        # inputs.shape[0]
        # inputs.shape[1]
        K = labels.shape[1]

        loss = 0

        for k in range(K):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(
                inputs * class_probs, dim=(2, 3, 4), keepdim=True
            ) / torch.add(torch.mean(class_probs, dim=(2, 3, 4), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.intensity_sigma**2))

            numerator = torch.sum(
                class_probs
                * F.conv3d(class_probs * weights, self.kernel, padding=self.radius),
                dim=(1, 2, 3, 4),
            )
            denominator = torch.sum(
                class_probs * F.conv3d(weights, self.kernel, padding=self.radius),
                dim=(1, 2, 3, 4),
            )
            loss += F.l1_loss(
                numerator / torch.add(denominator, 1e-6),
                torch.zeros_like(numerator),
            )
            # print(f"k {k} loss {loss}")

        return K - loss

    @staticmethod
    def gaussian_kernel(radius, sigma, dtype):
        """Computes the Gaussian kernel.

        Args:
            radius (int): The radius of the kernel.
            sigma (float): The standard deviation of the Gaussian distribution.
            dtype (torch.dtype): The data type for the kernel tensor.

        Returns:
            The Gaussian kernel of shape (1, 1, 2*radius+1, 2*radius+1, 2*radius+1).
        """
        x_2 = torch.linspace(-radius, radius, 2 * radius + 1, dtype=dtype) ** 2
        dist = (
            torch.sqrt(
                x_2.reshape(-1, 1, 1) + x_2.reshape(1, -1, 1) + x_2.reshape(1, 1, -1)
            )
            / sigma
        )
        kernel = torch.exp(-0.5 * (dist**2)) / torch.exp(torch.tensor(0.0, dtype=dtype))
        return kernel.view((1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2]))
