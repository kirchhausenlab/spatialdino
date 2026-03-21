"""Compute global histogram normalization bounds across volumes.

Reads all TIFF volumes specified in the inference config, crops and
optionally rescales them to isotropic resolution, then computes robust
min/max intensity values by thresholding the cumulative histogram of
non-zero voxels.  The resulting bounds are saved to a text file and can
be fed back into inference for consistent cross-volume normalisation.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.amp
import torch.distributed
import torch.nn.functional as F
from natsort import natsorted
from skimage import io

from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data.utils import median_fill, validate_crop_params
from spatialdino.utils.misc import make_3tuple

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

logger = logging.getLogger("inference_3d")


def main() -> None:
    """Compute and save global histogram normalization bounds.

    Loads every TIFF volume listed in the config, applies cropping,
    median filling, and optional isotropic rescaling, then builds a
    histogram of all non-zero voxel intensities.  The lower and upper
    intensity thresholds (determined by a cumulative-sum cutoff) are
    written to ``norm_per_vol.txt`` in the configured save directory.
    """
    config = parse_config(CONFIG_PATH.joinpath("inference.yaml"))  # type: ignore
    fnames = natsorted(list(Path(config.file_path).glob("*.tif")))  # type: ignore
    file_start = int(getattr(config, "file_start", 0) or 0)
    file_end = getattr(config, "file_end", None)
    fnames = fnames[file_start:file_end]
    print(f"running norm_per_vol for {len(fnames)} files for {config.file_path}")
    vols = []
    crop_start_z, crop_end_z, crop_start_y, crop_end_y, crop_start_x, crop_end_x = (
        config.crop_params
    )
    isotropic_scale_factor = make_3tuple(config.isotropic_scale_factor)
    for fname in fnames:
        raw_volume = io.imread(fname).astype(np.float32)
        start_z, end_z = crop_start_z, crop_end_z
        start_y, end_y = crop_start_y, crop_end_y
        start_x, end_x = crop_start_x, crop_end_x

        end_z = end_z if end_z > 0 else raw_volume.shape[0]
        end_y = end_y if end_y > 0 else raw_volume.shape[1]
        end_x = end_x if end_x > 0 else raw_volume.shape[2]

        crop_params = (start_z, end_z, start_y, end_y, start_x, end_x)

        validate_crop_params(crop_params, raw_volume.shape)

        raw_volume = raw_volume[
            start_z:end_z,
            start_y:end_y,
            start_x:end_x,
        ]

        raw_volume = median_fill(raw_volume)
        if isotropic_scale_factor != (1.0, 1.0, 1.0):
            raw_volume = (
                F
                .interpolate(
                    torch.from_numpy(raw_volume).unsqueeze(0).unsqueeze(0),
                    scale_factor=isotropic_scale_factor,
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )

        vols.append(raw_volume)

    # Stack all volumes along Z and build a single histogram.
    vols = np.concatenate(vols, axis=0)
    data = torch.from_numpy(vols)
    max_val = data.max().item()
    # Fraction of total non-zero voxels used as the cumulative-sum cutoff
    # to determine the robust min/max intensity values.
    threshold_divisor = 1.0 / 5000
    bins = 65536

    # Create mask for non-zero values
    non_zero_mask = data != 0
    non_zero_data = data[non_zero_mask]

    # Calculate histogram only on non-zero values
    histogram = torch.histc(non_zero_data.float(), bins=bins, min=0, max=max_val)

    threshold = int(non_zero_data.numel() * threshold_divisor)
    cumsum = torch.cumsum(histogram, 0)
    cumsum_reverse = torch.cumsum(histogram.flip(0), 0)

    # find the indices of the cumulative sum that exceed the threshold
    min_indices = torch.nonzero(cumsum > threshold)
    max_indices = torch.nonzero(cumsum_reverse > threshold)

    histogram_bins = np.linspace(0, max_val, bins)

    # convert the indices to intensity values
    min_v = (
        histogram_bins[min_indices[0]] if len(min_indices) > 0 else histogram_bins[0]
    )
    max_v = (
        histogram_bins[bins - 1 - max_indices[0]]
        if len(max_indices) > 0
        else histogram_bins[-1]
    )
    save_path = Path(config.save_path)  # type: ignore
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path.joinpath("norm_per_vol.txt"), "w") as f:
        f.write(f"Global hist min: {min_v}\nGlobal hist max: {max_v}")
    print(f"saved norm_per_vol to {save_path.joinpath('norm_per_vol.txt')}")


if __name__ == "__main__":
    main()
