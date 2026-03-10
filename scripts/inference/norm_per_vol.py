import logging
from pathlib import Path

import numpy as np
import torch
import torch.amp
import torch.distributed
from natsort import natsorted
from skimage import io

from spatialdino.config import CONFIG_PATH, parse_config
from spatialdino.data.utils import validate_crop_params


torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

logger = logging.getLogger("inference_3d")


def main() -> None:
    config = parse_config(CONFIG_PATH.joinpath("inference.yaml"))  # type: ignore
    fnames = natsorted(list(Path(config.file_path).glob("*.tif")))  # type: ignore
    print(f"running norm_per_vol for {len(fnames)} files for {config.file_path}")
    vols = []
    start_z, end_z, start_y, end_y, start_x, end_x = config.crop_params
    for fname in fnames:
        raw_volume = io.imread(fname).astype(np.float32)

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

        vols.append(raw_volume)

    vols = np.concatenate(vols, axis=0)
    data = torch.from_numpy(vols)
    max_val = data.max().item()
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
