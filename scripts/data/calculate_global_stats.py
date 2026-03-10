from typing import Final
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from spatialdino.data.dino3d.dataloaders import make_data_loader
from spatialdino.data.transforms import get_min_max
from spatialdino.train.dino3d.utils import make_3tuple
from spatialdino.utils.configs import CONFIG_PATH
import numpy as np
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(cfg: DictConfig) -> None:
    train_dataset = instantiate(
        cfg.data.dataset.train_dataset,
        transform=None,
        collation_fn=None,
        infinite=False,
    )

    CONST_AUTO_THRESHOLD: Final[int] = 5000
    MAX_VAL: Final[int] = 65535  # 2 ** 16 - 1
    THRESHOLD_DIVISOR: Final[float] = 1.0 / CONST_AUTO_THRESHOLD
    EPS: Final[float] = np.finfo(np.float32).eps

    # Create DataLoader with multiple workers
    num_workers = 128
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],  # Unwrap the single sample from the batch
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_voxels = 0

    for i, sample in enumerate(dataloader):
        mask = sample["metadata"]["mask"]
        vol = torch.from_numpy(sample["image"])
        min_val, max_val = get_min_max(vol, mask, MAX_VAL, THRESHOLD_DIVISOR)
        vol = vol.sub_(min_val).div_(max_val - min_val + EPS).clamp_(0.0, 1.0)
        flat_vol = vol.flatten()
        total_sum += flat_vol.sum().item()
        total_sq_sum += (flat_vol**2).sum().item()
        total_voxels += flat_vol.numel()
        if i % 1000 == 0:
            logger.info(f"Processed {i} samples")

        if i % 10000 == 0:
            logger.info(
                f"Current stats = total_sum: {total_sum}, total_sq_sum: {total_sq_sum}, total_voxels: {total_voxels}"
            )

    mean = total_sum / total_voxels
    # Variance = E[x^2] - (E[x])^2
    variance = (total_sq_sum / total_voxels) - (mean**2)
    std = np.sqrt(variance)
    # mean: 0.163, std: 0.0988
    print(f"Mean: {mean}, Std: {std}")


if __name__ == "__main__":
    main()
