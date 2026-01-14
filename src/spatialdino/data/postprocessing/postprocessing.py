from pathlib import Path
from typing import List, Tuple

import torch
from omegaconf import DictConfig
from skimage import io
from torch.utils.data import Dataset

from spatialdino.utils.misc import make_3tuple


class PostProcessingDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        fnames: List[Path],
    ) -> None:
        super().__init__()
        self.config = config
        self.file_path = Path(self.config.file_path)
        assert self.file_path.is_dir(), (
            f"file_path {self.file_path} does not exist, make sure to run inference.py first"
        )
        self.fnames = fnames
        self.save_path = Path(self.config.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.patch_size = make_3tuple(self.config.patch_size)
        self.stride = make_3tuple(self.config.stride)
        self.config.patch_size = self.patch_size

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        file = self.fnames[idx]
        lr_feats = torch.load(file.joinpath("lr_feats.pt"), weights_only=False)
        img_3d_unnorm = io.imread(file.joinpath("volume_unnorm.tif"))
        img_3d_unnorm = torch.from_numpy(img_3d_unnorm).float()
        assert img_3d_unnorm.dim() == 3, (
            f"img_3d_unnorm dim {img_3d_unnorm.dim()} != 3, expected [Z, Y, X] dims"
        )
        save_path = self.save_path.joinpath(file.stem)

        # Always return 4-tuple for consistency
        return (
            lr_feats,  # [Z_patch, Y_patch, X_patch, embed_dim]
            img_3d_unnorm,  # [Z, Y, X] or None
            str(save_path),
        )
