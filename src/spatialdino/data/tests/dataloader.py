from pathlib import Path
from spatialdino.data.dataset import CellDataset2D, ColorMode, ZFlattenMode
from torch.utils.data import DataLoader
from loguru import logger
import pytest
from spatialdino.data.collators import collate_fn_2d
from spatialdino.data.transforms import transforms_2d


class TestDataloader2D:
    @pytest.fixture
    def dataloader(self) -> DataLoader:
        num_workers = 4
        batch_size = 16
        shuffle = True
        color_mode = ColorMode.RGB
        z_flatten_mode = ZFlattenMode.MAX
        base_data_dir = Path("/nfs/scratch2/alavaee/data/processed/llsm")
        dataset = CellDataset2D(
            base_data_dir=base_data_dir,
            color_mode=color_mode,
            z_flatten_mode=z_flatten_mode,
            transforms=transforms_2d,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_2d,
        )
        return dataloader

    def test_dataloader(self, dataloader: DataLoader) -> None:
        """
        Check if the image is of the correct size of not
        """
        for i, batch in enumerate(dataloader):
            try:
                image = batch["images"]
                if i == 3:
                    break
                # batch dimension counts as well
                assert image.ndim == 4
            except Exception as err_:
                logger.error(
                    "Could not verify the size of every image %s for index %d"
                    % (str(err_), i)
                )
                assert False, f"Error: {str(err_)} for index {i}"
