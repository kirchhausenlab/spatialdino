import logging
from functools import partial
from typing import Callable, Optional

import webdataset as wds
from omegaconf import DictConfig
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from spatialdino.data.dataloader import make_dataloader
from spatialdino.data.dataset import custom_unbatched, make_webdataset
from spatialdino.models.lift.data.collate import collate_fn, collate_fn_test
from spatialdino.utils.misc import make_3tuple

logger = logging.getLogger("LiFT")


def setup_dataloader(
    config: DictConfig,
    transform: Optional[Callable] = None,
) -> DataLoader:
    train_dataset = make_webdataset_lift(
        base_data_dir=config.base_data_dir,
        batch_size=config.batch_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
        nodesplitter=wds.split_by_node,
        collation_fn=collate_fn,
        transform=transform,
    )
    train_dataloader = make_dataloader(
        train_dataset,
        batch_size=None,  # handled in web dataset
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
        drop_last=False,
        shuffle=False,  # web dataset already shuffled
        collate_fn=None,  # handled in web dataset
    )
    train_dataloader = (
        train_dataloader.compose(custom_unbatched())
        .shuffle(config.shuffle_buffer_size)
        .batched(
            config.batch_size,
            collation_fn=collate_fn,
        )
    )
    return train_dataloader


def get_test_dataset(
    config: DictConfig,
    transform: Optional[Callable] = None,
) -> wds.WebLoader:
    collate_fn = partial(
        collate_fn_test,
        patch_size=make_3tuple(config.lift_model.patch_size),
        dtype=config.dtype,
    )

    test_dataset = make_webdataset(
        base_data_dir=config.base_test_data_dir,
        batch_size=1,
        shuffle_buffer_size=0,
        nodesplitter=wds.split_by_node,
        collation_fn=collate_fn,
        transform=transform,
    )
    test_dataloader = make_dataloader(
        test_dataset,
        batch_size=None,  # handled in web dataset
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
        drop_last=False,
        shuffle=False,  # web dataset already shuffled
        collate_fn=None,  # handled in web dataset
    )
    # test_dataloader = (
    #     test_dataloader.compose(custom_unbatched())
    #     .shuffle(config.shuffle_buffer_size)
    #     .batched(
    #         config.batch_size,
    #         collation_fn=collate_fn_test,
    #     )
    # )

    return test_dataloader


def scheduler_setup(
    cfg: DictConfig,
    optimizer: Adam,
) -> ReduceLROnPlateau:
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=cfg.training.optim.config.scheduler.mode,
        factor=cfg.training.optim.config.scheduler.factor,
        patience=cfg.training.optim.config.scheduler.patience,
        cooldown=cfg.training.optim.config.scheduler.cooldown,
    )
    return scheduler
