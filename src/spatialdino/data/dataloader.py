from typing import Any, Callable, Optional

from torch.utils.data import Dataset, Sampler
import webdataset as wds


def make_dataloader(
    dataset: Dataset,
    num_workers: int,
    batch_size: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    sampler: Optional[Sampler] = None,
    drop_last: bool = False,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    **kwargs: Any,
) -> wds.WebLoader:
    return wds.WebLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        sampler=sampler,
        drop_last=drop_last,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **kwargs,
    )
