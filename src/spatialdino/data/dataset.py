import io
import logging
import os
from functools import partial
from itertools import islice
from typing import Any, Callable, Dict, Iterator, List, Optional
from webdataset.filters import pipelinefilter
import numpy as np
import torch
import webdataset as wds
from omegaconf import ListConfig
from webdataset import filters

from spatialdino.data.utils import median_fill


def load_and_transform_webdataset(
    sample: Dict[str, Any], transform: Optional[Callable] = None
) -> Dict[str, Any]:
    key = sample["__key__"]
    experiment_name, suffix = key.split("_", 1)
    image = sample["values.npy"].astype(np.float32)  # Ensure image is float32

    image = median_fill(image)
    metadata = sample["metadata.pth"]
    metadata["experiment_name"] = experiment_name

    assert image.dtype == np.float32, (
        f"{key}, {experiment_name}: Expected image dtype to be np.float32, but got {image.dtype}"
    )

    if "non_zero_stacks" in metadata:
        mask = np.asarray(metadata["non_zero_stacks"], dtype=bool)
        metadata["mask"] = mask
        del metadata["non_zero_stacks"]
    else:
        metadata["mask"] = np.ones(metadata["shape"][0], dtype=bool)

    data = {"image": image, "mask": metadata["mask"]}

    if transform:
        output = transform(data)
    else:
        output = data

    return output


def _custom_train_unbatched(
    data: Iterator[Dict[str, Any]],
    n_global_crops: int = 2,
    n_local_crops: int = 8,
) -> Iterator[Dict[str, Any]]:
    """
    Turn batched data back into unbatched data.

    Args:
        data: Iterator of batches.

    Yields:
        Individual samples from the batches.
    """
    for sample in data:
        batch_size = sample["collated_global_crops"].size(0) // n_global_crops

        for i in range(batch_size):
            output = {}
            output["global_crops"] = sample["collated_global_crops"][
                (i * n_global_crops) : (i + 1) * n_global_crops
            ]
            output["local_crops"] = sample["collated_local_crops"][
                (i * n_local_crops) : (i + 1) * n_local_crops
            ]
            yield output


def _custom_sinder_unbatched(
    data: Iterator[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """
    Turn batched data back into unbatched data.

    Args:
        data: Iterator of batches.

    Yields:
        Individual samples from the batches.
    """
    for sample in data:
        batch_size = sample["collated_images"].size(0)
        for i in range(batch_size):
            output = {}
            output["image"] = sample["collated_images"][i]
            yield output


def _custom_segmentation_unbatched(
    data: Iterator[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """
    Turn batched data back into unbatched data.

    Args:
        data: Iterator of batches.

    Yields:
        Individual samples from the batches.
    """
    for sample in data:
        batch_size = sample["collated_images"].size(0)
        for i in range(batch_size):
            output = {}
            output["image"] = sample["collated_images"][i]
            yield output


custom_train_unbatched = pipelinefilter(_custom_train_unbatched)
custom_sinder_unbatched = pipelinefilter(_custom_sinder_unbatched)
custom_segmentation_unbatched = pipelinefilter(_custom_segmentation_unbatched)


def make_webdataset(
    base_data_dir: str,
    batch_size: int,
    collation_fn: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    nodesplitter: Optional[Callable] = None,
    workersplitter: Optional[Callable] = None,
    shuffle_buffer_size: int = 1000,
    infinite: bool = True,
    num_samples: Optional[int] = None,
) -> wds.DataPipeline:
    nodesplitter = wds.single_node_only if nodesplitter is None else nodesplitter
    workersplitter = wds.split_by_worker if workersplitter is None else workersplitter

    dataset = wds.WebDataset(
        urls=base_data_dir,
        resampled=True,
        shardshuffle=False,
        nodesplitter=nodesplitter,
        workersplitter=workersplitter,
    )

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.decode().map(
        partial(load_and_transform_webdataset, transform=transform)
    )

    if collation_fn is not None:
        dataset = dataset.batched(batch_size, collation_fn=collation_fn)

    if not infinite:
        assert num_samples is not None
        dataset = dataset.with_epoch(num_samples // batch_size)

    return dataset