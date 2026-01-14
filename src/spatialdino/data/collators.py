import torch
import torch.nn.functional as F
from typing import Any, Dict, List


def collate_fn_2d(
    batch: List[Dict[str, Any]], height: int, width: int
) -> Dict[str, Any]:
    dtype = batch[0]["image"].dtype
    collated_batch = {
        "images": torch.zeros(len(batch), 3, height, width, dtype=dtype),
        "masks": torch.ones(len(batch), height, width, dtype=torch.bool),
        "stats": [],
        "metadata": [],
    }
    for idx, item in enumerate(batch):
        h, w = item["image"].size()[1:3]
        pad_h, pad_w = height - h, width - w
        padded_img = F.pad(
            item["image"], (0, pad_w, 0, pad_h), mode="constant", value=0
        )
        collated_batch["images"][idx] = padded_img
        collated_batch["masks"][idx][-pad_h:][-pad_w:] = 0
        collated_batch["stats"].append(item["stats"])
        collated_batch["metadata"].append(item["metadata"])

    return collated_batch


# def collate_fn_3d(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     pass
