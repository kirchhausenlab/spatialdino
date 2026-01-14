from typing import Any, Dict, List, Tuple

import torch

from spatialdino.data import DTYPE_MAPPING


def collate_fn(
    image_batch: List[Dict[str, Any]],
    dtype: str = "bf16",
) -> Dict[str, Any]:
    dtype = DTYPE_MAPPING[dtype]  # type: ignore
    images = torch.stack([item["image"].to(dtype) for item in image_batch], dim=0)
    images_half = torch.stack(
        [item["image_half"].to(dtype) for item in image_batch], dim=0
    )
    images_quarter = torch.stack(
        [item["image_quarter"].to(dtype) for item in image_batch], dim=0
    )

    metadata = [item["metadata"] for item in image_batch]
    res: Dict[str, Any] = {
        "image": images,
        "image_half": images_half,
        "image_quarter": images_quarter,
        "metadata": metadata,
    }

    return res


def collate_fn_test(
    samples_list: List[Dict[str, Any]],
    patch_size: Tuple[int, int, int],
    dtype: str,
) -> Dict[str, Any]:
    dtype = DTYPE_MAPPING[dtype]  # type: ignore
    C, Z, Y, X = samples_list[0]["image"].shape
    B = len(samples_list)

    for data in samples_list:
        Z = min(Z, data["metadata"]["shape"][0])
        Y = min(Y, data["metadata"]["shape"][1])
        X = min(X, data["metadata"]["shape"][2])

    Z = (Z // patch_size[0]) * patch_size[0]
    Y = (Y // patch_size[1]) * patch_size[1]
    X = (X // patch_size[2]) * patch_size[2]

    crop_or_pad = CropOrPad(target_shape=(Z, Y, X))

    batch = {
        "images": torch.zeros(size=(B, C, Z, Y, X), dtype=dtype),  # type: ignore
        "metadata": [],
    }

    for i, data in enumerate(samples_list):
        image, mask = crop_or_pad(data["image"], mask=data["metadata"]["mask"])
        batch["images"][i] = image
        data["metadata"]["mask"] = mask
        batch["metadata"].append(data["metadata"])

    return batch
