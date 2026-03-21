from typing import Tuple

import torch
from torch.amp.autocast_mode import autocast

from spatialdino.data import DTYPE_MAPPING
from spatialdino.models.lift import convert_shape
from spatialdino.models.lift.model.extractor import ViTExtractor


def get_feats_(
    extractor: ViTExtractor,
    image_tensor: torch.Tensor,
    patches: tuple[int, int, int],
    return_class_token: bool = False,
    use_fp16: bool = True,
    device_type: str = "cuda",
    dtype: str = "bf16",
) -> torch.Tensor:
    with (
        torch.no_grad(),
        torch.amp.autocast(
            enabled=use_fp16, device_type=device_type, dtype=DTYPE_MAPPING[dtype]
        ),
    ):
        feat1 = extractor.extract_descriptors(
            image_tensor,
            return_class_token=return_class_token,
            use_fp16=use_fp16,
            device_type=device_type,
            dtype=dtype,
        )
        feat1 = feat1.permute(0, 2, 1)
        feat1 = feat1.reshape(feat1.shape[0], -1, patches[0], patches[1], patches[2])
    return feat1


# x = x.permute(0, 2, 1)  # B, C, T
# x = x.reshape(x.shape[0], -1, patches[0], patches[1], patches[2])
