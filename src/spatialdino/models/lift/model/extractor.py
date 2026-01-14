from typing import Union

import torch
from torch import nn

from spatialdino.data import DTYPE_MAPPING


class ViTExtractor:
    def __init__(
        self,
        model: nn.Module,
        stride: int = 4,
        device: Union[str, torch.device] = "cuda",
    ):
        self.model = model
        self.embed_dim = model.embed_dim
        self.model.to(device)
        self.patch_size = model.patch_embed.patch_size[0]
        self.stride = stride

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        return_class_token: bool = True,
        device_type: str = "cuda",
        dtype: str = "bf16",
        use_fp16: bool = True,
    ) -> torch.Tensor:
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=device_type, dtype=DTYPE_MAPPING[dtype], enabled=use_fp16
            ),
        ):
            output = self.model(
                x=batch,
            )
        if not return_class_token:
            output = output["x_norm_patchtokens"]

        return output
