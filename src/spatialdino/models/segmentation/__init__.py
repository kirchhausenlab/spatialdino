from typing import Any, Dict, Literal, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from spatialdino.data.transforms import remap_image
from spatialdino.models.layers.decoder import Decoder
from spatialdino.models.layers.encoder import Encoder
from spatialdino.utils.misc import make_3tuple


class Segmentation(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.img_size = make_3tuple(config.global_crop_size)
        self.patch_size = make_3tuple(config.patch_size)
        self.stride = make_3tuple(config.stride)
        self.config = config
        self.config.patch_size = self.patch_size
        self.config.img_size = self.img_size
        self.encoder = Encoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            ffn_bias=config.ffn_bias,
            ffn_layer=config.ffn_layer,
            drop_path_rate=config.drop_path_rate,
            drop_path_uniform=config.drop_path_uniform,
            init_values=config.layerscale,
            interpolate_offset=config.interpolate_offset,
        )

        self.decoder = Decoder(
            img_size=config.crop_size,
            patch_size=self.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            dropout=config.decoder_dropout,
        )

    def train(self, mode: bool = True) -> "Segmentation":
        super().train(mode)
        return self

    def forward(
        self,
        x: torch.Tensor,
        device_type: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        with torch.amp.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
        ):
            out = {**self.encoder(x)}

            out["grid_size"] = (
                1 + (x.shape[-3] - self.patch_size[0]) // self.stride[0],
                1 + (x.shape[-2] - self.patch_size[1]) // self.stride[1],
                1 + (x.shape[-1] - self.patch_size[2]) // self.stride[2],
            )

            inp = out["x_norm_patchtokens"]

            inp = remap_image(inp)

            out.update(self.decoder(inp, grid_size=out["grid_size"]))

        return out

    def predict(
        self,
        x: torch.Tensor,
        device_type: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enabled: bool = True,
        roi_size: Union[int, tuple[int, int, int]] = (32, 32, 32),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction for segmentation returns tuple of (seg, recon, patch_tokens), do not return dict for sliding window inference
        """
        with torch.inference_mode():
            with torch.amp.autocast(  # type: ignore
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
            ):
                out = {}
                out["x_norm_patchtokens"] = self.encoder.predict(
                    img=x,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    vit_feat="patch",
                    norm_feat=self.config.norm_feat,
                    dtype=dtype,
                    use_amp=enabled,
                    device_type=device_type,
                    roi_size=roi_size,
                ).to(self.encoder.inferer.sw_device)  # [B, C, Z, Y, X]

                inp = out["x_norm_patchtokens"]

                # NOTE: from the original paper, convert to 0-100 range, reason is to accomodate the intensity sigma of SoftNCuts loss.
                # No percentile normalization
                inp = remap_image(inp)
                out["grid_size"] = inp.shape[-3:]

                B, C, Z, Y, X = inp.shape

                inp = inp.view(B, C, -1).transpose(1, 2)

                out.update(
                    self.decoder.predict(
                        inp,
                        grid_size=out["grid_size"],  # type: ignore
                        dtype=dtype,
                        use_amp=enabled,
                        device_type=device_type,
                        roi_size=roi_size,
                    )
                )

        return out["decoder_ncuts"], out["decoder_recon"], out["x_norm_patchtokens"]
