from pathlib import Path
from typing import Any, Mapping, Optional, Union
from omegaconf import DictConfig
import torch
import torch.nn as nn
from spatialdino.distributed import save_on_master
from spatialdino.models import Encoder
from spatialdino.models.segmentation import Segmentation
from spatialdino.models.ssl import SSL
from spatialdino.utils.misc import make_3tuple


def build_ssl_model(config: DictConfig) -> SSL:
    return SSL(config)


def build_segmentation_model(config: DictConfig) -> Segmentation:
    seg = Segmentation(config)

    if config.freeze_encoder:
        for param in seg.encoder.parameters():
            param.requires_grad = False

    if config.backbone_path:
        seg.encoder.load_state_dict(
            torch.load(
                config.backbone_path,
                map_location="cpu",
                weights_only=True,
            ),
            strict=False,
        )

    return seg


def init_backbone(config: DictConfig) -> Encoder:
    """Initialize the backbone model based on config parameters.

    Args:
        config: Configuration object containing model parameters

    Returns:
        nn.Module: Initialized encoder model
    """
    encoder = Encoder(
        img_size=make_3tuple(config.global_crop_size),
        patch_size=make_3tuple(config.patch_size),
        stride=make_3tuple(config.stride) if "stride" in config else None,
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
        num_register_tokens=getattr(config, "num_register_tokens", 0),
        num_tt_register_tokens=getattr(config, "num_tt_register_tokens", 0),
        interpolate_offset=config.interpolate_offset,
        interpolate_antialias=config.interpolate_antialias,
        interpolate_align_corners=config.interpolate_align_corners,
        pos_embed_type=config.pos_embed_type,
        rope_theta=getattr(config, "rope_theta", 10000.0),
        rope_normalize_coords=getattr(config, "rope_normalize_coords", False),
        rope_coord_shift=getattr(config, "rope_coord_shift", None),
        rope_coord_jitter=getattr(config, "rope_coord_jitter", None),
        rope_coord_rescale=getattr(config, "rope_coord_rescale", None),
        rope_drop_prob=getattr(config, "rope_drop_prob", 0.0),
    )

    if config.backbone_path:
        state_dict = torch.load(
            config.backbone_path,
            map_location="cpu",
            weights_only=True,
        )
        encoder.load_state_dict(
            state_dict,
            strict=False,
        )

    return encoder


def init_segmentation(config: DictConfig) -> Segmentation:
    """Initialize the segmentation model based on config parameters.

    Args:
        config: Configuration object containing model parameters

    Returns:
        nn.Module: Initialized segmentation model
    """
    seg = Segmentation(config)

    if config.seg_model_path:
        seg.load_state_dict(
            torch.load(
                config.seg_model_path,
                map_location="cpu",
                weights_only=True,
            ),
            strict=False,
        )

    return seg


def save_backbone(
    output_dir: Path,
    backbone: nn.Module,
) -> None:
    backbone_path = output_dir.joinpath("backbone.pth")
    save_on_master(backbone.state_dict(), backbone_path)


def load_model(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_scaler: Optional[torch.amp.GradScaler] = None,
    extra_modules: Optional[Mapping[str, nn.Module]] = None,
) -> int:
    if checkpoint_path.startswith("https"):  # type: ignore
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location="cpu", check_hash=True, weights_only=False
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint and loss_scaler is not None:
        loss_scaler.load_state_dict(checkpoint["scaler"])

    if extra_modules is not None:
        extra_state = checkpoint.get("extra_modules", {})
        for name, module in extra_modules.items():
            module_state = extra_state.get(name)
            if module_state is not None:
                module.load_state_dict(module_state)

    if checkpoint.get("step_semantics") == "optimizer_updates_completed":
        return checkpoint["step"]

    return checkpoint["step"] + 1
