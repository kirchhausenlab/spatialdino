from typing import Any, Callable, Dict, Optional, Tuple
from omegaconf import DictConfig
import torch
from spatialdino.models.layers.encoder import Encoder
from spatialdino.models.layers.dino_head import DINOHead
from spatialdino.utils.misc import make_3tuple
import torch.nn as nn


class SSL(nn.Module):
    def __init__(self, config: DictConfig, **kwargs: Any) -> None:
        super().__init__()
        self.img_size = make_3tuple(config.global_crop_size)
        self.patch_size = make_3tuple(config.patch_size)
        self.config = config
        self.config.patch_size = self.patch_size
        self.config.img_size = self.img_size
        self.student = nn.ModuleDict()
        num_register_tokens = getattr(config, "num_register_tokens", 0)
        encoder_kwargs = dict(
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
            num_register_tokens=num_register_tokens,
            num_tt_register_tokens=0,
            interpolate_offset=config.interpolate_offset,
            pos_embed_type=getattr(config, "pos_embed_type", "none"),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_normalize_coords=getattr(config, "rope_normalize_coords", False),
            rope_coord_shift=getattr(config, "rope_coord_shift", None),
            rope_coord_jitter=getattr(config, "rope_coord_jitter", None),
            rope_coord_rescale=getattr(config, "rope_coord_rescale", None),
            **kwargs,
        )
        self.student["encoder"] = Encoder(**encoder_kwargs)

        self.teacher = nn.ModuleDict()
        self.teacher["encoder"] = Encoder(**encoder_kwargs)
        self.do_dino = config.dino_loss_weight > 0
        self.do_ibot = config.ibot_loss_weight > 0
        if self.do_dino:
            self.student["dino_head"] = DINOHead(
                in_dim=config.embed_dim,
                out_dim=config.n_prototypes,
            )
            self.teacher["dino_head"] = DINOHead(
                in_dim=config.embed_dim,
                out_dim=config.n_prototypes,
            )
        if self.do_ibot:
            self.student["ibot_head"] = DINOHead(
                in_dim=config.embed_dim,
                out_dim=config.n_prototypes,
            )
            self.teacher["ibot_head"] = DINOHead(
                in_dim=config.embed_dim,
                out_dim=config.n_prototypes,
            )

        self.init_teacher_from_student()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def init_teacher_from_student(self) -> None:
        self.teacher.load_state_dict(self.student.state_dict())

    def train(self, mode: bool = True) -> "SSL":
        super().train(mode)
        self.teacher.eval()
        return self

    # student forward
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        masks: Dict[str, Optional[torch.Tensor]],
        upperbound: int,
        n_masked_patches: int,
        mask_indices_list: torch.Tensor,
        device_type: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enabled: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        with torch.amp.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
        ):
            global_output, local_output = self.student["encoder"](
                [x["collated_global_crops"], x["collated_local_crops"]],
                masks=[
                    masks["collated_global_crops"],
                    masks["collated_local_crops"],
                ],
            )

        if self.do_dino:
            global_output["cls_token_after_head"] = self.student["dino_head"](
                global_output["x_norm_clstoken"]
            )
            local_output["cls_token_after_head"] = self.student["dino_head"](
                local_output["x_norm_clstoken"]
            )
        if self.do_ibot:
            buffer_tensor = global_output["x_norm_patchtokens"].new_zeros(
                upperbound, self.config.embed_dim
            )
            buffer_tensor[:n_masked_patches].copy_(
                torch.index_select(
                    global_output["x_norm_patchtokens"].flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            global_output["patch_tokens_after_head"] = self.student["ibot_head"](
                buffer_tensor
            )[:n_masked_patches]

        return global_output, local_output

    def forward_teacher(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor],
        upperbound: int,
        n_masked_patches: int,
        mask_indices_list: torch.Tensor,
        n_masked_patches_tensor: torch.Tensor,
        teacher_temp: float,
        latent_loss_fn: Dict[str, Callable],
        device_type: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enabled: bool = True,
        ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
            ):
                output = self.teacher["encoder"](x, masks=masks)
            if self.do_dino:
                output["cls_token_after_head"] = self.teacher["dino_head"](
                    output["x_norm_clstoken"]
                )
                if self.config.centering == "centering":
                    output["cls_token_softmaxed_centered"] = (
                        latent_loss_fn["cls_token"].softmax_center_teacher(
                            output["cls_token_after_head"],
                            teacher_temp=teacher_temp,
                        )
                    )
                    latent_loss_fn["cls_token"].update_center(
                        output["cls_token_after_head"]
                    )
                elif self.config.centering == "sinkhorn_knopp":
                    output["cls_token_softmaxed_centered"] = (
                        latent_loss_fn["cls_token"].sinkhorn_knopp_teacher(
                            output["cls_token_after_head"],
                            teacher_temp=teacher_temp,
                        )
                    )
                else:
                    raise ValueError(
                        f"Invalid centering method: {self.config.centering}"
                    )
            if self.do_ibot:
                buffer_tensor = output["x_norm_patchtokens"].new_zeros(
                    upperbound, self.config.embed_dim
                )
                torch.index_select(
                    output["x_norm_patchtokens"].flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor[:n_masked_patches],
                )
                output["patch_tokens_after_head"] = self.teacher["ibot_head"](
                    buffer_tensor
                )[:n_masked_patches]
                if self.config.centering == "centering":
                    output["patch_tokens_after_head"] = output[
                        "patch_tokens_after_head"
                    ].unsqueeze(0)
                    output["patch_tokens_softmaxed_centered"] = latent_loss_fn[
                        "patch_tokens"
                    ].softmax_center_teacher(
                        output["patch_tokens_after_head"][:, :n_masked_patches],
                        teacher_temp=teacher_temp,
                    )
                    output["patch_tokens_softmaxed_centered"] = output[
                        "patch_tokens_softmaxed_centered"
                    ].squeeze(0)

                    latent_loss_fn["patch_tokens"].update_center(
                        output["patch_tokens_after_head"][:n_masked_patches]
                    )
                elif self.config.centering == "sinkhorn_knopp":
                    output["patch_tokens_softmaxed_centered"] = latent_loss_fn[
                        "patch_tokens"
                    ].sinkhorn_knopp_teacher(
                        output["patch_tokens_after_head"],
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
                else:
                    raise ValueError(
                        f"Invalid centering method: {self.config.centering}"
                    )
        return output

    def update_teacher(self, teacher_momentum: float) -> None:
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student.parameters(),
                self.teacher.parameters(),
            ):
                teacher_param.data.mul_(teacher_momentum)
                teacher_param.data.add_(
                    student_param.detach().data, alpha=1 - teacher_momentum
                )
