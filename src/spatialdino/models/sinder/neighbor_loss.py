from typing import Optional, Tuple, Dict, Any
import torch.nn.functional as F
import torch
import torch.nn as nn


def check_anomaly_theoretical(
    x: torch.Tensor,
    D: int,
    H: int,
    W: int,
    anomaly_dir: torch.Tensor,
    num_register_tokens: int,
    temperature: float = 0.1,
    mask_thr: float = 0.001,
    kernel: int = 3,
) -> Optional[Dict[str, Any]]:
    # batch size should be 1
    # if batch size is not 1, randomly select one sample
    idx = torch.randint(0, x.shape[0], (1,))
    x_token = x[idx, 1 + num_register_tokens :]
    x_token = x_token.view(H, W, D, -1)

    with torch.no_grad():
        feature = F.normalize(x_token, dim=-1)
        direction = F.normalize(anomaly_dir, dim=-1)

        logits = -(feature * direction).sum(dim=-1).abs()
        prob = torch.exp(logits / temperature)
        pad = kernel // 2

        w = prob.unfold(0, kernel, 1).unfold(1, kernel, 1).unfold(2, kernel, 1)
        w = w / w.sum(dim=(-1, -2, -3), keepdims=True)  # type: ignore

        gaussian = (
            torch.tensor(
                [
                    1 / 64,
                    1 / 32,
                    1 / 64,
                    1 / 32,
                    1 / 16,
                    1 / 32,
                    1 / 64,
                    1 / 32,
                    1 / 64,
                    1 / 32,
                    1 / 16,
                    1 / 32,
                    1 / 16,
                    1 / 8,
                    1 / 16,
                    1 / 32,
                    1 / 16,
                    1 / 32,
                    1 / 64,
                    1 / 32,
                    1 / 64,
                    1 / 32,
                    1 / 16,
                    1 / 32,
                    1 / 64,
                    1 / 32,
                    1 / 64,
                ],
                dtype=torch.float32,
                device=w.device,
            ).reshape(1, 1, 1, 3, 3, 3)  # Reshape for 3D convolution
        )
        w2 = w * gaussian
        w2 = w2 / w2.sum(dim=(-1, -2, -3), keepdims=True)

        T = x_token.unfold(0, kernel, 1).unfold(1, kernel, 1).unfold(2, kernel, 1)
        T = (T * w2[:, :, :, None].to(T.device)).sum(dim=(-1, -2, -3))

        mask_full = logits < logits.mean() - mask_thr * logits.std()
        mask_full[:pad, :, :] = False
        mask_full[:, :pad, :] = False
        mask_full[:, :, :pad] = False
        mask_full[-pad:, :, :] = False
        mask_full[:, -pad:, :] = False
        mask_full[:, :, -pad:] = False
        index_tensor = torch.nonzero(mask_full.flatten()).flatten()
        if len(index_tensor) == 0:
            return None
        z = index_tensor // (W * D)
        col = (index_tensor % (W * D)) // D
        row = (index_tensor % (W * D)) % D

        alpha = x_token[pad:-pad, pad:-pad, pad:-pad].norm(dim=-1).mean()
    loss_neighbor = (
        (x_token[z, col, row] - T[z - pad, col - pad, row - pad]).norm(dim=-1)
    ).mean() / alpha

    return {
        "loss_neighbor": loss_neighbor,
        "z": z,
        "col": col,
        "row": row,
        "T": T,
        "alpha": alpha,
        "mask_full": mask_full,
        "x_token": x_token,
    }


def get_neighbor_loss(
    model: nn.Module,
    x: torch.Tensor,
    skip_less_than: int = 1,
    temperature: float = 0.1,
    mask_thr: float = 0.001,
    kernel: int = 3,
) -> Optional[Dict[str, Any]]:
    spatial_size = x.shape[-3:]
    x = model.prepare_tokens_with_masks(x)
    grid_size = tuple(
        1 + (spatial_size[i] - model.patch_size[i]) // model.stride[i]
        for i in range(len(spatial_size))
    )

    for i, blk in enumerate(model.blocks):
        x = blk(x)
        assert len(model.singular_defect_directions) > 0
        result = check_anomaly_theoretical(
            x=x,
            D=grid_size[0],
            H=grid_size[1],
            W=grid_size[2],
            anomaly_dir=model.singular_defect_directions[i],
            num_register_tokens=model.num_register_tokens,
            temperature=temperature,
            mask_thr=mask_thr,
            kernel=kernel,
        )
        if result is not None:
            if len(result["row"]) >= skip_less_than:
                assert not torch.isnan(result["loss_neighbor"]).any()
                return {
                    "i": i,
                    **result,
                }
    return None
