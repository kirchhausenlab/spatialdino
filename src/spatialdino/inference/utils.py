from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchdr import IncrementalPCA

from spatialdino.data import DTYPE_MAPPING


def split_volume(
    volume: Union[torch.Tensor, np.ndarray],
    num_gpus: int,
    axis: int = -2,
) -> Dict[int, Any]:
    assert axis < volume.ndim, (
        f"Axis {axis} out of bounds for tensor with {volume.ndim} dimensions"
    )
    dim_size = volume.shape[axis]
    base_chunk_size = dim_size // num_gpus
    chunks = {}
    start_idx = 0

    for i in range(num_gpus):
        chunk_size = base_chunk_size + (1 if i < dim_size % num_gpus else 0)
        end_idx = start_idx + chunk_size
        slice_indices = [slice(None)] * volume.ndim
        slice_indices[axis] = slice(start_idx, end_idx)

        chunk = volume[tuple(slice_indices)]
        chunks[i] = {"chunk": chunk, "indices": [start_idx, end_idx]}

        start_idx = end_idx

    return chunks


def pca_features(
    lr_feats: np.ndarray,
    foreground_threshold: str,
    n_components: int = 3,
    eps: float = np.finfo(np.float32).eps,  # type: ignore
    remove_background: bool = False,
    batch_size: int = 8192,
    lowrank: bool = True,
    device: str = "auto",
) -> np.ndarray:
    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size,
        lowrank=lowrank,
        device=device,
    )

    pca_features = pca.fit_transform(lr_feats)

    if remove_background:
        min_val = np.min(pca_features[:, 0])
        max_val = np.max(pca_features[:, 0])
        pca_features[:, 0] = (pca_features[:, 0] - min_val) / (max_val - min_val + eps)  # type: ignore

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(pca_features[:, 0], bins=100)
        ax.set_title("Normalized PCA 1st Component Features Distribution")
        fig.savefig("bg_hist.png", dpi=300, bbox_inches=0)

        op, foreground_thresh = foreground_threshold.strip().split(" ")

        if op == ">=":
            pca_features_fg = pca_features[:, 0] >= float(foreground_thresh)
        elif op == "<=":
            pca_features_fg = pca_features[:, 0] <= float(foreground_thresh)
        else:
            raise ValueError(f"Invalid operator: {op}, expected >= or <=")

        pca_features_left = pca.fit_transform(lr_feats[pca_features_fg])
        min_val = np.min(pca_features_left, axis=0, keepdims=True)
        max_val = np.max(pca_features_left, axis=0, keepdims=True)
        pca_features_left = (pca_features_left - min_val) / (max_val - min_val + eps)

        pca_features_rgb = np.zeros_like(pca_features)

        # new scaled foreground features
        pca_features_rgb[pca_features_fg] = pca_features_left
    else:
        min_val = np.min(pca_features, axis=0, keepdims=True)
        max_val = np.max(pca_features, axis=0, keepdims=True)
        pca_features_rgb = (pca_features - min_val) / (max_val - min_val + eps)

    return pca_features_rgb
