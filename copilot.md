# Inference commands

1. Create a conda environment.

```bash
conda create -n cell3d python=3.12 --no-default-packages
```

4. Add required third-party libraries to the project.

DELETE

```bash
rm -rf ~/.cache/
```

```bash
rm -rf cell_interactome.egg-info
rm -rf cell_interactome/__pycache__
```

```bash
git submodule update --init --recursive
conda activate cell3d
python -m pip install -e ".[dev,flash-attn,viz]" \
--extra-index-url https://download.pytorch.org/whl/cu120 \
mamba install -c conda-forge pyclesperanto
mamba install -c conda-forge pybind11
conda install -c conda-forge ocl-icd-system
```

Ensure you have the `https://support.zivid.com/en/latest/getting-started/software-installation/gpu/install-opencl-drivers-ubuntu.html` OpenCL drivers installed.

## Training

```bash
export NCCL_SOCKET_IFNAME=ib # use all infiniband interfaces
export MASTER_ADDR=""
export MASTER_PORT=""
export RDZV_ID="" # job id
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT"
export NUM_ALLOWED_FAILURES=3
export OMP_NUM_THREADS=16
export NODE_RANK="" # rank of the node

torchrun \
--nnodes 3 \
--nproc_per_node 8 \
--node_rank $NODE_RANK \
--max-restarts $NUM_ALLOWED_FAILURES \
--rdzv-id $RDZV_ID \
--rdzv-backend $RDZV_BACKEND \
--rdzv-endpoint $RDZV_ENDPOINT \
scripts/train/pretrain.py \
--config=configs/pretrain.yaml
```

## Inference

```bash
export WEIGHTS_PATH=/nfs/scratch2/shared_image_recog_ml/weights/dinov2/pretrain_scratch_featup
python scripts/inference/inference.py \
inference.img_path="./test_images/sample.png" \
model.weights="${WEIGHTS_PATH}/ckpt_step_96999.pth" \
+inference=vitg14_reg4
```

# Inference scripts

```bash
#!/bin/bash

folders_lst=(
  # "/nfs/datasync4/tklab-llsm/20220829_p5_p55_sCMOS_Anand/processed/CS1_Dextranphrodogreen_aso_2PM/Ex05_488_100mW_z0p5/ch488nmCamA/DS"
  # "/nfs/datasync4/inacio/data/raw_data/llsm/virus_gu/ex13/ch560nmCamB/DS"
  "/nfs/data1expansion/datasync3/Gustavo/20210422_0p5_0p55_sCMOS_Gu_AP2/CS1_Ap2_live_3colorsDic/Ex07_488_60mW_z0p5/ch488nmCamA/DS"
  # "/nfs/scratch2/inacio/data/llsm_simulation/second_try/volumes_noise"
  # "/nfs/scratch/Anwesha/Zeiss/04102025/CS1/Imported/Ex03_488nm_100mW_560nm_100mW_642nm_100mW_z0p5/ch642nmCamB/DS"
)
# ------------- STEP 1: INFERENCE -------------
save_path="/raid1/cme_tests/results/ap2_gu/"
export OMP_NUM_THREADS=16
CONFIG_PATH="./src/cell_interactome/config/inference.yaml"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv-backend=c10d ./scripts/inference/inference.py \
          --config "$CONFIG_PATH" \
          --file_path "$folders_lst" \
          --save_path "$save_path"
echo "inference successful, running postprocessing"
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 ./scripts/inference/postprocessing.py \
  --config "$CONFIG_PATH" \
  --file_path "$folders_lst" \
  --save_path "$save_path"
```

# Tracking logic

```python
from pathlib import Path
from typing import List, TypedDict, Union, Literal
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import io
from cell_interactome.utils import (
    cosine_dist_func,
)
import stackview
from cell_interactome.config import parse_config
from omegaconf import DictConfig
import trackpy as tp

%load_ext autoreload
%autoreload 2

%config InlineBackend.figure_format = "retina"

# Set default tick label colors to white:
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"

# Set default axes (plot area) facecolor to black:
plt.rcParams["axes.facecolor"] = "black"

# (Optional) To also set the entire figure's background to black:
plt.rcParams["figure.facecolor"] = "black"
experiments = natsorted(list(BASE_DIR.iterdir()))
lst = []
for f in experiments:
    if f.is_dir():
        has_matching_file = any(
            "centroids.tif" in file.name or "features.tif" in file.name
            for file in f.iterdir()
        )
        if has_matching_file:
            lst.append(f.stem)
file_paths = [BASE_DIR.joinpath(f) for f in lst]
assert all([p.is_dir() for p in file_paths]), "Not all paths are directories"
print(f"Found {len(file_paths)} directories in {BASE_DIR},\nfile 0 is {file_paths[0]}")
def get_track_features(
    file_paths: List[Path],
    config: DictConfig,
) -> pd.DataFrame:
    """
    Track objects across multiple time frames.

    Parameters:
    -----------
    file_paths : list of Path objects
        List of paths to folders containing in_seg.tif.
    max_distance : float
        Maximum distance between frames for considering a match between frames. Must be greater than 0.
    max_gap : int
        Maximum allowed number of consecutive frames in which an object can be missed
        before its track is terminated.
    centroid_weight : float
        Weight for the centroid distance in the cost matrix.
    feature_weight : float
        Weight for the feature distance in the cost matrix.

    Returns:
    --------
    tracks : pd.DataFrame
        A DataFrame of Track objects representing track updates.
    """

    track_features = []  # list to store track updates (one per matched frame)
    # Process each frame by index t and its centroids
    for t, file_path in enumerate(file_paths):
        save_path = Path(file_path)
        centroid_path = save_path.joinpath("centroids.tif")
        feature_path = save_path.joinpath("features.tif")
        if centroid_path.is_file() and feature_path.is_file():
            print(f"Centroids exist at {save_path}")
            centroids, features = (
                io.imread(file_path.joinpath("centroids.tif")),
                io.imread(file_path.joinpath("features.tif")),
            )
        else:
            print(f"Generating centroids and features for {save_path}")
            centroids, features = generate_centroids(
                config=config,
                save_path=save_path,
            )
        vols = io.imread(file_path.joinpath("volume.tif"))
        for i in range(features.shape[0]):
            track_feature = {
                "z": centroids[i, 0],
                "y": centroids[i, 1],
                "x": centroids[i, 2],
                "t": t,
                "intensities": vols[
                    int(centroids[i, 0]), int(centroids[i, 1]), int(centroids[i, 2])
                ],
            }
            for j in range(features.shape[1]):
                track_feature[f"feature_{j}"] = features[i, j]
            track_features.append(track_feature)

    return pd.DataFrame(track_features)
```

# Feature Calculation

```python
def calculate_features(
    instance_segmentation: np.ndarray,
    features: np.ndarray,
    original_vol: np.ndarray,
    fg_mask: np.ndarray,
    centroid_calculation: Literal["mean", "max", "midpoint", "moment"] = "mean",
    n_components: int = 3,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    assert features.ndim == 2, "Features must be 2D"
    assert fg_mask.dtype == np.bool_, "fg_mask must be a boolean array"
    assert fg_mask.shape == instance_segmentation.shape, (
        "fg_mask must have the same shape as instance_segmentation"
    )

    labels, counts, centroids, object_features = _calculate_features(
        instance_segmentation=instance_segmentation,
        original_vol=original_vol,
        features=features,
        fg_mask=fg_mask,
        centroid_calculation=centroid_calculation,
    )

    if object_features.shape[1] != n_components:
        pca = PCA(n_components=n_components, use_torch_pca=True, device=device)
        object_features = pca.fit_transform(object_features)  # type: ignore

    return centroids, object_features


@njit(
    fastmath=True,
    cache=True,
)
def _calculate_features(
    instance_segmentation: np.ndarray,
    original_vol: np.ndarray,
    features: np.ndarray,
    fg_mask: np.ndarray,
    centroid_calculation: Literal["mean", "max", "midpoint", "moment"] = "moment",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_ind, y_ind, x_ind = np.nonzero(fg_mask)
    fg_mask_flat = fg_mask.flatten()
    labels = np.unique(instance_segmentation.flatten()[fg_mask_flat])
    assert len(labels) > 0, "No labels found"
    features_fg = features[fg_mask_flat]

    centroid_dim = 3 if centroid_calculation != "midpoint" else 6
    centroids = np.zeros((len(labels), centroid_dim), dtype=np.float64)
    if centroid_calculation == "midpoint":
        centroids[:, 0] = np.inf
        centroids[:, 1] = -np.inf
        centroids[:, 2] = np.inf
        centroids[:, 3] = -np.inf
        centroids[:, 4] = np.inf
        centroids[:, 5] = -np.inf
    elif centroid_calculation == "moment":
        m_0 = np.zeros(len(labels), dtype=np.float64)
        m_1_z = np.zeros(len(labels), dtype=np.float64)
        m_1_y = np.zeros(len(labels), dtype=np.float64)
        m_1_x = np.zeros(len(labels), dtype=np.float64)

    counts = np.zeros(len(labels), dtype=np.uint64)
    max_intensity = np.zeros(len(labels), dtype=np.float64)
    object_features = np.zeros((len(labels), features_fg.shape[-1]), dtype=np.float64)

    for idx, (z, y, x) in enumerate(zip(z_ind, y_ind, x_ind)):
        label_idx = instance_segmentation[z, y, x]
        counts[label_idx] += 1

        object_features[label_idx] += features_fg[idx]

        if centroid_calculation == "max":
            intensity = original_vol[z, y, x]
            if intensity > max_intensity[label_idx]:
                max_intensity[label_idx] = intensity
                centroids[label_idx, 0] = z
                centroids[label_idx, 1] = y
                centroids[label_idx, 2] = x
        elif centroid_calculation == "mean":
            centroids[label_idx, 0] += z
            centroids[label_idx, 1] += y
            centroids[label_idx, 2] += x
        elif centroid_calculation == "midpoint":
            # z_min, z_max, y_min, y_max, x_min, x_max
            centroids[label_idx, 0] = np.minimum(centroids[label_idx, 0], z)
            centroids[label_idx, 1] = np.maximum(centroids[label_idx, 1], z)
            centroids[label_idx, 2] = np.minimum(centroids[label_idx, 2], y)
            centroids[label_idx, 3] = np.maximum(centroids[label_idx, 3], y)
            centroids[label_idx, 4] = np.minimum(centroids[label_idx, 4], x)
            centroids[label_idx, 5] = np.maximum(centroids[label_idx, 5], x)
        elif centroid_calculation == "moment":
            weight = original_vol[z, y, x]
            m_0[label_idx] += weight  # type: ignore
            m_1_z[label_idx] += z * weight  # type: ignore
            m_1_y[label_idx] += y * weight  # type: ignore
            m_1_x[label_idx] += x * weight  # type: ignore

    if centroid_calculation == "mean":
        centroids /= counts[:, None]
    elif centroid_calculation == "midpoint":
        centroids[:, 0] = (centroids[:, 0] + centroids[:, 1]) / 2
        centroids[:, 1] = (centroids[:, 2] + centroids[:, 3]) / 2
        centroids[:, 2] = (centroids[:, 4] + centroids[:, 5]) / 2
    elif centroid_calculation == "moment":
        centroids[:, 0] = m_1_z / (m_0 + 1e-6)  # type: ignore
        centroids[:, 1] = m_1_y / (m_0 + 1e-6)  # type: ignore
        centroids[:, 2] = m_1_x / (m_0 + 1e-6)  # type: ignore

    object_features /= counts[:, None]

    return (
        labels,
        counts,
        centroids,
        object_features,
    )
```

# PCA calculation

```python
class PCA(object):
    def __init__(
        self,
        n_components: int,
        use_torch_pca: bool = True,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ):
        self.n_components = n_components
        self.use_torch_pca = use_torch_pca
        self.device = device
        # Initialize PCA once, avoiding repeated string comparison
        if self.use_torch_pca:
            self.pca = torch_pca.PCA(n_components=self.n_components, **kwargs)
        else:
            self.pca = skd.PCA(n_components=self.n_components, **kwargs)

    def _prepare_input(self, x):
        if self.use_torch_pca:
            # Combine conditional operations to reduce checks
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            elif x.dtype != torch.float32:
                x = x.float()
            # Only move to device if needed
            if self.device is not None and x.device != self.device:
                x = x.to(self.device, non_blocking=True)
        return x

    def fit(self, x):
        x = self._prepare_input(x)
        self.pca.fit(x)  # type: ignore
        return self

    def transform(self, x):
        x = self._prepare_input(x)
        transformed = self.pca.transform(x)  # type: ignore
        # Only convert to numpy if necessary
        return (
            transformed.detach().cpu().numpy()
            if isinstance(transformed, torch.Tensor)
            else transformed
        )

    def fit_transform(self, x):
        # Avoid duplicate preparation by leveraging existing methods
        x = self._prepare_input(x)
        transformed = self.pca.fit_transform(x)  # type: ignore
        return (
            transformed.detach().cpu().numpy()
            if isinstance(transformed, torch.Tensor)
            else transformed
        )


def run_pca(
    pca: PCA,
    feats: np.ndarray,
    non_zero_mask: np.ndarray,
    input_shape: Tuple[int, int, int],
    n_components: int,
    dtype: str = "float32",
) -> np.ndarray:
    Z, Y, X = input_shape
    pca_features_rgb = np.zeros((Z * Y * X, n_components), dtype=dtype)
    pca_features_rgb_fg = pca.fit_transform(feats)
    min_val = np.min(pca_features_rgb_fg, axis=0, keepdims=True)
    max_val = np.max(pca_features_rgb_fg, axis=0, keepdims=True)
    pca_features_rgb_fg = (pca_features_rgb_fg - min_val) / (max_val - min_val + 1e-6)
    pca_features_rgb[non_zero_mask] = pca_features_rgb_fg
    pca_features_rgb = pca_features_rgb.reshape(Z, Y, X, n_components)

    min_val = pca_features_rgb.min()
    max_val = pca_features_rgb.max()
    pca_features_rgb = (pca_features_rgb - min_val) / (max_val - min_val + 1e-6)
    pca_features_rgb = np.clip(pca_features_rgb * 255, 0, 255).astype(np.uint8)
    return pca_features_rgb
```
