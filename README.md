# Cell Interactome 🧬

_Automated detection and tracking of cellular interactions using self-supervised deep learning_

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-%234285F4.svg?style=flat&logo=medical&logoColor=white)](https://monai.io/)
[![Mamba](https://img.shields.io/badge/Mamba-conda--forge-green.svg)](https://mamba.readthedocs.io/)
[![DINOv2](https://img.shields.io/badge/DINOv2-Facebook%20AI-blue.svg)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/Ruff-linter-orange.svg)](https://docs.astral.sh/ruff/)
[![Napari](https://img.shields.io/badge/Napari-image--viewer-purple.svg)](https://napari.org/stable/)
[![TOML](https://img.shields.io/badge/TOML-configuration-lightgrey.svg)](https://toml.io/en/)

## 👥 Authors

[![Arkash Jain - LinkedIn](https://img.shields.io/badge/Arkash%20Jain-LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/arkashj/)
[![Alex Lavaee - LinkedIn](https://img.shields.io/badge/Alex%20Lavaee-LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/alexlavaee/)
[![Tom Kirchhausen - PI (Research)](https://img.shields.io/badge/Tom%20Kirchhausen-PI-0b6f9b?style=flat&logo=researchgate)](https://cellbio.hms.harvard.edu/faculty-staff/tomas-kirchhausen)

**Cell Interactome** uses advanced AI to automatically detect, segment, and track biological objects in 3D light-sheet microscopy data. Key capabilities:

- **🎯 Automated detection** of cellular structures (vesicles, organelles, proteins)
- **✂️ Precise segmentation** with no manual annotation required
- **🔗 Temporal tracking** for cellular dynamics analysis
- **📊 Multi-channel support** (488nm, 560nm, 642nm wavelengths)

---

## 📚 Resources

**Play around with the interactive CLI to get a feel for the pipeline and then tune the actual scripts. If you get stuck, please watch the video tutorials before asking questions.**

- **[Video Tutorials](wiki/run_spatial_dino/)**: Workflow demonstrations
- **[Technical Details](wiki/spatial_dino.pdf)**: Research methodology
- **[Getting Started](wiki/getting_started.md)**: Step-by-step guide

**Interested in infrastructure setup?** For low-level infrastructure details and configuration notes, see my GitHub issue on the PyTorch repository: [PyTorch issue #144779 — low-level infra details](https://github.com/pytorch/pytorch/issues/144779).



---

## 🚀 Quick Start

### Installation

```bash
git clone --recursive https://github.com/kirchhausenlab/cell_interactome.git
cd cell_interactome
mamba env create -f env.yaml -n cell3d
mamba activate cell3d
pip install -e . && pip install natsort rich tqdm click
```

### Interactive Analysis

```bash
python scripts/interactive_inf.py  # Feature extraction & segmentation
```

---

## 📋 Table of Contents

### 🚀 Getting Started

1. [Resources](#-resources) - Video tutorials, technical docs, and guides
2. [Quick Start](#-quick-start) - One-command installation and basic usage
3. [Machine Setup](#machine-setup-💻) - Ubuntu/MacOS/Windows compatibility guide
4. [CUDA Installation](#cuda-installation-🛠️) - GPU requirements and setup verification
5. [Installation](#installation-🛠️) - Complete setup process
   - [SSH Key Setup](#1-ssh-key-setup) - GitHub authentication
   - [Miniforge & Mamba](#2-install-miniforge--mamba) - Environment management
   - [Repository Clone](#3-clone-the-repository) - Source code download
   - [Environment Creation](#4-create-a-conda-environment) - Dependencies installation
   - [Troubleshooting](#-troubleshooting) - Common issues and fixes

### 🔬 Core Pipeline

6. [Training](#training-🏃) - Multi-node self-supervised model training
   - [Environment Setup](#environment-setup) - Configuration variables
   - [Multi-Node Setup](#multi-node-setup) - Distributed training configuration
   - [Training Command](#training-command) - Execution examples
7. [Inference](#inference-🔍) - Feature extraction from 3D volumes
   - [Interactive CLI](#inference-🔍) - Recommended user interface
   - [Input Size Guidelines](#input-size-guidelines) - Volume constraints
   - [Example Scripts](#example-inference-script) - Batch processing setup
   - [Script Parameters](#script-parameters) - Configuration options
8. [Segmentation](#segmentation-✂️) - Instance mask generation
   - [Interactive CLI](#segmentation-✂️) - User-friendly interface
   - [Example Scripts](#example-segmentation-script) - Automation examples
   - [Parameter Tuning](#parameter-tuning) - SNR-specific optimization

### 📊 Workflow & Analysis

9. [Pipeline Workflow](#-pipeline-workflow) - Complete processing pipeline
   - [Feature Extraction](#1-feature-extraction) - DINO feature computation
   - [Segmentation Process](#2-segmentation) - Attention-based clustering
   - [Sample Results](#sample-results-by-data-type) - Data type examples
   - [Tracking & Visualization](#3-tracking--visualization) - Temporal analysis
10. [Results & Features](#-results--features) - Output files and capabilities
    - [Output Files](#output-files) - Generated data formats
    - [Feature Visualization](#feature-visualization) - Analysis examples

### 🔧 Technical Details

11. [Architecture](#-architecture) - Codebase structure overview
12. [Configuration](#-configuration) - Key parameter settings
13. [Support](#-support) - Help and contact information
14. [Citation](#-citation) - Academic reference

### 📖 Advanced Technical Documentation

15. [Technical Supplement](#-technical-supplement) - Deep dive into algorithms
    - [Inference Pipeline](#inference-pipeline-deep-dive) - Architecture details
    - [Global Normalization](#global-normalization-strategy) - Preprocessing methods
    - [Distributed Inference](#distributed-inference-strategy) - Multi-GPU processing
    - [Feature Extraction](#feature-extraction-details) - DINO implementation
    - [Enhanced Normalization](#enhanced-normalization-for-complex-volumes) - Advanced preprocessing
    - [Segmentation Details](#postprocessing--segmentation) - Clustering algorithms
    - [Tracking Algorithm](#tracking-algorithm) - Temporal linking methods
    - [SINDER Post-training](#sinder-post-training) - Artifact correction technique

---

## Machine Setup 💻

You can use the following machines to run the Cell Interactome pipeline:

1. **Ubuntu** <img src="https://user-images.githubusercontent.com/25181517/186884153-99edc188-e4aa-4c84-91b0-e2df260ebc33.png" width="15">
2. **MacOS** <img src="https://user-images.githubusercontent.com/25181517/186884152-ae609cca-8cf1-4175-8d60-1ce1fa078ca2.png" width="15"> [*Please Use Docker*] <img src="https://user-images.githubusercontent.com/25181517/117207330-263ba280-adf4-11eb-9b97-0ac5b40bc3be.png" width="15">
3. **Windows** <img src="https://user-images.githubusercontent.com/25181517/186884150-05e9ff6d-340e-4802-9533-2c3f02363ee3.png" width="18"> [*Please Use Docker*] <img src="https://user-images.githubusercontent.com/25181517/117207330-263ba280-adf4-11eb-9b97-0ac5b40bc3be.png" width="18">

---

## CUDA Installation 🛠️

This project requires CUDA version 12.x. Verify the correct version of CUDA installed by running:

```bash
nvcc --version
```

---

## Installation 🛠️

### 1. SSH Key Setup

To clone this repository, you'll need to set up SSH keys for GitHub authentication. Follow the [GitHub SSH documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for complete details.

#### Generate a new SSH key:

```bash
# Use Ed25519 algorithm (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"

# For legacy systems, use RSA
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

When prompted:

- Press Enter to accept the default file location (`~/.ssh/id_ed25519`)
- Enter a secure passphrase (optional but recommended)

#### Add SSH key to ssh-agent:

```bash
# Start the ssh-agent
eval "$(ssh-agent -s)"

# Add your SSH private key to the ssh-agent
ssh-add ~/.ssh/id_ed25519
```

#### Add the public key to GitHub:

1. Copy your public key to clipboard:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

2. Go to GitHub → Settings → SSH and GPG keys → New SSH key
3. Paste your public key and save

### 2. Install Miniforge & Mamba

Mamba is a faster drop-in replacement for conda that we use for environment management:

```bash
# Download Miniforge (includes mamba)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge3.sh

# Install Miniforge
bash Miniforge3.sh -b -p $HOME/miniforge3

# Initialize shell integration
source $HOME/miniforge3/bin/activate
conda init

# Restart your shell or source bashrc
source ~/.bashrc
```

### 3. Clone the repository:

```bash
git clone --recursive git@github.com:kirchhausenlab/cell_interactome.git
```

### 4. Create a conda environment:

```bash
mamba env create -f env.yaml -n cell3d
mamba activate cell3d
pip install -e .
pip install natsort rich tqdm click xformers
```

### 📦 Troubleshooting

In case of errors, ensure you have the required dependencies for Python installed:

**Delete the cache:**

```bash
rm -rf ~/.cache/
rm -rf cell_interactome.egg-info
```

**Install system dependencies:**

```bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
ninja-build cmake libegl1-mesa-dev python3-dev
```

---

## Training 🏃

### Environment Setup

If you do not have a `.bashrc` file, create one:

```bash
touch ~/.bashrc
```

Add the following to your `.bashrc` file:

**XDG Configuration** (cross-platform directory layout):

```bash
export tname="tmux rename-window"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_DATA_DIRS="/usr/local/share:/usr/share"
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CONFIG_DIRS="/etc/xdg"
export XDG_STATE_HOME="$HOME/.local/state"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_RUNTIME_DIR="$HOME/.local/run"
export XDG_DESKTOP_DIR="$HOME/Desktop"
export XDG_DOCUMENTS_DIR="$HOME/Documents"
export XDG_DOWNLOAD_DIR="$HOME/Downloads"
export XDG_MUSIC_DIR="$HOME/Music"
export XDG_PICTURES_DIR="$HOME/Pictures"
export XDG_PUBLICSHARE_DIR="$HOME/Public"
export XDG_TEMPLATES_DIR="$HOME/Templates"
export XDG_VIDEOS_DIR="$HOME/Videos"
export HUGGINGFACE_HUB_CACHE="${XDG_CACHE_HOME}/huggingface/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export JUPYTER_PLATFORM_DIRS=1
```

**NCCL Configuration** (NVIDIA's communication library):

```bash
export NCCL_SOCKET_NTHREADS=4     # number of threads per socket
export NCCL_NSOCKS_PERTHREAD=4    # number of sockets per thread
export NCCL_IB_DISABLE=0          # enable Infiniband
export NCCL_IB_HCA="mlx5"         # use Mellanox Infiniband
export CUDA_HOME="/usr/local/cuda-12"  # choose the correct CUDA version
export PATH=$CUDA_HOME/bin:$PATH
export CPATH="$CUDA_HOME/include:$CPATH"
```

**C++ Library:**

```bash
export CXX=g++
```

**Distributed Training:**

```bash
export NCCL_SOCKET_IFNAME=ib      # use all infiniband interfaces
export RDZV_BACKEND="c10d"
export OMP_NUM_THREADS=16
export NUM_ALLOWED_FAILURES=3

export RDZV_ID="2001"             # set the rdzv id to be the same for all nodes
export MASTER_PORT="29500"        # set the master port to be the same for all nodes
```

### Multi-Node Setup

To get the Master Address, get the IP address of your infiniband interface:

```bash
ibstat
```

Example output:

```bash
ib0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST> mtu 2044
inet 10.1.0.11 netmask 255.255.0.0 broadcast 10.1.255.255
...

ib1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST> mtu 2044
inet 10.2.0.11 netmask 255.255.0.0 broadcast 10.2.255.255
inet6 fe80::63f:7203:d3:ae0e prefixlen 64 scopeid 0x20<link>
unspec 00-00-10-29-FE-80-00-00-00-00-00-00-00-00-00-00 txqueuelen 256 (UNSPEC)
RX packets 137081439 bytes 250938047445 (250.9 GB)
RX errors 0 dropped 0 overruns 0 frame 0
TX packets 786878109 bytes 1591832370392 (1.5 TB)
TX errors 0 dropped 0 overruns 0 carrier 0 collisions 0
```

Set the master address (in this case `10.1.0.11`):

```bash
export MASTER_ADDR="10.1.0.11"
export RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT"
```

### Training Command

For multi-node training with 3 nodes and 8 GPUs per node:

```bash
# Arguments explanation:
# --nnodes: number of nodes (e.g. 3)
# --node_rank: rank of the node (e.g. 0, 1, 2, ... n) for n nodes
# --nproc_per_node: number of processes/GPUs per node (e.g. 8)
# --master_addr: address of the master node (e.g. 10.10.10.10)
# --master_port: port of the master node (e.g. 29500)

torchrun --nnodes 3 --nproc_per_node 8 --node_rank $NODE_RANK \
         --rdzv-id $RDZV_ID --rdzv-backend $RDZV_BACKEND \
         --rdzv-endpoint $RDZV_ENDPOINT scripts/train/pretrain.py
```

---

## Inference 🔍

**Interactive CLI (Recommended):**

```bash
python3 scripts/interactive_inf.py
```

### Input Size Guidelines

For efficient inference, we recommend maximum `[Z,Y,X]` input of `[150, 750, 750]` (total volume ~84M pixels³). Alternative configurations:

- `[300, 300, 300]`
- `[80, 1000, 500]`

### Example Inference Script

```bash
#!/bin/bash

folder_path="/nfs/data1expansion/datasync3/Gustavo/20210422_0p5_0p55_sCMOS_Gu_AP2/CS1_Ap2_live_3colorsDic/Ex07_488_60mW_z0p5/ch488nmCamA/DS"
number_of_files=1                    # -1 for all timepoints, otherwise chose a number
save_path="/raid1/cme_tests/results/ablations/ap2_test"
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

torchrun --nnodes 1 --node_rank 0 --nproc_per_node $NUM_PROC_PER_NODE \
         --rdzv_endpoint=localhost:9999 ./scripts/inference/inference.py \
  file_path="$folder_path" \
  save_path="$save_path" \
  number_of_files=$number_of_files \
  crop_params="[0,0,0,0,0,0]"
```

### Script Parameters

- **`folder_path`**: Path to folder containing images
- **`number_of_files`**: Number of files to process (-1 for all files)
- **`save_path`**: Path to save results
- **`OMP_NUM_THREADS`**: Number of threads to use
- **`CUDA_VISIBLE_DEVICES`**: List of GPUs to use
- **`NUM_PROC_PER_NODE`**: Number of processes/GPUs per node
- **`crop_params`**: Parameters for cropping images

---

## Segmentation ✂️

**Interactive CLI (Recommended):**

```bash
python3 scripts/interactive_seg.py
```

### Example Segmentation Script

```bash
#!/bin/bash
save_path="/raid1/cme_tests/results/ap2_latest"
blur_image=False                     # true for most datasets
blur_factor=1.0                      # use 3.0 for very low SNR, 1.0 for SNR > 3
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
use_raw_mask=False                   # set to False unless SNR < 3
spot_sigma=2.0                       # for most endosomes, viruses
outline_sigma=2.0
attention_weight=0.5
background_removal_radius=10
skip_existing=False                  # set to True if continuing previous experiments

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$NUM_PROC_PER_NODE \
         --rdzv_endpoint=localhost:8999 ./scripts/inference/segmentation.py \
  file_path="$save_path" \
  save_path="$save_path" \
  skip_existing=$skip_existing \
  background_removal_radius=$background_removal_radius \
  attention_weight=$attention_weight \
  blur_factor=$blur_factor \
  use_raw_mask=$use_raw_mask \
  outline_sigma=$outline_sigma \
  spot_sigma=$spot_sigma \
  blur_image=$blur_image
```

### Parameter Tuning

For parameter optimization, check `scripts/notebooks/segmentation_misc/test_segmentation.ipynb`.

**Key guidelines:**

- **Low SNR datasets**: Increase `blur_factor` to 3.0 for Gaussian blur
- **Standard datasets (SNR > 3)**: Use `blur_factor=1.0`
- **Parameter visualization**: The sample experiment images above show optimal configurations for different data types

---

## 🔄 Pipeline Workflow

### 1. Feature Extraction

![Inference Configuration](scripts/images/inference_config.png)
![Path Selection](scripts/images/inference_select_paths.png)
![Setup Interface](scripts/images/inference_setup.png)

**Configure and extract DINO features from 3D volumes:**

- Multi-GPU distributed processing
- Automatic volume preprocessing and normalization
- Sliding window inference for large datasets

### 2. Segmentation

![Segmentation Interface](scripts/images/segmentation.png)
![File Operations](scripts/images/file_operations.png)

**Generate instance segmentations using attention-based clustering:**

- K-means clustering on normalized features
- Attention density-guided mask refinement
- Instance segmentation via Voronoi-Otsu labeling

#### Sample Results by Data Type

|                                   **AP2 (Endocytic Proteins)**                                   |                                       **Dextran (Fluid Markers)**                                        |
| :----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
|       ![AP2 Config](scripts/notebooks/segmentation_misc/sample_experiments/ap2/config.png)       |       ![Dextran Config](scripts/notebooks/segmentation_misc/sample_experiments/dextran/config.png)       |
| ![AP2 Segmentation](scripts/notebooks/segmentation_misc/sample_experiments/ap2/segmentation.png) | ![Dextran Segmentation](scripts/notebooks/segmentation_misc/sample_experiments/dextran/segmentation.png) |

|                                                **Transferrin (Low SNR)**                                                |                                                **Simulated Low SNR**                                                |
| :---------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|       ![Transferrin Config](scripts/notebooks/segmentation_misc/sample_experiments/transferrin_lowsnr/config.png)       |       ![Simulated Config](scripts/notebooks/segmentation_misc/sample_experiments/simulated_lowsnr/config.png)       |
| ![Transferrin Segmentation](scripts/notebooks/segmentation_misc/sample_experiments/transferrin_lowsnr/segmentation.png) | ![Simulated Segmentation](scripts/notebooks/segmentation_misc/sample_experiments/simulated_lowsnr/segmentation.png) |

### 3. Tracking & Visualization

![Tracking](scripts/images/tracking.png)
![Feature Visualization](scripts/images/visualize_features.png)
![Track Visualization](scripts/images/tracks.png)

**Temporal object linking and analysis:**

- Physics-based velocity prediction
- Feature-enhanced trackpy integration
- Multi-channel track visualization in Napari

![CLI Summary](scripts/images/summary_of_cli.png)
![CLI Interface](scripts/images/cli_image.png)

---

## 📊 Results & Features

### Output Files

- **`lr_feats.tif`**: DINO feature maps
- **`segmentation.tif`**: Instance masks
- **`centroids.csv`**: Object coordinates
- **`tracks.csv`**: Temporal trajectories

### Feature Visualization

Get comprehensive upsampled features as demonstrated in: `/nfs/scratch1/ajain/cell_interactome/scripts/notebooks/feature_visualizations/ap2/ap2_latest_ex07_CamA_ch0_stack0000_488nm_0000000msec_0087860321msecAbs_000x_000y_000z_0000t/features/`

---

## 📁 Architecture

```
cell_interactome/
├── src/cell_interactome/        # Core ML framework
│   ├── models/ssl/             # DINO self-supervised architecture
│   ├── models/segmentation/    # Encoder-decoder models
│   ├── data/transforms.py      # Preprocessing pipelines
│   ├── tracking/              # Temporal linking algorithms
│   └── config/                # Configuration files
├── scripts/
│   ├── train/pretrain.py      # Self-supervised training
│   ├── inference/             # Feature extraction & segmentation
│   ├── interactive_inf.py     # User interface
│   └── notebooks/             # Analysis workflows
└── models/                    # Pretrained weights
```

---

## 🔧 Configuration

**Key parameters in `src/cell_interactome/config/inference.yaml`:**

- `chunk_size`: Processing volume size
- `patch_size`: Transformer patch dimensions
- `stride`: Inference overlap
- `isotropic_scale_factor`: Spatial normalization

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kirchhausenlab/cell_interactome/issues)
- **Contact**: Araksh Jain ([arkashjain17@gmail.com](mailto:arkashjain17@gmail.com)), Alex Lavaee ([alavaee@bu.edu](mailto:alavaee@bu.edu))

---

## 📄 Citation

```bibtex
@article{cell_interactome2024,
  title={Cell Interactome: Automated Detection of Cellular Interactions using Self-Supervised Deep Learning},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

# 📖 Technical Supplement

## Inference Pipeline Deep Dive

### Architecture Overview

The inference pipeline implements a multi-stage transformation using DINO self-supervised vision transformers adapted for 3D biological imaging:

```
Raw Volume → Preprocessing → Feature Extraction → Postprocessing → Segmentation
```

### Global Normalization Strategy

**Step 1: Histogram-based Global Normalization**

- Remove first z-stack (focus artifacts)
- Concatenate volumes across time series
- Generate background mask using `EMPTY_VOXEL_THRESHOLD`
- Calculate percentile boundaries (excludes outliers/background noise)
- Save global min/max values for consistent normalization

**Step 2: Inference Transform Pipeline**

```python
transforms = [
    Maskd(),                    # Filter based on intensity thresholds
    DeleteItemsd(),             # Memory-efficient cleanup
    Interpolated(),             # Chunked trilinear interpolation
    AddChannelsd(),             # Add channel dimension
    ResizeWithPadOrCropd(),     # Median-value padding
    HistogramNormalize(),       # Global histogram clipping + min-max
    ToTensord()                 # Final tensor conversion
]
```

### Distributed Inference Strategy

**Volume Processing:**

- **Chunk size**: `[196, 196, 196]` (configurable)
- **Patch size**: `[8, 8, 8]` (ViT patch dimensions)
- **Isotropic scaling**: Applied before padding for aspect ratio correction
- **Target shape calculation**: Ensures divisibility by chunk size

**Example Volume Transformation:**

```
Original: [86, 350, 350]
Isotropic scaling: [2.401, 1.0, 1.0] → [206, 350, 350]
Target shape: [400, 400, 400] (padded to be divisible by chunk size)
```

**MONAI Sliding Window:**

- Modified `SlidingWindowInference` for 3D volumes
- Register tokens removed during inference (training stability only)
- CLS token excluded from output
- Contiguous tensor requirements enforced

### Feature Extraction Details

**Model Configuration:**

- **Backbone**: DINO ViT adapted for 3D
- **Batch size**: 1 (sliding window)
- **Output**: `lr_feats` with shape `[embed_dim + num_heads, Z_patch, Y_patch, X_patch]`
- **Attention channels**: Last 12 channels contain multi-head attention maps
- **Feature channels**: First `embed_dim` channels contain patch features

**Padding Removal:**

```python
scale_factor = volume.shape / lr_feats.shape  # Account for patch downsampling
padding_3 = padding * scale_factor           # Scale padding to feature resolution
lr_feats = lr_feats[padding_3[0]:, padding_3[1]:, padding_3[2]:]  # Unpad features
```

## Enhanced Normalization for Complex Volumes

### Problem: Patch-wise Contrast Variations

Large 3D volumes exhibit:

- **Illumination gradients**: Uneven lighting across regions
- **Acquisition artifacts**: Variable SNR in different areas
- **Patch boundary discontinuities**: After processing artifacts

### Solution: Multi-method Normalization

**1. Patch-wise Normalization**

```python
normalizer = PatchNormalize(
    patch_size=(448, 448, 448),
    overlap_factor=0.1,          # 10% overlap between patches
    method="minmax",             # or "histogram" for robust outlier handling
    blend_boundaries=True        # Smooth transitions
)
```

**2. Enhanced Flat-field Correction**

```python
corrected = enhanced_flat_field_correction(
    volume=volume,
    patch_size=(448, 448, 448),
    blur_radius=50.0,            # Larger = more global correction
    method="gaussian",           # "gaussian", "median", or "rolling_ball"
    device=cle_device           # GPU acceleration via pyclesperanto
)
```

**3. Hybrid Approach (Recommended)**

```python
def hybrid_normalization(volume):
    # Step 1: Remove illumination gradients
    corrected = enhanced_flat_field_correction(volume, blur_radius=30.0)

    # Step 2: Equalize contrast patch-wise
    normalizer = PatchNormalize(patch_size=(448, 448, 448), blend_boundaries=True)
    return normalizer(corrected)
```

### Performance Guidelines

| Method     | Speed  | Quality   | Use Case                      |
| ---------- | ------ | --------- | ----------------------------- |
| Patch-wise | Fast   | Good      | Regional contrast differences |
| Flat-field | Medium | Excellent | Illumination gradients        |
| Hybrid     | Slow   | Best      | Complex multi-issue datasets  |

## Postprocessing & Segmentation

### Feature Upsampling & Clustering

**1. Feature Processing:**

```python
# Split features: attention (last 12) vs patch features (remaining)
attn_feats = lr_feats[..., -config.num_heads:]
patch_feats = lr_feats[..., :-config.num_heads]

# L2 normalize patch features
patch_feats = F.normalize(patch_feats, dim=-1)

# Compute attention weights for K-means sampling
attn_sum = attn_feats.sum(dim=-1)
probs = max(attn_sum) - attn_sum  # Higher prob for less-attended regions
```

**2. K-means Clustering:**

```python
labels = kmeans_fit_predict(
    features=patch_feats.flatten(),
    weights=probs.flatten(),
    n_clusters=config.n_clusters,
    distance="cosine",
    init="kmeans++"
)
```

**3. Instance Segmentation Pipeline:**

```
Raw Mask → Dilation → Sobel Filter → CLAHE → Foreground Extraction →
Laplacian of Gaussian → Voronoi-Otsu Labeling → Instance Segmentation
```

### Attention Density Analysis

**Mask Generation:**

```python
def get_3d_mask_and_density(img_3d, labels, attn_feats):
    # For each cluster, compute attention density per unit area
    density = compute_attention_density_per_cluster(labels, attn_feats)

    # Generate binary mask using density threshold
    threshold = threshold_otsu(density)
    seg_3d_mask = density > threshold

    return seg_3d_mask, density
```

## Tracking Algorithm

### Centroid Calculation via Statistical Moments

**Moment-based Centers:**

```python
def calculate_centroids(volume, instance_labels, features):
    for label_id in unique_labels:
        # Get all pixels belonging to this instance
        mask = (instance_labels == label_id)

        # Zero moment: total intensity
        m_0 = volume[mask].sum()

        # First moments: weighted coordinate sums
        coords = np.where(mask)
        m_1_z = (coords[0] * volume[mask]).sum()
        m_1_y = (coords[1] * volume[mask]).sum()
        m_1_x = (coords[2] * volume[mask]).sum()

        # Centroid: first_moment / zero_moment
        centroid = np.array([m_1_z, m_1_y, m_1_x]) / m_0

        # Feature average
        avg_features = features[mask].mean(axis=0)
```

### GPU-Accelerated PCA & Tracking

**Dimensionality Reduction:**

```python
# Convert to torch tensors for GPU acceleration
features_tensor = torch.from_numpy(features).cuda()
pca_features = torch_pca(features_tensor, n_components=10)  # 99.5% variance
```

**Trackpy Integration:**

```python
# Create velocity predictor for physics-based tracking
predictor = trackpy.predict.NearestVelocityPredictor()

# Link particles using both spatial coordinates and PCA features
tracks = trackpy.link(
    centroids_df,
    search_range=config.search_range,
    predictor=predictor,
    adaptive_stop=config.adaptive_stop,
    adaptive_step=config.adaptive_step,
    memory=config.memory,  # 3-5 frames typical
    link_strategy='hybrid'  # Combines coordinates + features
)
```

### Multi-channel Track Visualization

**Combined Visualization:**

```python
# Color coding by channel
channel_colors = {
    '488': 'green',   # Transferrin
    '560': 'red',     # Zeiss
    '642': 'blue'     # Zeiss
}

# Track ID methods:
# Offset: Channel 488 (0-7059), 560 (7060-17198), 642 (17199-24870)
# Prefix: "track_id_channel" format
```

## SINDER Post-training

### Singular Defect Direction Theory

**Problem**: Vision transformers exhibit high-norm "defective" tokens that create artifacts in biological segmentation.

**Root Cause**: Artifacts correlate with leading left singular vectors of linearized transformer operations, independent of input data.

### Linearization Process

**Attention Block Decomposition:**

```python
A = A₄ @ A₃ @ A₂ @ A₁ @ A₀
where:
A₀ = (I - 1/N * 1ₙₓₙ)      # Centering matrix
A₁ = diag(norm.weight)       # Layer normalization scaling
A₂ = qkv.weight[-1/3:]      # Value projection from QKV
A₃ = attn.proj.weight       # Output projection
A₄ = diag(layer_scale.gamma) # Layer scaling
```

**MLP Block Linearization:**

```python
# Non-linear activation approximated via least-squares
X = torch.randn(100000, embed_dim)  # Random samples
Y = activation(mlp.fc1(X))          # Activation output
C₂ = solve_least_squares(X, Y)      # Linear approximation

C = C₄ @ C₃ @ C₂ @ C₁ @ C₀  # Full MLP linearization
```

**Layer Composition:**

```python
for layer_i in range(num_layers):
    E_i = MLP_linear_i @ (I + Attention_linear_i)  # Residual connection
    G_i = E_i @ G_{i-1}  # Accumulate transformations
    u_i, s_i, v_i = SVD(G_i)
    defect_directions[i] = u_i[:, 0]  # Leading left singular vector
```

### Training Strategy

**1. Replace Linear Layers:**

```python
class SVDLinearAddition(nn.Module):
    def __init__(self, linear_layer):
        # Decompose: W = U @ diag(S) @ V^T
        U, S, Vt = torch.svd(linear_layer.weight)
        self.U = nn.Parameter(U, requires_grad=False)
        self.S = nn.Parameter(S, requires_grad=False)
        self.Vt = nn.Parameter(Vt, requires_grad=False)
        self.epsilon = nn.Parameter(torch.zeros_like(S))  # Only trainable param

    def forward(self, x):
        W = self.U @ torch.diag(self.S + self.epsilon) @ self.Vt
        return F.linear(x, W, self.bias)
```

**2. Anomaly Detection:**

```python
def detect_anomalies(features, defect_direction, temperature=0.1, threshold=4):
    # Compute alignment with defect direction
    feature_norm = F.normalize(features, dim=-1)
    direction_norm = F.normalize(defect_direction, dim=-1)

    logits = -(feature_norm * direction_norm).sum(dim=-1).abs()

    # Identify anomalous tokens
    mask = logits < logits.mean() - threshold * logits.std()
    return mask, logits
```

**3. Neighbor Loss Computation:**

```python
def compute_neighbor_loss(x_token, anomaly_mask, kernel_size=3):
    # Apply 3D Gaussian smoothing to anomalous regions
    prob = torch.exp(logits / temperature)

    # Create sliding windows
    windows = x_token.unfold(0, kernel_size, 1).unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)

    # Weight by Gaussian + non-anomalousness scores
    gaussian_kernel = create_3d_gaussian(kernel_size)
    weights = prob_weights * gaussian_kernel
    weights = weights / weights.sum(dim=(-1,-2,-3), keepdims=True)

    # Compute smoothed representation
    smoothed = (windows * weights[..., None]).sum(dim=(-1,-2,-3))

    # Loss: deviation from smoothed neighbors
    alpha = x_token.norm(dim=-1).mean()  # Normalization factor
    loss = (x_token[anomaly_mask] - smoothed[anomaly_mask]).norm(dim=-1).mean() / alpha

    return loss
```

**4. Layer-Limited Updates:**

```python
if config.limit_layers:
    # Only update layers near detected anomalies
    for layer_idx in range(max(0, anomaly_layer - config.limit_layers + 1)):
        for param in model.blocks[layer_idx].parameters():
            if param.grad is not None:
                param.grad = None  # Freeze earlier layers
```

### SINDER Results

SINDER achieves targeted artifact removal while preserving semantic features:

- **Minimal parameters**: Only `epsilon` values trainable (~0.1% of total params)
- **Localized corrections**: Layer-limited updates prevent global disruption
- **Improved segmentation**: Reduced high-norm token artifacts in biological objects
- **Preserved features**: Maintains downstream task performance

The approach demonstrates that singular defects in self-supervised vision transformers can be systematically identified and corrected through targeted regularization, enabling more robust biological image analysis.
