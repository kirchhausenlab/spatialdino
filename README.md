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

[Alex Lavaee\*](https://www.linkedin.com/in/alexlavaee/), [Arkash Jain\*](https://www.linkedin.com/in/arkashj/), [Gustavo Scanavachi](https://scholar.google.com/citations?user=w4oASWoAAAAJ&hl=pt-BR), [Jose Inacio](https://scholar.google.com/citations?user=oZdz4lEAAAAJ&hl=en), [Tom Kirchhausen](https://cellbio.hms.harvard.edu/faculty-staff/tomas-kirchhausen)

_Equal contribution, equal first authorship_

**Cell Interactome** uses advanced AI to automatically detect, segment, and track biological objects in 3D light-sheet microscopy data. Key capabilities:

- **🎯 Automated detection** of cellular structures (vesicles, organelles, proteins)
- **✂️ Precise segmentation** with no manual annotation required
- **🔗 Temporal tracking** for cellular dynamics analysis
- **📊 Multi-channel support** (488nm, 560nm, 642nm wavelengths)

---

## 🧪 Experiment Results Gallery

The following showcase demonstrates segmentation results across various biological experiments, highlighting the pipeline's versatility across different cellular structures and imaging conditions.

| Experiment & Description                                                                      | Segmentation Results                                                    |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **AP2** - Clathrin-coated pits, endocytic proteins with easily fitted PSF<br/>_blur=1_        | ![AP2 Segmentation](scripts/images/ap2.png)                             |
| **Apilimod** - Small molecule that binds to ER and inhibits endocytic pathway<br/>_blur=1_    | ![Apilimod Segmentation](scripts/images/apilimod.png)                   |
| **Dextran** - Fluid phase markers for endosomal uptake<br/>_blur=1_                           | ![Dextran Segmentation](scripts/images/dextran.png)                     |
| **Membranes** - Cellular membrane structures<br/>_blur=1_                                     | ![Membranes Segmentation](scripts/images/membranes.png)                 |
| **Mitochondria** - Mitochondrial networks and dynamics<br/>_blur=1_                           | ![Mitochondria Segmentation](scripts/images/mitochondria.png)           |
| **Nuclei** - Nuclear segmentation and morphology<br/>_blur=1_                                 | ![Nuclei Segmentation](scripts/images/nuclei.png)                       |
| **Nuclei (Large)** - Large-scale nuclear imaging<br/>_spot_sigma=10_                          | ![Nuclei Big Segmentation](scripts/images/nuclei_big.png)               |
| **Transferrin (488nm)** - Iron transport protein, Channel 488nm<br/>_blur=3_                  | ![Transferrin 488 Segmentation](scripts/images/transferrin_chan488.png) |
| **Transferrin (642nm)** - Iron transport protein, Channel 642nm (pH insensitive)<br/>_blur=3_ | ![Transferrin 642 Segmentation](scripts/images/transferrin_chan642.png) |

**Additional Files Available for Each Experiment:**

- `instance_seg.tif` - 3D instance segmentation masks
- `patch_tokens.tif` - DINO feature visualizations
- `volume.tif` - Original volume data
- `segmentation.png` - 2D visualization shown above

---

## 📂 Public Data Access

**For researchers and biologists**: All datasets and pre-trained models are publicly available on AWS S3.

### S3 Bucket Structure

![AWS URI Structure](scripts/images/uri.png)

| **Datasets**                                     | **Models**                                    |
| ------------------------------------------------ | --------------------------------------------- |
| ![Dataset Structure](scripts/images/dataset.png) | ![Model Structure](scripts/images/models.png) |

### Download Commands

**Download datasets:**

```bash
aws s3 cp s3://spatialdino/dataset_part1/ ./datasets/ --recursive --no-sign-request
```

**Download models:**

```bash
aws s3 cp s3://spatialdino/models/ ./models/ --recursive --no-sign-request
```

**List available data:**

```bash
aws s3 ls s3://spatialdino/ --no-sign-request
```

---

## 📚 Resources

**Play around with the interactive CLI to get a feel for the pipeline and then tune the actual scripts. If you get stuck, please watch the video tutorials before asking questions.**

- **[Video Tutorials](wiki/run_spatial_dino/)**: Workflow demonstrations
- **[Technical Details](wiki/spatial_dino.pdf)**: Overview of the pipeline
- **[Technical Explanations](wiki/technical_summary/)**: Detailed explanations of the pipeline and advanced technical details
- **[Presentations](wiki/presentations/)**: Slides from presentations showing evolution of the project
- **[Interesting Papers](wiki/interesting_papers/)**: Some key papers that inspired the project
- **[Getting Started](wiki/getting_started.md)**: Step-by-step guide

**Tip for Non-technical users**: For people unfamiliar with pulling code from github, you need to first setup an ssh key and add the key to your github account. Information is provided on this path `wiki/setup_ssh.md`.

**Interested in infrastructure setup?** For low-level infrastructure details and configuration notes, see our GitHub issue on the PyTorch repository: [PyTorch issue #144779 — low-level infra details](https://github.com/pytorch/pytorch/issues/144779).

---

## 📋 Table of Contents

### 🚀 Getting Started

1. [Experiment Results Gallery](#-experiment-results-gallery) - Comprehensive experiment showcase
2. [Resources](#-resources) - Video tutorials, technical docs, and guides
3. [Machine Setup](#machine-setup-💻) - Ubuntu/MacOS/Windows compatibility guide
4. [CUDA Installation](#cuda-installation-🛠️) - GPU requirements and setup verification
5. [Installation](#installation-🛠️) - Complete setup process
   - [Miniforge & Mamba](#1-install-miniforge--mamba) - Environment management
   - [Repository Clone](#2-clone-the-repository) - Source code download
   - [Environment Creation](#3-create-a-conda-environment) - Dependencies installation
   - [Troubleshooting](#-troubleshooting) - Common issues and fixes

### 🔬 Core Pipeline

6. [Inference](#inference-🔍) - Feature extraction from 3D volumes
   - [Interactive CLI](#inference-🔍) - Recommended user interface
   - [Input Size Guidelines](#input-size-guidelines) - Volume constraints
   - [Example Scripts](#example-inference-script) - Batch processing setup
   - [Script Parameters](#script-parameters) - Configuration options
7. [Segmentation](#segmentation-✂️) - Instance mask generation
   - [Interactive CLI](#segmentation-✂️) - User-friendly interface
   - [Example Scripts](#example-segmentation-script) - Automation examples
   - [Parameter Tuning](#parameter-tuning) - SNR-specific optimization
8. [Training](#training-🏃) - Multi-node self-supervised model training
   - [Environment Setup](#environment-setup) - Configuration variables
   - [Multi-Node Setup](#multi-node-setup) - Distributed training configuration
   - [Training Command](#training-command) - Execution examples

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
13. [Contributions](#-contributions) - Summary of contributions
14. [Support](#-support) - Help and contact information
15. [Citation](#-citation) - Academic reference

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

### 1. Install Miniforge & Mamba

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

### 2. Clone the repository:

```bash
git clone --recursive git@github.com:kirchhausenlab/cell_interactome.git
```

### 3. Create a conda environment:

```bash
mamba env create -f env.yaml -n cell3d
mamba activate cell3d
pip install -e . && pip install natsort rich click xformers wheel setuptools
```

**Incase of torch issues, check your Cuda version at `/usr/local/` and install torch from source `pip install torch --index-url https://download.pytorch.org/whl/cu121`, where you can change cu121 for your version. Say you have `cuda-12.1` installed, then you can install torch with `pip install torch --index-url https://download.pytorch.org/whl/cu121`.**

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

## Inference 🔍

> **⚠️ Important**: Before running inference, ensure you have the pretrained model. Use the model path `../models/backbone.pth` which contains the pretrained weights for the DINO vision transformer.

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

When you run the code, you will see the following interface:

![CLI Interface](scripts/images/cli_image.png)

### 1. Feature Extraction

To run an inference, first you need to setup the configuration file.

![Inference Configuration](scripts/images/inference_config.png)

Then you need to select the path to the data.

![Path Selection](scripts/images/inference_select_paths.png)

Finally, you need to setup the parameters for the inference.

![Setup Interface](scripts/images/inference_setup.png)

**Configure and extract DINO features from 3D volumes:**

- Multi-GPU distributed processing
- Automatic volume preprocessing and normalization
- Sliding window inference for large datasets

### 2. Segmentation

![Segmentation Interface](scripts/images/segmentation.png)

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
|       ![Transferrin Config](scripts/notebooks/segmentation_misc/sample_experiments/transferrin_lowsnr/config.png)       | ![Simulated Segmentation](scripts/notebooks/segmentation_misc/sample_experiments/simulated_lowsnr/segmentation.png) |
| ![Transferrin Segmentation](scripts/notebooks/segmentation_misc/sample_experiments/transferrin_lowsnr/segmentation.png) |       ![Simulated Config](scripts/notebooks/segmentation_misc/sample_experiments/simulated_lowsnr/config.png)       |

**Moving data**
Once done, you can copy files from one location to another, delete them, etc.

![File Operations](scripts/images/file_operations.png)

### 3. Tracking & Visualization

Run the tracking notebook in the `scripts/notebooks/run_tracking/tracking_main.ipynb` folder.

![Tracking](scripts/images/tracking.png)

When done you can run the `scripts/notebooks/quantify_tracks/tracking_experiments_2chan.ipynb` notebook to visualize tracks.

![Track Visualization](scripts/images/tracks.png)

**Temporal object linking and analysis:**

- Physics-based velocity prediction
- Feature-enhanced trackpy integration
- Multi-channel track visualization in Napari

![CLI Summary](scripts/images/summary_of_cli.png)

---

## Training 🏃

### Environment Setup

If you do not have a `.bashrc` file, create one:

```bash
touch ~/.bashrc
```

Add the following to your `.bashrc` file:

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

## 📊 Results & Features

### Output Files

- **`lr_feats.pt`**: DINO feature maps
- **`segmentation.tif`**: Instance masks
- **`centroids.csv`**: Object coordinates
- **`tracks.csv`**: Temporal trajectories

### Feature Visualization

Get comprehensive upsampled features as demonstrated in-
In the codebase, find the pdf at the link below:
`https://github.com/kirchhausenlab/cell_interactome/tree/main/scripts/notebooks/feature_visualizations/ap2/ap2_latest_ex07_CamA_ch0_stack0000_488nm_0000000msec_0087860321msecAbs_000x_000y_000z_0000t/features`
For Kirchhausen lab members find the pdf at the link below:
`/nfs/datasync4/spatial_dino/code/cell_interactome/scripts/notebooks/feature_visualizations/ap2/ap2_latest_ex07_CamA_ch0_stack0000_488nm_0000000msec_0087860321msecAbs_000x_000y_000z_0000t/features/`

You can also run the `scripts/notebooks/upsample_feats.py` notebook to visualize the features.
![Feature Visualization](scripts/images/upsample.png)

The files will look like this:

![Feature Visualization](scripts/images/upsample_dextran.png)

**The detailed_analysis.pdf is an ordered list of features going from most variance to least variance.**

![Feature Visualization](scripts/images/visualize_features.png)

## 📁 Architecture

## 🧠 `src/cell_interactome/` - Core Framework

This document provides a comprehensive overview of the source code organization for the Cell Interactome project. The codebase is designed with modularity and extensibility in mind, supporting both research experimentation and production deployment.

---

## 📊 **Main Modules Overview**

```
src/cell_interactome/
├── 🏗️  models/              # Neural network architectures & training
├── 📊 data/                 # Data handling & preprocessing
├── 🔬 processing/           # Biological data processing
├── 🎯 inference/            # Model inference utilities
├── 🔗 tracking/             # Object tracking algorithms
├── 💸 loss/                 # Loss functions for training
├── ⚙️  config/              # Configuration management
├── 🔄 optim/                # Optimization algorithms
├── 📈 visualization/        # Result visualization
├── 🛠️  utils/               # Utility functions
└── 📝 logging/              # Training & inference logging
```

---

## 🏗️ **Models Module** (`models/`)

**Purpose**: Contains all neural network architectures and model definitions.

### Core Architecture Components

#### **Self-Supervised Learning** (`ssl/`)

- **Main Model**: `SSL` class - DINO-based self-supervised learning
- **Purpose**: Learns feature representations without manual labels
- **Key Features**:
  - Vision transformer backbone
  - Teacher-student training paradigm
  - Multi-scale feature extraction

#### **Segmentation** (`segmentation/`)

- **Main Model**: `Segmentation` class - Encoder-decoder architecture
- **Purpose**: Pixel-level object detection and segmentation
- **Key Features**:
  - 3D volume processing
  - Multi-scale feature fusion
  - Instance segmentation capabilities

#### **Network Layers** (`layers/`)

Essential building blocks for all models:

- **`encoder.py`**: Vision transformer encoder with 3D patches
- **`decoder.py`**: Feature decoder for reconstruction tasks
- **`attention.py`**: Multi-head attention mechanisms
- **`patch_embed.py`**: 3D patch embedding layers
- **`pos_embed.py`**: Positional encoding for transformers
- **`mlp.py`**: Multi-layer perceptron components
- **`dino_head.py`**: DINO-specific prediction heads
- **`unet.py`**: U-Net architecture components

#### **Specialized Models**

**DINO Variants** (`dino3d/`, `dinov2/`):

- 3D adaptations of DINO architecture
- Different model scales (small, base, large)
- Custom vision transformer implementations

**Lift Model** (`lift/`):

- Feature upsampling and enhancement
- Super-resolution for biological imaging
- Multi-scale feature fusion

**Sinder** (`sinder/`):

- Specialized defect detection
- Singular value analysis
- Image repair mechanisms

---

## 📊 **Data Module** (`data/`)

**Purpose**: Handles all data loading, preprocessing, and augmentation for biological imaging.

### Core Components

#### **Dataset Classes**

- **`dataset.py`**: Main dataset implementations

  - 3D volume datasets
  - Multi-channel microscopy support
  - Temporal sequence handling

- **`dataloader.py`**: Efficient data loading
  - Multi-GPU support
  - Memory optimization
  - Batch processing

#### **Data Transformations** (`transforms.py`)

Specialized preprocessing for biological data:

- **Normalization**: Channel-wise and global normalization
- **Augmentation**: 3D rotations, flips, noise injection
- **Scaling**: Anisotropic to isotropic conversion
- **Cropping**: Smart cropping for memory efficiency

#### **Specialized Data Handling**

**Inference Pipeline** (`inference/`):

- **`inference.py`**: Production inference dataset
- Memory-efficient loading for large volumes
- Sliding window processing
- Multi-scale inference support

**Post-processing** (`postprocessing/`):

- Result refinement datasets
- Segmentation mask processing
- Feature aggregation

**DINO-specific** (`dino3d/`, `dinov2/`):

- Self-supervised learning data pipelines
- Augmentation strategies for contrastive learning
- Multi-view generation

---

## 🔬 **Processing Module** (`processing/`)

**Purpose**: Biological data preprocessing and enhancement specific to microscopy.

### Key Components

---

## 🎯 **Inference Module** (`inference/`)

**Purpose**: Production-ready model inference utilities.

### Components

#### **`inference_3d.py`** - 3D Volume Inference

- Sliding window processing for large volumes
- Multi-GPU inference support
- Memory-efficient computation
- Patch-based processing with overlap handling

#### **`utils.py`** - Inference Utilities

- Model loading and initialization
- Configuration management
- Output post-processing
- Performance optimization

---

## 🔗 **Tracking Module** (`tracking/`)

**Purpose**: Temporal object tracking across time series.

### Key Components

#### **`utils.py`** - Core Tracking Algorithms

- **Hungarian algorithm**: Optimal assignment for object matching
- **Feature-based matching**: Using learned representations
- **Temporal consistency**: Maintaining track continuity
- **Track quality assessment**: Filtering spurious tracks

#### **`simple_decay_correction.py`** - Biological Corrections

- Photobleaching correction
- Intensity normalization over time
- Biological drift compensation

---

## 💸 **Loss Module** (`loss/`)

**Purpose**: Training loss functions optimized for biological imaging.

### Loss Functions

#### **Self-Supervised Losses**

- **`dino_clstoken_loss.py`**: DINO classification token loss
- **`ibot_patch_loss.py`**: Patch-level contrastive loss
- **`koleo_loss.py`**: Feature diversity regularization

#### **Segmentation Losses**

- **`soft_ncuts.py`**: Soft normalized cuts for segmentation
- **`charbonnier_loss.py`**: Robust reconstruction loss
- **`fourier_loss.py`**: Frequency domain loss

---

## ⚙️ **Config Module** (`config/`)

**Purpose**: Centralized configuration management using YAML files.

### Configuration Files

#### **Model Configurations** (`models/`)

- **`vitb8.yaml`**: Vision Transformer Base with 8x8 patches
- **`vits8.yaml`**: Vision Transformer Small with 8x8 patches

#### **Task Configurations**

- **`inference.yaml`**: Inference pipeline settings
- **`segmentation.yaml`**: Segmentation training parameters
- **`pretrain.yaml`**: Self-supervised pretraining config
- **`upsampler.yaml`**: Feature upsampling settings

---

## 🔄 **Optimization Module** (`optim/`)

**Purpose**: Advanced optimization algorithms for deep learning.

### Components

- **`lars.py`**: Large batch training optimizer
- **`lr_sched.py`**: Learning rate scheduling
- **`lr_decay.py`**: Learning rate decay strategies

---

## 🛠️ **Utils Module** (`utils/`)

**Purpose**: Shared utility functions across the project.

### Core Utilities

#### **General Utils**

- **`utils.py`**: Common helper functions
- **`misc.py`**: Miscellaneous utilities
- **`dcp_to_pth.py`**: Model format conversion

#### **Specialized Utils**

**Inference Utils** (`inference/`):

- **`instance_seg_utils.py`**: Instance segmentation utilities
- Object detection post-processing
- Centroid calculation
- Feature extraction from segmentations

**Tracking Utils** (`tracking/`):

- **`tracking_utils.py`**: Advanced tracking algorithms
- **`plotting.py`**: Tracking visualization
- Trajectory analysis
- Track quality metrics

**Model-specific Utils** (`dino3d/`, `dinov2/`):

- Model initialization
- Weight loading/saving
- Feature extraction utilities

---

## 📝 **Logging Module** (`logging/`)

**Purpose**: Comprehensive logging for training and inference.

### Components

- **`wandb.py`**: Weights & Biases integration
- **`utils.py`**: Logging utilities
- Training progress tracking
- Metric visualization
- Experiment management

---

_This architecture enables cutting-edge research while maintaining production-ready code quality for biological applications._

---

## 🔧 Configuration

**Key parameters in `src/cell_interactome/config/inference.yaml`:**

- `chunk_size`: Processing volume size
- `patch_size`: Transformer patch dimensions
- `stride`: Inference overlap
- `isotropic_scale_factor`: Spatial normalization

---

## 🤝 Contributions

**Key contributions of this work:**

- **Self-supervised learning**: DINO-based feature extraction for biological imaging without manual annotation
- **3D processing**: Full volumetric analysis of light-sheet microscopy data
- **Multi-scale inference**: Efficient processing of large biological datasets
- **Instance segmentation**: Attention-guided clustering for precise object detection
- **Temporal tracking**: Physics-based object linking across time series
- **Interactive tools**: User-friendly CLI interfaces for non-technical users

**Technical innovations:**

- 3D vision transformer adaptation for biological imaging
- Attention density-guided segmentation refinement
- Multi-channel processing pipeline for fluorescence microscopy
- Distributed training framework for large-scale self-supervised learning

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/kirchhausenlab/cell_interactome/issues)
- **Contact**: Alex Lavaee ([alavaee@bu.edu](mailto:alavaee@bu.edu)), Araksh Jain ([arkashjain17@gmail.com](mailto:arkashjain17@gmail.com))

---

## 📄 Citation

```bibtex
@article{cell_interactome2025,
  title={Cell Interactome: Automated Detection of Cellular Interactions using Self-Supervised Deep Learning},
  author={Alex Lavaee ⃰, Arkash Jain ⃰, Tom Kirchhausen},
  journal={[Journal]},
  year={2025}
}
```
