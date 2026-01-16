# SpatialDINO

_Automated detection and tracking of cellular interactions using self-supervised deep learning_

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/DINOv2-Facebook%20AI-blue.svg)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/Ruff-linter-orange.svg)](https://docs.astral.sh/ruff/)
[![TOML](https://img.shields.io/badge/TOML-configuration-lightgrey.svg)](https://toml.io/en/)

## 👥 Authors

[Alex Lavaee\*](https://www.linkedin.com/in/alexlavaee/), [Arkash Jain\*](https://www.linkedin.com/in/arkashj/), [Gustavo Scanavachi Moreira Campos](https://scholar.google.com/citations?user=w4oASWoAAAAJ&hl=pt-BR), [Jose Inacio Costa-Filho](https://scholar.google.com/citations?user=oZdz4lEAAAAJ&hl=en), [Adam Ingemansson](https://kirchhausen.hms.harvard.edu/people/adam-ingemansson-bs), [Tom Kirchhausen](https://cellbio.hms.harvard.edu/faculty-staff/tomas-kirchhausen)

_Equal contribution, equal first authorship_

**SpatialDINO** uses advanced AI to automatically detect, segment, and track biological objects in 3D light-sheet microscopy data. Key capabilities:

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

This project requires CUDA version 12 or higher. Verify the correct version of CUDA installed by running:

```bash
nvcc --version
```

---

### 1. Install uv

uv is a faster drop-in replacement for conda that we use for environment management. Download and install it via either

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone --recursive git@github.com:kirchhausenlab/spatialdino.git
```

### 3. Create a uv environment:

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
rm -rf spatialdino.egg-info
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

- **Issues**: [GitHub Issues](https://github.com/kirchhausenlab/spatialdino/issues)
- **Contact**: Alex Lavaee ([alavaee@bu.edu](mailto:alavaee@bu.edu)), Araksh Jain ([arkashjain17@gmail.com](mailto:arkashjain17@gmail.com))

---

## 📄 Citation

```bibtex
@article {spatialdino2025,
	author = {Lavaee, Alex and Jain, Arkash and Scanavachi Moreira Campos, Gustavo and Costa-Filho, Jose Inacio and Ingemansson, Adam and Kirchhausen, Tom},
	title = {SpatialDINO: A Self-Supervised 3D Vision Transformer that enables Segmentation and Tracking in Crowded Cellular Environments},
	year = {2026},
	doi = {10.64898/2025.12.31.697247},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/01/02/2025.12.31.697247},
	journal = {bioRxiv}
}
```
