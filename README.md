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

**SpatialDINO** uses advanced AI to automatically detect, segment, and track biological objects in 3D light-sheet microscopy data.

---

## 📂 Public Data Access

All datasets and pre-trained models are publicly available on AWS S3.

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

In the repository directory, run
```bash
uv venv --python 3.12
uv sync --all-packages
```

This creates a single root `.venv` shared by the core `spatialdino` package and the GUI server in
`apps/server`.

---

## Inference

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

uv run torchrun --nnodes 1 --node_rank 0 --nproc_per_node $NUM_PROC_PER_NODE \
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

## Segmentation

### Work in progress

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

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/kirchhausenlab/spatialdino/issues)
- **Contact**: Alex Lavaee ([alavaee@bu.edu](mailto:alavaee@bu.edu)), Arkash Jain ([arkashjain17@gmail.com](mailto:arkashjain17@gmail.com))

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
