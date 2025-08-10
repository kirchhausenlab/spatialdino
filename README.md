# Cell Interactome 🧬

_Automated detection and tracking of cellular interactions using self-supervised deep learning_

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🔬 For Biologists: What This Tool Does

**Cell Interactome** uses advanced AI to automatically detect, segment, and track biological objects in your microscopy data. Think of it as an intelligent assistant that can:

- **🎯 Detect cellular structures** in 3D light-sheet microscopy images
- **✂️ Segment individual objects** (vesicles, organelles, proteins) with high precision
- **🔗 Track movements** over time to understand cellular dynamics
- **📊 Quantify interactions** between different cellular components

### Key Advantages for Biology Research

- **No manual annotation needed** - learns patterns automatically
- **Works with multi-channel data** (488nm, 560nm, 642nm wavelengths)
- **Handles complex 3D volumes** with temporal sequences
- **Provides quantitative tracking** for statistical analysis

---

## 🚀 Quick Start for Biologists

### 1. **Prepare Your Data**

- Light-sheet microscopy TIFF files
- Multi-channel supported (up to 3 channels)
- 3D volumes with time series

### 2. **Run Analysis Pipeline**

```bash
# Interactive mode for easy use
python scripts/interactive_inf.py
```

### 3. **Get Results**

- Segmented objects as TIFF files
- Tracking data as CSV files
- Visualization-ready outputs for Napari/Fiji

---

## 🛠️ Technical Architecture

This project implements a state-of-the-art **DINO (self-supervised vision transformer)** architecture adapted for 3D biological imaging:

### Core Components

- **SSL Model**: Self-supervised learning using DINO backbone
- **Segmentation Model**: Encoder-decoder architecture for object detection
- **Tracking System**: Temporal linking with feature-based matching
- **Processing Pipeline**: End-to-end automation from raw data to results

### Model Architecture Flow

```
Raw 3D Data → Preprocessing → DINO Features → Segmentation → Instance Detection → Tracking → Results
```

---

## 📁 Project Structure

```
cell_interactome/
├── 📚 README.md                    # This file
├── ⚙️  pyproject.toml              # Package configuration
├── 📋 requirements files           # Dependencies (env.yaml, uv.lock)
│
├── 🧠 src/cell_interactome/        # Core ML modules
│   ├── 🏗️  models/                 # Neural network architectures
│   │   ├── ssl/                   # Self-supervised learning (DINO)
│   │   ├── segmentation/          # Encoder-decoder models
│   │   ├── layers/                # Custom network layers
│   │   └── utils.py               # Model utilities
│   ├── 📊 data/                    # Data handling and transforms
│   │   ├── dataset.py             # Dataset classes
│   │   ├── transforms.py          # Image preprocessing
│   │   └── inference/             # Inference data loaders
│   ├── 🔬 processing/              # Biological data processing
│   │   ├── data_loading.py        # Light-sheet microscopy loading
│   │   └── deconvolution.py       # Image enhancement
│   ├── 🎯 inference/               # Model inference utilities
│   ├── 🔗 tracking/                # Object tracking algorithms
│   ├── 📈 visualization/           # Result visualization
│   └── ⚙️  config/                 # Configuration management
│
├── 🏃 scripts/                     # Executable workflows
│   ├── 🎓 train/                   # Model training scripts
│   │   ├── pretrain.py            # Self-supervised pretraining
│   │   ├── segmentation.py        # Segmentation training
│   │   └── upsample.py            # Feature upsampling
│   ├── 🔍 inference/               # Inference pipeline
│   │   ├── inference.py           # Feature extraction
│   │   ├── segmentation.py        # Object segmentation
│   │   └── postprocessing.py      # Result refinement
│   ├── 📊 data/                    # Data preparation utilities
│   ├── 🎨 visualize/               # Visualization tools
│   ├── 📓 notebooks/               # Jupyter analysis notebooks
│   └── 🖥️  interactive_inf.py      # User-friendly interface
│
├── 🤖 models/                      # Pretrained model weights
├── 📊 reports/                     # Generated results and figures
├── 📖 wiki/                        # Detailed documentation
└── 🎯 icons/                       # UI assets
```

### Key Directories Explained

#### 🧠 `src/cell_interactome/` - Core ML Framework

- **`models/`**: Neural network implementations
  - `ssl/`: DINO self-supervised architecture
  - `segmentation/`: Encoder-decoder for object detection
  - `layers/`: Custom transformer and CNN components
- **`data/`**: Biological data handling optimized for microscopy
- **`processing/`**: Specialized tools for light-sheet microscopy
- **`tracking/`**: Temporal object linking algorithms

#### 🏃 `scripts/` - Research Workflows

- **`train/`**: Model training for different tasks
- **`inference/`**: Complete analysis pipeline
- **`notebooks/`**: Interactive analysis and visualization
- **Interactive tools**: User-friendly interfaces for biologists

---

## 🔄 Biological Workflow Pipeline

### 1. **Data Preprocessing**

```bash
# Prepare and deskew light-sheet data
python scripts/data/prepare.py
python scripts/data/deskew.py
```

### 2. **Feature Extraction**

```bash
# Extract DINO features from 3D volumes
./scripts/inference.sh
```

### 3. **Object Segmentation**

```bash
# Segment individual biological objects
./scripts/segmentation.sh
```

### 4. **Tracking Analysis**

Run the notebook `scripts/notebooks/run_tracking/tracking_tests.ipynb` to get the tracking results.

### 5. **Visualization & Analysis**

Run the first part of the notebook `scripts/notebooks/visualize_features.ipynb` to visualize all features and get detailed pdfs.

---

## 💻 Installation

### Prerequisites

- **Python 3.12+**
- **CUDA 12.x** (for GPU acceleration)
- **16GB+ RAM** (32GB recommended for large datasets)

### Quick Installation

```bash
# Clone repository
git clone --recursive https://github.com/kirchhausenlab/cell_interactome.git
cd cell_interactome

# Create environment
mamba env create -f env.yaml -n cell3d

mamba activate cell3d
pip install -e .
pip install natsort rich tqdm click
```

---

## 📊 Understanding Results

### Output Files

- **`lr_feats.tif`**: Extracted DINO features
- **`segmentation.tif`**: Object masks
- **`centroids.csv`**: Object positions
- **`tracks.csv`**: Temporal trajectories
- **`visualization/`**: Images for biological interpretation

### Biological Interpretation

- **Centroids**: Precise 3D locations of detected objects
- **Tracks**: Movement patterns over time
- **Features**: High-dimensional representations for similarity analysis
- **Segmentation**: Binary masks for quantitative analysis

---

## 🔧 Configuration

Key configuration files:

- **`src/cell_interactome/config/inference.yaml`**: Inference parameters
- **`src/cell_interactome/config/pretrain.yaml`**: Pretraining parameters

---

## 📚 Additional Resources

### 📖 Documentation

- **[Getting Started Guide](wiki/getting_started.md)**: Step-by-step tutorial
- **[Pipeline Overview](wiki/pipeline.md)**: Detailed workflow explanation
- **[Technical Details](wiki/spatial_dino.pdf)**: Research paper and methods

### 🎥 Video Tutorials

- **[Running Inference](wiki/run_spatial_dino/)**: Video demonstrations

---

## 🤝 Contributing

We welcome contributions from both biologists and developers!

### For Biologists

- Report issues with specific datasets
- Request new features for your research
- Share example data for testing

### For Developers

- Implement new model architectures with notes found in `wiki/supplementary` folder.
- Pretrain the model with `scripts/train/pretrain.py`

---

## 📄 Citation

If you use Cell Interactome in your research, please cite:

```bibtex
@article{cell_interactome2024,
  title={Cell Interactome: Automated Detection of Cellular Interactions using Self-Supervised Deep Learning},
  author={[Your Names]},
  journal={[Journal Name]},
  year={2024}
}
```

---

## 📞 Support

- **🐛 Issues**: [GitHub Issues](https://github.com/kirchhausenlab/cell_interactome/issues)
- **📧 Email**: Araksh Jain [arkashjain17@gmail.com](mailto:arkashjain17@gmail.com), Alex Lavaee [alavaee@bu.edu](mailto:alavaee@bu.edu)

---

## 🏆 Acknowledgments

This project builds upon:

- **DINO**: Self-supervised vision transformers ([Caron et al., 2021](https://arxiv.org/abs/2104.14294))

# Cell Interactome

## Table of Contents

1. [Machine Setup](#Machine-Setup) 💻
2. [Cuda Installation](#Cuda-Installation) 🛠️
3. [Introductions](#Introduction) 📘
4. [Installation](#Installation) 🛠️
5. [Convert Pretrained Weights](#Convert-pretrained-weights) 🔄
6. [Training](#Training) 🏃

### Machine Setup 💻

You can use the following machines to run the Cell Interactome pipeline:

1. Ubuntu <img src="https://user-images.githubusercontent.com/25181517/186884153-99edc188-e4aa-4c84-91b0-e2df260ebc33.png" width="15">
2. MacOS <img src="https://user-images.githubusercontent.com/25181517/186884152-ae609cca-8cf1-4175-8d60-1ce1fa078ca2.png" width="15"> [*Please Use Docker*] <img src="https://user-images.githubusercontent.com/25181517/117207330-263ba280-adf4-11eb-9b97-0ac5b40bc3be.png" width="15">
3. Windows - <img src="https://user-images.githubusercontent.com/25181517/186884150-05e9ff6d-340e-4802-9533-2c3f02363ee3.png" width="18"> [*Please Use Docker*]<img src="https://user-images.githubusercontent.com/25181517/117207330-263ba280-adf4-11eb-9b97-0ac5b40bc3be.png" width="18">

### CUDA Installation 🛠

This project requires CUDA version 12.x. Veri️fy the correct version of CUDA installed by running the following command:

```bash
nvcc --version
```

### Installation 🛠️

1. Clone the repository:

```bash
git clone --recursive git@github.com:kirchhausenlab/cell_interactome.git
```

2. Create a conda environment.

```bash
mamba env create -f env.yaml -n cell3d
mamba activate cell3d
pip install -e .
pip install natsort rich tqdm click xformers
```

📦 In case of errors, please ensure you have the required dependencies for Python installed:

Delete the cache.

```bash
rm -rf ~/.cache/
rm -rf cell_interactome.egg-info
```

```bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
ninja-build cmake libegl1-mesa-dev python3-dev
```

## Training

If you do not have a .bashrc file, create one.

```
touch ~/.bashrc
```

To your .bashrc file, add the following:
**XDG** is a standard for cross-platform directory layout.

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

**NCCL** is a nvidia's communication library.

```bash
export NCCL_SOCKET_NTHREADS=4 # number of threads per socket
export NCCL_NSOCKS_PERTHREAD=4 # number of sockets per thread
export NCCL_IB_DISABLE=0 # enable Infiniband
export NCCL_IB_HCA="mlx5" # use Mellanox Infiniband
export CUDA_HOME="/usr/local/cuda-12" # choose the correct CUDA version
export PATH=$CUDA_HOME/bin:$PATH
export CPATH="$CUDA_HOME/include:$CPATH"
```

**Add c++ library**

```bash
export CXX=g++
```

**Distributed training**

```bash
export NCCL_SOCKET_IFNAME=ib # use all infiniband interfaces
export RDZV_BACKEND="c10d"
export OMP_NUM_THREADS=16
export NUM_ALLOWED_FAILURES=3

export RDZV_ID="2001" # set the rdvz id to be the same for all nodes
export MASTER_PORT="29500" # set the master port to be the same for all nodes, it can be any random number
```

To get the Master Address, get the IP address of your infiniband interface. Say you have 3 nodes, you need to set one node to be your master node. Run `ibstat` to confirm infiniband ids. One done, do as follows:

```bash
ibstat
```

Which will give you the following output:

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

In this case, the master node is `10.1.0.11` where inet is the IP address of the master node.

```bash
export MASTER_ADDR="10.1.0.11"
export RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT"
```

If you chose another machine to be your master node, run `ifconfig` and get the IP address of the machine. To run multi-node training or inference, you will need to ssh into all the machines. For the master node, run `export NODE_RANK=0`, for the other nodes, run `export NODE_RANK=1, 2, 3, ...`. Once done, run the command below starting with the master node and in consectuive order of node ranks running the same command. The example below is for 3 nodes with 8 GPUs per node.

**Note this is done for the same set of hardware for all nodes.**

```bash
# Meaning of the arguments:
# --nnodes: number of nodes (e.g. n)
# --node_rank: rank of the node (e.g. 0, 1, 2, 3, ... n) for n nodes
# --nproc_per_node: number of processes/GPUs per node (e.g. 8)
# --master_addr: address of the master node (e.g. 10.10.10.10)
# --master_port: port of the master node (e.g. 29500)

torchrun --nnodes 3 --nproc_per_node 8 --node_rank $NODE_RANK --rdzv-id $RDZV_ID --rdzv-backend $RDZV_BACKEND --rdzv-endpoint $RDZV_ENDPOINT scripts/train/pretrain.py
```

## Inference

**To run inference and segmentation, you simply need to run the interactive cli** `python3 scripts/interactive_inf.py`

Similar to how you setup training, you can setup multi-node inference. However, we usually run inference on a single node, to run it you do not need to worry about the master node/children nodes.

**Input size**
Note that for efficient inference, we recommend using a maximum [Z,Y,X] input of size [150, 750, 750] at the maximum, however this is pertaining to the total volume of Z.X.Y = 84,000,000 pixels^3. Basically you could have volumes of sizes [300, 300, 300], [80, 1000, 500] and so on.
**Example inference script**

```bash
#!/bin/bash

folder_path="/nfs/data1expansion/datasync3/Gustavo/20210422_0p5_0p55_sCMOS_Gu_AP2/CS1_Ap2_live_3colorsDic/Ex07_488_60mW_z0p5/ch488nmCamA/DS"
number_of_files=1 # -1 for all timepoints, otherwise chose a number
save_path="/raid1/cme_tests/results/ablations/ap2_test"
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

torchrun --nnodes 1 --node_rank 0 --nproc_per_node $NUM_PROC_PER_NODE --rdzv_endpoint=localhost:9999 ./scripts/inference/inference.py \
  file_path="$folder_path" \
  save_path="$save_path" \
  number_of_files=$number_of_files \
  crop_params="[0,0,0,0,0,0]"
```

**Explain the script**

- `folder_path`: path to the folder containing the images
- `number_of_files`: number of files to process, -1 for all files
- `save_path`: path to save the results
- `OMP_NUM_THREADS`: number of threads to use
- `CUDA_VISIBLE_DEVICES`: list of GPUs to use
- `NUM_PROC_PER_NODE`: number of processes/GPUs per node
- `torchrun`: command to run the inference
- `scripts/inference/inference.py`: path to the inference script
- `file_path`: path to the folder containing the images
- `save_path`: path to save the results
- `crop_params`: parameters for cropping the images

## Segmentation

**To run segmentation, you simply need to run the interactive cli** `python3 scripts/interactive_seg.py`
Similar to the inference script, you can run segmentation on a single node or multi-node.

**Example segmentation script**

```bash
#!/bin/bash
save_path="/raid1/cme_tests/results/ap2_latest" #"/nfs/scratch2/shared_image_recog_ml/spatial_dino_exp/zeiss_experiments/zeiss_560_left"
blur_image=False # true for most datasets
blur_factor=1.0 # unless you have very low snr for which I'd recommend setting this to 3, use 1.0 for most datasets
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
use_raw_mask=False # Unless you have the non-care'd virus, with a snr of <3, please set this to False
spot_sigma=2.0 # for most endosomes, viruses
outline_sigma=2.0
attention_weight=0.5
background_removal_radius=10
skip_existing=False # if you ran a few experiments, then set this to True, but if you have never run any experiments, then set this to False

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$NUM_PROC_PER_NODE --rdzv_endpoint=localhost:8999 ./scripts/inference/segmentation.py \
  file_path="$save_path" \
  save_path="$save_path" \
  skip_existing=$skip_existing \
  background_removal_radius=$background_removal_radius \
  attention_weight=$attention_weight \
  blur_factor=$blur_factor \
  use_raw_mask=$use_raw_mask \
  outline_sigma=$outline_sigma \
  spot_sigma=$spot_sigma \
  blur_image=$blur_image \
  outline_sigma=$outline_sigma
```

You should checkout the `scripts/notebooks/segmentation_misc/test_segmentation.ipynb` for tuning parameters for segmentation. In the set of screenshots below, you can see how I change the parameters to get good results. For examples where SNR is low, you should increase the blur_factor for the guassian blur and see the results in the notebook. Usually, you should set the blur_factor to 1.0 for most datasets when the SNR > 3.
