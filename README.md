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
python scripts/inference/inference.py
```

### 3. **Object Segmentation**

```bash
# Segment individual biological objects
python scripts/inference/segmentation.py
```

### 4. **Tracking Analysis**

```bash
# Link objects across time points
python scripts/tracking/track_objects.py
```

### 5. **Visualization & Analysis**

```bash
# Generate results for biological interpretation
python scripts/visualize/generate_reports.py
```

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

### Verify Installation

```bash
python -c "import cell_interactome; print('✅ Installation successful!')"
```

---

## 🧪 Usage Examples

### For Biologists: Interactive Mode

```bash
# Launch user-friendly interface
python scripts/interactive_inf.py
```

### For Developers: Command Line

```bash
# Run inference on specific data
python scripts/inference/inference.py \
    inference.img_path="./data/sample.tif" \
    model.weights="./models/pretrained.pth"

# Batch processing
python scripts/inference/segmentation.py \
    file_path="./data/experiment_folder/"
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
- **`src/cell_interactome/config/models/`**: Model architectures

### Important Parameters for Biology

```yaml
# Inference settings
batch_size: 1 # Adjust based on GPU memory
overlap: 0.25 # Overlap between patches
upsample_factor: 2 # Feature resolution enhancement

# Segmentation settings
spot_sigma: 2 # Object detection sensitivity
outline_sigma: 2 # Boundary refinement
min_object_size: 10 # Filter small artifacts
```

---

## 📚 Additional Resources

### 📖 Documentation

- **[Getting Started Guide](wiki/getting_started.md)**: Step-by-step tutorial
- **[Pipeline Overview](wiki/pipeline.md)**: Detailed workflow explanation
- **[Technical Details](wiki/spatial_dino.pdf)**: Research paper and methods

### 🎥 Video Tutorials

- **[Running Inference](wiki/run_spatial_dino/)**: Video demonstrations
- **[Segmentation Workflow](wiki/run_spatial_dino/)**: Step-by-step examples

### 💡 Example Data

- **[Experiments](wiki/experiments/)**: Sample datasets and results
- **[Analysis Notebooks](scripts/notebooks/)**: Interactive examples

---

## 🤝 Contributing

We welcome contributions from both biologists and developers!

### For Biologists

- Report issues with specific datasets
- Request new features for your research
- Share example data for testing

### For Developers

- Implement new model architectures
- Optimize performance for larger datasets
- Add support for new microscopy formats

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
- **💬 Discussions**: [GitHub Discussions](https://github.com/kirchhausenlab/cell_interactome/discussions)
- **📧 Email**: [contact@kirchhausenlab.org](mailto:contact@kirchhausenlab.org)

---

## 🏆 Acknowledgments

This project builds upon:

- **DINO**: Self-supervised vision transformers ([Caron et al., 2021](https://arxiv.org/abs/2104.14294))
