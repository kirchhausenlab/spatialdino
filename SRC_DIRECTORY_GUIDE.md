# 📁 Source Code Architecture Guide

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

#### **`data_loading.py`** - Light Sheet Microscopy Support

- **`LightSheetMetadata`**: Metadata extraction from microscopy files
- **`DirectoryParser`**: Automated file structure parsing
- **Multi-channel support**: 488nm, 560nm, 642nm wavelengths
- **Anisotropy correction**: Converting non-isotropic voxels
- **Frame rate detection**: Automatic temporal resolution parsing

#### **`deconvolution.py`** - Image Enhancement

- Richardson-Lucy deconvolution
- Point spread function estimation
- Noise reduction algorithms
- Resolution enhancement

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

## 📈 **Visualization Module** (`visualization/`)

**Purpose**: Scientific visualization for biological analysis.

### Components

- **`visualize.py`**: Main visualization utilities
- 3D volume rendering
- Multi-channel overlay
- Tracking trajectory visualization
- Quantitative plots and statistics

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

## 🔧 **Usage Patterns**

### For Biologists

```python
# Load and process microscopy data
from cell_interactome.processing.data_loading import LightSheetMetadata
from cell_interactome.inference import inference_3d

# Run inference on your data
results = inference_3d.predict(your_microscopy_data)
```

### For ML Researchers

```python
# Extend the segmentation model
from cell_interactome.models.segmentation import Segmentation
from cell_interactome.models.layers import Encoder, Decoder

# Create custom architecture
custom_model = Segmentation(config)
```

### For Developers

```python
# Add new loss functions
from cell_interactome.loss import SoftNCuts
from torch import nn

class CustomLoss(nn.Module):
    # Your implementation
```

---

## 🧪 **Testing Structure**

The codebase includes comprehensive testing:

- **Unit tests**: In `data/tests/`
- **Integration tests**: Cross-module functionality
- **Performance tests**: GPU memory and speed optimization
- **Biological validation**: Against known ground truth

---

## 📚 **Extension Points**

The architecture is designed for easy extension:

1. **New Model Architectures**: Add to `models/`
2. **Custom Data Loaders**: Extend `data/dataset.py`
3. **Novel Loss Functions**: Add to `loss/`
4. **Specialized Processing**: Extend `processing/`
5. **Custom Tracking**: Add algorithms to `tracking/`

---

## 💡 **Best Practices**

### Code Organization

- **Modularity**: Each component has a single responsibility
- **Configuration-driven**: Use YAML files for all parameters
- **Type hints**: Comprehensive type annotations
- **Documentation**: Docstrings for all public functions

### Performance

- **Memory efficiency**: Sliding window processing for large volumes
- **GPU optimization**: Multi-GPU support throughout
- **Caching**: Intelligent caching of computed features
- **Batch processing**: Vectorized operations where possible

### Biological Considerations

- **Multi-channel support**: Native support for microscopy data
- **Anisotropic handling**: Proper 3D spacing considerations
- **Time series**: Temporal consistency in tracking
- **Scale awareness**: Multi-scale processing for different magnifications

---

_This architecture enables cutting-edge research while maintaining production-ready code quality for biological applications._
