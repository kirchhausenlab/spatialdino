# Cell Interactome 🧬

_Automated detection and tracking of cellular interactions using self-supervised deep learning_

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Cell Interactome** uses advanced AI to automatically detect, segment, and track biological objects in 3D light-sheet microscopy data. Key capabilities:

- **🎯 Automated detection** of cellular structures (vesicles, organelles, proteins)
- **✂️ Precise segmentation** with no manual annotation required
- **🔗 Temporal tracking** for cellular dynamics analysis
- **📊 Multi-channel support** (488nm, 560nm, 642nm wavelengths)

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

## 🔄 Pipeline Workflow

### 1. Feature Extraction

![Inference Configuration](icons/inference_config.png)
![Path Selection](icons/inference_select_paths.png)
![Setup Interface](icons/inference_setup.png)

**Configure and extract DINO features from 3D volumes:**

- Multi-GPU distributed processing
- Automatic volume preprocessing and normalization
- Sliding window inference for large datasets

### 2. Segmentation

![Segmentation Interface](icons/segmentation.png)
![File Operations](icons/file_operations.png)

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

![Tracking](icons/tracking.png)
![Feature Visualization](icons/visualize_features.png)
![Track Visualization](icons/tracks.png)

**Temporal object linking and analysis:**

- Physics-based velocity prediction
- Feature-enhanced trackpy integration
- Multi-channel track visualization in Napari

![CLI Summary](icons/summary_of_cli.png)
![CLI Interface](icons/cli_image.png)

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

## 📚 Resources

- **[Technical Details](wiki/spatial_dino.pdf)**: Research methodology
- **[Video Tutorials](wiki/run_spatial_dino/)**: Workflow demonstrations
- **[Getting Started](wiki/getting_started.md)**: Step-by-step guide

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
