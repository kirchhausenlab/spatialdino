# 🏃 Scripts Directory Guide

## 🎯 **Research Workflows & Production Scripts**

The `scripts/` directory contains all executable workflows, from data preparation to final analysis. This is where biologists and researchers interact with the Cell Interactome system to process their microscopy data and train new models.

---

## 📊 **Directory Overview**

```
scripts/
├── 🎓 train/                    # Model training workflows
├── 🔍 inference/                # Production inference pipeline
├── 📊 data/                     # Data preparation & quality control
├── 📓 notebooks/                # Interactive analysis & visualization
├── 🎨 visualize/                # Result visualization tools
├── 🔗 tracking/                 # Object tracking workflows
├── 🖥️  interactive_inf.py       # User-friendly GUI interface
├── 👁️  napari_viewer.py         # 3D visualization with Napari
├── 🛠️  support_funcs_for_tklab/ # Laboratory-specific utilities
├── 📁 vols_to_save/            # Output storage organization
└── ⚙️  *.sh                    # Automation shell scripts
```

---

## 🎓 **Training Module** (`train/`)

**Purpose**: Train new models or fine-tune existing ones for specific biological applications.

### Core Training Scripts

#### **`pretrain.py`** - Self-Supervised Learning

```bash
# Train DINO backbone on your microscopy data
torchrun --nnodes 1 --nproc_per_node 8 scripts/train/pretrain.py
```

**Features**:

- Multi-GPU distributed training
- Self-supervised learning (no labels needed)
- Supports various Vision Transformer sizes
- Automatic checkpointing and resuming

**Biological Use Cases**:

- Creating features for new cell types
- Adapting to different microscopy setups
- Domain adaptation for new imaging conditions

#### **`segmentation.py`** - Object Detection Training

```bash
# Train segmentation model for specific biological objects
torchrun --nnodes 1 --nproc_per_node 8 scripts/train/segmentation.py
```

**Features**:

- Encoder-decoder architecture training
- 3D volume segmentation
- Multi-channel input support
- Custom loss functions for biology

**Biological Use Cases**:

- Training for specific organelles
- Cell type-specific segmentation
- Multi-protein co-localization

#### **`sinder.py`** - Defect Detection Training

```bash
# Train models for artifact detection and correction
python scripts/train/sinder.py
```

**Features**:

- Detects imaging artifacts
- Learns normal vs. abnormal patterns
- Quality control automation

#### **`upsample.py`** - Feature Enhancement Training

```bash
# Train upsampling models for higher resolution
torchrun --nnodes 1 --nproc_per_node 8 scripts/train/upsample.py
```

**Features**:

- Super-resolution for biological features
- Multi-scale feature enhancement
- Resolution adaptation

---

## 🔍 **Inference Module** (`inference/`)

**Purpose**: Production-ready analysis pipeline for processing biological data.

### Core Inference Scripts

#### **`inference.py`** - Feature Extraction

```bash
# Extract DINO features from your data
python scripts/inference/inference.py \
    inference.img_path="./data/experiment/" \
    model.weights="./models/pretrained.pth"
```

**What it does**:

- Extracts high-dimensional features from 3D volumes
- Multi-GPU parallel processing
- Sliding window for large datasets
- Memory-efficient processing

**Output**:

- `lr_feats.tif`: Feature maps for downstream analysis
- `volume.tif`: Preprocessed volume data

#### **`segmentation.py`** - Object Segmentation

```bash
# Segment biological objects in your data
python scripts/inference/segmentation.py \
    file_path="./results/inference_output/"
```

**What it does**:

- Detects and segments individual objects
- Uses PCA for feature visualization
- Instance segmentation with unique labels
- Post-processing for biological objects

**Output**:

- `segmentation.tif`: Binary masks for each object
- `centroids.csv`: 3D coordinates of detected objects
- `features.tif`: Object-level feature vectors

#### **`postprocessing.py`** - Result Refinement

```bash
# Refine segmentation results
python scripts/inference/postprocessing.py \
    config.file_path="./segmentation_results/"
```

**What it does**:

- Removes noise and artifacts
- Merges fragmented objects
- Applies biological constraints
- Quality filtering

#### **`decoder_seg.py`** - Advanced Segmentation

```bash
# Advanced segmentation with decoder
python scripts/inference/decoder_seg.py
```

**Features**:

- Uses decoder network for reconstruction
- More accurate boundary detection
- Handles complex object shapes

### Supporting Scripts

#### **`enhanced_normalization_example.py`**

- Demonstrates advanced normalization techniques
- Channel-specific processing
- Intensity correction methods

#### **`norm_per_vol.py`**

- Volume-specific normalization
- Adaptive intensity scaling
- Background correction

---

## 📊 **Data Module** (`data/`)

**Purpose**: Prepare and quality-check biological data before analysis.

### Data Preparation Scripts

#### **`prepare.py`** - Main Data Preparation

```bash
# Prepare raw microscopy data for analysis
python scripts/data/prepare.py \
    --input_dir "/path/to/raw/data" \
    --output_dir "/path/to/processed"
```

**What it does**:

- Converts various microscopy formats
- Standardizes file organization
- Creates metadata files
- Validates data integrity

#### **`deskew.py`** - Light Sheet Correction

```bash
# Correct light sheet microscopy distortions
python scripts/data/deskew.py \
    --input_path "/path/to/data" \
    --angle 32.8
```

**What it does**:

- Corrects geometric distortions from light sheet imaging
- Adjustable skew angle
- Preserves spatial relationships
- Outputs corrected volumes

### Quality Control Scripts

#### **`calculate_global_stats.py`** - Dataset Statistics

```bash
# Calculate statistics across your dataset
python scripts/data/calculate_global_stats.py
```

**Features**:

- Mean/std calculation for normalization
- Channel-wise statistics
- Quality metrics
- Dataset summary reports

#### **`per_file_stats.py`** - Individual File Analysis

```bash
# Analyze individual files for quality
python scripts/data/per_file_stats.py
```

**Output**: Detailed quality reports in `per_file_stats.md`

#### **`prepare_data_quality_check.py`** - Automated QC

```bash
# Automated quality control pipeline
python scripts/data/prepare_data_quality_check.py
```

**Features**:

- Detects corrupt files
- Identifies outliers
- Flags potential issues
- Generates QC reports

### Data Management Scripts

#### **`save.py` & `save.sh`** - Data Archiving

```bash
# Archive processed data
bash scripts/data/save.sh
```

**Features**:

- Organized data storage
- Metadata preservation
- Backup creation
- Version control

#### **`select_experiments.py`** - Experiment Selection

```bash
# Select specific experiments for analysis
python scripts/data/select_experiments.py
```

#### **`update_data.py`** - Data Updates

```bash
# Update dataset with new files
python scripts/data/update_data.py
```

#### **`create_experiment_txt.py`** - Experiment Documentation

```bash
# Create experiment documentation
python scripts/data/create_experiment_txt.py
```

---

## 📓 **Notebooks Module** (`notebooks/`)

**Purpose**: Interactive analysis and visualization for biological research.

### Feature Analysis

#### **Feature Visualizations** (`feature_visualizations/`)

- **`visualize_features_ap2.ipynb`**: AP2 protein analysis
- **`visualize_features_test.ipynb`**: General feature exploration
- **`visualize_features.ipynb`**: Main feature analysis notebook

**Use Cases**:

- Explore learned features
- Validate model representations
- Compare different conditions

### Tracking Analysis

#### **Quantify Tracks** (`quantify_tracks/`)

- **`tracking_experiments_2chan.ipynb`**: 2-channel tracking analysis
- **`tracking_experiments_3chan.ipynb`**: 3-channel tracking analysis

**Features**:

- Track quality metrics
- Movement pattern analysis
- Statistical comparisons
- Publication-ready plots

#### **Run Tracking** (`run_tracking/`)

- **`tracking_main.ipynb`**: Main tracking workflow
- **`tracking_tests.ipynb`**: Tracking algorithm testing

### Visualization Tools

#### **Viewer Notebooks** (`viewer/`)

- **`viewer_2chan.ipynb`**: 2-channel data viewer
- **`viewer_3chan.ipynb`**: 3-channel data viewer
- **`viewer_2chan_3movies.ipynb`**: Multi-movie comparison

**Features**:

- Interactive 3D visualization
- Multi-channel overlay
- Time series playback
- Annotation tools

### Processing Utilities

#### **Segmentation Misc** (`segmentation_misc/`)

- **`test_segmentation.ipynb`**: Segmentation testing
- **`convert_movie_to_tiff.ipynb`**: Format conversion
- **`deconv.ipynb`**: Deconvolution testing

---

## 🎨 **Visualization Module** (`visualize/`)

**Purpose**: Generate publication-quality visualizations and organize results.

### Scripts

#### **`save_2d.py`** - 2D Projection Export

```bash
# Create 2D projections from 3D data
python scripts/visualize/save_2d.py
```

**Features**:

- Maximum intensity projections
- Multi-channel overlays
- Scalebar addition
- Publication formatting

#### **`copy_files.py`** - File Organization

```bash
# Organize output files
python scripts/visualize/copy_files.py
```

#### **`move_good_images.py`** - Result Curation

```bash
# Curate high-quality results
python scripts/visualize/move_good_images.py
```

---

## 🔗 **Tracking Module** (`tracking/`)

**Purpose**: Temporal analysis and object tracking workflows.

### Structure

- **`ap2_tracks/`**: AP2 protein tracking data

  - `ap2_detections.csv`: Object detections
  - `ap2_track.csv`: Linked trajectories

- **`transferrin/`**: Transferrin tracking data
  - `transferrin_centroids.csv`: Centroid positions

---

## 🖥️ **Interactive Tools**

### **`interactive_inf.py`** - User-Friendly Interface

```bash
# Launch interactive analysis interface
python scripts/interactive_inf.py
```

**Features**:

- GUI for non-programmers
- Step-by-step workflow guidance
- Real-time progress monitoring
- Parameter adjustment
- Batch processing support

**Perfect for Biologists**:

- No command-line expertise needed
- Visual parameter selection
- Automated workflow execution
- Result preview and validation

### **`napari_viewer.py`** - 3D Visualization

```bash
# Launch 3D viewer for your data
python scripts/napari_viewer.py
```

**Features**:

- Interactive 3D volume viewing
- Multi-channel visualization
- Segmentation overlay
- Track visualization
- Measurement tools

---

## ⚙️ **Automation Scripts** (Shell Scripts)

### **`inference.sh`** - Batch Inference

```bash
# Run inference on multiple datasets
bash scripts/inference.sh
```

### **`segmentation.sh`** - Batch Segmentation

```bash
# Run segmentation on multiple experiments
bash scripts/segmentation.sh
```

### **`save_vols.sh`** - Volume Archiving

```bash
# Archive processed volumes
bash scripts/save_vols.sh
```

---

## 🛠️ **Laboratory Support** (`support_funcs_for_tklab/`)

**Purpose**: Specialized tools for Kirchhausen lab workflows.

### Scripts

#### **`process_experiment_tiffs.py`** - Lab-Specific Processing

```bash
# Process lab-standard TIFF files
python scripts/support_funcs_for_tklab/process_experiment_tiffs.py
```

#### **`xml_to_csv_converter.py`** - Metadata Conversion

```bash
# Convert XML metadata to CSV
python scripts/support_funcs_for_tklab/xml_to_csv_converter.py
```

---

## 📁 **Output Organization** (`vols_to_save/`)

**Purpose**: Organized storage of analysis results.

### Structure

- **`ap2_new_instance_seg/`**: AP2 segmentation results
- **`ap2_new_instance_seg_Raw/`**: Raw AP2 data
- **`zeiss_488/`, `zeiss_560/`, `zeiss_640/`**: Channel-specific results
- **`gu/`**: GU lab specific data

### Subdirectories by Analysis Type

- **`attn/`**: Attention maps
- **`decon/`**: Deconvolved images
- **`instance/`**: Instance segmentation masks
- **`raw/`**: Raw input data
- **`vol_unnorm/`**: Unnormalized volumes

---

## 🚀 **Typical Workflows**

### For Biologists: Full Analysis Pipeline

1. **Data Preparation**

```bash
python scripts/data/prepare.py --input_dir /path/to/raw
python scripts/data/deskew.py --input_path /path/to/data
```

2. **Quality Control**

```bash
python scripts/data/calculate_global_stats.py
python scripts/data/prepare_data_quality_check.py
```

3. **Interactive Analysis**

```bash
python scripts/interactive_inf.py
```

4. **Visualization**

```bash
python scripts/napari_viewer.py
```

### For Researchers: Custom Training

1. **Data Preparation**

```bash
python scripts/data/prepare.py
```

2. **Self-Supervised Pretraining**

```bash
torchrun --nnodes 1 --nproc_per_node 8 scripts/train/pretrain.py
```

3. **Task-Specific Training**

```bash
torchrun --nnodes 1 --nproc_per_node 8 scripts/train/segmentation.py
```

4. **Evaluation**

```bash
python scripts/inference/inference.py
python scripts/inference/segmentation.py
```

### For Production: Batch Processing

1. **Automated Pipeline**

```bash
bash scripts/inference.sh
bash scripts/segmentation.sh
bash scripts/save_vols.sh
```

---

## 💡 **Tips for Usage**

### Configuration Management

- Most scripts use YAML configuration files
- Override parameters with command-line arguments
- Use `+parameter=value` syntax for new parameters

### Memory Management

- Scripts automatically handle GPU memory optimization
- Adjust batch sizes for your hardware
- Use distributed processing for large datasets

### Output Organization

- Results are automatically organized by experiment
- Use consistent naming for easy tracking
- Archive important results regularly

### Monitoring Progress

- Most scripts include progress bars
- Check log files for detailed information
- Use interactive mode for real-time feedback

---

## 📊 **Performance Expectations**

### Typical Processing Times

- **Feature extraction**: 1-5 minutes per volume (depending on size)
- **Segmentation**: 30 seconds - 2 minutes per volume
- **Tracking**: Seconds to minutes (depending on object count)
- **Training**: Hours to days (depending on dataset size)

### Resource Requirements

- **GPU memory**: 8-24GB recommended
- **System RAM**: 32GB+ for large datasets
- **Storage**: 100GB+ for processed results

---

_These scripts provide a complete research environment from raw data to published results, designed specifically for biological microscopy workflows._
