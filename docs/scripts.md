# Scripts Reference

All runnable scripts live under `scripts/` and are organized into four stages of the SpatialDINO pipeline: **data preparation**, **training**, **inference**, and **post-processing**.

---

## Data Preparation (`scripts/data/`)

### `save.py` — WebDataset Builder

The primary data-ingestion script. Reads raw `.tif` microscopy volumes from experiment directories and writes chunked 2D or 3D data into [WebDataset](https://github.com/webdataset/webdataset) `.tar` shards.

**Notable classes and functions:**

| Name | Signature | Description |
|------|-----------|-------------|
| `DataExtractor` | `@dataclass` | Orchestrates the full extraction pipeline. Configurable chunk sizes (`z/y/x_chunk_size`), voxel spacing (`dz/dy/dx`), auto-crop parameters, and worker count. Call instances directly — `extractor(data_paths_txt, save_path)`. |
| `DataExtractor.extract_experiments` | `(data_paths, found_experiments_limit, min_tif_files, use_deskewed_data, experiment_filter_pattern, existing_experiments) -> List[Experiment]` | Scans data directories for valid experiments matching a regex filter. Skips already-processed experiments via a hash-based deduplication scheme. |
| `DataExtractor.get_experiment_name` | `(directory: str) -> str` | Produces an MD5 hash of the directory path as a stable, unique experiment identifier. |
| `_data_iterator` | `(save_2d, save_3d, auto_crop, tif_file, ...) -> List[Dict]` | Worker function submitted to `loky` executor. Reads a single TIF, chunks it into 2D or 3D patches, and returns WebDataset-compatible dicts with `__key__`, `values.npy`, and `metadata.pth`. |
| `chunk_2d_images` | `(image_data, stack, base_metadata, y_chunk_size, x_chunk_size) -> Generator[Image2D]` | Yields non-empty 2D tiles from a Z-projected image. Skips tiles below `EMPTY_VOXEL_THRESHOLD`. |
| `chunk_3d_images` | `(image_data, stack, base_metadata, z/y/x_chunk_size) -> Generator[Image3D]` | Yields 3D sub-volumes with position metadata for spatial reassembly. |
| `chunk_3d_images_auto_crop` | `(image_data, ..., k, lower_percentile, upper_percentile, min_bbox_ratio, max_bbox_ratio) -> Generator[Image3D]` | Uses k-means-based auto-cropping (`utils.auto_extract_crop_params`) to extract regions of interest before chunking. |
| `Experiment` | `TypedDict` | Schema: `tif_files`, `name`, `path`, `metadata`. |

**CLI:** `python scripts/data/save.py --save_3d --auto_crop --data_paths_txt <file> --save_path <dir>`

---

### `deskew.py` — Distributed Lattice Light-Sheet Deskewing

Corrects the geometric skew inherent to lattice light-sheet microscopy data using GPU-accelerated affine transforms. Runs distributed across GPUs via `torchrun`.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Loads `deskew.yaml` config, sets up distributed process group, builds the affine deskew matrix from the first volume, computes a valid-data mask, then iterates all frames through `DeskewDataset` + `DistributedSampler`. Fills invalid (border) voxels with the median of the masked region. |

**Config:** `src/spatialdino/config/deskew.yaml`
**Launch:** `torchrun --nproc_per_node N scripts/data/deskew.py`

---

### `calculate_global_stats.py` — Dataset-Wide Mean/Std

Computes the global mean and standard deviation across the entire training dataset for input normalization.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `(cfg: DictConfig) -> None` | Hydra entry point. Iterates the full training dataset with 128 workers, applies per-volume min/max normalization via `get_min_max`, then accumulates running sums for Welford-style mean/variance computation. |

**Constants:** `CONST_AUTO_THRESHOLD = 5000`, `MAX_VAL = 65535` (16-bit)

---

### `per_file_stats.py` — Annotation File Statistics

Reports per-annotation-file statistics: total data size (GB), average z-frames per experiment, and average experiment size.

| Name | Signature | Description |
|------|-----------|-------------|
| `analyze_folder` | `(path) -> (total_size, tif_count)` | Recursively walks a directory counting `.tif` files and summing file sizes. |
| `expand_experiments` | `(path) -> List[str]` | Expands a directory into its immediate subdirectories (each treated as an experiment). |

**Output:** Writes `scripts/data/per_file_stats.md`

---

### `prepare.py` — Placeholder

Contains a single TODO comment: "Normalize the data created from save_data.py here." Reserved for future normalization pipeline.

---

### `prepare_data_quality_check.py` — Visual Grid QC

Creates labeled grid images of maximum-intensity projections for manual quality review of channel data.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Iterates AO-LLSM data directories, reads one TIF per channel, computes max-Z projections, and tiles them into 6x6 grids saved as PNGs with CSV metadata. |

---

### `create_experiment_txt.py` — Experiment Path Discovery

Crawls data directories for valid experiment channel paths matching a regex pattern (dextran, lamp1, npc1, eea1, transferrin).

**Output:** `channel_paths_valid.txt`

---

### `select_experiments.py` — Experiment Filtering

Filters saved experiment paths by matching metadata against target fluorophore patterns. Reads `.pth` metadata and writes filtered paths to a text file.

---

### `update_data.py` — Metadata Updater

Thin wrapper around `spatialdino.data.utils.update_pth_file_metadata` to batch-update `.pth` file metadata across a directory tree.

---

## Training (`scripts/train/`)

All training scripts use distributed data parallel (DDP) via `torchrun`, WebDataset for I/O, OmegaConf YAML configs, and WandB logging.

### `pretrain.py` — Self-Supervised Pretraining (DINO + iBOT)

The core pretraining script. Implements DINOv2-style joint-embedding self-supervised learning on 3D microscopy volumes.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Builds the SSL model, WebDataset pipeline with multi-crop transforms, optimizer (AdamW/Adam), and loss scaler. Handles checkpoint resume. Launches `train()`. |
| `train` | `(config, step, model: SSL, model_ddp: DDP, optimizer, train_dataloader, rank, world_size, loss_scaler?, run?) -> dict[str, float]` | The training loop. Manages five schedules (LR, weight decay, momentum, teacher temp, last-layer LR). Computes DINO cls-token loss, iBOT patch-token loss, and KoLeo regularization. Performs teacher EMA updates, gradient accumulation, mixed-precision training, and periodic checkpointing. |

**Losses:** `DINOLoss` (cls token), `iBOTPatchLoss` (masked patch tokens), `KoLeoLoss` (uniformity regularization)
**Config:** `pretrain.yaml`

---

### `segmentation.py` — U-Net Decoder Training

Trains the segmentation decoder head on top of frozen ViT encoder features.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Builds the segmentation model, WebDataset pipeline with `SegmentationTransform`, and optimizer. |
| `train` | `(config, step, model: Segmentation, model_ddp, ...) -> dict[str, float]` | Training loop with two losses: pixel reconstruction (L1/L2/Charbonnier selectable via config) and Soft Normalized Cuts (`SoftNCutsLoss`). Uses `ReduceLROnPlateau` scheduler. |

**Config:** `segmentation.yaml`

---

### `sinder.py` — Singular Defect Direction Training

Trains the SinDer (Singular Defect) module that identifies and repairs singular defect directions in ViT attention blocks.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Initializes backbone, computes `singular_defect_directions`, calls `replace_linear_addition_noqk` to inject learnable epsilon parameters. Freezes all parameters except `.epsilon` entries, optimizes with SGD. |
| `train` | `(config, step, model, ...) -> dict[str, float]` | Training loop using `get_neighbor_loss` with temperature-scaled spatial neighborhood consistency. Handles NaN gradients gracefully and supports optional layer-limiting. |

**Config:** `sinder.yaml`

---

### `upsample.py` — LiFT Upsampler Training

Trains the LiFT (Learned image Feature Transform) model to upsample low-resolution ViT patch features to full image resolution.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Initializes a frozen ViT backbone (optionally `torch.compile`-d), wraps it in a `ViTExtractor`, and creates a `LiFT` module. Configures multi-scale feature targets at L1, L2, L4 resolutions. Loss function selectable: MSE, L1, or CosineEmbedding. |

**Config:** `upsampler.yaml`

---

## Inference (`scripts/inference/`)

### `inference.py` — Distributed Feature Extraction

The main inference entry point. Extracts dense ViT patch features from 3D microscopy volumes.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Sets up distributed inference, loads the backbone in eval mode, and creates an optional `StreamingEncoder` for memory-efficient processing of large volumes. Iterates files via `InferenceDataset` + `DistributedSampler`. |
| `predict` | `() -> None` (inner) | For each volume: applies `InferenceTransform`, runs either streaming or standard forward pass to get `lr_feats`, removes padding, and saves as `lr_feats.npy` in `[Z, Y, X, C]` layout. |

**Key features:**
- Streaming inference via Triton kernels for volumes exceeding GPU memory
- Per-volume or global histogram normalization (see `norm_per_vol.py`)
- Configurable crop parameters and isotropic rescaling

**Launch:** `torchrun --nproc_per_node N scripts/inference/inference.py file_path=... save_path=...`

---

### `norm_per_vol.py` — Global Histogram Normalization

Computes dataset-wide histogram bounds for consistent normalization across all inference volumes.

| Name | Signature | Description |
|------|-----------|-------------|
| `main` | `() -> None` | Reads all TIFs, applies cropping and isotropic rescaling, concatenates along Z, then computes a 65536-bin histogram on non-zero voxels. Finds min/max intensity thresholds at configurable percentile cutoffs. |

**Output:** `norm_per_vol.txt` with `Global hist min` and `Global hist max` values.

---

## Post-Processing (`scripts/post_processing/`)

### `process_features.py` — PCA & High-Resolution Feature Export

Processes saved `lr_feats.npy` feature volumes: computes PCA projections and/or upsamples all channels to full resolution.

| Name | Signature | Description |
|------|-----------|-------------|
| `compute_pca_volume` | `(lr_feats, *, n_components, device) -> np.ndarray` | GPU-accelerated PCA via batched covariance computation and `torch.linalg.eigh`. Memory-efficient: processes features in chunks controlled by `DEFAULT_PCA_BATCH_BYTES` (256 MB). |
| `upsample_channels` | `(channels_zyx, *, target_shape, device) -> np.ndarray` | Trilinear upsampling of feature channels to the original volume resolution. |
| `export_pca` | `(subfolder, lr_feats, *, target_shape, n_components, save_format, device) -> None` | Full PCA pipeline: compute, upsample, normalize to uint8, save as TIF or NPY. |
| `export_high_resolution_features` | `(subfolder, lr_feats, *, target_shape, ...) -> None` | Upsamples all feature channels individually with threaded I/O. Memory-bounded by `DEFAULT_MAX_UPSAMPLE_BYTES` (512 MB). |
| `process_subfolder` | `(input, output, *, save_pca, pca_components, ...) -> None` | Entry point per sample folder. |

**CLI:** `python scripts/post_processing/process_features.py --input-path <dir> --save-pca --pca-components 3`

---

### `segmentation.py` — Voronoi-Otsu Segmentation

Applies Voronoi-Otsu labeling to SpatialDINO features for instance segmentation. Uses [pyclesperanto](https://github.com/clEsperanto/pyclesperanto) for GPU-accelerated morphological operations.

| Name | Signature | Description |
|------|-----------|-------------|
| `segment_subfolder` | `(subfolder, *, output_path, gaussian_blur_sigma, rolling_ball_radius, device, cle_device, log_kernel) -> Path` | Sums patch token features, upsamples to full resolution, applies rolling-ball background subtraction, LoG convolution, Gaussian blur, then Voronoi-Otsu labeling. Saves uint32 label masks as TIFF. |
| `_log_kernel` | `(size, sigma) -> np.ndarray` | Generates a 3D Laplacian-of-Gaussian kernel for blob detection. |
| `upsample_scalar_volume` | `(volume_zyx, *, target_shape, device) -> np.ndarray` | Trilinear upsampling helper for scalar volumes. |

**CLI:** `python scripts/post_processing/segmentation.py --input-path <dir> --output-path <dir> --enable-voronoi-otsu`

---

### `probability_map.py` — Two-Stage Probability Map Classification

Generates voxel-level foreground/background probability maps from SpatialDINO features using density estimation.

| Name | Signature | Description |
|------|-----------|-------------|
| `TimepointPaths` | `@dataclass(frozen=True)` | Holds paths for a single timepoint: subfolder, lr_feats, raw volume. |
| `ProbabilityMapParams` | `@dataclass(frozen=True)` | Full parameter set: density method (KDE or GPU-histogram), batch size, bandwidth, probability thresholds, seed. |
| `PackedDens` | `@dataclass(frozen=True)` | Packed density tensors for batched Stage 2 classification. |
| `discover_timepoints` | `(input_path, *, exclude_paths) -> list[TimepointPaths]` | Validates and discovers all timepoint subfolders with required files. |

**Two stages:**
1. **Density estimation** — fits per-feature foreground/background density functions from a labeled training timepoint
2. **Classification** — applies learned densities to classify every voxel across all timepoints

**CLI:** `python scripts/post_processing/probability_map.py --input-path <dir> --run-density-estimation --training-timepoint <name> --seg-tif <path>`

---

### `tracking.py` — Multi-Object Tracking Across Timepoints

Links segmented objects across time using spatial proximity, feature correlation, and Dice overlap voting.

| Name | Signature | Description |
|------|-----------|-------------|
| `LabelGeometry` | `@dataclass` | Per-label geometry: voxel coordinates, centroid, volume, bounding box, local mask. |
| `PreparedTimepoint` | `@dataclass` | Complete timepoint data: geometries, label IDs, centroids, volumes, mean intensities. |
| `TrackingParams` | `@dataclass(frozen=True)` | All tracking hyperparameters: `max_distance_xy/z`, `z_distance_weight`, `vote_thresholds`, `dice_threshold`, `corr_threshold`. |
| `Track` | `@dataclass` | A track: ordered list of `TrackPoint` entries with track ID, start time, and length. |
| `extract_label_geometries` | `(segmentation_yxz) -> dict[int, LabelGeometry]` | Extracts all foreground labels and computes their spatial geometry in a single vectorized pass. |
| `compute_label_mean_intensities` | `(segmentation_yxz, raw_yxz) -> (label_ids, amplitudes)` | Computes mean raw intensity per label using `np.bincount`. |
| `prepare_timepoint` | `(index, paths: TimepointPaths) -> PreparedTimepoint` | Full timepoint preparation: loads raw + segmentation volumes, extracts geometries, computes amplitudes. |
| `find_spatial_candidates` | `(ref_tp, cand_tp, *, spatial_radius, zratio) -> dict[int, (cand_ids, distances)]` | Finds candidate matches within an anisotropic spatial radius, sorted by distance. |
| `compute_alignment_overlap` | `(ref_geom, cand_geom) -> (dice, ref_coords, cand_coords, overlap)` | Centroid-aligned Dice overlap between two label masks. |
| `anisotropic_distance` | `(a, b, zratio) -> float` | Weighted Euclidean distance accounting for Z anisotropy. |

**Algorithm:** For each consecutive timepoint pair, spatial candidates are found, then feature-channel voting determines assignments. Unassigned labels are handled by distance-based fallback. Tracks are assembled greedily and exported as CSV.

**CLI:** `python scripts/post_processing/tracking.py --input-path <dir> --segmentation-path <dir> --output-path <dir>`
