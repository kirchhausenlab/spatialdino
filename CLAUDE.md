# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpatialDINO is a self-supervised 3D vision transformer (DINOv2-style) for 3D fluorescence microscopy. The repo ships three things together: the `spatialdino` Python package (training/inference/segmentation), a FastAPI infra server (`apps/server`), and a React+Vite GUI (`apps/web`) that drives the server.

## Environment & install

- Python 3.12, managed with `uv`. Single root `.venv` shared across the workspace.
- `uv venv --python 3.12 && uv sync --all-packages` from the repo root installs both `spatialdino` and the `apps/server` workspace member (declared in `pyproject.toml` `[tool.uv.workspace]`).
- GUI: `cd apps/web && npm install`.
- Requires CUDA 12+ for training/inference (uses `torch>=2.8`, `xformers`, `monai`).

## Common commands

GUI (recommended dev entrypoint — spawns both vite and the uvicorn API):
```bash
cd apps/web && npm run dev          # starts FastAPI server + vite proxy
cd apps/web && npm run build        # tsc + vite build
cd apps/web && npm run typecheck
```
The dev script (`apps/web/scripts/dev.mjs`) auto-picks a backend port; override with `SPATIALDINO_DEV_API_HOST`, `SPATIALDINO_DEV_API_PORT`, `SPATIALDINO_DEV_API_TARGET`. It uses `SPATIALDINO_UV_CACHE_DIR` (defaults to `<repo>/.uv-cache`).

Server standalone:
```bash
uv run spatialdino-server --host 0.0.0.0 --port 8000 [--reload]
```

Inference (distributed via torchrun — see README for full example):
```bash
uv run torchrun --nnodes 1 --nproc_per_node $NUM_PROC_PER_NODE \
    --rdzv_endpoint=localhost:9999 ./scripts/inference/inference.py \
    file_path=... save_path=... file_start=0 file_end=1 \
    global_hist_min=null global_hist_max=null crop_params="[0,0,0,0,0,0]"
```
CLI args after the script path are merged into the OmegaConf config via `merge_with_cli()`.

Training:
```bash
torchrun --nnodes N --nproc_per_node 8 --node_rank $NODE_RANK \
    --rdzv-id $RDZV_ID --rdzv-backend $RDZV_BACKEND --rdzv-endpoint $RDZV_ENDPOINT \
    scripts/train/pretrain.py
```
Other entrypoints in `scripts/`: `train/segmentation.py`, `train/sinder.py`, `train/upsample.py`, `inference/{segmentation,tracking,probability_map,process_features,norm_per_vol}.py`, and data prep scripts under `scripts/data/`.

Tests / lint:
```bash
uv run pytest tests/                           # full suite
uv run pytest tests/test_inference_pipeline.py # single file
uv run pytest tests/test_ssl.py::test_name     # single test
uv run ruff check .                            # lint (config in ruff.toml)
uv run ruff format .                           # format
```

## Architecture

Source layout is `src/spatialdino/<module>` with the entry scripts living in `scripts/<area>/` — scripts are thin drivers that import from the package, so changes to behavior generally belong in `src/spatialdino/`, not the scripts.

Key modules:
- `spatialdino.config` — OmegaConf-based config loader. `parse_config(path)` loads a YAML, then if it has a `model:` key it merges `config/models/<name>.yaml`, then layers in CLI overrides. Configs live in `src/spatialdino/config/*.yaml` (`pretrain`, `inference`, `segmentation`, `sinder`, `upsampler`, `deskew`).
- `spatialdino.distributed` — torch.distributed helpers; nearly every script calls `dist.init()` and uses `dist.is_main_process()` etc. Inference and training both assume torchrun-style launch.
- `spatialdino.data` — `webdataset` pipeline for training (`dataset.make_webdataset`, `collate_fn_train`), 3D TIFF inference dataset (`data.inference.InferenceDataset`/`InferenceTransform`), and a large `transforms.py` with the 3D augmentation stack (`PreTrainTransform`).
- `spatialdino.models` — backbone + heads. `models.utils.init_backbone` / `load_model` are the constructors used by inference and training. Submodules: `layers/` (ViT building blocks), `ssl/` (SSL wrapper class `SSL` plus `save_model`/`load_model` helpers), `lift/` (2D→3D lifting), `sinder/`, `segmentation/`.
- `spatialdino.loss` — `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss` (consumed by `pretrain.py`).
- `spatialdino.inference` — streaming encoder (`StreamingEncoder`) used to walk large volumes patch-wise and the file enumeration helper `list_tiff_paths`.
- `spatialdino.logging` — `setup_logging`, `MetricLogger`, `SmoothedValue`, plus `logging.wandb.init_wandb`.

Server / GUI:
- `apps/server/spatialdino_server/app.py` is a single ~110KB FastAPI app; `cli.py` boots it via uvicorn (`spatialdino-server` console script). `fs_api.py` and `fs_roots.py` expose filesystem browsing; `jobs_api.py` runs/tracks long jobs; `status.py` reports health.
- `apps/web` is a Vite/React/TypeScript SPA; `npm run dev` proxies API calls to the FastAPI server started in the same process tree.

Pretrained weights are expected at `../models/backbone.pth` (relative to the working directory of inference invocations); models and datasets are pulled from the public `s3://spatialdino/` bucket with `--no-sign-request`.

## Core innovations (load-bearing — read before changing the model)

**No position embeddings (by design)** — `models/layers/encoder.py:164-192` makes `pos_embed_type` a config switch with three modes: `"sincos"`, `"learned"`, or absent (`self.pos_embed = None`, logs `"no positional encoding"`). The forward path gates on it: `if self.use_pos_embed: x = x + self.interpolate_pos_encoding(...)` (`encoder.py:294-296`). The `Conv3d` patch embed (`patch_embed.py:48-49`) carries the spatial inductive bias instead — there is no RoPE replacement. When pos-embed *is* enabled, interpolation is **trilinear** (`encoder.py:225-274`) so volumes can vary in size. Don't reintroduce mandatory positional encodings.

**Pretrain loss combo** (`scripts/train/pretrain.py:175-184`): DINO cls-token loss on global↔global *and* local↔global pairs, plus iBOT masked-patch loss (mask ratios 15-50%, line 275), plus KoLeo regularizer. Teacher EMA at line 659; LR/WD/teacher-momentum/teacher-temperature schedules at lines 412-418; gradient accumulation with DDP sync control at 496-505.

**SSL wrapper** (`models/ssl/__init__.py:10-241`): student + teacher are independent `Encoder` instances, teacher under `no_grad`. `forward()` runs student over global+local crops with mask token replacement (lines 100, 126-138); `forward_teacher()` does centering / Sinkhorn-Knopp and updates centers (lines 142-230).

**3D transform stack** (`data/transforms.py`, ~2.4k lines):
- `PreTrainTransform` (1568-1694) — 2 globals (96³) + 8 locals (48³); each crop goes through `KMeansSpatialCropSamplesd` + RandFlip + RandRotate90d on Z,Y plane only (1346-1352, preserves X scan semantics) + RandAdjustContrastd (γ 0.25-2.0) + RandGaussianNoise.
- `KMeansSpatialCropSamplesd` (984-1202) — **content-aware 3D crop**: KMeans on top-percentile (99.75-99.9) intensity voxels so crops land on signal, not background. This is the main "no random crops on empty volume" innovation.
- Histogram normalization (64-202) — 16-bit fluorescence stretch with `threshold_divisor=1/5000` for background suppression.
- Chunked trilinear interp with Gaussian antialiasing (461-759) — splits 5D tensors to fit memory.
- `Deskew` (`data/deskew.py:203-241`) — lattice light-sheet correction: 31.5° rotation + anisotropic dz/dx scaling.
- `collate_fn_train` (`collate.py:73-205`) — 3D iBOT block/random masking with aspect-ratio jitter; weight ∝ 1/mask-count.
- Other variants: `SegmentationTransform`, `SinderTransform`, `LiFTtransforms` (multi-scale full/half/quarter).

**Inference path** (`scripts/inference/inference.py` + `spatialdino.inference`):
- `InferenceDataset` (`data/inference.py:117-305`) — crop tuple, isotropic resample, histogram or min-max norm (median-fills NaN/inf at line 217), pads to multiples of patch + chunk size (269-286).
- `global_hist_min` / `global_hist_max` (line 141) optionally override per-volume stats — populated by `scripts/inference/norm_per_vol.py` for cross-volume consistency.
- Multi-GPU sharding (`inference.py:99,102`) is plain `DistributedSampler(shuffle=False)` over the **file list** — each rank owns whole files, no within-volume sharding.
- **`StreamingEncoder`** is the headline inference innovation: tiles a volume into `streaming_patch_chunk_size` chunks, embeds each via `Conv3d`, stores tokens in a `TokenStore` (GPU/CPU/disk-backed), and streams Q/KV in blocks (`q_block_tokens`, `kv_block_tokens`) through the transformer with online-softmax attention (Triton kernels with LogSumExp). Tokens write directly into the unified feature grid via `store.patch_view`, so stitching is **index-based, no blending**.
- Outputs under `save_path/`: `lr_feats/{t}.npy` + `raw/{t}.tif` (from `inference.py`); then `hr_feats/`, `pca_n/` (`process_features.py` — trilinear upsample + PCA), `seg_probmap/` + `probmap_densities.npz` (`probability_map.py` — KDE), `seg_voronoi/*.tif` (`segmentation.py` — Voronoi-Otsu on attention heads), `tracks.csv` (`tracking.py` — feature-similarity linking).

**SINDER (Singular Defect Repair)** — 3D adaptation of the 2D SINDER paper, run as a fine-tuning pass on a frozen pretrained backbone (`scripts/train/sinder.py`).
- `singular_defect.py:164-223` — per block, build composed linear `A = A4·A3·A2·A1` for the attention V-branch (`anomaly_dir_attn`) and for the MLP via 100k-sample LSQ approx of the nonlinearity (`anomaly_dir_mlp_ls`). Accumulate left-singular vectors block-by-block to get one cumulative anomaly direction per depth.
- `repair.py` — each linear becomes `U @ diag(S + ε) @ Vᵀ`. **U, S, V are frozen; only ε is trained.** For QKV, only the V-branch ε learns (`replace_linear_addition_noqk`).
- `neighbor_loss.py:107-143` — per-voxel anomaly score `−|f·d|`, softmax-tempered, then a **3×3×3 Gaussian** weighted average of token features in the 3D neighborhood. Loss pulls anomalous tokens toward that mean. Unfolds along D,H,W (lines 31-34) — explicitly volumetric.
- Train loop: `init_backbone` → compute defect dirs → `replace_linear_addition_noqk` → freeze all except ε → SGD(momentum=0.9) on `loss_neighbor` only. `config/sinder.yaml` uses 256³ crops with `isotropic_scale_factor=[2.404, 1.0, 1.0]` for anisotropic z and `limit_layers: 10` to restrict gradient depth.

There are no Jupyter notebooks in the repo; all inference workflows are the scripts under `scripts/inference/`.

## Conventions

- Ruff config (`ruff.toml`): line length 88, double quotes, target `py310` for lint despite project requiring 3.12. Lazy imports (`PLC0415`) are intentionally allowed — don't auto-fix them.
- Configs are OmegaConf YAML; pass overrides as `key=value` CLI args (dot paths work).
- Distributed code paths assume `torchrun`; running a script with plain `python` will not initialize the process group correctly.
