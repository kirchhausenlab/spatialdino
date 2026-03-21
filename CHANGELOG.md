# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-20

### Added
- Self-supervised 3D Vision Transformer (ViT-S/8, ViT-B/8) with DINOv2-style training
- 3D patch embedding with configurable patch size and stride
- Memory-efficient attention via xFormers for scalable training
- Multi-crop self-supervised training with DINO + iBOT losses
- Sinkhorn-Knopp centering for teacher output normalization
- 3D sinusoidal and learned positional embeddings with trilinear interpolation
- Stochastic depth (DropPath) and LayerScale for training stability
- SwiGLU and MLP feed-forward network variants
- U-Net decoder for semantic segmentation from ViT patch tokens
- LiFT (Learned image Feature Transform) upsampling module
- SinDer: singular defect direction analysis for ViT blocks
- Distributed multi-node training with DDP and NCCL backend
- Streaming inference engine with Triton kernels for large volumes
- WebDataset-based data pipeline for efficient I/O with shard shuffling
- 3D data augmentations: random resized crops, flips, isotropic rescaling
- Per-volume and global histogram normalization for inference
- Voronoi-Otsu segmentation post-processing
- Probability map generation and object tracking pipelines
- Web GUI (FastAPI + React) for interactive inference and visualization
- AWS S3 integration for public dataset and model access
- CLI for launching the GUI server (`spatialdino-server`)
- Comprehensive configuration system via OmegaConf YAML files
- CI pipeline with Ruff linting, mypy type checking, and pytest
- Dependabot for automated dependency updates (pip, npm, GitHub Actions)

### Infrastructure
- Project managed with `uv` and `pyproject.toml` (PEP 621)
- Ruff for linting (E, W, F, I, N, UP, B, A, C4, SIM, TCH, RUF) and formatting
- Mypy for static type checking with strict optional and unused-ignore warnings
- GitHub Actions CI workflow for lint, typecheck, and test jobs

[Unreleased]: https://github.com/kirchhausenlab/spatialdino/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kirchhausenlab/spatialdino/releases/tag/v0.1.0
