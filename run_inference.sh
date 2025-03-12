#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=8 scripts/inference/inference_3d_multigpu.py 
python3 src/cell_interactome/utils/inference.py