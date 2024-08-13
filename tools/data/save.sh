#!/bin/bash

python save.py \
--z_voxel_size 32 \
--y_voxel_size 224 \
--x_voxel_size 224 \
--parent_directory /nfs/datasync4/tklab-llsm \
--save_path /nfs/scratch/alavaee/data/processed/llsm \
--found_experiments_limit 50 \
--min_tif_files 20 \
--max_workers 200
