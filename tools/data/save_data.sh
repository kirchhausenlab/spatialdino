#!/bin/bash

python save_data.py \
--z_voxel_size 32 \
--y_voxel_size 256 \
--x_voxel_size 256 \
--parent_directory /nfs/scratch \
--save_path /nfs/scratch/alavaee/data/processed/tklab-llsm \
--found_experiments_limit 1 \
--max_workers 100
