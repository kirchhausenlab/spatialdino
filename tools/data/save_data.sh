#!/bin/bash

python save_data.py \
--parent_directory /nfs/scratch \
--save_path /nfs/scratch/alavaee/data/processed/tklab-llsm \
--found_experiments_limit 1 \
--max_workers 10
