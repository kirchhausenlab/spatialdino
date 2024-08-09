#!/bin/bash

python save_data.py \
--parent_directory /nfs/scratch/Anand \
--search_pattern "**/DS" \
--save_path /nfs/datasync4/ajain/tklab-llsm \
--found_directories_limit 5 \
--min_tif_files_in_folder 8 \
--max_workers 10
