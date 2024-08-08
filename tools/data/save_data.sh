#!/bin/bash

python save_data.py \
--parent_directory /nfs/datasync4/tklab-llsm \
--search_pattern "**/DS" \
--save_path /nfs/datasync4/ajain/tklab-llsm \
--found_directories_limit 1 \
--min_tif_files_in_folder 20 \
--max_workers 100
